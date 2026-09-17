"""Sequential, artifact-producing reversible diffusion controls for G1."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import torch

from .dataset import SourceSpec, build_digest_records, generate_source_messages
from .runner import ExperimentConfig, _build_model, _codec, _conditions, _training_data
from .models import GaussianDiffusion, parameter_count


@dataclass(frozen=True)
class G1Config:
    representation: str
    source: str = "printable"
    seed: int = 0
    train_sizes: tuple[int, ...] = (1, 4, 16, 64)
    validation_size: int = 16
    test_size: int = 16
    training_steps: int = 1_500
    batch_size: int = 16
    learning_rate: float = 1e-3
    diffusion_steps: int = 50
    sampling_steps: int = 50
    beta_end: float = 0.4
    prediction_type: str = "sample"
    width: int | None = None
    evaluation_interval: int = 500
    sampling_seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_sizes", tuple(self.train_sizes))
        object.__setattr__(self, "sampling_seeds", tuple(self.sampling_seeds))
        if self.representation not in {"bgv", "cgge", "bits"}:
            raise ValueError("representation must be bgv, cgge, or bits")
        if self.source not in {"printable", "random_bytes"} or self.representation == "cgge" and self.source != "printable":
            raise ValueError("invalid source for representation")
        if self.train_sizes != (1, 4, 16, 64):
            raise ValueError("G1 must run the fixed 1, 4, 16, 64 ladder")
        if min(self.validation_size, self.test_size, self.training_steps, self.batch_size, self.evaluation_interval) < 1:
            raise ValueError("sizes, steps, batch size, and interval must be positive")
        if not self.sampling_seeds:
            raise ValueError("at least one sampling seed is required")


def _experiment_config(config: G1Config, condition_dim: int, dataset_size: int) -> ExperimentConfig:
    return ExperimentConfig(
        representation=config.representation,  # type: ignore[arg-type]
        source=config.source,  # type: ignore[arg-type]
        algorithm="md5",
        q=8,
        dataset_size=dataset_size,
        data_seed=config.seed,
        split_seed=config.seed,
        model_seed=config.seed,
        method="diffusion",
        condition_mode="reversible_record",
        training_steps=config.training_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        diffusion_steps=config.diffusion_steps,
        sampling_steps=config.sampling_steps,
        beta_end=config.beta_end,
        prediction_type=config.prediction_type,  # type: ignore[arg-type]
        width=config.width or (64 if config.representation == "bits" else 8),
        condition_dim=condition_dim,
        device=config.device,
    )


def _byte_score(candidate: bytes | None, target: bytes) -> tuple[int, int]:
    if candidate is None:
        return 0, len(target)
    return sum(left == right for left, right in zip(candidate, target)), max(len(candidate), len(target))


@torch.no_grad()
def _evaluate(model, diffusion, records, config, encoder, decoder, shape, device, seeds) -> tuple[dict[str, object], list[dict[str, object]], torch.Tensor]:
    model.eval()
    clean, conditions = _training_data(records, config, encoder, device)
    mse = bit_correct = bit_total = byte_correct = byte_total = exact = valid = 0
    samples = []
    last = clean
    for seed in seeds:
        last = diffusion.sample(
            model,
            conditions,
            shape,
            sampling_steps=config.sampling_steps,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        mse += torch.nn.functional.mse_loss(last, clean, reduction="sum").item()
        bit_correct += ((last >= 0) == (clean >= 0)).sum().item()
        bit_total += clean.numel()
        for record, value in zip(records, last):
            decoded = decoder.decode((value + 1) / 2)
            correct, total = _byte_score(decoded.message if decoded.valid else None, record.message)
            byte_correct += correct
            byte_total += total
            valid += decoded.valid
            exact += decoded.valid and decoded.message == record.message
            samples.append(
                {
                    "seed": seed,
                    "target_hex": record.message.hex(),
                    "candidate_hex": decoded.message.hex() if decoded.message is not None else None,
                    "valid": decoded.valid,
                    "reason": decoded.reason,
                    "exact": decoded.valid and decoded.message == record.message,
                }
            )
    count = len(records) * len(seeds)
    metrics = {
        "sample_count": count,
        "reconstruction_mse": mse / (bit_total or 1),
        "bit_accuracy": bit_correct / (bit_total or 1),
        "byte_accuracy": byte_correct / (byte_total or 1),
        "valid_decode_rate": valid / (count or 1),
        "exact_recovery_rate": exact / (count or 1),
    }
    return metrics, samples, last


@torch.no_grad()
def _diagnostics(model, diffusion, records, config, encoder, decoder, shape, device, final_sample) -> dict[str, object]:
    model.eval()
    clean, conditions = _training_data(records, config, encoder, device)
    generator = torch.Generator(device=device).manual_seed(config.model_seed + 10_000)
    noise = torch.randn(clean.shape, device=device, generator=generator)
    zero_index = torch.zeros(len(clean), dtype=torch.long, device=device)
    no_noise_input = diffusion.add_noise(clean, torch.zeros_like(clean), zero_index)
    no_noise_output = model(no_noise_input, torch.zeros(len(clean), device=device), conditions)
    no_noise_clean = diffusion.predicted_clean(no_noise_output, no_noise_input, zero_index).clamp(-1, 1)
    levels = sorted({0, 1, config.diffusion_steps // 4, config.diffusion_steps // 2, config.diffusion_steps - 1})
    sweep = []
    for timestep in levels:
        index = torch.full((len(clean),), timestep, dtype=torch.long, device=device)
        noisy = diffusion.add_noise(clean, noise, index)
        output = model(noisy, index.float() / (config.diffusion_steps - 1), conditions)
        predicted = diffusion.predicted_clean(output, noisy, index).clamp(-1, 1)
        sweep.append(
            {
                "timestep": timestep,
                "alpha_bar": diffusion.alpha_bar[timestep].item(),
                "mse": torch.nn.functional.mse_loss(predicted, clean).item(),
                "bit_accuracy": ((predicted >= 0) == (clean >= 0)).float().mean().item(),
            }
        )
    initial = torch.randn(clean.shape, device=device, generator=generator)
    index = torch.full((len(clean),), config.diffusion_steps - 1, dtype=torch.long, device=device)
    direct = diffusion.predicted_clean(
        model(initial, index.float() / (config.diffusion_steps - 1), conditions), initial, index
    ).clamp(-1, 1)
    return {
        "no_noise_reconstruction": {
            "timestep": 0,
            "alpha_bar": diffusion.alpha_bar[0].item(),
            "mse": torch.nn.functional.mse_loss(no_noise_clean, clean).item(),
            "bit_accuracy": ((no_noise_clean >= 0) == (clean >= 0)).float().mean().item(),
        },
        "one_step_denoising": sweep[1],
        "fixed_timestep_noise_sweep": sweep,
        "sampler_bypass": {
            "mse": torch.nn.functional.mse_loss(direct, clean).item(),
            "bit_accuracy": ((direct >= 0) == (clean >= 0)).float().mean().item(),
        },
        "decoder_bypass": {
            "mse": torch.nn.functional.mse_loss(final_sample, clean).item(),
            "bit_accuracy": ((final_sample >= 0) == (clean >= 0)).float().mean().item(),
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_stage(name: str, train_messages, evaluation_sets, config: G1Config, output: Path, command: str) -> dict[str, object]:
    stage = output / name
    stage.mkdir(parents=True, exist_ok=True)
    encoder, decoder, shape = _codec(config.representation)  # type: ignore[arg-type]
    experiment = _experiment_config(config, int(torch.tensor(shape).prod().item()), len(train_messages))
    device = torch.device(config.device)
    train_records = build_digest_records(train_messages, source=config.source, algorithm="md5", q=8)  # type: ignore[arg-type]
    sets = {
        split: build_digest_records(messages, source=config.source, algorithm="md5", q=8)  # type: ignore[arg-type]
        for split, messages in evaluation_sets.items()
    }
    torch.manual_seed(config.seed)
    model = _build_model(experiment, shape).to(device)
    values, conditions = _training_data(train_records, experiment, encoder, device)
    diffusion = GaussianDiffusion(
        config.diffusion_steps,
        beta_end=config.beta_end,
        prediction_type=config.prediction_type,  # type: ignore[arg-type]
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    log = []
    started = perf_counter()
    model.train()
    for step in range(1, config.training_steps + 1):
        indices = torch.randint(len(values), (config.batch_size,), device=device, generator=generator)
        loss = diffusion.loss(model, values[indices], conditions[indices], generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config.evaluation_interval == 0 or step == config.training_steps:
            metrics, _, _ = _evaluate(
                model, diffusion, sets.get("validation", sets["train"]), experiment, encoder, decoder, shape, device, (config.sampling_seeds[0],)
            )
            log.append({"step": step, "loss": loss.item(), **metrics})
            model.train()
    metrics = {}
    sample_rows = []
    final_sample = None
    for split, records in sets.items():
        split_metrics, samples, final_sample = _evaluate(
            model, diffusion, records, experiment, encoder, decoder, shape, device, config.sampling_seeds
        )
        metrics[split] = split_metrics
        sample_rows.extend({"split": split, **sample} for sample in samples)
    required = ("train",) if "test" not in sets else ("train", "validation", "test")
    passed = all(metrics[split]["exact_recovery_rate"] == 1.0 for split in required)
    result = {
        "gate": "G1-C" if "test" in sets else "G1-A" if len(train_messages) == 1 else "G1-B",
        "stage": name,
        "status": "PASS" if passed else "FAIL",
        "seed": config.seed,
        "sampling_seeds": list(config.sampling_seeds),
        "dataset_size": len(train_messages),
        "checkpoint": str(stage / "checkpoint.pt"),
        "command": command,
        "configuration": asdict(config),
        "effective_model_configuration": asdict(experiment),
        "parameter_count": parameter_count(model),
        "torch_version": torch.__version__,
        "training_wall_seconds": perf_counter() - started,
        "terminal_alpha_bar": diffusion.alpha_bar[-1].item(),
        "metrics": metrics,
    }
    torch.save({"config": asdict(experiment), "model_state": model.state_dict()}, stage / "checkpoint.pt")
    _write_json(stage / "metrics.json", result)
    _write_json(stage / "diagnostics.json", _diagnostics(model, diffusion, sets[required[-1]], experiment, encoder, decoder, shape, device, final_sample))
    with (stage / "training.jsonl").open("w", encoding="utf-8") as stream:
        for row in log:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (stage / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in sample_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return result


def run_g1(config: G1Config, output_dir: str | Path, *, command: str = "") -> dict[str, object]:
    """Run G1-A, the 4/16/64 G1-B ladder, then held-out G1-C; stop on failure."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    total = max(config.train_sizes) + config.validation_size + config.test_size
    messages = generate_source_messages(SourceSpec(config.source, total, seed=config.seed))  # type: ignore[arg-type]
    train_pool = messages[: max(config.train_sizes)]
    validation = messages[max(config.train_sizes) : max(config.train_sizes) + config.validation_size]
    test = messages[-config.test_size :]
    if set(train_pool) & set(validation) or set(train_pool) & set(test) or set(validation) & set(test):
        raise RuntimeError("G1 split leakage detected")
    with (output / "split.jsonl").open("w", encoding="utf-8") as stream:
        for split, values in (("train", train_pool), ("validation", validation), ("test", test)):
            for value in values:
                stream.write(json.dumps({"split": split, "message_hex": value.hex(), "sha256": hashlib.sha256(value).hexdigest()}, sort_keys=True) + "\n")
    results = []
    stopped = False
    for size in config.train_sizes:
        name = f"g1-{'a' if size == 1 else 'b'}-n{size}-seed{config.seed}"
        if stopped:
            results.append({"gate": "G1-A" if size == 1 else "G1-B", "stage": name, "status": "NOT RUN"})
            continue
        result = _run_stage(name, train_pool[:size], {"train": train_pool[:size]}, config, output, command)
        results.append(result)
        stopped = result["status"] != "PASS"
    if stopped:
        results.append({"gate": "G1-C", "stage": f"g1-c-n64-seed{config.seed}", "status": "NOT RUN"})
    else:
        results.append(
            _run_stage(
                f"g1-c-n64-seed{config.seed}",
                train_pool,
                {"train": train_pool, "validation": validation, "test": test},
                config,
                output,
                command,
            )
        )
    summary = {
        "status": "PASS" if all(result["status"] == "PASS" for result in results) else "FAIL",
        "torch_version": torch.__version__,
        "configuration": asdict(config),
        "results": results,
    }
    _write_json(output / "gate_results.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the sequential reversible diffusion G1 controls.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config = G1Config(**json.loads(arguments.config.read_text(encoding="utf-8")))
    command = shlex.join([sys.executable, "-m", "diffusion_hash_inv.positive_control", *(argv or sys.argv[1:])])
    result = run_g1(config, arguments.output, command=command)
    print(json.dumps(result, sort_keys=True))
    return 0 if all(item["status"] == "PASS" for item in result["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
