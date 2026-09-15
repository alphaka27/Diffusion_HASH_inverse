"""One reproducible path for controls, baselines, and conditional diffusion."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path
from time import perf_counter
from typing import Literal, Sequence

import torch
from torch import Tensor

from .baselines import nearest_training_digest, source_prior_random_search
from .controls import Representation, representation_control
from .dataset import (
    DigestRecord,
    SourceDistribution,
    SourceSpec,
    build_digest_records,
    generate_source_messages,
    hash_caption,
    select_digest_representatives,
    split_validation_report,
    split_digest_groups,
    write_split,
)
from .encoding import (
    BGVDecoder,
    BGVEncoder,
    CGGEDecoder,
    CGGEEncoder,
    DirectBitsDecoder,
    DirectBitsEncoder,
    GLYPH_TABLE_VERSION,
    glyph_table_checksum,
)
from .evaluation import CandidateAttempt, EvaluationSummary, write_evaluation
from .models import BitDenoiser, DirectPredictor, GaussianDiffusion, ImageUNet, parameter_count


Method = Literal["diffusion", "predictor", "random", "nearest_train", "reversible", "shuffled", "zero"]
ConditionMode = Literal["hash", "zero", "shuffled_hash", "reversible_record"]


@dataclass(frozen=True)
class ExperimentConfig:
    representation: Representation
    source: SourceDistribution
    algorithm: str
    q: int
    dataset_size: int = 12_000
    data_seed: int = 0
    split_seed: int = 0
    model_seed: int = 0
    k: int = 1
    method: Method = "diffusion"
    condition_mode: ConditionMode = "hash"
    length_conditioning: bool = False
    training_steps: int = 1_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    diffusion_steps: int = 100
    sampling_steps: int = 100
    width: int = 32
    condition_dim: int = 256
    test_limit: int | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.representation not in {"bgv", "cgge", "bits"}:
            raise ValueError("unsupported representation")
        if self.method not in {"diffusion", "predictor", "random", "nearest_train", "reversible", "shuffled", "zero"}:
            raise ValueError("unsupported method")
        if self.condition_mode not in {"hash", "zero", "shuffled_hash", "reversible_record"}:
            raise ValueError("unsupported condition mode")
        if self.source not in {"printable", "random_bytes"}:
            raise ValueError("unsupported source")
        if self.representation == "cgge" and self.source != "printable":
            raise ValueError("CGGE is Printable-only")
        if self.representation == "bits" and self.length_conditioning:
            raise ValueError("Direct Bits has no separate length-conditioning ablation")
        if self.method == "predictor" and self.k != 1:
            raise ValueError("the deterministic predictor is only a K=1 baseline")
        if self.dataset_size < 3 or self.k < 1 or self.training_steps < 1 or self.batch_size < 1:
            raise ValueError("dataset size, budget, training steps, and batch size must be positive")
        if (
            self.diffusion_steps < 2
            or self.sampling_steps < 1
            or self.sampling_steps > self.diffusion_steps
            or self.learning_rate <= 0
            or self.width < 2
            or self.condition_dim < 1
            or self.test_limit is not None and self.test_limit < 1
        ):
            raise ValueError("invalid model or sampling dimensions")


@dataclass(frozen=True)
class ExperimentResult:
    summary: EvaluationSummary
    output_dir: Path
    training_loss: float | None
    parameter_count: int | None


def _caption_condition(record: DigestRecord, config: ExperimentConfig) -> Tensor:
    caption = hash_caption(
        record.algorithm,
        record.q,
        record.digest,
        length=len(record.message) if config.length_conditioning else None,
    ).encode("ascii")
    if len(caption) > config.condition_dim:
        raise ValueError("condition_dim is too small for the fixed caption")
    condition = torch.zeros(config.condition_dim, dtype=torch.float32)
    condition[: len(caption)] = torch.tensor(list(caption), dtype=torch.float32) / 127
    return condition


def _digest_condition(record: DigestRecord, config: ExperimentConfig) -> Tensor:
    if record.q > config.condition_dim:
        raise ValueError("condition_dim must contain every q-bit digest condition")
    condition = torch.zeros(config.condition_dim, dtype=torch.float32)
    bits = [
        (byte >> shift) & 1
        for byte in record.digest
        for shift in range(7, -1, -1)
    ][: record.q]
    condition[: record.q] = torch.tensor(bits, dtype=torch.float32)
    return condition


def _record_condition(record: DigestRecord, config: ExperimentConfig) -> Tensor:
    bits = DirectBitsEncoder().encode(record.message).flatten()
    if len(bits) > config.condition_dim:
        raise ValueError("condition_dim is too small for the reversible record")
    condition = torch.zeros(config.condition_dim, dtype=torch.float32)
    condition[: len(bits)] = bits
    return condition


def _condition(record: DigestRecord, config: ExperimentConfig) -> Tensor:
    if config.condition_mode == "zero":
        return torch.zeros(config.condition_dim, dtype=torch.float32)
    if config.condition_mode == "reversible_record":
        return _record_condition(record, config)
    return _digest_condition(record, config) if config.representation == "bits" else _caption_condition(record, config)


def _conditions(records: Sequence[DigestRecord], config: ExperimentConfig) -> Tensor:
    conditions = torch.stack([_condition(record, config) for record in records])
    if config.condition_mode == "shuffled_hash":
        if len(conditions) < 2:
            raise ValueError("shuffled condition requires at least two records")
        return torch.roll(conditions, shifts=1, dims=0)
    return conditions


def _codec(representation: Representation):
    if representation == "bgv":
        return BGVEncoder(), BGVDecoder(), (2, 32, 128)
    if representation == "cgge":
        return CGGEEncoder(), CGGEDecoder(), (2, 32, 64)
    return DirectBitsEncoder(), DirectBitsDecoder(), (32, 8)


def _build_model(config: ExperimentConfig, shape: tuple[int, ...]):
    if config.representation == "bits":
        return BitDenoiser(shape[0] * shape[1], config.condition_dim, width=max(64, config.width * 4))
    return ImageUNet(shape[0], config.condition_dim, width=config.width)


def _training_data(records: Sequence[DigestRecord], config: ExperimentConfig, encoder, device: torch.device) -> tuple[Tensor, Tensor]:
    values = torch.stack([encoder.encode(record.message) * 2 - 1 for record in records]).to(device)
    return values, _conditions(records, config).to(device)


def _train(
    model, records: Sequence[DigestRecord], config: ExperimentConfig, encoder, device: torch.device
) -> tuple[GaussianDiffusion, float]:
    values, conditions = _training_data(records, config, encoder, device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device=device).manual_seed(config.model_seed)
    diffusion = GaussianDiffusion(config.diffusion_steps, device=device)
    loss = torch.tensor(float("nan"))
    for _ in range(config.training_steps):
        indices = torch.randint(len(values), (config.batch_size,), device=device, generator=generator)
        loss = diffusion.loss(model, values[indices], conditions[indices], generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return diffusion, loss.item()


def _diffusion_attempts(
    model,
    diffusion: GaussianDiffusion,
    targets: Sequence[DigestRecord],
    config: ExperimentConfig,
    decoder,
    shape: tuple[int, ...],
    device: torch.device,
):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config.model_seed + 1)
    attempts = []
    for target, condition in zip(targets, _conditions(targets, config).to(device)):
        group = []
        for _ in range(config.k):
            sample = diffusion.sample(model, condition[None, :], shape, sampling_steps=config.sampling_steps, generator=generator)[0]
            decoded = decoder.decode((sample + 1) / 2)
            group.append((decoded.message, decoded.valid, decoded.reason))
        attempts.append(tuple(CandidateAttempt(*value) for value in group))
    return tuple(attempts)


def _train_predictor(
    model, records: Sequence[DigestRecord], config: ExperimentConfig, encoder, shape: tuple[int, ...], device: torch.device
) -> float:
    values, conditions = _training_data(records, config, encoder, device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device=device).manual_seed(config.model_seed)
    loss = torch.tensor(float("nan"))
    for _ in range(config.training_steps):
        indices = torch.randint(len(values), (config.batch_size,), device=device, generator=generator)
        loss = torch.nn.functional.mse_loss(model(conditions[indices], shape), values[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return loss.item()


@torch.no_grad()
def _predictor_attempts(model, targets: Sequence[DigestRecord], config: ExperimentConfig, decoder, shape: tuple[int, ...], device: torch.device):
    model.eval()
    attempts = []
    for target, condition in zip(targets, _conditions(targets, config).to(device)):
        value = model(condition[None, :], shape)[0].clamp(-1, 1)
        decoded = decoder.decode((value + 1) / 2)
        attempts.append(tuple(CandidateAttempt(decoded.message, decoded.valid, decoded.reason) for _ in range(config.k)))
    return tuple(attempts)


def _target_subset(records: Sequence[DigestRecord], config: ExperimentConfig) -> tuple[DigestRecord, ...]:
    limit = config.test_limit
    if config.k == 100:
        limit = min(1_000, len(records)) if limit is None else min(limit, 1_000)
    return tuple(records if limit is None else records[:limit])


def _round_trip_pass(config: ExperimentConfig) -> bool:
    """Run the fixed representation corpus needed for this run's G1 artifact."""
    if config.source == "printable":
        corpus = (b"!Ab9", b"~~~~", b"A1!z" * 7 + b"A1!", b"!" * 31)
    else:
        corpus = (bytes(range(4)), b"\x00\xff\x00\xff", bytes(range(31)), b"\x00" * 31)
    encoder, decoder, _ = _codec(config.representation)
    return all((decoded := decoder.decode(encoder.encode(message))).valid and decoded.message == message for message in corpus)


def run_experiment(config: ExperimentConfig, output_dir: str | Path) -> ExperimentResult:
    """Create data, execute one declared method, and preserve reproducibility artifacts."""
    torch.manual_seed(config.model_seed)
    random.seed(config.model_seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_spec = SourceSpec(config.source, config.dataset_size, seed=config.data_seed)
    messages = generate_source_messages(source_spec)
    records = build_digest_records(messages, source=config.source, algorithm=config.algorithm, q=config.q)
    split = split_digest_groups(records, seed=config.split_seed)
    write_split(split, output / "data", source_spec=source_spec, split_seed=config.split_seed)
    split_report = split_validation_report(split)
    (output / "data" / "split_validation.json").write_text(json.dumps(split_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not split_report["passed"]:
        raise RuntimeError("G0 failed: split contains message or digest overlap")
    round_trip_pass = _round_trip_pass(config)
    if not round_trip_pass:
        raise RuntimeError("G1 failed: representation round-trip corpus did not decode exactly")
    targets = _target_subset(select_digest_representatives(split["test"]), config)
    if not split["train"] or not targets:
        raise ValueError("digest-group split produced an empty train or test partition; increase dataset_size")

    metadata = {
        "config": asdict(config),
        "train_count": len(split["train"]),
        "test_record_count": len(split["test"]),
        "test_count": len(targets),
    }
    training_loss = parameter_total = None
    if config.method == "random":
        attempts = source_prior_random_search(targets, k=config.k, seed=config.model_seed, length_aware=config.length_conditioning)
    elif config.method == "nearest_train":
        attempts = nearest_training_digest(targets, split["train"], k=config.k)
    elif config.method in {"reversible", "shuffled", "zero"}:
        attempts = representation_control(targets, representation=config.representation, kind=config.method, k=config.k, seed=config.model_seed)
    else:
        encoder, decoder, shape = _codec(config.representation)
        device = torch.device(config.device)
        model = _build_model(config, shape).to(device)
        started = perf_counter()
        if config.method == "predictor":
            model = DirectPredictor(config.condition_dim, prod(shape), width=max(64, config.width * 4)).to(device)
            training_loss = _train_predictor(model, split["train"], config, encoder, shape, device)
            diffusion = None
        else:
            diffusion, training_loss = _train(model, split["train"], config, encoder, device)
        metadata.update({"parameter_count": parameter_count(model), "training_wall_seconds": perf_counter() - started})
        parameter_total = parameter_count(model)
        torch.save({"config": asdict(config), "model_state": model.state_dict()}, output / "checkpoint.pt")
        attempts = (
            _predictor_attempts(model, targets, config, decoder, shape, device)
            if config.method == "predictor"
            else _diffusion_attempts(model, diffusion, targets, config, decoder, shape, device)
        )

    summary = write_evaluation(targets, attempts, output / "evaluation", method=config.method, k=config.k, metadata=metadata)
    _, decoder, shape = _codec(config.representation)
    manifest = {
        **metadata,
        "representation": config.representation,
        "representation_version": "bgv-v1" if config.representation == "bgv" else GLYPH_TABLE_VERSION if config.representation == "cgge" else "direct-bits-v1",
        "image_shape": list(shape),
        "glyph_table_checksum": glyph_table_checksum() if config.representation == "cgge" else None,
        "glyph_distance_metric": "mse" if config.representation == "cgge" else None,
        "glyph_valid_threshold": decoder.config.glyph_valid_threshold if config.representation == "cgge" else None,
        "mask_threshold": decoder.config.mask_threshold if config.representation in {"bgv", "cgge"} else None,
        "condition_format": "digest_bits" if config.representation == "bits" else "ascii_caption_bytes",
        "torch_version": torch.__version__,
        "training_loss": training_loss,
        "run_gates": {
            "g0_split_independence": split_report["passed"],
            "g1_round_trip": round_trip_pass,
            "g2_candidate_budget": summary.candidate_attempt_count == summary.target_count * config.k,
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ExperimentResult(summary, output, training_loss, parameter_total)


__all__ = ["ExperimentConfig", "ExperimentResult", "run_experiment"]
