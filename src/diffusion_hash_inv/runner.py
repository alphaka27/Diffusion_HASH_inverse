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
from .encoding.bgv import BGVConfig
from .encoding.cgge import CGGEConfig
from .encoding.tokens import TokenCodec, TOKENIZER_VERSION
from .discrete import MaskedDiffusion, SequenceDenoiser
from .conditioning import digest_condition, shuffled_donors


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
    beta_end: float = 0.02
    prediction_type: Literal["epsilon", "sample"] = "epsilon"
    width: int = 32
    condition_dim: int = 256
    test_limit: int | None = None
    device: str = "cpu"
    max_length: int = 31
    condition_format: str = "legacy"
    masking_schedule: tuple[float, ...] | None = None
    token_embedding_dim: int | None = None
    token_temperature: float | None = None

    def __post_init__(self) -> None:
        if self.representation not in {"bgv", "cgge", "bits", "tokens"}:
            raise ValueError("unsupported representation")
        if self.method not in {"diffusion", "predictor", "random", "nearest_train", "reversible", "shuffled", "zero"}:
            raise ValueError("unsupported method")
        if self.condition_mode not in {"hash", "zero", "shuffled_hash", "reversible_record", "length_only"}:
            raise ValueError("unsupported condition mode")
        if self.source not in {"printable", "random_bytes"}:
            raise ValueError("unsupported source")
        if self.representation == "cgge" and self.source != "printable":
            raise ValueError("CGGE is Printable-only")
        if self.representation == "bits" and self.length_conditioning:
            raise ValueError("Direct Bits has no separate length-conditioning ablation")
        if self.method == "predictor" and self.k != 1:
            raise ValueError("the deterministic predictor is only a K=1 baseline")
        if not 4 <= self.max_length <= 255 or self.condition_format not in {"legacy", "canonical_bits"}:
            raise ValueError("invalid maximum length or condition format")
        if self.representation == "bits" and self.max_length != 31:
            raise ValueError("legacy Direct Bits is fixed-length and is not a new core family")
        if self.condition_format == "canonical_bits" and self.condition_mode != "reversible_record":
            if self.condition_dim != 259 + int(self.length_conditioning):
                raise ValueError("canonical condition needs 259 dimensions, plus one for known length")
        if self.condition_mode == "length_only" and (not self.length_conditioning or self.condition_format != "canonical_bits"):
            raise ValueError("length-only requires canonical known-length conditioning")
        if self.representation == "tokens":
            if self.condition_format != "canonical_bits" or self.masking_schedule is None or self.token_embedding_dim is None or self.token_temperature is None:
                raise ValueError("discrete specification requires explicit condition, schedule, embedding and temperature")
            if self.method in {"reversible", "shuffled", "zero"}:
                raise ValueError("codec-only controls are not actual-model positive/negative controls")
            if self.token_embedding_dim < 1 or not 0 < self.token_temperature < float("inf"):
                raise ValueError("invalid discrete model specification")
        if self.dataset_size < 1 or self.k < 1 or self.training_steps < 1 or self.batch_size < 1:
            raise ValueError("dataset size, budget, training steps, and batch size must be positive")
        if (
            self.diffusion_steps < 2
            or self.sampling_steps < 1
            or self.sampling_steps > self.diffusion_steps
            or self.learning_rate <= 0
            or not 1e-4 < self.beta_end < 1
            or self.prediction_type not in {"epsilon", "sample"}
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
    encoder, _, _ = _codec(config.representation, max_length=config.max_length, source=config.source)
    value = encoder.encode(record.message).flatten()
    if len(value) > config.condition_dim:
        raise ValueError("condition_dim is too small for the reversible record")
    condition = torch.zeros(config.condition_dim, dtype=torch.float32)
    condition[: len(value)] = value
    return condition


def _condition(record: DigestRecord, config: ExperimentConfig) -> Tensor:
    if config.condition_mode == "zero":
        return torch.zeros(config.condition_dim, dtype=torch.float32)
    if config.condition_mode == "reversible_record":
        return _record_condition(record, config)
    if config.condition_format == "canonical_bits":
        condition = digest_condition(record.algorithm, record.q, record.digest,
                                     length=len(record.message) if config.length_conditioning else None)
        if config.condition_mode == "length_only":
            condition[3:259] = 0
        return condition
    return _digest_condition(record, config) if config.representation == "bits" else _caption_condition(record, config)


def _conditions(records: Sequence[DigestRecord], config: ExperimentConfig) -> Tensor:
    conditions = torch.stack([_condition(record, config) for record in records])
    if config.condition_mode == "shuffled_hash":
        if config.condition_format == "canonical_bits":
            donors = shuffled_donors(records, seed=config.data_seed, same_length=config.length_conditioning)
            return conditions[donors]
        if len(conditions) < 2:
            raise ValueError("shuffled condition requires at least two records")
        return torch.roll(conditions, shifts=1, dims=0)
    return conditions


def _codec(representation: Representation, *, max_length: int = 31, source: str = "printable"):
    if representation == "bgv":
        settings = BGVConfig(max_message_length=max_length)
        return BGVEncoder(settings), BGVDecoder(settings), (2, settings.image_height, settings.image_width)
    if representation == "cgge":
        settings = CGGEConfig(max_message_length=max_length)
        return CGGEEncoder(settings), CGGEDecoder(settings), (2, settings.image_height, settings.image_width)
    if representation == "tokens":
        codec = TokenCodec(source, max_length)
        return codec, codec, (max_length + 1,)
    return DirectBitsEncoder(), DirectBitsDecoder(), (32, 8)


def _build_model(config: ExperimentConfig, shape: tuple[int, ...]):
    if config.representation == "tokens":
        codec = TokenCodec(config.source, config.max_length)
        return SequenceDenoiser(codec.vocabulary_size, shape[0], config.condition_dim,
                                width=config.width, embedding_dim=config.token_embedding_dim)
    if config.representation == "bits":
        return BitDenoiser(
            shape[0] * shape[1],
            config.condition_dim,
            width=max(64, config.width * 4),
            aligned_condition=config.condition_mode == "reversible_record" and config.condition_dim == prod(shape),
        )
    condition_shape = shape if config.condition_mode == "reversible_record" and config.condition_dim == prod(shape) else None
    return ImageUNet(shape[0], config.condition_dim, width=config.width, condition_shape=condition_shape)


def _training_data(records: Sequence[DigestRecord], config: ExperimentConfig, encoder, device: torch.device) -> tuple[Tensor, Tensor]:
    values = torch.stack([encoder.encode(record.message) for record in records]).to(device)
    if config.representation != "tokens":
        values = values * 2 - 1
    return values, _conditions(records, config).to(device)


def _train(
    model, records: Sequence[DigestRecord], config: ExperimentConfig, encoder, device: torch.device,
    *, checkpoint_path: Path | None = None,
) -> tuple[GaussianDiffusion, float]:
    values, conditions = _training_data(records, config, encoder, device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device=device).manual_seed(config.model_seed)
    diffusion = MaskedDiffusion(encoder.mask, config.masking_schedule, device=device) if config.representation == "tokens" else GaussianDiffusion(
        config.diffusion_steps,
        beta_end=config.beta_end,
        prediction_type=config.prediction_type,
        device=device,
    )
    loss = torch.tensor(float("nan"))
    first_step = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        from .experiment_state import sha256
        checksum = checkpoint_path.with_suffix('.sha256').read_text().strip()
        if sha256(checkpoint_path) != checksum:
            raise RuntimeError("checkpoint corruption; refuse automatic replacement")
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if saved['config'] != asdict(config):
            raise RuntimeError("incompatible checkpoint configuration; use a new run ID")
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator'].cpu())
        first_step, loss = saved['step'], torch.tensor(saved['loss'])
    for step in range(first_step, config.training_steps):
        indices = torch.randint(len(values), (config.batch_size,), device=device, generator=generator)
        loss = diffusion.loss(model, values[indices], conditions[indices], generator=generator)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if checkpoint_path is not None and ((step + 1) % 100 == 0 or step + 1 == config.training_steps):
            from .experiment_state import sha256
            temporary = checkpoint_path.with_suffix('.tmp')
            torch.save({'config': asdict(config), 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'generator': generator.get_state(), 'step': step + 1, 'loss': loss.item()}, temporary)
            temporary.replace(checkpoint_path)
            checkpoint_path.with_suffix('.sha256').write_text(sha256(checkpoint_path) + '\n')
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
            started = perf_counter()
            options = {"temperature": config.token_temperature} if config.representation == "tokens" else {}
            sample = diffusion.sample(model, condition[None, :], shape, sampling_steps=config.sampling_steps, generator=generator, **options)[0]
            decoded = decoder.decode(sample if config.representation == "tokens" else (sample + 1) / 2)
            group.append(CandidateAttempt(decoded.message, decoded.valid, decoded.reason,
                                          config.model_seed + 1, sample.cpu().tolist(), perf_counter() - started))
        attempts.append(tuple(group))
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
    corpus = tuple(message[:config.max_length] for message in corpus)
    encoder, decoder, _ = _codec(config.representation, max_length=config.max_length, source=config.source)
    return all((decoded := decoder.decode(encoder.encode(message))).valid and decoded.message == message for message in corpus)


def _run_experiment(config: ExperimentConfig, output_dir: str | Path) -> ExperimentResult:
    """Create data, execute one declared method, and preserve reproducibility artifacts."""
    torch.manual_seed(config.model_seed)
    random.seed(config.model_seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_spec = SourceSpec(config.source, config.dataset_size, seed=config.data_seed, max_length=config.max_length)
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
        attempts = source_prior_random_search(targets, k=config.k, seed=config.model_seed, length_aware=config.length_conditioning, max_length=config.max_length)
    elif config.method == "nearest_train":
        attempts = nearest_training_digest(targets, split["train"], k=config.k)
    elif config.method in {"reversible", "shuffled", "zero"}:
        attempts = representation_control(targets, representation=config.representation, kind=config.method, k=config.k, seed=config.model_seed)
    else:
        encoder, decoder, shape = _codec(config.representation, max_length=config.max_length, source=config.source)
        device = torch.device(config.device)
        model = _build_model(config, shape).to(device)
        started = perf_counter()
        if config.method == "predictor":
            if config.representation == "tokens":
                raise ValueError("use the registered shared non-diffusion baseline codec; categorical predictor not implemented")
            model = DirectPredictor(config.condition_dim, prod(shape), width=max(64, config.width * 4)).to(device)
            training_loss = _train_predictor(model, split["train"], config, encoder, shape, device)
            diffusion = None
        else:
            diffusion, training_loss = _train(model, split["train"], config, encoder, device,
                                              checkpoint_path=output / "training_resume.pt")
        metadata.update({"parameter_count": parameter_count(model), "training_wall_seconds": perf_counter() - started})
        parameter_total = parameter_count(model)
        torch.save({"config": asdict(config), "model_state": model.state_dict()}, output / "checkpoint.pt")
        attempts = (
            _predictor_attempts(model, targets, config, decoder, shape, device)
            if config.method == "predictor"
            else _diffusion_attempts(model, diffusion, targets, config, decoder, shape, device)
        )

    metadata["scope"] = "engineering_or_legacy_only_not_confirmatory"
    summary = write_evaluation(targets, attempts, output / "evaluation", method=config.method, k=config.k, metadata=metadata, max_length=config.max_length)
    _, decoder, shape = _codec(config.representation, max_length=config.max_length, source=config.source)
    manifest = {
        **metadata,
        "representation": config.representation,
        "representation_version": TOKENIZER_VERSION if config.representation == "tokens" else "bgv-strict-v2" if config.representation == "bgv" else GLYPH_TABLE_VERSION if config.representation == "cgge" else "direct-bits-v1",
        "image_shape": list(shape),
        "glyph_table_checksum": glyph_table_checksum() if config.representation == "cgge" else None,
        "glyph_distance_metric": "mse" if config.representation == "cgge" else None,
        "glyph_valid_threshold": decoder.config.glyph_valid_threshold if config.representation == "cgge" else None,
        "mask_threshold": decoder.config.mask_threshold if config.representation in {"bgv", "cgge"} else None,
        "condition_format": (
            "encoded_record"
            if config.condition_mode == "reversible_record"
            else "canonical_bits" if config.condition_format == "canonical_bits" else "digest_bits" if config.representation == "bits" else "ascii_caption_bytes"
        ),
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


def run_experiment(config: ExperimentConfig, output_dir: str | Path) -> ExperimentResult:
    """Engineering/legacy runner with exclusive output lock and compatible replay.

    A new-study confirmatory orchestrator must additionally require full G0/G1,
    external dataset manifests and validation-only checkpoint selection.
    """
    import fcntl
    from .automation import freeze_json, source_version
    from .experiment_state import sha256, write_json
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = output / 'configuration_frozen.json'
        if not snapshot.exists() and any(p.name != '.run.lock' for p in output.iterdir()):
            raise RuntimeError("unversioned existing output; preserve it and use a new run ID")
        freeze_json(snapshot, {'config': asdict(config), 'code_version': source_version(),
                               'scope': 'engineering_or_legacy_only_not_confirmatory'})
        complete = output / 'complete.json'
        if complete.exists():
            for name, checksum in json.loads(complete.read_text())['sha256'].items():
                if sha256(output / name) != checksum:
                    raise RuntimeError(f"completed artifact corrupted: {name}")
            manifest = json.loads((output / 'run_manifest.json').read_text())
            metrics = json.loads((output / 'evaluation/metrics.json').read_text())
            summary = EvaluationSummary(**{name: metrics[name] for name in EvaluationSummary.__dataclass_fields__})
            return ExperimentResult(summary, output, manifest['training_loss'], manifest.get('parameter_count'))
        result = _run_experiment(config, output)
        write_json(complete, {'sha256': {str(p.relative_to(output)): sha256(p) for p in output.rglob('*')
                                         if p.is_file() and p.name not in {'.run.lock','complete.json'}}})
        return result


__all__ = ["ExperimentConfig", "ExperimentResult", "run_experiment"]
