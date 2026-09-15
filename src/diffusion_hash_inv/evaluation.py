"""Hashlib-based candidate verification and fixed-budget summary metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from typing import Mapping, Sequence

from .dataset import DigestRecord, digest_prefix_hex


@dataclass(frozen=True)
class CandidateAttempt:
    """One generation attempt; invalid decodes intentionally still consume budget."""

    message: bytes | None
    valid: bool
    reason: str | None = None


@dataclass(frozen=True)
class Verification:
    digest: str | None
    prefix_match: bool


@dataclass(frozen=True)
class EvaluationSummary:
    target_count: int
    candidate_budget: int
    candidate_attempt_count: int
    hash_verification_count: int
    valid_decode_rate: float
    preimage_success_at_k: float
    in_domain_preimage_success_at_k: float
    exact_source_recovery_at_k: float
    length_match_rate: float
    length_matched_preimage_success_at_k: float
    preimage_success_ci95: tuple[float, float]
    zero_success_upper95: float | None


@dataclass(frozen=True)
class TargetOutcome:
    target_id: int
    target_prefix: str
    preimage_success: bool
    exact_source_recovery: bool


@dataclass(frozen=True)
class PairedComparison:
    target_count: int
    model_successes: int
    baseline_successes: int
    n10: int
    n01: int
    absolute_gain: float
    relative_gain: float | None
    additional_solved_targets: int
    mcnemar_pvalue: float
    delta_ci95: tuple[float, float]


def verify_candidate(candidate: bytes, target: DigestRecord) -> Verification:
    """Rehash a decoded candidate and compare only the experiment's q bits."""
    digest = hashlib.new(target.algorithm, candidate).digest()
    return Verification(digest.hex(), digest_prefix_hex(digest, target.q) == target.prefix)


def _in_source_domain(candidate: bytes, target: DigestRecord) -> bool:
    if not 4 <= len(candidate) <= 31:
        return False
    return target.source == "random_bytes" or all(0x21 <= byte <= 0x7E for byte in candidate)


def binomial_ci95(successes: int, trials: int) -> tuple[float, float]:
    """Wilson 95% interval; for zero successes this also gives the useful bound."""
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("successes must be within a positive trial count")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def target_outcomes(
    targets: Sequence[DigestRecord], attempts_by_target: Sequence[Sequence[CandidateAttempt]], *, k: int
) -> tuple[TargetOutcome, ...]:
    """Return one binary success outcome for every unique evaluated target."""
    if k < 1 or len(targets) != len(attempts_by_target):
        raise ValueError("targets and attempt groups must align and k must be positive")
    outcomes = []
    for target, attempts in zip(targets, attempts_by_target):
        if len(attempts) != k:
            raise ValueError("every target must consume exactly k candidate attempts")
        success = exact = False
        for attempt in attempts:
            if attempt.valid and attempt.message is not None:
                success |= verify_candidate(attempt.message, target).prefix_match
                exact |= attempt.message == target.message
        outcomes.append(TargetOutcome(target.id, target.prefix, success, exact))
    return tuple(outcomes)


def paired_comparison(
    model_outcomes: Sequence[bool],
    baseline_outcomes: Sequence[bool],
    *,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 10_000,
) -> PairedComparison:
    """Compute the preregistered paired McNemar test and bootstrap gain interval."""
    if not model_outcomes or len(model_outcomes) != len(baseline_outcomes) or bootstrap_samples < 1:
        raise ValueError("paired outcomes must be non-empty, aligned, and use positive bootstrap samples")
    n = len(model_outcomes)
    model_successes = sum(model_outcomes)
    baseline_successes = sum(baseline_outcomes)
    n10 = sum(model and not baseline for model, baseline in zip(model_outcomes, baseline_outcomes))
    n01 = sum(not model and baseline for model, baseline in zip(model_outcomes, baseline_outcomes))
    discordant = n10 + n01
    pvalue = sum(comb(discordant, value) for value in range(n10, discordant + 1)) / 2**discordant if discordant else 1.0
    differences = tuple(int(model) - int(baseline) for model, baseline in zip(model_outcomes, baseline_outcomes))
    generator = random.Random(bootstrap_seed)
    samples = sorted(sum(generator.choices(differences, k=n)) / n for _ in range(bootstrap_samples))
    return PairedComparison(
        target_count=n,
        model_successes=model_successes,
        baseline_successes=baseline_successes,
        n10=n10,
        n01=n01,
        absolute_gain=(model_successes - baseline_successes) / n,
        relative_gain=model_successes / baseline_successes if baseline_successes else None,
        additional_solved_targets=model_successes - baseline_successes,
        mcnemar_pvalue=pvalue,
        delta_ci95=(samples[int(0.025 * (bootstrap_samples - 1))], samples[int(0.975 * (bootstrap_samples - 1))]),
    )


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Return Holm-adjusted p-values for one preregistered comparison family."""
    if any(not 0 <= value <= 1 for value in pvalues.values()):
        raise ValueError("p-values must be within [0, 1]")
    adjusted: dict[str, float] = {}
    previous = 0.0
    count = len(pvalues)
    for index, (name, value) in enumerate(sorted(pvalues.items(), key=lambda item: item[1])):
        previous = max(previous, min(1.0, (count - index) * value))
        adjusted[name] = previous
    return adjusted


def score_attempts(
    targets: Sequence[DigestRecord], attempts_by_target: Sequence[Sequence[CandidateAttempt]], *, k: int
) -> EvaluationSummary:
    """Score exactly k attempts per target without retrying invalid candidates."""
    if k < 1 or len(targets) != len(attempts_by_target):
        raise ValueError("targets and attempt groups must align and k must be positive")
    valid_count = verification_count = length_matches = successes = in_domain_successes = exact_recoveries = length_matched_successes = 0
    for target, attempts in zip(targets, attempts_by_target):
        if len(attempts) != k:
            raise ValueError("every target must consume exactly k candidate attempts")
        target_success = target_in_domain_success = target_exact = target_length_matched_success = False
        for attempt in attempts:
            if not attempt.valid or attempt.message is None:
                continue
            valid_count += 1
            length_matches += len(attempt.message) == len(target.message)
            verification_count += 1
            verified = verify_candidate(attempt.message, target)
            target_success |= verified.prefix_match
            target_in_domain_success |= verified.prefix_match and _in_source_domain(attempt.message, target)
            target_length_matched_success |= verified.prefix_match and len(attempt.message) == len(target.message)
            target_exact |= attempt.message == target.message
        successes += target_success
        in_domain_successes += target_in_domain_success
        length_matched_successes += target_length_matched_success
        exact_recoveries += target_exact
    total_attempts = len(targets) * k
    return EvaluationSummary(
        target_count=len(targets),
        candidate_budget=k,
        candidate_attempt_count=total_attempts,
        hash_verification_count=verification_count,
        valid_decode_rate=valid_count / total_attempts if total_attempts else 0.0,
        preimage_success_at_k=successes / len(targets) if targets else 0.0,
        in_domain_preimage_success_at_k=in_domain_successes / len(targets) if targets else 0.0,
        exact_source_recovery_at_k=exact_recoveries / len(targets) if targets else 0.0,
        length_match_rate=length_matches / total_attempts if total_attempts else 0.0,
        length_matched_preimage_success_at_k=length_matched_successes / len(targets) if targets else 0.0,
        preimage_success_ci95=binomial_ci95(successes, len(targets)) if targets else (0.0, 0.0),
        zero_success_upper95=min(1.0, 3 / len(targets)) if not successes and targets else None,
    )


def write_evaluation(
    targets: Sequence[DigestRecord],
    attempts_by_target: Sequence[Sequence[CandidateAttempt]],
    output_dir: str | Path,
    *,
    method: str,
    k: int,
    metadata: dict[str, object] | None = None,
) -> EvaluationSummary:
    """Persist a summary and every attempt, including invalid budget spend."""
    summary = score_attempts(targets, attempts_by_target, k=k)
    outcomes = target_outcomes(targets, attempts_by_target, k=k)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "candidates.jsonl").open("w", encoding="utf-8") as output:
        for record, attempts in zip(targets, attempts_by_target):
            for index, attempt in enumerate(attempts):
                verification = verify_candidate(attempt.message, record) if attempt.valid and attempt.message is not None else None
                output.write(
                    json.dumps(
                        {
                            "target_id": record.id,
                            "candidate_index": index,
                            "valid": attempt.valid,
                            "reason": attempt.reason,
                            "message_hex": attempt.message.hex() if attempt.message is not None else None,
                            "candidate_length": len(attempt.message) if attempt.message is not None else None,
                            "actual_digest": verification.digest if verification else None,
                            "prefix_match": verification.prefix_match if verification else False,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    result = {"method": method, "metadata": metadata or {}, **asdict(summary)}
    (target / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (target / "metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(asdict(summary)))
        writer.writeheader()
        writer.writerow(asdict(summary))
    with (target / "outcomes.jsonl").open("w", encoding="utf-8") as output:
        for outcome in outcomes:
            output.write(json.dumps(asdict(outcome), sort_keys=True) + "\n")
    return summary


__all__ = [
    "CandidateAttempt",
    "EvaluationSummary",
    "PairedComparison",
    "TargetOutcome",
    "Verification",
    "binomial_ci95",
    "holm_adjust",
    "paired_comparison",
    "score_attempts",
    "target_outcomes",
    "verify_candidate",
    "write_evaluation",
]
