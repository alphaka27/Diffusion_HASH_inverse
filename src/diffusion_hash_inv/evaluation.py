"""Hashlib-based candidate verification and fixed-budget summary metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .dataset import DigestRecord

EVALUATOR_VERSION = "fixed-budget-domain-v2"
VERIFIER_VERSION = "hashlib-independent-msb-v2"


@dataclass(frozen=True)
class CandidateAttempt:
    """One generation attempt; invalid decodes intentionally still consume budget."""

    message: bytes | None
    valid: bool
    reason: str | None = None
    generation_seed: int | None = None
    raw_representation: list | None = None
    generation_seconds: float | None = None


@dataclass(frozen=True)
class Verification:
    digest: str | None
    prefix_match: bool
    full_digest_match: bool = False


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
    """Independent full rehash; binary-string extraction does not reuse generator code."""
    digest = hashlib.new(target.algorithm, candidate).digest()
    if not 1 <= target.q <= len(digest) * 8:
        raise ValueError("invalid target q")
    prefix = int("".join(f"{byte:08b}" for byte in digest)[:target.q], 2)
    return Verification(digest.hex(), prefix == int(target.prefix, 16), digest == target.digest)


def _in_source_domain(candidate: bytes, target: DigestRecord, *, min_length: int, max_length: int) -> bool:
    return (min_length <= len(candidate) <= max_length
            and (target.source == "random_bytes" or
                 target.source == "printable" and all(0x21 <= byte <= 0x7E for byte in candidate)))


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


def target_outcomes(targets, attempts_by_target, *, k: int, min_length: int = 4, max_length: int = 31):
    """Same domain validation and exact budget rules as the summary and ledger."""
    return _evaluate(targets, attempts_by_target, k=k, min_length=min_length, max_length=max_length)[1]


def exact_mcnemar(n10: int, n01: int, *, alternative: str = "greater") -> float:
    if min(n10, n01) < 0 or alternative not in {"greater", "two-sided"}:
        raise ValueError("invalid discordances or alternative")
    n = n10 + n01
    if not n:
        return 1.0
    upper = n01 if alternative == "greater" else min(n10, n01)
    coefficient = total = 1
    for k in range(1, upper + 1):
        coefficient = coefficient * (n - k + 1) // k
        total += coefficient
    return min(1.0, (2 if alternative == "two-sided" else 1) * total / 2**n)


def paired_comparison(
    model_outcomes: Sequence[bool],
    baseline_outcomes: Sequence[bool],
    *,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 10_000,
    alternative: str = "greater",
) -> PairedComparison:
    """Compute the preregistered paired McNemar test and bootstrap gain interval."""
    if not model_outcomes or len(model_outcomes) != len(baseline_outcomes) or bootstrap_samples < 1:
        raise ValueError("paired outcomes must be non-empty, aligned, and use positive bootstrap samples")
    n = len(model_outcomes)
    model_successes = sum(model_outcomes)
    baseline_successes = sum(baseline_outcomes)
    n10 = sum(model and not baseline for model, baseline in zip(model_outcomes, baseline_outcomes))
    n01 = sum(not model and baseline for model, baseline in zip(model_outcomes, baseline_outcomes))
    pvalue = exact_mcnemar(n10, n01, alternative=alternative)
    # Exactly target-level bootstrap for binary pairs, via their difference counts.
    import numpy as np
    generator = np.random.default_rng(bootstrap_seed)
    counts = generator.multinomial(n, [n01 / n, (n - n10 - n01) / n, n10 / n], size=bootstrap_samples)
    samples = sorted((counts[:, 2] - counts[:, 0]) / n)
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


def _evaluate(targets, attempts_by_target, *, k, min_length, max_length):
    if k < 1 or not targets or len(targets) != len(attempts_by_target) or not 0 <= min_length <= max_length:
        raise ValueError("nonempty targets, aligned attempts, positive budget and valid lengths required")
    keys = [(r.algorithm, r.q, r.prefix) for r in targets]
    if len(set(keys)) != len(keys):
        raise ValueError("evaluation unit must be a unique digest condition")
    valid_count = verification_count = length_matches = successes = exact_recoveries = length_successes = 0
    outcomes, ledger = [], []
    for target, attempts in zip(targets, attempts_by_target):
        if len(attempts) != k:
            raise ValueError("every target must consume exactly k candidate attempts")
        success = exact = length_success = False
        for index, attempt in enumerate(attempts):
            candidate = attempt.message
            if candidate is not None and not isinstance(candidate, bytes):
                raise TypeError("decoded candidate must be raw bytes or None")
            # Rehash every interpretable payload, including invalid-domain and duplicate outputs.
            verified = verify_candidate(candidate, target) if candidate is not None else None
            verification_count += verified is not None
            valid = bool(attempt.valid and candidate is not None and _in_source_domain(
                candidate, target, min_length=min_length, max_length=max_length))
            reason = attempt.reason if not attempt.valid else None
            if attempt.valid and not valid:
                reason = "missing_payload" if candidate is None else "source_domain"
            match_length = valid and len(candidate) == len(target.message)
            matched = valid and verified.prefix_match
            recovered = valid and candidate == target.message
            valid_count += valid
            length_matches += match_length
            success |= matched
            exact |= recovered
            length_success |= matched and match_length
            ledger.append(dict(target_id=target.id, target_digest=target.digest.hex(), target_prefix=target.prefix,
                               algorithm=target.algorithm, q=target.q, candidate_index=index, k_position=index + 1,
                               valid=valid, decode_success=attempt.valid, reason=reason,
                               message_hex=candidate.hex() if candidate is not None else None,
                               candidate_length=len(candidate) if candidate is not None else None,
                               actual_digest=verified.digest if verified else None,
                               prefix_match=verified.prefix_match if verified else False,
                               full_digest_match=verified.full_digest_match if verified else False,
                               exact_source_match=recovered, generation_seed=attempt.generation_seed,
                               raw_representation=attempt.raw_representation, generation_seconds=attempt.generation_seconds))
        outcomes.append(TargetOutcome(target.id, target.prefix, bool(success), bool(exact)))
        successes += success
        exact_recoveries += exact
        length_successes += length_success
    n, total = len(targets), len(targets) * k
    summary = EvaluationSummary(n, k, total, verification_count, valid_count / total,
                                successes / n, successes / n, exact_recoveries / n,
                                length_matches / total, length_successes / n,
                                binomial_ci95(successes, n), -math.expm1(math.log(0.05) / n) if successes == 0 else None)
    return summary, tuple(outcomes), ledger


def score_attempts(targets, attempts_by_target, *, k: int, min_length: int = 4, max_length: int = 31):
    """Score exactly k attempts. Format AND source domain are primary validity."""
    return _evaluate(targets, attempts_by_target, k=k, min_length=min_length, max_length=max_length)[0]


def write_evaluation(targets, attempts_by_target, output_dir, *, method: str, k: int,
                     metadata=None, min_length: int = 4, max_length: int = 31):
    """One verification pass feeds the metrics, outcomes and canonical JSON ledger."""
    summary, outcomes, ledger = _evaluate(targets, attempts_by_target, k=k,
                                          min_length=min_length, max_length=max_length)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "candidates.jsonl").open("w", encoding="utf-8") as output:
        for row in ledger:
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    result = {"method": method, "metadata": metadata or {}, "evaluator_version": EVALUATOR_VERSION,
              "verifier_version": VERIFIER_VERSION, "min_length": min_length, "max_length": max_length,
              "zero_success_rule_of_three": min(1.0, 3 / len(targets)) if summary.zero_success_upper95 else None,
              **asdict(summary)}
    (target / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
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
