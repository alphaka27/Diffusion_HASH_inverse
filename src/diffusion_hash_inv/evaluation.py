"""Hashlib-based candidate verification and fixed-budget summary metrics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

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
    hash_verification_count: int
    valid_decode_rate: float
    preimage_success_at_k: float
    exact_source_recovery_at_k: float
    length_match_rate: float
    preimage_success_ci95: tuple[float, float]


def verify_candidate(candidate: bytes, target: DigestRecord) -> Verification:
    """Rehash a decoded candidate and compare only the experiment's q bits."""
    digest = hashlib.new(target.algorithm, candidate).digest()
    return Verification(digest.hex(), digest_prefix_hex(digest, target.q) == target.prefix)


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


def score_attempts(
    targets: Sequence[DigestRecord], attempts_by_target: Sequence[Sequence[CandidateAttempt]], *, k: int
) -> EvaluationSummary:
    """Score exactly k attempts per target without retrying invalid candidates."""
    if k < 1 or len(targets) != len(attempts_by_target):
        raise ValueError("targets and attempt groups must align and k must be positive")
    valid_count = verification_count = length_matches = successes = exact_recoveries = 0
    for target, attempts in zip(targets, attempts_by_target):
        if len(attempts) != k:
            raise ValueError("every target must consume exactly k candidate attempts")
        target_success = target_exact = False
        for attempt in attempts:
            if not attempt.valid or attempt.message is None:
                continue
            valid_count += 1
            length_matches += len(attempt.message) == len(target.message)
            verification_count += 1
            verified = verify_candidate(attempt.message, target)
            target_success |= verified.prefix_match
            target_exact |= attempt.message == target.message
        successes += target_success
        exact_recoveries += target_exact
    total_attempts = len(targets) * k
    return EvaluationSummary(
        target_count=len(targets),
        candidate_budget=k,
        hash_verification_count=verification_count,
        valid_decode_rate=valid_count / total_attempts if total_attempts else 0.0,
        preimage_success_at_k=successes / len(targets) if targets else 0.0,
        exact_source_recovery_at_k=exact_recoveries / len(targets) if targets else 0.0,
        length_match_rate=length_matches / total_attempts if total_attempts else 0.0,
        preimage_success_ci95=binomial_ci95(successes, len(targets)) if targets else (0.0, 0.0),
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
    return summary


__all__ = [
    "CandidateAttempt",
    "EvaluationSummary",
    "Verification",
    "binomial_ci95",
    "score_attempts",
    "verify_candidate",
    "write_evaluation",
]
