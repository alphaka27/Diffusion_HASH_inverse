"""Baseline and representation-control execution checks."""

import json

from diffusion_hash_inv.baselines import source_prior_random_search
from diffusion_hash_inv.controls import representation_control
from diffusion_hash_inv.dataset import SourceSpec, build_digest_records, generate_source_messages
from diffusion_hash_inv.evaluation import score_attempts, write_evaluation


def _targets() -> tuple:
    messages = generate_source_messages(SourceSpec("printable", 4, seed=2))
    return build_digest_records(messages, source="printable", algorithm="md5", q=128)


def test_reversible_controls_use_the_real_bgv_and_cgge_decoders() -> None:
    targets = _targets()
    for representation in ("bgv", "cgge", "bits"):
        result = score_attempts(targets, representation_control(targets, representation=representation, kind="reversible", k=2), k=2)
        assert result.preimage_success_at_k == result.exact_source_recovery_at_k == 1.0
        assert result.hash_verification_count == len(targets) * 2


def test_shuffled_control_has_no_exact_source_recoveries() -> None:
    targets = _targets()
    attempts = representation_control(targets, representation="cgge", kind="shuffled", k=1, seed=5)
    assert score_attempts(targets, attempts, k=1).exact_source_recovery_at_k == 0.0


def test_length_aware_random_baseline_is_reproducible_and_artifacts_include_all_attempts(tmp_path) -> None:
    targets = _targets()
    attempts = source_prior_random_search(targets, k=3, seed=9, length_aware=True)
    assert attempts == source_prior_random_search(targets, k=3, seed=9, length_aware=True)
    assert all(len(attempt.message) == len(target.message) for target, group in zip(targets, attempts) for attempt in group)
    summary = write_evaluation(targets, attempts, tmp_path, method="random", k=3)
    assert summary.hash_verification_count == len(targets) * 3
    assert len((tmp_path / "candidates.jsonl").read_text().splitlines()) == len(targets) * 3
    assert json.loads((tmp_path / "metrics.json").read_text())["method"] == "random"
