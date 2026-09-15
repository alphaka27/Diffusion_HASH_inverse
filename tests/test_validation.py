"""End-to-end G0--G4 artifact validation."""

import json

from diffusion_hash_inv.validation import validate_confirmatory_runs


def _run(root, *, seed: int, method: str, condition: str = "hash", successes: bool = False) -> None:
    root.mkdir()
    evaluation = root / "evaluation"
    evaluation.mkdir()
    config = {
        "representation": "bits",
        "source": "printable",
        "algorithm": "md5",
        "q": 8,
        "dataset_size": 12,
        "data_seed": 0,
        "split_seed": 0,
        "model_seed": seed,
        "k": 1,
        "method": method,
        "condition_mode": condition,
        "length_conditioning": False,
        "test_limit": None,
    }
    metrics = {
        "target_count": 6,
        "candidate_budget": 1,
        "candidate_attempt_count": 6,
        "exact_source_recovery_at_k": 1.0 if condition == "reversible_record" else 0.0,
    }
    (root / "run_manifest.json").write_text(
        json.dumps({"config": config, "run_gates": {"g0_split_independence": True, "g1_round_trip": True, "g2_candidate_budget": True}}),
        encoding="utf-8",
    )
    (evaluation / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (evaluation / "outcomes.jsonl").write_text(
        "".join(json.dumps({"target_id": index, "target_prefix": f"{index:02x}", "preimage_success": successes, "exact_source_recovery": successes}) + "\n" for index in range(6)),
        encoding="utf-8",
    )


def test_validator_reports_truncated_evidence_for_a_complete_matrix(tmp_path) -> None:
    models = []
    baselines = []
    for seed in range(3):
        model = tmp_path / f"model-{seed}"
        baseline = tmp_path / f"baseline-{seed}"
        _run(model, seed=seed, method="diffusion", successes=True)
        _run(baseline, seed=seed, method="random")
        models.append(model)
        baselines.append(baseline)
    positive = tmp_path / "positive"
    _run(positive, seed=0, method="diffusion", condition="reversible_record", successes=True)
    result = validate_confirmatory_runs(models, {"random": baselines}, positive_control_run=positive, bootstrap_seed=7)
    assert result["gates"] == {"g0": True, "g1": True, "g2": True, "g3": True, "g4": True}
    assert result["evidence_level"] == "L2_TRUNCATED_EVIDENCE"
