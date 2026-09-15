"""Aggregate independently-run methods into the preregistered G0--G4 decision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .evaluation import holm_adjust, paired_comparison


_MATCHED_SETTINGS = (
    "source",
    "algorithm",
    "q",
    "k",
    "dataset_size",
    "data_seed",
    "split_seed",
    "test_limit",
    "length_conditioning",
)


@dataclass(frozen=True)
class RunArtifact:
    path: Path
    config: dict[str, object]
    metrics: dict[str, object]
    outcomes: dict[tuple[int, str], bool]
    run_gates: dict[str, bool]


def read_run_artifact(path: str | Path) -> RunArtifact:
    """Load the minimal immutable artifacts required for paired comparison."""
    root = Path(path)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    outcomes: dict[tuple[int, str], bool] = {}
    for line in (root / "evaluation" / "outcomes.jsonl").read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        key = (value["target_id"], value["target_prefix"])
        if key in outcomes:
            raise ValueError(f"duplicate target outcome in {root}: {key}")
        outcomes[key] = bool(value["preimage_success"])
    if not outcomes:
        raise ValueError(f"no target outcomes in {root}")
    return RunArtifact(root, manifest["config"], metrics, outcomes, manifest.get("run_gates", {}))


def _by_seed(paths: Sequence[str | Path]) -> dict[int, RunArtifact]:
    artifacts = {int((artifact := read_run_artifact(path)).config["model_seed"]): artifact for path in paths}
    if len(artifacts) != len(paths):
        raise ValueError("each seed must have exactly one run artifact")
    return artifacts


def _matched(model: RunArtifact, baseline: RunArtifact) -> bool:
    if any(model.config.get(name) != baseline.config.get(name) for name in _MATCHED_SETTINGS):
        return False
    if model.outcomes.keys() != baseline.outcomes.keys():
        return False
    for artifact in (model, baseline):
        target_count = artifact.metrics.get("target_count")
        budget = artifact.config.get("k")
        if artifact.metrics.get("candidate_attempt_count") != target_count * budget:
            return False
        if artifact.metrics.get("candidate_budget") != budget or not artifact.run_gates.get("g2_candidate_budget"):
            return False
    return True


def _outcome_vector(model: RunArtifact, baseline: RunArtifact) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    keys = sorted(model.outcomes)
    return tuple(model.outcomes[key] for key in keys), tuple(baseline.outcomes[key] for key in keys)


def _positive_control_passes(model: RunArtifact, positive: RunArtifact | None) -> bool:
    if positive is None:
        return False
    if positive.config.get("method") not in {"diffusion", "predictor"} or positive.config.get("condition_mode") != "reversible_record":
        return False
    if any(model.config.get(name) != positive.config.get(name) for name in _MATCHED_SETTINGS if name != "k"):
        return False
    return (
        positive.metrics.get("exact_source_recovery_at_k", 0.0) >= 0.99
        and positive.run_gates.get("g0_split_independence", False)
        and positive.run_gates.get("g1_round_trip", False)
        and positive.run_gates.get("g2_candidate_budget", False)
    )


def validate_confirmatory_runs(
    model_runs: Sequence[str | Path],
    baseline_runs: Mapping[str, Sequence[str | Path]],
    *,
    positive_control_run: str | Path | None = None,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Evaluate a fixed three-seed model/baseline matrix and return G0--G4."""
    models = _by_seed(model_runs)
    baselines = {name: _by_seed(paths) for name, paths in baseline_runs.items()}
    expected_seeds = {0, 1, 2}
    model_zero = models.get(0)
    positive = read_run_artifact(positive_control_run) if positive_control_run is not None else None
    all_artifacts = [*models.values(), *(artifact for runs in baselines.values() for artifact in runs.values())]
    g0 = bool(all_artifacts) and all(artifact.run_gates.get("g0_split_independence", False) for artifact in all_artifacts)
    g1 = model_zero is not None and all(artifact.run_gates.get("g1_round_trip", False) for artifact in models.values()) and _positive_control_passes(model_zero, positive)
    g2 = model_zero is not None and bool(baselines) and all(
        seed in models and seed in runs and _matched(models[seed], runs[seed])
        for runs in baselines.values()
        for seed in expected_seeds
    )

    comparisons = {}
    if model_zero is not None and g2:
        raw = {}
        for name, runs in baselines.items():
            model_outcomes, baseline_outcomes = _outcome_vector(model_zero, runs[0])
            result = paired_comparison(model_outcomes, baseline_outcomes, bootstrap_seed=bootstrap_seed)
            comparisons[name] = asdict(result)
            raw[name] = result.mcnemar_pvalue
        for name, adjusted in holm_adjust(raw).items():
            comparisons[name]["holm_adjusted_pvalue"] = adjusted
    g3 = bool(comparisons) and all(
        value["holm_adjusted_pvalue"] < 0.05 and value["delta_ci95"][0] > 0 for value in comparisons.values()
    )

    seed_gains = {}
    for name, runs in baselines.items():
        gains = {}
        for seed in expected_seeds & models.keys() & runs.keys():
            model_outcomes, baseline_outcomes = _outcome_vector(models[seed], runs[seed])
            gains[str(seed)] = (sum(model_outcomes) - sum(baseline_outcomes)) / len(model_outcomes)
        seed_gains[name] = gains
    g4 = bool(seed_gains) and all(set(gains) == {"0", "1", "2"} and all(gain > 0 for gain in gains.values()) for gains in seed_gains.values())

    if not (g0 and g1 and g2):
        evidence = "L0_INVALID"
    elif not (g3 and g4):
        evidence = "L1_NO_EVIDENCE"
    elif model_zero is not None and int(model_zero.config["q"]) < hashlib.new(str(model_zero.config["algorithm"])).digest_size * 8:
        evidence = "L2_TRUNCATED_EVIDENCE"
    else:
        evidence = "L3_FULL_DIGEST_EVIDENCE"
    return {
        "gates": {"g0": g0, "g1": g1, "g2": g2, "g3": g3, "g4": g4},
        "comparisons": comparisons,
        "seed_gains": seed_gains,
        "evidence_level": evidence,
        "test_adaptation_note": "External preregistration remains required; artifacts can only verify split and run consistency.",
    }


def write_confirmatory_validation(
    output_path: str | Path,
    model_runs: Sequence[str | Path],
    baseline_runs: Mapping[str, Sequence[str | Path]],
    *,
    positive_control_run: str | Path | None = None,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Persist one machine-readable G0--G4 decision."""
    result = validate_confirmatory_runs(
        model_runs,
        baseline_runs,
        positive_control_run=positive_control_run,
        bootstrap_seed=bootstrap_seed,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


__all__ = ["RunArtifact", "read_run_artifact", "validate_confirmatory_runs", "write_confirmatory_validation"]
