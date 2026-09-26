"""Check the v3.1 plan's arithmetic and fixed contracts; never run experiments."""

import hashlib
import json
import math
import re
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "examples/poc-v3.1-protocol.json"
    spec = json.loads(path.read_text())
    document = (root / "RESEARCH_PLAN_V3_1.md").read_text()
    pipelines = spec["pipelines"]
    sources = {entry["source"] for entry in pipelines.values()}
    profiles = spec["model_profiles"]
    selection = spec["profile_selection"]
    pilot = spec["pilot"]
    p1, p2a, p2b, p3 = (pilot[key] for key in ("P1", "P2A", "P2B", "P3"))
    e0, formal = spec["rehearsal"], spec["main"]
    batch = spec["training"]["batch_size"]
    assert spec["revision"] == "3.1"
    assert spec["execution_status_at_authoring"] == "PLAN_ONLY_IMPLEMENTATION_REQUIRED"
    assert spec["planned_cli_implemented_at_authoring"] is False
    assert spec["objective"]["positive_scientific_result_required_for_completion"] is False
    assert spec["objective"]["partial_counts_as_complete"] is False
    assert spec["pipeline_order"] == list(pipelines)
    assert len(pipelines) == 5 and len(sources) == 2
    assert spec["seeds"]["model_labels"] == formal["model_seeds"] == p3["model_seeds"] == [0, 1, 2]
    assert p2a["model_seeds"] == [99] and p2b["model_seeds"] == [100]
    assert selection["gaussian_order"] == ["G0", "G1", "G2"]
    assert selection["discrete_order"] == ["D0", "D1"]
    orders = [selection[entry["model"] + "_order"] for entry in pipelines.values()]
    pairs = sum(map(len, orders))
    assert pairs == selection["maximum_profile_pipeline_pairs"] == 13
    assert pilot["P0"]["model_profile_pipeline_pairs"] == pairs
    assert profiles["G2"]["prediction"] == "sample" and profiles["G2"]["intermediate_clean_clipping"]
    assert profiles["D1"]["length_head"]["length_values"] == list(range(4, 32))
    assert profiles["D1"]["sampling_nfe_per_candidate"] == 1 + profiles["D1"]["sampling_steps"] == 33
    assert p1["training_runs"] == 2 * pairs == 26
    assert p1["learned_candidates"] == p1["training_runs"] * p1["evaluation_trials"] * p1["k"]
    assert p1["random_candidates"] == len(sources) * p1["evaluation_trials"] * p1["k"]
    all_nfe = sum(profiles[name]["sampling_nfe_per_candidate"] for order in orders for name in order)
    assert p1["sampling_nfe"] == all_nfe * 2 * p1["evaluation_trials"] * p1["k"]
    assert p2a["evaluation_conditions"] == 2 * p2a["train_complement_pairs"] == 16
    assert p2a["gate_denominator_per_variant"] == p2a["evaluation_conditions"] * p2a["trajectories_per_condition_variant"]
    assert p2a["candidates_per_run"] == 2 * p2a["gate_denominator_per_variant"] == 128
    assert p2b["probe_candidates_per_run"] == len(p2b["probe_epochs"]) * p2b["probe_conditions"] * 2 * p2b["k"]
    for stage, denominator, joint, wrong in ((p2a, 64, 61, 1), (p2b, 128, 122, 3)):
        assert stage["gate_denominator_per_variant"] == denominator
        assert stage["normal_joint_min"] == stage["flipped_joint_min"] == joint <= denominator
        assert stage["wrong_original_max"] == wrong < denominator
        assert stage["training_runs_min"] == len(pipelines) and stage["training_runs_max"] == pairs
    gate = spec["synthetic"]["formal_gate"]
    assert gate["normal_joint_min"] == gate["flipped_joint_min"] == 475
    assert gate["wrong_original_max"] == 15 and gate["denominator"] == 512
    nfe_bounds = [sum(choose(profiles[name]["sampling_nfe_per_candidate"] for name in order) for order in orders)
                  for choose in (min, max)]
    for stage in (p1, p2b, e0, p3, formal):
        messages = stage.get("train_messages", stage.get("train_unique_messages"))
        assert stage["optimizer_updates_per_run"] == math.ceil(messages / batch) * stage["epochs"]
    for stage, methods, trials, variants in ((e0, 2, e0["evaluation_trials"], 1),
                                            (p3, 1, p3["evaluation_unique_conditions"], 2),
                                            (formal, 2, formal["evaluation_trials"], 1)):
        seeds = len(stage["model_seeds"])
        runs = len(pipelines) * methods * seeds
        assert stage.get("training_runs", stage.get("training_runs_full_matrix")) == runs
        per_run = trials * variants * stage["k"]
        learned = stage.get("learned_candidates", stage.get("candidates", stage.get("learned_candidates_full_matrix")))
        assert learned == runs * per_run
        for bound, nfe in zip(("min", "max"), nfe_bounds):
            assert stage["sampling_nfe_" + bound] == nfe * methods * seeds * per_run
        if methods == 2:
            assert stage.get("random_candidates", stage.get("shared_random_candidates")) == len(sources) * seeds * per_run
    assert formal["train_digest_slots"] + formal["validation_digest_slots"] + formal["test_pool_size"] == 4096
    assert formal["candidate_rows_full_matrix"] == formal["learned_candidates_full_matrix"] + formal["shared_random_candidates"]
    assert formal["requires_all_pipelines_qualified"] and not formal["partial_execution_counts_as_complete"]
    statistics = spec["statistics"]
    assert statistics["family_size"] == len(pipelines) and not statistics["ci_is_additional_gate"]
    assert statistics["calibration_repetitions_per_scenario"] == 20000
    assert set(statistics["calibration_scenarios"]) == set(statistics["calibration_case_probabilities"])
    assert len(statistics["calibration_scenarios"]) == 7
    execution = spec["execution"]
    profile_candidates = sum(execution["profile_batch_candidates"]) * (execution["profile_warmup_batches_per_size"] + execution["profile_measured_batches_per_size"])
    assert profile_candidates == 340
    for bound, nfe in zip(("min", "max"), nfe_bounds):
        assert execution["profile_sampling_nfe_" + bound] == profile_candidates * nfe
    caps = execution["hard_stage_active_wall_seconds"]
    assert sum(caps[key] for key in ("P0", "P1", "P2", "P3", "E0", "STATISTICS")) <= caps["POC_TOTAL"]
    totals = []
    for count in (len(pipelines), pairs):
        totals.append({
            "runs": p1["training_runs"] + count * 2 + e0["training_runs"] + p3["training_runs"] + formal["training_runs_full_matrix"],
            "updates": p1["training_runs"] * p1["optimizer_updates_per_run"] + count * (p2a["optimizer_updates_per_run"] + p2b["optimizer_updates_per_run"]) + e0["training_runs"] * e0["optimizer_updates_per_run"] + p3["training_runs"] * p3["optimizer_updates_per_run"] + formal["training_runs_full_matrix"] * formal["optimizer_updates_per_run"],
            "evaluation_rows": p1["learned_candidates"] + p1["random_candidates"] + count * (p2a["candidates_per_run"] + p2b["probe_candidates_per_run"]) + e0["learned_candidates"] + e0["random_candidates"] + p3["candidates"] + formal["candidate_rows_full_matrix"],
        })
    assert totals == [{"runs": 91, "updates": 795288, "evaluation_rows": 7435520},
                      {"runs": 107, "updates": 936888, "evaluation_rows": 7442688}]
    for key in totals[0]:
        assert f'{totals[0][key]:,}–{totals[1][key]:,}' in document
    for stage in (formal, p3):
        assert f'{stage["sampling_nfe_min"]:,}–{stage["sampling_nfe_max"]:,}' in document
    for link in re.findall(r"\]\(([^)]+)\)", document):
        if not link.startswith("https://"):
            assert (root / link).is_file(), link
    print(json.dumps({"status": "PLAN_CONSISTENCY_PASS", "experiment_gates": "NOT_RUN",
                      "spec_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "success_path_min": totals[0], "success_path_max": totals[1]}, indent=2))


if __name__ == "__main__":
    main()
