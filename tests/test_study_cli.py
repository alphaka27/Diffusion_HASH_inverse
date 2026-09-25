"""Small real-model integration checks; these do not award formal Pilot gates."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
import torch

from diffusion_hash_inv import study_cli
from diffusion_hash_inv import study_pilot as pilot


SPEC = Path(__file__).parents[1] / "examples" / "poc-v3-protocol.json"


def small_protocol():
    p = deepcopy(pilot.load_protocol(SPEC))
    p["synthetic"]["train_unique_messages"] = 16
    p["training"]["batch_size"] = 2
    p["training"]["validation_draws_per_condition"] = 1
    p["pilot"]["P1"].update(train_messages=8, validation_conditions=2, evaluation_trials=2)
    p["models"]["gaussian"].update(diffusion_steps=10, sampling_steps=2)
    p["models"]["discrete"]["sampling_steps"] = 2
    p["execution"]["minimum_disk_free_gib"] = 0
    return p


def arguments(root, stage, resume=False):
    return SimpleNamespace(workdir=root, stage=stage, device="cpu", development=True, threads=1, resume=resume)


def test_cli_validation_and_read_only_dry_run(tmp_path, capsys):
    root = tmp_path / "never-created"
    common = ["--protocol", str(SPEC), "--workdir", str(root)]
    assert study_cli.main(["pilot", "--stage", "P1", *common, "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["settings"]["training_runs"] == 10
    assert not root.exists()
    assert study_cli.main(["pilot", "--stage", "P0", *common, "--device", "cpu"]) == 2
    changed = json.loads(SPEC.read_text())
    changed["pilot"]["P1"]["epochs"] = 1
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(changed))
    assert study_cli.main(["pilot", "--stage", "P1", "--protocol", str(invalid), "--workdir", str(root), "--dry-run"]) == 2
    assert not root.exists()


def test_small_all_pipeline_training_generation_and_recovery(tmp_path):
    torch.set_num_threads(1)
    p = small_protocol()
    assert pilot.run_pilot(p, arguments(tmp_path, "P0"))["status"] == "PASS"
    result = pilot.run_pilot(p, arguments(tmp_path, "P1"))
    assert result["status"] == "PASS" and result["development_only"]
    summary = pilot.read_json(tmp_path / "pilot/P1/summary.json")
    assert len(summary["runs"]) == 12  # ten learned runs and two shared random streams
    for name, row in summary["runs"].items():
        if name.startswith("random/"):
            assert row["candidates"] == 20
            continue
        assert row["updates"] == 8
        assert row["metrics"]["candidates"] == 20
        assert set(row["metrics"]["success_at_k"]) == {"1", "10"}
        if name.endswith("/main"):
            assert row["recovery"]["status"] == "PASS"
    assert pilot.write_report(tmp_path, p)["main_ready"] is False
    with pytest.raises(pilot.PilotError, match="already complete"):
        pilot.run_pilot(p, arguments(tmp_path, "P1", resume=True))
    ledger = tmp_path / "pilot/P1/runs/P-DISC/0/main/candidates.sqlite"
    with sqlite3.connect(ledger) as connection:
        connection.execute("DELETE FROM candidates WHERE rowid=1")
    with pytest.raises(pilot.PilotError, match="modified"):
        pilot.write_report(tmp_path, p)


def test_independent_rng_sampler_matches_existing_single_candidate():
    torch.set_num_threads(1)
    p = small_protocol()
    device = torch.device("cpu")
    for name in p["pipeline_order"]:
        model, diffusion = pilot.model_and_diffusion(p, name, device, 7)
        shape = pilot.codecs(p["pipelines"][name])[2]
        expected = diffusion.sample(model, pilot.condition([3], device), shape,
                                    sampling_steps=2, generator=pilot.generator(device, 81),
                                    **({"temperature": 1.} if isinstance(diffusion, pilot.MaskedDiffusion) else {}))
        actual = pilot.sample(p, name, model, diffusion, [3], [81], device)
        assert torch.equal(expected.cpu(), actual)
    assert pilot.seed(p, "P1", "generation", unit_id="1", attempt=1) != pilot.seed(p, "P1", "generation", unit_id="2", attempt=1)


def test_resume_rejects_changed_environment_and_budget_is_cumulative(tmp_path):
    p = small_protocol()
    pilot.run_pilot(p, arguments(tmp_path, "P0"))
    args = arguments(tmp_path, "P1")
    args.threads = 2
    with pytest.raises(pilot.PilotError, match="environment"):
        pilot.run_pilot(p, args)
    state = {"active_seconds": 601.}
    budget = pilot.Budget(tmp_path, "P0", state, p, torch.device("cpu"))
    with pytest.raises(pilot.PilotError) as caught:
        budget.check()
    assert caught.value.code == 5 and state["active_seconds"] >= 601


def test_p3_normal_flip_counts_and_failure_are_separate_from_completion(tmp_path):
    p = small_protocol()
    p["pipeline_order"] = ["P-DISC"]
    p["pilot"]["P3"].update(train_messages=4, validation_conditions=2, epochs=1, validation_epochs=[1], model_seeds=[0, 1, 2])
    data = pilot.make_data(p)
    data["pairs"]["test"] = [[1, 4094]]
    pilot.atomic_json(tmp_path / "resources.json", {"pipelines": {"P-DISC": {"batch_size": 2, "p3_run_seconds": 200}}, "p3_soft_wall_seconds": 600})
    state = {"active_seconds": 0.}
    budget = pilot.Budget(tmp_path, "P3", state, p, torch.device("cpu"))
    result = pilot.execute_models(p, "P3", tmp_path, data, torch.device("cpu"), budget)
    assert result["evaluation_complete"] is True
    assert result["status"] == "PILOT_EVALUATION_COMPLETE" and result["exit_code"] == 2
    assert len(result["runs"]) == 3
    for row in result["runs"].values():
        assert row["metrics"]["candidates"] == 4


def test_profile_grid_and_resource_block(tmp_path):
    p = small_protocol()
    data = pilot.make_data(p)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    budget = pilot.Budget(tmp_path, "P2", {"active_seconds": 0.}, p, device)
    model, diffusion = pilot.model_and_diffusion(p, "P-DISC", device, 0)
    state = {"training_update_seconds": [.01] * 320, "validation_seconds": [.1], "checkpoint_seconds": [.01]}
    profile = pilot.profile(p, "P-DISC", model, diffusion, state, data, tmp_path, device, budget)
    assert profile["profile_candidates"] == 340
    assert profile["selected_batch"] in (1, 4, 16, 64)
    profiles = {name: dict(profile, update_seconds=100.) for name in p["pipeline_order"]}
    assert pilot.seal_resources(p, tmp_path, profiles)["status"] == "BLOCKED_RESOURCE"


def test_deterministic_embedding_preserves_lookup_and_gradient():
    torch.manual_seed(8)
    ordinary = torch.nn.Embedding(259, 16)
    dense = pilot.DeterministicEmbedding.from_pretrained(ordinary.weight.detach().clone(), freeze=False)
    indices = torch.tensor([[258, 258, 1, 0, 258, 1, 257]])
    weights = torch.randn(1, 7, 16)
    assert torch.equal(ordinary(indices), dense(indices))
    (ordinary(indices) * weights).sum().backward()
    (dense(indices) * weights).sum().backward()
    assert torch.allclose(ordinary.weight.grad, dense.weight.grad, atol=1e-6, rtol=1e-6)


def test_interrupted_stage_resumes_only_once_without_replacing_committed_work(tmp_path, monkeypatch):
    p = small_protocol()
    p["pipeline_order"] = ["P-DISC"]
    # P0's matrix count is a separate full-matrix fixture, already tested above.
    monkeypatch.setattr(pilot, "preflight", lambda *args: {"status": "PASS"})
    pilot.run_pilot(p, arguments(tmp_path, "P0"))
    original = pilot.train_model

    def interrupted(*args, **kwargs):
        return original(*args, **dict(kwargs, pause_update=5))

    monkeypatch.setattr(pilot, "train_model", interrupted)
    with pytest.raises(pilot.RecoveryPause):
        pilot.run_pilot(p, arguments(tmp_path, "P1"))
    before = pilot.read_json(tmp_path / "pilot/P1/state.json")["active_seconds"]
    with pytest.raises(pilot.PilotError, match="--resume"):
        pilot.run_pilot(p, arguments(tmp_path, "P1"))
    monkeypatch.setattr(pilot, "train_model", original)
    assert pilot.run_pilot(p, arguments(tmp_path, "P1", resume=True))["status"] == "PASS"
    after = pilot.read_json(tmp_path / "pilot/P1/state.json")
    assert after["resume_attempts"] == 1 and after["active_seconds"] > before
