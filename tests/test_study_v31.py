"""Engineering checks only: shortened internal fixtures never confer formal gates."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from diffusion_hash_inv import study_cli, study_pilot as pilot
from diffusion_hash_inv.study_profiles import LengthMaskedDiffusion
from diffusion_hash_inv.study_v31 import profile_checks


SPEC = Path(__file__).parents[1] / "examples/poc-v3.1-protocol.json"


def small_protocol():
    p = deepcopy(pilot.load_protocol(SPEC))
    p["synthetic"]["train_unique_messages"] = 16
    p["training"].update(batch_size=2, validation_draws_per_condition=1)
    p["pilot"]["P1"].update(train_messages=8, validation_conditions=2, evaluation_trials=2)
    p["execution"]["minimum_disk_free_gib"] = 0
    for name, profile in p["model_profiles"].items():
        profile.update(sampling_steps=2, sampling_nfe_per_candidate=3 if name == "D1" else 2)
        if profile["family"] == "gaussian":
            profile["diffusion_steps"] = 10
    return p


def args(root, stage, resume=False, device="cpu"):
    return SimpleNamespace(workdir=root, stage=stage, device=device, development=True, threads=1, resume=resume)


def test_cli_plan_and_unsupported_execution_are_read_only(tmp_path, capsys):
    root = tmp_path / "not-created"
    common = ["--protocol", str(SPEC), "--workdir", str(root)]
    assert study_cli.main(["pilot", *common, "--stage", "P2", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert not result["executable"] and set(result["settings"]) == {"P2A", "P2B"}
    assert study_cli.main(["pilot", *common, "--stage", "P0"]) == 2
    assert study_cli.main(["pilot", *common, "--stage", "P2", "--development"]) == 2
    assert not root.exists()
    modified = json.loads(SPEC.read_text())
    modified["model_profiles"]["G2"]["prediction"] = "epsilon"
    bad = tmp_path / "modified.json"
    bad.write_text(json.dumps(modified))
    with pytest.raises(pilot.PilotError, match="Unsupported/modified"):
        pilot.load_protocol(bad)


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS requires native Metal access"))])
def test_all_thirteen_profiles_train_generate_and_recover(tmp_path, device):
    p = small_protocol()
    torch.set_num_threads(1)
    assert pilot.run_pilot(p, args(tmp_path, "P0", device=device))["status"] == "PASS"
    assert len(pilot.read_json(tmp_path / "pilot/P0/model_profiles.json")) == 13
    result = pilot.run_pilot(p, args(tmp_path, "P1", device=device))
    assert result["status"] == "PASS" and result["development_only"]
    summary = pilot.read_json(tmp_path / "pilot/P1/summary.json")
    assert len(summary["runs"]) == 28
    for name, run in summary["runs"].items():
        if name.startswith("random/"):
            assert run["candidates"] == 20 and run["sampling_nfe"] == 0
            continue
        profile_id = run["profile_id"]
        assert run["updates"] == 8
        assert run["metrics"]["candidates"] == 20
        assert run["metrics"]["sampling_nfe"] == 20 * (3 if profile_id == "D1" else 2)
        if profile_id == "D1":
            assert run["metrics"]["valid_rate"] == 1.
            ledger = pilot.logical_ledger(tmp_path / "pilot/P1/runs" / name)
            assert all(row["length_rng_identity"] != row["rng_identity"] for row in ledger)
            assert all(row["sampled_length"] == row["byte_length"] for row in ledger)
        if name.endswith("/main"):
            assert run["recovery"]["status"] == "PASS"
        timing = pilot.read_json(tmp_path / "pilot/P1/runs" / name / "training.json")
        assert len(timing["full_loop_update_seconds"]) == 8
    assert not pilot.read_json(tmp_path / "implementation_readiness.json")["implementation_ready"]
    assert not pilot.write_report(tmp_path, p)["main_ready"]


def test_x0_validation_target_and_d1_loss_boundary():
    p, device = small_protocol(), torch.device("cpu")
    encoder, _, shape = pilot.codecs(p["pipelines"]["P-G-BGV"])
    clean = encoder.encode(b"000!")[None] * 2 - 1

    class Oracle(torch.nn.Module):
        def forward(self, value, time, condition):
            return clean.expand_as(value)

    _, diffusion = pilot.model_and_diffusion(p, "P-G-BGV", device, 0, profile_id="G2")
    assert pilot.validation_loss(p, "P1", "P-G-BGV", Oracle(), diffusion,
                                 [[0, b"000!".hex()]], device) == 0.
    assert torch.equal(pilot.sample(p, "P-G-BGV", Oracle(), diffusion, [0], [8], device, profile_id="G2"), clean)
    model, diffusion = pilot.model_and_diffusion(p, "P-DISC", device, 0, profile_id="D1")
    assert isinstance(diffusion, LengthMaskedDiffusion)
    codec = pilot.codecs(p["pipelines"]["P-DISC"])[0]
    tokens = torch.stack([codec.encode(b"000!"), codec.encode(b"f" * 31)])
    cond = pilot.condition([0, 4095], device)
    with torch.no_grad():
        model.length_head.weight.zero_()
        model.length_head.bias.zero_()
    losses = diffusion.losses(model, tokens, cond, torch.zeros(2), torch.zeros_like(tokens, dtype=torch.bool))
    assert torch.allclose(losses, torch.full((2,), math.log(28)))
    losses.mean().backward()
    assert model.length_head.weight.grad.abs().sum() > 0
    corrupt = tokens.clone()
    corrupt[0, -1] = codec.eos
    with pytest.raises(ValueError, match="exactly one EOS"):
        diffusion.lengths(corrupt)
    with pytest.raises(ValueError, match="separate length RNG"):
        pilot.sample(p, "P-DISC", model, diffusion, [0], [8], device, profile_id="D1")
    assert pilot.seed(p, "P1", "initialization", profile_id="G0") != pilot.seed(p, "P1", "initialization", profile_id="G1")
    assert pilot.seed(p, "P1", "generation", profile_id="G0") == pilot.seed(p, "P1", "generation", profile_id="G1")


def test_full_gaussian_schedule_batch_check_keeps_single_reference_exact():
    p, device = pilot.load_protocol(SPEC), torch.device("cpu")
    torch.set_num_threads(1)
    model, diffusion = pilot.model_and_diffusion(p, "P-G-BGV", device, 0, profile_id="G0")
    result = profile_checks(p, "P-G-BGV", "G0", model, diffusion, device)
    assert result["passed"] and result["single_reference_exact"] and result["batch_decoder_equal"]
    assert result["batch_max_absolute_difference"] <= .0011


def test_budget_soft_warning_hard_cap_and_bounded_scans(tmp_path, monkeypatch):
    p = small_protocol()
    run = tmp_path / "pilot/P3/runs/P-DISC/D1/0/main"
    run.mkdir(parents=True)
    pilot.atomic_json(tmp_path / "resources.json", {"p3_soft_wall_seconds": .01,
                                                   "pipelines": {"P-DISC": {"p3_run_seconds": .01}}})
    calls = []
    original = Path.rglob

    def counted(path, pattern):
        calls.append(path)
        return original(path, pattern)

    monkeypatch.setattr(Path, "rglob", counted)
    state = {"active_seconds": 1., "run_active_seconds": {str(run): 1.}}
    budget = pilot.Budget(tmp_path, "P3", state, p, torch.device("cpu"))
    for _ in range(20):
        budget.check(run)
    assert calls.count(tmp_path) == 1 and calls.count(run) == 1
    assert state["warnings"]["RESOURCE_ESTIMATE_EXCEEDED"] is True
    p["execution"]["hard_run_storage_gib"] = 1024 / pilot.GIB
    with pytest.raises(pilot.PilotError, match="before write"):
        budget.reserve(run, 1025)
    state["active_seconds"] = p["execution"]["hard_stage_active_wall_seconds"]["P3"]
    with pytest.raises(pilot.PilotError, match="active wall-clock cap"):
        budget.check()


def test_partial_report_reads_completed_run_seals_without_stage_gate(tmp_path):
    p = small_protocol()
    pilot.atomic_json(tmp_path / "protocol.frozen.json", p)
    pilot.atomic_json(tmp_path / "manifest.json", {"identity": {"protocol_sha256": pilot.digest(p), "development": True}, "data_sha256": None})
    pilot.atomic_json(tmp_path / "gates.json", {})
    run = tmp_path / "pilot/P1/runs/P-DISC/D1/0/main"
    pilot.atomic_json(run / "metrics.json", {"normal_joint": 7, "flipped_joint": 0})
    pilot.atomic_json(run / "complete.json", {"summary": {"metrics": pilot.read_json(run / "metrics.json")},
                                              "sha256": pilot.seal_directory(run)})
    pilot.atomic_json(tmp_path / "pilot/P1/runs/P-DISC/D1/0/shuffled/configuration.json", {})
    result = pilot.write_report(tmp_path, p)
    assert result["run_progress"]["P1/P-DISC/D1/0/main"] == "COMPLETE"
    assert result["run_progress"]["P1/P-DISC/D1/0/shuffled"] == "INCOMPLETE"
    assert result["run_progress"]["P1/P-DISC/D0/0/main"] == "NOT_RUN"
    pilot.atomic_json(run / "metrics.json", {"normal_joint": 8})
    with pytest.raises(pilot.PilotError, match="modified"):
        pilot.write_report(tmp_path, p)


def test_resume_rejects_unknown_time_and_exhausted_hard_budget(tmp_path):
    p = small_protocol()
    pilot.run_pilot(p, args(tmp_path, "P0"))
    for status, code in (("RUNNING", 3), ("INCOMPLETE", 5)):
        pilot.atomic_json(tmp_path / "pilot/P1/state.json", {"status": status, "exit_code": code,
                                                           "active_seconds": 1., "history": []})
        with pytest.raises(pilot.PilotError, match="Unknown unaccounted time"):
            pilot.run_pilot(p, args(tmp_path, "P1", resume=True))
