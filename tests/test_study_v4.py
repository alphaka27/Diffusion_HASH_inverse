"""v4 contract and one compact production lifecycle check; never a scientific run."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from diffusion_hash_inv import study_v4 as v4, study_v4_data as data, study_v4_runtime as rt
from diffusion_hash_inv import study_v4_statistics as stats


class NoBudget:
    def check(self, *args, **kwargs):
        pass

    def reserve(self, *args, **kwargs):
        pass

    def account(self, **counts):
        pass


def test_cli_contract_statistics_and_gates(tmp_path, capsys):
    root = tmp_path / "uncreated"
    assert v4.main(["run", "--workdir", str(root), "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["main_rows"] == 14745600 and result["main_learned_nfe"] == 324403200
    assert result["updates_per_main_run"] == 15700 and not root.exists()
    altered = deepcopy(data.PROTOCOL)
    altered["MAIN"]["trials"] = 10
    path = tmp_path / "changed.json"
    rt.atomic_json(path, altered)
    assert v4.main(["plan", "--protocol", str(path)]) == 3
    assert v4.main(["run", "--workdir", str(root), "--stage", "V3"]) == 3
    assert not (root / "MAIN").exists()
    stats.self_check()
    assert stats.analyze_counts({})["scientific_decision"] == "INVALID_OR_INCOMPLETE"
    cases = [(800, 200, "GO"), (200, 200, "NO_GO_SMALL"), (500, 300, "INCONCLUSIVE")]
    for n10, n01, expected in cases:
        counts = {f"{s}/{c}": {"n10": n10, "n01": n01, "n11": 0, "n00": 16384-n10-n01}
                  for s in range(3) for c in ("random", "shuffled")}
        assert stats.analyze_counts(counts)["scientific_decision"] == expected
    counts["2/random"] = {"n10": 200, "n01": 200, "n11": 0, "n00": 15984}
    assert stats.analyze_counts(counts)["scientific_decision"] == "NO_GO_REPRODUCIBILITY"
    lower, upper = np.array([[.01]*6]), np.array([[.02]*6])
    assert stats.decisions(lower, upper)["INCONCLUSIVE"][0]
    with pytest.raises(ValueError):
        stats.analyze_counts({key: {**row, "n00": -1} for key, row in counts.items()})


def reviewed_inventory(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"review": "fixture only"}')
    return {"schema": 1, "reviewer": "fixture", "reviewed_scopes": dict.fromkeys(data.SCOPES, True), "unresolved": [],
            "entries": [{"id": "fixture", "scope": "local_runs", "purpose": "evaluation", "source": "printable",
                         "prefixes12": list(range(1885)), "notes": "test inventory, not a real exposure audit",
                         "evidence": [{"path": str(evidence), "sha256": data.file_hash(evidence)}]}]}


def test_exposure_ownership_data_boundary_and_seed_contract(tmp_path):
    p = data.PROTOCOL
    assert data.audit_exposure(data.exposure_template())["status"] == "BLOCKED_EXPOSURE"
    with pytest.raises(ValueError, match="inventory"):
        data.audit_exposure([])
    inventory = reviewed_inventory(tmp_path)
    assert data.audit_exposure(inventory)["status"] == "PASS"
    exposure = list(range(1885))
    groups = data.ownership(p, "MAIN", exposure)
    assert not set(groups["test"]) & set(exposure)
    assert {k: len(v) for k, v in groups.items()} == {"train": 1536, "validation": 512, "reserve": 1024, "test": 1024}
    assert len(set(sum(groups.values(), []))) == 4096
    with pytest.raises(ValueError, match="EXPOSURE"):
        data.ownership(p, "MAIN", range(3073))
    inventory["entries"][0]["prefixes12"].append(4096)
    assert data.audit_exposure(inventory)["status"] == "BLOCKED_EXPOSURE"
    q = deepcopy(p)
    q["E0"].update(train_messages=32, train_groups=32, validation_groups=2, test_groups=2)
    directory = tmp_path / "data"
    groups = data.make_data(q, "E0", directory, range(128))
    assert len(data.read_json(directory / "train.json")) == 32
    for name in ("train.json", "validation.json", "evaluator/test_representatives.json"):
        for target, raw in data.read_json(directory / name):
            assert data.h12(bytes.fromhex(raw)) == target
    data.verify_data(directory)
    assert data.make_data(q, "E0", directory, range(128)) == groups
    rngs = {data.seed(p, "MAIN", namespace, method=method, label=s, trial=t, attempt=a)
            for namespace in ("length", "payload") for method in data.METHODS
            for s in range(3) for t in range(3) for a in range(1, 4)}
    assert len(rngs) == 162
    assert data.seed(p, "MAIN", "train-corruption", label=0, epoch=1, trial=0) == 9254768291057528196


def test_full_ledger_fixture_and_random_resume(tmp_path):
    assert v4.ledger_fixture(data.PROTOCOL, tmp_path / "fixture", NoBudget())["status"] == "PASS"
    p = deepcopy(data.PROTOCOL)
    p["TEST"] = {"k": 100}
    targets = [17, 17]
    first, resumed = tmp_path / "first", tmp_path / "resumed"
    expected = rt.evaluate(p, "TEST", "random", 0, targets, first, None, 16, NoBudget())
    with pytest.raises(rt.Pause):
        rt.evaluate(p, "TEST", "random", 0, targets, resumed, None, 16, NoBudget(), pause_batch=1)
    partial = rt.verify_ledger(p, "TEST", "random", 0, targets, resumed, None, 16, complete=False)
    assert partial["outcomes"]["normal"] == [None, None] and partial["success_at_k"]["100"] is None
    actual = rt.evaluate(p, "TEST", "random", 0, targets, resumed, None, 16, NoBudget())
    assert expected["rows"] == actual["rows"] == 200
    assert expected["md5_calls"] == actual["md5_calls"] == 200
    with sqlite3.connect(first / "candidates.sqlite") as a, sqlite3.connect(resumed / "candidates.sqlite") as b:
        assert a.execute("SELECT * FROM candidates").fetchall() == b.execute("SELECT * FROM candidates").fetchall()
    assert actual["valid_rate"] == 1 and actual["nfe"] == 0
    with sqlite3.connect(resumed / "candidates.sqlite") as db:
        record = json.loads(db.execute("SELECT record FROM candidates WHERE rowid=1").fetchone()[0])
        record["success"] = not record["success"]
        db.execute("UPDATE candidates SET record=? WHERE rowid=1", (json.dumps(record),))
    with pytest.raises(ValueError, match="verifier"):
        rt.verify_ledger(p, "TEST", "random", 0, targets, resumed, None, 16)


def test_mlx_v4_trainer_uniform_loss_and_resume(tmp_path):
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    from diffusion_hash_inv import mlx_backend as backend
    backend.resolve_device("gpu")
    p = deepcopy(data.PROTOCOL)
    p["V1"].update(train_messages=128, validation_groups=4, train_groups=4088, test_groups=4, epochs=2, validation_every=1)
    data.make_data(p, "V1", tmp_path / "data")
    result = v4.recovery(p, "V1", tmp_path / "data", tmp_path / "recovery", NoBudget(), 4)
    assert result["status"] == "PASS"
    model, diffusion = rt.build_model(p, "V1", 0)
    assert diffusion.factorized and not diffusion.prefix_balanced_loss
    clean = backend.clean_batch(rt.TokenCodec("printable", 31), [[0, b"000!".hex()]], diffusion)
    cond = backend.models.condition([0])
    mask = mx.zeros((1, 32), dtype=mx.bool_)
    loss, parts = diffusion.losses(model, clean, cond, mx.array([.5]), mask, return_components=True)
    assert parts["payload_ce"].item() == 0
    np.testing.assert_allclose(np.array(loss), np.array(parts["length_ce"])[None])
    a = rt.train(p, "V1", "main", 0, tmp_path / "data", tmp_path / "main", NoBudget())
    b = rt.train(p, "V1", "shuffled", 0, tmp_path / "data", tmp_path / "shuffled", NoBudget())
    main, shuffled = (data.read_json(tmp_path / m / "training.json") for m in ("main", "shuffled"))
    assert [r["order_sha256"] for r in main["history"]] == [r["order_sha256"] for r in shuffled["history"]]
    assert main["updates"] == shuffled["updates"] == 4
    assert a[2] != b[2]


def test_e0_actual_md5_end_to_end_and_sealed_resume(tmp_path):
    pytest.importorskip("mlx.core", exc_type=ImportError)
    from diffusion_hash_inv.mlx_backend import resolve_device
    resolve_device("gpu")
    p = data.PROTOCOL
    directory = tmp_path / "E0"
    data.make_data(p, "E0", directory / "data", range(128))
    seal = v4.train_all(p, "E0", directory, NoBudget())
    results = v4.evaluate_all(p, "E0", directory, NoBudget(), 16)
    assert sum(row["rows"] for row in results.values()) == 9600
    assert sum(row["nfe"] for row in results.values()) == 211200
    assert sum(row["md5_calls"] for row in results.values()) == 9600
    assert v4.train_all(p, "E0", directory, NoBudget()) == seal
    # Already committed rows are verified without another candidate stream.
    assert v4.evaluate_all(p, "E0", directory, NoBudget(), 16)["99/main"]["outcomes"] == results["99/main"]["outcomes"]
    counts = rt.paired_counts(results, (99,))
    assert all(sum(row.values()) == 32 for row in counts.values())
    path = directory / "trials.json"
    altered = data.read_json(path)
    altered["targets"][0] ^= 1
    data.atomic_json(path, altered)
    with pytest.raises(ValueError, match="deterministic"):
        data.trial_schedule(p, "E0", directory, seal)


def test_budget_and_partial_report_do_not_award_science(tmp_path):
    p = deepcopy(data.PROTOCOL)
    p["resources"]["disk_free_gib"] = 0
    budget = rt.Budget(tmp_path, p, "V0")
    budget.state["stages"]["V0"] = 86400
    with pytest.raises(rt.ResourceStop, match="cumulative"):
        budget.check()
    budget.finish()
    assert data.read_json(tmp_path / "budget.json")["running_since"] is None
    assert v4.analyze(p, tmp_path)["scientific_decision"] == "NOT_EVALUATED"


def test_measured_resource_profile(tmp_path):
    pytest.importorskip("mlx.core", exc_type=ImportError)
    from diffusion_hash_inv.mlx_backend import resolve_device
    resolve_device("gpu")
    p = data.PROTOCOL
    data.make_data(p, "E0", tmp_path / "V2/E0/data", range(128))
    result = v4.profile(p, tmp_path, NoBudget())
    assert result["status"] in {"PASS", "BLOCKED_RESOURCE"}
    assert result["batch_size"] in (1, 4, 16, 64)
    assert len(result["training_update_windows"]) == 3
    assert result["storage_required_bytes"] > 14745600 * 100
    assert result["soft_seconds"] == result["estimate_seconds"] * 1.5
