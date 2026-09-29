"""V5 contract regression checks; optional Metal checks run in a subprocess."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dhi_v5 import data
from dhi_v5.checks import hash_gate, ledger_gate
from dhi_v5.protocol import (atomic_json, audit_exposure, environment, file_hash, freeze,
                             read_json, registration, sealed_json, source_manifest, verify_frozen)
from dhi_v5.runtime import Ledger, trial_schedule
from dhi_v5.statistics import (DELTA, Z_C, compare, cp_bound, decide, final_decision,
                               interval_from_moments, stage_c, compute_advantage)
from dhi_v5.study import report


def test_independent_package_and_fixed_plan():
    root = Path(__file__).parents[1]/"src/dhi_v5"
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("diffusion_hash_inv")
            if isinstance(node, ast.Import):
                assert all(not n.name.startswith(("diffusion_hash_inv", "torch")) for n in node.names)
    protocol = registration()
    assert protocol["backend"] == "mlx" and protocol["trials_c"] == 65536
    assert protocol["k"] == 100 and protocol["updates"] == 40000


def test_hash_codec_and_streams():
    assert hash_gate(128)["passed"]
    groups = data.split("test", "W2", 64, [0, 1, 2])
    assert [len(groups[k]) for k in ("train", "validation", "test")] == [2816, 256, 1024]
    assert not set(groups["test"]) & {0, 1, 2}
    a = data.fresh_batch(("test",), 7, 64, groups["train"], window="W2")
    b = data.fresh_batch(("test",), 7, 64, groups["train"], window="W2")
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert np.isin(a[2], groups["train"]).all()
    assert np.array_equal(data.hash_batch(a[0], a[1], window="W2"), a[2])
    synthetic = data.synthetic_split()
    assert set(synthetic["acceptance"]) == {x ^ 4095 for x in synthetic["acceptance"]}
    assert not set(synthetic["test"]) & set(synthetic["train"] + synthetic["validation"])
    donors = data.derangement(("test",), 100)
    assert np.all(donors != np.arange(100)) and len(set(donors)) == 100
    assert data.key_words(("x",), [0])[0].tobytes() != data.key_words(("x",), [100])[0].tobytes()
    assert data.decode(np.full((1, 32), data.MASK)) == [None]
    with pytest.raises(ValueError): data.condition_bits(np.array([4096]))
    with pytest.raises(ValueError): data.derangement(("x",), 1)


def test_statistics_branches_and_cp():
    assert cp_bound(8, 30, .025) == pytest.approx(.1227948098723548)
    assert cp_bound(8, 30, .025, upper=True) == pytest.approx(.458893651394751)
    assert cp_bound(0, 30) == 0 and cp_bound(30, 30, upper=True) == 1
    assert cp_bound(0, 30, upper=True) == pytest.approx(1-.05**(1/30))
    def bands(r, s):
        return {"Random": {"lower": r[0], "upper": r[1]}, "Shuffled": {"lower": s[0], "upper": s[1]}}
    assert decide(bands((.0001, .001), (.0001, .001))) == "POSITIVE"
    assert decide(bands((-.001, .001), (-.001, .001))) == "REJECTED_BOUNDED"
    assert decide(bands((-.001, .004), (-.001, .001))) == "REJECTED_NO_CONDITION_GAIN"
    assert decide(bands((-.001, .001), (-.001, .004))) == "REJECTED_NO_RANDOM_ADVANTAGE"
    assert decide(bands((-.001, .004), (-.001, .004))) == "EXTEND"
    assert decide(bands((-.001, .004), (-.001, .004)), 2) == "NOT_ESTABLISHED_UNRESOLVED"
    assert decide(bands((.001, .004), (.001, .004)), replication=True) == "SUPPORTED"
    assert decide(bands((-.001, .004), (.001, .004)), replication=True) == "NOT_ESTABLISHED_NOT_REPLICATED"
    assert final_decision("FAIL", "REJECTED_BOUNDED") == "FINAL_NOT_ESTABLISHED"
    m = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]])
    c = np.zeros_like(m)
    d = m.mean(axis=0)
    result = compare(m, c)
    assert result["se"] == pytest.approx(d.std(ddof=1)/2)
    assert compare(c, m)["estimate"] == -result["estimate"]
    assert stage_c(m, c, c)["comparisons"]["Random"]["seeds"][0]["n10"] == 2
    with pytest.raises(ValueError): stage_c(m[:2], c[:2], c[:2])
    with pytest.raises(ValueError): compare(np.array([[float('nan'), 0]]), np.zeros((1, 2)))


def test_ledger_and_trials(tmp_path):
    # This fixture stays on the NumPy/hashlib path; it does not import MLX.
    ns = ("test-ledger",)
    meta = {"task": "synthetic", "rung": 64, "window": "W1", "method": "Random", "rng_namespace": list(ns)}
    ledger = Ledger(tmp_path/"l.sqlite", meta, [0, 0], 3)
    payloads = [b"000!", b"000!", None, b"FFFF", b"000!", b"~~~~"]
    ledger.append(0, payloads[:2], data.key_words(ns, [0, 1]))
    ledger.close()
    ledger = Ledger(tmp_path/"l.sqlite", meta, [0, 0], 3)
    with pytest.raises(ValueError): ledger.verify()
    ledger.append(2, payloads[2:], data.key_words(ns, np.arange(2, 6)))
    hits, metrics = ledger.verify()
    assert hits.tolist() == [1, 1] and metrics["rows"] == 6 and metrics["hits"] == 3
    with ledger.db:
        ledger.db.execute("UPDATE candidates SET hit=0 WHERE trial=0 AND attempt=0")
    with pytest.raises(ValueError, match="rehash"): ledger.verify()
    ledger.close()
    with pytest.raises(ValueError): trial_schedule(tmp_path/"trials.json", ns, [0, 1], [], 10)
    meta = {**meta, "task": "md5", "window": "W2"}
    real = Ledger(tmp_path/"md5.sqlite", meta, [data.hash_one(b"abcd", window="W2")], 2)
    real.append(0, [b"abcd", None], data.key_words(ns, [0, 1]))
    outcome, metrics = real.verify()
    assert outcome.tolist() == [1] and metrics["md5_calls"] == metrics["hash_family_calls"] == 2
    real.close()


def test_frozen_protocol_and_audit_refuse_changes(tmp_path, monkeypatch):
    from dhi_v5 import protocol
    monkeypatch.setattr(protocol, "environment", lambda: {"backend": "fixture"})
    incomplete = tmp_path/"inventory.json"
    atomic_json(incomplete, {"schema": "v5-exposure-1", "scope_complete": False})
    assert not audit_exposure(incomplete)["certified"]
    for name in ("A-impl", "A-prof", "A-dev", "fallback", "exposure-audit"):
        atomic_json(tmp_path/f"{name}.json", {"passed": True})
    # Environment metadata is available without importing the Metal runtime.
    freeze(tmp_path, {}, {"lr": .001}, {"certified": True, "primary": "W2"}, {})
    assert verify_frozen(tmp_path)["window"]["replication"] == "W3"
    frozen = read_json(tmp_path/"protocol.frozen.json")
    frozen["registration"]["updates"] = 1
    atomic_json(tmp_path/"protocol.frozen.json", frozen)
    with pytest.raises(ValueError): verify_frozen(tmp_path)


def test_partial_report_is_not_a_rejection(tmp_path):
    result = report(tmp_path)
    assert result["status"] == "INCOMPLETE" and result["overall"] == "NOT_FINAL"
    atomic_json(tmp_path/"A.json", {"C1": "FAIL"})
    result = report(tmp_path)
    assert result["C3"] == "NOT_ESTABLISHED_UNTESTABLE"
    assert result["overall"] == "FINAL_NOT_ESTABLISHED"


def test_mlx_end_to_end(tmp_path):
    if importlib.util.find_spec("mlx") is None:
        pytest.skip("MLX is optional outside Apple Silicon")
    available = subprocess.run([sys.executable, "-c", "import mlx.core"], capture_output=True)
    if available.returncode:
        pytest.skip("Metal unavailable in this process sandbox")
    code = "from pathlib import Path; from dhi_v5.checks import sampler_gate, resume_gate, ledger_gate; import sys; p=Path(sys.argv[1]); assert sampler_gate(12)['passed']; assert resume_gate(p)['passed']; assert ledger_gate(p)['passed']; print('MLX integration PASS')"
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_stage_order_and_final_report(tmp_path, monkeypatch):
    from dhi_v5 import study
    events = []
    class NoBudget:
        def __init__(self, root, stage): pass
        def check(self): pass
    monkeypatch.setattr(study, "Budget", NoBudget)
    monkeypatch.setattr(study, "release", lambda: None)
    frozen = {"window": {"primary": "W2", "replication": "W3"}, "fallback": {"settings": {
        "c_mc_trials": 16384, "b_trials": 16384, "ladder": list(data.LADDER), "upper_replication": True, "s_updates": 160000}}}
    qualification = {"C1": "PASS", "architecture": "D1-T", "scale_architecture": "D1-T-L", "clp_disabled": False}
    atomic_json(tmp_path/"A.json", qualification)
    atomic_json(tmp_path/"exposure-audit.json", {"excluded": {w: [] for w in data.WINDOWS}})
    def learned(root, stage, arch, method, seed_id, *args, **kwargs):
        folder = root/stage/f"test-{method}-{seed_id}"
        sealed_json(folder/"complete.json", {"method": method, "seed": seed_id})
        events.append(("train", method, seed_id))
        return folder, None
    def streams(root, stage, arch, seed_id, groups, targets, *args, **kwargs):
        assert sum(e[0] == "train" for e in events) == 6
        events.append(("eval", seed_id))
        outcome = {m: np.zeros(len(targets), dtype=np.int8) for m in ("Main", "Shuffled", "Random")}
        metric = {"independent_verified": True, "successful_training_matches": 0, "top_1pct_target_hit_share": 0,
                  "hits": 0, "rows": len(targets)*100, "success_at_100": 0}
        return outcome, {m: dict(metric) for m in outcome}, None
    monkeypatch.setattr(study, "learned_run", learned)
    monkeypatch.setattr(study, "run_streams", streams)
    with pytest.raises(FileNotFoundError): study.stage_b(tmp_path, frozen, qualification)
    c = study.stage_confirmatory(tmp_path, "C", frozen, qualification)
    assert c["decision"] == "REJECTED_BOUNDED" and len(c["looks"]) == 1
    assert len(read_json(tmp_path/"C/trials.json")["checkpoints"]) == 6
    with pytest.raises(ValueError, match="audited"): study.stage_confirmatory(tmp_path, "R", frozen, qualification)
    def ladder(root, stage, rung, seed_id, *args, **kwargs):
        return {"GEN": rung <= 16 and seed_id != 2, "INFO": rung <= 12 if stage == "B" else True, "clp": None}
    monkeypatch.setattr(study, "ladder_run", ladder)
    b = study.stage_b(tmp_path, frozen, qualification)
    assert b["r_gen_confirmed"] == 16 and b["r_edge"] == 16 and b["pivot"] == "PIVOT_SUPPORTED"
    s = study.stage_s(tmp_path, frozen, qualification)
    assert s["SCALE_SHIFT"] and s["CLP_64_anomaly"]
    final = report(tmp_path)
    assert final["overall"] == "FINAL_REJECTED"
    assert final["C3"] == "REJECTED_BOUNDED"  # B/S positives cannot change C3.


def test_qualification_remediation_once(tmp_path, monkeypatch):
    from dhi_v5 import study
    calls = []
    def one(root, arch, seed_id, updates, *args):
        calls.append((arch, seed_id, updates))
        return {"generation_pass": updates == 160000 and arch == "D1-S", "clp": {"positive": False}}
    monkeypatch.setattr(study, "qualify_one", one)
    monkeypatch.setattr(study, "release", lambda: None)
    frozen = {"lr": .0003, "profile": {"models": {a: {"batch": 1024} for a in ("D1-S", "D1-T", "D1-T-L")}},
              "fallback": {"settings": {"prefer_d1s": False}}}
    q = study.qualify(tmp_path, frozen, None)
    assert q["Q"] == ["D1-S"] and q["architecture"] == "D1-S"
    assert q["remediation"] and q["clp_disabled"] and q["scale_architecture"] == "D1-T"
    assert len(calls) == 13 and sum(c[2] == 160000 for c in calls) == 6


def test_exposure_projection_and_c4(tmp_path):
    code, archive = tmp_path/"code", tmp_path/"archive"
    code.mkdir(); archive.mkdir()
    (code/"conditioning.py").write_text("condition_use = 'prefix'\n")
    atomic_json(archive/"run.json", {"condition_use": "full"})
    atomic_json(archive/"representatives.json", [b"abcd".hex()])
    records = []
    for file, use in ((code/"conditioning.py", "prefix"), (archive/"run.json", "full"), (archive/"representatives.json", "none")):
        record = {"path": str(file.relative_to(tmp_path)), "sha256": file_hash(file), "condition_use": use}
        if use == "full": record["representatives"] = "archive/representatives.json"
        records.append(record)
    inventory = {"schema": "v5-exposure-1", "scope_complete": True, "w1_exposure_complete": False,
                 "scopes": {"code": ["code"], "archive": ["archive"]}, "reviewed_files": records}
    atomic_json(tmp_path/"inventory.json", inventory)
    result = audit_exposure(tmp_path/"inventory.json")
    assert result["certified"] and result["primary"] == "W2"
    for w in data.WINDOWS:
        assert data.hash_one(b"abcd", window=w) in result["excluded"][w]
    (code/"new.py").write_text("new = True\n")
    assert not audit_exposure(tmp_path/"inventory.json")["certified"]
    yes = compute_advantage("SUPPORTED", 100, 100, 1, 100, 10000, 1000)
    no = compute_advantage("REJECTED_BOUNDED", 100, 100, 1, 100, 10000, 1000)
    assert yes["decision"] == "PER_QUERY_ADVANTAGE" and not yes["amortized_advantage"]
    assert no["decision"] == "NO_ADVANTAGE"


def test_extension_reuses_trials_and_replication_changes_window(tmp_path, monkeypatch):
    from dhi_v5 import study
    class NoBudget:
        def __init__(self, *args): pass
        def check(self): pass
    monkeypatch.setattr(study, "Budget", NoBudget)
    monkeypatch.setattr(study, "release", lambda: None)
    frozen = {"window": {"primary": "W2", "replication": "W3"}, "fallback": {"settings": {"c_mc_trials": 16384}}}
    q = {"C1": "PASS", "architecture": "D1-S"}
    atomic_json(tmp_path/"exposure-audit.json", {"excluded": {w: [] for w in data.WINDOWS}})
    seen_targets, trained, looks = {}, [], []
    def learned(root, stage, arch, method, seed_id, *args, **kwargs):
        trained.append((stage, method, seed_id, kwargs["window"]))
        folder = root/stage/f"{method}-{seed_id}"
        sealed_json(folder/"complete.json", {"window": kwargs["window"]})
        return folder, None
    def streams(root, stage, arch, seed_id, groups, targets, *args, **kwargs):
        if stage in seen_targets:
            assert np.array_equal(seen_targets[stage], targets)
        else:
            seen_targets[stage] = targets.copy()
        outcomes = {m: np.zeros(len(targets), dtype=np.int8) for m in ("Main", "Shuffled", "Random")}
        metrics = {m: {"independent_verified": True, "successful_training_matches": 0, "top_1pct_target_hit_share": 0} for m in outcomes}
        return outcomes, metrics, None
    def classify(main, random, shuffled, look, replication):
        looks.append((look, replication, main.shape[0]))
        return {"decision": "SUPPORTED" if replication else ("EXTEND" if look == 1 else "POSITIVE"), "look": look, "comparisons": {}}
    monkeypatch.setattr(study, "learned_run", learned)
    monkeypatch.setattr(study, "run_streams", streams)
    monkeypatch.setattr(study, "stage_c", classify)
    c = study.stage_confirmatory(tmp_path, "C", frozen, q)
    assert c["decision"] == "POSITIVE" and len(c["looks"]) == 2
    assert len(read_json(tmp_path/"C/extension-checkpoints.json")) == 6
    r = study.stage_confirmatory(tmp_path, "R", frozen, q)
    assert r["decision"] == "SUPPORTED" and r["window"] == "W3"
    assert looks == [(1, False, 3), (2, False, 6), (1, True, 3)]
    assert {row[2] for row in trained if row[0] == "R"} == {0, 1, 2}
    assert not np.array_equal(seen_targets["C"], seen_targets["R"])


def test_one_command_run_preflight_and_order(tmp_path, monkeypatch):
    from dhi_v5 import study
    inventory = tmp_path/"inventory.json"
    atomic_json(inventory, {"schema": "v5-exposure-1", "scope_complete": False})
    def never(*args, **kwargs):
        raise AssertionError("Expensive stage started before exposure certification")
    monkeypatch.setattr(study, "stage_a", never)
    with pytest.raises(ValueError, match="incomplete"):
        study.run_all(tmp_path, inventory)
    assert not (tmp_path/"A-impl.json").exists()

    events = []
    monkeypatch.setattr(study, "audit_exposure", lambda path: {"certified": True, "primary": "W2", "inventory_sha256": file_hash(path)})
    def stage_a(root, phase):
        events.append("A")
        atomic_json(root/"A.json", {"C1": "PASS", "architecture": "D1-S"})
    def confirm(root, stage, frozen, qualification):
        events.append(stage)
        result = {"decision": "POSITIVE", "artifact_audit": {"passed": True}} if stage == "C" else {"decision": "SUPPORTED"}
        sealed_json(root/f"{stage}.json", result)
        return result
    def ladder(root, *args):
        events.append("B")
        sealed_json(root/"B.json", {"partial": False})
    def scale(root, *args):
        events.append("S")
        sealed_json(root/"S.json", {"partial": False})
    monkeypatch.setattr(study, "stage_a", stage_a)
    monkeypatch.setattr(study, "verify_frozen", lambda root: {"window": {"primary": "W2", "replication": "W3"}})
    monkeypatch.setattr(study, "stage_confirmatory", confirm)
    monkeypatch.setattr(study, "stage_b", ladder)
    monkeypatch.setattr(study, "stage_s", scale)
    monkeypatch.setattr(study, "report", lambda root: {"status": "TERMINAL", "overall": "FINAL_SUPPORTED"})
    assert study.run_all(tmp_path, inventory)["overall"] == "FINAL_SUPPORTED"
    assert events == ["A", "C", "R", "B", "S"]
    assert study.run_all(tmp_path, inventory)["overall"] == "FINAL_SUPPORTED"
    assert events == ["A", "C", "R", "B", "S"]
