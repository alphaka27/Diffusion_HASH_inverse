"""One gated CLI for the fixed v4 decision study. No reduced scientific run mode."""
import argparse
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import tempfile

import numpy as np

from .study_pilot import (atomic_json, canonical, digest, equal_tensors, file_hash, load_checkpoint,
                          lock, read_json, seal_directory, verify_seal)
from .study_v4_data import (METHODS, PROTOCOL, audit_exposure, h12, load_protocol, make_data,
                            seed, trial_schedule, verify_data)
from . import study_v4_runtime as rt
from .study_v4_statistics import analyze_counts, calibrate, self_check

STAGES = ("V0", "V1", "V2", "V3", "V4", "V5")


class Blocked(RuntimeError):
    def __init__(self, status, reason):
        self.status = status
        super().__init__(reason)


def plan(p):
    n, k = p["MAIN"]["trials"], p["MAIN"]["k"]
    return {"protocol_id": p["protocol_id"], "dry_run": True, "device_checked": False,
            "stages": {"V0": "integrity and actual MLX recovery checks", "V1": "three-seed synthetic qualification",
                       "V2": "exposure audit, E0, 8x20000 calibration, resource/batch seal",
                       "V3": "fresh ownership/data and six learned checkpoints", "V4": "common trials and nine complete streams",
                       "V5": "independent verifier, six simultaneous intervals and Korean report"},
            "main_rows": 9 * n * k, "main_learned_nfe": 6 * n * k * p["model"]["nfe"],
            "updates_per_main_run": math.ceil(p["MAIN"]["train_messages"] / p["training"]["batch_size"]) * p["MAIN"]["epochs"],
            "prerequisites": ["MLX Metal GPU", "reviewed exposure inventory before V2", "V1 qualification", "measured resource seal"],
            "scientific_decision": "NOT_RUN", "settings": p}


def initialize(root, p):
    root.mkdir(parents=True, exist_ok=True)
    current = rt.manifest(p)
    if (root / "protocol.frozen.json").exists():
        if read_json(root / "protocol.frozen.json") != p or read_json(root / "manifest.json") != current:
            raise ValueError("frozen protocol/code/environment changed; continuation is not permitted")
    else:
        if any(path.name not in {".lock", "exposure_inventory.json", "exposure_audit.json"} for path in root.iterdir()):
            raise ValueError("workdir contains unrecognized artifacts; use an empty v4 directory")
        atomic_json(root / "protocol.frozen.json", p)
        atomic_json(root / "manifest.json", current)


def audited(root):
    inventory_path, audit_path = root / "exposure_inventory.json", root / "exposure_audit.json"
    if not inventory_path.exists() or not audit_path.exists():
        raise Blocked("BLOCKED_EXPOSURE", "run audit with a complete reviewed exposure inventory before V2")
    audit = audit_exposure(read_json(inventory_path))
    if audit != read_json(audit_path) or audit["status"] != "PASS":
        raise Blocked("BLOCKED_EXPOSURE", "; ".join(audit["issues"]) or "exposure evidence changed")
    return audit["exposed_prefixes12"]


def audit(root, inventory):
    if (root / "V2").exists() or (root / "MAIN").exists():
        raise ValueError("exposure is frozen once V2 starts")
    result = audit_exposure(inventory)
    atomic_json(root / "exposure_inventory.json", inventory)
    atomic_json(root / "exposure_audit.json", result)
    return result


def recovery(p, task, data, directory, budget, batch):
    """Same production trainer and sampler; interrupted tensors and candidates must match."""
    from unittest.mock import patch
    from .mlx_backend import models
    from .study_pilot import logical_ledger
    directory.mkdir(parents=True, exist_ok=True)
    a, b = directory / "continuous", directory / "resumed"
    original_read = rt.read_json
    def boundary(path):
        if "evaluator" in Path(path).parts or Path(path).name in {"targets.json", "test_representatives.json"}:
            raise AssertionError("training accessed hidden test data")
        return original_read(path)
    with patch.object(rt, "read_json", side_effect=boundary):
        model, diffusion, checksum = rt.train(p, task, "main", 99, data, a, budget)
        try:
            rt.train(p, task, "main", 99, data, b, budget, pause_update=1)
        except rt.Pause:
            pass
        recovered, recovered_diff, recovered_checksum = rt.train(p, task, "main", 99, data, b, budget)
    left, _ = load_checkpoint(a)
    right, _ = load_checkpoint(b)
    if not all(equal_tensors(left[k], right[k]) for k in ("model", "optimizer", "update", "epoch", "best_epoch", "best_loss")):
        raise ValueError("training recovery differs from continuous MLX execution")
    # Repeated targets must still receive independent trial seeds.
    y = read_json(data / "validation.json")[0][0]
    targets = [y, y]
    rt.evaluate(p, task, "main", 99, targets, a, checksum, batch, budget, model, diffusion)
    try:
        rt.evaluate(p, task, "main", 99, targets, b, recovered_checksum, batch, budget, recovered, recovered_diff, pause_batch=0)
    except rt.Pause:
        pass
    rt.evaluate(p, task, "main", 99, targets, b, recovered_checksum, batch, budget, recovered, recovered_diff)
    def logical(path):
        return [{k: v for k, v in row.items() if k not in {"checkpoint_sha256", "config_sha256"}} for row in logical_ledger(path)]
    if logical(a) != logical(b):
        raise ValueError("generation recovery differs at frozen inference batch")
    try:
        models.condition([{"target": y, "length": 4}])
    except ValueError:
        pass
    else:
        raise ValueError("hidden representative length entered public condition")
    return {"status": "PASS", "training_tensors_exact": True, "generation_rows_exact": True,
            "training_test_boundary": True, "hidden_length_rejected": True, "batch": batch}


def ledger_fixture(p, directory, budget):
    """Full production ledger->trial->counts path, including duplicates and invalid attempts."""
    q = deepcopy(p)
    q["FIXTURE"] = {"k": 100}
    payload = b"!!!!"
    target = h12(payload)
    results = {}
    for label in range(3):
        for method in METHODS:
            folder = directory / str(label) / method
            folder.mkdir(parents=True, exist_ok=True)
            targets = [target, target]
            identity = rt.evaluation_identity(q, "FIXTURE", method, label, targets, None, 4)
            atomic_json(folder / "evaluation.json", identity)
            with closing(rt.ledger_open(folder / "candidates.sqlite")) as db:
                rows = []
                for trial in range(2):
                    for attempt in range(1, 101):
                        # Main only trial0, controls only trial1; every stream has a zero-success trial.
                        success = not (label == 2 and method == "random") and trial == (0 if method == "main" else 1) and attempt in (1, 2)
                        value = payload if success else None
                        output = ((value, value is not None, None if value else "mask_remaining"),
                                  seed(q, "FIXTURE", "payload", method=method, label=label, trial=trial, attempt=attempt),
                                  seed(q, "FIXTURE", "length", method=method, label=label, trial=trial, attempt=attempt))
                        row = rt.candidate(q, "FIXTURE", method, label, (trial, target, "normal", attempt), output, None, digest(identity))
                        rows.append((f'{q["protocol_id"]}/FIXTURE/{method}/{label}', str(trial), "normal", attempt, canonical(row).decode()))
                with db:
                    db.executemany("INSERT INTO candidates VALUES (?,?,?,?,?)", rows)
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            metrics = rt.verify_ledger(q, "FIXTURE", method, label, targets, folder, None, 4, budget=budget)
            zero = label == 2 and method == "random"
            assert metrics["rows"] == 200 and metrics.get("valid", 0) == (0 if zero else 2)
            assert metrics.get("duplicate_within_trial", 0) == (0 if zero else 1)
            assert metrics["md5_calls"] == (0 if zero else 2)
            assert metrics["outcomes"]["normal"] == ([False, False] if zero else [True, False] if method == "main" else [False, True])
            results[f"{label}/{method}"] = metrics
    counts = rt.paired_counts(results)
    assert all(row == {"n11": 0, "n10": 1, "n01": int(key != "2/random"), "n00": int(key == "2/random")} for key, row in counts.items())
    assert analyze_counts(counts, n=2)["scientific_decision"] == "INCONCLUSIVE"
    assert analyze_counts({})["scientific_decision"] == "INVALID_OR_INCOMPLETE"
    folder = directory / "0/main"
    with closing(sqlite3.connect(folder / "candidates.sqlite")) as db:
        with db:
            db.execute("DELETE FROM candidates WHERE rowid=100")
    try:
        rt.verify_ledger(q, "FIXTURE", "main", 0, [target, target], folder, None, 4)
    except ValueError:
        pass
    else:
        raise ValueError("missing ledger attempt accepted")
    return {"status": "PASS", "paired_counts": counts, "invalid_duplicate_post_success_attempts": True,
            "missing_rows_rejected": True, "repeated_target_trials": True}


def prepare_data(p, task, directory, budget, exposed=()):
    started = time.monotonic()
    try:
        return make_data(p, task, directory, exposed, check=budget.check)
    finally:
        budget.account(data_preparation_seconds_completed=time.monotonic() - started)


def v0(p, root, budget):
    self_check()
    vectors = {b"": "d41d8cd98f00b204e9800998ecf8427e", b"a": "0cc175b9c0f1b6a831c399e269772661",
               b"abc": "900150983cd24fb0d6963f7d28e17f72", b"message digest": "f96b697d7cb7938d525a2f31aaf161d0",
               b"abcdefghijklmnopqrstuvwxyz": "c3fcd3d76192e4007dfb496cca67e13b",
               b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789": "d174ab98d277d9f5a5611c2c9f419d9f",
               b"1234567890" * 8: "57edf4a22be3c955ac49da2e2107b67a"}
    for raw, expected in vectors.items():
        assert hashlib.md5(raw).hexdigest() == expected and h12(raw) == int(expected[:3], 16)
    assert h12(b"a") == 0x0cc
    codec = rt.TokenCodec("printable", 31)
    for length in range(4, 32):
        raw = bytes(33 + i % 94 for i in range(length))
        assert codec.decode(codec.encode(raw)).message == raw
    assert seed(p, "MAIN", "payload", method="main", trial=0) != seed(p, "MAIN", "payload", method="main", trial=1)
    q = deepcopy(p)
    q["V1"].update(train_messages=128, validation_groups=4, train_groups=4088, test_groups=4,
                   epochs=2, validation_every=1, k=1)
    prepare_data(q, "V1", root / "V0/data", budget)
    recovered = recovery(q, "V1", root / "V0/data", root / "V0/recovery", budget, 4)
    fixtures = ledger_fixture(p, root / "V0/ledger_fixture", budget)
    return {"status": "PASS", "hash_vectors": len(vectors), "leading_zero": True, "raw_roundtrips": 28,
            "recovery": recovered, "ledger_fixture": fixtures, "uniform_payload_loss": True}


def train_all(p, task, directory, budget):
    if (directory / "checkpoints.seal.json").exists():
        return verified_checkpoints(directory)
    checksums, failures = {}, {}
    for label in p[task]["seeds"]:
        for method in (("main",) if task == "V1" else ("main", "shuffled")):
            run = directory / "runs" / str(label) / method
            failure = run / "failure.json"
            if failure.exists():
                failures[f"{label}/{method}"] = read_json(failure)
                continue
            try:
                if (run / "training.json").exists():
                    _, _, checksum = rt.load_model(p, task, label, run)
                else:
                    _, _, checksum = rt.train(p, task, method, label, directory / "data", run, budget)
                checksums[f"{label}/{method}"] = {"checkpoint_sha256": checksum,
                                                  "files": seal_directory(run)}
            except (FloatingPointError, rt.RunStop) as error:
                # Numerical/local-time failures never acquire replacement seeds.
                failures[f"{label}/{method}"] = {"status": "FAILED", "reason": str(error)}
                atomic_json(failure, failures[f"{label}/{method}"])
                budget.check()
    if failures:
        atomic_json(directory / "training_failures.json", failures)
        raise ValueError("one or more registered learned runs failed; independent runs were preserved")
    seal = {"protocol_sha256": digest(p), "runs": checksums}
    path = directory / "checkpoints.seal.json"
    if path.exists() and read_json(path) != seal:
        raise ValueError("checkpoint seal changed")
    atomic_json(path, seal)
    return seal


def verified_checkpoints(directory):
    seal = read_json(directory / "checkpoints.seal.json")
    for key, entry in seal["runs"].items():
        verify_seal(directory / "runs" / key, entry["files"])
    return seal


def evaluate_all(p, task, directory, budget, batch):
    seal = verified_checkpoints(directory)
    targets = trial_schedule(p, task, directory, seal)
    results, failures = {}, {}
    for label in p[task]["seeds"]:
        for method in (("main",) if task == "V1" else METHODS):
            model = diffusion = checksum = None
            run = directory / "evaluations" / str(label) / method
            if (run / "failure.json").exists():
                failures[f"{label}/{method}"] = read_json(run / "failure.json")
                continue
            try:
                if method != "random":
                    model, diffusion, checksum = rt.load_model(p, task, label, directory / "runs" / str(label) / method)
                results[f"{label}/{method}"] = rt.evaluate(p, task, method, label, targets, run, checksum, batch, budget, model, diffusion)
            except (FloatingPointError, rt.RunStop) as error:
                failures[f"{label}/{method}"] = {"status": "FAILED", "reason": str(error)}
                atomic_json(run / "failure.json", failures[f"{label}/{method}"])
                budget.check()
    atomic_json(directory / "evaluation_summary.json", {"results": results, "failures": failures})
    if failures:
        raise ValueError("incomplete evaluation; independent streams were preserved")
    return results


def v1(p, root, budget):
    directory = root / "V1"
    prepare_data(p, "V1", directory / "data", budget)
    train_all(p, "V1", directory, budget)
    results = evaluate_all(p, "V1", directory, budget, 4)
    cfg, checks = p["V1"], {}
    for name, row in results.items():
        checks[name] = (all(sum(row["outcomes"][v]) >= cfg["joint_min"] and row[v + "_valid"] == cfg["test_groups"] for v in ("normal", "flipped"))
                        and row.get("wrong_original", 0) <= cfg["wrong_max"])
    atomic_json(directory / "qualification.json", {"checks": checks, "thresholds": cfg})
    if not all(checks.values()):
        raise Blocked("BLOCKED_QUALIFICATION", "uniform-payload D1 synthetic acceptance failed; no seed/loss/profile retry")
    return {"status": "PASS", "checks": checks, "results": results}


def profile(p, root, budget):
    directory = root / "V2/profile"
    directory.mkdir(parents=True, exist_ok=True)
    q = deepcopy(p)
    q["PROFILE"] = {**p["E0"], "epochs": 80, "validation_every": 10}
    # 256 messages / 64 = 4 updates/epoch: 20 warm-up + 3*100 measured updates.
    data = root / "V2/E0/data"
    model, diffusion, checksum = rt.train(q, "PROFILE", "main", 99, data, directory / "training", budget)
    timings = read_json(directory / "training/training.json")
    windows = [sum(timings["update_seconds"][20 + i * 100:120 + i * 100]) / 100 for i in range(3)]
    if len(timings["update_seconds"]) != 320:
        raise ValueError("profiling needs exactly 320 full-loop updates")
    pool = read_json(data / "targets.json")
    measurements = {}
    for batch in p["resources"]["batches"]:
        q["MEASURE"] = {"k": batch}
        values = {}
        for method in ("main", "random"):
            run = directory / f"batch-{batch}" / method
            before = time.monotonic()
            rt.evaluate(q, "MEASURE", method, 99, pool[:4], run, checksum if method == "main" else None,
                        batch, budget, model, diffusion)
            rows = [json.loads(line) for line in (run / "telemetry.jsonl").read_text().splitlines()]
            times = [r["seconds"] for r in rows if r["kind"] == "generation_batch"]
            if len(times) != 4:
                raise ValueError("profile must have one warmup and three measured batches")
            values[method + "_seconds_per_candidate"] = max(times[1:]) / batch
            values[method + "_full_seconds"] = time.monotonic() - before
        measurements[str(batch)] = values
    fastest = min(v["main_seconds_per_candidate"] for v in measurements.values())
    batch = min(int(k) for k, v in measurements.items() if v["main_seconds_per_candidate"] <= fastest / .99)
    selected = measurements[str(batch)]
    # Large populated WAL/index fixture, kept as explicit storage-only data.
    stress = directory / "storage.sqlite"
    with closing(rt.ledger_open(stress)) as db:
        template = rt.candidate(p, "E0", "main", 99, (16383, pool[0], "normal", 100),
                                ((b"~" * 31, True, None), 2**64-1, 2**64-2), checksum, "f" * 64)
        count = db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        if count > 100000 or count % 1000:
            raise ValueError("corrupt storage profiling fixture")
        for start in range(count, 100000, 1000):
            budget.check()
            with db:
                db.executemany("INSERT INTO candidates VALUES (?,?,?,?,?)",
                               [(p["protocol_id"] + "/MAIN/shuffled/2", str(i), "normal", 100, canonical(template).decode())
                                for i in range(start, start + 1000)])
        wal_size = sum(path.stat().st_size for path in directory.glob("storage.sqlite*"))
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        bytes_per_row = max(wal_size, stress.stat().st_size) / 100000
    row_count, learned, random_rows = 14745600, 9830400, 4915200
    val_seconds = max(row["validation_seconds"] for row in timings["history"] if "validation_seconds" in row) * 16
    checkpoint_seconds = max(timings["checkpoint_seconds"])
    train_seconds = 6 * (15700 * max(windows) + 10 * val_seconds + 257 * checkpoint_seconds)
    # Measure production verifier independently at the chosen batch, with all MD5 calls.
    measured_run = directory / f"batch-{batch}" / "main"
    q["MEASURE"] = {"k": batch}
    verification = rt.verify_ledger(q, "MEASURE", "main", 99, pool[:4], measured_run, checksum, batch, budget=budget)
    verify_per_row = verification["verification_seconds"] / verification["rows"]
    generation_seconds = learned * selected["main_seconds_per_candidate"] + random_rows * selected["random_seconds_per_candidate"]
    # V4 audits after generation and V5 independently audits again; allow one replay audit too.
    reporting_seconds = row_count * verify_per_row * 3
    estimate = train_seconds + generation_seconds + reporting_seconds
    prep_storage = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    storage = (row_count * bytes_per_row + prep_storage + 6 * 3 * 20 * 1024**2) * 2
    result = {"status": "PASS", "batch_size": batch, "measurements": measurements, "training_update_windows": windows,
              "estimate_seconds": estimate, "soft_seconds": estimate * 1.5, "storage_required_bytes": storage,
              "storage_bytes_per_row": bytes_per_row, "training_seconds": train_seconds,
              "generation_seconds": generation_seconds, "verification_report_seconds": reporting_seconds,
              "measurement_protocol_sha256": digest(q), "performance_selection": False}
    if result["soft_seconds"] > p["resources"]["main_seconds"] or storage > p["resources"]["storage_gib"] * rt.GIB or storage + p["resources"]["disk_free_gib"] * rt.GIB > shutil.disk_usage(root).free:
        result["status"] = "BLOCKED_RESOURCE"
    atomic_json(root / "resource.seal.json", result)
    return result


def v2(p, root, budget):
    exposed = audited(root)
    directory = root / "V2"
    calibration_path = directory / "calibration.json"
    if calibration_path.exists():
        calibration = read_json(calibration_path)
    else:
        budget.check()
        started = time.monotonic()
        calibration = calibrate()
        budget.account(calibration_seconds_completed=time.monotonic() - started)
        atomic_json(calibration_path, calibration)
        budget.check()
    if calibration["status"] != "PASS":
        raise ValueError("production calibration failed; no test access permitted")
    prepare_data(p, "E0", directory / "E0/data", budget, exposed)
    train_all(p, "E0", directory / "E0", budget)
    resources_path = root / "resource.seal.json"
    resources = read_json(resources_path) if resources_path.exists() else profile(p, root, budget)
    if resources["status"] != "PASS":
        raise Blocked("BLOCKED_RESOURCE", "measured forecast exceeds the registered time/storage budget")
    batch = resources["batch_size"]
    recovery_path = directory / "recovery.json"
    if not recovery_path.exists():
        result = recovery(p, "E0", directory / "E0/data", directory / "recovery", budget, batch)
        atomic_json(recovery_path, result)
    results = evaluate_all(p, "E0", directory / "E0", budget, batch)
    assert sum(r["rows"] for r in results.values()) == 9600
    assert sum(r.get("nfe", 0) for r in results.values()) == 211200
    # E0 gets the same paired counts and report primitives but cannot award a scientific decision.
    e0 = {"status": "PASS", "scientific_decision": "NOT_APPLICABLE_ENGINEERING_REHEARSAL",
          "paired_counts": rt.paired_counts(results, (99,)), "rows": 9600}
    atomic_json(directory / "E0/report.json", e0)
    audited(root)
    return {"status": "PASS", "calibration": calibration["status"], "E0": e0, "batch_size": batch,
            "resources": file_hash(resources_path), "exposure": file_hash(root / "exposure_audit.json")}


def v3(p, root, budget):
    exposed = audited(root)
    prepare_data(p, "MAIN", root / "MAIN/data", budget, exposed)
    seal = train_all(p, "MAIN", root / "MAIN", budget)
    return {"status": "PASS", "checkpoint_count": len(seal["runs"])}


def v4(p, root, budget):
    batch = read_json(root / "resource.seal.json")["batch_size"]
    results = evaluate_all(p, "MAIN", root / "MAIN", budget, batch)
    return {"status": "PASS", "streams": len(results), "rows": sum(row["rows"] for row in results.values())}


def partial_status(root):
    return {"stage_statuses": {stage: read_json(root / stage / "state.json") for stage in STAGES if (root / stage / "state.json").exists()},
            "completed_training": [str(path.parent.relative_to(root)) for path in root.rglob("training.json")],
            "failures": {str(path.parent.relative_to(root)): read_json(path) for path in root.rglob("failure.json")},
            "active_costs": read_json(root / "budget.json") if (root / "budget.json").exists() else {}}


def analyze(p, root, budget=None):
    directory, results, missing = root / "MAIN", {}, []
    if not (directory / "checkpoints.seal.json").exists() or not (directory / "trials.json").exists():
        return {"execution_status": "INVALID_OR_INCOMPLETE", "scientific_decision": "NOT_EVALUATED", "missing": ["checkpoints or trials"], **partial_status(root)}
    seal = verified_checkpoints(directory)
    targets = trial_schedule(p, "MAIN", directory, seal)
    train_hex = [row[1] for row in read_json(directory / "data/train.json")]
    batch = read_json(root / "resource.seal.json")["batch_size"]
    for label in p["MAIN"]["seeds"]:
        for method in METHODS:
            name = f"{label}/{method}"
            run = directory / "evaluations" / name
            checkpoint = seal["runs"][name]["checkpoint_sha256"] if method != "random" else None
            if not (run / "candidates.sqlite").exists():
                missing.append(name)
                continue
            result = rt.verify_ledger(p, "MAIN", method, label, targets, run, checkpoint, batch, complete=False,
                                      training_hex=train_hex, budget=budget)
            results[name] = result
            if result["status"] != "COMPLETE":
                missing.append(name)
    statistics = analyze_counts(rt.paired_counts(results), n=p["MAIN"]["trials"])
    all_stages = all((root / stage / "complete.json").exists() for stage in STAGES[:5])
    if missing or not all_stages:
        statistics["scientific_decision"] = "NOT_EVALUATED"
    costs = read_json(root / "budget.json") if (root / "budget.json").exists() else {}
    storage = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    per_stream_cost = {}
    for name, result in results.items():
        successes = sum(value is True for value in result["outcomes"]["normal"])
        telemetry = directory / "evaluations" / name / "telemetry.jsonl"
        events = [json.loads(line) for line in telemetry.read_text().splitlines()] if telemetry.exists() else []
        inference = sum(row["seconds"] for row in events if row["kind"] in {"generation_batch", "generation_uncommitted"})
        training = costs.get("runs", {}).get(str(directory / "runs" / name), 0)
        per_stream_cost[name] = {"inference_seconds": inference, "training_seconds": training,
                                 "inference_seconds_per_success": inference / successes if successes else None,
                                 "training_inclusive_seconds_per_success": (inference + training) / successes if successes else None}
    return {"execution_status": "COMPLETE" if not missing and all_stages else "INVALID_OR_INCOMPLETE", **statistics,
            "streams": results, "missing": missing, "active_costs": costs, "cost_per_stream": per_stream_cost,
            "cost_breakdown_seconds": {k.removesuffix("_seconds_completed"): v for k, v in costs.items() if k.endswith("_seconds_completed")},
            "storage_bytes": storage, "scope": "fixed Printable MD5-12 pool, six checkpoints and three registered seeds only"}


def write_report(root, result):
    result["next_action"] = {"GO": "별도 재현·두 번째 source·계산비용 연구를 설계한다.",
                             "NO_GO_SMALL": "현재 설정의 확대를 중단하고 효과 크기 상한과 음성 결과를 정리한다.",
                             "NO_GO_REPRODUCIBILITY": "세 seed 모두의 유용성 기준을 충족하지 못했으므로 현재 접근을 확대하지 않는다.",
                             "INCONCLUSIVE": "현 revision을 종료하며 자동 증액하지 않는다."}.get(result["scientific_decision"],
                             "차단·미완료 사유를 확인한다. MD5 효과에 대한 결론은 내리지 않는다.")
    atomic_json(root / "decision.json", result)
    lines = ["# v4 실행 보고서", "", f"실행 상태: `{result['execution_status']}`", "",
             f"과학적 판정: `{result['scientific_decision']}`", "",
             "고정 Printable MD5-12 pool·checkpoint·seed에 조건부인 결과입니다. Full MD5 역상이나 계산비용 우위를 뜻하지 않습니다.", ""]
    lines += ["다음 행동: " + result["next_action"], ""]
    if result.get("reason"):
        lines += ["사유: " + result["reason"], ""]
    if result.get("comparisons"):
        lines += ["| 비교 | Δ | 동시 CI 하한 | 상한 | n11/n10/n01/n00 |", "|---|---:|---:|---:|---|"]
        for name, row in result["comparisons"].items():
            lines.append(f"| {name} | {row['delta']:.6f} | {row['lower']:.6f} | {row['upper']:.6f} | {row['n11']}/{row['n10']}/{row['n01']}/{row['n00']} |")
    lines += ["", "누락: " + (", ".join(result.get("missing", [])) or "없음"), "",
              "유효한 모든 비교의 하한이 1%p를 초과할 때만 GO입니다. 등호는 경계에 포함하며 자동 증액하지 않습니다.",
              "무결성·양성 대조·노출·자원 실패는 MD5 효과가 없다는 결론으로 해석하지 않습니다.",
              "비용·길이·중복·학습 메시지 일치·@1/@10·NFE·재해시 비용은 decision.json에 기록됩니다.", ""]
    (root / "report.ko.md").write_text("\n".join(lines))
    return result


def stage_seal(root, stage):
    directories = {"V0": ["V0"], "V1": ["V1"], "V2": ["V2"], "V3": ["MAIN/data", "MAIN/runs"],
                   "V4": ["MAIN/evaluations"], "V5": []}[stage]
    seal = {}
    for name in directories:
        seal.update({str(Path(name) / k): v for k, v in seal_directory(root / name).items()})
    extra = {"V2": ["resource.seal.json", "exposure_inventory.json", "exposure_audit.json"],
             "V3": ["MAIN/checkpoints.seal.json"], "V4": ["MAIN/trials.json", "MAIN/evaluation_summary.json"]}.get(stage, [])
    seal.update({name: file_hash(root / name) for name in extra})
    return seal


def verify_completed(root):
    for stage in STAGES:
        path = root / stage / "complete.json"
        if path.exists():
            verify_seal(root, read_json(path)["seal"])


def execute(p, root, stage, resume=False):
    from .mlx_backend import resolve_device
    if stage != "V0":
        previous = STAGES[STAGES.index(stage) - 1]
        if not (root / previous / "complete.json").exists():
            raise Blocked("INVALID_OR_INCOMPLETE", f"{previous} must pass before {stage}")
    if stage == "V2":
        audited(root)  # Missing inventory does not consume the stage/resume allowance.
    resolve_device("gpu")
    initialize(root, p)
    verify_completed(root)
    directory = root / stage
    directory.mkdir(exist_ok=True)
    state_path = directory / "state.json"
    if (directory / "complete.json").exists():
        raise ValueError(f"{stage} already complete")
    state = read_json(state_path) if state_path.exists() else {"status": "NEW", "resumes": 0}
    if state["status"] != "NEW":
        if state["status"] not in {"INTERRUPTED", "RUNNING"} or not resume:
            raise ValueError("stage cannot retry; an interrupted stage requires --resume")
        old_budget = read_json(root / "budget.json") if (root / "budget.json").exists() else {}
        run = state.get("interrupted_run") or old_budget.get("running_run") or f"stage:{stage}"
        run = run.replace("/evaluations/", "/learned/").replace("/runs/", "/learned/")
        resume_path = root / "resumes.json"
        counts = read_json(resume_path) if resume_path.exists() else {}
        if counts.get(run, 0) >= p["resources"]["max_resumes_per_run"]:
            raise ValueError("this run has exhausted its one exact resume")
        counts[run] = counts.get(run, 0) + 1
        atomic_json(resume_path, counts)
        state["resumes"] += 1
    elif resume:
        raise ValueError("nothing to resume")
    state["status"] = "RUNNING"
    atomic_json(state_path, state)
    budget = rt.Budget(root, p, stage)
    try:
        budget.check()
        if stage == "V5":
            result = write_report(root, analyze(p, root, budget))
            if result["execution_status"] != "COMPLETE":
                raise ValueError("V5 cannot complete with missing streams")
            result = {"status": "PASS", "scientific_decision": result["scientific_decision"]}
        else:
            result = {"V0": v0, "V1": v1, "V2": v2, "V3": v3, "V4": v4}[stage](p, root, budget)
        budget.check()
        atomic_json(directory / "result.json", result)
        state["status"] = "PASS"
        atomic_json(directory / "complete.json", {"status": "PASS", "seal": stage_seal(root, stage)})
        return {"stage": stage, **result}
    except (KeyboardInterrupt, rt.Pause):
        state.update(status="INTERRUPTED", interrupted_run=budget.run)
        raise
    except Exception as error:
        state.update(status=getattr(error, "status", "BLOCKED_RESOURCE" if isinstance(error, rt.ResourceStop) else "INVALID_OR_INCOMPLETE"), reason=str(error))
        write_report(root, {"execution_status": state["status"], "scientific_decision": "NOT_EVALUATED", "reason": str(error), **partial_status(root)})
        raise
    finally:
        budget.finish()
        atomic_json(state_path, state)


def parser():
    result = argparse.ArgumentParser(description="고정 v4 Printable MD5-12 실험: V0–V5, 단일 CLI")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("plan", "audit", "run", "report"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--protocol", type=Path, help="optional; only the registered examples/v4-protocol.json is accepted")
        if name != "plan":
            cmd.add_argument("--workdir", type=Path, required=True)
        if name == "audit":
            cmd.add_argument("--inventory", type=Path, required=True)
        if name == "run":
            cmd.add_argument("--stage", choices=(*STAGES, "all"), default="all")
            cmd.add_argument("--resume", action="store_true")
            cmd.add_argument("--dry-run", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if not __debug__:
            raise ValueError("v4 integrity checks require Python without -O")
        p = load_protocol(args.protocol)
        if args.command == "plan" or getattr(args, "dry_run", False):
            result = plan(p)
        else:
            root = args.workdir.resolve()
            if args.command == "report" and not (root / "protocol.frozen.json").exists():
                raise ValueError("no initialized v4 study to report")
            with lock(root / ".lock"):
                if args.command == "audit":
                    result = audit(root, read_json(args.inventory))
                elif args.command == "report":
                    initialize(root, p)
                    verify_completed(root)
                    result = write_report(root, analyze(p, root))
                else:
                    stages = STAGES if args.stage == "all" else (args.stage,)
                    result = {}
                    for stage in stages:
                        if args.stage == "all" and (root / stage / "complete.json").exists():
                            continue
                        with lock(Path(tempfile.gettempdir()) / "dhi-v4-mlx-gpu.lock"):
                            result = execute(p, root, stage, args.resume)
                        args.resume = False
                    if not result:
                        initialize(root, p)
                        verify_completed(root)
                        result = read_json(root / "decision.json")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if result.get("status", "").startswith("BLOCKED") else 0
    except KeyboardInterrupt:
        print("중단 상태 보존 완료. 같은 명령에 --resume을 추가하여 한 번 재개할 수 있습니다.", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, RuntimeError, ImportError, AssertionError) as error:
        status = getattr(error, "status", "BLOCKED_RESOURCE" if isinstance(error, rt.ResourceStop) else "INVALID_OR_INCOMPLETE")
        print(json.dumps({"execution_status": status, "scientific_decision": "NOT_EVALUATED", "reason": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2 if status.startswith("BLOCKED") else 3


if __name__ == "__main__":
    raise SystemExit(main())
