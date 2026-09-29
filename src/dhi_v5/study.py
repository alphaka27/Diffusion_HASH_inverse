"""V5 stage runner: A -> C -> optional extension/R -> B -> S -> report."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from .data import (decode, encode, fresh_batch, hash_one, key_words, messages,
                   prior_candidates, rng, source, split, synthetic_split, valid)
from .protocol import (Budget, BudgetExceeded, atomic_json, audit_exposure, environment,
                       fallback_for, file_hash, freeze, read_json, registration, sealed_json,
                       source_manifest, study_lock, verify_frozen)
from .runtime import Ledger, evaluate, likelihood_probe, load_checkpoint, train, trial_schedule
from .statistics import compute_advantage, final_decision, ladder_cell, stage_c


def release():
    import mlx.core as mx
    gc.collect()
    mx.clear_cache()


def profile(root, budget):
    import mlx.core as mx
    import mlx.optimizers as optim
    from .models import candidate_keys, make_model, parameter_count, sample, train_step
    groups = synthetic_split()
    payload, lengths, labels, _ = fresh_batch(("profile",), 0, 256, groups["train"], task="synthetic")
    tokens = encode(payload, lengths)
    models = {}
    for arch in ("D1-S", "D1-T", "D1-T-L"):
        budget.check()
        model = make_model(arch, ("profile", arch))
        optimizer = optim.Adam(.001, bias_correction=True)
        keys = candidate_keys(("profile-train",), np.arange(256))
        timings = []
        for update in range(12):
            started = time.monotonic()
            train_step(model, optimizer, tokens, lengths, labels, keys, .001, update)
            if update >= 2:
                timings.append(time.monotonic()-started)
        samples = {}
        for size in (256, 1024, 4096):
            budget.check()
            y = np.resize(labels, size)
            k = candidate_keys(("profile-sample",), np.arange(size))
            sample(model, y, k)
            mx.reset_peak_memory()
            elapsed = []
            for _ in range(3):
                started = time.monotonic()
                sample(model, y, k)
                elapsed.append(time.monotonic()-started)
            samples[str(size)] = {"candidates_per_second": size/float(np.median(elapsed)), "peak_memory": mx.get_peak_memory()}
        best = max(samples, key=lambda b: samples[b]["candidates_per_second"])
        size = int(best)
        from .data import condition_bits
        x = mx.array(np.resize(tokens, (size, 32)))
        c = mx.array(condition_bits(np.resize(labels, size)))
        n = mx.array(np.resize(lengths, size))
        mx.eval(model(x, mx.full((size,), .5), c, n))
        started = time.monotonic()
        for _ in range(10):
            mx.eval(model(x, mx.full((size,), .5), c, n))
        models[arch] = {"parameters": parameter_count(model), "update_seconds": float(np.median(timings)),
                        "batch": size, "batches": samples, "candidates_per_second": samples[best]["candidates_per_second"],
                        "forward_rows_per_second": 10*size/(time.monotonic()-started)}
        del model, optimizer
        release()
    count = 65536
    started = time.monotonic()
    p, n = source(rng("profile-prior-md5"), count)
    for message in messages(p, n):
        hashlib.md5(message).digest()
    prior_md5 = count/(time.monotonic()-started)
    started = time.monotonic()
    for update in range(100):
        fresh_batch(("profile-fixture",), update, 256, groups["train"], window="W2")
    fresh_seconds = (time.monotonic()-started)/100
    ns = ("profile-ledger",)
    meta = {"task": "synthetic", "window": "W1", "rung": 64, "method": "Random", "rng_namespace": list(ns)}
    ledger = Ledger(root/"profile-ledger.sqlite", meta, np.zeros(100, dtype=np.int32))
    started = time.monotonic()
    p, n = prior_candidates(ns, np.arange(10000))
    if not ledger.count():
        ledger.append(0, messages(p, n), key_words(ns, np.arange(10000)))
    ledger.verify(budget=budget)
    elapsed = time.monotonic()-started
    ledger.close()
    return {"models": models, "prior_md5_per_second": prior_md5, "fresh_batch_seconds": fresh_seconds,
            "ledger_verified_rows_per_second": 10000/elapsed, "bytes_per_ledger_row": (root/"profile-ledger.sqlite").stat().st_size/10000,
            "a_estimated_hours": (7*40000*models["D1-T"]["update_seconds"]+6*10000*models["D1-T"]["update_seconds"])/3600,
            "training_storage_gib": 40*40000*256*60/1024**3, "environment": environment(),
            "measurement_scope": "synthetic-and-disposable-hash-fixtures-no-study-condition-data"}


def develop(root, budget):
    groups = synthetic_split()
    scores = []
    for lr in registration()["dev_lrs"]:
        rows = []
        for seed_id in registration()["dev_seeds"]:
            folder = root/"A-dev"/f"lr-{lr}"/str(seed_id)
            model = train(folder, "D1-T", ("A-dev", seed_id), groups, updates=10000, lr=lr, task="synthetic", budget=budget)
            db = sqlite3.connect(folder/"training.sqlite")
            rows.append(float(db.execute("SELECT validation_loss FROM diagnostics ORDER BY update_id DESC LIMIT 1").fetchone()[0]))
            db.close()
            del model
            release()
        scores.append({"lr": lr, "seed_losses": rows, "mean_loss": float(np.mean(rows))})
    selected = min(scores, key=lambda row: row["mean_loss"])
    return {"lr": selected["lr"], "scores": scores, "selection": registration()["dev_selection"], "acceptance_used": False}


def qualify_one(root, arch, seed_id, updates, lr, batch, budget):
    from .models import candidate_keys, sample
    groups = synthetic_split()
    namespace = ("A-Q", arch, seed_id)
    folder = root/"A-Q"/str(updates)/arch/str(seed_id)
    model = train(folder, arch, namespace, groups, updates=updates, lr=lr, task="synthetic", budget=budget)
    result_path = folder/"qualification.json"
    if result_path.exists():
        return read_json(result_path)
    labels = np.asarray(groups["acceptance"], dtype=np.int32)
    candidates = []
    for flipped in (False, True):
        target = labels ^ 4095 if flipped else labels
        rows = []
        for offset in range(0, len(labels), batch):
            budget.check()
            rows.extend(decode(sample(model, target[offset:offset+batch], candidate_keys((*namespace, "acceptance"), np.arange(offset, min(offset+batch, len(labels)))))))
        candidates.append(rows)
    normal, flipped = candidates
    normal_joint = sum(valid(x) and hash_one(x, task="synthetic") == y for x, y in zip(normal, labels))
    flipped_joint = sum(valid(x) and hash_one(x, task="synthetic") == (y ^ 4095) for x, y in zip(flipped, labels))
    wrong = sum(valid(x) and hash_one(x, task="synthetic") == y for x, y in zip(flipped, labels))
    normal_valid, flipped_valid = sum(map(valid, normal)), sum(map(valid, flipped))
    clp = likelihood_probe(model, (*namespace, "qualification"), groups["acceptance"], pairs=registration()["clp_a"], task="synthetic", budget=budget)
    clp["positive"] = clp["estimate"] > 3.26*clp["se"]
    clp["threshold"] = 3.26
    generation_pass = normal_joint >= 461 and flipped_joint >= 461 and normal_valid == flipped_valid == 512 and wrong <= 25
    result = {"normal_joint": int(normal_joint), "flipped_joint": int(flipped_joint), "wrong_original": int(wrong),
              "normal_valid": normal_valid, "flipped_valid": flipped_valid, "generation_pass": bool(generation_pass), "clp": clp,
              "normal_payloads": [x.hex() if x else None for x in normal], "flipped_payloads": [x.hex() if x else None for x in flipped],
              "md5_calls": 0, "checkpoint": read_json(folder/"checkpoint.json")}
    sealed_json(result_path, result)
    return result


def qualify(root, frozen, budget):
    rounds = []
    for updates in (40000, 160000):
        rows = {}
        for arch in ("D1-S", "D1-T"):
            lr = .001 if arch == "D1-S" else frozen["lr"]
            rows[arch] = [qualify_one(root, arch, s, updates, lr, frozen["profile"]["models"][arch]["batch"], budget) for s in (0, 1, 2)]
            release()
        qualified = [arch for arch, values in rows.items() if all(x["generation_pass"] for x in values)]
        rounds.append({"updates": updates, "rows": rows, "Q": qualified})
        if qualified:
            break
    large = qualify_one(root, "D1-T-L", 0, 40000, frozen["lr"]*192/256, frozen["profile"]["models"]["D1-T-L"]["batch"], budget)
    clp_disabled = any(x["generation_pass"] and not x["clp"]["positive"] for r in rounds for row in r["rows"].values() for x in row)
    clp_disabled |= large["generation_pass"] and not large["clp"]["positive"]
    arch = ("D1-T" if "D1-T" in qualified else "D1-S") if qualified else None
    if frozen["fallback"]["settings"]["prefer_d1s"] and "D1-S" in qualified:
        arch = "D1-S"
    return {"C1": "PASS" if qualified else "FAIL", "Q": qualified, "architecture": arch, "rounds": rounds,
            "remediation": len(rounds) == 2, "large": large, "scale_architecture": "D1-T-L" if large["generation_pass"] else "D1-T",
            "clp_disabled": bool(clp_disabled), "clp_disabled_reason": "GENERATION_PASS_CLP_FAILURE" if clp_disabled else None,
            "d1s_fallback_requested": frozen["fallback"]["settings"]["prefer_d1s"],
            "d1s_fallback_applied": frozen["fallback"]["settings"]["prefer_d1s"] and "D1-S" in qualified}


def stage_a(root, phase):
    from .checks import implementation_gate
    budget = Budget(root, "A")
    phases = ("impl", "prof", "dev", "freeze", "qualify") if phase == "all" else (phase,)
    for current in phases:
        if current == "impl":
            if not (root/"A-impl.json").exists():
                result = implementation_gate(root, budget=budget)
                if not result["passed"]:
                    atomic_json(root/"A-impl-failure.json", result)
                    raise ValueError("Implementation/calibration gate failed")
                sealed_json(root/"A-impl.json", {**result, "source": source_manifest()})
        else:
            gate = read_json(root/"A-impl.json")
            if not gate["passed"] or gate["source"] != source_manifest():
                raise ValueError("Current implementation must pass A-impl first")
            if current == "prof" and not (root/"A-prof.json").exists():
                result = profile(root, budget)
                sealed_json(root/"A-prof.json", result)
                sealed_json(root/"fallback.json", fallback_for(result))
            elif current == "dev" and not (root/"A-dev.json").exists():
                read_json(root/"fallback.json")
                sealed_json(root/"A-dev.json", develop(root, budget))
            elif current == "freeze":
                freeze(root, read_json(root/"A-prof.json"), read_json(root/"A-dev.json"), read_json(root/"exposure-audit.json"), read_json(root/"fallback.json"))
            elif current == "qualify" and not (root/"A.json").exists():
                sealed_json(root/"A.json", qualify(root, verify_frozen(root), budget))
        budget.check()


def get_groups(root, stage, window, rung):
    audit = read_json(root/"exposure-audit.json")
    excluded = audit["excluded"][window] if rung == 64 else []
    groups = split(stage, window, rung, excluded)
    sealed_json(root/stage/f"groups-{window}-r{rung}.json", groups)
    return groups


def learned_run(root, stage, arch, method, seed_id, groups, frozen, budget, *, window, rung=64, updates=40000):
    namespace = (stage, window, rung, arch, seed_id)
    folder = root/stage/f"{window}-r{rung}"/arch/f"{method}-{seed_id}"
    lr = .001 if arch == "D1-S" else frozen["lr"]*(192/256 if arch == "D1-T-L" else 1)
    model = train(folder, arch, namespace, groups, updates=updates, lr=lr, window=window, rung=rung, shuffled=method == "Shuffled", budget=budget)
    return folder, model


def run_streams(root, stage, arch, seed_id, groups, targets, frozen, budget, qualification, *, window, rung=64,
                methods=("Main", "Shuffled", "Random", "MC"), clp_pairs=65536, mc_trials=None):
    from .models import make_model
    outcomes, metrics, clp = {}, {}, None
    for method in methods:
        checkpoint_method = "Main" if method == "MC" else method
        folder = root/stage/f"{window}-r{rung}"/arch/f"{checkpoint_method}-{seed_id}"
        model, checkpoint, training_db = None, None, None
        if method != "Random":
            model = make_model(arch, ("load-only",))
            if not (folder/"complete.json").exists():
                raise ValueError("Unsealed model")
            checkpoint = read_json(folder/"checkpoint.json")
            load_checkpoint(folder, model)
            training_db = folder/"training.sqlite"
        stream_targets = targets[:mc_trials] if method == "MC" and mc_trials else targets
        evaluation_folder = root/stage/f"{window}-r{rung}"/arch/f"eval-{method}-{seed_id}"
        values, result = evaluate(evaluation_folder, model, (stage, window, rung, method, seed_id), stream_targets,
                                 method=method, seed_id=seed_id, window=window, rung=rung, training_db=training_db,
                                 checkpoint=checkpoint, batch_size=frozen["profile"]["models"][arch]["batch"], budget=budget)
        outcomes[method], metrics[method] = values, {key: value for key, value in result.items() if key != "outcomes"}
        if method == "Main" and not qualification["clp_disabled"]:
            path = evaluation_folder/"clp.json"
            if path.exists():
                clp = read_json(path)
            else:
                clp = likelihood_probe(model, (stage, window, rung, seed_id, "test-probe"), groups["test"], pairs=clp_pairs, window=window, rung=rung, budget=budget)
                sealed_json(path, clp)
        del model
        release()
    return outcomes, metrics, clp


def stage_confirmatory(root, stage, frozen, qualification):
    if qualification["C1"] != "PASS":
        raise ValueError("Stage A qualification did not pass")
    if stage == "R":
        c = read_json(root/"C.json")
        if c["decision"] != "POSITIVE" or not c["artifact_audit"]["passed"]:
            raise ValueError("Replication requires audited Stage C POSITIVE")
    arch = qualification["architecture"]
    window = frozen["window"]["primary" if stage == "C" else "replication"]
    groups = get_groups(root, stage, window, 64)
    all_outcomes = {name: [] for name in ("Main", "Random", "Shuffled")}
    all_metrics, all_clp, look_results = {}, {}, []
    for look, seeds in ((1, (0, 1, 2)), (2, (3, 4, 5))):
        budget = Budget(root, "C_extension" if look == 2 else stage)
        checkpoint_folders = []
        for seed_id in seeds:
            for method in ("Main", "Shuffled"):
                folder, model = learned_run(root, stage, arch, method, seed_id, groups, frozen, budget, window=window)
                checkpoint_folders.append(folder)
                del model
                release()
        path = root/stage/"trials.json"
        if look == 1:
            targets = trial_schedule(path, (stage, window, 64), groups["test"], checkpoint_folders, 65536)
        else:
            sealed_json(root/stage/"extension-checkpoints.json", {str(p): file_hash(p/"complete.json") for p in checkpoint_folders})
            targets = np.asarray(read_json(path)["targets"], dtype=np.int32)
        for seed_id in seeds:
            outcomes, metrics, clp = run_streams(root, stage, arch, seed_id, groups, targets, frozen, budget, qualification,
                                               window=window, methods=("Main", "Shuffled", "Random", "MC") if stage == "C" else ("Main", "Shuffled", "Random"),
                                               mc_trials=frozen["fallback"]["settings"]["c_mc_trials"])
            for name in all_outcomes:
                all_outcomes[name].append(outcomes[name])
            all_metrics[str(seed_id)], all_clp[str(seed_id)] = metrics, clp
        result = stage_c(*(np.stack(all_outcomes[name]) for name in ("Main", "Random", "Shuffled")), look=look, replication=stage == "R")
        look_results.append(result)
        sealed_json(root/stage/f"look-{look}.json", result)
        budget.check()
        if result["decision"] != "EXTEND" or stage == "R":
            break
    audit_passed = all(m["independent_verified"] and m["successful_training_matches"] == 0 for rows in all_metrics.values() for m in rows.values())
    main_rate = np.mean(np.stack(all_outcomes["Main"]))
    mc_signal = bool(any("MC" in rows and rows["MC"]["success_at_100"]/(rows["MC"]["rows"]/100) < main_rate for rows in all_metrics.values()))
    clp_signal = bool(any(c and c["estimate"] > 3.26*c["se"] for c in all_clp.values()))
    final = {**result, "looks": look_results, "window": window, "architecture": arch, "metrics": all_metrics, "clp": all_clp,
             "artifact_audit": {"passed": audit_passed, "hit_only": not mc_signal and not clp_signal,
                                "top_1pct_target_hit_share": {s: m["Main"]["top_1pct_target_hit_share"] for s, m in all_metrics.items()}}}
    sealed_json(root/f"{stage}.json", final)
    return final


def ladder_run(root, stage, rung, seed_id, arch, frozen, qualification, budget, *, updates=40000, trials=16384, clp_pairs=32768):
    window = frozen["window"]["primary"] if rung == 64 else "W1"
    groups = get_groups(root, stage, window, rung)
    folder, model = learned_run(root, stage, arch, "Main", seed_id, groups, frozen, budget, window=window, rung=rung, updates=updates)
    if rung == 64 and stage == "S":
        clp = None if qualification["clp_disabled"] else likelihood_probe(model, (stage, window, rung, seed_id), groups["test"], pairs=clp_pairs, window=window, rung=rung, budget=budget)
        if clp:
            clp["positive"] = clp["estimate"] > 3.26*clp["se"]
            clp["threshold"] = 3.26
        return {"INFO": clp["positive"] if clp else None, "clp": clp, "GEN": None}
    del model
    release()
    targets = trial_schedule(root/stage/f"trials-r{rung}-s{seed_id}.json", (stage, window, rung, seed_id), groups["test"], [folder], trials)
    outcomes, metrics, clp = run_streams(root, stage, arch, seed_id, groups, targets, frozen, budget, qualification,
                                       window=window, rung=rung, methods=("Main", "MC", "Random"), clp_pairs=clp_pairs)
    return {**ladder_cell(outcomes["Main"], outcomes["Random"], outcomes["MC"], clp), "metrics": metrics}


def require_c_terminal(root):
    c = read_json(root/"C.json")
    if c["decision"] == "POSITIVE":
        read_json(root/"R.json")
    elif c["decision"] == "EXTEND":
        raise ValueError("C extension must precede B/S")


def stage_b(root, frozen, qualification):
    require_c_terminal(root)
    budget = Budget(root, "B")
    settings = frozen["fallback"]["settings"]
    arch = qualification["architecture"]
    cells = {}
    for rung in settings["ladder"]:
        path = root/"B"/f"cell-r{rung}-s0.json"
        if not path.exists():
            sealed_json(path, ladder_run(root, "B", rung, 0, arch, frozen, qualification, budget, trials=settings["b_trials"]))
        cells[str(rung)] = {"0": read_json(path)}
    gen = [int(r) for r, rows in cells.items() if rows["0"]["GEN"]]
    info = [int(r) for r, rows in cells.items() if rows["0"]["INFO"]]
    peak = max(gen, default=None)
    replicate = [peak] if peak is not None else []
    if peak is not None and settings["upper_replication"]:
        above = [r for r in settings["ladder"] if r > peak]
        if above:
            replicate.append(min(above))
    for rung in replicate:
        for seed_id in (1, 2):
            path = root/"B"/f"cell-r{rung}-s{seed_id}.json"
            if not path.exists():
                sealed_json(path, ladder_run(root, "B", rung, seed_id, arch, frozen, qualification, budget, trials=settings["b_trials"]))
            cells[str(rung)][str(seed_id)] = read_json(path)
    confirmed = max((int(r) for r, rows in cells.items() if len(rows) == 3 and sum(x["GEN"] for x in rows.values()) >= 2), default=None)
    edge = next((r for r in settings["ladder"] if cells[str(r)]["0"]["INFO"] is False), None)
    result = {"cells": cells, "r_gen_seed0": peak, "r_gen_confirmed": confirmed, "r_info": max(info, default=None), "r_edge": edge,
              "pivot": "PIVOT_SUPPORTED" if confirmed is not None and confirmed >= 16 else "PIVOT_NOT_SUPPORTED",
              "r4_positive_control": cells["4"]["0"]["GEN"], "partial": False, "clp_disabled": qualification["clp_disabled"],
              "omitted_rungs": [r for r in registration()["ladder"] if r not in settings["ladder"]]}
    sealed_json(root/"B.json", result)
    budget.check()
    return result


def stage_s(root, frozen, qualification):
    require_c_terminal(root)
    b = read_json(root/"B.json")
    budget = Budget(root, "S")
    arch = qualification["scale_architecture"]
    rungs = [r for r in (b["r_edge"], 64) if r is not None]
    cells = {}
    for rung in rungs:
        path = root/"S"/f"cell-r{rung}.json"
        if not path.exists():
            sealed_json(path, ladder_run(root, "S", rung, 0, arch, frozen, qualification, budget,
                        updates=frozen["fallback"]["settings"]["s_updates"], trials=4096, clp_pairs=65536))
        cells[str(rung)] = read_json(path)
    shift = cells[str(b["r_edge"])]["INFO"] if b["r_edge"] is not None else None
    result = {"architecture": arch, "cells": cells, "SCALE_SHIFT": shift, "rung_shift_lower_bound": 1 if shift else 0,
              "CLP_64_anomaly": cells["64"]["INFO"], "partial": False}
    sealed_json(root/"S.json", result)
    budget.check()
    return result


def report(root):
    artifacts = {name: read_json(root/f"{name}.json") for name in ("A", "B", "C", "R", "S", "failure") if (root/f"{name}.json").exists()}
    c1 = artifacts.get("A", {}).get("C1", "NOT_MEASURED")
    c = artifacts.get("C", {})
    if not c:
        for look in (2, 1):
            path = root/"C"/f"look-{look}.json"
            if path.exists():
                c = read_json(path)
                break
    c3 = c.get("decision", "NOT_ESTABLISHED_INCOMPLETE")
    if c1 == "FAIL": c3 = "NOT_ESTABLISHED_UNTESTABLE"
    elif c3 == "POSITIVE": c3 = artifacts.get("R", {}).get("decision", "NOT_ESTABLISHED_REPLICATION_PENDING")
    failure = artifacts.get("failure", {})
    if failure.get("stage") in ("C", "R", "A") and failure.get("reason"):
        c3 = failure["reason"]
    c4 = None
    if c3 == "SUPPORTED":
        frozen = verify_frozen(root)
        metrics = c["metrics"]
        main_hits = sum(v["Main"]["hits"] for v in metrics.values())
        main_count = sum(v["Main"]["rows"] for v in metrics.values())
        random_hits = sum(v["Random"]["hits"] for v in metrics.values())
        random_count = sum(v["Random"]["rows"] for v in metrics.values())
        c4 = compute_advantage(c3, main_hits, main_count, random_hits, random_count, frozen["profile"]["prior_md5_per_second"], frozen["profile"]["models"][c["architecture"]]["candidates_per_second"])
    else:
        c4 = compute_advantage(c3, 0, 0, 0, 0, 0, 0)
    completed = c1 == "FAIL" or bool(failure) or (bool(c) and c3 not in ("NOT_ESTABLISHED_REPLICATION_PENDING", "EXTEND") and "B" in artifacts and "S" in artifacts)
    partial_cells = {stage: {p.stem: read_json(p) for p in (root/stage).glob("cell-*.json")} for stage in ("B", "S")}
    decision = {"status": "TERMINAL" if completed else "INCOMPLETE", "C1": c1, "C2": {"B": artifacts.get("B", partial_cells["B"] or None), "S": artifacts.get("S", partial_cells["S"] or None), "partial": "B" not in artifacts or "S" not in artifacts},
                "C3": c3, "C4": c4, "overall": final_decision(c1, c3) if completed else "NOT_FINAL",
                "comparisons": c.get("comparisons"), "failure": failure or None,
                "budget": read_json(root/"budget.json") if (root/"budget.json").exists() else None}
    atomic_json(root/"decision.json", decision)
    lines = ["# V5 최종 연구 보고서", "", f"실행 상태: {decision['status']}", f"종합 판정: {decision['overall']}", "",
             "| 질문 | 판정 |", "|---|---|", f"| C1 기계 | {c1} |", f"| C2 구조 이용 | {'부분 측정' if decision['C2']['partial'] else 'B·S 완료'} |",
             f"| C3 연구 가설 | {c3} |", f"| C4 계산 우위 | {c4['decision']} |", "",
             "실험이 미완료인 경우 효과가 없다는 결론으로 해석하지 않는다. 미측정 구간은 null로 기록한다.", "",
             "## 기존 증거와 D0", "", "PoC는 형식 실패, v3.1은 적격성 미완, v4는 checkpoint 선택 규칙으로 차단되었다. "
             "V4 D0의 최종 checkpoint 정상/반전 joint는 469/460, 469/469, 462/466이었다. "
             "기존 처리량은 D1-S vmap 17,565 후보/s, Transformer proxy 약 1,418 후보/s였다. 이는 V5 실측과 구분한다.", ""]
    for title, value in (("C1 적격성·보완·Q", artifacts.get("A")), ("C2 사다리·규모·pivot", decision["C2"]),
                         ("C3 모든 대조·seed별 구간·감사·재현", {"C": c or None, "R": artifacts.get("R")}), ("C4 계산 비용", c4)):
        lines += [f"## {title}", "", "```json", json.dumps(value, ensure_ascii=False, indent=2), "```", ""]
    profile_path = root/"A-prof.json"
    if profile_path.exists():
        lines += ["## 같은 기계의 실측 처리량", "", "```json", json.dumps(read_json(profile_path), ensure_ascii=False, indent=2), "```", ""]
    lines += ["## 적용 범위", "", "실측은 Printable source, 등록된 MD5 12-bit window, D1 계열, 균일 masked-diffusion loss, K=100에 한정된다. "
              "학습 1 run의 MD5 호출은 약 14,894,545회이며, 4,096개 전체 target lookup 기대 비용 약 36,434회의 409배다. 학습비 포함 계산 우위는 성립하지 않는다.", "",
              "Random-function 논증을 Gaussian BGV/CGGE, Random Bytes, q=8/16에 적용하는 것은 이론적 기대이며 측정 결과가 아니다. "
              "Full MD5 역상, 임의 target 역상, SHA-256, 보안 붕괴, 모든 diffusion의 불가능성을 주장하지 않는다. v3.1 다섯 pipeline의 적격성도 완료로 보고하지 않는다.", ""]
    path = root/"FINAL_REPORT_KO.md"
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)
    return decision


def run_all(root, inventory=None):
    """Resume the registered sequence, stopping when C1 fails or a cap is reached."""
    stage = "A"
    try:
        if (root/"protocol.frozen.json").exists():
            verify_frozen(root)
            if inventory is not None and file_hash(inventory) != read_json(root/"exposure-audit.json")["inventory_sha256"]:
                raise ValueError("Inventory differs from the frozen exposure audit")
        elif inventory is not None:
            audit = audit_exposure(inventory)
            if not audit.get("certified"):
                raise ValueError(f"Exposure inventory is incomplete: {audit.get('reason', 'uncertified windows')}")
            atomic_json(root/"exposure-audit.json", audit)
        elif not (root/"exposure-audit.json").exists() or not read_json(root/"exposure-audit.json").get("certified"):
            raise ValueError("A certified exposure inventory is required before the full run")

        if not (root/"A.json").exists():
            stage_a(root, "all" if not (root/"protocol.frozen.json").exists() else "qualify")
        qualification = read_json(root/"A.json")
        if qualification["C1"] == "FAIL":
            return report(root)
        frozen = verify_frozen(root)

        stage = "C"
        c = read_json(root/"C.json") if (root/"C.json").exists() else stage_confirmatory(root, "C", frozen, qualification)
        if c["decision"] == "POSITIVE" and not c["artifact_audit"]["passed"]:
            sealed_json(root/"failure.json", {"stage": "C", "reason": "NOT_ESTABLISHED_INTEGRITY"})
            return report(root)
        if c["decision"] == "POSITIVE":
            stage = "R"
            if not (root/"R.json").exists():
                stage_confirmatory(root, "R", frozen, qualification)
        stage = "B"
        if not (root/"B.json").exists():
            stage_b(root, frozen, qualification)
        stage = "S"
        if not (root/"S.json").exists():
            stage_s(root, frozen, qualification)
        return report(root)
    except (BudgetExceeded, ValueError, RuntimeError, FloatingPointError) as error:
        error.stage = stage
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--output", type=Path)
    audit = commands.add_parser("audit")
    audit.add_argument("--inventory", required=True, type=Path)
    audit.add_argument("--root", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--root", required=True, type=Path)
    run.add_argument("--stage", choices=("all", "A", "C", "R", "B", "S"), required=True)
    run.add_argument("--phase", choices=("all", "impl", "prof", "dev", "freeze", "qualify"), default="all")
    run.add_argument("--inventory", type=Path, help="Completed exposure inventory for --stage all")
    reporting = commands.add_parser("report")
    reporting.add_argument("--root", required=True, type=Path)
    check = commands.add_parser("check")
    check.add_argument("--root", required=True, type=Path)
    check.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = registration()
        if args.output:
            sealed_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    root = args.root.resolve()
    if root.exists() and not (root/"v5-study.json").exists() and any(root.iterdir()):
        parser.error("Choose an empty directory or an existing V5 study; prior artifacts are read-only")
    if args.command == "run" and args.stage != "A" and args.phase != "all":
        parser.error("--phase only applies to Stage A")
    if args.command == "run" and args.inventory and args.stage != "all":
        parser.error("--inventory only applies to --stage all; use the audit command for staged runs")
    with study_lock(root):
        sealed_json(root/"v5-study.json", {"protocol": registration()["protocol"]})
        if args.command == "audit":
            if (root/"protocol.frozen.json").exists():
                raise ValueError("Audit cannot change after protocol freeze")
            result = audit_exposure(args.inventory)
            atomic_json(root/"exposure-audit.json", result)
        elif args.command == "report":
            result = report(root)
        elif args.command == "check":
            from .checks import implementation_gate
            result = implementation_gate(root, args.quick)
            atomic_json(root/("quick-check.json" if args.quick else "implementation-check.json"), result)
        else:
            if (root/"failure.json").exists():
                raise RuntimeError("This study has a terminal failure; registered caps/rules cannot be reset")
            try:
                if args.stage == "all":
                    result = run_all(root, args.inventory)
                elif args.stage == "A":
                    stage_a(root, args.phase)
                    result = {"stage": "A", "phase": args.phase, "completed": True}
                else:
                    frozen = verify_frozen(root)
                    qualification = read_json(root/"A.json")
                    if (root/f"{args.stage}.json").exists():
                        result = read_json(root/f"{args.stage}.json")
                    elif args.stage in ("C", "R"):
                        result = stage_confirmatory(root, args.stage, frozen, qualification)
                    elif args.stage == "B":
                        result = stage_b(root, frozen, qualification)
                    else:
                        result = stage_s(root, frozen, qualification)
            except BudgetExceeded as error:
                result = {"stage": getattr(error, "stage", args.stage), "reason": "NOT_ESTABLISHED_BY_BUDGET", "error": str(error)}
                sealed_json(root/"failure.json", result)
                report(root)
                print(json.dumps(result))
                return 2
            except (ValueError, FloatingPointError, RuntimeError) as error:
                failed_stage = getattr(error, "stage", args.stage)
                atomic_json(root/"last-error.json", {"stage": failed_stage, "error": str(error), "scientific_result": "NOT_EVALUATED"})
                if "retry exhausted" in str(error).lower() or "retry exhausted" in str(error).lower().replace("exact ", ""):
                    sealed_json(root/"failure.json", {"stage": failed_stage, "reason": "NOT_ESTABLISHED_INTEGRITY", "error": str(error)})
                    report(root)
                raise
            if args.stage != "all":
                report(root)
        if args.command == "run" and args.stage == "all":
            result = {key: result[key] for key in ("status", "overall", "C1", "C3")}
            result["decision"] = str(root/"decision.json")
            result["report"] = str(root/"FINAL_REPORT_KO.md")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
