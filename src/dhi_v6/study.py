"""V6 단계 실행기와 CLI: A -> C -> (R) -> P -> (S) -> 보고서. 틀은 dhi_v5/study.py를 따른다."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time

import numpy as np

from . import PROTOCOL, data, runtime
from . import statistics as st
from .protocol import (Budget, BudgetExceeded, approve_caps, atomic_json, audit_exposure, budget_plan,
                       effective_caps, environment, file_hash, freeze, inventory_draft, read_json,
                       registration, sealed_json, source_manifest, study_lock, verify_frozen)

REG = registration()
PIPELINES = tuple(REG["pipeline_order"])
GAUSSIAN = tuple(p for p in PIPELINES if REG["pipelines"][p]["model"] == "G3-U")
SEEDS = tuple(REG["stage_c"]["seeds"])
UPDATES = REG["training"]["updates"]
REPAIR_UPDATES = REG["training"]["remediation_updates"]
A_PHASES = ("impl", "prof1", "train", "dev", "evaluate", "repair", "prof2", "budget", "freeze")
MD5_STAGES = ("C", "R", "P", "S")
RANDOM_BATCH = 2048
PROFILE_BLOCK_SECONDS = 30
FORWARD_SECONDS = 5
# BudgetExceeded is a RuntimeError; every handler below re-raises it before this tuple.
INTEGRITY_ERRORS = (ValueError, RuntimeError, FloatingPointError)
BLOCK_RESULT = re.compile(r"block-\d+\.json")
BLINDING = {"procedure": "automatic looks; CLI shows continue/stop only"}
RELEASE = "headline after C (and R when an audited POSITIVE exists); P and S follow as supplementary stages"


class Halt(RuntimeError):
    """The budget plan needs a human cap decision before any MD5 condition data exist."""


def emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def release():
    import mlx.core as mx
    gc.collect()
    mx.clear_cache()


@contextmanager
def charged(root, stage):
    """One Budget per stage session; its time is persisted when the session ends."""
    budget = Budget(root, stage)
    try:
        yield budget
    finally:
        budget.flush()


def require(root, *names):
    missing = [name for name in names if not (Path(root) / name).exists()]
    if missing:
        raise ValueError(f"Missing prerequisite artifacts: {', '.join(missing)}")


def optional(root, name):
    path = Path(root) / name
    return read_json(path) if path.exists() else None


def source_of(pipeline):
    return REG["pipelines"][pipeline]["source"]


def steps_key(pipeline, steps):
    return str(steps) if pipeline in GAUSSIAN else "none"


def run_folder(root, stage, pipeline, method, seed_id, updates, architecture=None):
    label = pipeline if architecture is None else f"{pipeline}-{architecture}"
    return Path(root) / stage / "runs" / label / f"{method}-{seed_id}" / f"u{updates}"


def load_trained(folder, pipeline, stage, seed_id, architecture=None):
    from .models import make_model
    complete = runtime.verify_training(folder)
    model = make_model(pipeline, stage, seed_id, architecture)
    runtime.load_checkpoint(folder, model)
    return model, complete["checkpoint"]


def generate(model, labels, keys, steps, batch):
    from .models import sample
    messages, margins, strict = [], [], []
    for offset in range(0, len(labels), batch):
        m, g, s = sample(model, labels[offset:offset + batch], keys[offset:offset + batch], steps or 25)
        messages += list(m)
        margins.append(g)
        strict.append(s)
    return messages, np.concatenate(margins), np.concatenate(strict)


# ---------------------------------------------------------------- Stage A: 기계 적격성과 예산

def require_impl(root):
    require(root, "A-impl.json")
    gate = read_json(root / "A-impl.json")
    if not gate["passed"] or gate["quick"] or gate["source"] != source_manifest():
        raise ValueError("Current implementation must pass the formal A-impl gate first")
    return gate


def phase_impl(root, budget):
    from .checks import implementation_gate
    path = root / "A-impl.json"
    if path.exists():
        return read_json(path)
    try:
        result = implementation_gate(root, quick=False, budget=budget)
    except BudgetExceeded:
        raise
    except Exception as error:
        atomic_json(root / "A-impl-failure.json", {"passed": False, "error": repr(error), "source": source_manifest()})
        raise
    if not result["passed"]:
        atomic_json(root / "A-impl-failure.json", {**result, "source": source_manifest()})
        raise ValueError("Implementation gate failed")
    sealed_json(path, {**result, "source": source_manifest()})
    return read_json(path)


def choose_batch(batches):
    """Largest throughput; within 1% the smaller batch wins."""
    best = max(v["candidates_per_second"] for v in batches.values())
    floor = best * (1 - REG["profiling"]["tie_relative"])
    return min(int(b) for b, v in batches.items() if v["candidates_per_second"] >= floor)


def _timed_updates(pipeline, architecture, scratch, budget):
    import mlx.optimizers as optim
    from .models import candidate_keys, make_model, train_step
    spec = REG["pipelines"][pipeline]
    architecture = architecture or spec["model"]
    setting = REG["models"][architecture]
    model = make_model(pipeline, "A-prof", 0, architecture)
    optimizer = optim.Adam(learning_rate=setting["lr"], betas=[.9, .999], eps=1e-8, bias_correction=True)
    groups = data.synthetic_split()["train"]
    warmup, timed = REG["profiling"]["train_warmup_updates"], REG["profiling"]["train_timed_updates"]
    seconds = []
    for update in range(warmup + timed):
        budget.check()
        started = time.perf_counter()
        payload, lengths, labels, calls = data.fresh_batch(("A-prof", spec["source"], 0), update, REG["training"]["batch"],
                                                           groups, task="synthetic")
        clean = runtime.encoded_batch(payload, lengths, spec["representation"], spec["source"])
        keys = candidate_keys(("A-prof", pipeline, architecture, update), np.arange(len(labels)))
        train_step(model, optimizer, clean, lengths, labels, keys, setting["lr"], update)
        runtime.charge_work(scratch, updates=1, md5_calls=calls)
        if update >= warmup:
            seconds.append(time.perf_counter() - started)
    return model, float(np.median(seconds))


def _forward_rows(model, pipeline):
    """Forward rows per second at the CLP scoring batch (256 pairs = 512 rows)."""
    import mlx.core as mx
    spec = REG["pipelines"][pipeline]
    payload, lengths, labels, _ = data.fresh_batch(("A-prof", spec["source"], 1), 0, 512, data.synthetic_split()["train"],
                                                   task="synthetic")
    clean = runtime.encoded_batch(payload, lengths, spec["representation"], spec["source"])
    clean = mx.array(clean if spec["representation"] == "tokens" else clean.astype(np.float32))
    n, c, t = mx.array(lengths), mx.array(data.condition_bits(labels)), mx.full((512,), .5)
    if spec["representation"] == "tokens":
        def call():
            return model(clean, t, c, n)
    else:
        condition = mx.concatenate((c, n[:, None].astype(mx.float32) / 31), axis=1)

        def call():
            return model(clean, t, condition)
    mx.eval(call())
    started, repeats = time.perf_counter(), 0
    while repeats < 3 or time.perf_counter() - started < FORWARD_SECONDS:
        mx.eval(call())
        repeats += 1
    return 512 * repeats / (time.perf_counter() - started)


def _burst(model, pipeline, batch, steps, budget):
    import mlx.core as mx
    from .models import candidate_keys, sample
    labels = np.resize(np.array(data.synthetic_split()["validation"], dtype=np.int32), batch)
    keys = candidate_keys(("A-prof", pipeline, "burst", batch, steps), np.arange(batch))
    sample(model, labels, keys, steps)
    mx.reset_peak_memory()
    started, produced = time.perf_counter(), 0
    while produced == 0 or time.perf_counter() - started < REG["profiling"]["burst_seconds"]:
        budget.check()
        sample(model, labels, keys, steps)
        produced += batch
    return {"candidates_per_second": produced / (time.perf_counter() - started), "peak_memory": int(mx.get_peak_memory())}


def _generation_profile(model, pipeline, steps_options, budget):
    rows = {}
    for steps in steps_options:
        batches = {str(b): _burst(model, pipeline, b, steps or 25, budget) for b in REG["generation"]["batch_options"]}
        batch = choose_batch(batches)
        rows["none" if steps is None else str(steps)] = {"batches": batches, "batch": batch,
                                                         "candidates_per_second": batches[str(batch)]["candidates_per_second"]}
    return rows


def _md5_rate():
    """C4 baseline: source-prior sampling plus single-core hashlib MD5 over 65,536 messages."""
    started = time.perf_counter()
    payload, lengths = data.source(data.rng("A-prof", "md5"), 65536, "P")
    for message in data.messages(payload, lengths):
        hashlib.md5(message).digest()
    return 65536 / (time.perf_counter() - started)


def _ledger_rates(scratch, budget, trials=10240):
    """Verifier throughput on a 1,024,000-row Random fixture (random 12-bit targets, exposed W1)."""
    batch = RANDOM_BATCH
    targets = data.rng("A-prof", "verify").integers(0, 4096, trials).astype(np.int32)
    meta = {"protocol": PROTOCOL, "stage": "A-prof", "source": "P", "method": "Random", "task": "md5",
            "window": "W1", "rung": 64, "representation": None}
    namespace = ("A-prof", "P", "W1", 64, "Random", 0)
    ledger = runtime.Ledger(scratch / "ledger", 1, meta, targets, batch=batch)
    started = time.perf_counter()
    for offset in range(ledger.position, trials * 100, batch):
        budget.check()
        payload, lengths = data.prior_candidates(namespace, np.arange(offset, offset + batch, dtype=np.uint64), "P")
        ledger.append(offset, data.messages(payload, lengths))
    ledger.commit()
    ledger.close()
    written = trials * 100 / (time.perf_counter() - started)
    started = time.perf_counter()
    runtime.verify_ledger(ledger.path, targets, meta, budget=budget)
    return written, trials * 100 / (time.perf_counter() - started)


def phase_prof1(root, budget):
    from .models import parameter_count
    path = root / "A-prof-1.json"
    if path.exists():
        return read_json(path)
    scratch = root / "A-prof-tmp"
    shutil.rmtree(scratch, ignore_errors=True)
    models = {}
    for pipeline in PIPELINES:
        model, update_seconds = _timed_updates(pipeline, None, scratch, budget)
        options = REG["models"]["G3-U"]["sampling_steps_options"] if pipeline in GAUSSIAN else [None]
        models[pipeline] = {"update_seconds": update_seconds, "parameters": parameter_count(model),
                            "forward_rows_per_second": _forward_rows(model, pipeline),
                            "by_steps": _generation_profile(model, pipeline, options, budget)}
        del model
        release()
    model, update_seconds = _timed_updates("P-DISC", REG["stage_s"]["model"], scratch, budget)
    scale = {"model": REG["stage_s"]["model"], "update_seconds": update_seconds, "parameters": parameter_count(model),
             "forward_rows_per_second": _forward_rows(model, "P-DISC"),
             "by_steps": _generation_profile(model, "P-DISC", [None], budget)}
    del model
    release()
    random_ledger_cps, verify_rows = _ledger_rates(scratch, budget)
    result = {"models": models, "scale": scale, "md5_per_second": _md5_rate(), "random_ledger_cps": random_ledger_cps,
              "verify_rows_per_second": verify_rows, "record_bytes": runtime.RECORD.itemsize,
              "environment": environment(),
              "scope": "random-weight models, synthetic stream and labels, Random fixture on exposed W1; no study condition data"}
    shutil.rmtree(scratch, ignore_errors=True)
    sealed_json(path, result)
    return result


def a_q_folder(root, pipeline, seed_id, updates):
    return run_folder(root, "A-Q", pipeline, "Main", seed_id, updates)


def phase_train(root, budget):
    groups = data.synthetic_split()
    for pipeline in PIPELINES:
        for seed_id in SEEDS:
            runtime.train(a_q_folder(root, pipeline, seed_id, UPDATES), pipeline, "A-Q", seed_id, groups,
                          updates=UPDATES, task="synthetic", budget=budget)
            release()


def dev_rows(model, pipeline, budget, *, generator=generate):
    """Joint on dev 256 x {normal, flipped} and duplicates on the first 64 dev conditions x 100, per S_G."""
    from .models import candidate_keys
    src = source_of(pipeline)
    per = REG["synthetic"]["diversity_candidates"]
    dev = np.array(sorted(data.synthetic_split()["validation"]), dtype=np.int32)
    diversity = np.repeat(dev[:REG["synthetic"]["diversity_conditions"]], per)
    rows = {}
    for steps in REG["models"]["G3-U"]["sampling_steps_options"]:
        budget.check()
        row = {}
        for variant, labels in (("normal", dev), ("flipped", dev ^ 4095)):
            keys = candidate_keys(("A-dev", pipeline, variant, 0), np.arange(len(labels)))
            messages = generator(model, labels, keys, steps, 256)[0]
            row[f"{variant}_joint"] = int(sum(data.valid(m, src) and data.synthetic_label(m, src) == int(y)
                                              for m, y in zip(messages, labels)))
        keys = candidate_keys(("A-dev", pipeline, "diversity", 0), np.arange(len(diversity)))
        messages = generator(model, diversity, keys, steps, 256)[0]
        duplicates = sum(per - len(set(messages[i:i + per])) for i in range(0, len(messages), per))
        row.update(duplicates=int(duplicates), duplicate_rate=duplicates / len(messages))
        rows[str(steps)] = row
    return rows


def select_steps(rows):
    rule = REG["dev_selection"]
    baseline = rows[str(rule["default_steps"])]["duplicate_rate"]
    for steps in REG["models"]["G3-U"]["sampling_steps_options"]:
        row = rows[str(steps)]
        if (row["normal_joint"] >= rule["joint_min"] and row["flipped_joint"] >= rule["joint_min"]
                and row["duplicate_rate"] <= baseline + rule["duplicate_tolerance"]):
            return steps
    return rule["default_steps"]


def dev_selection(root, pipeline, updates, budget):
    path = root / "A-dev" / pipeline / f"u{updates}" / "selection.json"
    if path.exists():
        return read_json(path)
    model, checkpoint = load_trained(a_q_folder(root, pipeline, 0, updates), pipeline, "A-Q", 0)
    rows = dev_rows(model, pipeline, budget)
    del model
    release()
    result = {"updates": updates, "checkpoint": checkpoint, "by_steps": rows, "selected": select_steps(rows)}
    sealed_json(path, result)
    return result


def phase_dev(root, budget):
    path = root / "A-dev.json"
    if path.exists():
        return read_json(path)
    result = {"rule": REG["dev_selection"], "pipelines": {p: dev_selection(root, p, UPDATES, budget) for p in GAUSSIAN}}
    sealed_json(path, result)
    return result


def qualify_seed(root, pipeline, seed_id, updates, steps, batch, budget):
    from .models import candidate_keys
    folder = a_q_folder(root, pipeline, seed_id, updates)
    path = folder / "qualification.json"
    if path.exists():
        result = read_json(path)
        if result["sampling_steps"] != steps:
            raise ValueError("Qualification was sealed with different sampling steps")
        return result
    model, checkpoint = load_trained(folder, pipeline, "A-Q", seed_id)
    src, q = source_of(pipeline), REG["qualification"]
    labels = np.array(data.synthetic_split()["acceptance"], dtype=np.int32)
    batch = min(batch, len(labels))
    observed = {}
    for variant, targets in (("normal", labels), ("flipped", labels ^ 4095)):
        budget.check()
        keys = candidate_keys(("A-Q", pipeline, variant, seed_id), np.arange(len(targets)))
        messages = generate(model, targets, keys, steps, batch)[0]
        observed[variant] = [data.synthetic_label(m, src) if data.valid(m, src) else -1 for m in messages]
        observed[variant + "_valid"] = int(sum(data.valid(m, src) for m in messages))
    values, _ = runtime.likelihood_probe(model, "A-Q", "qualification", pipeline, seed_id, labels.tolist(),
                                         pairs=q["clp_pairs"], task="synthetic", budget=budget)
    original = labels.tolist()
    result = {"pipeline": pipeline, "seed_id": seed_id, "updates": updates, "sampling_steps": steps, "batch": batch,
              "normal_joint": int(sum(v == y for v, y in zip(observed["normal"], original))),
              "flipped_joint": int(sum(v == (y ^ 4095) for v, y in zip(observed["flipped"], original))),
              "wrong_original": int(sum(v == y for v, y in zip(observed["flipped"], original))),
              "normal_valid": observed["normal_valid"], "flipped_valid": observed["flipped_valid"],
              "clp": st.probe(values, q["clp_z"]), "checkpoint": checkpoint, "md5_calls": 0}
    result["generation_pass"] = bool(result["normal_joint"] >= q["joint_min"] and result["flipped_joint"] >= q["joint_min"]
                                     and result["normal_valid"] == result["flipped_valid"] == q["valid"]
                                     and result["wrong_original"] <= q["wrong_max"])
    del model
    release()
    sealed_json(path, result)
    return result


def qualification_record(rounds, steps, profile):
    """Q holds pipelines whose three seeds pass the generation criteria; a CLP-only miss disables CLP (plan §5.3)."""
    first = rounds[0]["rows"]
    qualified = [p for p in PIPELINES if all(r["generation_pass"] for r in first[p])]
    failing = [p for p in PIPELINES if p not in qualified]
    updates = {p: UPDATES for p in PIPELINES}
    state = "pending" if failing and len(rounds) == 1 else ("done" if failing else "not-needed")
    for p, rows in (rounds[1]["rows"].items() if len(rounds) > 1 else ()):
        if all(r["generation_pass"] for r in rows):
            qualified.append(p)
            updates[p] = REPAIR_UPDATES
    qualified = [p for p in PIPELINES if p in qualified]
    c1 = {p: "PASS" if p in qualified else ("PENDING_REPAIR" if state == "pending" else "UNTESTABLE") for p in PIPELINES}
    clp_disabled = {p: any(r["generation_pass"] and not r["clp"]["positive"] for rnd in rounds for r in rnd["rows"].get(p, []))
                    for p in PIPELINES}
    settings = {p: {"updates": updates[p], "sampling_steps": steps.get(p),
                    "batch": profile["models"][p]["by_steps"][steps_key(p, steps.get(p))]["batch"]}
                for p in qualified}
    return {"rounds": rounds, "Q": qualified, "C1": c1, "repair_state": state, "repair_used": len(rounds) > 1,
            "sampling_steps": steps, "clp_disabled": clp_disabled, "settings": settings}


def phase_evaluate(root, budget):
    path = root / "A.json"
    if path.exists():
        return read_json(path)
    require(root, "A-prof-1.json", "A-dev.json")
    profile = read_json(root / "A-prof-1.json")
    dev = read_json(root / "A-dev.json")["pipelines"]
    steps = {p: dev[p]["selected"] for p in GAUSSIAN}
    rows = {}
    for p in PIPELINES:
        batch = profile["models"][p]["by_steps"][steps_key(p, steps.get(p))]["batch"]
        rows[p] = [qualify_seed(root, p, s, UPDATES, steps.get(p), batch, budget) for s in SEEDS]
    record = qualification_record([{"updates": UPDATES, "rows": rows}], steps, profile)
    atomic_json(path, record)
    return record


def phase_repair(root, budget):
    """Plan §5.5: continue failing runs to 80k, re-select S_G for Gaussian ones, evaluate round 2."""
    record = read_json(root / "A.json")
    if record["repair_state"] != "pending":
        return record
    profile = read_json(root / "A-prof-1.json")
    failing = [p for p in PIPELINES if record["C1"][p] == "PENDING_REPAIR"]
    groups = data.synthetic_split()
    for p in failing:
        for s in SEEDS:
            runtime.train(a_q_folder(root, p, s, REPAIR_UPDATES), p, "A-Q", s, groups, updates=REPAIR_UPDATES,
                          task="synthetic", resume_from=a_q_folder(root, p, s, UPDATES), budget=budget)
            release()
    path = root / "A-dev-repair.json"
    if path.exists():
        repaired = read_json(path)["pipelines"]
    else:
        repaired = {p: dev_selection(root, p, REPAIR_UPDATES, budget) for p in failing if p in GAUSSIAN}
        sealed_json(path, {"rule": REG["dev_selection"], "pipelines": repaired})
    steps = {**record["sampling_steps"], **{p: v["selected"] for p, v in repaired.items()}}
    rows = {}
    for p in failing:
        batch = profile["models"][p]["by_steps"][steps_key(p, steps.get(p))]["batch"]
        rows[p] = [qualify_seed(root, p, s, REPAIR_UPDATES, steps.get(p), batch, budget) for s in SEEDS]
    record = qualification_record([*record["rounds"], {"updates": REPAIR_UPDATES, "rows": rows}], steps, profile)
    atomic_json(root / "A.json", record)
    return record


_TRAINING_FIXTURE = []


def _training_fixture():
    """Random digests as many as one 40k-update stream: the lookup cost of Stage C, never a match."""
    if not _TRAINING_FIXTURE:
        count = UPDATES * REG["training"]["batch"]
        raw = data.rng("A-prof-2", "training").bytes(16 * count)
        _TRAINING_FIXTURE.append(np.sort(np.frombuffer(raw, dtype="V16")))
    return _TRAINING_FIXTURE[0]


def profile_trials(cps):
    """Blocks of about PROFILE_BLOCK_SECONDS; a multiple of 512 trials keeps every batch option whole."""
    return int(max(512, 512 * round(PROFILE_BLOCK_SECONDS * cps / 100 / 512)))


def sustained(folder, budget, *, pipeline=None, model=None, source=None, batch=RANDOM_BATCH, steps=None,
              cps=None, warmup=None, measure=None):
    """Stage C path per block: generate, decode, hash, training lookup, ledger, verifier, regeneration audit."""
    profile = REG["profiling"]
    warmup = profile["warmup_seconds"] if warmup is None else warmup
    measure = profile["measure_seconds"] if measure is None else measure
    last = min(profile["last_window_seconds"], measure)
    src = source or source_of(pipeline)
    trials = profile_trials(cps) if cps else 2048
    targets_rng = data.rng("A-prof-2", pipeline or f"Random-{src}")
    training = _training_fixture() if model is not None else None
    folder = Path(folder)
    shutil.rmtree(folder, ignore_errors=True)
    targets, marks, block = np.zeros(0, dtype=np.int32), [], 0
    started = time.perf_counter()
    while not marks or marks[-1][0] < warmup + measure:
        budget.check()
        targets = np.concatenate((targets, targets_rng.integers(0, 4096, trials).astype(np.int32)))
        runtime.evaluate_block(folder, block, targets, stage="A-prof-2", source=src,
                               method="Random" if model is None else "Main", seed_id=0, pipeline=pipeline,
                               model=model, window="W1", rung=64, task="md5", start_trial=block * trials,
                               trials=trials, batch=batch, steps=steps or 25, training=training, budget=budget)
        (folder / f"block-{block}.bin").unlink()
        block += 1
        marks.append((time.perf_counter() - started, block * trials * 100))
    shutil.rmtree(folder, ignore_errors=True)
    t1, n1 = marks[-1]
    t0, n0 = next(((t, n) for t, n in marks if t >= warmup), marks[0])
    cumulative = (n1 - n0) / (t1 - t0) if t1 > t0 else n1 / t1
    tl, nl = next(((t, n) for t, n in reversed(marks) if t <= t1 - last), (t0, n0))
    last_window = (n1 - nl) / (t1 - tl) if t1 > tl else cumulative
    return {"sustained_cps": min(cumulative, last_window), "cumulative_cps": cumulative, "last_window_cps": last_window,
            "candidates": n1, "blocks": block, "block_trials": trials, "seconds": t1,
            "warmup_seconds": warmup, "measure_seconds": measure}


def phase_prof2(root, budget):
    path = root / "A-prof-2.json"
    if path.exists():
        return read_json(path)
    record = read_json(root / "A.json")
    if record["repair_state"] == "pending":
        raise ValueError("Run the repair phase before A-prof-2")
    profile = read_json(root / "A-prof-1.json")
    scratch = root / "A-prof-2-tmp"
    models = {}
    for p in record["Q"]:
        setting = record["settings"][p]
        burst = profile["models"][p]["by_steps"][steps_key(p, setting["sampling_steps"])]["candidates_per_second"]
        model, _ = load_trained(a_q_folder(root, p, 0, setting["updates"]), p, "A-Q", 0)
        models[p] = {**sustained(scratch / p, budget, pipeline=p, model=model, batch=setting["batch"],
                                 steps=setting["sampling_steps"], cps=burst), **setting}
        del model
        release()
    # Random is CPU-bound and steady; a short run measures the same full path.
    random = {src: sustained(scratch / f"Random-{src}", budget, source=src, warmup=10, measure=120)
              for src in sorted({source_of(p) for p in record["Q"]})}
    result = {"models": models, "random_cps": min(v["sustained_cps"] for v in random.values()), "random": random,
              "verify_rows_per_second": profile["verify_rows_per_second"], "environment": environment(),
              "scope": "seed-0 synthetic checkpoints, random 12-bit targets on exposed W1; no study condition data"}
    shutil.rmtree(scratch, ignore_errors=True)
    sealed_json(path, result)
    return result


def phase_budget(root):
    require(root, "A.json", "A-prof-1.json", "A-prof-2.json")
    record = read_json(root / "A.json")
    seconds = read_json(root / "budget.json")["seconds"]
    caps = effective_caps(root)
    plan = budget_plan(read_json(root / "A-prof-1.json"), read_json(root / "A-prof-2.json"), record,
                       seconds.get("A", 0) + seconds.get("A_repair", 0), caps=caps, repair_used=record["repair_used"])
    atomic_json(root / "budget-plan.json", plan)
    if plan["decision"] == "HALT":
        atomic_json(root / "halt.json", {"decision": "HALT", "required_c_hours": plan["required_c_hours"],
                                         "estimates": plan["estimates"], "caps_hours": caps})
        raise Halt(f"Stage C needs a cap of at least {plan['required_c_hours']:.1f} h; approve-caps before any MD5 data")
    return plan


def stage_a(root, phase="all"):
    if (root / "protocol.frozen.json").exists():
        return verify_frozen(root)
    current = None
    try:
        for current in (A_PHASES if phase == "all" else (phase,)):
            if current != "impl":
                require_impl(root)
            if current in ("repair", "prof2", "budget", "freeze"):
                require(root, "A.json")
                record = read_json(root / "A.json")
                if current == "repair" and record["repair_state"] != "pending":
                    continue
                if current != "repair" and not record["Q"]:
                    return None
            if current == "budget":
                phase_budget(root)
            elif current == "freeze":
                require(root, "budget-plan.json", "exposure-audit.json")
                freeze(root)
            else:
                with charged(root, "A_repair" if current == "repair" else "A") as budget:
                    {"impl": phase_impl, "prof1": phase_prof1, "train": phase_train, "dev": phase_dev,
                     "evaluate": phase_evaluate, "repair": phase_repair, "prof2": phase_prof2}[current](root, budget)
    except BudgetExceeded as error:
        sealed_json(root / "failure.json", {"stage": "A_repair" if current == "repair" else "A", "phase": current,
                                            "reason": "NOT_ESTABLISHED_BY_BUDGET", "error": str(error)})
        raise
    return None


# ---------------------------------------------------------------- 공통: 학습 run, stream, 무결성

def frozen_context(root):
    require(root, "protocol.frozen.json")
    frozen = verify_frozen(root)
    plan = frozen["settings"]
    return plan, plan["pipelines"], list(frozen["Q"])


def load_failures(root, stage):
    path = Path(root) / stage / "integrity.json"
    return read_json(path) if path.exists() else {}


def record_failure(root, stage, pipeline, reason):
    failures = load_failures(root, stage)
    failures.setdefault(pipeline, reason)
    atomic_json(Path(root) / stage / "integrity.json", failures)
    emit({"stage": stage, "integrity_failure": pipeline})
    return failures


def train_stage(root, stage, pipelines, settings, groups, window, rung, methods, seeds, budget, *,
                architecture=None, updates=None):
    folders = {}
    for p in pipelines:
        if p in load_failures(root, stage):
            continue
        count = updates or settings[p]["updates"]
        try:
            for method in methods:
                for s in seeds:
                    folder = run_folder(root, stage, p, method, s, count, architecture)
                    runtime.train(folder, p, stage, s, groups, updates=count, method=method, task="md5", window=window,
                                  rung=rung, architecture=architecture, stream_root=root / stage / "streams", budget=budget)
                    folders[(p, method, s)] = folder
                    release()
        except BudgetExceeded:
            raise
        except INTEGRITY_ERRORS as error:
            record_failure(root, stage, p, f"training: {error}")
    return folders, load_failures(root, stage)


def schedule(root, stage, window, rung, groups, folders, trials):
    path = root / stage / "trials.json"
    if path.exists():
        value = read_json(path)
        for folder, digest in value["checkpoints"].items():
            if file_hash(Path(folder) / "complete.json") != digest:
                raise ValueError("A checkpoint changed after the trial schedule was drawn")
        return np.asarray(value["targets"], dtype=np.int32)
    return runtime.trial_schedule(path, stage, window, rung, groups["test"], folders, trials)


_INDEX = {}


def training_lookup(folder):
    """Sorted training digests of a run; runs sharing a source-seed stream share one index."""
    key = tuple(sorted(runtime.verify_training(folder)["digest_segments"]))
    if key not in _INDEX:
        _INDEX[key] = runtime.training_index([folder])
    return _INDEX[key]


def stream_folder(root, stage, method, seed_id, *, pipeline=None, source=None, architecture=None):
    label = source if method == "Random" else (pipeline if architecture is None else f"{pipeline}-{architecture}")
    return Path(root) / stage / "eval" / label / f"{method}-{seed_id}"


def stream_block(root, stage, block, targets, start, trials, budget, *, method, seed_id, window, rung,
                 pipeline=None, run=None, source=None, architecture=None, batch=None, steps=None):
    folder = stream_folder(root, stage, method, seed_id, pipeline=pipeline, source=source, architecture=architecture)
    if method == "Random":
        return runtime.evaluate_block(folder, block, targets, stage=stage, source=source, method="Random", seed_id=seed_id,
                                      window=window, rung=rung, task="md5", start_trial=start, trials=trials,
                                      batch=RANDOM_BATCH, budget=budget)
    model, checkpoint = load_trained(run, pipeline, stage, seed_id, architecture)
    try:
        return runtime.evaluate_block(folder, block, targets, stage=stage, source=source_of(pipeline), method=method,
                                      seed_id=seed_id, pipeline=pipeline, model=model, window=window, rung=rung, task="md5",
                                      start_trial=start, trials=trials, batch=batch, steps=steps or 25,
                                      training=training_lookup(run), checkpoint=checkpoint, budget=budget)
    finally:
        del model
        release()


def evaluate_streams(root, stage, block, targets, start, trials, pipelines, settings, folders, budget, *, window, rung,
                     methods, seeds, architecture=None, batch=None):
    """Learned streams, then Random per source; an integrity error removes only the affected pipelines."""
    for p in pipelines:
        if p in load_failures(root, stage):
            continue
        try:
            for method in methods:
                for s in seeds:
                    stream_block(root, stage, block, targets, start, trials, budget, method=method, seed_id=s,
                                 window=window, rung=rung, pipeline=p, run=folders[(p, "Main" if method == "MC" else method, s)],
                                 architecture=architecture, batch=batch or settings[p]["batch"],
                                 steps=None if settings is None else settings[p]["sampling_steps"])
        except BudgetExceeded:
            raise
        except INTEGRITY_ERRORS as error:
            record_failure(root, stage, p, f"stream: {error}")
    for src in sorted({source_of(p) for p in pipelines if p not in load_failures(root, stage)}):
        try:
            for s in seeds:
                stream_block(root, stage, block, targets, start, trials, budget, method="Random", seed_id=s,
                             window=window, rung=rung, source=src)
        except BudgetExceeded:
            raise
        except INTEGRITY_ERRORS as error:
            for p in pipelines:
                if source_of(p) == src:
                    record_failure(root, stage, p, f"Random-{src}: {error}")


def success(folder, blocks):
    """Success@100 bits of sealed blocks, checked against each block's seal."""
    values = []
    for b in blocks:
        result = read_json(folder / f"block-{b}.json")
        name = f"block-{b}.trials.npy"
        if file_hash(folder / name) != result["artifacts"][name]:
            raise ValueError(f"Trial summary changed: {folder / name}")
        values.append(np.load(folder / name, allow_pickle=False)["at100"].astype(int))
    return np.concatenate(values)


def outcomes(root, stage, pipelines, blocks, seeds=SEEDS, controls=("Random", "Shuffled"), architecture=None):
    result = {}
    for p in pipelines:
        row = {"Main": np.stack([success(stream_folder(root, stage, "Main", s, pipeline=p, architecture=architecture), blocks)
                                 for s in seeds])}
        for control in controls:
            folders = [stream_folder(root, stage, "Random", s, source=source_of(p)) if control == "Random"
                       else stream_folder(root, stage, control, s, pipeline=p, architecture=architecture) for s in seeds]
            row[control] = np.stack([success(folder, blocks) for folder in folders])
        result[p] = row
    return result


def clp_path(root, stage, pipeline, seed_id, architecture=None):
    label = pipeline if architecture is None else f"{pipeline}-{architecture}"
    return Path(root) / stage / "clp" / f"{label}-seed{seed_id}.json"


def clp_probe(root, stage, pipeline, folder, seed_id, groups, *, window, rung, pairs, threshold, budget, architecture=None):
    """Compute and seal one CLP probe inside a stage session; a sealed probe is reused."""
    path = clp_path(root, stage, pipeline, seed_id, architecture)
    if path.exists():
        return read_json(path)
    model, _ = load_trained(folder, pipeline, stage, seed_id, architecture)
    differences, _ = runtime.likelihood_probe(model, stage, "test", pipeline, seed_id, groups["test"], pairs=pairs,
                                              task="md5", window=window, rung=rung, budget=budget)
    del model
    release()
    row = {"differences": differences.tolist(), "probe": st.probe(differences, threshold)}
    sealed_json(path, row)
    return row


def sealed_clp(root, stage, pipeline, seeds, architecture=None):
    """Sealed CLP rows for every seed, or None while any is missing."""
    paths = [clp_path(root, stage, pipeline, s, architecture) for s in seeds]
    return [read_json(path) for path in paths] if all(path.exists() for path in paths) else None


def sealed_blocks(root, stage, pipeline, blocks, seeds, controls, architecture=None):
    """True when every stream a pipeline's analysis needs has sealed these blocks."""
    folders = [stream_folder(root, stage, "Main", s, pipeline=pipeline, architecture=architecture) for s in seeds]
    for control in controls:
        folders += [stream_folder(root, stage, "Random", s, source=source_of(pipeline)) if control == "Random"
                    else stream_folder(root, stage, control, s, pipeline=pipeline, architecture=architecture) for s in seeds]
    return all((folder / f"block-{b}.json").exists() for folder in folders for b in blocks)


def stream_metrics(root, stage, pipeline, blocks):
    """Quality counts over the analysed blocks of a pipeline's Main streams."""
    rows = [read_json(stream_folder(root, stage, "Main", s, pipeline=pipeline) / f"block-{b}.json") for s in SEEDS for b in blocks]
    keys = ("rows", "hits", "valid", "duplicates", "training_matches", "strict_valid")
    return {k: int(sum(r[k] for r in rows)) for k in keys}


def artifact_audit(root, stage, pipeline, blocks, trials, updates, clp):
    """Plan §6.5: independent re-hash of every success, no training match, target concentration, CLP direction."""
    targets = np.asarray(read_json(root / stage / "trials.json")["targets"], dtype=np.int64)
    spec, shift = data.SOURCES[source_of(pipeline)], data.WINDOWS["W3"]
    per_target, successes, mismatches, matches, sealed = {}, 0, 0, 0, True
    for s in SEEDS:
        folder = stream_folder(root, stage, "Main", s, pipeline=pipeline)
        training = training_lookup(run_folder(root, stage, pipeline, "Main", s, updates))
        for b in blocks:
            result = read_json(folder / f"block-{b}.json")
            sealed &= all(file_hash(folder / name) == digest for name, digest in result["artifacts"].items())
            records = np.memmap(folder / f"block-{b}.bin", dtype=runtime.RECORD, mode="r")
            for row in np.flatnonzero(records["flags"] & 2):
                message = records[row]["payload"][:records[row]["length"]].tobytes()
                target = int(targets[(result["start"] + int(row)) // 100])
                value = (int.from_bytes(hashlib.md5(message).digest(), "big") >> shift) & 4095
                valid = 4 <= len(message) <= 31 and all(spec["byte_min"] <= x <= spec["byte_max"] for x in message)
                mismatches += int(not valid or value != target)
                digest = np.frombuffer(hashlib.sha256(message).digest()[:16], dtype="V16")
                matches += int(runtime.lookup_digests(digest, training)[0])
                per_target[target] = per_target.get(target, 0) + 1
                successes += 1
    top = max(1, math.ceil(len(set(targets[:trials].tolist())) * .01))
    share = sum(sorted(per_target.values(), reverse=True)[:top]) / successes if successes else 0.
    info = None if clp is None or clp.get("disabled") else clp["INFO_64"]
    return {"passed": bool(sealed and mismatches == 0 and matches == 0), "sealed_artifacts": bool(sealed),
            "successful_payloads": successes, "rehash_mismatches": mismatches, "successful_training_matches": matches,
            "top_1pct_targets": top, "top_1pct_target_hit_share": share,
            "clp_direction": None if info is None else ("positive" if clp["pooled"]["estimate"] > 0 else "non-positive"),
            "INFO_64": info, "hit_only": None if info is None else not info}


def budget_stops(root):
    """Stages whose sealed result was cut by their cap (no effect information)."""
    flags = {"C": ("C.json", "budget_stop"), "R": ("R.json", "budget_stop"), "P": ("P.json", "partial"),
             "S": ("S.json", "partial")}
    return [stage for stage, (name, key) in flags.items() if (optional(root, name) or {}).get(key)]


# ---------------------------------------------------------------- Stage C: 주 판정
# C·R·P·S는 세션 안에서 학습·생성·CLP를 수행해 봉인하고, 판정은 세션 밖에서 봉인된 자료만으로 조립한다.
# 예산이 도중에 소진되든 재실행 세션을 여는 순간 이미 소진되어 있든 같은 규칙으로 결과가 정해진다.

def next_block_fits(root, stage, previous, budget):
    remaining = effective_caps(root)[stage] * 3600 - budget.state["seconds"].get(stage, 0)
    return previous["block_seconds"] * REG["budget"]["next_block_safety"] <= remaining


def sealed_looks(root):
    looks = []
    for look in range(1, REG["stage_c"]["looks"] + 1):
        path = root / "C" / "looks" / f"look-{look}.json"
        if not path.exists():
            break
        looks.append(read_json(path))
    return looks


def final_look(root, block, active, completed):
    """The sealed stopping look, else the full classification at the last completed look (budget rule)."""
    if not completed or not active:
        return None
    if completed[-1]["action"] == "stop":
        return completed[-1]
    last = completed[-1]["look"]
    return {**st.stage_c(outcomes(root, "C", active, range(1, last + 1)), last, budget_stop=True),
            "trials": block * last, "active": active}


def run_looks(root, plan, settings, qualified, groups, budget):
    """Train, draw the schedule, then generate blocks and seal looks until the joint rule or the next-block check stops."""
    block, looks = plan["block"], REG["stage_c"]["looks"]
    folders, failures = train_stage(root, "C", qualified, settings, groups, "W3", 64, ("Main", "Shuffled"), SEEDS, budget)
    active = [p for p in qualified if p not in failures]
    if not active:
        return
    targets = schedule(root, "C", "W3", 64, groups, [f for (p, _, _), f in folders.items() if p in active], looks * block)
    previous = None
    for look in range(1, looks + 1):
        path = root / "C" / "looks" / f"look-{look}.json"
        if not path.exists():
            start = root / "C" / "looks" / f"block-{look}-start.json"
            if not start.exists():
                if previous and not next_block_fits(root, "C", previous, budget):
                    return
                atomic_json(start, {"seconds": budget.state["seconds"].get("C", 0)})
            evaluate_streams(root, "C", look, targets, (look - 1) * block, block, active, settings, folders, budget,
                             window="W3", rung=64, methods=("Main", "Shuffled"), seeds=SEEDS)
            active = [p for p in qualified if p not in load_failures(root, "C")]
            if not active:
                return
            result = st.stage_c(outcomes(root, "C", active, range(1, look + 1)), look)
            elapsed = budget.state["seconds"].get("C", 0) - read_json(start)["seconds"]
            sealed_json(path, {**result, "trials": block * look, "active": active, "block_seconds": elapsed})
        previous = read_json(path)
        emit({"look": look, "action": previous["action"]})
        if previous["action"] == "stop":
            return


def stage_c(root):
    if (root / "C.json").exists():
        return read_json(root / "C.json")
    plan, settings, qualified = frozen_context(root)
    clp_disabled = read_json(root / "A.json")["clp_disabled"]
    audit = read_json(root / "exposure-audit.json")
    block = plan["block"]
    groups = data.split("W3", 64, audit["excluded"]["W3"])
    sealed_json(root / "C" / "groups-W3-r64.json", groups)
    try:
        with charged(root, "C") as budget:
            run_looks(root, plan, settings, qualified, groups, budget)
            active = [p for p in qualified if p not in load_failures(root, "C")]
            final = final_look(root, block, active, sealed_looks(root))
            for p in [p for p in active if final and p in final["pipelines"] and not clp_disabled.get(p)]:
                for s in SEEDS:
                    clp_probe(root, "C", p, run_folder(root, "C", p, "Main", s, settings[p]["updates"]), s, groups,
                              window="W3", rung=64, pairs=REG["stage_c"]["clp_pairs_per_seed"], threshold=st.Z_P,
                              budget=budget)
    except BudgetExceeded:
        pass
    failures = load_failures(root, "C")
    active = [p for p in qualified if p not in failures]
    completed = sealed_looks(root)
    final = final_look(root, block, active, completed)
    final_active = [p for p in active if final and p in final["pipelines"]]
    blocks = range(1, final["look"] + 1) if final else range(0)
    decisions = {p: final["pipelines"][p]["decision"] if p in final_active else "NOT_ESTABLISHED_BY_BUDGET"
                 for p in active}
    decisions.update({p: "NOT_ESTABLISHED_INTEGRITY" for p in failures if p in qualified})
    clp = {}
    for p in final_active:
        rows = None if clp_disabled.get(p) else sealed_clp(root, "C", p, SEEDS)
        if clp_disabled.get(p):
            clp[p] = {"disabled": True, "z": None, "INFO_64": None}
        elif rows:
            pooled = st.probe(np.concatenate([r["differences"] for r in rows]), st.Z_P)
            clp[p] = {"per_seed": {str(s): r["probe"] for s, r in zip(SEEDS, rows)}, "pooled": pooled,
                      "z": pooled["z"], "INFO_64": bool(pooled["z"] is not None and pooled["z"] > st.Z_P)}
    audits = {p: artifact_audit(root, "C", p, blocks, block * final["look"], settings[p]["updates"], clp.get(p))
              for p in final_active if decisions[p] == "POSITIVE"}
    result = {"block": block, "looks": completed, "final": final,
              "budget_stop": bool(active) and (not completed or completed[-1]["action"] != "stop"),
              "decisions": decisions, "integrity": failures, "clp": clp,
              "clp_partial": any(p not in clp for p in final_active), "audit": audits,
              "contrasts": st.contrasts(outcomes(root, "C", final_active, blocks)) if final_active else [],
              "metrics": {p: stream_metrics(root, "C", p, blocks) for p in final_active}}
    sealed_json(root / "C.json", result)
    return result


def replication_entrants(c):
    return [p for p in PIPELINES if c["decisions"].get(p) == "POSITIVE" and c["audit"].get(p, {}).get("passed")]


def require_order(root, stage):
    """Registered order A -> C -> (R) -> P -> (S)."""
    require(root, "C.json")
    if stage in ("P", "S") and replication_entrants(read_json(root / "C.json")) and not (root / "R.json").exists():
        raise ValueError("Stage R must finish before P and S")
    if stage == "S":
        require(root, "P.json")


# ---------------------------------------------------------------- Stage R: W4 재현

def stage_r(root):
    if (root / "R.json").exists():
        return read_json(root / "R.json")
    require_order(root, "R")
    _, settings, _ = frozen_context(root)
    entrants = replication_entrants(read_json(root / "C.json"))
    trials = REG["stage_r"]["trials"]
    if entrants:
        audit = read_json(root / "exposure-audit.json")
        groups = data.split("W4", 64, audit["excluded"]["W4"])
        sealed_json(root / "R" / "groups-W4-r64.json", groups)
        try:
            with charged(root, "R") as budget:
                folders, failures = train_stage(root, "R", entrants, settings, groups, "W4", 64, ("Main", "Shuffled"),
                                                SEEDS, budget)
                active = [p for p in entrants if p not in failures]
                if active:
                    targets = schedule(root, "R", "W4", 64, groups,
                                       [f for (p, _, _), f in folders.items() if p in active], trials)
                    evaluate_streams(root, "R", 1, targets, 0, trials, active, settings, folders, budget,
                                     window="W4", rung=64, methods=("Main", "Shuffled"), seeds=SEEDS)
        except BudgetExceeded:
            pass
    failures = load_failures(root, "R")
    results = {}
    for p in entrants:
        if p not in failures and sealed_blocks(root, "R", p, [1], SEEDS, ("Random", "Shuffled")):
            row = outcomes(root, "R", [p], [1])[p]
            results[p] = st.replication(row["Main"], row["Random"], row["Shuffled"], len(entrants))
    decisions = {p: "NOT_ESTABLISHED_INTEGRITY" if p in failures else
                 results[p]["decision"] if p in results else "NOT_ESTABLISHED_BY_BUDGET" for p in entrants}
    result = {"entrants": entrants, "decisions": decisions, "results": results, "integrity": failures,
              "budget_stop": any(v == "NOT_ESTABLISHED_BY_BUDGET" for v in decisions.values())}
    sealed_json(root / "R.json", result)
    return result


# ---------------------------------------------------------------- Stage P: r=4 양성 대조

def stage_p(root):
    if (root / "P.json").exists():
        return read_json(root / "P.json")
    require_order(root, "P")
    plan, settings, qualified = frozen_context(root)
    clp_disabled = read_json(root / "A.json")["clp_disabled"]
    groups = data.split(REG["stage_p"]["window"], REG["stage_p"]["rung"])
    sealed_json(root / "P" / "groups-W1-r4.json", groups)
    trials = plan["p_trials"]
    try:
        with charged(root, "P") as budget:
            folders, failures = train_stage(root, "P", qualified, settings, groups, "W1", 4, ("Main",), (0,), budget)
            active = [p for p in qualified if p not in failures]
            if active:
                targets = schedule(root, "P", "W1", 4, groups, [folders[(p, "Main", 0)] for p in active], trials)
                evaluate_streams(root, "P", 1, targets, 0, trials, active, settings, folders, budget, window="W1", rung=4,
                                 methods=("Main", "MC"), seeds=(0,))
            for p in active:
                if p not in load_failures(root, "P") and not clp_disabled.get(p):
                    clp_probe(root, "P", p, folders[(p, "Main", 0)], 0, groups, window="W1", rung=4,
                              pairs=REG["stage_p"]["clp_pairs"], threshold=st.Z_P, budget=budget)
    except BudgetExceeded:
        pass
    failures = load_failures(root, "P")
    results = {}
    for p in qualified:
        if p in failures or not sealed_blocks(root, "P", p, [1], (0,), ("Random", "MC")):
            continue
        row = outcomes(root, "P", [p], [1], seeds=(0,), controls=("Random", "MC"))[p]
        rows = None if clp_disabled.get(p) else sealed_clp(root, "P", p, (0,))
        results[p] = {**st.positive_control(row["Main"][0], row["Random"][0], row["MC"][0],
                                            rows[0]["probe"] if rows else {"z": None}),
                      "INFO_4_disabled": bool(clp_disabled.get(p)),
                      "success_at_100": {k: float(v.mean()) for k, v in row.items()}}
        if not rows:
            results[p]["INFO_4"] = None
    missing = [p for p in qualified if p not in failures
               and (p not in results or (results[p]["INFO_4"] is None and not clp_disabled.get(p)))]
    result = {"trials": trials, "pipelines": results, "partial": bool(missing), "integrity": failures}
    sealed_json(root / "P.json", result)
    return result


# ---------------------------------------------------------------- Stage S: 규모 탐침 (선택)

def stage_s(root):
    if (root / "S.json").exists():
        return read_json(root / "S.json")
    require_order(root, "S")
    plan, _, qualified = frozen_context(root)
    cfg = REG["stage_s"]
    if not plan["stage_s"] or cfg["pipeline"] not in qualified:
        result = {"skipped": True, "reason": "budget plan omitted Stage S" if not plan["stage_s"] else "P-DISC not in Q"}
        sealed_json(root / "S.json", result)
        return result
    require(root, "C/groups-W3-r64.json")
    groups = read_json(root / "C" / "groups-W3-r64.json")
    batch = read_json(root / "A-prof-1.json")["scale"]["by_steps"]["none"]["batch"]
    p, model = cfg["pipeline"], cfg["model"]
    try:
        with charged(root, "S") as budget:
            folders, failures = train_stage(root, "S", [p], None, groups, "W3", 64, ("Main",), (0,), budget,
                                            architecture=model, updates=cfg["updates"])
            if not failures:
                folder = folders[(p, "Main", 0)]
                clp_probe(root, "S", p, folder, 0, groups, window="W3", rung=64, pairs=cfg["clp_pairs"],
                          threshold=st.Z_S, budget=budget, architecture=model)
                targets = schedule(root, "S", "W3", 64, groups, [folder], cfg["trials"])
                evaluate_streams(root, "S", 1, targets, 0, cfg["trials"], [p], None, folders, budget, window="W3",
                                 rung=64, methods=("Main", "MC"), seeds=(0,), architecture=model, batch=batch)
    except BudgetExceeded:
        pass
    failures = load_failures(root, "S")
    result = {"skipped": False, "pipeline": p, "model": model, "updates": cfg["updates"], "integrity": failures}
    rows = sealed_clp(root, "S", p, (0,), model)
    if rows:
        clp = rows[0]["probe"]
        result.update(clp=clp, SCALE_SIGNAL=bool(clp["z"] is not None and clp["z"] > st.Z_S))
    if p not in failures and sealed_blocks(root, "S", p, [1], (0,), ("Random", "MC"), architecture=model):
        row = outcomes(root, "S", [p], [1], seeds=(0,), controls=("Random", "MC"), architecture=model)[p]
        result["success_at_100"] = {k: float(v.mean()) for k, v in row.items()}
    result["partial"] = bool(not failures and (rows is None or "success_at_100" not in result))
    sealed_json(root / "S.json", result)
    return result


# ---------------------------------------------------------------- 보고서

def terminal(root):
    if (root / "failure.json").exists():
        return True
    record = optional(root, "A.json")
    if record and record["repair_state"] != "pending" and not record["Q"]:
        return True
    c = optional(root, "C.json")
    if c is None or (replication_entrants(c) and not (root / "R.json").exists()):
        return False
    return (root / "P.json").exists() and (root / "S.json").exists()


def headline_ready(root):
    """The registered headline depends only on C1, C3 and R; P and S cannot change it."""
    if terminal(root):
        return True
    c = optional(root, "C.json")
    return c is not None and (not replication_entrants(c) or (root / "R.json").exists())


def pipeline_verdicts(root):
    record, c, r = optional(root, "A.json"), optional(root, "C.json"), optional(root, "R.json")
    verdicts = {}
    for p in PIPELINES:
        c1 = record["C1"].get(p) if record else None
        if c1 == "UNTESTABLE":
            verdicts[p] = "UNTESTABLE"
        elif c1 != "PASS" or c is None or p not in c["decisions"]:
            verdicts[p] = "NOT_ESTABLISHED_BY_BUDGET"
        elif c["decisions"][p] == "POSITIVE":
            if not c["audit"].get(p, {}).get("passed"):
                verdicts[p] = "NOT_ESTABLISHED_INTEGRITY"
            else:
                verdicts[p] = (r or {}).get("decisions", {}).get(p, "NOT_ESTABLISHED_BY_BUDGET")
        else:
            verdicts[p] = c["decisions"][p]
    return verdicts


def compute_c4(root, c, verdicts):
    profile, frozen = optional(root, "A-prof-1.json"), optional(root, "protocol.frozen.json")
    if not c or not profile or not frozen or not c["final"]:
        return {}
    settings = frozen["settings"]["pipelines"]
    blocks = range(1, c["final"]["look"] + 1)
    result = {}
    for p, metrics in c["metrics"].items():
        random = [read_json(stream_folder(root, "C", "Random", s, source=source_of(p)) / f"block-{b}.json")
                  for s in SEEDS for b in blocks]
        model_cps = profile["models"][p]["by_steps"][steps_key(p, settings[p]["sampling_steps"])]["candidates_per_second"]
        result[p] = st.compute_advantage(verdicts[p], metrics["hits"], metrics["rows"], sum(x["hits"] for x in random),
                                         sum(x["rows"] for x in random), profile["md5_per_second"], model_cps)
    return result


def quality(metrics, pipeline):
    if not metrics:
        return None
    rows = metrics["rows"]
    return {"duplicates_per_trial": metrics["duplicates"] / (rows / 100), "training_match_rate": metrics["training_matches"] / rows,
            "valid_rate": metrics["valid"] / rows,
            "strict_valid_rate": metrics["strict_valid"] / rows if pipeline in GAUSSIAN else None}


def failure_list(root):
    rows = []
    study = optional(root, "failure.json")
    if study:
        rows.append(study)
    for stage in MD5_STAGES:
        rows += [{"stage": stage, "pipeline": p, "reason": reason} for p, reason in load_failures(root, stage).items()]
    return rows


def report(root):
    root = Path(root)
    if not headline_ready(root):
        decision = {"status": "INCOMPLETE", "headline": "NOT_FINAL", "progress": status(root), "blinding": BLINDING}
        atomic_json(root / "decision.json", decision)
        lines = ["# V6 최종 연구 보고서", "", "실행 상태: INCOMPLETE", "종합 판정: NOT_FINAL", "",
                 "종합 판정은 C(감사를 통과한 POSITIVE가 있으면 R까지)가 끝나야 확정된다. 그 전에는 판정과 구간을 공개하지 "
                 "않는다(블라인드). 미완료 상태를 효과 없음이나 기각으로 해석하지 않는다."]
        (root / "FINAL_REPORT_KO.md").write_text("\n".join(lines) + "\n")
        return decision
    done = terminal(root)
    pending = [] if done else [s for s in ("P", "S") if not (root / f"{s}.json").exists()]
    record, c = optional(root, "A.json") or {}, optional(root, "C.json") or {}
    p_stage, r_stage, s_stage = optional(root, "P.json") or {}, optional(root, "R.json"), optional(root, "S.json")
    verdicts = pipeline_verdicts(root)
    c4 = compute_c4(root, c, verdicts)
    final = c.get("final") or {}
    pipelines = {}
    for p in PIPELINES:
        setting = record.get("settings", {}).get(p, {})
        comparisons = final.get("pipelines", {}).get(p, {}).get("comparisons", {})
        clp = c.get("clp", {}).get(p)
        pipelines[p] = {
            "C1": record.get("C1", {}).get(p, "NOT_EVALUATED"), "updates": setting.get("updates"),
            "sampling_steps": setting.get("sampling_steps"), "batch": setting.get("batch"),
            "C2": p_stage.get("pipelines", {}).get(p), "C3": verdicts[p],
            "C3_look": final.get("look") if comparisons else None,
            "intervals": {k: {x: v[x] for x in ("estimate", "lower", "upper", "se", "trials")} for k, v in comparisons.items()},
            "seed_estimates": {k: v["seeds"] for k, v in comparisons.items()},
            "CLP_64": None if clp is None else {"z": clp["z"], "INFO_64": clp["INFO_64"], "disabled": clp.get("disabled", False)},
            "audit": c.get("audit", {}).get(p), "replication": (r_stage or {}).get("results", {}).get(p),
            "C4": c4.get(p, {"decision": "NOT_EVALUATED"}), "quality": quality(c.get("metrics", {}).get(p), p)}
    budget = optional(root, "budget.json") or {"seconds": {}}
    plan = optional(root, "budget-plan.json")
    decision = {"status": "TERMINAL" if done else "HEADLINE_FINAL", "pending": pending, "release": RELEASE,
                "headline": st.headline([verdicts[p] for p in PIPELINES]), "Q": record.get("Q", []),
                "pipelines": pipelines, "contrasts": c.get("contrasts", []),
                "stage_c": {k: c.get(k) for k in ("block", "budget_stop", "clp_partial")},
                "stage_r": r_stage, "stage_s": s_stage,
                "budget": {"hours_used": {k: v / 3600 for k, v in budget["seconds"].items()}, "caps_hours": effective_caps(root),
                           "plan": None if plan is None else {k: plan.get(k) for k in ("decision", "block", "p_trials", "stage_s")}},
                "failures": failure_list(root), "blinding": BLINDING}
    atomic_json(root / "decision.json", decision)
    (root / "FINAL_REPORT_KO.md").write_text(render_report(root, decision, record, p_stage, s_stage))
    return decision


def pp(value):
    return "—" if value is None else f"{100 * value:+.3f}%p"


def yes(value):
    return "—" if value is None else ("예" if value else "아니오")


def interval_text(v):
    return f"{pp(v['estimate'])} [{pp(v['lower'])}, {pp(v['upper'])}]"


def intervals_text(row):
    iv = row["intervals"]
    return f"Δ_R {interval_text(iv['Random'])}, Δ_S {interval_text(iv['Shuffled'])}" if iv else "측정 없음"


def aq_text(record, p):
    updates = (record.get("settings", {}).get(p) or {}).get("updates")
    rows = [r for rnd in record.get("rounds", []) for r in rnd["rows"].get(p, []) if updates in (None, r["updates"])]
    if not rows:
        return "측정 없음"
    return f"정상 {min(r['normal_joint'] for r in rows)}/512, 반전 {min(r['flipped_joint'] for r in rows)}/512"


def contrast_summary(contrasts):
    counts = {}
    for row in contrasts:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    return ", ".join(f"{k} {v}건" for k, v in sorted(counts.items())) or "비교 없음"


def conclusion(decision, record):
    """Plan §15.2 pre-written sentence for the headline, filled with the measured values."""
    rows, head = decision["pipelines"], decision["headline"]
    verdicts = {p: v["C3"] for p, v in rows.items()}
    if head == "FINAL_SUPPORTED":
        supported = [p for p, v in verdicts.items() if v == "SUPPORTED"]
        others = ", ".join(f"{p} {v}" for p, v in verdicts.items() if v != "SUPPORTED") or "없음"
        c4 = ", ".join(f"{p} {rows[p]['C4']['decision']}" for p in supported)
        return (f"고정 source, MD5 12-bit window W3, 등록된 학습량에서 {', '.join(supported)}의 해시 조건 모델은 Random과 "
                f"Shuffled 대비 Success@100 이득을 보였고({'; '.join(f'{p}: {intervals_text(rows[p])}' for p in supported)}), "
                f"이 이득은 W4에서 재현되었다. 다른 파이프라인의 판정은 {others}다. 계산 우위는 {c4}다. Full MD5 역상, "
                "보안 붕괴, 학습비 포함 계산 우위는 주장하지 않는다.")
    if head == "FINAL_REJECTED":
        bounds = "; ".join(f"{p} U_R {pp(rows[p]['intervals']['Random']['upper'])}, U_S "
                           f"{pp(rows[p]['intervals']['Shuffled']['upper'])}" for p in PIPELINES)
        gen4 = [p for p, v in rows.items() if v["C2"] and v["C2"].get("GEN_4")]
        structure = ("step-reduced MD5 r=4 양성 대조(Stage P)는 진행 중이다" if "P" in decision["pending"] else
                     f"step-reduced MD5 r=4에서는 {', '.join(gen4)}이(가) 구조를 이용했다" if gen4
                     else "step-reduced MD5 r=4에서 구조를 이용한 파이프라인은 없었다")
        return ("Gaussian BGV, Gaussian CGGE, Discrete token 표현과 Printable, Random Bytes source로 구성한 5개 파이프라인 "
                "모두에서, 해시 조건 diffusion 생성의 Success@100 이득은 사전 최소 관심 효과 0.5%p 미만으로 배제되었다"
                f"(파이프라인별 상한 {bounds}). 같은 기계들은 synthetic 조건을 "
                f"{'; '.join(f'{p} {aq_text(record, p)}' for p in PIPELINES)} 정확도로 사용했고, {structure}. "
                f"파이프라인 간 효과 차이는 {contrast_summary(decision['contrasts'])}이다. 계산 우위는 없다.")
    if head == "FINAL_REJECTED_WITH_EXCEPTIONS":
        rejected = [p for p, v in verdicts.items() if v.startswith("REJECTED_")]
        exceptions = ", ".join(f"{p}({verdicts[p]}; {intervals_text(rows[p])})" for p in verdicts if p not in rejected)
        return (f"{len(rejected)}개 파이프라인({', '.join(rejected)})에서 0.5%p 이상의 이득을 배제했다. {exceptions}은(는) "
                "판정하지 못했다. 이득이 지지된 파이프라인은 없다.")
    measured = [f"{p} U_R {pp(v['intervals']['Random']['upper'])}, U_S {pp(v['intervals']['Shuffled']['upper'])}"
                for p, v in rows.items() if v["intervals"]]
    return (f"사전 규칙으로 판정하지 못했다. 사유는 {', '.join(f'{p} {v}' for p, v in verdicts.items())}이고, 측정된 경우 "
            f"이득의 상한은 {'; '.join(measured) or '측정 없음'}이다.")


def render_report(root, decision, record, p_stage, s_stage):
    """Plan §15.3 layout."""
    rows = decision["pipelines"]
    prof1, prof2 = optional(root, "A-prof-1.json"), optional(root, "A-prof-2.json")
    pending = decision["pending"]
    state = ("TERMINAL" if not pending else
             f"HEADLINE_FINAL (종합 판정 확정. 보조 단계 {'·'.join(pending)}가 진행 중이며, 끝나면 이 보고서가 갱신된다)")
    lines = ["# V6 최종 연구 보고서", "", f"실행 상태: {state}", f"종합 판정: **{decision['headline']}**", "",
             conclusion(decision, record), "",
             "## 1. 최종 결론 카드", "", "| 파이프라인 | C1 기계 | C2 구조 이용(r=4) | C3 연구 가설 | C4 계산 우위 |",
             "|---|---|---|---|---|"]
    for p, v in rows.items():
        c2 = v["C2"]
        c2_text = ("진행 중" if "P" in pending and v["C1"] == "PASS" else "—" if not c2
                   else f"GEN_4 {yes(c2['GEN_4'])}, INFO_4 {yes(c2['INFO_4'])}")
        lines.append(f"| {p} | {v['C1']} | {c2_text} | {v['C3']} | {v['C4'].get('decision')} |")
    lines += ["", "## 2. 버전별 결산과 처리량", "",
              "버전별 결산과 V5 부분 결과(C1 PASS, C3 NOT_ESTABLISHED_BY_BUDGET)는 RESEARCH_PLAN_V6.md §1을 따른다.", ""]
    if prof1:
        lines += [f"MD5 prior 처리량(prior sampling + hashlib 단일 core): {prof1['md5_per_second']:,.0f} 메시지/초. "
                  f"검증기 처리량: {prof1['verify_rows_per_second']:,.0f} 행/초.", "",
                  "| 파이프라인 | 초/update | burst 후보/초(선택 설정) | 지속 후보/초(A-prof-2) |", "|---|---:|---:|---:|"]
        for p in PIPELINES:
            key = steps_key(p, rows[p]["sampling_steps"]) if rows[p]["batch"] else None
            burst = prof1["models"][p]["by_steps"][key]["candidates_per_second"] if key else None
            sustained_cps = (prof2 or {}).get("models", {}).get(p, {}).get("sustained_cps")
            lines.append(f"| {p} | {prof1['models'][p]['update_seconds']:.4f} | "
                         f"{'—' if burst is None else f'{burst:,.0f}'} | {'—' if sustained_cps is None else f'{sustained_cps:,.0f}'} |")
    lines += ["", "## 3. C1 기계 적격성 (A-Q)", "", f"Q = {', '.join(decision['Q']) or '없음'}. 보완 사용: "
              f"{'예' if record.get('repair_used') else '아니오'}.", "",
              "| 파이프라인 | C1 | updates | S_G | B* | seed별 정상 / 반전 joint | CLP 제외 |", "|---|---|---:|---:|---:|---|---|"]
    for p in PIPELINES:
        seeds = [f"{r['normal_joint']}/{r['flipped_joint']}" for rnd in record.get("rounds", []) for r in rnd["rows"].get(p, [])]
        lines.append(f"| {p} | {rows[p]['C1']} | {rows[p]['updates'] or '—'} | {rows[p]['sampling_steps'] or '—'} | "
                     f"{rows[p]['batch'] or '—'} | {', '.join(seeds) or '—'} | {'예' if record.get('clp_disabled', {}).get(p) else '아니오'} |")
    lines += ["", "## 4. C3 연구 가설 (Success@100, 동시 구간)", "",
              "| 파이프라인 | C3 | look | Δ_R 추정 [구간] | Δ_S 추정 [구간] | seed별 Δ_R / Δ_S | CLP_64 z | 감사 |",
              "|---|---|---:|---|---|---|---:|---|"]
    for p, v in rows.items():
        iv, seeds = v["intervals"], v["seed_estimates"]
        seed_text = ("; ".join(f"{pp(a['estimate'])} / {pp(b['estimate'])}" for a, b in zip(seeds["Random"], seeds["Shuffled"]))
                     if iv else "—")
        clp = v["CLP_64"]
        clp_text = "—" if not clp else ("제외" if clp["disabled"] else f"{clp['z']:.2f}")
        audit = v["audit"]
        audit_text = "—" if not audit else (f"{'통과' if audit['passed'] else '실패'}, 상위 1% target 비중 "
                                            f"{audit['top_1pct_target_hit_share']:.3f}, hit-only {yes(audit['hit_only'])}")
        lines.append(f"| {p} | {v['C3']} | {v['C3_look'] or '—'} | {interval_text(iv['Random']) if iv else '—'} | "
                     f"{interval_text(iv['Shuffled']) if iv else '—'} | {seed_text} | {clp_text} | {audit_text} |")
    r_stage = decision["stage_r"]
    if r_stage and r_stage["entrants"]:
        lines += ["", "Stage R(W4 재현):", ""]
        for p in r_stage["entrants"]:
            result = r_stage["results"].get(p)
            detail = ("" if not result else
                      f", Δ_R {interval_text(result['comparisons']['Random'])}, Δ_S {interval_text(result['comparisons']['Shuffled'])}")
            lines.append(f"- {p}: {r_stage['decisions'][p]}{detail}")
    lines += ["", "## 5. C2 구조 이용 (W1, r=4)", ""]
    if p_stage.get("pipelines"):
        lines += ["| 파이프라인 | GEN_4 | INFO_4 | Main − Random | Main − MC | CLP_4 z |", "|---|---|---|---|---|---:|"]
        for p, v in p_stage["pipelines"].items():
            z = v["clp"]["z"]
            lines.append(f"| {p} | {yes(v['GEN_4'])} | {'제외' if v['INFO_4_disabled'] else yes(v['INFO_4'])} | "
                         f"{interval_text(v['comparisons']['Random'])} | {interval_text(v['comparisons']['MC'])} | "
                         f"{'—' if z is None else f'{z:.2f}'} |")
    if "P" in pending:
        lines.append("Stage P는 진행 중이다.")
    elif p_stage.get("partial") or not p_stage:
        lines.append("Stage P는 예산 때문에 부분 측정되었거나 실행되지 않았다.")
    lines += ["", "## 6. C5 파이프라인 대비", "", "| 대비 | 대조 | 추정 [구간] | 분류 |", "|---|---|---|---|"]
    for row in decision["contrasts"]:
        lines.append(f"| {row['pair'][0]} − {row['pair'][1]} | {row['control']} | "
                     f"{interval_text(row) if 'estimate' in row else '—'} | {row['decision']} |")
    lines += ["", "## 7. C4 계산 우위", ""]
    measured = [(p, v["C4"]) for p, v in rows.items() if "rho" in v["C4"]]
    for p, c4 in measured:
        lines.append(f"- {p}: ρ = {c4['rho']:.1f}, 손익분기 prior {c4['break_even_prior']:.4f}, 산술 {c4['arithmetic']}, "
                     f"판정 {c4['decision']}")
    if not measured:
        lines.append("A-prof-1 처리량이나 Stage C 후보 수가 없어 계산하지 않았다.")
    lines += ["", "## 8. Stage S (D1-T-L 규모 탐침)", ""]
    if "S" in pending:
        lines.append("진행 중이다.")
    elif not s_stage:
        lines.append("실행되지 않았다.")
    elif s_stage.get("skipped"):
        lines.append(f"생략: {s_stage['reason']}.")
    else:
        clp = s_stage.get("clp")
        z = None if not clp else clp["z"]
        lines.append(f"CLP_64 z = {'—' if z is None else f'{z:.2f}'}, SCALE_SIGNAL = {yes(s_stage.get('SCALE_SIGNAL'))}, "
                     f"부분 측정 = {yes(s_stage['partial'])}.")
        if s_stage.get("success_at_100"):
            lines.append("Success@100(기술통계): " + ", ".join(f"{k} {100 * v:.3f}%" for k, v in s_stage["success_at_100"].items()))
    if decision["failures"]:
        lines += ["", "## 실패와 제외", ""] + [f"- {json.dumps(row, ensure_ascii=False, sort_keys=True)}" for row in decision["failures"]]
    lines += ["", "## 9. 적용 범위와 일반화", "",
              "실측 범위는 Printable과 Random Bytes source, MD5 12-bit window W3(재현 시 W4), 5개 파이프라인(G3-U BGV·CGGE, "
              "D1-S), 등록된 decoder·sampler·학습량, K=100, r=4 양성 대조다. q=8/16과 다른 구조에 대한 결론은 random-function "
              "논증에 따른 이론적 기대이며 측정이 아니다. Full MD5 역상, 임의 target MD5 역상, SHA-256, 보안 붕괴, 최신 "
              "암호분석 공격과의 비교, 모든 diffusion의 불가능성은 주장하지 않는다.", "",
              "V6 이후 같은 질문의 revision은 없다. 세부 수치는 `decision.json`에 있다."]
    return "\n".join(lines) + "\n"


def stream_progress(root, stage):
    folder = Path(root) / stage / "eval"
    if not folder.exists():
        return {"completed_stream_blocks": 0, "committed_rows": 0, "recent_rows_per_second": None}
    results = sorted((p for p in folder.rglob("block-*.json") if BLOCK_RESULT.fullmatch(p.name)), key=lambda p: p.stat().st_mtime)
    commits = [read_json(p)["committed_rows"] for p in folder.rglob("block-*.commit.json")]
    recent = [read_json(p) for p in results[-3:]]
    seconds = sum(r["elapsed_seconds"] for r in recent)
    return {"completed_stream_blocks": len(results), "committed_rows": int(sum(commits)),
            "recent_rows_per_second": sum(r["rows"] for r in recent) / seconds if seconds else None}


def status(root):
    """Progress only: no success rates, intervals or decisions."""
    root = Path(root)
    seconds = (optional(root, "budget.json") or {"seconds": {}})["seconds"]
    caps = effective_caps(root)
    plan = optional(root, "budget-plan.json")
    streams = {stage: stream_progress(root, stage) for stage in MD5_STAGES}
    looks = root / "C" / "looks"
    eta = None
    if plan and plan["decision"] == "PROCEED" and streams["C"]["recent_rows_per_second"]:
        q = list(plan["pipelines"])
        planned = (6 * len(q) + 3 * len({source_of(p) for p in q})) * REG["stage_c"]["looks"] * plan["block"] * 100
        eta = max(0, planned - streams["C"]["committed_rows"]) / streams["C"]["recent_rows_per_second"] / 3600
    names = ("A-impl", "A-prof-1", "A-dev", "A", "A-dev-repair", "A-prof-2", "budget-plan", "exposure-audit",
             "protocol.frozen", *MD5_STAGES, "decision")
    required = sum(seconds.get(k, 0) for k in ("A", "A_repair", "C", "P")) / 3600
    return {"protocol": PROTOCOL, "artifacts": {name: (root / f"{name}.json").exists() for name in names},
            "required_path_hours": {"used": round(required, 3),
                                    "cap": caps["required_with_repair" if seconds.get("A_repair") else "required"]},
            "budget_stops": budget_stops(root),
            "streams": streams, "c_looks_sealed": len(list(looks.glob("look-*.json"))) if looks.exists() else 0,
            "eta_hours_c_max": eta, "hours_used": {k: round(v / 3600, 3) for k, v in seconds.items()},
            "caps_hours": caps, "remaining_hours": {k: round(caps[k] - seconds.get(k, 0) / 3600, 3)
                                                    for k in ("A", "A_repair", "C", "R", "P", "S")},
            "halt": bool(plan and plan["decision"] == "HALT"), "failure": (root / "failure.json").exists()}


# ---------------------------------------------------------------- 실행 순서와 CLI

def run_all(root):
    stage_a(root, "all")
    if read_json(root / "A.json")["Q"]:
        c = stage_c(root)
        if replication_entrants(c):
            stage_r(root)
        # The headline is final once C (and R) are sealed; P and S follow as supplementary stages.
        decision = report(root)
        emit({"status": decision["status"], "headline": decision["headline"], "pending": decision["pending"]})
        stage_p(root)
        stage_s(root)
    return report(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--output", type=Path)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--draft", action="store_true", required=True)
    inventory.add_argument("--v5", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    for name in ("audit", "run", "approve-caps", "status", "report", "check"):
        commands.add_parser(name).add_argument("--root", type=Path, required=True)
    commands.choices["audit"].add_argument("--inventory", type=Path, required=True)
    commands.choices["run"].add_argument("--stage", choices=("all", "A", *MD5_STAGES), required=True)
    commands.choices["run"].add_argument("--phase", choices=("all", *A_PHASES), default="all")
    commands.choices["approve-caps"].add_argument("--stage", choices=("C",), required=True)
    commands.choices["approve-caps"].add_argument("--hours", type=float, required=True)
    commands.choices["approve-caps"].add_argument("--reason", required=True)
    commands.choices["check"].add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = registration()
        if args.output:
            sealed_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "inventory":
        draft = inventory_draft(args.v5)
        atomic_json(args.output, draft)
        emit({"output": str(args.output), "files": len(draft["reviewed_files"]),
              "unreviewed": sum(r["condition_use"] == "unreviewed" for r in draft["reviewed_files"])})
        return 0
    root = args.root.resolve()
    if root.exists() and any(root.iterdir()) and not (root / "v6-study.json").exists():
        parser.error("Choose an empty directory or an existing V6 study; other artifacts are read-only")
    if args.command == "run" and args.stage != "A" and args.phase != "all":
        parser.error("--phase only applies to Stage A")
    if args.command == "status":
        emit(status(root))
        return 0
    with study_lock(root):
        sealed_json(root / "v6-study.json", {"protocol": PROTOCOL})
        if args.command == "audit":
            if (root / "protocol.frozen.json").exists():
                raise ValueError("Audit cannot change after protocol freeze")
            result = audit_exposure(args.inventory)
            atomic_json(root / "exposure-audit.json", result)
            emit({"certified": result["certified"], "reason": result.get("reason"),
                  "excluded_counts": {w: len(v) for w, v in result.get("excluded", {}).items()}})
        elif args.command == "approve-caps":
            emit(approve_caps(root, args.stage, args.hours, args.reason))
        elif args.command == "report":
            decision = report(root)
            emit({"status": decision["status"], "headline": decision["headline"], "report": str(root / "FINAL_REPORT_KO.md")})
        elif args.command == "check":
            from .checks import implementation_gate
            result = implementation_gate(root, args.quick)
            atomic_json(root / ("quick-check.json" if args.quick else "implementation-check.json"), result)
            emit({"quick": result["quick"], "passed": result["passed"], "quick_paths_passed": result["quick_paths_passed"],
                  "certifies_study": result["certifies_study"], "gates": {k: v["passed"] for k, v in result["gates"].items()}})
        else:
            if (root / "failure.json").exists():
                raise RuntimeError("This study has a terminal failure; registered caps and rules cannot be reset")
            try:
                if args.stage == "all":
                    decision = run_all(root)
                    emit({"status": decision["status"], "headline": decision["headline"], "budget_stops": budget_stops(root)})
                elif args.stage == "A":
                    stage_a(root, args.phase)
                    emit({"stage": "A", "phase": args.phase, "completed": True})
                else:
                    {"C": stage_c, "R": stage_r, "P": stage_p, "S": stage_s}[args.stage](root)
                    emit({"stage": args.stage, "completed": True, "budget_stop": args.stage in budget_stops(root),
                          "headline_ready": headline_ready(root)})
            except Halt as error:
                emit({"halt": str(error)})
                return 3
            except BudgetExceeded as error:
                # C/R/P/S seal a partial result under their own caps; only Stage A ends the study here.
                if not (root / "failure.json").exists():
                    sealed_json(root / "failure.json", {"stage": args.stage, "reason": "NOT_ESTABLISHED_BY_BUDGET",
                                                        "error": str(error)})
                report(root)
                emit(read_json(root / "failure.json"))
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
