"""MLX training, full-attempt SQLite evaluation, recovery and hard resource limits."""
from collections import Counter
from contextlib import closing
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import sqlite3
import sys
import time

import numpy as np

from .encoding.tokens import TokenCodec
from .study_pilot import (atomic_json, canonical, digest, event, file_hash, ledger_open,
                          load_checkpoint, read_json, save_checkpoint, verify as synthetic_verify)
from .study_v4_data import METHODS, h12, prior, seed, valid_source

GIB = 1024 ** 3


class ResourceStop(RuntimeError):
    pass


class RunStop(RuntimeError):
    pass


class Pause(RuntimeError):
    """Deterministic injected interruption; never a qualification result."""


def manifest(p):
    root = Path(__file__).resolve().parents[2]
    return {"protocol_sha256": digest(p), "python": sys.version, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "mlx")},
            "sources": {str(path.relative_to(root)): file_hash(path) for path in sorted((root / "src/diffusion_hash_inv").rglob("*.py"))},
            "lock_sha256": file_hash(root / "uv.lock"), "precision": "float32", "device": "mlx/gpu"}


class Budget:
    def __init__(self, root, p, stage):
        self.root, self.p, self.stage = Path(root), p, stage
        self.path = self.root / "budget.json"
        self.state = read_json(self.path) if self.path.exists() else {"stages": {}, "runs": {}, "nfe_reserved": 0, "warnings": []}
        pending = self.state.get("running_since")
        if pending is not None:
            elapsed = max(0., time.time() - pending)
            self.add_time(self.state["running_stage"], self.state.get("running_run"), elapsed)
            self.state["warnings"].append("unclean interruption: elapsed wall time includes downtime; reserved NFE is an upper bound")
        self.last, self.run, self.checks, self.size = time.monotonic(), None, 0, 0
        self.persist()

    def add_time(self, stage, run, elapsed):
        self.state["stages"][stage] = self.state["stages"].get(stage, 0.) + elapsed
        if run:
            self.state["runs"][run] = self.state["runs"].get(run, 0.) + elapsed

    def persist(self, running=True):
        self.state.update(running_since=time.time() if running else None, running_stage=self.stage, running_run=self.run)
        atomic_json(self.path, self.state)

    def tick(self, run=None):
        now = time.monotonic()
        self.add_time(self.stage, self.run, now - self.last)
        self.last, self.run = now, str(run) if run is not None else None
        self.persist()

    def reserve(self, run, count, *, refresh=False):
        if refresh or self.checks % 100 == 0:
            self.size = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        if self.size + count > self.p["resources"]["storage_gib"] * GIB:
            raise ResourceStop("study storage cap reached")
        self.size += count

    def check(self, run=None, nfe=0):
        self.tick(run)
        self.checks += 1
        cfg = self.p["resources"]
        prep = sum(v for k, v in self.state["stages"].items() if k in {"V0", "V1", "V2"})
        main = sum(v for k, v in self.state["stages"].items() if k in {"V3", "V4", "V5"})
        if prep >= cfg["preparation_seconds"] or main >= cfg["main_seconds"] or prep + main >= cfg["total_seconds"]:
            raise ResourceStop("cumulative active-time hard cap reached")
        if run and self.state["runs"].get(str(run), 0.) >= cfg["run_seconds"]:
            raise RunStop("per-run active-time cap reached")
        if shutil.disk_usage(self.root).free < cfg["disk_free_gib"] * GIB:
            raise ResourceStop("minimum free disk reached")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if rss > cfg["rss_gib"] * GIB:
            raise ResourceStop("process RSS cap reached")
        if "mlx.core" in sys.modules:
            import mlx.core as mx
            if mx.get_active_memory() > cfg["gpu_gib"] * GIB:
                raise ResourceStop("GPU allocation cap reached")
        resources = self.root / "resource.seal.json"
        if resources.exists() and main > read_json(resources)["soft_seconds"]:
            if "soft time estimate exceeded" not in self.state["warnings"]:
                self.state["warnings"].append("soft time estimate exceeded")
        self.reserve(run, 65536)
        self.state["nfe_reserved"] += nfe
        self.persist()

    def account(self, **counts):
        for key, value in counts.items():
            self.state[key] = self.state.get(key, 0) + value
        self.persist()

    def finish(self):
        self.tick()
        self.persist(running=False)


def build_model(p, task, label):
    from . import mlx_models as models
    import mlx.core as mx
    codec = TokenCodec("printable", 31)
    mx.random.seed(seed(p, task, "initialization", label=label))
    spec = p["model"]
    model = models.SequenceDenoiser(codec.vocabulary_size, width=spec["width"], embedding_dim=spec["embedding_dim"],
                                    factorized=True, condition_output=True)
    diffusion = models.MaskedDiffusion(codec.mask, spec["sampling_steps"], factorized=True, prefix_balanced_loss=False)
    mx.eval(model.parameters())
    return model, diffusion


def validation(p, task, model, diffusion, rows, budget, directory):
    from . import mlx_backend as backend
    import mlx.core as mx
    started = time.monotonic()
    codec, total = TokenCodec("printable", 31), 0.
    model.eval()
    for start in range(0, len(rows), p["training"]["batch_size"]):
        batch = rows[start:start + p["training"]["batch_size"]]
        clean = backend.clean_batch(codec, batch, diffusion)
        for draw in range(p["training"]["validation_draws"]):
            budget.check(directory)
            keys = [mx.random.key(seed(p, task, "validation", trial=row[0], attempt=draw)) for row in batch]
            values = diffusion.losses(model, clean, backend.models.condition([row[0] for row in batch]), *diffusion.noise_inputs(clean, keys))
            if not backend.models.finite(values):
                raise FloatingPointError("non-finite validation objective")
            total += values.sum().item()
    budget.account(validation_seconds_completed=time.monotonic() - started)
    return total / (len(rows) * p["training"]["validation_draws"])


def train(p, task, method, label, data, directory, budget, *, pause_update=None):
    """Trainers open only train/validation; common noise has method=null."""
    from . import mlx_backend as backend
    import mlx.core as mx
    if method not in {"main", "shuffled"}:
        raise ValueError("only main/shuffled are learned methods")
    directory.mkdir(parents=True, exist_ok=True)
    budget.check(directory)
    rows, val = read_json(data / "train.json"), read_json(data / "validation.json")
    cfg = p[task]
    if len(rows) != cfg["train_messages"] or len(val) != cfg["validation_groups"]:
        raise ValueError("registered data quota mismatch")
    identity = {"backend": "mlx", "protocol_sha256": digest(p), "task": task, "method": method, "seed": label,
                "train_sha256": file_hash(data / "train.json"), "validation_sha256": file_hash(data / "validation.json")}
    config = directory / "configuration.json"
    if config.exists() and read_json(config) != identity:
        raise ValueError("training identity changed")
    atomic_json(config, identity)
    model, diffusion = build_model(p, task, label)
    optimizer, step = backend.optimizer_and_step(model, diffusion, p["training"])
    state = {"identity": identity, "epoch": 1, "offset": 0, "update": 0, "best_loss": None, "best_epoch": None,
             "history": [], "epoch_losses": [], "update_seconds": [], "checkpoint_seconds": []}
    codec = TokenCodec("printable", 31)

    def save(best=False):
        mx.synchronize()
        started = time.monotonic()
        state.update(model=model.parameters(), optimizer=optimizer.state)
        save_checkpoint(directory, state, best=best, budget=budget)
        elapsed = time.monotonic() - started
        state["checkpoint_seconds"].append(elapsed)
        budget.account(checkpoint_seconds_completed=elapsed)

    if (directory / "checkpoints/LATEST.json").exists():
        state, _ = load_checkpoint(directory)
        if state["identity"] != identity:
            raise ValueError("checkpoint identity changed")
        backend.load_weights(model, state["model"])
        optimizer.state = state["optimizer"]
    else:
        save()
    while state["epoch"] <= cfg["epochs"]:
        epoch = state["epoch"]
        order, donors = list(range(len(rows))), list(range(len(rows)))
        random.Random(seed(p, task, "train-order", label=label, epoch=epoch)).shuffle(order)
        if method == "shuffled":
            random.Random(seed(p, task, "shuffle", label=label, epoch=epoch)).shuffle(donors)
        same = sum(rows[i][0] == rows[donors[i]][0] for i in range(len(rows)))
        while state["offset"] < len(rows):
            started = time.monotonic()
            budget.check(directory)
            offset = state["offset"]
            indices = order[offset:offset + p["training"]["batch_size"]]
            clean = backend.clean_batch(codec, [rows[i] for i in indices], diffusion)
            cond = backend.models.condition([rows[donors[i]][0] for i in indices])
            model.train()
            loss, parts = step(clean, cond, seed(p, task, "train-corruption", label=label, epoch=epoch, trial=offset))
            mx.synchronize()
            budget.account(training_update_seconds_completed=time.monotonic() - started)
            state["offset"] += len(indices)
            state["update"] += 1
            state["epoch_losses"].append(float(loss.item()))
            event(directory, kind="training_update", update=state["update"], epoch=epoch,
                  loss=float(loss.item()), seconds=time.monotonic() - started,
                  components={k: float(v.item()) for k, v in parts.items()})
            if state["update"] == pause_update:
                save()
                raise Pause("training checkpoint boundary")
            if state["update"] % p["training"]["checkpoint_every"] == 0:
                save()
            state["update_seconds"].append(time.monotonic() - started)
        improved = False
        row = {"epoch": epoch, "updates": state["update"], "mean_training_loss": float(np.mean(state["epoch_losses"])),
               "same_condition_count": same, "order_sha256": digest(order), "donors_sha256": digest(donors)}
        if epoch % cfg["validation_every"] == 0:
            started = time.monotonic()
            value = validation(p, task, model, diffusion, val, budget, directory)
            row.update(validation_loss=value, validation_seconds=time.monotonic() - started)
            improved = state["best_loss"] is None or value < state["best_loss"]
            if improved:
                state["best_loss"], state["best_epoch"] = value, epoch
        state["history"].append(row)
        state.update(epoch=epoch + 1, offset=0, epoch_losses=[])
        save(best=improved)
    best, checksum = load_checkpoint(directory, best=True)
    backend.load_weights(model, best["model"])
    result = {"status": "COMPLETE", "updates": state["update"], "best_epoch": state["best_epoch"],
              "best_validation_loss": state["best_loss"], "checkpoint_sha256": checksum, "history": state["history"],
              "update_seconds": state["update_seconds"], "checkpoint_seconds": state["checkpoint_seconds"]}
    atomic_json(directory / "training.json", result)
    budget.check(directory)
    return model, diffusion, checksum


def load_model(p, task, label, directory):
    from .mlx_backend import load_weights
    state, checksum = load_checkpoint(directory, best=True)
    model, diffusion = build_model(p, task, label)
    load_weights(model, state["model"])
    return model, diffusion, checksum


def generated(p, task, method, label, jobs, model=None, diffusion=None):
    """Input is (trial, public target, variant, attempt), never a hidden representative."""
    payload_seeds = [seed(p, task, "payload", method=method, label=label, trial=t, attempt=a) for t, y, v, a in jobs]
    length_seeds = [seed(p, task, "length", method=method, label=label, trial=t, attempt=a) for t, y, v, a in jobs]
    if method == "random":
        # Two independent source-prior namespaces, including uniform length 4..31.
        messages = []
        for s, l in zip(payload_seeds, length_seeds):
            rng = random.Random(s)
            messages.append(bytes(rng.randint(33, 126) for _ in range(random.Random(l).randint(4, 31))))
        decoded = [(x, True, None) for x in messages]
    else:
        from . import mlx_backend as backend
        samples = backend.models.sample(model, diffusion, backend.models.condition([y for _, y, _, _ in jobs]), (32,),
                                         steps=p["model"]["sampling_steps"], seeds=payload_seeds, length_seeds=length_seeds)
        codec = TokenCodec("printable", 31)
        decoded = []
        for sample in samples:
            row = codec.decode(backend.to_codec_tensor(sample))
            decoded.append((row.message, row.valid, row.reason))
    return list(zip(decoded, payload_seeds, length_seeds))


def candidate(p, task, method, label, job, generated_row, checkpoint, config_hash):
    trial, y, variant, attempt = job
    (payload, decoded, reason), rng, length_rng = generated_row
    domain = valid_source(payload)
    prefix = h12(payload) if payload is not None and task != "V1" else None
    valid = bool(decoded and domain)
    hit = synthetic_verify(payload, "printable", y)[1] if task == "V1" else prefix == y
    return {"protocol_id": p["protocol_id"], "task": task, "method": method, "seed": label,
            "trial_id": trial, "target": y, "variant": variant, "attempt": attempt,
            "candidate_hex": payload.hex() if payload is not None else None, "decode_valid": decoded,
            "domain_valid": domain, "valid": valid, "reason": reason if not decoded else None if domain else "source_domain",
            "digest_prefix": prefix, "success": bool(valid and hit),
            "wrong_original": bool(task == "V1" and variant == "flipped" and valid and synthetic_verify(payload, "printable", y ^ 4095)[1]),
            "md5_calls": int(payload is not None and task != "V1"),
            "nfe": 0 if method == "random" else p["model"]["nfe"],
            "rng_identity": str(rng), "length_rng_identity": str(length_rng),
            "checkpoint_sha256": checkpoint, "config_sha256": config_hash}


def evaluation_identity(p, task, method, label, targets, checkpoint, batch_size):
    return {"protocol_sha256": digest(p), "task": task, "method": method, "seed": label,
            "targets_sha256": digest(targets), "checkpoint_sha256": checkpoint, "batch_size": batch_size}


def evaluate(p, task, method, label, targets, directory, checkpoint, batch_size, budget,
             model=None, diffusion=None, *, pause_batch=None):
    if method not in METHODS or batch_size not in p["resources"]["batches"] or not targets or any(type(y) is not int or not 0 <= y < 4096 for y in targets):
        raise ValueError("invalid method, frozen inference batch or public targets")
    directory.mkdir(parents=True, exist_ok=True)
    identity = evaluation_identity(p, task, method, label, targets, checkpoint, batch_size)
    path = directory / "evaluation.json"
    if path.exists() and read_json(path) != identity:
        raise ValueError("evaluation identity changed")
    atomic_json(path, identity)
    cfg, variants = p[task], ("normal", "flipped") if task == "V1" else ("normal",)
    total = len(targets) * len(variants) * cfg["k"]
    config_hash, run_id = digest(identity), f'{p["protocol_id"]}/{task}/{method}/{label}'
    with closing(ledger_open(directory / "candidates.sqlite")) as db:
        count = db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        if count > total or (count != total and count % batch_size):
            raise ValueError("ledger is not a complete prefix of committed batches")
        # Verify the entire existing prefix before any replay; missing rows are never zero outcomes.
        if count:
            verify_ledger(p, task, method, label, targets, directory, checkpoint, batch_size, complete=False, budget=budget)
        for start in range(count, total, batch_size):
            started = time.monotonic()
            jobs = []
            for offset in range(start, min(start + batch_size, total)):
                trial, remainder = divmod(offset, len(variants) * cfg["k"])
                variant, attempt0 = divmod(remainder, cfg["k"])
                target = targets[trial] ^ (4095 if variants[variant] == "flipped" else 0)
                jobs.append((trial, target, variants[variant], attempt0 + 1))
            budget.check(directory, nfe=len(jobs) * (p["model"]["nfe"] if method != "random" else 0))
            try:
                rows = [candidate(p, task, method, label, job, output, checkpoint, config_hash)
                        for job, output in zip(jobs, generated(p, task, method, label, jobs, model, diffusion))]
            except BaseException:
                budget.account(generation_seconds_completed=time.monotonic() - started)
                event(directory, kind="generation_uncommitted", rows=0, seconds=time.monotonic() - started)
                raise
            budget.account(sampling_nfe_completed=sum(r["nfe"] for r in rows), md5_calls_completed=sum(r["md5_calls"] for r in rows))
            if pause_batch == start // batch_size:
                budget.account(generation_seconds_completed=time.monotonic() - started)
                event(directory, kind="generation_uncommitted", rows=len(rows), seconds=time.monotonic() - started)
                raise Pause("candidate batch before commit")
            with db:
                db.executemany("INSERT INTO candidates VALUES (?,?,?,?,?)",
                               [(run_id, str(row["trial_id"]), row["variant"], row["attempt"], canonical(row).decode()) for row in rows])
            event(directory, kind="generation_batch", rows=len(rows), seconds=time.monotonic() - started)
            budget.account(generation_seconds_completed=time.monotonic() - started)
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    result = verify_ledger(p, task, method, label, targets, directory, checkpoint, batch_size, budget=budget)
    atomic_json(directory / "metrics.json", result)
    budget.check(directory)
    return result


def verify_ledger(p, task, method, label, targets, directory, checkpoint, batch_size, *, complete=True, training_hex=(), budget=None):
    """Stream independent rehash and trial counts, bounding memory by one candidate batch."""
    import json
    cfg = p[task]
    variants = ("normal", "flipped") if task == "V1" else ("normal",)
    identity = evaluation_identity(p, task, method, label, targets, checkpoint, batch_size)
    if read_json(directory / "evaluation.json") != identity:
        raise ValueError("ledger configuration mismatch")
    path = directory / "candidates.sqlite"
    if not path.is_file():
        raise ValueError("missing candidate ledger")
    outcomes = {variant: [False] * len(targets) for variant in variants}
    prefixes = {str(k): [False] * len(targets) for k in (1, 10, 100) if k <= cfg["k"]}
    counters, lengths, first_hits = Counter(), Counter(), Counter()
    training_hex, trial_seen = set(training_hex), set()
    last_trial = None
    started = time.monotonic()
    with closing(sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)) as db:
        for index, (run_id, unit, variant, attempt, record) in enumerate(db.execute("SELECT * FROM candidates ORDER BY rowid")):
            if budget and index % 10000 == 0:
                budget.check(directory)
            trial, remainder = divmod(index, len(variants) * cfg["k"])
            v, a = divmod(remainder, cfg["k"])
            if trial >= len(targets):
                raise ValueError("extra candidate rows")
            y = targets[trial] ^ (4095 if variants[v] == "flipped" else 0)
            if (run_id, unit, variant, attempt) != (f'{p["protocol_id"]}/{task}/{method}/{label}', str(trial), variants[v], a + 1):
                raise ValueError("ledger gap/duplicate/order mismatch")
            row = json.loads(record)
            payload = bytes.fromhex(row["candidate_hex"]) if row["candidate_hex"] is not None else None
            # A strict token decoder returns either a valid payload or no bytes.
            if row["decode_valid"] != (payload is not None) or (payload is None and not row["reason"]):
                raise ValueError("invalid decoder evidence")
            output = ((payload, row["decode_valid"], row["reason"]),
                      seed(p, task, "payload", method=method, label=label, trial=trial, attempt=a+1),
                      seed(p, task, "length", method=method, label=label, trial=trial, attempt=a+1))
            expected = candidate(p, task, method, label, (trial, y, variant, a+1), output, checkpoint, digest(identity))
            if canonical(row) != canonical(expected):
                raise ValueError("independent candidate verifier mismatch")
            if last_trial != (trial, variant):
                trial_seen.clear()
                last_trial = (trial, variant)
            if payload is not None:
                counters["duplicate_within_trial"] += int(payload in trial_seen)
                trial_seen.add(payload)
                counters["training_message_matches"] += int(row["candidate_hex"] in training_hex)
                lengths[len(payload)] += 1
            counters.update(rows=1, valid=int(row["valid"]), hits=int(row["success"]),
                            md5_calls=row["md5_calls"], nfe=row["nfe"], wrong_original=int(row["wrong_original"]))
            counters[variant + "_valid"] += int(row["valid"])
            if row["success"]:
                if not outcomes[variant][trial]:
                    first_hits[attempt] += 1
                outcomes[variant][trial] = True
                if variant == "normal":
                    for k in prefixes:
                        if attempt <= int(k):
                            prefixes[k][trial] = True
        # SQLite performs the distinct aggregation without an unbounded Python payload set.
        duplicates = db.execute("SELECT COUNT(*) - COUNT(DISTINCT json_extract(record, '$.candidate_hex')) FROM candidates WHERE json_extract(record, '$.candidate_hex') IS NOT NULL").fetchone()[0]
        expected_count = len(targets) * len(variants) * cfg["k"]
        if complete and counters["rows"] != expected_count:
            raise ValueError("incomplete candidate ledger")
    if budget:
        budget.account(verifier_md5_calls_completed=counters["md5_calls"], verifier_seconds_completed=time.monotonic() - started)
    n = counters["rows"]
    # Incomplete trials remain null even if their observed prefix already hit.
    for v, variant in enumerate(variants):
        for trial in range(len(targets)):
            if (trial * len(variants) + v + 1) * cfg["k"] > n:
                outcomes[variant][trial] = None
    for k, values in prefixes.items():
        for trial in range(len(targets)):
            if trial * len(variants) * cfg["k"] + int(k) > n:
                values[trial] = None
    return {"status": "COMPLETE" if n == expected_count else "PARTIAL", **dict(counters),
            "outcomes": outcomes, "success_at_k": {k: sum(v) / len(targets) if all(x is not None for x in v) else None for k, v in prefixes.items()},
            "valid_rate": counters["valid"] / n if n else None,
            "duplicate_payloads_stream": duplicates, "duplicate_rate_stream": duplicates / n if n else None,
            "duplicate_rate_within_trial": counters["duplicate_within_trial"] / n if n else None,
            "hash_hit_given_valid": counters["hits"] / counters["valid"] if counters["valid"] else None,
            "length_counts": dict(lengths), "first_hit_counts": dict(first_hits),
            "verifier_md5_calls": counters["md5_calls"], "verification_seconds": time.monotonic() - started,
            "trials": len(targets), "unique_targets": len(set(targets)), "batch_size": batch_size}


def paired_counts(results, seeds=(0, 1, 2)):
    counts = {}
    for label in seeds:
        for control in ("random", "shuffled"):
            a, b = results.get(f"{label}/main"), results.get(f"{label}/{control}")
            if a is None or b is None or a["status"] != "COMPLETE" or b["status"] != "COMPLETE":
                continue
            av, bv = a["outcomes"]["normal"], b["outcomes"]["normal"]
            if len(av) != len(bv):
                raise ValueError("paired trials differ in length")
            counts[f"{label}/{control}"] = {f"n{i}{j}": sum(x == bool(i) and y == bool(j) for x, y in zip(av, bv))
                                           for i in (0, 1) for j in (0, 1)}
    return counts
