"""Atomic MLX checkpoints, streaming ledgers and independent payload verification."""
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from . import PROTOCOL
from .data import (decode, derangement, encode, fresh_batch, hash_one, identity,
                   key_words, messages, prior_candidates, rng, shuffle_permutation, valid)
from .protocol import atomic_json, canonical, file_hash, read_json, sealed_json
from .statistics import probe


def connect(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    return db


def charge_work(folder, **counts):
    """Persist work before a model call; an interrupted batch is charged in full."""
    path = Path(folder)/"work.json"
    totals = read_json(path) if path.exists() else {}
    for name, count in counts.items():
        totals[name] = totals.get(name, 0) + int(count)
    atomic_json(path, totals)
    return totals


def save_checkpoint(folder, model, optimizer, update):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    folder = Path(folder)
    name = f"checkpoint-{update:08d}.safetensors"
    arrays = {"model." + key: value for key, value in tree_flatten(model.parameters())}
    arrays.update({"optimizer." + key: mx.array(value) for key, value in tree_flatten(optimizer.state)})
    temporary = folder / "checkpoint.tmp.safetensors"
    mx.save_safetensors(str(temporary), arrays)
    with temporary.open("rb") as stream:
        import os
        os.fsync(stream.fileno())
    temporary.replace(folder / name)
    old = read_json(folder / "checkpoint.json") if (folder / "checkpoint.json").exists() else None
    atomic_json(folder / "checkpoint.json", {"file": name, "sha256": file_hash(folder / name), "update": update})
    if old and old["file"] != name:
        (folder / old["file"]).unlink()


def load_checkpoint(folder, model, optimizer=None):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    folder = Path(folder)
    pointer = read_json(folder / "checkpoint.json")
    if file_hash(folder / pointer["file"]) != pointer["sha256"]:
        raise ValueError("Checkpoint digest mismatch")
    weights = mx.load(str(folder / pointer["file"]))
    model.load_weights([(key[6:], value) for key, value in weights.items() if key.startswith("model.")], strict=True)
    if optimizer is not None:
        optimizer.state = tree_unflatten([(key[10:], value) for key, value in weights.items() if key.startswith("optimizer.")])
    mx.eval(model.parameters())
    return pointer["update"]


def train(folder, architecture, namespace, groups, *, updates=40000, lr=.001,
          task="md5", window="W1", rung=64, shuffled=False, budget=None,
          batch_size=256, checkpoint_every=4000, diagnostic_pairs=256):
    import mlx.optimizers as optim
    from .models import candidate_keys, make_model, train_step
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    contract = {"protocol": PROTOCOL, "architecture": architecture, "namespace": list(namespace), "updates": updates,
                "lr": lr, "task": task, "window": window, "rung": rung, "shuffled": shuffled,
                "batch_size": batch_size, "groups": groups, "checkpoint_every": checkpoint_every,
                "diagnostic_pairs": diagnostic_pairs}
    sealed_json(folder / "contract.json", contract)
    model = make_model(architecture, namespace)
    optimizer = optim.Adam(learning_rate=lr, betas=[.9, .999], eps=1e-8, bias_correction=True)
    if (folder / "complete.json").exists():
        complete = read_json(folder / "complete.json")
        if complete["training_sha256"] != file_hash(folder / "training.sqlite") or complete["checkpoint"] != read_json(folder / "checkpoint.json"):
            raise ValueError("Modified training artifact")
        if load_checkpoint(folder, model) != updates:
            raise ValueError("Only final update checkpoints may be evaluated")
        return model
    if (folder / "attempt.json").exists():
        attempt = read_json(folder / "attempt.json")
        if attempt["retries"] >= 1:
            raise RuntimeError("Registered single exact retry exhausted")
        attempt["retries"] += 1
    else:
        attempt = {"retries": 0}
    atomic_json(folder / "attempt.json", attempt)
    start = load_checkpoint(folder, model, optimizer) if (folder / "checkpoint.json").exists() else 0
    db = connect(folder / "training.sqlite")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS hashes (digest BLOB PRIMARY KEY, first_update INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS batches (update_id INTEGER PRIMARY KEY, loss REAL, md5_calls INTEGER, permutation BLOB, data_sha256 TEXT);
        CREATE TABLE IF NOT EXISTS diagnostics (update_id INTEGER PRIMARY KEY, validation_loss REAL, clp_json TEXT);
    """)
    with db:
        db.execute("DELETE FROM hashes WHERE first_update > ?", (start,))
        db.execute("DELETE FROM batches WHERE update_id > ?", (start,))
        db.execute("DELETE FROM diagnostics WHERE update_id > ?", (start,))
    try:
        for update in range(start, updates):
            if budget:
                budget.check()
            payload, lengths, labels, calls = fresh_batch(namespace, update, batch_size, groups["train"], task=task, window=window, rung=rung)
            charge_work(folder, md5_calls=calls, updates=1, training_pairs=batch_size)
            permutation = shuffle_permutation(namespace, update, batch_size) if shuffled else np.arange(batch_size)
            tokens = encode(payload, lengths)
            keys = candidate_keys((*namespace, "train-corruption", update), np.arange(batch_size))
            loss = train_step(model, optimizer, tokens, lengths, labels[permutation], keys, lr, update)
            with db:
                db.executemany("INSERT OR IGNORE INTO hashes VALUES (?,?)",
                               ((hashlib.sha256(message).digest(), update+1) for message in messages(payload, lengths)))
                db.execute("INSERT INTO batches VALUES (?,?,?,?,?)", (update+1, loss, calls, permutation.astype("<u2").tobytes(), hashlib.sha256(tokens.tobytes()+labels.tobytes()).hexdigest()))
            if (update+1) % checkpoint_every == 0 or update+1 == updates:
                diagnostic = likelihood_probe(model, (*namespace, "validation"), groups["validation"], pairs=diagnostic_pairs,
                                              task=task, window=window, rung=rung, budget=budget)
                with db:
                    db.execute("INSERT INTO diagnostics VALUES (?,?,?)", (update+1, diagnostic["validation_loss"], json.dumps(diagnostic)))
                save_checkpoint(folder, model, optimizer, update+1)
        calls = db.execute("SELECT coalesce(sum(md5_calls),0) FROM batches").fetchone()[0]
        hashes = db.execute("SELECT count(*) FROM hashes").fetchone()[0]
    finally:
        db.close()
    sealed_json(folder / "complete.json", {"updates": updates, "checkpoint": read_json(folder / "checkpoint.json"),
                "training_sha256": file_hash(folder / "training.sqlite"), "md5_calls": read_json(folder/"work.json")["md5_calls"],
                "committed_md5_calls": calls, "training_hashes": hashes, "work": read_json(folder/"work.json")})
    return model


def likelihood_probe(model, namespace, groups, *, pairs, task="md5", window="W1", rung=64, batch_size=256, budget=None):
    from .models import candidate_keys, clp_pairs, score
    differences, losses = [], []
    for offset in range(0, pairs, batch_size):
        if budget:
            budget.check()
        n = min(batch_size, pairs-offset)
        payload, lengths, labels, _ = fresh_batch((*namespace, "clp"), offset, 2*n, groups, task=task, window=window, rung=rung)
        keys = candidate_keys((*namespace, "clp-corruption"), np.arange(2*offset, 2*(offset+n)))
        tokens = encode(payload, lengths)
        differences.append(clp_pairs(model, tokens, lengths, labels, keys))
        losses.append(-score(model, tokens, lengths, labels, keys))
    return {**probe(np.concatenate(differences)), "validation_loss": float(np.concatenate(losses).mean())}


class Ledger:
    def __init__(self, path, metadata, targets, k=100):
        self.path, self.metadata, self.k = Path(path), metadata, k
        self.targets = np.asarray(targets, dtype=np.int32)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = connect(self.path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1), json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS candidates (
                trial INTEGER NOT NULL, attempt INTEGER NOT NULL, payload_hex TEXT,
                hit INTEGER NOT NULL CHECK(hit IN (0,1)), rng BLOB NOT NULL,
                PRIMARY KEY (trial,attempt)) WITHOUT ROWID;
        """)
        contract = canonical({**metadata, "k": k, "targets": self.targets.tolist()}).decode()
        old = self.db.execute("SELECT json FROM metadata WHERE id=1").fetchone()
        if old and old[0] != contract:
            raise ValueError("Ledger identity/targets changed")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES (1,?)", (contract,))
        self.position = self.count()

    def count(self):
        count = self.db.execute("SELECT count(*) FROM candidates").fetchone()[0]
        if count:
            last = self.db.execute("SELECT trial,attempt FROM candidates ORDER BY trial DESC,attempt DESC LIMIT 1").fetchone()
            if last[0]*self.k + last[1] + 1 != count:
                raise ValueError("Ledger has missing attempts")
        return count

    def append(self, offset, payloads, keys):
        if offset != self.position:
            raise ValueError("Only contiguous, exactly-once candidate commits allowed")
        if offset + len(payloads) > len(self.targets) * self.k or len(keys) != len(payloads):
            raise ValueError("Candidate budget exceeded")
        rows = []
        for index, (payload, key) in enumerate(zip(payloads, keys, strict=True), offset):
            trial, attempt = divmod(index, self.k)
            hit = valid(payload) and hash_one(payload, self.metadata["rung"], self.metadata["window"], self.metadata["task"]) == self.targets[trial]
            rows.append((trial, attempt, payload.hex() if payload is not None else None, int(hit), np.asarray(key, dtype="<u4").tobytes()))
        with self.db:
            self.db.executemany("INSERT INTO candidates VALUES (?,?,?,?,?)", rows)
        self.position += len(rows)

    def verify(self, training_db=None, budget=None):
        expected = len(self.targets)*self.k
        if self.count() != expected:
            raise ValueError("Incomplete ledger cannot enter statistical analysis")
        outcomes = np.zeros(len(self.targets), dtype=np.int8)
        at1, at10 = np.zeros_like(outcomes), np.zeros_like(outcomes)
        valid_count = hits = train_matches = 0
        lengths = {str(i): 0 for i in range(4, 32)}
        by_target = {}
        training = sqlite3.connect(f"file:{Path(training_db).resolve()}?mode=ro", uri=True) if training_db else None
        try:
            cursor = self.db.execute("SELECT trial,attempt,payload_hex,hit,rng FROM candidates ORDER BY trial,attempt")
            position = 0
            while rows := cursor.fetchmany(4096):
                if budget:
                    budget.check()
                expected_keys = key_words(self.metadata["rng_namespace"], np.arange(position, position+len(rows)))
                for row, key in zip(rows, expected_keys, strict=True):
                    trial, attempt, hex_payload, recorded, recorded_key = row
                    if (trial, attempt) != divmod(position, self.k) or recorded_key != key.astype("<u4").tobytes():
                        raise ValueError("Ledger trial/attempt/RNG identity mismatch")
                    payload = bytes.fromhex(hex_payload) if hex_payload is not None else None
                    is_valid = valid(payload)
                    hit = bool(is_valid and hash_one(payload, self.metadata["rung"], self.metadata["window"], self.metadata["task"]) == self.targets[trial])
                    if recorded != int(hit):
                        raise ValueError("Independent rehash disagrees with ledger")
                    valid_count += is_valid
                    hits += hit
                    if is_valid:
                        lengths[str(len(payload))] += 1
                    match = bool(training and payload is not None and training.execute("SELECT 1 FROM hashes WHERE digest=?", (hashlib.sha256(payload).digest(),)).fetchone())
                    train_matches += match
                    if hit and match:
                        raise ValueError("Successful payload overlaps training messages")
                    outcomes[trial] |= hit
                    at1[trial] |= hit and attempt < 1
                    at10[trial] |= hit and attempt < 10
                    if hit:
                        target = str(int(self.targets[trial]))
                        by_target[target] = by_target.get(target, 0) + 1
                    position += 1
        finally:
            if training:
                training.close()
        unique = self.db.execute("SELECT count(*) FROM (SELECT payload_hex FROM candidates WHERE payload_hex IS NOT NULL GROUP BY payload_hex)").fetchone()[0]
        top = max(1, int(np.ceil(len(set(self.targets)) * .01)))
        return outcomes, {"rows": expected, "hits": hits, "success_at_100": int(outcomes.sum()), "success_at_1": int(at1.sum()),
                "success_at_10": int(at10.sum()), "valid": valid_count, "duplicate_count": expected-unique-self.db.execute("SELECT count(*) FROM candidates WHERE payload_hex IS NULL").fetchone()[0],
                "training_matches": train_matches, "lengths": lengths, "hit_given_valid": hits/valid_count if valid_count else None,
                "top_1pct_target_hit_share": sum(sorted(by_target.values(), reverse=True)[:top])/hits if hits else 0.0,
                "nfe": 0 if self.metadata["method"] == "Random" else expected*33,
                "hash_family_calls": 2*valid_count if self.metadata["task"] == "md5" else 0,
                "md5_calls": 2*valid_count if self.metadata["task"] == "md5" and self.metadata["rung"] == 64 else 0,
                "independent_verified": True, "successful_training_matches": 0}

    def close(self):
        self.db.close()


def evaluate(folder, model, namespace, targets, *, method, seed_id, task="md5", window="W1", rung=64,
             batch_size=1024, k=100, training_db=None, checkpoint=None, budget=None):
    from .models import candidate_keys, sample
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    meta = {"protocol": PROTOCOL, "task": task, "window": window, "rung": rung, "method": method,
            "seed": seed_id, "rng_namespace": list(namespace), "checkpoint": checkpoint}
    targets = np.asarray(targets, dtype=np.int32)
    donors = derangement(namespace, len(targets)) if method == "MC" else np.arange(len(targets))
    sealed_json(folder / "conditions.json", {"targets": targets.tolist(), "donors": donors.tolist() if method == "MC" else None})
    ledger = Ledger(folder / "candidates.sqlite", meta, targets, k)
    try:
        if result_path.exists():
            result = read_json(result_path)
            if result["ledger_sha256"] != file_hash(ledger.path):
                raise ValueError("Sealed candidate ledger changed")
            return np.asarray(result["outcomes"], dtype=np.int8), result
        started = time.monotonic()
        if (folder / "attempt.json").exists():
            retry = read_json(folder / "attempt.json")["retries"] + 1
        else:
            retry = 0
        if retry > 1:
            raise RuntimeError("Candidate stream retry exhausted")
        atomic_json(folder / "attempt.json", {"retries": retry})
        for offset in range(ledger.count(), len(targets)*k, batch_size):
            if budget:
                budget.check()
            indices = np.arange(offset, min(offset+batch_size, len(targets)*k))
            keys = key_words(namespace, indices)
            charge_work(folder, generated_attempts=len(indices), nfe=0 if method == "Random" else len(indices)*33)
            if method == "Random":
                payload, lengths = prior_candidates(namespace, indices)
                candidates = messages(payload, lengths)
            else:
                labels = targets[donors[indices // k]]
                candidates = decode(sample(model, labels, candidate_keys(namespace, indices)))
            ledger.append(offset, candidates, keys)
        outcomes, metrics = ledger.verify(training_db, budget)
    finally:
        ledger.close()
    result = {**metrics, "outcomes": outcomes.tolist(), "elapsed_seconds": time.monotonic()-started,
              "ledger_sha256": file_hash(folder / "candidates.sqlite"), "metadata": meta,
              "charged_work_including_replay": read_json(folder/"work.json")}
    sealed_json(result_path, result)
    return outcomes, result


def trial_schedule(path, namespace, groups, checkpoints, trials):
    if not checkpoints or any(not (Path(p)/"complete.json").exists() for p in checkpoints):
        raise ValueError("All learned checkpoints must be sealed before drawing trials")
    seals = {str(Path(p).resolve()): file_hash(Path(p)/"complete.json") for p in checkpoints}
    targets = rng("trials", *namespace).choice(groups, trials).astype(int).tolist()
    sealed_json(path, {"targets": targets, "checkpoints": seals, "rng_identity": identity("trials", *namespace)})
    return np.asarray(targets, dtype=np.int32)
