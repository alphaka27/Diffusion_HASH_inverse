"""학습·평가 저장소. checkpoint와 작업량 계상은 dhi_v5/runtime.py에서 복사했다."""
import hashlib
import os
from pathlib import Path
import time

import numpy as np

from . import PROTOCOL, codecs, data
from .protocol import atomic_json, canonical, file_hash, read_json, registration, sealed_json



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


def atomic_array(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        if path.suffix == ".npy":
            np.save(stream, arrays["value"], allow_pickle=False)
        else:
            np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def message_digests(messages):
    return np.frombuffer(b"".join(hashlib.sha256(m).digest()[:16] for m in messages), dtype="V16").copy()


def encoded_batch(payload, lengths, representation, src):
    return (data.encode(payload, lengths, src) if representation == "tokens"
            else codecs.encode(payload, lengths, representation, src))


def likelihood_probe(model, stage, purpose, pipeline, seed_id, groups, *, pairs,
                     task="md5", window="W1", rung=64, batch_size=256, budget=None):
    import mlx.core as mx
    from .models import candidate_keys, clp_pairs, score
    differences, losses = [], []
    for offset in range(0, pairs, batch_size):
        if budget:
            budget.check()
        n = min(batch_size, pairs - offset)
        payload, lengths, labels, _ = data.fresh_batch(("clp", stage, purpose, model.src, seed_id),
            offset, 2 * n, groups, task=task, window=window, rung=rung)
        keys = candidate_keys(("clp-corruption", stage, purpose, pipeline, seed_id), np.arange(offset, offset + n))
        clean = encoded_batch(payload, lengths, model.representation, model.src)
        differences.append(clp_pairs(model, clean, lengths, labels, keys))
        losses.append(-score(model, clean, lengths, labels, mx.repeat(keys, 2, axis=0)))
    return np.concatenate(differences), float(np.concatenate(losses).mean())


def verify_training(folder):
    folder = Path(folder)
    complete = read_json(folder / "complete.json")
    if complete["checkpoint"] != read_json(folder / "checkpoint.json"):
        raise ValueError("Training checkpoint pointer changed")
    if file_hash(folder / complete["checkpoint"]["file"]) != complete["checkpoint"]["sha256"]:
        raise ValueError("Training checkpoint digest mismatch")
    read_json(folder / "contract.json")
    for name, digest in complete["segments"].items():
        if file_hash(folder / name) != digest:
            raise ValueError("Training segment changed")
    for name, digest in complete["diagnostics"].items():
        if file_hash(folder / name) != digest:
            raise ValueError("Training diagnostic changed")
    for name, digest in complete["digest_segments"].items():
        if file_hash(name) != digest:
            raise ValueError("Shared training digest segment changed")
    return complete


def train(folder, pipeline, stage, seed_id, groups, *, updates=40000, method="Main",
          task="md5", window="W1", rung=64, architecture=None, batch_size=256,
          checkpoint_every=4000, diagnostic_pairs=256, stream_root=None,
          resume_from=None, budget=None):
    import mlx.optimizers as optim
    from .models import candidate_keys, make_model, train_step
    if method not in ("Main", "Shuffled") or updates < 1 or checkpoint_every < 1:
        raise ValueError("Invalid training contract")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    spec = registration()["pipelines"][pipeline]
    architecture = architecture or spec["model"]
    setting = registration()["models"][architecture]
    src, representation = spec["source"], spec["representation"]
    namespace = (stage, src, seed_id)
    namespaces = {"fresh": ["fresh", *namespace, task, window, rung], "shuffle": ["shuffle", *namespace],
                  "weights": ["weights", stage, pipeline, seed_id],
                  "corruption": ["train-corruption", stage, pipeline, seed_id],
                  "validation": ["clp", stage, "validation", src, seed_id],
                  "clp_corruption": ["clp-corruption", stage, "validation", pipeline, seed_id]}
    parent = verify_training(resume_from) if resume_from is not None else None
    resume = ({"path": str(Path(resume_from).resolve()), "checkpoint_sha256": parent["checkpoint"]["sha256"],
               "update": parent["updates"]} if parent else None)
    contract = {"protocol": PROTOCOL, "pipeline": pipeline, "model": architecture, "stage": stage,
                "task": task, "window": window, "rung": rung, "method": method, "seed_id": seed_id,
                "updates": updates, "lr": setting["lr"], "warmup": setting["warmup"], "grad_clip": setting["grad_clip"],
                "batch": batch_size, "namespaces": namespaces, "groups_sha256": hashlib.sha256(canonical(groups)).hexdigest(),
                "checkpoint_every": checkpoint_every, "diagnostic_pairs": diagnostic_pairs, "resume_from": resume}
    if parent:
        previous = read_json(Path(resume_from) / "contract.json")
        for key in contract.keys() - {"updates", "resume_from"}:
            if previous[key] != contract[key]:
                raise ValueError(f"Continuation contract mismatch: {key}")
        if parent["updates"] >= updates:
            raise ValueError("Continuation must advance update count")
    sealed_json(folder / "contract.json", contract)
    model = make_model(pipeline, stage, seed_id, architecture)
    optimizer = optim.Adam(learning_rate=setting["lr"], betas=[.9, .999], eps=1e-8, bias_correction=True)
    if (folder / "complete.json").exists():
        verify_training(folder)
        if load_checkpoint(folder, model) != updates:
            raise ValueError("Only final checkpoints may be evaluated")
        return model
    attempt_path = folder / "attempt.json"
    retries = read_json(attempt_path)["retries"] + 1 if attempt_path.exists() else 0
    if retries > 1:
        raise RuntimeError("Registered single exact retry exhausted")
    atomic_json(attempt_path, {"retries": retries})
    if (folder / "checkpoint.json").exists():
        start = load_checkpoint(folder, model, optimizer)
    elif parent:
        start = load_checkpoint(resume_from, model, optimizer)
    else:
        start = 0
    for directory, pattern in (("segments", "seg-*.npz"), ("diagnostics", "diag-*.json")):
        for path in (folder / directory).glob(pattern):
            if int(path.stem.split("-")[-1]) > start:
                path.unlink()
                path.with_name(path.name + ".sha256").unlink(missing_ok=True)
    if task == "md5" and stream_root is None:
        raise ValueError("MD5 training requires the shared stage stream directory")
    digest_root = Path(stream_root) / f"{src}-seed{seed_id}" if stream_root is not None else None
    rows, digests = [], []
    digest_manifest = dict(parent["digest_segments"]) if parent else {}
    for update in range(start, updates):
        if budget:
            budget.check()
        payload, lengths, labels, calls = data.fresh_batch(namespace, update, batch_size, groups["train"],
                                                          task=task, window=window, rung=rung)
        permutation = data.shuffle_permutation(namespace, update, batch_size) if method == "Shuffled" else np.arange(batch_size)
        clean = encoded_batch(payload, lengths, representation, src)
        keys = candidate_keys(("train-corruption", stage, pipeline, seed_id, update), np.arange(batch_size))
        loss = train_step(model, optimizer, clean, lengths, labels[permutation], keys, setting["lr"], update)
        charge_work(folder, updates=1, training_pairs=batch_size, md5_calls=calls)
        digest = hashlib.sha256(payload.tobytes() + lengths.tobytes() + labels.tobytes()).digest()
        perm_digest = hashlib.sha256(permutation.astype("<u4").tobytes()).digest()
        rows.append((update + 1, loss, calls, digest, perm_digest))
        if task == "md5":
            digests.append(message_digests(data.messages(payload, lengths)))
        if (update + 1) % checkpoint_every == 0 or update + 1 == updates:
            values, validation_loss = likelihood_probe(model, stage, "validation", pipeline, seed_id,
                groups["validation"], pairs=diagnostic_pairs, task=task, window=window, rung=rung, budget=budget)
            end = update + 1
            atomic_json(folder / "diagnostics" / f"diag-{end:08d}.json",
                        {"update": end, "validation_loss": validation_loss, "clp_differences": values.tolist()})
            arrays = {"update_id": np.array([r[0] for r in rows], dtype="<u4"),
                      "loss": np.array([r[1] for r in rows], dtype="<f4"),
                      "md5_calls": np.array([r[2] for r in rows], dtype="<u4"),
                      "data_sha256": np.frombuffer(b"".join(r[3] for r in rows), dtype=np.uint8).reshape(-1, 32),
                      "perm_sha256": np.frombuffer(b"".join(r[4] for r in rows), dtype=np.uint8).reshape(-1, 32)}
            atomic_array(folder / "segments" / f"seg-{end:08d}.npz", **arrays)
            if task == "md5":
                path = digest_root / f"digests-{end:08d}.npy"
                expected = np.sort(np.concatenate(digests))
                if path.exists():
                    if not np.array_equal(np.load(path, allow_pickle=False), expected):
                        raise ValueError("Shared training digest mismatch")
                else:
                    atomic_array(path, value=expected)
                digest_manifest[str(path.resolve())] = file_hash(path)
            save_checkpoint(folder, model, optimizer, end)
            rows, digests = [], []
    # Earlier committed segments also belong to a resumed run's digest manifest.
    if task == "md5":
        for segment in sorted((folder / "segments").glob("seg-*.npz")):
            end = int(segment.stem.split("-")[-1])
            path = digest_root / f"digests-{end:08d}.npy"
            digest_manifest[str(path.resolve())] = file_hash(path)
    segment_manifest = {str(p.relative_to(folder)): file_hash(p) for p in sorted((folder / "segments").glob("*.npz"))}
    diagnostics = {str(p.relative_to(folder)): file_hash(p) for p in sorted((folder / "diagnostics").glob("*.json"))}
    sealed_json(folder / "complete.json", {"updates": updates, "checkpoint": read_json(folder / "checkpoint.json"),
        "segments": segment_manifest, "segment_manifest_sha256": hashlib.sha256(canonical(segment_manifest)).hexdigest(),
        "diagnostics": diagnostics, "digest_segments": digest_manifest, "work": read_json(folder / "work.json"),
        "md5_calls": read_json(folder / "work.json").get("md5_calls", 0)})
    return model


RECORD = np.dtype([("payload", "u1", (31,)), ("length", "u1"), ("flags", "u1"),
                   ("margin", "<f2"), ("reserved", "u1")], align=False)
TRIAL = np.dtype([("at1", "u1"), ("at10", "u1"), ("at100", "u1"), ("hits", "u1"),
                  ("first", "u1"), ("duplicates", "u1"), ("training_matches", "u1")])


def training_index(folders):
    paths = {}
    for folder in folders:
        paths.update(verify_training(folder)["digest_segments"])
    return np.unique(np.concatenate([np.load(p, allow_pickle=False) for p in paths])) if paths else np.array([], dtype="V16")


def lookup_digests(digests, training):
    if not len(training):
        return np.zeros(len(digests), dtype=bool)
    positions = np.searchsorted(training, digests)
    return (positions < len(training)) & (training[np.minimum(positions, len(training) - 1)] == digests)


def candidate_records(messages, targets, metadata, training, margins=None, strict=None):
    records = np.zeros(len(messages), dtype=RECORD)
    records["margin"] = np.nan if margins is None else margins
    valid = np.array([data.valid(m, metadata["source"]) for m in messages])
    for i, message in enumerate(messages):
        if message is not None and len(message) <= 31:
            records["length"][i] = len(message)
            records["payload"][i, :len(message)] = np.frombuffer(message, dtype=np.uint8)
    hit = np.zeros(len(messages), dtype=bool)
    if metadata["task"] == "md5":
        hit[valid] = data.hash_batch(records["payload"][valid], records["length"][valid],
                                     metadata["rung"], metadata["window"]) == np.asarray(targets)[valid]
    else:
        hit[valid] = np.array([data.synthetic_label(m, metadata["source"]) for m, ok in zip(messages, valid) if ok]) == np.asarray(targets)[valid]
    digests = message_digests([m if m is not None else b"" for m in messages])
    match = lookup_digests(digests, training) & valid
    strict = np.zeros(len(messages), dtype=bool) if strict is None else np.asarray(strict, dtype=bool)
    records["flags"] = valid.astype(np.uint8) | (hit.astype(np.uint8) << 1) | (match.astype(np.uint8) << 2) | (strict.astype(np.uint8) << 3)
    return records


class Ledger:
    def __init__(self, folder, block, metadata, targets, *, batch, k=100, start=0, training=None):
        self.folder, self.block, self.metadata = Path(folder), block, metadata
        self.targets, self.batch, self.k, self.start = np.asarray(targets), batch, k, start
        data.condition_bits(self.targets)
        if batch <= 0 or k < 1 or k > 100 or start < 0 or start % k:
            raise ValueError("Invalid ledger boundary")
        self.training = np.array([], dtype="V16") if training is None else training
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / f"block-{block}.bin"
        self.commit_path = self.folder / f"block-{block}.commit.json"
        self.contract = {"metadata": metadata, "batch": batch, "k": k, "start": start,
                         "targets_sha256": hashlib.sha256(self.targets.astype("<u2").tobytes()).hexdigest(),
                         "rows": len(targets) * k, "training_sha256": hashlib.sha256(self.training.tobytes()).hexdigest()}
        sealed_json(self.folder / f"block-{block}.contract.json", self.contract)
        committed = read_json(self.commit_path) if self.commit_path.exists() else {"committed_rows": 0}
        if self.commit_path.exists() and (committed.get("generator") != metadata or committed.get("batch") != batch
                or committed.get("contract_sha256") != file_hash(self.folder / f"block-{block}.contract.json")):
            raise ValueError("Ledger commit contract changed")
        self.position = committed["committed_rows"]
        if self.position < 0 or self.position % batch or self.position > len(targets) * k:
            raise ValueError("Invalid committed row boundary")
        if self.position and (not self.path.exists() or self.path.stat().st_size < self.position * RECORD.itemsize):
            raise ValueError("Committed ledger bytes missing")
        self.stream = self.path.open("r+b" if self.path.exists() else "w+b")
        self.stream.truncate(self.position * RECORD.itemsize)
        self.stream.seek(0, 2)
        self.last_commit, self.pending_batches = time.monotonic(), 0

    def append(self, offset, messages, margins=None, strict=None):
        if offset != self.position or len(messages) != self.batch or offset + len(messages) > len(self.targets) * self.k:
            raise ValueError("Append must be a contiguous complete batch")
        targets = self.targets[np.arange(offset, offset + len(messages)) // self.k]
        records = candidate_records(messages, targets, self.metadata, self.training, margins, strict)
        self.stream.write(records.tobytes())
        self.position += len(records)
        self.pending_batches += 1
        if self.pending_batches >= 8 or time.monotonic() - self.last_commit >= 30:
            self.commit()

    def commit(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        atomic_json(self.commit_path, {"committed_rows": self.position, "batch": self.batch,
                                      "generator": self.metadata, "contract_sha256": file_hash(self.folder / f"block-{self.block}.contract.json")})
        self.last_commit, self.pending_batches = time.monotonic(), 0

    def close(self):
        self.stream.close()


def verify_ledger(path, targets, metadata, *, training=None, k=100, budget=None):
    targets = np.asarray(targets)
    training = np.array([], dtype="V16") if training is None else training
    expected = len(targets) * k
    if Path(path).stat().st_size != expected * RECORD.itemsize:
        raise ValueError("Incomplete ledger cannot enter analysis")
    records = np.memmap(path, dtype=RECORD, mode="r", shape=(expected,))
    summaries = np.zeros(len(targets), dtype=TRIAL)
    total_valid = total_hits = total_matches = total_strict = 0
    entropy_counts = np.zeros((31, 256), dtype=np.int64)
    hit_targets = {}
    for trial_start in range(0, len(targets), 1024):
        if budget:
            budget.check()
        subset = records[trial_start * k:min(trial_start + 1024, len(targets)) * k]
        lengths = subset["length"].astype(int)
        if np.any(lengths > 31) or np.any(subset["reserved"]) or np.any(subset["flags"] & 240):
            raise ValueError("Invalid ledger layout/reserved bits")
        if np.any(subset["payload"][np.arange(31) >= lengths[:, None]]):
            raise ValueError("Nonzero payload padding")
        messages = [row[:n].tobytes() for row, n in zip(subset["payload"], lengths)]
        spec = data.SOURCES[metadata["source"]]
        valid = np.fromiter((4 <= len(m) <= 31 and all(spec["byte_min"] <= b <= spec["byte_max"] for b in m)
                             for m in messages), dtype=bool, count=len(messages))
        shift = data.WINDOWS[metadata["window"]]
        if metadata["task"] == "md5":
            # Independent scalar path: no digest_batch/hash_batch calls.
            hashes = np.fromiter(((int.from_bytes(hashlib.md5(m).digest() if metadata["rung"] == 64
                                  else data.digest_reference(m, metadata["rung"]), "big") >> shift) & 4095
                                  for m in messages), dtype=np.uint16, count=len(messages))
        else:
            hashes = np.fromiter((data.synthetic_label(m, metadata["source"]) for m in messages), dtype=np.int32)
        hit = valid & (hashes == np.repeat(targets[trial_start:trial_start + len(subset) // k], k))
        digests = np.frombuffer(b"".join(hashlib.sha256(m).digest()[:16] for m in messages), dtype="V16")
        match = lookup_digests(digests, training) & valid
        flags = valid.astype(np.uint8) | (hit.astype(np.uint8) << 1) | (match.astype(np.uint8) << 2)
        if not np.array_equal(subset["flags"] & 7, flags):
            raise ValueError("Independent rehash/valid/training flags disagree")
        if np.any(hit & match):
            raise ValueError("Successful payload overlaps training messages")
        gaussian = metadata.get("representation") in ("bgv", "cgge") and metadata["method"] != "Random"
        if not gaussian and (np.any(subset["flags"] & 8) or not np.isnan(subset["margin"]).all()):
            raise ValueError("Non-Gaussian diagnostic fields changed")
        if gaussian and not np.isfinite(subset["margin"]).all():
            raise ValueError("Non-finite Gaussian margin")
        for j in range(len(subset) // k):
            a, b = j * k, (j + 1) * k
            hits = np.flatnonzero(hit[a:b])
            duplicate = k - len(set((int(lengths[i]), messages[i]) for i in range(a, b)))
            summaries[trial_start + j] = (int(hit[a]), int(hit[a:a + min(10, k)].any()), int(bool(len(hits))),
                                          len(hits), int(hits[0]) if len(hits) else 255, duplicate, int(match[a:b].sum()))
            if len(hits):
                target = str(int(targets[trial_start + j]))
                hit_targets[target] = hit_targets.get(target, 0) + len(hits)
        for position in range(31):
            selected = valid & (lengths > position)
            entropy_counts[position] += np.bincount(subset["payload"][selected, position], minlength=256)
        total_valid += int(valid.sum())
        total_hits += int(hit.sum())
        total_matches += int(match.sum())
        total_strict += int(((subset["flags"] & 8) != 0).sum())
    probabilities = entropy_counts / np.maximum(entropy_counts.sum(axis=1, keepdims=True), 1)
    entropy = -(probabilities * np.log2(np.maximum(probabilities, np.finfo(float).tiny))).sum(axis=1)
    top = max(1, int(np.ceil(len(set(targets.tolist())) * .01)))
    metrics = {"rows": expected, "hits": total_hits, "valid": total_valid,
        "success_at_1": int(summaries["at1"].sum()), "success_at_10": int(summaries["at10"].sum()),
        "success_at_100": int(summaries["at100"].sum()), "duplicates": int(summaries["duplicates"].sum()),
        "training_matches": total_matches, "successful_training_matches": 0, "strict_valid": total_strict,
        "position_entropy": entropy.tolist(), "independent_verified": True,
        "strict_recomputed": False, "margin_recomputed": False,
        "top_1pct_target_hit_share": sum(sorted(hit_targets.values(), reverse=True)[:top]) / total_hits if total_hits else 0.}
    return summaries, metrics


def regeneration_audit(path, stream_id, block, start, count, generator, *, batch=64, budget=None):
    records = np.memmap(path, dtype=RECORD, mode="r")
    selected = []
    for offset in range(0, count, 65536):
        indices = np.arange(start + offset, start + min(offset + 65536, count), dtype=np.uint64)
        mask = data.key_words(("regen-audit", stream_id, block), indices)[:, 1] % 100 == 0
        selected.extend(indices[mask].tolist())
    for offset in range(0, len(selected), batch):
        if budget:
            budget.check()
        chosen = np.asarray(selected[offset:offset + batch], dtype=np.uint64)
        padded = np.pad(chosen, (0, batch - len(chosen)), mode="edge")
        messages, _, _ = generator(padded)
        for index, message in zip(chosen, messages):
            row = records[int(index) - start]
            expected = row["payload"][:row["length"]].tobytes()
            if message is None:
                message = b""
            if message != expected or len(message) != row["length"]:
                raise ValueError("Regeneration audit payload mismatch")
    return {"selected": len(selected), "batch": batch, "passed": True}


def trial_schedule(path, stage, window, rung, groups, checkpoints, trials):
    if not checkpoints:
        raise ValueError("All learned checkpoints must be sealed before drawing trials")
    seals = {}
    for folder in checkpoints:
        verify_training(folder)
        seals[str(Path(folder).resolve())] = file_hash(Path(folder) / "complete.json")
    targets = data.rng("trials", stage, window, rung).choice(groups, trials).astype(int).tolist()
    sealed_json(path, {"targets": targets, "checkpoints": seals,
                       "rng_identity": data.identity("trials", stage, window, rung)})
    return np.asarray(targets, dtype=np.int32)


def evaluate_block(folder, block, targets, *, stage, source, method, seed_id, pipeline=None, model=None,
                   window="W1", rung=64, task="md5", start_trial=0, trials=None,
                   batch=256, steps=25, training=None, checkpoint=None, budget=None, k=100):
    folder = Path(folder)
    targets = np.asarray(targets, dtype=np.int32)
    trials = len(targets) - start_trial if trials is None else trials
    rows, start = trials * k, start_trial * k
    if rows % batch or trials < 1 or start_trial + trials > len(targets):
        raise ValueError("Block must contain full generation batches")
    namespace = (stage, source if method == "Random" else pipeline, window, rung, method, seed_id)
    if method not in ("Main", "Shuffled", "Random", "MC") or (method != "Random" and model is None):
        raise ValueError("Invalid generator")
    representation = model.representation if model is not None and method != "Random" else None
    nfe = 0 if method == "Random" else (steps + 1 if representation != "tokens" else 33)
    metadata = {"protocol": PROTOCOL, "stage": stage, "source": source, "pipeline": pipeline,
                "method": method, "seed_id": seed_id, "window": window, "rung": rung, "task": task,
                "namespace": list(namespace), "checkpoint": checkpoint, "representation": representation, "nfe": nfe}
    donors = data.derangement((stage, pipeline, window, rung, seed_id), len(targets)) if method == "MC" else np.arange(len(targets))
    def generate(indices):
        if method == "Random":
            payload, lengths = data.prior_candidates(namespace, indices, source)
            return data.messages(payload, lengths), np.full(len(indices), np.nan), np.zeros(len(indices), dtype=bool)
        from .models import candidate_keys, sample
        labels = targets[donors[(indices // k).astype(int)]]
        return sample(model, labels, candidate_keys(namespace, indices), steps)
    result_path = folder / f"block-{block}.json"
    if result_path.exists():
        result = read_json(result_path)
        contract = read_json(folder / f"block-{block}.contract.json")
        training_bytes = b"" if training is None else training.tobytes()
        if (contract["targets_sha256"] != hashlib.sha256(targets[start_trial:start_trial + trials].astype("<u2").tobytes()).hexdigest()
                or contract["training_sha256"] != hashlib.sha256(training_bytes).hexdigest() or contract["k"] != k):
            raise ValueError("Completed block targets/training changed")
        if result["metadata"] != metadata or result["batch"] != batch or result["start"] != start or result["rows"] != rows:
            raise ValueError("Completed block contract changed")
        for name, digest in result["artifacts"].items():
            if file_hash(folder / name) != digest:
                raise ValueError("Sealed block artifact changed")
        return np.load(folder / f"block-{block}.trials.npy", allow_pickle=False), result
    ledger = Ledger(folder, block, metadata, targets[start_trial:start_trial + trials], batch=batch, k=k, start=start, training=training)
    retry_path = folder / f"block-{block}.attempt.json"
    retry = read_json(retry_path)["retries"] + 1 if retry_path.exists() else 0
    if retry > 1:
        ledger.close()
        raise RuntimeError("Candidate stream retry exhausted")
    atomic_json(retry_path, {"retries": retry})
    started = time.monotonic()
    try:
        for offset in range(ledger.position, rows, batch):
            if budget:
                budget.check()
            charge_work(folder, generated_attempts=batch, nfe=batch * nfe)
            indices = np.arange(start + offset, start + offset + batch, dtype=np.uint64)
            messages, margins, strict = generate(indices)
            ledger.append(offset, messages, margins, strict)
        ledger.commit()
    finally:
        ledger.close()
    summary, metrics = verify_ledger(ledger.path, targets[start_trial:start_trial + trials], metadata,
                                     training=training, k=k, budget=budget)
    # Regenerate at the generation batch: D1 kernels differ between batch 64/256 and 1,024 in about 2e-6 of rows.
    audit = regeneration_audit(ledger.path, list(namespace), block, start, rows, generate, batch=batch, budget=budget)
    summary_path = folder / f"block-{block}.trials.npy"
    atomic_array(summary_path, value=summary)
    result = {**metrics, "metadata": metadata, "batch": batch, "start": start, "nfe": nfe,
              "elapsed_seconds": time.monotonic() - started, "regeneration": audit,
              "work": read_json(folder / "work.json"),
              "artifacts": {p.name: file_hash(p) for p in (ledger.path, summary_path, ledger.commit_path,
                                                         folder / f"block-{block}.contract.json")}}
    sealed_json(result_path, result)
    return summary, result
