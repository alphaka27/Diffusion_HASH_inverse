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
