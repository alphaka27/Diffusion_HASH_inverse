"""V2 tensor-only training/generation path and a sealed synthetic development pilot.

No primary hash corpus is loaded. The smaller pilot cannot certify G1-B.
Models, diffusion equations, codecs, Wilson intervals and atomic IO are reused.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch

from .automation import freeze_json, source_version
from .devices import resolve_device, synchronize
from .discrete import MaskedDiffusion, SequenceDenoiser
from .evaluation import binomial_ci95
from .experiment_state import sha256
from .models import GaussianDiffusion, ImageUNet, parameter_count
from .poc import save_torch, write
from .runner import _codec


def seed(spec, namespace, *, method="", target="", position="", shared_source=False):
    fields = (spec["protocol_id"], spec["master_seed"], namespace, spec["source"],
              "" if shared_source else spec["pipeline"], method,
              "" if shared_source else spec["model_seed"], target, position)
    return int.from_bytes(hashlib.sha256(":".join(map(str, fields)).encode()).digest()[:8], "big")


def condition_bits(targets, device):
    """Only public 12-bit integers cross this boundary; no DigestRecord is accepted."""
    if not targets or any(type(y) is not int or not 0 <= y < 4096 for y in targets):
        raise ValueError("conditions must be nonempty public 12-bit integers")
    return torch.tensor([[(y >> bit) & 1 for bit in range(11, -1, -1)] for y in targets],
                        dtype=torch.float32, device=device)


def check_conditions(value):
    if value.ndim != 2 or value.shape[1] != 12 or not torch.isfinite(value).all() or not ((value == 0) | (value == 1)).all():
        raise ValueError("model input must contain exactly 12 binary condition bits")


def components(spec, device):
    # Repeated-token embedding gradients on MPS otherwise differ at the last bits.
    torch.use_deterministic_algorithms(True)
    representation = "tokens" if spec["pipeline"].endswith("DISC") else "cgge" if spec["pipeline"].endswith("CGGE") else "bgv"
    encoder, decoder, shape = _codec(representation, source=spec["source"])
    torch.manual_seed(seed(spec, "initialization"))  # Matched Main/Shuffled initialization.
    if representation == "tokens":
        cfg = spec["discrete"]
        model = SequenceDenoiser(encoder.vocabulary_size, 32, 12, width=cfg["width"], embedding_dim=cfg["embedding_dim"])
        steps = cfg["sampling_steps"]
        diffusion = MaskedDiffusion(encoder.mask, [i / steps for i in range(steps + 1)], device=device)
    else:
        cfg = spec["gaussian"]
        model = ImageUNet(2, 12, width=cfg["width"])
        diffusion = GaussianDiffusion(cfg["diffusion_timesteps"], beta_start=cfg["beta_start"],
                                      beta_end=cfg["beta_end"], prediction_type=cfg["prediction"], device=device)
    return model.to(device), diffusion, encoder, decoder, shape


def synthetic_prefix(y, source):
    values = [(y >> bit) & 15 for bit in (8, 4, 0)]
    return bytes(b"0123456789abcdef"[v] for v in values) if source == "printable" else bytes(values)


def synthetic_verifier(message, y, source):
    """Independent payload/domain check; no MD5 and no generator feedback."""
    valid = (isinstance(message, bytes) and 4 <= len(message) <= 31
             and (source == "random_bytes" or all(33 <= x <= 126 for x in message)))
    return bool(valid), bool(valid and message[:3] == synthetic_prefix(y, source))


def synthetic_dataset(spec):
    pairs = list(range(2048))
    random.Random(seed(spec, "positive-control-partition", shared_source=True)).shuffle(pairs)
    expand = lambda group: sorted(y for a in group for y in (a, a ^ 4095))
    train_ids = expand(pairs[:1536])
    val_ids = expand(pairs[1536:1536 + spec["validation_conditions"] // 2])
    test_ids = expand(pairs[1792:1792 + spec["test_conditions"] // 2])
    assert not (set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids))
    rng = random.Random(seed(spec, "positive-control-corpus", shared_source=True))

    def message(y):
        length = rng.randrange(4, 32)
        rest = (bytes(rng.randrange(33, 127) for _ in range(length - 3)) if spec["source"] == "printable"
                else bytes(rng.randrange(256) for _ in range(length - 3)))
        return synthetic_prefix(y, spec["source"]) + rest

    train, seen = [], set()
    for _ in range(1_000_000):
        y = rng.choice(train_ids)
        value = message(y)
        if value not in seen:
            seen.add(value)
            train.append({"condition": y, "message_hex": value.hex()})
        if len(train) == spec["train_messages"]:
            break
    if len(train) != spec["train_messages"]:
        raise RuntimeError("synthetic construction draw cap reached")
    validation = [{"condition": y, "message_hex": message(y).hex()} for y in val_ids]
    assert not (seen & {bytes.fromhex(r["message_hex"]) for r in validation})
    return {"task": "three_nibble_payload_constraints_v1", "train": train,
            "validation": validation, "test_conditions": test_ids,
            "train_condition_pool": train_ids, "primary_hash_data_used": False}


def encode_rows(rows, encoder, device):
    clean = torch.stack([encoder.encode(bytes.fromhex(r["message_hex"])) for r in rows]).to(device)
    if clean.dtype != torch.long:
        clean = clean * 2 - 1
    return clean, condition_bits([r["condition"] for r in rows], device)


def epoch_pairing(conditions, spec, epoch):
    """Called only by training; validation and generation always use true conditions."""
    generator = torch.Generator().manual_seed(seed(spec, "train-order", position=epoch))
    order = torch.randperm(len(conditions), generator=generator)
    donors = torch.arange(len(conditions))
    if spec["method"] == "shuffled":
        generator.manual_seed(seed(spec, "shuffle", method="shuffled", position=epoch))
        donors = torch.randperm(len(conditions), generator=generator)
    paired = conditions[donors.to(conditions.device)]
    info = {"epoch": epoch + 1, "donor_sha256": hashlib.sha256(donors.numpy().tobytes()).hexdigest(),
            "same_condition_fraction": float((paired == conditions).all(1).float().mean().item())}
    return order, paired, info


@torch.no_grad()
def validation_loss(model, diffusion, clean, conditions, spec):
    model.eval()
    total = 0.
    batch = spec["training"]["batch_size"]
    for draw in range(4):
        generator = torch.Generator(device=clean.device).manual_seed(seed(spec, "validation-noise", position=draw))
        for start in range(0, len(clean), batch):
            loss = diffusion.loss(model, clean[start:start + batch], conditions[start:start + batch], generator=generator)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite validation loss")
            total += loss.item() * len(clean[start:start + batch])
    return total / (4 * len(clean))


def train(model, diffusion, clean, conditions, validation, spec, output):
    """Epoch training with recoverable batch cursor, local RNG and validation-only selection."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    check_conditions(conditions)
    check_conditions(validation[1])
    if len(clean) != len(conditions) or len(validation[0]) != len(validation[1]):
        raise ValueError("data and condition counts differ")
    cfg = spec["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["betas"]),
                                 eps=cfg["eps"], weight_decay=cfg["weight_decay"])
    checkpoint = output / "training_resume.pt"
    identity_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    for value in (clean, conditions, *validation):
        identity_hash.update(str((value.dtype, tuple(value.shape))).encode())
        identity_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    identity = identity_hash.hexdigest()
    state = dict(epoch=0, offset=0, updates=0, epoch_loss=0., best_loss=None, best_epoch=None,
                 best_model=None, history=[], pairing=[], seconds=0.)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=clean.device, weights_only=True)
        if saved["identity"] != identity:
            raise RuntimeError("checkpoint identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
    elapsed = state["seconds"]
    synchronize(clean.device)
    started = perf_counter()

    def save(generator):
        synchronize(clean.device)
        state["seconds"] = elapsed + perf_counter() - started
        state["noise_rng"] = generator.get_state()
        save_torch(checkpoint, dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(), state=state))
        write(output / "progress.json", {k: state[k] for k in ("epoch", "offset", "updates", "best_epoch", "best_loss", "seconds")})

    for epoch in range(state["epoch"], cfg["epochs"]):
        order, paired, pairing = epoch_pairing(conditions, spec, epoch)
        if len(state["pairing"]) <= epoch:
            state["pairing"].append(pairing)
        generator = torch.Generator(device=clean.device).manual_seed(seed(spec, "train-noise", position=epoch))
        if state["offset"]:
            generator.set_state(state["noise_rng"].cpu())
        model.train()
        for start in range(state["offset"], len(clean), cfg["batch_size"]):
            indices = order[start:start + cfg["batch_size"]].to(clean.device)
            loss = diffusion.loss(model, clean[indices], paired[indices], generator=generator)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            state["updates"] += 1
            state["offset"] = start + len(indices)
            state["epoch_loss"] += loss.item() * len(indices)
            if state["updates"] % spec["checkpoint_every_updates"] == 0:
                save(generator)
        row = {"epoch": epoch + 1, "training_loss": state["epoch_loss"] / len(clean)}
        if (epoch + 1) % cfg["validation_every_epochs"] == 0:
            value = validation_loss(model, diffusion, *validation, spec)
            row["validation_loss"] = value
            if state["best_loss"] is None or value < state["best_loss"]:
                state.update(best_loss=value, best_epoch=epoch + 1,
                             best_model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        state["history"].append(row)
        state.update(epoch=epoch + 1, offset=0, epoch_loss=0.)
        save(generator)
        print(json.dumps({**row, "updates": state["updates"], "seconds": state["seconds"]}), flush=True)
    if state["best_model"] is None:
        raise RuntimeError("no scheduled validation checkpoint exists")
    selected = output / "selected.pt"
    if not selected.exists():
        save_torch(selected, dict(model=state["best_model"], epoch=state["best_epoch"], validation_loss=state["best_loss"], identity=identity))
    selected_state = torch.load(selected, map_location=clean.device, weights_only=True)
    if selected_state["identity"] != identity or selected_state["epoch"] != state["best_epoch"]:
        raise RuntimeError("selected checkpoint mismatch")
    model.load_state_dict(selected_state["model"])
    result = {k: state[k] for k in ("updates", "best_epoch", "best_loss", "history", "pairing", "seconds")}
    result.update(checkpoint_sha256=sha256(selected), parameter_count=parameter_count(model))
    freeze_json(output / "selection.json", result)
    return result


@torch.no_grad()
def generate(model, diffusion, conditions, shape, spec, *, generator):
    """Model-facing API accepts only bits, public configuration and independent RNG."""
    check_conditions(conditions)
    model.eval()
    discrete = spec["pipeline"].endswith("DISC")
    cfg = spec["discrete"] if discrete else spec["gaussian"]
    options = {"temperature": cfg["temperature"]} if discrete else {}
    return diffusion.sample(model, conditions, shape, sampling_steps=cfg["sampling_steps"], generator=generator, **options)


def candidate_stream(model, diffusion, decoder, shape, spec, output, targets, *, variant, k):
    """Append each attempt transactionally; replay an uncommitted attempt with its fixed seed."""
    if variant not in {"normal", "flipped"} or not 1 <= k <= 100 or targets != sorted(set(targets)):
        raise ValueError("invalid candidate stream")
    device = next(model.parameters()).device
    database = Path(output) / "candidates.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS attempts (variant TEXT, target INTEGER, position INTEGER, input_target INTEGER, rng_seed TEXT, candidate_hex TEXT, decoder_valid INTEGER, reason TEXT, valid INTEGER, correct INTEGER, original_match INTEGER, seconds REAL, raw BLOB, PRIMARY KEY(variant,target,position))")
        for rank, target in enumerate(targets):
            for position in range(1, k + 1):
                if db.execute("SELECT 1 FROM attempts WHERE variant=? AND target=? AND position=?", (variant, target, position)).fetchone():
                    continue
                input_target = target if variant == "normal" else target ^ 4095
                rng_seed = seed(spec, "generation", method=spec["method"], target=target, position=position)
                generator = torch.Generator(device=device).manual_seed(rng_seed)  # Same initial RNG for normal/flip.
                synchronize(device)
                started = perf_counter()
                value = generate(model, diffusion, condition_bits([input_target], device), shape, spec, generator=generator)[0].cpu()
                decoded = decoder.decode(value if spec["pipeline"].endswith("DISC") else (value + 1) / 2)
                valid, correct = synthetic_verifier(decoded.message, input_target, spec["source"])
                _, original = synthetic_verifier(decoded.message, target, spec["source"])
                valid = bool(decoded.valid and valid)
                raw = value.numpy().tobytes() if variant == "normal" and rank < 16 and position == 1 else None
                db.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    variant, target, position, input_target, str(rng_seed),
                    None if decoded.message is None else decoded.message.hex(), int(decoded.valid), decoded.reason,
                    int(valid), int(valid and correct), int(valid and original), perf_counter() - started, raw))
                db.commit()  # A completed row and its optional raw tensor are one durable unit.
            if (rank + 1) % 16 == 0 or rank + 1 == len(targets):
                print(json.dumps({"generation": variant, "targets_done": rank + 1, "k": k}), flush=True)


def summarize(output, spec, targets, training):
    with sqlite3.connect(Path(output) / "candidates.sqlite") as db:
        db.row_factory = sqlite3.Row
        # ponytail: bounded pilot (<=51,712 rows); stream aggregation for a future primary orchestrator.
        rows = [dict(r) for r in db.execute("SELECT variant,target,position,input_target,rng_seed,candidate_hex,decoder_valid,reason,valid,correct,original_match,seconds FROM attempts ORDER BY variant,target,position")]
    for row in rows:
        expected_input = row["target"] if row["variant"] == "normal" else row["target"] ^ 4095
        message = None if row["candidate_hex"] is None else bytes.fromhex(row["candidate_hex"])
        valid, correct = synthetic_verifier(message, expected_input, spec["source"])
        _, original = synthetic_verifier(message, row["target"], spec["source"])
        valid = bool(valid and row["decoder_valid"])
        expected_seed = seed(spec, "generation", method=spec["method"], target=row["target"], position=row["position"])
        if (row["input_target"] != expected_input or row["rng_seed"] != str(expected_seed)
                or (row["valid"], row["correct"], row["original_match"]) != (valid, valid and correct, valid and original)):
            raise RuntimeError("candidate ledger verifier mismatch")
    n = len(targets)
    scores = {}
    for name, variant, field in (("normal", "normal", "correct"), ("flipped", "flipped", "correct"), ("flipped_matches_original", "flipped", "original_match")):
        subset = [r for r in rows if r["variant"] == variant and r["position"] == 1]
        if {r["target"] for r in subset} != set(targets) or len(subset) != n:
            raise RuntimeError("incomplete positive-control attempts")
        successes = sum(r[field] for r in subset)
        scores[name] = dict(successes=successes, trials=n, rate=successes / n, wilson95=binomial_ci95(successes, n),
                            valid_rate=sum(r["valid"] for r in subset) / n)
    prefix_targets = targets[:spec["prefix_diagnostic_targets"]]
    curves = {}
    for k in (1, 10, 100):
        if k > spec["prefix_diagnostic_k"]:
            continue
        outcomes = []
        for target in prefix_targets:
            subset = [r for r in rows if r["variant"] == "normal" and r["target"] == target and r["position"] <= k]
            if len(subset) != k or {r["position"] for r in subset} != set(range(1, k + 1)):
                raise RuntimeError("candidate prefix budget mismatch")
            outcomes.append(any(r["correct"] for r in subset))
        curves[str(k)] = dict(successes=sum(outcomes), targets=len(outcomes), rate=sum(outcomes) / len(outcomes))
    expected = 2 * n + len(prefix_targets) * (spec["prefix_diagnostic_k"] - 1)
    if len(rows) != expected:
        raise RuntimeError("candidate count mismatch")
    diagnostic_pass = (scores["normal"]["wilson95"][0] >= .9 and scores["flipped"]["wilson95"][0] >= .9
                       and scores["flipped_matches_original"]["wilson95"][1] <= .05)
    result = dict(status="COMPLETED", scope=spec["scope"], pipeline=spec["pipeline"], model_seed=spec["model_seed"],
                  device=spec["device"], training=training, scores=scores, candidate_count=len(rows),
                  prefix_diagnostic=curves, decoder_reasons=dict(Counter(r["reason"] or "valid" for r in rows)),
                  generation_decode_seconds=sum(r["seconds"] for r in rows),
                  pilot_thresholds="PASS" if diagnostic_pass else "FAIL", g1_b="NOT_CERTIFIED_REDUCED_PILOT",
                  primary_hash_evaluation="NOT_RUN")
    write(Path(output) / "report.json", result)
    # The durable source is SQLite; this sorted export is rebuilt safely after interruption.
    path = Path(output) / "candidates.jsonl"
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    temporary.replace(path)
    return result


def run(protocol, profile, output):
    if protocol["q"] != 12 or protocol["condition_dim"] != 12:
        raise ValueError("this path requires the v2 12-bit protocol")
    if profile["pipeline"] not in protocol["pipelines"] or profile["method"] not in {"main", "shuffled"}:
        raise ValueError("unknown pipeline or method")
    if (profile["scope"] != "development_synthetic_pilot_not_g1_certification" or profile["model_seed"] not in (0, 1, 2)
            or not 1 <= profile["train_messages"] <= 10000 or not 1 <= profile["epochs"] <= 100
            or profile["epochs"] % protocol["training"]["validation_every_epochs"]
            or any(profile[k] < 2 or profile[k] > 512 or profile[k] % 2 for k in ("validation_conditions", "test_conditions"))
            or not 1 <= profile["prefix_diagnostic_targets"] <= profile["test_conditions"]
            or not 1 <= profile["prefix_diagnostic_k"] <= 100 or profile["checkpoint_every_updates"] < 1):
        raise ValueError("invalid development pilot sizes or schedule")
    device = resolve_device(profile["device"])
    spec = {**profile, "device": str(device), "protocol_id": protocol["protocol_id"],
            "deterministic_algorithms": True,
            "master_seed": protocol["dataset"]["engineering_seed"],
            "source": "printable" if profile["pipeline"].startswith("P-") else "random_bytes",
            "gaussian": protocol["gaussian"], "discrete": protocol["discrete"],
            "training": {**protocol["training"], "epochs": profile["epochs"]}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_json(output / "run.json", {"spec": spec, "protocol": protocol, "code_sha256": source_version(),
                                           "torch": torch.__version__, "cpu_threads": torch.get_num_threads()})
        if (output / "complete.json").exists():
            checksums = json.loads((output / "complete.json").read_text())
            if any(sha256(output / name) != value for name, value in checksums.items()):
                raise RuntimeError("completed pilot artifact checksum mismatch")
            return json.loads((output / "report.json").read_text())
        dataset = synthetic_dataset(spec)
        freeze_json(output / "synthetic_data.json", dataset)
        model, diffusion, encoder, decoder, shape = components(spec, device)
        clean, conditions = encode_rows(dataset["train"], encoder, device)
        validation = encode_rows(dataset["validation"], encoder, device)
        training = train(model, diffusion, clean, conditions, validation, spec, output)
        del clean, conditions, validation
        targets = dataset["test_conditions"]
        candidate_stream(model, diffusion, decoder, shape, spec, output, targets, variant="normal", k=1)
        candidate_stream(model, diffusion, decoder, shape, spec, output, targets, variant="flipped", k=1)
        candidate_stream(model, diffusion, decoder, shape, spec, output,
                         targets[:spec["prefix_diagnostic_targets"]], variant="normal", k=spec["prefix_diagnostic_k"])
        result = summarize(output, spec, targets, training)
        files = ("run.json", "synthetic_data.json", "training_resume.pt", "selected.pt", "selection.json", "candidates.sqlite", "candidates.jsonl", "report.json")
        write(output / "complete.json", {name: sha256(output / name) for name in files})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = run(json.loads(args.protocol.read_text()), json.loads(args.config.read_text()), args.output)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
