"""v3 synthetic Pilot execution, with frozen inputs and auditable continuation.

No primary MD5 training/evaluation is performed by this module.
"""
from collections import Counter
from contextlib import closing, contextmanager
import fcntl
import hashlib
import io
import json
import math
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
import torch

from .devices import resolve_device, synchronize as torch_synchronize
from .discrete import MaskedDiffusion, SequenceDenoiser
from .encoding.bgv import BGVDecoder, BGVEncoder
from .encoding.cgge import CGGEDecoder, CGGEEncoder, glyph_table_checksum
from .encoding.tokens import TokenCodec
from .evaluation import exact_mcnemar, holm_adjust
from .models import GaussianDiffusion, ImageUNet, parameter_count


PROTOCOL_SHA256 = "8e3ceefe3c833202273af82411f797b37349d1a648ec868f2d874989c9116fa1"
PROTOCOL_V31_SHA256 = "d2b3a09e97a4243fa4a27e62d3b35f39618502c7bbfb678f1b244aa52674d7c4"
PROTOCOL_V31_P2A10K_SHA256 = "5c89245cc1b1d712492a6dacf84c0c1a5696b22f92d46d10983b50d91c28f41c"
PROTOCOL_V31_P2FIX_SHA256 = "e761b641e7f46cb4cdcd2f135c6be4cc4a90057fd5362b4fdf25e6d1239db3e0"
PROTOCOL_V31_P2STRUCT_SHA256 = "6e2f450767b586bba0e7069c0df9c9bc24d8a4e870cda1e2ad85308cd15d3916"
GIB = 1024 ** 3


def synchronize(device):
    if isinstance(device, torch.device):
        torch_synchronize(device)
    else:
        import mlx.core as mx
        mx.synchronize()


class PilotError(Exception):
    def __init__(self, message, code=4):
        super().__init__(message)
        self.code = code


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_protocol(path):
    try:
        value = read_json(path)
    except (OSError, ValueError) as error:
        raise PilotError(f"Cannot read protocol: {error}", 2) from error
    # A whitelist prevents silently ignoring a changed scientific field or typo.
    if digest(value) not in {PROTOCOL_SHA256, PROTOCOL_V31_SHA256, PROTOCOL_V31_P2A10K_SHA256, PROTOCOL_V31_P2FIX_SHA256, PROTOCOL_V31_P2STRUCT_SHA256}:
        raise PilotError("Unsupported/modified protocol. Only frozen v3.0/v3.1 and registered P2A-10k/P2-fix/P2-struct amendments are accepted; amendments require implementation review.", 2)
    return value


def seed(p, stage, namespace, *, source=None, pipeline=None, method=None,
         model_seed=None, epoch=None, unit_id=None, attempt=None, profile_id=None):
    fields = [p["seeds"].get("protocol_namespace", p["protocol_id"]), p["seeds"]["engineering_master"], "synthetic_nibbles",
              stage, namespace, source, pipeline, method, model_seed, epoch, unit_id, attempt]
    if p["revision"] == "3.1":
        fields.insert(8, profile_id if namespace in p["seeds"]["profile_nonnull_namespaces"] else None)
    raw = json.dumps(fields, separators=(",", ":"), ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def condition(values, device):
    values = list(values)
    if any(type(y) is not int or not 0 <= y < 4096 for y in values):
        raise ValueError("Only public 12-bit conditions may enter the model")
    return torch.tensor([[(y >> shift) & 1 for shift in range(11, -1, -1)] for y in values], dtype=torch.float32, device=device)


def generator(device, value):
    return torch.Generator(device=device).manual_seed(value)


def codecs(pipeline):
    source, representation = pipeline["source"], pipeline["representation"]
    if representation == "bgv":
        return BGVEncoder(), BGVDecoder(), (2, 32, 128)
    if representation == "cgge":
        return CGGEEncoder(), CGGEDecoder(), (2, 32, 64)
    codec = TokenCodec(source, 31)
    return codec, codec, (32,)


class DeterministicEmbedding(torch.nn.Embedding):
    """Same embedding table, with dense matmul gradients instead of MPS scatter-add.

    Repeated tokens caused last-bit differences in MPS embedding backward even
    with deterministic algorithms enabled. The v3 vocabulary is at most 259;
    its small one-hot matrix gives an exact-recovery path without CPU fallback.
    """
    def forward(self, indices):
        return torch.nn.functional.one_hot(indices, self.num_embeddings).to(self.weight.dtype) @ self.weight


def model_and_diffusion(p, pipeline, device, initial_seed, *, profile_id=None, backend="torch"):
    cfg = p["pipelines"][pipeline]
    if backend == "mlx":
        from .mlx_models import build_profile
        return build_profile(p, pipeline, profile_id, seed=initial_seed)
    torch.manual_seed(initial_seed)
    if p["revision"] == "3.1":
        from .study_profiles import build_profile
        codec, _, shape = codecs(cfg)
        model, diffusion = build_profile(p, pipeline, profile_id, shape, codec, device)
        if cfg["model"] == "discrete" and device.type == "mps":
            model.embedding = DeterministicEmbedding.from_pretrained(model.embedding.weight, freeze=False)
        return model.to(device), diffusion
    if cfg["model"] == "gaussian":
        spec = p["models"]["gaussian"]
        model = ImageUNet(2, 12, spec["width"])
        diffusion = GaussianDiffusion(spec["diffusion_steps"], beta_start=spec["beta_start"], beta_end=spec["beta_end"], device=device)
    else:
        spec = p["models"]["discrete"]
        codec = TokenCodec(cfg["source"], 31)
        model = SequenceDenoiser(codec.vocabulary_size, 32, 12, width=spec["width"], embedding_dim=spec["embedding_dim"])
        if device.type == "mps":
            model.embedding = DeterministicEmbedding.from_pretrained(model.embedding.weight, freeze=False)
        diffusion = MaskedDiffusion(codec.mask, torch.linspace(0, 1, spec["sampling_steps"] + 1), device=device)
    return model.to(device), diffusion


def prior(source, rng):
    low, high = (33, 126) if source == "printable" else (0, 255)
    return bytes(rng.randint(low, high) for _ in range(rng.randint(4, 31)))


def synthetic_message(y, source, rng):
    prefix = [y >> 8, (y >> 4) & 15, y & 15]
    if source == "printable":
        prefix = [b"0123456789abcdef"[n] for n in prefix]
    return bytes(prefix) + prior(source, rng)[3:]


def verify(message, source, y):
    if message is None:
        return False, False
    valid = 4 <= len(message) <= 31 and (source == "random_bytes" or all(33 <= b <= 126 for b in message))
    try:
        prefix = int(message[:3].decode("ascii"), 16) if source == "printable" else int.from_bytes(bytes(message[:3]), "big")
        if source == "random_bytes":
            prefix = (message[0] << 8) | (message[1] << 4) | message[2] if all(b < 16 for b in message[:3]) else -1
        elif any(b not in b"0123456789abcdef" for b in message[:3]):
            prefix = -1
    except (ValueError, UnicodeError, IndexError):
        prefix = -1
    return valid, valid and prefix == y


def make_data(p):
    pairs = [(y, y ^ 4095) for y in range(2048)]
    random.Random(seed(p, "SYN_DATA", "split")).shuffle(pairs)
    ntrain = p["synthetic"]["train_conditions"] // 2
    nval = p["synthetic"]["validation_conditions"] // 2
    groups = {"train": pairs[:ntrain], "validation": pairs[ntrain:ntrain + nval], "test": pairs[ntrain + nval:]}
    result = {"pairs": groups, "sources": {}}
    for source in ("printable", "random_bytes"):
        rng = random.Random(seed(p, "SYN_DATA", "corpus", source=source))
        train_conditions = [y for pair in groups["train"] for y in pair]
        records, seen = [], set()
        for _ in range(p["synthetic"]["draw_cap_per_source"]):
            y = rng.choice(train_conditions)
            message = synthetic_message(y, source, rng)
            if message in seen:
                continue
            records.append([y, message.hex()])
            seen.add(message)
            if len(records) == p["synthetic"]["train_unique_messages"]:
                break
        if len(records) != p["synthetic"]["train_unique_messages"]:
            raise PilotError("Synthetic corpus draw cap exhausted", 2)
        validation = [[y, synthetic_message(y, source, rng).hex()] for pair in groups["validation"] for y in pair]
        result["sources"][source] = {"train": records, "validation": validation}
    return result


def stage_targets(p, data, stage, source):
    cfg = p["pilot"][stage]
    if stage == "P1":
        pool = [y for pair in data["pairs"]["validation"] for y in pair][:cfg["validation_conditions"]]
        rng = random.Random(seed(p, stage, "trial-list", source=source))
        return [(str(i + 1), rng.choice(pool)) for i in range(cfg["evaluation_trials"])]
    pairs = data["pairs"]["test"] if stage == "P3" else random.Random(seed(p, stage, "dev-probe")).sample(data["pairs"]["validation"], cfg["probe_conditions"] // 2)
    return [(f"case:{y:03x}", y) for pair in pairs for y in pair]


def plan(p, stage, device, development):
    if p["revision"] == "3.1":
        from .study_v31 import plan as v31_plan
        return v31_plan(p, stage, device, development)
    cfg = p["pilot"][stage]
    return {"protocol": p["protocol_id"], "stage": stage, "mode": "DEVELOPMENT_ONLY" if development else "V3_PILOT",
            "device": device, "dry_run": True, "device_checked": False,
            "prerequisite": None if stage == "P0" else f"P{int(stage[1]) - 1}",
            "pipelines": p["pipeline_order"], "settings": cfg,
            "wall_cap_seconds": p["execution"]["hard_stage_active_wall_seconds"][stage],
            "note": "P3 uses the batch and resource budgets sealed by P2. No training was performed."}


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PilotError(f"Another process holds {path}", 2) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def environment(device, threads):
    base = Path(__file__).parent
    return {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
            "os": platform.platform(), "machine": platform.machine(), "device": str(device),
            "threads": threads, "precision": "float32", "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
            "source_sha256": {str(path.relative_to(base)): file_hash(path) for path in sorted(base.rglob("*.py"))}}


class Budget:
    def __init__(self, root, stage, state, p, device, *, backend="torch"):
        self.root, self.stage, self.state, self.p, self.device = root, stage, state, p, device
        self.last = time.monotonic()
        self.run = None
        self.backend = backend
        self.storage_bytes = None
        self.reserved_bytes = 0
        self.run_storage = {}
        self.other_poc_seconds = (sum(read_json(path).get("active_seconds", 0.)
                                      for path in (root / "pilot").glob("*/state.json")
                                      if path.parent.name != stage) if p["revision"] == "3.1" else 0.)

    def reserve(self, run, count, *, refresh=False):
        """Conservatively reserve writes between full scans; one study writer holds the lock."""
        if self.p["revision"] != "3.1":
            return
        if count < 0:
            raise ValueError("negative write reservation")
        cfg = self.p["execution"]
        if refresh or self.storage_bytes is None or self.reserved_bytes >= 16 * 2 ** 20:
            self.storage_bytes = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())
            self.reserved_bytes = 0
            self.run_storage.clear()
        if self.storage_bytes + self.reserved_bytes + count > cfg["hard_study_storage_gib"] * GIB:
            raise PilotError("Study storage cap reached before write", 5)
        if shutil.disk_usage(self.root).free < cfg["minimum_disk_free_gib"] * GIB + count:
            raise PilotError("Insufficient free disk space for reserved write", 5)
        if run is not None:
            key = str(run)
            if key not in self.run_storage:
                self.run_storage[key] = sum(path.stat().st_size for path in Path(run).rglob("*") if path.is_file())
            if self.run_storage[key] + count > cfg["hard_run_storage_gib"] * GIB:
                raise PilotError("Run storage cap reached before write", 5)
            self.run_storage[key] += count
        self.reserved_bytes += count

    def tick(self, run=None):
        now = time.monotonic()
        elapsed = now - self.last
        self.state["active_seconds"] += elapsed
        if self.run is not None:
            self.state.setdefault("run_active_seconds", {}).setdefault(self.run, 0)
            self.state["run_active_seconds"][self.run] += elapsed
        self.last = now
        self.run = str(run) if run is not None else None
        self.reserve(None, len(canonical(self.state)) + 1024)
        atomic_json(self.root / "pilot" / self.stage / "state.json", self.state)

    def check(self, run=None):
        self.tick(run)
        cfg = self.p["execution"]
        cap = cfg["hard_stage_active_wall_seconds"][self.stage]
        resources_path = self.root / "resources.json"
        resources = read_json(resources_path) if resources_path.exists() else None
        v31 = self.p["revision"] == "3.1"
        if self.stage == "P3" and resources is not None and not v31:
            cap = min(cap, resources["p3_soft_wall_seconds"])
        if self.state["active_seconds"] >= cap:
            raise PilotError(f"{self.stage} active wall-clock cap reached", 5)
        if run is not None:
            run_cap = cfg["hard_formal_run_active_wall_seconds"]
            if self.stage == "P3" and not v31:
                relative = Path(run).relative_to(self.root / "pilot" / self.stage / "runs")
                run_cap = min(run_cap, resources["pipelines"][relative.parts[0]]["p3_run_seconds"])
            if self.state.get("run_active_seconds", {}).get(str(run), 0) >= run_cap:
                raise PilotError("Per-run active wall-clock cap reached", 5)
        if v31:
            if self.other_poc_seconds + self.state["active_seconds"] >= cfg["hard_stage_active_wall_seconds"]["POC_TOTAL"]:
                raise PilotError("Cumulative PoC active wall-clock cap reached", 5)
            if self.stage == "P3" and resources is not None:
                exceeded = self.state["active_seconds"] > resources["p3_soft_wall_seconds"]
                if run is not None:
                    pipeline = Path(run).relative_to(self.root / "pilot" / self.stage / "runs").parts[0]
                    exceeded |= self.state.get("run_active_seconds", {}).get(str(run), 0) > resources["pipelines"][pipeline]["p3_run_seconds"]
                if exceeded:
                    self.state.setdefault("warnings", {})["RESOURCE_ESTIMATE_EXCEEDED"] = True
        if shutil.disk_usage(self.root).free < cfg["minimum_disk_free_gib"] * GIB:
            raise PilotError("Minimum free disk space reached", 5)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if self.backend == "mlx":
            import mlx.core as mx
            mps = mx.get_active_memory() if self.device.type == mx.gpu else 0
        else:
            mps = torch.mps.current_allocated_memory() if self.device.type == "mps" else 0
        if rss > cfg["hard_process_rss_gib"] * GIB or mps > cfg["hard_mps_allocated_gib"] * GIB:
            raise PilotError("Process/GPU memory cap reached", 5)
        if v31:
            self.reserve(run, 65536)  # Budget for telemetry/metadata until the next check.
            return
        size = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())
        if size > cfg["hard_study_storage_gib"] * GIB:
            raise PilotError("Study storage cap reached", 5)
        if run and Path(run).is_dir() and sum(f.stat().st_size for f in Path(run).rglob("*") if f.is_file()) > cfg["hard_run_storage_gib"] * GIB:
            raise PilotError("Run storage cap reached", 5)


def event(directory, **row):
    with (directory / "telemetry.jsonl").open("a") as stream:
        stream.write(canonical({"wall_time": time.time(), **row}).decode() + "\n")


def save_checkpoint(directory, state, *, best=False, budget=None):
    folder = directory / "checkpoints"
    folder.mkdir(exist_ok=True)
    native_mlx = state["identity"].get("backend") == "mlx"
    extension = "safetensors" if native_mlx else "pt"
    name = f"update-{state['update']:08d}-epoch-{state['epoch']:04d}.{extension}"
    path = folder / name
    payload = None
    if native_mlx:
        from .mlx_backend import checkpoint_bytes
        payload = checkpoint_bytes(state)
        if budget is not None:
            budget.reserve(directory, len(payload) + 16384, refresh=True)
    elif budget is not None and budget.p["revision"] == "3.1":
        buffer = io.BytesIO()
        torch.save(state, buffer)
        payload = buffer.getvalue()
        budget.reserve(directory, len(payload) + 16384, refresh=True)
    with (folder / "pending.tmp").open("wb") as stream:
        if payload is None:
            torch.save(state, stream)
        else:
            stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(folder / "pending.tmp", path)
    entry = {"file": name, "sha256": file_hash(path)}
    pointer = folder / "LATEST.json"
    old = read_json(pointer)["history"] if pointer.exists() else []
    history = [entry] + [x for x in old if x["file"] != name][:1]
    atomic_json(pointer, {"history": history})
    if best:
        atomic_json(folder / "BEST.json", entry)
    kept = {x["file"] for x in history}
    if (folder / "BEST.json").exists():
        kept.add(read_json(folder / "BEST.json")["file"])
    for stale in folder.glob("update-*"):
        if stale.suffix in {".pt", ".safetensors"} and stale.name not in kept:
            stale.unlink()


def load_checkpoint(directory, *, best=False):
    folder = directory / "checkpoints"
    pointer = read_json(folder / ("BEST.json" if best else "LATEST.json"))
    entry = pointer if best else pointer["history"][0]
    path = folder / entry["file"]
    if path.parent != folder or file_hash(path) != entry["sha256"]:
        raise PilotError("Checkpoint checksum mismatch")
    if path.suffix == ".safetensors":
        from .mlx_backend import load_checkpoint as load_mlx_checkpoint
        return load_mlx_checkpoint(path), entry["sha256"]
    return torch.load(path, map_location="cpu", weights_only=True), entry["sha256"]


class RecoveryPause(Exception):
    """Deliberate interruption used only by the required recovery comparison."""


def clean_batch(encoder, records, device, discrete):
    clean = torch.stack([encoder.encode(bytes.fromhex(row[1])) for row in records]).to(device)
    return clean if discrete else clean * 2 - 1


@torch.no_grad()
def validation_loss(p, stage, pipeline, model, diffusion, records, device):
    """Fixed per-case/draw noise, shared across epochs, methods and model seeds."""
    if getattr(diffusion, "backend", None) == "mlx":
        from .mlx_backend import validation_loss as mlx_validation_loss
        return mlx_validation_loss(p, stage, pipeline, model, diffusion, records)
    source = p["pipelines"][pipeline]["source"]
    encoder, _, _ = codecs(p["pipelines"][pipeline])
    discrete = isinstance(diffusion, MaskedDiffusion)
    total = 0.0
    model.eval()
    for start in range(0, len(records), p["training"]["batch_size"]):
        rows = records[start:start + p["training"]["batch_size"]]
        clean = clean_batch(encoder, rows, device, discrete)
        cond = condition([r[0] for r in rows], device)
        for draw in range(1, p["training"]["validation_draws_per_condition"] + 1):
            gens = [generator(device, seed(p, stage, "validation-noise", source=source, pipeline=pipeline,
                                          unit_id=f"case:{row[0]:03x}", attempt=draw)) for row in rows]
            if discrete:
                times = torch.stack([torch.rand((), device=device, generator=g) for g in gens])
                masks = torch.stack([torch.rand(clean.shape[1:], device=device, generator=g) for g in gens]) < times[:, None]
                from .study_profiles import LengthMaskedDiffusion
                if isinstance(diffusion, LengthMaskedDiffusion):
                    values = diffusion.losses(model, clean, cond, times, masks)
                else:
                    logits = model(clean.masked_fill(masks, diffusion.mask_token), times, cond)
                    losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), clean, reduction="none")
                    values = (losses * masks).sum(1) / masks.sum(1).clamp_min(1)
            else:
                indices = torch.stack([torch.randint(diffusion.steps, (), device=device, generator=g) for g in gens])
                noise = torch.stack([torch.randn(clean.shape[1:], device=device, generator=g) for g in gens])
                if getattr(diffusion, "loss_regions", None):
                    values = diffusion.losses(model, clean, cond, indices, noise)
                else:
                    noisy = diffusion.add_noise(clean, noise, indices)
                    target = noise if diffusion.prediction_type == "epsilon" else clean
                    values = (model(noisy, indices.float() / (diffusion.steps - 1), cond) - target).square().flatten(1).mean(1)
            if not torch.isfinite(values).all():
                raise PilotError("Non-finite validation loss", 3)
            total += values.sum().item()
    return total / (len(records) * p["training"]["validation_draws_per_condition"])


def train_model(p, stage, pipeline, method, label, data, directory, device, budget, *, pause_update=None, profile_id=None,
                training_rows=None, epoch_callback=None):
    native_mlx = budget.backend == "mlx"
    if native_mlx:
        from . import mlx_backend as mlx
    if p["revision"] == "3.1":
        budget.check(directory)
        budget.reserve(directory, 65536, refresh=True)
    directory.mkdir(parents=True, exist_ok=True)
    source = p["pipelines"][pipeline]["source"]
    cfg, training = p["pilot"][stage], p["training"]
    rows = training_rows if training_rows is not None else data["sources"][source]["train"][:cfg["train_messages"]]
    validation = data["sources"][source]["validation"][:cfg.get("validation_conditions", 0)]
    batch_size = cfg.get("batch_size", training["batch_size"])
    final_only = stage == "P2A"
    epochs = cfg["optimizer_updates_per_run"] if final_only else cfg["epochs"]
    if len(rows) != cfg["train_messages"] or (final_only and len(rows) != batch_size):
        raise PilotError("Training corpus does not satisfy the registered stage size", 2)
    identity = {"protocol_sha256": digest(p), "stage": stage, "pipeline": pipeline, "method": method,
                "model_seed": label, "data_sha256": digest([rows, validation]), "device": str(device)}
    if p["revision"] == "3.1":
        identity["profile_id"] = profile_id
    if native_mlx:
        identity.update(backend="mlx", rng_scheme="explicit_mlx_key64_v1", optimizer_bias_correction=True)
    config_path = directory / "configuration.json"
    if config_path.exists() and read_json(config_path) != identity:
        raise PilotError("Run configuration/data mismatch")
    atomic_json(config_path, identity)
    initial = seed(p, stage, "initialization", source=source, pipeline=pipeline, model_seed=label, profile_id=profile_id)
    model, diffusion = model_and_diffusion(p, pipeline, device, initial, profile_id=profile_id, backend=budget.backend)
    if native_mlx:
        optimizer, train_step = mlx.optimizer_and_step(model, diffusion, training)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"], betas=tuple(training["betas"]),
                                     eps=training["eps"], weight_decay=training["weight_decay"], foreach=False, fused=False)
    encoder, _, _ = codecs(p["pipelines"][pipeline])
    state = {"identity": identity, "epoch": 1, "offset": 0, "update": 0, "best_loss": None,
             "best_epoch": None, "curve": [], "training_update_seconds": [], "validation_seconds": [],
             "checkpoint_seconds": [], "full_loop_update_seconds": [],
             "full_loop_timing_excludes": ["validation", "checkpoint"]}

    def save(best=False):
        synchronize(device)
        started = time.monotonic()
        if native_mlx:
            state.update(model=model.parameters(), optimizer=optimizer.state)
        else:
            state.update(model=model.state_dict(), optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
                         device_rng=torch.mps.get_rng_state() if device.type == "mps" else torch.get_rng_state())
        save_checkpoint(directory, state, best=best, budget=budget)
        state["checkpoint_seconds"].append(time.monotonic() - started)

    if (directory / "checkpoints" / "LATEST.json").exists():
        state, _ = load_checkpoint(directory)
        if state["identity"] != identity:
            raise PilotError("Checkpoint identity mismatch")
        if native_mlx:
            mlx.load_weights(model, state["model"])
            optimizer.state = state["optimizer"]
        else:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["cpu_rng"])
            if device.type == "mps":
                torch.mps.set_rng_state(state["device_rng"])
    else:
        save()
    while state["epoch"] <= epochs:
        # The preceding epoch is durable before its probe. On resume an unfinished
        # probe is replayed before training can advance or prune its checkpoint.
        if epoch_callback and state["offset"] == 0 and state["epoch"] > 1:
            epoch_callback(state["epoch"] - 1)
        epoch = state["epoch"]
        order = list(range(len(rows)))
        random.Random(seed(p, stage, "train-order", source=source, pipeline=pipeline, model_seed=label, epoch=epoch)).shuffle(order)
        donors = list(range(len(rows)))
        if method == "shuffled":
            random.Random(seed(p, stage, "shuffle", source=source, pipeline=pipeline, model_seed=label, epoch=epoch)).shuffle(donors)
        state["permutation"] = order
        state["donors"] = donors
        if state["offset"] == 0:
            event(directory, kind="epoch", epoch=epoch, order_sha256=digest(order), donor_sha256=digest(donors),
                  same_condition_fraction=sum(rows[i][0] == rows[donors[i]][0] for i in range(len(rows))) / len(rows))
        while state["offset"] < len(rows):
            loop_started = time.monotonic()
            budget.check(directory)
            indices = order[state["offset"]:state["offset"] + batch_size]
            synchronize(device)
            started = time.monotonic()
            if native_mlx:
                clean = mlx.clean_batch(encoder, [rows[i] for i in indices], diffusion)
                cond = mlx.models.condition([rows[donors[i]][0] for i in indices])
            else:
                clean = clean_batch(encoder, [rows[i] for i in indices], device, isinstance(diffusion, MaskedDiffusion))
                cond = condition([rows[donors[i]][0] for i in indices], device)
            noise_seed = seed(p, stage, "train-noise", source=source, pipeline=pipeline, method=method, model_seed=label,
                              epoch=epoch, unit_id=state["offset"] + 1, attempt=state["update"] + 1)
            model.train()
            components = {}
            if native_mlx:
                loss, components = train_step(clean, cond, noise_seed)
            else:
                optimizer.zero_grad(set_to_none=True)
                if getattr(diffusion, "factorized", False):
                    loss, components = diffusion.loss(model, clean, cond, generator=generator(device, noise_seed), return_components=True)
                else:
                    loss = diffusion.loss(model, clean, cond, generator=generator(device, noise_seed))
                if not torch.isfinite(loss):
                    raise PilotError("Non-finite training loss", 3)
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise PilotError("Non-finite gradient", 3)
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise PilotError("Non-finite model weights", 3)
            synchronize(device)
            seconds = time.monotonic() - started
            state["update"] += 1
            state["offset"] += len(indices)
            state["training_update_seconds"].append(seconds)
            event(directory, kind="update", epoch=epoch, update=state["update"], loss=loss.item(), seconds=seconds,
                  **{name: value.item() for name, value in components.items()})
            state.setdefault("full_loop_update_seconds", []).append(time.monotonic() - loop_started)
            if pause_update == state["update"]:
                save()
                raise RecoveryPause("optimizer boundary")
            if state["update"] % training["checkpoint_every_optimizer_updates"] == 0:
                save()
        improved = False
        if epoch in cfg.get("validation_epochs", []):
            budget.check(directory)
            synchronize(device)
            started = time.monotonic()
            value = validation_loss(p, stage, pipeline, model, diffusion, validation, device)
            synchronize(device)
            state["validation_seconds"].append(time.monotonic() - started)
            state["curve"].append({"epoch": epoch, "validation_loss": value})
            improved = state["best_loss"] is None or value < state["best_loss"]
            if improved:
                state["best_loss"], state["best_epoch"] = value, epoch
            event(directory, kind="validation", **state["curve"][-1], best_epoch=state["best_epoch"])
        state["epoch"], state["offset"] = epoch + 1, 0
        if not final_only or epoch == epochs or epoch in cfg.get("diagnostic_updates", []):
            save(best=improved)
    if epoch_callback:
        epoch_callback(epochs)
    best, checksum = load_checkpoint(directory, best=not final_only and cfg.get("selection_checkpoint") != "final_epoch")
    if native_mlx:
        state["model"] = model.parameters()
        mlx.load_weights(model, best["model"])
    else:
        state["model"] = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        model.load_state_dict(best["model"])
    model.eval()
    return model, diffusion, state, checksum


@torch.no_grad()
def sample(p, pipeline, model, diffusion, targets, seeds, device, *, to_cpu=True, profile_id=None, length_seeds=None, trace=None):
    """Batch model forwards with an independent backend RNG for every trajectory."""
    if getattr(diffusion, "backend", None) == "mlx":
        from . import mlx_backend as mlx
        shape = codecs(p["pipelines"][pipeline])[2]
        values = mlx.models.sample(model, diffusion, mlx.models.condition(targets), shape,
                                    steps=p["model_profiles"][profile_id]["sampling_steps"], seeds=seeds, length_seeds=length_seeds, trace=trace)
        return mlx.to_codec_tensor(values) if to_cpu else values
    cond = condition(targets, device)
    gens = [generator(device, value) for value in seeds]
    _, _, shape = codecs(p["pipelines"][pipeline])
    model.eval()
    if p["revision"] == "3.1":
        from .study_profiles import profile_ids, sample_independent
        if profile_id not in profile_ids(p, pipeline):
            raise PilotError("Explicit registered profile required for v3.1 sampling", 2)
        length_gens = [generator(device, value) for value in length_seeds] if length_seeds is not None else None
        value = sample_independent(model, diffusion, cond, shape, p["model_profiles"][profile_id]["sampling_steps"],
                                   gens, length_generators=length_gens, trace=trace)
        return value.cpu() if to_cpu else value
    if isinstance(diffusion, GaussianDiffusion):
        steps = p["models"]["gaussian"]["sampling_steps"]
        times = torch.linspace(diffusion.steps - 1, 0, steps, device=device).round().long().unique_consecutive()
        value = torch.stack([torch.randn(shape, device=device, generator=g) for g in gens])
        for position, index in enumerate(times):
            clock = torch.full((len(targets),), index.item() / (diffusion.steps - 1), device=device)
            noise = model(value, clock, cond)
            clean = diffusion.predicted_clean(noise, value, index.repeat(len(targets)))
            alpha = diffusion.alpha_bar[times[position + 1]] if position + 1 < len(times) else torch.tensor(1., device=device)
            value = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        if not torch.isfinite(value).all():
            raise PilotError("Non-finite Gaussian sampler", 3)
        value = value.clamp(-1, 1)
        return value.cpu() if to_cpu else value
    value = torch.full((len(targets), *shape), diffusion.mask_token, dtype=torch.long, device=device)
    for current in range(diffusion.steps, 0, -1):
        clock = diffusion.probabilities[current].expand(len(targets))
        logits = model(value, clock, cond)
        if not torch.isfinite(logits).all():
            raise PilotError("Non-finite discrete sampler", 3)
        tokens = torch.stack([torch.multinomial(row.softmax(-1), 1, generator=g).squeeze(-1) for row, g in zip(logits, gens)])
        probability = 1 - diffusion.probabilities[current - 1] / diffusion.probabilities[current]
        reveal = torch.stack([torch.rand(shape, device=device, generator=g) for g in gens]) < probability
        value = torch.where((value == diffusion.mask_token) & reveal, tokens, value)
    return value.cpu() if to_cpu else value


def ledger_open(path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("CREATE TABLE IF NOT EXISTS candidates (run_id TEXT, unit_id TEXT, variant TEXT, attempt INTEGER, record TEXT NOT NULL, PRIMARY KEY(run_id, unit_id, variant, attempt))")
    return connection


def evaluation_jobs(p, stage, targets, settings=None):
    cfg = settings if settings is not None else p["pilot"][stage]
    return [(unit, y, variant, attempt) for variant in cfg["condition_variants"] for unit, y in targets for attempt in range(1, cfg["k"] + 1)]


def evaluate(p, stage, pipeline, method, label, targets, directory, model, diffusion, checkpoint,
             batch_size, device, budget, *, source=None, pause_after=None, profile_id=None, settings=None):
    if p["revision"] == "3.1":
        budget.check(directory)
        budget.reserve(directory, 2 ** 20, refresh=True)
    directory.mkdir(parents=True, exist_ok=True)
    source = source or p["pipelines"][pipeline]["source"]
    run_id = f"{stage}/{pipeline or source}/{method}/{label}"
    if profile_id is not None:
        run_id += f"/{profile_id}"
    cfg = settings if settings is not None else p["pilot"][stage]
    jobs = evaluation_jobs(p, stage, targets, cfg)
    config = {"protocol": digest(p), "run_id": run_id, "targets": targets, "batch_size": batch_size, "checkpoint": checkpoint}
    if p["revision"] == "3.1":
        config["profile_id"] = profile_id
        config["backend"] = budget.backend
    if settings is not None:
        config["evaluation_settings"] = settings
    config_sha = digest(config)
    config_path = directory / "evaluation.json"
    if config_path.exists() and digest(read_json(config_path)) != config_sha:
        raise PilotError("Evaluation identity/batch/checkpoint mismatch")
    atomic_json(config_path, config)
    connection = ledger_open(directory / "candidates.sqlite")
    decoder = codecs(p["pipelines"][pipeline])[1] if pipeline else None
    expected_keys = [(run_id, unit, variant, attempt) for unit, _, variant, attempt in jobs]
    try:
        stored = {tuple(row[:4]): json.loads(row[4]) for row in connection.execute("SELECT * FROM candidates")}
        if not set(stored).issubset(set(expected_keys)):
            raise PilotError("Unexpected ledger keys")
        # Completed batches are immutable; partially committed batches are corruption.
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start + batch_size]
            keys = expected_keys[start:start + batch_size]
            present = [key in stored for key in keys]
            if any(present) and not all(present):
                raise PilotError("Incomplete inference transaction")
            if all(present):
                if any(stored[key]["config_sha256"] != config_sha for key in keys):
                    raise PilotError("Ledger configuration checksum mismatch")
                continue
            loop_started = time.monotonic()
            budget.check(directory)
            values = [y if variant == "normal" else y ^ 4095 for _, y, variant, _ in batch]
            namespace = cfg.get("rng_namespace") or ("generation-payload" if profile_id in {"D0", "D1"} else "generation" if pipeline else "random")
            seeds = [seed(p, stage, namespace, source=source, pipeline=pipeline,
                          method=method, model_seed=label, unit_id=unit, attempt=attempt, profile_id=profile_id) for unit, _, _, attempt in batch]
            length_seeds = ([seed(p, stage, "generation-length", source=source, pipeline=pipeline,
                                 method=method, model_seed=label, unit_id=unit, attempt=attempt)
                             for unit, _, _, attempt in batch] if getattr(diffusion, "factorized", False) else None)
            prefix_reveals = [[] for _ in batch]

            def record_reveals(step, fraction, logits, tokens, revealed):
                array = lambda x: np.array(x) if budget.backend == "mlx" else x.detach().cpu().numpy()
                scores, selected, visible = array(logits), array(tokens), array(revealed)
                probabilities = np.exp(scores - scores.max(-1, keepdims=True))
                probabilities /= probabilities.sum(-1, keepdims=True)
                offset = 33 if source == "printable" else 0
                for i, position in zip(*np.nonzero(visible)):
                    y = values[i]
                    expected = f"{y:03x}".encode() if source == "printable" else bytes([y >> 8, (y >> 4) & 15, y & 15])
                    token = int(selected[i, position])
                    prefix_reveals[i].append({"position": int(position), "step": step, "mask_fraction": fraction,
                                              "expected_byte": expected[position], "sampled_byte": token + offset,
                                              "argmax_byte": int(scores[i, position].argmax()) + offset,
                                              "correct_probability": float(probabilities[i, position, expected[position] - offset]),
                                              "sampled_probability": float(probabilities[i, position, token])})

            trace_enabled = profile_id == "D1" and cfg.get("record_prefix_reveals", False)
            synchronize(device)
            started = time.monotonic()
            samples = sample(p, pipeline, model, diffusion, values, seeds, device,
                             profile_id=profile_id, length_seeds=length_seeds,
                             trace=record_reveals if trace_enabled else None) if pipeline else None
            synchronize(device)
            generation_seconds = time.monotonic() - started
            if p["revision"] == "3.1":
                raw_bytes = samples.numel() * samples.element_size() if samples is not None else 0
                budget.reserve(directory, len(batch) * 16384 + raw_bytes + 65536)
            rows = []
            for index, ((unit, original, variant, attempt), target, rng_seed) in enumerate(zip(batch, values, seeds)):
                if pipeline:
                    decoded = decoder.decode(samples[index]) if isinstance(decoder, TokenCodec) else decoder.decode(samples[index], normalized=True)
                    payload, decoded_valid, reason = decoded.message, decoded.valid, decoded.reason
                else:
                    payload, decoded_valid, reason = prior(source, random.Random(rng_seed)), True, None
                domain, hit = verify(payload, source, target)
                valid = decoded_valid and domain
                row = {"source": source, "pipeline": pipeline, "method": method, "model_seed": label,
                       "task": "synthetic_nibbles", "unit_id": unit, "requested_target": target,
                       "original_target": original, "variant": variant, "attempt": attempt,
                       "candidate_hex": payload.hex() if payload is not None else None,
                       "byte_length": len(payload) if payload is not None else None, "valid": valid,
                       "reason": reason if not decoded_valid else None if domain else "source_domain",
                       "success": bool(valid and hit), "wrong_original": bool(valid and verify(payload, source, original)[1]),
                       "verifier_kind": "synthetic_nibbles", "verifier_calls": int(payload is not None), "md5_calls": 0,
                       "rng_identity": str(rng_seed), "checkpoint_sha256": checkpoint, "config_sha256": config_sha,
                       "generation_seconds_per_candidate": generation_seconds / len(batch)}
                if p["revision"] == "3.1":
                    row.update(profile_id=profile_id,
                               backend=budget.backend,
                               sampling_nfe=p["model_profiles"][profile_id]["sampling_nfe_per_candidate"] if pipeline else 0)
                    if length_seeds is not None:
                        sampled_length = (int((samples[index] == decoder.eos).nonzero()[0].item()) if isinstance(decoder, TokenCodec)
                                          else int((samples[index, 1] > 0).sum().item()) // (8 * (decoder.config.image_width // decoder.config.cols))
                                          - int(p["pipelines"][pipeline]["representation"] == "bgv"))
                        row.update(length_rng_identity=str(length_seeds[index]),
                                   sampled_length=sampled_length)
                    if trace_enabled:
                        if sorted(x["position"] for x in prefix_reveals[index]) != [0, 1, 2]:
                            raise PilotError("Missing or repeated prefix reveal diagnostic")
                        row["prefix_reveals"] = prefix_reveals[index]
                rows.append(row)
                if pipeline and variant == "normal" and attempt == 1 and unit in {u for u, _ in targets[:16]}:
                    raw = directory / "raw"
                    raw.mkdir(exist_ok=True)
                    torch.save(samples[index], raw / f"{unit.replace(':', '-')}.pt")
            # The P1 attempt-7 fault occurs before commit: replay the whole batch.
            if pause_after and any(unit == targets[0][0] and variant == "normal" and attempt == pause_after for unit, _, variant, attempt in batch):
                raise RecoveryPause("generation before batch commit")
            with connection:
                connection.executemany("INSERT INTO candidates VALUES (?,?,?,?,?)", [(*key, canonical(row).decode()) for key, row in zip(keys, rows)])
            event(directory, kind="inference_batch", candidates=len(rows), generation_seconds=generation_seconds,
                  total_seconds=time.monotonic() - started, full_loop_seconds=time.monotonic() - loop_started)
        records = [json.loads(row[0]) for row in connection.execute("SELECT record FROM candidates ORDER BY rowid")]
        if len(records) != len(jobs):
            raise PilotError("Missing ledger rows")
        by_key = {(r["unit_id"], r["variant"], r["attempt"]): r for r in records}
        for unit, original, variant, attempt in jobs:
            row = by_key[(unit, variant, attempt)]
            target = original if variant == "normal" else original ^ 4095
            payload = bytes.fromhex(row["candidate_hex"]) if row["candidate_hex"] is not None else None
            if row["requested_target"] != target or row["original_target"] != original or row["success"] != bool(row["valid"] and verify(payload, source, target)[1]):
                raise PilotError("Independent ledger verifier mismatch")
        metrics = {"candidates": len(records), "trials_or_cases": len(targets), "unique_targets": len(set(y for _, y in targets)),
                   "valid_rate": sum(r["valid"] for r in records) / len(records), "invalid_reasons": dict(Counter(r["reason"] for r in records if not r["valid"])),
                   "duplicate_payloads": sum(n - 1 for n in Counter(r["candidate_hex"] for r in records if r["candidate_hex"] is not None).values()),
                   "verifier_calls": sum(r["verifier_calls"] for r in records), "md5_calls": 0,
                   "normal_joint": sum(r["success"] for r in records if r["variant"] == "normal"),
                   "flipped_joint": sum(r["success"] for r in records if r["variant"] == "flipped"),
                   "wrong_original": sum(r["wrong_original"] for r in records if r["variant"] == "flipped"),
                   "success_at_k": {str(k): sum(any(by_key[(unit, "normal", a)]["success"] for a in range(1, k + 1)) for unit, _ in targets) / len(targets)
                                    for k in (1, 10, 100) if k <= cfg["k"]},
                   "measured_k": cfg["k"], "batch_size": batch_size}
        if p["revision"] == "3.1":
            variants = {}
            for variant in cfg["condition_variants"]:
                subset = [r for r in records if r["variant"] == variant]
                valid_count = sum(r["valid"] for r in subset)
                prefix_correct = [0, 0, 0]
                for row in subset:
                    payload = bytes.fromhex(row["candidate_hex"]) if row["candidate_hex"] is not None else b""
                    y = row["requested_target"]
                    expected = f"{y:03x}".encode() if source == "printable" else bytes([y >> 8, (y >> 4) & 15, y & 15])
                    for position in range(3):
                        prefix_correct[position] += bool(row["valid"] and len(payload) > position and payload[position] == expected[position])
                variants[variant] = {"candidates": len(subset), "valid": valid_count,
                                     "joint": sum(r["success"] for r in subset),
                                     "wrong_original": sum(r["wrong_original"] for r in subset),
                                     "prefix_position_correct_strict": prefix_correct,
                                     "conditional_accuracy_given_valid": sum(r["success"] for r in subset) / valid_count if valid_count else None}
            metrics.update(profile_id=profile_id, sampling_nfe=sum(r["sampling_nfe"] for r in records),
                           variants=variants,
                           conditional_accuracy_given_valid=(sum(r["success"] for r in records) / sum(r["valid"] for r in records)
                                                             if any(r["valid"] for r in records) else None),
                           warnings=["JOINT_SUCCESS_ZERO"] if metrics["normal_joint"] == 0 else [])
        atomic_json(directory / "metrics.json", metrics)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return metrics
    finally:
        connection.close()


def family_decision(components):
    """The v3 intersection test; missing components only pad the p-value family."""
    pvalues = {name: max(values) if values is not None and len(values) == 6 else 1.0 for name, values in components.items()}
    if len(pvalues) != 5:
        raise ValueError("v3 requires the fixed family of five pipelines")
    return holm_adjust(pvalues)


def preflight(p, directory, device, budget):
    from .hashing.md5 import MD5Tracer
    checks = {}
    native_mlx = budget.backend == "mlx"
    if native_mlx:
        from . import mlx_backend as mlx
    codec_device = torch.device("cpu") if native_mlx else device

    def require(name, passed, detail=None):
        checks[name] = {"passed": bool(passed), "detail": detail}
        atomic_json(directory / "checks.json", checks)
        if not passed:
            raise PilotError(f"P0 fixture failed: {name}", 3)

    value = mlx.mx.arange(16, dtype=mlx.mx.float32) if native_mlx else torch.arange(16, dtype=torch.float32, device=device)
    synchronize(device)
    total = (value * value).sum().item()
    require("device_tensor", total == 1240, str(device))
    require("font", glyph_table_checksum() == p["codecs"]["cgge"]["font_sha256"])
    v31 = p["revision"] == "3.1"
    if v31:
        budget.reserve(directory, 8 * 2 ** 20, refresh=True)
    else:
        for filename, kind in (("models.py", "gaussian"), ("discrete.py", "discrete")):
            require(f"source_{filename}", file_hash(Path(__file__).with_name(filename)) == p["models"][kind]["reference_source_sha256"])
    count = 0
    registered = {}
    for pipeline in p["pipeline_order"]:
        budget.check()
        cfg = p["pipelines"][pipeline]
        encoder, decoder, shape = codecs(cfg)
        low, high = (33, 126) if cfg["source"] == "printable" else (0, 255)
        fixtures = [bytes([b]) * 4 for b in range(low, high + 1)]
        fixtures += [bytes([low]) * n for n in range(4, 32)]
        fixtures += [bytes([high]) * n for n in range(4, 32)]
        fixtures += [bytes(low + i % (high - low + 1) for i in range(n)) for n in range(4, 32)]
        require(f"roundtrip_{pipeline}", all(decoder.decode(encoder.encode(x)).message == x for x in fixtures), len(fixtures))
        count += len(fixtures)
        encoded = encoder.encode(fixtures[0])
        bad = encoded.clone()
        if isinstance(decoder, TokenCodec):
            bad[0] = decoder.mask
            require(f"mask_rejected_{pipeline}", not decoder.decode(bad).valid)
            bad = encoded.clone()
            bad[-1] = decoder.eos
            require(f"eos_rejected_{pipeline}", not decoder.decode(bad).valid)
        else:
            bad[0, 0, 0] = float("nan")
            require(f"nonfinite_rejected_{pipeline}", not decoder.decode(bad).valid)
            bad = encoded.clone()
            bad[1] = 0
            require(f"mask_rejected_{pipeline}", not decoder.decode(bad).valid)
        require(f"shape_rejected_{pipeline}", not decoder.decode(encoded[:1]).valid)
        from .study_profiles import LengthMaskedDiffusion, profile_ids
        for profile_id in profile_ids(p, pipeline) if v31 else [None]:
            model, diffusion = model_and_diffusion(p, pipeline, device, 0, profile_id=profile_id, backend=budget.backend)
            spec = p["baseline_reference_models" if v31 else "models"][cfg["model"]]
            expected = spec["expected_parameters"] if cfg["model"] == "gaussian" else spec[f"expected_parameters_{cfg['source']}"]
            if profile_id in {"G1", "G2", "G3"}:
                expected += 2 * 3 * 3 * p["model_profiles"][profile_id]["width"]
            elif profile_id == "D1":
                expected += p["model_profiles"][profile_id]["width"] + 12 * 28 + 28
            if profile_id == "G3":
                expected += p["model_profiles"][profile_id]["width"] * 3 + 12 * 28 + 28
            if v31 and p["model_profiles"][profile_id].get("condition_output"):
                profile = p["model_profiles"][profile_id]
                expected += ((profile["width"] * 3 + 1) * math.prod(shape) if cfg["model"] == "gaussian"
                             else (profile["model_condition_dim"] + 1) * 32 * (encoder.vocabulary_size - 1))
            name = f"{pipeline}/{profile_id}" if v31 else pipeline
            actual_parameters = sum(value.size for _, value in mlx.tree_flatten(model.parameters())) if native_mlx else parameter_count(model)
            require(f"parameters_{name}", actual_parameters == expected, expected)
            if native_mlx:
                detail = mlx.profile_checks(p, pipeline, profile_id, model, diffusion, encoded)
                require(f"model_input_{name}", detail["model_input_valid"])
            else:
                x = encoded[None].to(device)
                cond = condition([0], device)
                if profile_id == "G3":
                    x = x * 2 - 1
                if getattr(diffusion, "factorized", False):
                    cond = diffusion.payload_condition(cond, diffusion.lengths(x))
                with torch.no_grad():
                    output = model(x, torch.zeros(1, device=device), cond)
                require(f"model_input_{name}", bool(torch.isfinite(output).all()) and x.shape[1:] == shape)
            if v31:
                if not native_mlx:
                    from .study_v31 import profile_checks
                    detail = profile_checks(p, pipeline, profile_id, model, diffusion, device)
                require(f"sampler_{name}", detail["passed"], detail)
                registered[name] = {"parameters": actual_parameters, "profile_id": profile_id, "backend": budget.backend}
            del model, diffusion
    require("clean_fixture_total", count == 1214, count)
    vectors = {b"": "d41d8cd98f00b204e9800998ecf8427e", b"abc": "900150983cd24fb0d6963f7d28e17f72", b"a" * 100: hashlib.md5(b"a" * 100).hexdigest()}
    require("independent_md5", all(MD5Tracer().trace(x)["digest"] == expected == hashlib.md5(x).hexdigest() for x, expected in vectors.items()))
    require("md5_msb12", int.from_bytes(hashlib.md5(b"abc").digest(), "big") >> 116 == 0x900)
    require("condition_leading_zeros", (mlx.models.condition([1]) if native_mlx else condition([1], codec_device)).tolist() == [[0.] * 11 + [1.]])
    try:
        if native_mlx:
            mlx.models.condition([{"y": 1, "length": 9, "message": "hidden"}])
        else:
            condition([{"y": 1, "length": 9, "message": "hidden"}], codec_device)
    except ValueError:
        require("metadata_boundary", True)
    else:
        require("metadata_boundary", False)
    require("trial_rng_identity", seed(p, "P1", "generation", unit_id="1", attempt=1) != seed(p, "P1", "generation", unit_id="2", attempt=1))
    require("synthetic_verifier", verify(b"00f!", "printable", 15) == (True, True) and verify(b"00f!", "printable", 16) == (True, False)
            and verify(b"\x00\x00\x0f\x00", "random_bytes", 15) == (True, True)
            and verify(b"GGG!", "printable", 0) == (True, False))
    require("mcnemar", exact_mcnemar(3, 0) == .125 and exact_mcnemar(0, 0) == 1 and exact_mcnemar(0, 3) == 1)
    family = {str(i): [0.001] * 6 for i in range(5)}
    family["0"][-1] = .9
    family["1"] = None
    adjusted = family_decision(family)
    require("maxp_holm_missing", adjusted["0"] >= .9 and adjusted["1"] == 1 and adjusted["2"] == .005)
    attempts = [False] * 99 + [True]
    require("hundred_attempt_accounting", [any(attempts[:k]) for k in (1, 10, 100)] == [False, False, True] and len(attempts) == 100)
    ci = np.quantile(np.zeros(100), [.025, .975], method="linear")
    require("degenerate_ci", ci.tolist() == [0., 0.])
    connection = ledger_open(directory / "fixture.sqlite")
    try:
        with connection:
            connection.execute("DELETE FROM candidates")
            connection.execute("INSERT INTO candidates VALUES ('fixture','1','normal',1,'{}')")
        try:
            with connection:
                connection.execute("INSERT INTO candidates VALUES ('fixture','1','normal',1,'{}')")
        except sqlite3.IntegrityError:
            require("duplicate_key_rejected", True)
        else:
            require("duplicate_key_rejected", False)
        require("missing_row_detected", connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] != 100)
    finally:
        connection.close()
    if v31:
        require("profile_matrix", len(registered) == p["pilot"]["P0"]["model_profile_pipeline_pairs"])
        atomic_json(directory / "model_profiles.json", registered)
    return {"status": "PASS", "clean_roundtrips": count, "checks": len(checks), "statistics_calibration": "NOT_RUN"}


def equal_tensors(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if type(a).__module__ == "mlx.core":
        return type(b) is type(a) and np.array_equal(np.array(a), np.array(b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tensors(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tensors(x, y) for x, y in zip(a, b))
    return a == b


def logical_ledger(directory):
    with closing(sqlite3.connect(directory / "candidates.sqlite")) as connection:
        rows = [json.loads(row[0]) for row in connection.execute("SELECT record FROM candidates ORDER BY unit_id,variant,attempt")]
    return [{key: value for key, value in row.items() if key not in {"generation_seconds_per_candidate"}} for row in rows]


def recovery_check(p, pipeline, data, directory, reference, model, diffusion, checksum, state, device, budget, batch_size, *, train=True, profile_id=None):
    result_path = directory / "recovery.json"
    if result_path.exists():
        return read_json(result_path)
    if train:
        replay = directory / "training"
        try:
            train_model(p, "P1", pipeline, "main", 0, data, replay, device, budget, pause_update=5, profile_id=profile_id)
        except RecoveryPause:
            pass
        _, _, recovered, _ = train_model(p, "P1", pipeline, "main", 0, data, replay, device, budget, profile_id=profile_id)
        replay_best, _ = load_checkpoint(replay, best=True)
        original_best, _ = load_checkpoint(reference, best=True)
        if not all(equal_tensors(recovered[k], state[k]) for k in ("model", "optimizer", "update", "best_epoch", "best_loss")) or not equal_tensors(replay_best["model"], original_best["model"]):
            raise PilotError(f"Training recovery diverged for {pipeline}")
    targets = stage_targets(p, data, "P1", p["pipelines"][pipeline]["source"])
    baseline = reference if train else directory / "reference"
    if not train:
        evaluate(p, "P1", pipeline, "main", 0, targets, baseline, model, diffusion, checksum, batch_size, device, budget, profile_id=profile_id)
    replay = directory / "generation"
    try:
        evaluate(p, "P1", pipeline, "main", 0, targets, replay, model, diffusion, checksum, batch_size, device, budget, pause_after=7, profile_id=profile_id)
    except RecoveryPause:
        pass
    evaluate(p, "P1", pipeline, "main", 0, targets, replay, model, diffusion, checksum, batch_size, device, budget, profile_id=profile_id)
    if logical_ledger(replay) != logical_ledger(baseline):
        raise PilotError(f"Generation recovery diverged for {pipeline}")
    result = {"status": "PASS", "training_update_5": "PASS" if train else "not_repeated",
              "generation_attempt_7_precommit": "PASS", "batch_size": batch_size,
              "reference_rows": len(logical_ledger(baseline)), "timing_compared": False}
    atomic_json(result_path, result)
    return result


def profile(p, pipeline, model, diffusion, state, data, directory, device, budget):
    cfg = p["execution"]
    source = p["pipelines"][pipeline]["source"]
    pool = [r[0] for r in data["sources"][source]["validation"]]
    encoder, decoder, _ = codecs(p["pipelines"][pipeline])
    rows = []
    for batch in cfg["profile_batch_candidates"]:
        measured, transfers, costs = [], [], []
        try:
            for repetition in range(4):
                budget.check(directory)
                targets = [pool[i % len(pool)] for i in range(batch)]
                seeds = [seed(p, "P2", "profile", source=source, pipeline=pipeline, unit_id=f"batch:{batch}:{repetition}", attempt=i + 1) for i in range(batch)]
                synchronize(device)
                start = time.monotonic()
                values = sample(p, pipeline, model, diffusion, targets, seeds, device, to_cpu=False)
                synchronize(device)
                elapsed = time.monotonic() - start
                transfer_start = time.monotonic()
                values = values.cpu()
                transfer_seconds = time.monotonic() - transfer_start
                decode_start = time.monotonic()
                for y, value in zip(targets, values):
                    decoded = decoder.decode(value) if isinstance(decoder, TokenCodec) else decoder.decode(value, normalized=True)
                    verify(decoded.message, source, y)
                if repetition:
                    measured.append(elapsed)
                    transfers.append(transfer_seconds)
                    costs.append(time.monotonic() - decode_start)
                event(directory, kind="profile", batch=batch, repetition=repetition, candidates=batch,
                      model_seconds=elapsed, cpu_transfer_seconds=transfer_seconds, decode_verify_seconds=time.monotonic() - decode_start)
            rows.append({"batch": batch, "eligible": True, "candidates_per_second": 3 * batch / (sum(measured) + sum(transfers)),
                         "model_seconds_per_candidate": sum(measured) / (3 * batch),
                         "cpu_transfer_seconds_per_candidate": sum(transfers) / (3 * batch),
                         "decode_verify_seconds_per_candidate": sum(costs) / (3 * batch)})
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            rows.append({"batch": batch, "eligible": False, "reason": str(error)})
            if device.type == "mps":
                torch.mps.empty_cache()
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        raise PilotError("No inference batch fits the device", 5)
    fastest = max(r["candidates_per_second"] for r in eligible)
    selected = min((r for r in eligible if r["candidates_per_second"] >= fastest * (1 - cfg["profile_tie_relative_throughput"])), key=lambda r: r["batch"])
    fixture = encoder.encode(bytes([33 if source == "printable" else 255]) * 31)
    start = time.monotonic()
    for _ in range(10):
        decoded = decoder.decode(fixture)
        if not decoded.valid:
            raise PilotError("Maximum-length decode fixture failed", 3)
        verify(decoded.message, source, 0)
        hashlib.md5(decoded.message).digest()
    valid_decode_seconds = (time.monotonic() - start) / 10
    telemetry = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    batches = [row for row in telemetry if row["kind"] == "inference_batch"]
    measured_io = max((max(0, row["total_seconds"] - row["generation_seconds"]) / row["candidates"] for row in batches), default=0.)
    updates = state["training_update_seconds"]
    windows = [sum(updates[start:start + 100]) / 100 for start in (20, 120, 220)]
    if len(updates) < 320:
        raise PilotError("P2 requires the registered 20 warm-up + three 100-update timing windows", 2)
    result = {"grid": rows, "selected_batch": selected["batch"], "update_seconds_windows": windows,
              "update_seconds": sum(windows) / 3, "validation_seconds": sum(state["validation_seconds"]) / len(state["validation_seconds"]),
              "checkpoint_seconds": max(state["checkpoint_seconds"], default=0),
              "candidate_seconds": selected["model_seconds_per_candidate"] + selected["cpu_transfer_seconds_per_candidate"] + max(measured_io, selected["decode_verify_seconds_per_candidate"], valid_decode_seconds),
              "maximum_length_decode_verify_md5_seconds": valid_decode_seconds,
              "ledger_decode_io_seconds_per_candidate": measured_io,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
              "mps_allocated_bytes_at_profile_end": torch.mps.current_allocated_memory() if device.type == "mps" else 0,
              "profile_candidates": 4 * sum(cfg["profile_batch_candidates"])}
    atomic_json(directory / "profile.json", result)
    return result


def seal_resources(p, root, profiles):
    cfg = p["execution"]
    estimates, p3, main = {}, 0., 0.
    for pipeline, row in profiles.items():
        train = p["pilot"]["P3"]["optimizer_updates_per_run"] * row["update_seconds"]
        validation = len(p["pilot"]["P3"]["validation_epochs"]) * row["validation_seconds"]
        io = (p["pilot"]["P3"]["optimizer_updates_per_run"] / 100 + p["pilot"]["P3"]["epochs"]) * row["checkpoint_seconds"]
        pilot_run = (train + validation + io + 1024 * row["candidate_seconds"]) * cfg["time_budget_multiplier"]
        main_run = (train + validation + io + 204800 * row["candidate_seconds"]) * cfg["time_budget_multiplier"]
        estimates[pipeline] = {"batch_size": row["selected_batch"], "p3_run_seconds": pilot_run, "main_run_seconds": main_run}
        p3 += 3 * pilot_run
        main += 6 * main_run
    stored = sum(path.stat().st_size for path in (root / "pilot" / "P2").rglob("*") if path.is_file())
    # Deliberately conservative: retain three P3 seeds and twice measured storage.
    storage = stored * 3 * cfg["storage_budget_multiplier"]
    current_storage = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    ready = p3 <= cfg["hard_stage_active_wall_seconds"]["P3"] and current_storage + storage <= cfg["hard_study_storage_gib"] * GIB
    ready &= all(row["p3_run_seconds"] <= cfg["hard_formal_run_active_wall_seconds"] for row in estimates.values())
    ready &= storage + cfg["minimum_disk_free_gib"] * GIB <= shutil.disk_usage(root).free
    result = {"status": "PASS" if ready else "BLOCKED_RESOURCE", "pipelines": estimates,
              "p3_soft_wall_seconds": math.ceil(p3), "p3_storage_estimate_bytes": storage,
              "main_learned_estimate_seconds": main, "main_ready": False,
              "main_note": "Primary data, analysis calibration, random-baseline cost and main runner remain unvalidated."}
    atomic_json(root / "resources.json", result)
    return result


def seal_directory(directory):
    return {str(path.relative_to(directory)): file_hash(path) for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name not in {"state.json", "complete.json"} and not path.name.endswith((".tmp", "-wal", "-shm"))}


def verify_seal(directory, seal):
    for name, checksum in seal.items():
        path = directory / name
        if not path.is_relative_to(directory) or not path.is_file() or file_hash(path) != checksum:
            raise PilotError(f"Artifact missing or modified: {path}")
        wal = Path(str(path) + "-wal")
        if path.suffix == ".sqlite" and wal.exists() and wal.stat().st_size:
            raise PilotError(f"Sealed SQLite ledger modified through WAL: {path}")


def execute_models(p, stage, root, data, device, budget):
    if p["revision"] == "3.1" and stage == "P2":
        from .study_p2 import execute
        return execute(p, root, data, device, budget)
    cfg = p["pilot"][stage]
    summaries, profiles = {}, {}
    resources = read_json(root / "resources.json") if stage == "P3" else None
    matrix = [(pipeline, None) for pipeline in p["pipeline_order"]]
    if p["revision"] == "3.1":
        from .study_profiles import profile_ids
        if stage != "P1":
            raise PilotError("v3.1 P2/P3 execution is not implemented", 2)
        matrix = [(pipeline, profile_id) for pipeline in p["pipeline_order"] for profile_id in profile_ids(p, pipeline)]
    for pipeline, profile_id in matrix:
        source = p["pipelines"][pipeline]["source"]
        targets = stage_targets(p, data, stage, source)
        for label in cfg["model_seeds"]:
            for method in cfg["methods"]:
                name = f"{pipeline}/{label}/{method}"
                if profile_id is not None:
                    name = f"{pipeline}/{profile_id}/{label}/{method}"
                directory = root / "pilot" / stage / "runs" / name
                complete = directory / "complete.json"
                if complete.exists():
                    saved = read_json(complete)
                    verify_seal(directory, saved["sha256"])
                    summaries[name] = saved["summary"]
                    if stage == "P2":
                        profiles[pipeline] = read_json(directory / "profile.json")
                    continue
                print(f"[{stage}] {name}: training/evaluation", file=sys.stderr, flush=True)
                model, diffusion, state, checkpoint = train_model(p, stage, pipeline, method, label, data, directory, device, budget, profile_id=profile_id)
                batch = cfg.get("inference_batch", cfg.get("probe_inference_batch")) if stage != "P3" else resources["pipelines"][pipeline]["batch_size"]
                metrics = evaluate(p, stage, pipeline, method, label, targets, directory, model, diffusion, checkpoint, batch, device, budget, profile_id=profile_id)
                summary = {"pipeline": pipeline, "method": method, "model_seed": label, "updates": state["update"],
                           "best_epoch": state["best_epoch"], "best_validation_loss": state["best_loss"], "metrics": metrics}
                if profile_id is not None:
                    summary["profile_id"] = profile_id
                atomic_json(directory / "training.json", {"update": state["update"], "best_epoch": state["best_epoch"], "curve": state["curve"],
                                                           "full_loop_update_seconds": state.get("full_loop_update_seconds", []),
                                                           "full_loop_timing_excludes": ["validation", "checkpoint"]})
                if stage == "P1" and method == "main":
                    summary["recovery"] = recovery_check(p, pipeline, data, directory / "recovery", directory, model,
                                                         diffusion, checkpoint, state, device, budget, batch, profile_id=profile_id)
                if stage == "P2":
                    profiles[pipeline] = profile(p, pipeline, model, diffusion, state, data, directory, device, budget)
                    summary["recovery"] = recovery_check(p, pipeline, data, directory / "selected_batch_recovery", directory,
                                                         model, diffusion, checkpoint, state, device, budget,
                                                         profiles[pipeline]["selected_batch"], train=False)
                    summary["warning"] = "LEARNING_SIGNAL_ABSENT" if metrics["normal_joint"] == 0 else None
                if stage == "P3":
                    gate = p["synthetic"]["formal_gate"]
                    summary["qualified"] = (metrics["normal_joint"] >= gate["normal_joint_min"] and
                                            metrics["flipped_joint"] >= gate["flipped_joint_min"] and
                                            metrics["wrong_original"] <= gate["wrong_original_max"])
                atomic_json(complete, {"summary": summary, "sha256": seal_directory(directory)})
                summaries[name] = summary
                del model, diffusion, state
    if stage == "P1":
        for source in ("printable", "random_bytes"):
            directory = root / "pilot" / stage / "random" / source
            targets = stage_targets(p, data, stage, source)
            summaries[f"random/{source}"] = evaluate(p, stage, None, "random", 0, targets, directory, None, None, None,
                                                      cfg["inference_batch"], device, budget, source=source)
    result = {"status": "PASS", "stage": stage, "runs": summaries}
    if stage == "P2":
        result["resources"] = seal_resources(p, root, profiles)
    if stage == "P3":
        qualified = [pipeline for pipeline in p["pipeline_order"] if all(summaries[f"{pipeline}/{label}/main"]["qualified"] for label in cfg["model_seeds"])]
        result.update(qualified_pipelines=qualified, evaluation_complete=True,
                      status="PILOT_QUALIFIED" if len(qualified) == len(p["pipeline_order"]) else "PILOT_PARTIAL_QUALIFICATION" if qualified else "PILOT_EVALUATION_COMPLETE",
                      exit_code=0 if len(qualified) == len(p["pipeline_order"]) else 2)
    return result


def run_pilot(p, args):
    v31 = p["revision"] == "3.1"
    backend = getattr(args, "backend", "torch")
    if backend not in {"torch", "mlx"} or (backend == "mlx" and not v31):
        raise PilotError("MLX requires v3.1; unsupported backend/protocol", 2)
    if v31 and (args.stage not in {"P0", "P1", "P2"} or not args.development):
        raise PilotError("v3.1 supports development P0/P1/P2 only (--development). E0, P3, audit and main/calibration must finish before formal execution.", 2)
    if v31 and args.stage != "P0" and not (args.workdir / "manifest.json").exists():
        raise PilotError("A new Pilot must start with P0", 2)
    if backend == "torch" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise PilotError("Unset PYTORCH_ENABLE_MPS_FALLBACK: v3 forbids silent CPU fallback", 2)
    try:
        if backend == "mlx":
            try:
                from . import mlx_backend as mlx
            except ImportError as error:
                raise PilotError("MLX is not installed. Install the project's [mlx] extra on Apple silicon.", 2) from error
            device = mlx.resolve_device(args.device)
        else:
            device = resolve_device(args.device)
    except (ValueError, RuntimeError) as error:
        raise PilotError(str(error), 2) from error
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_device = ("cpu" if device.type == mlx.mx.cpu else "mps") if backend == "mlx" else device.type
    with lock(root / ".study.lock"), lock(Path(os.environ.get("TMPDIR", "/tmp")) / f"dhi-v3-{os.getuid()}-{lock_device}.lock"):
        manifest_path = root / "manifest.json"
        identity = {"protocol_sha256": digest(p), "environment": environment(device, args.threads), "development": args.development}
        if backend == "mlx":
            from importlib.metadata import version
            identity.update(backend="mlx", rng_scheme="explicit_mlx_key64_v1")
            identity["environment"].update(mlx=version("mlx"), backend="mlx", model_layout="NCHW_interface_NHWC_convolutions")
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["identity"] != identity or digest(read_json(root / "protocol.frozen.json")) != digest(p):
                raise PilotError("Protocol/code/environment/device/development mode changed; exact continuation is not permitted")
        else:
            if args.resume:
                raise PilotError("--resume requires an existing study", 2)
            if args.stage != "P0":
                raise PilotError("A new Pilot must start with P0", 2)
            if any(path.name != ".study.lock" for path in root.iterdir()):
                raise PilotError("Refusing to initialize a nonempty output directory", 2)
            manifest = {"identity": identity, "created_at": time.time(), "commands": [], "data_sha256": None}
            atomic_json(root / "protocol.frozen.json", p)
            atomic_json(root / "gates.json", {})
            atomic_json(root / "analysis_validation.json", {"status": "NOT_RUN", "note": "Pilot CLI does not perform main-study statistical calibration."})
            atomic_json(manifest_path, manifest)
            if v31:
                from .study_v31 import readiness
                atomic_json(root / "implementation_readiness.json", readiness())
        gates = read_json(root / "gates.json")
        for number in range(int(args.stage[1])):
            previous = f"P{number}"
            if gates.get(previous, {}).get("status") != "PASS":
                raise PilotError(f"{args.stage} requires completed {previous}", 2)
            verify_seal(root / "pilot" / previous, gates[previous]["sha256"])
        if args.stage == "P3":
            if not (root / "resources.json").exists() or read_json(root / "resources.json")["status"] != "PASS":
                raise PilotError("P3 blocked by P2 resource estimates", 5)
            if file_hash(root / "resources.json") != gates["P2"]["resources_sha256"]:
                raise PilotError("Resource manifest changed after P2")
        directory = root / "pilot" / args.stage
        state_path = directory / "state.json"
        if state_path.exists():
            state = read_json(state_path)
            if state["status"] == "COMPLETE":
                raise PilotError("Stage already complete; use report or a new study directory", 2)
            if not args.resume:
                raise PilotError("Stage already exists; exact continuation requires --resume", 2)
            if v31 and (state["status"] == "RUNNING" or state.get("exit_code") == 5):
                raise PilotError("Unknown unaccounted time after an abrupt stop, or an exhausted hard budget: v3.1 exact continuation is blocked", 4)
            if state.get("resume_attempts", 0) >= 1 or state.get("exit_code") == 4:
                raise PilotError("Resume allowance exhausted or integrity failure; investigate without replacing results", 4)
            state["resume_attempts"] = state.get("resume_attempts", 0) + 1
        else:
            if args.resume:
                raise PilotError("No interrupted stage exists to resume", 2)
            state = {"status": "RUNNING", "active_seconds": 0., "resume_attempts": 0, "history": []}
        manifest["commands"].append({"argv": sys.argv, "time": time.time(), "stage": args.stage, "resume": args.resume})
        atomic_json(manifest_path, manifest)
        state["status"] = "RUNNING"
        state["history"].append({"status": "RUNNING", "time": time.time()})
        budget = Budget(root, args.stage, state, p, device, backend=backend)
        try:
            budget.check()
            if args.stage == "P0":
                result = preflight(p, directory, device, budget)
            else:
                path = root / "data" / "synthetic.json"
                if manifest["data_sha256"] is None:
                    if path.exists():
                        raise PilotError("Unsealed synthetic dataset exists")
                    data = make_data(p)
                    if v31:
                        budget.reserve(None, len(canonical(data)) + 65536, refresh=True)
                    atomic_json(path, data)
                    manifest["data_sha256"] = file_hash(path)
                    atomic_json(manifest_path, manifest)
                else:
                    if file_hash(path) != manifest["data_sha256"]:
                        raise PilotError("Synthetic data checksum mismatch")
                    data = read_json(path)
                result = execute_models(p, args.stage, root, data, device, budget)
            result["development_only"] = args.development
            result["backend"] = backend
            if args.development and args.stage == "P3":
                result.update(status="DEVELOPMENT_EVALUATION_COMPLETE", qualified_pipelines=[], exit_code=0)
            atomic_json(directory / "summary.json", result)
            state.update(status="COMPLETE", exit_code=result.get("exit_code", 0))
            gates[args.stage] = {"status": result["status"], "development_only": args.development, "sha256": seal_directory(directory)}
            if args.stage == "P2":
                gates[args.stage]["resources_sha256"] = file_hash(root / "resources.json")
                if v31:
                    gates[args.stage]["selection_sha256"] = file_hash(root / "profile_selection.json")
                    gates[args.stage]["profile_sha256"] = file_hash(root / "profile.frozen.json")
            atomic_json(root / "gates.json", gates)
        except BaseException as error:
            state.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "INCOMPLETE",
                         error=str(error), exit_code=error.code if isinstance(error, PilotError) else 130 if isinstance(error, KeyboardInterrupt) else 3)
            state["history"].append({"status": state["status"], "error": str(error), "time": time.time()})
            raise
        finally:
            budget.tick()
            if v31:
                write_report(root, p, acquire_lock=False)
        write_report(root, p, acquire_lock=False)
        return {"stage": args.stage, "status": result["status"], "development_only": args.development,
                "backend": backend, "workdir": str(root), "report": str(root / "report.md"), "exit_code": result.get("exit_code", 0)}


def write_report(root, p, *, acquire_lock=True):
    root = Path(root).resolve()
    if not (root / "manifest.json").exists():
        raise PilotError("No study manifest exists at --workdir", 2)
    if acquire_lock:
        with lock(root / ".study.lock"):
            return write_report(root, p, acquire_lock=False)
    manifest = read_json(root / "manifest.json")
    if manifest["identity"]["protocol_sha256"] != digest(p) or digest(read_json(root / "protocol.frozen.json")) != digest(p):
        raise PilotError("Report protocol mismatch")
    gates = read_json(root / "gates.json")
    v31 = p["revision"] == "3.1"
    lines = [f"# v{p['revision']} Pilot 실행 보고", "", f"Protocol: {p['protocol_id']}",
             f"Development only: {manifest['identity']['development']}",
             f"Backend: {manifest['identity'].get('backend', 'torch')}", "",
             "| 단계 | 실행 상태 | 판정 | Active seconds |", "|---|---|---|---:|"]
    for stage in ("P0", "P1", "P2", "P3"):
        directory = root / "pilot" / stage
        state = read_json(directory / "state.json") if (directory / "state.json").exists() else {}
        if stage in gates:
            verify_seal(directory, gates[stage]["sha256"])
        lines.append(f"| {stage} | {state.get('status', 'NOT_RUN')} | {gates.get(stage, {}).get('status', 'NOT_RUN')} | {state.get('active_seconds', 0):.2f} |")
    if manifest["data_sha256"] is not None and file_hash(root / "data" / "synthetic.json") != manifest["data_sha256"]:
        raise PilotError("Dataset checksum mismatch")
    if "P2" in gates and file_hash(root / "resources.json") != gates["P2"]["resources_sha256"]:
        raise PilotError("Resource checksum mismatch")
    if v31 and "P2" in gates:
        for filename, key in (("profile_selection.json", "selection_sha256"), ("profile.frozen.json", "profile_sha256")):
            if file_hash(root / filename) != gates["P2"][key]:
                raise PilotError("P2 profile selection checksum mismatch")
    if "P3" in gates:
        summary = read_json(root / "pilot" / "P3" / "summary.json")
        lines += ["", "적격 pipeline: " + (", ".join(summary["qualified_pipelines"]) or "없음"), "",
                  "| Pipeline / seed | 정상 joint | 반전 joint | 원래 조건 오성공 | 기준 충족 |", "|---|---:|---:|---:|---|"]
        for name, run in summary["runs"].items():
            m = run["metrics"]
            lines.append(f"| {name} | {m['normal_joint']} | {m['flipped_joint']} | {m['wrong_original']} | {run['qualified']} |")
    progress = {}
    for stage in ("P1", "P2", "P3"):
        directory = root / "pilot" / stage / "runs"
        paths = {path.parent for path in directory.rglob("complete.json")}
        paths |= {path.parent for path in directory.rglob("configuration.json")
                  if not any("recovery" in part for part in path.relative_to(directory).parts)}
        if v31 and stage == "P1":
            from .study_profiles import profile_ids
            paths |= {directory / pipeline / profile_id / str(label) / method
                      for pipeline in p["pipeline_order"] for profile_id in profile_ids(p, pipeline)
                      for label in p["pilot"][stage]["model_seeds"] for method in p["pilot"][stage]["methods"]}
        elif not v31:
            cfg = p["pilot"][stage]
            paths |= {directory / pipeline / str(label) / method for pipeline in p["pipeline_order"]
                      for label in cfg["model_seeds"] for method in cfg["methods"]}
        if not paths:
            continue
        lines += ["", f"## {stage} run 상태 (stage gate와 별도)", "",
                  "| Run | 실행 상태 | 정상 joint | 반전 joint |", "|---|---|---:|---:|"]
        for path in sorted(paths):
            name = str(path.relative_to(directory))
            completed = path / "complete.json"
            metrics = {}
            if completed.exists():
                saved = read_json(completed)
                verify_seal(path, saved["sha256"])
                metrics = saved["summary"]["metrics"]
                status = "COMPLETE"
            else:
                status = "INCOMPLETE" if (path / "configuration.json").exists() else "NOT_RUN"
            progress[f"{stage}/{name}"] = status
            flipped = "NOT_MEASURED" if stage == "P1" else metrics.get('flipped_joint', '—')
            lines.append(f"| {name} | {status} | {metrics.get('normal_joint', '—')} | {flipped} |")
    if v31:
        from .study_p2 import report_rows
        report_rows(root, p, lines, progress)
        from .study_v31 import readiness
        lines += ["", "v3.1 구현: IMPLEMENTATION_IN_PROGRESS. 현재 P0/P1/P2 결과는 개발 검사이며 formal PoC gate가 아니다.",
                  "남은 구현: " + ", ".join(readiness()["remaining"]) + "."]
    lines += ["", "본실험: NOT_IMPLEMENTED / NOT_RUN. 통계 calibration: NOT_RUN.",
              "Pilot 실행 완료와 합성 과제 성능 통과는 별도 판정이다. Development 결과는 v3 GPU 적격성 증거가 아니다.", ""]
    (root / "report.md").write_text("\n".join(lines))
    return {"status": "REPORT_WRITTEN", "report": str(root / "report.md"), "pilot_gates": {key: value["status"] for key, value in gates.items()},
            "main_ready": False, "run_progress": progress, "exit_code": 0}
