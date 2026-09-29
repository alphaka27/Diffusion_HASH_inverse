"""Frozen registration, evidence-backed exposure audit and resource accounting."""
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

from . import PROTOCOL, MASTER_SEED
from .data import LADDER, hash_one, identity, valid


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    path = Path(path)
    seal = path.with_name(path.name + ".sha256")
    if seal.exists() and seal.read_text().strip() != file_hash(path):
        raise ValueError(f"Sealed JSON was modified: {path}")
    return json.loads(path.read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sealed_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Refusing to change sealed artifact: {path}")
    else:
        atomic_json(path, value)
    seal = path.with_name(path.name + ".sha256")
    if not seal.exists():
        seal.write_text(file_hash(path) + "\n")


def registration():
    return {"protocol": PROTOCOL, "master_seed": MASTER_SEED, "backend": "mlx", "dtype": "float32",
            "implementation": "independent-v5", "sampler_gate": "fresh-scalar-reference-4096-and-batch-invariance",
            "synthetic_encoding": "uppercase-ASCII-hex-prefix-3", "updates": 40000, "batch": 256,
            "qualification_remediation_updates": 160000, "scale_updates": 160000,
            "intervals": 32, "temperature": 1, "nfe": 33, "k": 100,
            "trials_c": 65536, "trials_b": 16384, "trials_mc": 16384, "trials_s": 4096,
            "seeds": [0, 1, 2], "extension_seeds": [3, 4, 5], "ladder": list(LADDER),
            "groups": {"test": 1024, "validation": 256, "train": 2816},
            "delta": .0025, "checkpoint_every": 4000, "clp_draws": 8,
            "clp_c": 65536, "clp_b": 32768, "clp_s": 65536, "clp_a": 4096,
            "dev_updates": 10000, "dev_seeds": [0, 1], "dev_lrs": [.001, .0003, .0001],
            "dev_selection": "lowest-mean-final-validation-objective-tie-in-listed-order",
            "sampling_batches": [256, 1024, 4096], "r_mix": 12,
            "caps_hours": {"A": 8, "C": 16, "C_extension": 16, "R": 16, "B": 16, "S": 10, "required": 50},
            "caps_gib": {"disk": 64, "rss": 64, "gpu": 64, "free": 10},
            "fallback_order": ["halve_b_trials", "halve_s_updates", "omit_upper_replication", "reduce_c_mc", "omit_r32", "prefer_d1s"],
            "retry_limit": 1, "calibration_repetitions": 20000}


def source_manifest():
    root = Path(__file__).parent
    return {p.name: file_hash(p) for p in sorted(root.glob("*.py"))}


def environment():
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "mlx")}}


def verify_frozen(root):
    root = Path(root)
    frozen = read_json(root / "protocol.frozen.json")
    if frozen["registration"] != registration() or frozen["source"] != source_manifest() or frozen["environment"] != environment():
        raise ValueError("Frozen protocol/code/environment changed; resume refused")
    seal = read_json(root / "protocol.seal.json")
    if seal["sha256"] != file_hash(root / "protocol.frozen.json"):
        raise ValueError("Modified protocol JSON")
    for filename, digest in frozen["artifacts"].items():
        if file_hash(root / filename) != digest:
            raise ValueError(f"Frozen artifact changed: {filename}")
    return frozen


def freeze(root, profile, dev, audit, fallback):
    root = Path(root)
    if not audit.get("certified"):
        raise ValueError("Window exposure certification is incomplete")
    primary = audit["primary"]
    replication = "W2" if primary == "W1" else "W3"
    sealed_json(root / "window.json", {"primary": primary, "replication": replication, "audit": file_hash(root / "exposure-audit.json")})
    files = ("A-impl.json", "A-prof.json", "A-dev.json", "fallback.json", "exposure-audit.json", "window.json")
    frozen = {"registration": registration(), "source": source_manifest(), "environment": environment(),
              "artifacts": {p: file_hash(root / p) for p in files}, "profile": profile,
              "lr": dev["lr"], "fallback": fallback, "window": read_json(root / "window.json")}
    sealed_json(root / "protocol.frozen.json", frozen)
    sealed_json(root / "protocol.seal.json", {"sha256": file_hash(root / "protocol.frozen.json")})
    return frozen


def audit_exposure(inventory_path):
    """Fail closed: every file in each declared scope must be reviewed by digest."""
    path = Path(inventory_path).resolve()
    inventory = read_json(path)
    if inventory.get("schema") != "v5-exposure-1":
        raise ValueError("Unsupported exposure inventory")
    scopes = inventory.get("scopes", {})
    if not scopes.get("code") or not scopes.get("archive") or not inventory.get("scope_complete"):
        return {"certified": False, "reason": "INCOMPLETE_SCOPE", "primary": "W2"}
    files = set()
    scope_paths = []
    for group in ("code", "archive"):
        for item in scopes[group]:
            folder = (path.parent / item).resolve()
            if not folder.is_dir():
                raise ValueError(f"Missing audit scope: {folder}")
            scope_paths.append(folder)
            files.update(p.resolve() for p in folder.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".jsonl", ".sqlite", ".db", ".csv", ".md") and "__pycache__" not in p.parts)
    reviewed = inventory.get("reviewed_files", [])
    entries = {(path.parent / x["path"]).resolve(): x for x in reviewed}
    if len(entries) != len(reviewed) or set(entries) != files:
        return {"certified": False, "reason": "UNREVIEWED_FILES", "primary": "W2",
                "missing": sorted(str(p) for p in files - set(entries)), "extra": sorted(str(p) for p in set(entries) - files)}
    excluded = {w: set() for w in ("W1", "W2", "W3")}
    evidence = []
    for file, entry in sorted(entries.items()):
        if entry.get("sha256") != file_hash(file):
            raise ValueError(f"Audit evidence changed: {file}")
        use = entry.get("condition_use")
        if use not in ("prefix", "full", "raw", "toy", "none"):
            raise ValueError(f"Unclassified condition path: {file}")
        if use in ("full", "raw"):
            representative_path = (path.parent / entry["representatives"]).resolve()
            if not any(representative_path.is_relative_to(p) for p in scope_paths):
                raise ValueError("Representative evidence lies outside reviewed scope")
            representatives = read_json(representative_path)
            if not isinstance(representatives, list) or not representatives:
                raise ValueError("Full/raw digest exposure requires validation/test representative payloads")
            for payload_hex in representatives:
                payload = bytes.fromhex(payload_hex)
                if not valid(payload):
                    raise ValueError("Invalid exposure representative")
                for window in excluded:
                    excluded[window].add(hash_one(payload, window=window))
        for window, groups in entry.get("exposed_groups", {}).items():
            if window not in excluded or any(type(g) is not int or not 0 <= g < 4096 for g in groups):
                raise ValueError("Invalid exposed group")
            excluded[window].update(groups)
        evidence.append({"path": str(file), "sha256": entry["sha256"], "condition_use": use})
    w1_complete = inventory.get("w1_exposure_complete") is True
    primary = "W1" if w1_complete and len(excluded["W1"]) <= 3072 else "W2"
    needed = (primary, "W2" if primary == "W1" else "W3")
    certified = all(len(excluded[w]) <= 3072 for w in needed)
    return {"certified": certified, "primary": primary, "excluded": {w: sorted(x) for w, x in excluded.items()},
            "scope": scopes, "inventory_sha256": file_hash(path), "evidence": evidence,
            "w1_exposure_complete": w1_complete}


@contextmanager
def study_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is writing this V5 study") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, root, stage):
        self.root, self.stage = Path(root), stage
        self.path = self.root / "budget.json"
        self.state = read_json(self.path) if self.path.exists() else {"seconds": {}, "sessions": []}
        # A killed session consumes wall time through its last persisted heartbeat.
        self.last = time.monotonic()
        self.persisted = -float("inf")
        self.last_storage = -float("inf")
        self.storage = 0
        self.state["sessions"].append({"stage": stage, "started": time.time()})
        self.check()

    def check(self):
        now = time.monotonic()
        seconds = self.state["seconds"]
        seconds[self.stage] = seconds.get(self.stage, 0) + now - self.last
        self.last = now
        if now - self.persisted >= 1:
            atomic_json(self.path, self.state)
            self.persisted = now
        caps = registration()
        required = sum(seconds.get(s, 0) for s in ("A", "C", "B", "S"))
        if seconds[self.stage] > 3600 * caps["caps_hours"][self.stage] or required > 3600 * caps["caps_hours"]["required"]:
            raise BudgetExceeded("Registered wall-time cap exhausted")
        gib = 1024**3
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
        if rss > caps["caps_gib"]["rss"] * gib:
            raise BudgetExceeded("RSS cap exhausted")
        if shutil.disk_usage(self.root).free < caps["caps_gib"]["free"] * gib:
            raise BudgetExceeded("Minimum free disk violated")
        if "mlx.core" in sys.modules:
            mx = sys.modules["mlx.core"]
            if mx.get_active_memory() > caps["caps_gib"]["gpu"] * gib:
                raise BudgetExceeded("MLX active-memory cap exhausted")
        if now - self.last_storage > 30:
            self.storage = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
            self.last_storage = now
        if self.storage > caps["caps_gib"]["disk"] * gib:
            raise BudgetExceeded("Study storage cap exhausted")


def fallback_for(profile):
    settings = {"b_trials": 16384, "s_updates": 160000, "upper_replication": True, "c_mc_trials": 16384,
                "ladder": list(LADDER), "prefer_d1s": False}
    applied = []

    def estimate():
        arch = "D1-S" if settings["prefer_d1s"] else "D1-T"
        p = profile["models"][arch]
        large = profile["models"]["D1-T-L"]
        update, cps = p["update_seconds"], p["candidates_per_second"]
        random_cps = profile["prior_md5_per_second"]
        train_io = profile["fresh_batch_seconds"]
        ledger = profile["ledger_verified_rows_per_second"]
        b_runs = len(settings["ladder"]) + (4 if settings["upper_replication"] else 2)
        c_rows = 3 * (3 * 65536 + settings["c_mc_trials"]) * 100
        b_rows = b_runs * 3 * settings["b_trials"] * 100
        s_rows = 3 * 4096 * 100
        c = 6*40000*(update+train_io) + 3*(2*65536+settings["c_mc_trials"])*100/cps + 3*65536*100/random_cps + c_rows/ledger
        b = b_runs*40000*(update+train_io) + b_runs*2*settings["b_trials"]*100/cps + b_runs*settings["b_trials"]*100/random_cps + b_rows/ledger
        s = 2*settings["s_updates"]*(large["update_seconds"]+train_io) + 2*4096*100/large["candidates_per_second"] + 4096*100/random_cps + s_rows/ledger
        # Include the measured forward-only probes and all validation diagnostics.
        clp_c = 3*65536*32 / p["forward_rows_per_second"]
        c += clp_c
        b += b_runs*32768*32 / p["forward_rows_per_second"]
        s += 2*65536*32 / large["forward_rows_per_second"]
        estimates = {"C": c/3600, "B": b/3600, "S": s/3600}
        estimates["required"] = profile["a_estimated_hours"] + sum(estimates.values())
        estimates["disk_gib"] = (c_rows+b_rows+s_rows)*profile["bytes_per_ledger_row"] / 1024**3 + profile["training_storage_gib"]
        return estimates

    for action in [None, *registration()["fallback_order"]]:
        if action:
            applied.append(action)
            if action == "halve_b_trials": settings["b_trials"] = 8192
            elif action == "halve_s_updates": settings["s_updates"] = 80000
            elif action == "omit_upper_replication": settings["upper_replication"] = False
            elif action == "reduce_c_mc": settings["c_mc_trials"] = 4096
            elif action == "omit_r32": settings["ladder"].remove(32)
            elif action == "prefer_d1s": settings["prefer_d1s"] = True
        estimates = estimate()
        if all(1.5 * estimates[s] <= registration()["caps_hours"][s] for s in ("C", "B", "S", "required")) and estimates["disk_gib"] * 1.5 <= 64:
            break
    return {"settings": settings, "applied": applied, "estimates": estimates,
            "fits_caps": all(1.5*estimates[s] <= registration()["caps_hours"][s] for s in ("C", "B", "S", "required")) and estimates["disk_gib"]*1.5 <= 64,
            "d1s_requires_qualification": settings["prefer_d1s"]}
