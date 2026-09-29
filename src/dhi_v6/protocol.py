"""등록값과 JSON 봉인. 공통 함수는 dhi_v5/protocol.py에서 복사했다."""
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
import time
import sys

from . import MASTER_SEED, PROTOCOL


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
    return {
        'adam': {'betas': [0.9, 0.999], 'bias_correction': True, 'eps': 1e-08, 'weight_decay': 0},
        'backend': 'mlx',
        'budget': {'look2_safety': 1.5, 'next_block_safety': 1.2},
        'calibration_repetitions': 2000,
        'caps_gib': {'disk': 64, 'free': 20, 'gpu': 64, 'rss': 64},
        'caps_hours': {'A': 24,
                       'A_repair': 12,
                       'C': 80,
                       'P': 10,
                       'R': 36,
                       'S': 10,
                       'required': 114,
                       'required_with_repair': 126},
        'clp': {'common_draws_within_pair': True, 'draws': 8, 'gaussian_t': 'uniform-integer-0-999'},
        'contrast_alpha': 0.05,
        'contrasts': [['P-G-BGV', 'P-G-CGGE'], ['P-G-BGV', 'P-DISC'], ['P-G-CGGE', 'P-DISC'], ['R-G-BGV', 'R-DISC'],
                      ['P-G-BGV', 'R-G-BGV'], ['P-DISC', 'R-DISC']],
        'decoders': {'bgv': 'nearest-source-byte-prototype-on-4x4-block-means-tie-smallest',
                     'cgge': 'nearest-glyph-prototype-mse-tie-smallest',
                     'strict_diagnostic': {'bgv_bit_threshold': 0.5, 'cgge_max_mse': 0.1},
                     'tokens': 'payload-states-only'},
        'dev_selection': {'default_steps': 100,
                          'duplicate_tolerance': 0.005,
                          'joint_min': 231,
                          'of': 256,
                          'rule': 'smallest-steps-passing-joint-both-variants-and-duplicates-within-tolerance-of-100'},
        'dtype': 'float32',
        'generation': {'batch_multiple': 64, 'batch_options': [256, 1024, 2048]},
        'groups': {'test': 1024, 'train': 2816, 'validation': 256},
        'hash_gate': {'messages_per_source': 100000, 'rungs': [4, 5, 6, 7, 8, 10, 12, 16, 32, 64]},
        'implementation': 'independent-v6',
        'k': 100,
        'ledger': {'commit_batches': 8,
                   'commit_seconds': 30,
                   'record_bytes': 36,
                   'regeneration_batch': 64,
                   'regeneration_fraction': 0.01},
        'length_max': 31,
        'length_min': 4,
        'master_seed': MASTER_SEED,
        'models': {'D1-S': {'embedding': 16,
                            'grad_clip': None,
                            'hidden': 128,
                            'intervals': 32,
                            'lr': 0.001,
                            'nfe': 33,
                            'parameters': {'P': 508668, 'R': 1252572},
                            'warmup': 0},
                   'D1-T-L': {'dim': 256,
                              'ffn': 1024,
                              'grad_clip': 1.0,
                              'heads': 8,
                              'intervals': 32,
                              'layers': 8,
                              'lr': 0.00075,
                              'nfe': 33,
                              'parameters': 6415308,
                              'warmup': 1000},
                   'G3-U': {'beta_end': 0.02,
                            'beta_start': 0.0001,
                            'condition_output': True,
                            'coordinates': True,
                            'diffusion_steps': 1000,
                            'grad_clip': 1.0,
                            'loss': 'length-ce-plus-uniform-payload-glyph-mse',
                            'lr': 0.001,
                            'parameters': {'bgv': 910638, 'cgge': 513326},
                            'prediction': 'x0',
                            'sampler': 'ddim-eta0',
                            'sampling_steps_options': [25, 50, 100],
                            'warmup': 1000,
                            'width': 32,
                            'x0_clip': [-1, 1]}},
        'pipeline_order': ['P-G-BGV', 'P-G-CGGE', 'P-DISC', 'R-G-BGV', 'R-DISC'],
        'pipelines': {'P-DISC': {'model': 'D1-S', 'representation': 'tokens', 'source': 'P'},
                      'P-G-BGV': {'model': 'G3-U', 'representation': 'bgv', 'source': 'P'},
                      'P-G-CGGE': {'model': 'G3-U', 'representation': 'cgge', 'source': 'P'},
                      'R-DISC': {'model': 'D1-S', 'representation': 'tokens', 'source': 'R'},
                      'R-G-BGV': {'model': 'G3-U', 'representation': 'bgv', 'source': 'R'}},
        'profiling': {'burst_seconds': 60,
                      'last_window_seconds': 300,
                      'measure_seconds': 900,
                      'tie_relative': 0.01,
                      'train_timed_updates': 50,
                      'train_warmup_updates': 10,
                      'warmup_seconds': 600},
        'protocol': PROTOCOL,
        'qualification': {'acceptance': 512,
                          'clp_pairs': 4096,
                          'clp_z': 3.26,
                          'joint_min': 461,
                          'seeds': [0, 1, 2],
                          'valid': 512,
                          'wrong_max': 25},
        'retry_limit': 1,
        'sources': {'P': {'byte_max': 126, 'byte_min': 33, 'states': 94},
                    'R': {'byte_max': 255, 'byte_min': 0, 'states': 256}},
        'stage_c': {'alpha': 0.05,
                    'alpha_share': [0.1, 0.4, 0.5],
                    'block': 8192,
                    'clp_pairs_per_seed': 65536,
                    'delta': 0.005,
                    'fallback_block': 6144,
                    'family': {'controls': 2, 'pipelines': 5, 'sides': 2},
                    'info_alpha': 0.01,
                    'looks': 3,
                    'seeds': [0, 1, 2]},
        'stage_p': {'alpha': 0.01,
                    'clp_pairs': 16384,
                    'fallback_trials': 2048,
                    'rung': 4,
                    'seed': 0,
                    'trials': 4096,
                    'window': 'W1'},
        'stage_r': {'alpha_one_sided': 0.025, 'seeds': [0, 1, 2], 'trials': 16384, 'window': 'W4'},
        'stage_s': {'alpha_one_sided': 0.0005,
                    'clp_pairs': 65536,
                    'model': 'D1-T-L',
                    'optional': True,
                    'pipeline': 'P-DISC',
                    'rung': 64,
                    'seed': 0,
                    'trials': 4096,
                    'updates': 160000,
                    'window': 'W3'},
        'synthetic': {'P': 'uppercase-ASCII-hex-prefix-3',
                      'R': 'nibble-byte-prefix-3',
                      'acceptance': 512,
                      'dev': 256,
                      'diversity_candidates': 100,
                      'diversity_conditions': 64},
        'temperature': 1,
        'tokens': {'P': {'eos': 95, 'mask': 96, 'pad': 94, 'vocab': 97},
                   'R': {'eos': 257, 'mask': 258, 'pad': 256, 'vocab': 259}},
        'training': {'batch': 256,
                     'checkpoint_every': 4000,
                     'diagnostic_clp_pairs': 256,
                     'remediation_updates': 80000,
                     'scale_updates': 160000,
                     'updates': 40000},
        'window_roles': {'forbidden': ['W2'], 'positive_control': 'W1', 'primary': 'W3', 'replication': 'W4'},
        'windows': {'W1': 116, 'W2': 0, 'W3': 52, 'W4': 84},
    }


def source_manifest():
    root = Path(__file__).parent
    return {p.name: file_hash(p) for p in sorted(root.glob("*.py"))}


def environment():
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "mlx")}}



@contextmanager
def study_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is writing this V6 study") from error
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
        caps["caps_hours"] = effective_caps(self.root)
        required = sum(seconds.get(s, 0) for s in ("A", "A_repair", "C", "P"))
        if seconds[self.stage] > 3600 * caps["caps_hours"][self.stage] or required > 3600 * caps["caps_hours"]["required_with_repair" if seconds.get("A_repair", 0) else "required"]:
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

    def flush(self):
        """Persist elapsed time when a stage session ends, so the next Budget starts from it."""
        now = time.monotonic()
        self.state["seconds"][self.stage] = self.state["seconds"].get(self.stage, 0) + now - self.last
        self.last = self.persisted = now
        atomic_json(self.path, self.state)


def effective_caps(root):
    caps = registration()["caps_hours"]
    path = Path(root) / "cap-override.json"
    if path.exists():
        value = read_json(path)
        caps[value["stage"]] = value["hours"]
    return caps


def approve_caps(root, stage, hours, reason):
    root = Path(root)
    if stage != "C" or not isinstance(hours, (int, float)) or not 0 < hours < float("inf") or not reason.strip():
        raise ValueError("A finite positive C cap and a reason are required")
    if not (root / "halt.json").exists() or (root / "protocol.frozen.json").exists():
        raise ValueError("Cap approval requires HALT before freeze")
    if any((root / s).exists() for s in ("C", "R", "P", "S")):
        raise ValueError("Caps cannot change after MD5 condition data exist")
    result = {"stage": stage, "hours": hours, "reason": reason, "approved_at": time.time()}
    sealed_json(root / "cap-override.json", result)
    return result


def budget_plan(profile1, profile2, qualification, a_seconds, *, caps=None, repair_used=False):
    caps = caps or registration()["caps_hours"]
    q = qualification["Q"]
    sources = {registration()["pipelines"][p]["source"] for p in q}
    settings = qualification["settings"]
    train = sum(6 * settings[p]["updates"] * profile1["models"][p]["update_seconds"] for p in q)
    clp = sum(3 * 65536 * 32 / profile1["models"][p]["forward_rows_per_second"] for p in q)
    def costs(block):
        generated = sum(6 * block * 100 / profile2["models"][p]["sustained_cps"] for p in q)
        generated += len(sources) * 3 * block * 100 / profile2["random_cps"]
        rows = (6 * len(q) + 3 * len(sources)) * block * 100
        verify = rows / profile1["verify_rows_per_second"]
        regen = .01 * generated
        return {"train_C": train, "gen_block": generated, "verify_block": verify, "regen_block": regen,
                "clp_C": clp, "look2": train + 2 * (generated + regen + verify) + clp,
                "worst_C": train + 3 * (generated + regen + verify) + clp}
    estimates = {str(b): costs(b) for b in (8192, 6144)}
    block = next((b for b in (8192, 6144) if 1.5 * estimates[str(b)]["look2"] <= caps["C"] * 3600), None)
    if block is None:
        return {"decision": "HALT", "estimates": estimates, "required_c_hours": 1.5 * estimates["6144"]["look2"] / 3600}
    def p_cost(trials):
        train_p = sum(settings[p]["updates"] * profile1["models"][p]["update_seconds"] for p in q)
        gen_p = sum(2 * trials * 100 / profile2["models"][p]["sustained_cps"] for p in q)
        gen_p += len(sources) * trials * 100 / profile2["random_cps"]
        verify = (2 * len(q) + len(sources)) * trials * 100 / profile1["verify_rows_per_second"]
        clp_p = sum(16384 * 32 / profile1["models"][p]["forward_rows_per_second"] for p in q)
        return train_p + 1.01 * gen_p + verify + clp_p
    required_cap = caps["required_with_repair" if repair_used else "required"] * 3600
    reduce_p = a_seconds + estimates[str(block)]["worst_C"] + p_cost(4096) > required_cap
    return {"decision": "PROCEED", "block": block, "p_trials": 2048 if reduce_p else 4096,
            "stage_s": not reduce_p, "pipelines": settings, "caps": caps, "estimates": estimates,
            "predicted_p_seconds": p_cost(2048 if reduce_p else 4096), "a_seconds": a_seconds,
            "required_cap_seconds": required_cap, "repair_used": repair_used}


AUDIT_SUFFIXES = {".py", ".json", ".jsonl", ".sqlite", ".db", ".csv", ".md", ".npy", ".npz", ".bin"}


def inventory_files(base, scopes):
    folders, files = [], set()
    for paths in scopes.values():
        for item in paths:
            folder = (Path(base) / item).resolve()
            if not folder.is_dir():
                raise ValueError(f"Missing audit scope: {folder}")
            folders.append(folder)
            files.update(p.resolve() for p in folder.rglob("*") if p.is_file() and p.suffix in AUDIT_SUFFIXES
                         and "__pycache__" not in p.parts)
    return folders, files


def audit_exposure(path):
    from .data import WINDOWS, hash_one, valid
    path = Path(path).resolve()
    inventory = read_json(path)
    if inventory.get("schema") != "v6-exposure-1":
        raise ValueError("Unsupported exposure inventory")
    scopes = inventory.get("scopes", {})
    if not inventory.get("scope_complete") or not scopes.get("code") or not scopes.get("archive"):
        return {"certified": False, "reason": "INCOMPLETE_SCOPE"}
    folders, files = inventory_files(path.parent, scopes)
    rows = inventory.get("reviewed_files", [])
    entries = {(path.parent / row["path"]).resolve(): row for row in rows}
    if len(entries) != len(rows) or set(entries) != files:
        return {"certified": False, "reason": "UNREVIEWED_FILES", "missing": sorted(str(p) for p in files - set(entries)),
                "extra": sorted(str(p) for p in set(entries) - files)}
    excluded = {w: set() for w in WINDOWS}
    excluded["W2"] = set(range(4096))
    def add(groups):
        for window, values in groups.items():
            if window not in WINDOWS or any(type(g) is not int or not 0 <= g < 4096 for g in values):
                raise ValueError("Invalid exposure group")
            excluded[window].update(values)
    add(inventory.get("exposed_groups", {}))
    for file, row in sorted(entries.items()):
        if row.get("sha256") != file_hash(file):
            raise ValueError(f"Audit evidence changed: {file}")
        use = row.get("condition_use")
        if use not in {"prefix", "full", "raw", "toy", "none", "window"}:
            return {"certified": False, "reason": "UNREVIEWED_CLASSIFICATION", "path": str(file)}
        add(row.get("exposed_groups", {}))
        for role in row.get("window_roles", []):
            if role.get("window") not in WINDOWS or role.get("role") not in {"evaluation", "training", "validation", "fixture", "hash-test"}:
                raise ValueError("Invalid window role")
            if role.get("rung") == 64 and role["role"] in {"evaluation", "training", "validation"}:
                excluded[role["window"]].update(range(4096))
        if use in {"full", "raw"}:
            if not row.get("representatives"):
                raise ValueError("Full/raw exposure needs representative messages")
            representative = (path.parent / row["representatives"]).resolve()
            if representative not in files:
                raise ValueError("Representative evidence must be reviewed within scope")
            messages = read_json(representative)
            if not isinstance(messages, list) or not messages:
                raise ValueError("Empty representative evidence")
            for message_hex in messages:
                message = bytes.fromhex(message_hex)
                if not valid(message, "R"):
                    raise ValueError("Invalid exposure representative")
                for window in WINDOWS:
                    excluded[window].add(hash_one(message, window=window))
    return {"certified": all(len(excluded[w]) <= 3072 for w in ("W3", "W4")),
            "excluded": {w: sorted(values) for w, values in excluded.items()}, "primary": "W3", "replication": "W4",
            "sources": ["P", "R"], "inventory_sha256": file_hash(path), "scopes": scopes,
            "scope_boundary": inventory.get("scope_boundary", ""),
            "evidence": [{"path": str(p), "sha256": row["sha256"]} for p, row in sorted(entries.items())]}


def inventory_draft(v5_path):
    path = Path(v5_path).resolve()
    old = read_json(path)
    if old.get("schema") != "v5-exposure-1":
        raise ValueError("Expected V5 inventory")
    scopes = {name: [str((path.parent / p).resolve()) for p in values] for name, values in old["scopes"].items()}
    scopes["studies"] = [str(path.parent.parent / "local_experiment_archive/runs" / name)
                         for name in ("v5-study", "v5-study-certified")]
    _, files = inventory_files(path.parent, scopes)
    previous = {(path.parent / row["path"]).resolve(): row for row in old.get("reviewed_files", [])}
    studies = [Path(p).resolve() for p in scopes["studies"]]
    reviewed = []
    for file in sorted(files):
        digest = file_hash(file)
        if file in previous and previous[file]["sha256"] == digest:
            row = {**previous[file], "path": str(file)}
            if row.get("representatives"):
                row["representatives"] = str((path.parent / row["representatives"]).resolve())
        else:
            row = {"path": str(file), "sha256": digest, "condition_use": "unreviewed"}
            relative = next((file.relative_to(folder) for folder in studies if file.is_relative_to(folder)), None)
            if relative is not None:
                if "C" in relative.parts:
                    row.update(condition_use="window", window_roles=[{"window": "W2", "rung": 64, "role": role}
                                                                      for role in ("evaluation", "training")])
                elif any(p in relative.parts for p in ("A-Q", "A-dev")) or file.name == "profile-ledger.sqlite":
                    row["condition_use"] = "none"
                elif file.name == "A-impl.json":
                    row.update(condition_use="window", window_roles=[{"window": "W2", "rung": 64, "role": "fixture"}]
                        + [{"window": w, "rung": 64, "role": "hash-test"} for w in ("W1", "W2", "W3")])
        reviewed.append(row)
    return {"schema": "v6-exposure-1", "scope_complete": False,
            "scope_boundary": "V5 범위와 V5 study 두 root를 포함한 초안. 사람이 범위와 미검토 항목을 확인해야 한다.",
            "scopes": scopes, "reviewed_files": reviewed, "exposed_groups": old.get("exposed_groups", {})}


def freeze(root):
    root = Path(root)
    if (root / "protocol.frozen.json").exists():
        return verify_frozen(root)
    gate = read_json(root / "A-impl.json")
    if not gate.get("passed") or gate.get("quick") or gate.get("source") != source_manifest():
        raise ValueError("Current source requires formal A-impl PASS")
    calibration = gate.get("gates", {}).get("G9", {})
    if not calibration.get("production") or not calibration.get("passed") or calibration.get("repetitions", 0) < 2000:
        raise ValueError("Regression-only calibration cannot certify the study")
    plan = read_json(root / "budget-plan.json")
    if plan["decision"] != "PROCEED":
        raise ValueError("Budget HALT must be resolved before freeze")
    audit = read_json(root / "exposure-audit.json")
    if not audit.get("certified"):
        raise ValueError("W3/W4 exposure certification required")
    window = {"primary": "W3", "replication": "W4", "forbidden": ["W2"], "audit_sha256": file_hash(root / "exposure-audit.json")}
    sealed_json(root / "window.json", window)
    names = ["A-impl.json", "A-prof-1.json", "A-prof-2.json", "A-dev.json", "A.json", "budget-plan.json", "exposure-audit.json", "window.json"]
    for optional in ("A-dev-repair.json", "cap-override.json"):
        if (root / optional).exists():
            names.append(optional)
    for name in names:
        sealed_json(root / name, read_json(root / name))
    qualification = read_json(root / "A.json")
    frozen = {"registration": registration(), "source": source_manifest(), "environment": environment(),
              "artifacts": {name: file_hash(root / name) for name in names}, "Q": qualification["Q"],
              "settings": plan, "window": window, "caps": effective_caps(root)}
    sealed_json(root / "protocol.frozen.json", frozen)
    sealed_json(root / "protocol.seal.json", {"sha256": file_hash(root / "protocol.frozen.json")})
    return frozen


def verify_frozen(root):
    root = Path(root)
    frozen = read_json(root / "protocol.frozen.json")
    if frozen["registration"] != registration() or frozen["source"] != source_manifest() or frozen["environment"] != environment():
        raise ValueError("Frozen protocol/code/environment changed; resume refused")
    if read_json(root / "protocol.seal.json")["sha256"] != file_hash(root / "protocol.frozen.json"):
        raise ValueError("Modified protocol JSON")
    for name, digest in frozen["artifacts"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Frozen artifact changed: {name}")
    if frozen["caps"] != effective_caps(root):
        raise ValueError("Frozen caps changed")
    return frozen
