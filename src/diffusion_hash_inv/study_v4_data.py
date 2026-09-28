"""Frozen v4 contract, explicit seed namespaces, exposure audit and split ownership."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .study_pilot import atomic_json, canonical, digest, file_hash, prior, read_json, synthetic_message

PROTOCOL_ID = "dhi-v4-decision-20260928"
METHODS = ("main", "shuffled", "random")
SCOPES = ("local_runs", "external_runs", "q8_raw_or_full_digest", "q_ge_12",
          "development_rehearsals", "cross_source_reuse")
PROTOCOL = {
    "protocol_id": PROTOCOL_ID, "revision": "4", "master_seed": 2026092804,
    "source": {"name": "printable", "min_byte": 33, "max_byte": 126, "min_length": 4, "max_length": 31},
    "hash": "md5_full_digest_first_12_bits_big_endian_payload_only",
    "backend": "mlx", "device": "gpu", "precision": "float32", "concurrent_runs": 1,
    "model": {"width": 128, "embedding_dim": 16, "factorized": True, "condition_output": True,
              "prefix_balanced_loss": False, "sampling_steps": 32, "nfe": 33, "temperature": 1},
    "training": {"batch_size": 64, "learning_rate": .001, "betas": [.9, .999], "eps": 1e-8,
                 "weight_decay": 0, "validation_draws": 4, "checkpoint_every": 100},
    "V1": {"seeds": [0, 1, 2], "train_groups": 3072, "validation_groups": 512,
           "test_groups": 512, "train_messages": 10000, "epochs": 100, "validation_every": 10,
           "trials": 512, "k": 1, "joint_min": 461, "wrong_max": 25},
    "E0": {"seeds": [99], "train_groups": 64, "validation_groups": 32, "test_groups": 32,
           "train_messages": 256, "epochs": 2, "validation_every": 1, "trials": 32, "k": 100},
    "MAIN": {"seeds": [0, 1, 2], "train_groups": 1536, "validation_groups": 512,
             "test_groups": 1024, "reserve_groups": 1024, "train_messages": 10000,
             "epochs": 100, "validation_every": 10, "trials": 16384, "k": 100},
    "draw_cap": 1000000, "historical_exposure_lower_bound": 1885,
    "statistics": {"delta": .01, "tail": .05 / 24, "repetitions": 20000},
    "resources": {"preparation_seconds": 86400, "main_seconds": 259200, "total_seconds": 345600,
                  "run_seconds": 86400, "storage_gib": 64, "rss_gib": 64, "gpu_gib": 64,
                  "disk_free_gib": 10, "time_multiplier": 1.5, "storage_multiplier": 2,
                  "batches": [1, 4, 16, 64], "warmup_updates": 20, "measurement_updates": 100,
                  "measurement_windows": 3, "max_resumes_per_run": 1},
    "seed_fields": ["protocol_id", "master_seed", "task", "namespace", "method", "model_seed", "epoch", "trial", "attempt"],
    "seed_conversion": "UTF8 compact JSON array; SHA256 first 8 bytes big endian unsigned; Random/NumPy integer and MLX random.key(uint64)",
}


def load_protocol(path=None):
    p = read_json(path) if path else PROTOCOL
    if digest(p) != digest(PROTOCOL):
        raise ValueError("v4 fixed protocol mismatch; a changed design requires a new implementation revision")
    return p


def seed(p, task, namespace, *, method=None, label=None, epoch=None, trial=None, attempt=None):
    fields = [p["protocol_id"], p["master_seed"], task, namespace, method, label, epoch, trial, attempt]
    return int.from_bytes(hashlib.sha256(json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode()).digest()[:8], "big")


def h12(message):
    return int.from_bytes(hashlib.md5(message).digest(), "big") >> 116


def valid_source(message):
    return message is not None and 4 <= len(message) <= 31 and all(33 <= b <= 126 for b in message)


def exposure_template():
    return {"schema": 1, "reviewer": "", "reviewed_scopes": {scope: False for scope in SCOPES},
            "unresolved": ["Inventory local/external evaluations, including missing artifacts, before sealing."],
            "entries": [], "entry_example": {
                "id": "replace-with-run-id", "scope": "local_runs", "purpose": "evaluation",
                "source": "printable", "prefixes12": [], "evidence": [{"path": "/absolute/evidence.json", "sha256": ""}],
                "notes": "Include selection/development and linked cross-source exposure; q>=12 projects to leading 12 bits."}}


def audit_exposure(inventory):
    """Completeness is an explicit human attestation, never inferred from file absence."""
    if not isinstance(inventory, dict) or not isinstance(inventory.get("reviewer", ""), str):
        raise ValueError("exposure inventory must be an object with a string reviewer")
    if (not isinstance(inventory.get("reviewed_scopes", {}), dict)
            or not isinstance(inventory.get("entries", []), list)
            or not isinstance(inventory.get("unresolved", []), list)
            or any(not isinstance(x, str) for x in inventory.get("unresolved", []))):
        raise ValueError("invalid exposure inventory schema")
    issues = list(inventory.get("unresolved", []))
    if inventory.get("schema") != 1 or not inventory.get("reviewer", "").strip():
        issues.append("schema 1 and a named reviewer required")
    if any(inventory.get("reviewed_scopes", {}).get(scope) is not True for scope in SCOPES):
        issues.append("all six local/external exposure scopes must be reviewed")
    # These fixed integrity fixtures request MD5 targets; always exclude them from MAIN.
    fixture_groups = ownership(PROTOCOL, "E0", range(128))
    fixture_exclusions = sorted({17, h12(b"!!!!"), *fixture_groups["validation"], *fixture_groups["test"]})
    exposed, ids, evidence_hashes = set(fixture_exclusions), set(), {}
    historical = set()
    for entry in inventory.get("entries", []):
        if not isinstance(entry, dict):
            raise ValueError("inventory entries must be objects")
        eid = entry.get("id")
        if not isinstance(eid, str) or not eid or eid in ids:
            issues.append("missing/duplicate inventory entry id")
            continue
        ids.add(eid)
        prefixes = entry.get("prefixes12")
        if not isinstance(prefixes, list) or any(type(y) is not int or not 0 <= y < 4096 for y in prefixes):
            issues.append(f"{eid}: explicit integer prefixes12 required; unresolved q8 data blocks the audit")
            continue
        if entry.get("scope") not in SCOPES or entry.get("purpose") not in {"evaluation", "selection", "development", "training_only", "unrelated_hash"}:
            issues.append(f"{eid}: unresolved scope/purpose")
        if entry.get("source") not in {"printable", "cross_source_linked", "unrelated_source"}:
            issues.append(f"{eid}: unresolved source")
        if not entry.get("notes") or not entry.get("evidence"):
            issues.append(f"{eid}: evidence and explanation required (including unavailable remote runs)")
        if not isinstance(entry.get("evidence", []), list):
            raise ValueError("evidence must be an array")
        for evidence in entry.get("evidence", []):
            if not isinstance(evidence, dict) or not isinstance(evidence.get("path"), str):
                raise ValueError("evidence requires a string path and sha256")
            path = Path(evidence.get("path", ""))
            if not path.is_absolute() or not path.is_file() or file_hash(path) != evidence.get("sha256"):
                issues.append(f"{eid}: missing/changed evidence {path}")
            else:
                evidence_hashes[str(path)] = evidence["sha256"]
        if entry.get("source") in {"printable", "cross_source_linked"} and entry.get("purpose") in {"evaluation", "selection", "development"}:
            exposed.update(prefixes)
            historical.update(prefixes)
    if len(historical) < PROTOCOL["historical_exposure_lower_bound"]:
        issues.append("inventory falls below the documented 1885 historical exposure lower bound")
    if len(exposed) < 128:
        issues.append("E0 needs at least 128 previously exposed groups")
    if 4096 - len(exposed) < 1024:
        issues.append("fewer than 1024 unexposed groups remain")
    return {"status": "BLOCKED_EXPOSURE" if issues else "PASS", "issues": issues,
            "exposed_prefixes12": sorted(exposed), "available_count": 4096 - len(exposed),
            "inventory_sha256": digest(inventory), "evidence_sha256": evidence_hashes,
            "builtin_fixture_exclusions": fixture_exclusions,
            "completeness_basis": "named reviewer attestation plus evidence checks, not automatic discovery"}


def ownership(p, task, exposed):
    cfg = p[task]
    rng = random.Random(seed(p, task, "ownership"))
    if task == "V1":
        pairs = [(y, y ^ 4095) for y in range(2048)]
        rng.shuffle(pairs)
        values = [y for pair in pairs for y in pair]
        a, b = cfg["train_groups"], cfg["validation_groups"]
        return {"train": values[:a], "validation": values[a:a+b], "test": values[a+b:]}
    if task == "E0":
        values = sorted(exposed)
        rng.shuffle(values)
        if len(values) < sum(cfg[key] for key in ("train_groups", "validation_groups", "test_groups")):
            raise ValueError("BLOCKED_EXPOSURE: E0 exposed pool too small")
        a, b, c = (cfg[key] for key in ("train_groups", "validation_groups", "test_groups"))
        return {"train": values[:a], "validation": values[a:a+b], "test": values[a+b:a+b+c]}
    available = sorted(set(range(4096)) - set(exposed))
    rng.shuffle(available)
    test = available[:cfg["test_groups"]]
    if len(test) != cfg["test_groups"]:
        raise ValueError("BLOCKED_EXPOSURE: insufficient fresh holdout")
    remaining = sorted(set(range(4096)) - set(test))
    random.Random(seed(p, task, "remaining-ownership")).shuffle(remaining)
    a, b = cfg["train_groups"], cfg["validation_groups"]
    return {"train": remaining[:a], "validation": remaining[a:a+b], "reserve": remaining[a+b:], "test": test}


def make_data(p, task, directory, exposed=(), check=lambda: None):
    """Only this constructor handles representatives; trainers receive train.json/validation.json."""
    directory = Path(directory)
    if (directory / "seal.json").exists():
        verify_data(directory)
        return read_json(directory / "ownership.json")
    directory.mkdir(parents=True, exist_ok=True)
    cfg, groups = p[task], ownership(p, task, exposed)
    owners = {y: name for name, values in groups.items() for y in values}
    if len(owners) != sum(map(len, groups.values())):
        raise ValueError("group ownership overlap")
    rng = random.Random(seed(p, task, "data"))
    train, representatives, seen = [], {"validation": {}, "test": {}}, set()
    counts, lengths, characters = Counter(), Counter(), Counter()
    for draw in range(p["draw_cap"]):
        if draw % 1000 == 0:
            check()
        if task == "V1":
            # Synthetic pairs are jointly owned; validation/acceptance have separate suffixes.
            y = rng.choice(groups["train"])
            message = synthetic_message(y, "printable", rng)
        else:
            message = prior("printable", rng)
            y = h12(message)
        counts["draws"] += 1
        if message in seen:
            counts["duplicates"] += 1
            continue
        seen.add(message)
        owner = owners.get(y, "reject")
        if owner == "train" and len(train) < cfg["train_messages"]:
            train.append([y, message.hex()])
            lengths[len(message)] += 1
            characters.update(message)
        elif owner in representatives and y not in representatives[owner]:
            representatives[owner][y] = message.hex()
        else:
            counts[owner + "_surplus_or_reject"] += 1
        if task == "V1" and len(train) == cfg["train_messages"]:
            for name in representatives:
                for target in groups[name]:
                    representatives[name][target] = synthetic_message(target, "printable", rng).hex()
            break
        if len(train) == cfg["train_messages"] and all(len(representatives[k]) == len(groups[k]) for k in representatives):
            break
    if len(train) != cfg["train_messages"] or any(len(representatives[k]) != len(groups[k]) for k in representatives):
        atomic_json(directory / "failure.json", {"status": "INVALID_OR_INCOMPLETE", "counts": dict(counts)})
        raise ValueError("data draw cap exhausted")
    validation = [[y, representatives["validation"][y]] for y in groups["validation"]]
    test = [[y, representatives["test"][y]] for y in groups["test"]]
    raw_sets = [set(row[1] for row in rows) for rows in (train, validation, test)]
    if any(raw_sets[i] & raw_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("raw split overlap")
    for name, value in (("ownership.json", groups), ("train.json", train), ("validation.json", validation),
                        ("evaluator/test_representatives.json", test), ("targets.json", groups["test"]),
                        ("audit.json", {"counts": dict(counts), "train_length_counts": dict(lengths),
                                        "train_character_counts": dict(characters), "raw_overlap": 0, "group_overlap": 0,
                                        "train_distribution": "source prior conditional on train group ownership"})):
        if (directory / name).exists() and read_json(directory / name) != value:
            raise ValueError("partial data artifact differs from deterministic reconstruction")
        atomic_json(directory / name, value)
    files = {str(path.relative_to(directory)): file_hash(path) for path in directory.rglob("*.json")}
    atomic_json(directory / "seal.json", files)
    return groups


def verify_data(directory):
    for name, checksum in read_json(directory / "seal.json").items():
        if not (directory / name).resolve().is_relative_to(directory.resolve()) or file_hash(directory / name) != checksum:
            raise ValueError("data checksum mismatch")


def trial_schedule(p, task, directory, checkpoint_seal):
    if not checkpoint_seal:
        raise ValueError("checkpoint seal is required before drawing trials")
    path = directory / "trials.json"
    identity = {"checkpoints": digest(checkpoint_seal), "targets": file_hash(directory / "data/targets.json")}
    pool = read_json(directory / "data/targets.json")
    rng = random.Random(seed(p, task, "trials"))
    targets = pool if task == "V1" else [rng.choice(pool) for _ in range(p[task]["trials"])]
    value = {"identity": identity, "targets": targets, "replacement": task != "V1"}
    if path.exists():
        if read_json(path) != value:
            raise ValueError("trial schedule differs from its deterministic sealed draw")
    else:
        atomic_json(path, value)
    return targets
