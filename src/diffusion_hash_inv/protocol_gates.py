"""Fail-closed new-protocol G0/G2 checks, separate from legacy artifact schemas."""
import hashlib
import json
from pathlib import Path

from .dataset import DigestRecord, split_validation_report
from .experiment_state import sha256
from .evaluation import EVALUATOR_VERSION, VERIFIER_VERSION


def validate_dataset(root, frozen_config):
    root, frozen_config = Path(root), Path(frozen_config)
    errors = []
    if not frozen_config.is_file():
        return {"gate": "G0", "status": "FAIL", "errors": ["missing frozen configuration"]}
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        if sha256(frozen_config) != manifest["frozen_config_sha256"]:
            errors.append("frozen configuration checksum")
        if sha256(root / "records.jsonl") != manifest["records_sha256"]:
            errors.append("dataset checksum")
        partitions = {name: [] for name in ("train", "validation", "test")}
        for line in (root / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            message = bytes.fromhex(row["message_hex"])
            digest = hashlib.new(row["algorithm"], message).digest()
            if digest.hex() != row["full_digest"]:
                errors.append("dataset digest differs from independent rehash")
            if (row["source"], row["algorithm"], row["q"]) != (manifest["source"], manifest["algorithm"], manifest["q"]):
                errors.append("manifest setting mismatch")
            if not manifest["l_min"] <= len(message) <= manifest["l_max"]:
                errors.append("source length outside manifest")
            if row["source"] == "printable" and any(not 0x21 <= value <= 0x7e for value in message):
                errors.append("source alphabet violation")
            record = DigestRecord(row["id"], row["source"], message, row["algorithm"], row["q"], digest)
            if record.prefix != row["prefix"] or row["length"] != len(message):
                errors.append("record metadata mismatch")
            partitions[row["split"]].append(record)
        report = split_validation_report(partitions)
        if not report["passed"] or any(not records for records in partitions.values()):
            errors.append("empty or overlapping split")
        if report["record_counts"] != manifest["counts"]:
            errors.append("message quotas mismatch")
        for records in partitions.values():
            if len({r.message for r in records}) != len(records) or len({r.id for r in records}) != len(records):
                errors.append("duplicate source/id within split")
        return {"gate": "G0", "status": "FAIL" if errors else "PASS", "errors": sorted(set(errors)),
                "scope": manifest["scope"], "audit": report}
    except (KeyError, ValueError, OSError, TypeError) as error:
        return {"gate": "G0", "status": "FAIL", "errors": [str(error)]}


MATCHED_FIELDS = ("source", "algorithm", "q", "k", "l_min", "l_max", "condition_type",
                  "verifier_version", "validity_rule", "evaluator_version", "dataset_id")


def validate_comparison(model_manifest, baseline_manifest, model_ledger, baseline_ledger):
    errors = [key for key in MATCHED_FIELDS if key not in model_manifest or key not in baseline_manifest
              or model_manifest[key] != baseline_manifest[key]]
    for manifest, ledger in ((model_manifest, model_ledger), (baseline_manifest, baseline_ledger)):
        if manifest.get("verifier_version") != VERIFIER_VERSION or manifest.get("evaluator_version") != EVALUATOR_VERSION:
            errors.append("unsupported verifier/evaluator")
        targets, k = manifest.get("target_order", []), manifest.get("k", 0)
        if not targets or len(set(targets)) != len(targets) or k not in {1, 10, 100}:
            errors.append("target order or K")
            continue
        expected = [(key, position) for key in targets for position in range(1, k + 1)]
        actual = [(row["target_prefix"], row["k_position"]) for row in ledger]
        if actual != expected:
            errors.append("target ordering / actual candidate budget")
        if any(row.get("algorithm") != manifest["algorithm"] or row.get("q") != manifest["q"] for row in ledger):
            errors.append("ledger setting mismatch")
    if model_manifest.get("target_order") != baseline_manifest.get("target_order"):
        errors.append("paired target ordering")
    return {"gate": "G2", "status": "FAIL" if errors else "PASS", "errors": sorted(set(errors))}
