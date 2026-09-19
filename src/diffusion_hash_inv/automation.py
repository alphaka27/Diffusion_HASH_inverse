"""New-study immutable snapshots and restart integrity; legacy state stays untouched."""
import argparse
import datetime
import json
import os
import tempfile
from pathlib import Path
import hashlib

from .experiment_state import write_json, sha256

STATE_PATH = Path("artifacts/automation/EXPERIMENT_STATE.json")


def source_version():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for file in sorted(root.rglob("*.py")):
        digest.update(str(file.relative_to(root)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def freeze_json(path, value):
    """Atomically create a snapshot; identical replay is OK, replacement is forbidden."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise RuntimeError(f"immutable snapshot differs: {path}; use a new run ID")
        return sha256(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic create-if-absent, never replaces a file.
    finally:
        os.unlink(temporary)
    return sha256(path)


def verify_completed(state):
    checked = 0
    for job in state["completed_jobs"]:
        if not job.get("artifacts"):
            raise RuntimeError(f"completed job without checksummed evidence: {job['job_id']}")
        for name, checksum in job["artifacts"].items():
            if not Path(name).is_file() or sha256(name) != checksum:
                raise RuntimeError(f"completed artifact changed or missing: {name}")
            checked += 1
    for item in state["config_snapshots"]:
        if sha256(item["path"]) != item["sha256"]:
            raise RuntimeError(f"configuration changed: {item['path']}")
    return checked


def complete_job(job_id, artifacts, *, code_version, config_checksum, state_path=STATE_PATH):
    state = json.loads(Path(state_path).read_text())
    verify_completed(state)
    existing = next((job for job in state["completed_jobs"] if job["job_id"] == job_id), None)
    if existing:
        if existing["code_version"] != code_version or existing["config_checksum"] != config_checksum:
            raise RuntimeError("incompatible code/configuration; use a new job ID")
        return False
    paths = list(artifacts)
    if not paths:
        raise ValueError("completed job needs artifacts")
    state["completed_jobs"].append(dict(job_id=job_id, status="COMPLETED", code_version=code_version,
                                        config_checksum=config_checksum,
                                        artifacts={str(path): sha256(path) for path in paths}))
    state["pending_jobs"] = [job for job in state["pending_jobs"] if job["job_id"] != job_id]
    state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(state_path, state)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", required=True)
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    checked = verify_completed(state)
    print(json.dumps({"status": "PASS", "artifact_hash_checks": checked,
                      "current_phase": state["current_phase"], "blocked_jobs": len(state["blocked_jobs"]),
                      "pending_jobs": len(state["pending_jobs"])}))


if __name__ == "__main__":
    main()
