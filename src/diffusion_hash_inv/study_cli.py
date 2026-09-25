"""Executable, gated v3 Pilot entry point (not the legacy ExperimentConfig CLI)."""
import argparse
import json
from pathlib import Path
import sys


def parser():
    result = argparse.ArgumentParser(description="v3 Pilot P0–P3 and Pilot reporting; main-study commands are not implemented.")
    commands = result.add_subparsers(dest="command", required=True)
    pilot = commands.add_parser("pilot", help="execute one Pilot stage, checking prerequisites")
    report = commands.add_parser("report", help="verify and summarize existing Pilot artifacts")
    for command in (pilot, report):
        command.add_argument("--protocol", type=Path, required=True, help="supported v3 protocol JSON (not ExperimentConfig)")
        command.add_argument("--workdir", type=Path, required=True, help="study output directory; use the same directory for P0–P3")
    pilot.add_argument("--stage", required=True, choices=("P0", "P1", "P2", "P3"))
    pilot.add_argument("--device", default="mps", choices=("mps", "cpu"), help="explicit backend; cpu requires --development")
    pilot.add_argument("--development", action="store_true", help="development-only namespace; never confers v3 GPU qualification")
    pilot.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads, positive integer; frozen for continuation (default: 1)")
    pilot.add_argument("--resume", action="store_true", help="one exact continuation of an interrupted stage; never overwrite a completed stage")
    pilot.add_argument("--dry-run", action="store_true", help="validate protocol and print workload/requirements without GPU access or writes")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    from .study_pilot import PilotError, load_protocol, plan, run_pilot, write_report
    try:
        protocol = load_protocol(args.protocol)
        if args.command == "report":
            result = write_report(args.workdir, protocol)
        else:
            if args.threads < 1:
                raise PilotError("--threads must be positive", 2)
            if args.device == "cpu" and not args.development:
                raise PilotError("CPU execution requires --development and cannot qualify a v3 GPU Pilot", 2)
            result = plan(protocol, args.stage, args.device, args.development) if args.dry_run else run_pilot(protocol, args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return result.get("exit_code", 0)
    except PilotError as error:
        print(json.dumps({"status": "BLOCKED" if error.code == 2 else "FAILED", "error": str(error), "exit_code": error.code}, ensure_ascii=False), file=sys.stderr)
        return error.code
    except KeyboardInterrupt:
        print("Interrupted; stage state was preserved. Use --resume with identical arguments.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "FAILED", "error": str(error), "exit_code": 3}), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
