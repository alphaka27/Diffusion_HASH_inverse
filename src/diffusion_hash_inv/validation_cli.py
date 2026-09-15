"""CLI for aggregating frozen experiment runs into one G0--G4 decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .validation import write_confirmatory_validation


def _baseline_runs(values: Sequence[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise argparse.ArgumentTypeError("--baseline-run must be NAME=PATH")
        result.setdefault(name, []).append(Path(path))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write one paired G0--G4 validation report from frozen runs.")
    parser.add_argument("--model-run", action="append", type=Path, required=True, help="Model run directory; repeat for seeds 0, 1, 2")
    parser.add_argument("--baseline-run", action="append", required=True, help="Baseline run as NAME=PATH; repeat for every seed")
    parser.add_argument("--positive-control", type=Path, required=True, help="Seed-0 reversible-record model-control run")
    parser.add_argument("--output", type=Path, required=True, help="JSON validation report path")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    try:
        baselines = _baseline_runs(arguments.baseline_run)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    result = write_confirmatory_validation(
        arguments.output,
        arguments.model_run,
        baselines,
        positive_control_run=arguments.positive_control,
        bootstrap_seed=arguments.bootstrap_seed,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
