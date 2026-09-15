"""CLI for a reproducible baseline, control, or diffusion experiment run."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .runner import ExperimentConfig, run_experiment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one diffusion hash-inverse experiment from JSON config.")
    parser.add_argument("--config", type=Path, required=True, help="ExperimentConfig JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Directory for data, checkpoint, and evaluation")
    arguments = parser.parse_args(argv)
    config = ExperimentConfig(**json.loads(arguments.config.read_text(encoding="utf-8")))
    result = run_experiment(config, arguments.output)
    print(json.dumps({"summary": asdict(result.summary), "output": str(result.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
