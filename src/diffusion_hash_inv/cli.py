"""Command-line interface for creating hash execution traces."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Sequence

from .hashing import trace_hash
from .config.configuration import CHARACTER_GROUPS, select_characters
from .generator.message import generate_message


def _non_negative_length(value: str) -> int:
    length = int(value)
    if length < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return length


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without performing any hash work."""
    parser = argparse.ArgumentParser(
        description="Save a JSON trace of MD5 or SHA-256 intermediate calculations."
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        default="sha256",
        choices=("md5", "sha256", "sha-256"),
        help="hash algorithm to use (default: sha256)",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-text",
        "-t",
        help="text to hash; it is encoded as UTF-8 by default",
    )
    input_group.add_argument(
        "--input-length",
        type=_non_negative_length,
        help="generate this many characters to hash",
    )
    character_group = parser.add_mutually_exclusive_group()
    character_group.add_argument(
        "--characters",
        help="candidate characters for --input-length",
    )
    character_group.add_argument(
        "--character-groups",
        nargs="+",
        choices=tuple(CHARACTER_GROUPS),
        help="named candidate groups for --input-length",
    )
    parser.add_argument(
        "--seed",
        help="make --input-length generation repeatable; omit for CSPRNG output",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="text encoding used for --input-text (default: utf-8)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        help="output root; traces are stored in its trace/ subdirectory (default: output)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and invoke the package's single hash entry point."""
    experiment_started_at = datetime.now().astimezone()
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.seed is not None and arguments.input_length is None:
        parser.error("--seed requires --input-length")
    if (arguments.characters is not None or arguments.character_groups is not None) and arguments.input_length is None:
        parser.error("--characters and --character-groups require --input-length")
    if arguments.input_length is None:
        message = arguments.input_text
    else:
        characters = arguments.characters or (
            select_characters(*arguments.character_groups) if arguments.character_groups else ""
        )
        if not characters:
            parser.error("--input-length requires non-empty --characters or --character-groups")
        message = generate_message(arguments.input_length, characters, seed=arguments.seed)
    output_path = arguments.output / "trace" / (
        f"{arguments.algorithm.replace('-', '')}_{len(message)}_"
        f"{experiment_started_at:%Y%m%dT%H%M%S}.json"
    )
    trace = trace_hash(
        algorithm=arguments.algorithm,
        message=message,
        output_path=output_path,
        encoding=arguments.encoding,
    )
    print(
        json.dumps(
            {"algorithm": trace["algorithm"], "digest": trace["digest"], "output": str(output_path)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
