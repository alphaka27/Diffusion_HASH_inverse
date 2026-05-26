"""
Command line entry point for hash generation and validation.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from diffusion_hash_inv.config import (
    Byte2RGBConfig,
    HashConfig,
    MainConfig,
    MessageConfig,
    OutputConfig,
)
from diffusion_hash_inv.main import MainEP, RuntimeConfig
from diffusion_hash_inv.utils.ecc48 import SUPPORTED_METHODS


DEFAULT_LENGTH = 256
RGB_ENCODINGS = ("golay24", "legacy-bin", "cube-id", *SUPPORTED_METHODS)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser used by both ``python -m`` and the console script.
    """
    parser = argparse.ArgumentParser(description="Hash generation and image creation")
    parser.add_argument(
        "--hash-alg",
        "--hash_alg",
        dest="hash_alg",
        type=str,
        default="md5",
        help="Hash algorithm to use (default: md5)",
    )

    length_group = parser.add_mutually_exclusive_group()
    length_group.add_argument(
        "-l",
        "--length",
        type=int,
        default=None,
        help=f"Length of input bits to generate (default: {DEFAULT_LENGTH})",
    )
    length_group.add_argument(
        "-e",
        "--exponentiation",
        type=int,
        default=None,
        help="Use 2 to the power of this value as the input bit length",
    )

    parser.add_argument(
        "-i",
        "--iteration",
        type=int,
        default=0,
        help="Number of hash generation iterations (default: 0)",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-m",
        "--message",
        action="store_true",
        dest="message",
        help="Use text message input mode",
    )
    mode_group.add_argument(
        "-b",
        "--bit",
        action="store_false",
        dest="message",
        help="Use bit-string input mode",
    )
    parser.set_defaults(message=False)

    value_group = parser.add_mutually_exclusive_group()
    value_group.add_argument(
        "--random",
        action="store_true",
        dest="random",
        help="Generate random values for each iteration",
    )
    value_group.add_argument(
        "--sequential",
        action="store_true",
        dest="sequential",
        help="Generate deterministic sequential values from the iteration index",
    )
    parser.set_defaults(random=True, sequential=False)

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Enable verbose output",
    )
    parser.add_argument(
        "-c",
        "--clear",
        action="store_true",
        default=False,
        help="Clear generated files before running",
    )
    parser.add_argument(
        "--make-image",
        action="store_true",
        default=False,
        help="Create PNG images and HDF5 tensor shards from generated JSON logs after hashing",
    )
    parser.add_argument(
        "--make-png",
        action="store_true",
        default=False,
        help="Create PNG images from JSON logs",
    )
    parser.add_argument(
        "--make-hdf5",
        action="store_true",
        default=False,
        help="Create HDF5 tensor shards from JSON logs",
    )
    parser.add_argument(
        "--skip-hash-json",
        "--artifacts-only",
        action="store_false",
        dest="run_hash_json",
        default=True,
        help="Skip binary/hash JSON generation and create requested artifacts from existing JSON logs",
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=1,
        help="Number of processes used for PNG/HDF5 conversion (default: 1)",
    )
    parser.add_argument(
        "--rgb-encoding",
        choices=RGB_ENCODINGS,
        default=Byte2RGBConfig.encoding,
        help=f"Byte-to-RGB encoding mode (default: {Byte2RGBConfig.encoding})",
    )
    return parser


def _run_png_from_args(args: argparse.Namespace) -> bool:
    return bool(args.make_image or args.make_png)


def _run_hdf5_from_args(args: argparse.Namespace) -> bool:
    return bool(args.make_image or args.make_hdf5)


def _validate_stage_args(args: argparse.Namespace) -> None:
    if not args.run_hash_json and args.clear:
        raise ValueError("--clear cannot be used with --skip-hash-json because it removes existing JSON logs.")
    if not args.run_hash_json and not _run_png_from_args(args) and not _run_hdf5_from_args(args):
        raise ValueError(
            "At least one stage must run. Use --make-png, --make-hdf5, or --make-image "
            "with --skip-hash-json."
        )


def resolve_length(args: argparse.Namespace) -> int:
    """
    Resolve bit length from mutually exclusive CLI flags.
    """
    if args.length is not None:
        return args.length
    if args.exponentiation is not None:
        return 2 ** args.exponentiation
    return DEFAULT_LENGTH


def config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    """
    Convert parsed CLI arguments into the runtime configuration object.
    """
    _validate_stage_args(args)
    length = resolve_length(args)
    random_flag = bool(args.random and not args.sequential)
    run_artifacts = _run_png_from_args(args) or _run_hdf5_from_args(args)

    return RuntimeConfig(
        main=MainConfig(
            verbose_flag=args.verbose,
            clean_flag=args.clear,
            debug_flag=False,
            make_image_flag=run_artifacts,
            image_workers=args.image_workers,
        ),
        message=MessageConfig(
            message_flag=args.message,
            length=length,
            random_flag=random_flag,
        ),
        hash=HashConfig(
            hash_alg=args.hash_alg,
            length=length,
        ),
        output=OutputConfig(),
        rgb=Byte2RGBConfig(encoding=args.rgb_encoding),
    )


def run_from_args(args: argparse.Namespace) -> None:
    """
    Execute the application using parsed CLI arguments.
    """
    runtime_config = config_from_args(args)
    entry_point = MainEP(runtime_config)
    mode = "sequential" if args.sequential else "default"
    iteration = args.iteration if args.run_hash_json else None
    entry_point.run(
        iteration=iteration,
        mode=mode,
        run_hash_json=args.run_hash_json,
        run_png=_run_png_from_args(args),
        run_hdf5=_run_hdf5_from_args(args),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """
    Console-script compatible main function.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
