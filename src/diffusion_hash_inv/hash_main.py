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


DEFAULT_LENGTH = 256


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
        help="Create RGB images from generated JSON logs after hashing",
    )
    return parser


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
    length = resolve_length(args)
    random_flag = bool(args.random and not args.sequential)

    return RuntimeConfig(
        main=MainConfig(
            verbose_flag=args.verbose,
            clean_flag=args.clear,
            debug_flag=False,
            make_image_flag=args.make_image,
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
        rgb=Byte2RGBConfig(),
    )


def run_from_args(args: argparse.Namespace) -> None:
    """
    Execute the application using parsed CLI arguments.
    """
    runtime_config = config_from_args(args)
    entry_point = MainEP(runtime_config)
    mode = "sequential" if args.sequential else "default"
    entry_point.run(iteration=args.iteration, mode=mode)


def main(argv: Sequence[str] | None = None) -> None:
    """
    Console-script compatible main function.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
