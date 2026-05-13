"""
Unified command line interface for diffusion_hash_inv.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from diffusion_hash_inv import hash_main


CommandHandler = Callable[[list[str]], None]


def _preparse_device(argv: Sequence[str]) -> str:
    for idx, arg in enumerate(argv):
        if arg == "--device" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--device="):
            return arg.split("=", 1)[1]
    return "cpu"


def _configure_mlx_device(argv: Sequence[str]) -> None:
    device = _preparse_device(argv)
    import mlx.core as mx

    if device == "cpu":
        mx.set_default_device(mx.cpu)
    elif device == "gpu":
        mx.set_default_device(mx.gpu)
    else:
        raise ValueError(f"Unsupported MLX device: {device}")


def _run_hash(argv: list[str]) -> None:
    hash_main.main(argv)


def _run_mlx_toy(argv: list[str]) -> None:
    _configure_mlx_device(argv)
    from diffusion_hash_inv.models import diffusion_with_mlx

    diffusion_with_mlx.main(argv)


def _run_mlx_conditional(argv: list[str]) -> None:
    _configure_mlx_device(argv)
    from diffusion_hash_inv.models import conditional_diffusion_mlx

    conditional_diffusion_mlx.main(argv)


COMMANDS: dict[str, tuple[str, CommandHandler]] = {
    "hash": ("Run hash generation and validation", _run_hash),
    "mlx-toy": ("Run the synthetic MLX conditional DDPM demo", _run_mlx_toy),
    "mlx-conditional": (
        "Train the MLX conditional DDPM on generated hash images",
        _run_mlx_conditional,
    ),
}


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the top-level CLI parser.
    """
    parser = argparse.ArgumentParser(
        prog="diffhash",
        description="Unified CLI for hash generation and diffusion model experiments.",
        epilog=(
            "Use 'diffhash <command> --help' for command-specific options. "
            "Legacy hash options are still accepted directly, e.g. 'diffhash -i 10 -l 128'."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    for command, (help_text, _) in COMMANDS.items():
        subparsers.add_parser(command, help=help_text)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """
    Console-script compatible entry point.
    """
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help"):
        build_arg_parser().print_help()
        return

    command = args[0]
    if command in COMMANDS:
        _, handler = COMMANDS[command]
        handler(args[1:])
        return

    if command.startswith("-"):
        _run_hash(args)
        return

    valid_commands = ", ".join(sorted(COMMANDS))
    raise SystemExit(f"Unknown command: {command}\nValid commands: {valid_commands}")


__all__ = ["build_arg_parser", "main"]


if __name__ == "__main__":
    main()
