"""
Generate 512-bit random number and save it as binary file
"""

from secrets import randbits
import argparse
import math

from typing import Optional

from diffusion_hash_inv.config import MainConfig, OutputConfig
from diffusion_hash_inv.logger import Logs
from diffusion_hash_inv.utils import FileIO, bytes_to_binary_block, bytes_to_hex_block


DEFAULT_BIT_LENGTH = 512
DEFAULT_EXPONENT = 9


class GenerateRandomNBits:
    """
    Generate a random number of specified bit length.
    """
    def __init__(
        self,
        verbose_flag: bool = True,
        clean_flag: bool = False,
        start_timestamp: Optional[int] = None,
        file_io: Optional[FileIO] = None,
    ):
        self.is_verbose = verbose_flag
        self.start_time = start_timestamp
        self.file_io = file_io or FileIO(
            MainConfig(
                verbose_flag=verbose_flag,
                clean_flag=clean_flag,
                debug_flag=False,
                make_image_flag=False,
            ),
            OutputConfig(),
        )
        if self.is_verbose:
            print(f"Flags - Verbose: {verbose_flag}\n")

    @staticmethod
    def print_hex(msg, x):
        """
        Print the hexadecimal representation of the given bytes.
        """
        if msg.endswith("\n"):
            print(msg, end="")
        else:
            print(msg+":")
        print(bytes_to_hex_block(x))
        print()

    @staticmethod
    def print_bin(msg, data):
        """
        Print the binary representation of the given integer.
        """
        if msg.endswith("\n"):
            print(msg, end="")
        else:
            print(msg+":")
        print(bytes_to_binary_block(data))
        print()

    def main(self, length: int = 512, \
            timestamp: Optional[str] = None) -> bytes:
        """
        Generate a random 512-bit number and return its hexadecimal and binary representations.
        """
        timer_start = self.start_time or Logs.perftimer_start()
        timestamp = timestamp or Logs.get_current_timestamp()
        _n = randbits(length)
        _length = math.ceil(length / 8)
        bytes_n = _n.to_bytes(_length, byteorder='big', signed=False)
        elapsed_time = Logs.perftimer_end(timer_start)

        assert len(bytes_n) == _length, "Binary length does not match specified length."

        if self.is_verbose:
            print(f"Generated {length}-bit random number.")
            print(f"Generated at: {timestamp}")
            self.print_bin("Data in Binary", bytes_n)
            print(f"Binary length in Bytes: \n{len(bytes_n)}\n") # type: str
        self.print_hex(f"Data in Hexadecimal (len: {len(bytes_n)}bytes)", bytes_n)
        filename = f"random_{length}_bits_{timestamp}.bin"
        self.file_io.file_writer(
            filename,
            bytes_n,
            length=length,
            timestamp=timestamp,
            elapsed_time=elapsed_time,
            byteorder="big",
        )

        return bytes_n


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for random bit generation.
    """
    parser = argparse.ArgumentParser(description="Generate random bits and save to a binary file.")
    parser.add_argument('-l', '--length', type=int, default=None,
                        help=f'Length of random bits to generate (default: {DEFAULT_BIT_LENGTH})')

    parser.add_argument('-i', '--iterations', type=int,
                        default=1, help='Number of iterations to run (default: 1)')

    parser.add_argument('-e', '--exponentiation', type=int, default=None,
                        help=f'2 to the power of <exponentiation> (default: {DEFAULT_EXPONENT})')

    gv = parser.add_mutually_exclusive_group()
    gv.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                    help='Enable verbose output')
    gv.add_argument('-q', '--quiet', action='store_false', dest='verbose',
                    help='Suppress output')
    parser.set_defaults(verbose=True)

    gc = parser.add_mutually_exclusive_group()
    gc.add_argument('-c', '--clear', action='store_true',
                    dest='clear', help='Clear generated files')
    gc.add_argument('-C', '--no-clear', action='store_false', dest='clear',
                    help='Do not clear generated files (default)')
    parser.set_defaults(clear=False)
    return parser


def resolve_bit_length(args: argparse.Namespace) -> int:
    """
    Resolve bit length from parsed CLI arguments.
    """
    if args.length is not None:
        return args.length
    if args.exponentiation is not None:
        return 2 ** args.exponentiation
    return DEFAULT_BIT_LENGTH


def main(argv: Optional[list[str]] = None) -> None:
    """
    Console entry point for random bit generation.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bit_len = resolve_bit_length(args)
    generator = GenerateRandomNBits(args.verbose, clean_flag=args.clear)

    for idx in range(args.iterations):
        print(f"Iteration: {idx + 1}")
        generator.main(bit_len)
        print()


if __name__ == "__main__":
    main()
