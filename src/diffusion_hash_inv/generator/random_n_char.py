"""
Random N character Generation
Password Generator
"""

import unicodedata
from secrets import choice
import random
import argparse
import string
from typing import Optional, ClassVar

from diffusion_hash_inv.config import MainConfig, OutputConfig
from diffusion_hash_inv.utils import FileIO
from diffusion_hash_inv.logger import Logs


DEFAULT_BIT_LENGTH = 512
DEFAULT_EXPONENT = 9


class GenerateRandomNChar:
    """
    Generate a random string of N characters.
    """
    alphabet: ClassVar[str]

    def __init__(self, main_config: MainConfig, file_io: FileIO) -> None:
        self.main_config = main_config
        self.file_io = file_io
        self.is_verbose = self.main_config.verbose_flag
        self.start_time = Logs.get_current_timestamp()
        main_config_values = object.__getattribute__(self.main_config, "__dict__")
        self.seed_flag = main_config_values.get("seed_flag", True)
        self.seed = main_config_values.get("seed", None)

        type(self).alphabet = string.ascii_letters \
            + string.digits + string.punctuation + " "

    def help(self):
        """
        Provide help information for the password generator.
        """
        description = "Generate a random string of N characters\n"
        alphabet_info = (
            "Includes uppercase, lowercase, digits, punctuation, and space.\n" + 
            f"Alphabet List Length: {len(self.alphabet)}\n"
        )
        alphabet_list = f"Alphabet List: {self.alphabet}"
        print(description + alphabet_info + alphabet_list, end="\n\n")

    def generate(self, length: int) -> str:
        """
        Generate a random string of N characters.
        """
        if self.seed_flag:
            _pwd = ''.join(choice(GenerateRandomNChar.alphabet) for _ in range(length))
        else:
            rng = random.Random(self.seed)
            _pwd = ''.join(rng.choice(GenerateRandomNChar.alphabet) for _ in range(length))

        return _pwd

    def normalize(self, s: str, form: str = "NFKC") -> bytes:
        """
        Normalize a string to the specified Unicode normalization form.
        """
        assert form in ["NFKC", "NFKD", "NFC", "none"], "Invalid normalization form"
        if form != "none":
            s = unicodedata.normalize(form.upper(), s)
        return s.encode("utf-8")

    def main(self, length: int = 256, \
        byteorder: Optional[str] = None, \
        timer_start: Optional[int] = None) -> bytes:
        """
        Main function to generate random strings and display their entropy.
        """
        assert byteorder is not None and byteorder in ("big", "little"), \
            "byteorder must be 'big' or 'little'"

        if timer_start is None:
            raise ValueError("timer_start must be provided")

        _pwd = self.generate(length // 8)
        _pwd = self.normalize(_pwd)
        elapsed_time = Logs.perftimer_end(timer_start)
        if self.is_verbose:
            self.help()
        filename = f"random_{length}_char_{self.start_time[:19]}.char"

        self.file_io.file_writer(filename, _pwd, length=length, timestamp=self.start_time, \
            elapsed_time=elapsed_time, byteorder=byteorder)

        return _pwd


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for random character generation.
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
    Console entry point for random character generation.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bit_len = resolve_bit_length(args)

    main_config = MainConfig(
        verbose_flag=args.verbose,
        clean_flag=args.clear,
        debug_flag=False,
        make_image_flag=False,
    )
    file_io = FileIO(main_config, OutputConfig())
    pw_gen = GenerateRandomNChar(main_config, file_io)

    for _ in range(args.iterations):
        timer = Logs.perftimer_start()
        print(f"Iteration: {_ + 1}")
        pw_gen.main(length=bit_len, byteorder='little', timer_start=timer)


if __name__ == "__main__":
    main()
