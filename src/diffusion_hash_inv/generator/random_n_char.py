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

from diffusion_hash_inv.config import MainConfig
from diffusion_hash_inv.utils import FileIO
from diffusion_hash_inv.logger import Logs

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
        if self.main_config.seed_flag:
            _pwd = ''.join(choice(GenerateRandomNChar.alphabet) for _ in range(length))
        else:
            random.seed(self.main_config.seed)
            _pwd = ''.join(random.choice(GenerateRandomNChar.alphabet) for _ in range(length))

        return _pwd

    def normalize(self, s: str, form: str = "NFKC") -> bytes:
        """
        Normalize a string to the specified Unicode normalization form.
        """
        assert form in ["NFKC", "NFKD", "NFC", "none"], "Invalid normalization form"
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random bits and save to a binary file.")
    parser.add_argument('-l', '--length', type=int, default=argparse.SUPPRESS,
                        help='Length of random bits to generate (default: 512)')

    parser.add_argument('-i', '--iterations', type=int,
                        default=1, help='Number of iterations to run (default: 1)')

    parser.add_argument('-e', '--exponentiation', type=int, default=argparse.SUPPRESS,
                        help='2 to the power of <exponentiation> (default: 9)')

    gv = parser.add_mutually_exclusive_group()
    gv.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                    help='Enable verbose output')
    gv.add_argument('-q', '--quiet', action='store_false', dest='verbose',
                    help='Suppress output')
    parser.set_defaults(verbose=True)

    gc = parser.add_mutually_exclusive_group()
    gc.add_argument('-c', '--clear', action='store_true',
                    dest='clear', help='Clear generated files')
    gc.add_argument('-C', '--no-clear', action='store_true', dest='clear',
                    help='Do not clear generated files (default)')
    parser.set_defaults(clear=False)
    parser.set_defaults(length=512)
    parser.set_defaults(exponentiation=9)

    args = parser.parse_args()
    LEN_FLAG = False
    EXP_FLAG = False

    if hasattr(args, 'length'):
        LENGTH = args.length
        LEN_FLAG = True

    else:
        LENGTH = 512

    if hasattr(args, 'exponentiation'):
        EXP = args.exponentiation
        EXP_FLAG = True
    else:
        EXP = 9

    BIT_LEN = None
    if LEN_FLAG:
        BIT_LEN = LENGTH

    elif EXP_FLAG:
        BIT_LEN = 2 ** EXP

    assert BIT_LEN is not None, "Either length or exponentiation must be specified."

    pw_gen = GenerateRandomNChar()

    for _ in range(args.iterations):
        timer = Logs.perftimer_start()
        print(f"Iteration: {_ + 1}")
        pw_gen.main(length=BIT_LEN, byteorder='little', timer_start=timer)
