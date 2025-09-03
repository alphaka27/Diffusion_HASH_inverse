"""
Random N character Generation
Password Generator
"""

import unicodedata
from secrets import choice
import argparse
import string
import math

try:
    from diffusion_hash_inv.utils import FileIO
except ImportError as e:
    print(f"Error importing FileIO: {e}")

class GenerateRandomNChar(FileIO):
    """
    Generate a random string of N characters.
    """
    def __init__(self, clear_flag=False, verbose_flag=True, main_flag=False):
        super().__init__(main_flag, start_time=super().encode_timestamp())
        print(f"Flags - Clear: {clear_flag}, Verbose: {verbose_flag}\n")
        if clear_flag:
            print("Clearing generated files...")
            super().file_clean(clear_flag=clear_flag)
        self.__verbose__ = verbose_flag
        self.ts: bytes = super().encode_timestamp()

        GenerateRandomNChar.alphabet = string.ascii_letters \
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

    @staticmethod
    def calc_entropy(char_len: int, _pwd: str) -> float:
        """
        Calculate the entropy of the generated password.
        """
        entropy = char_len * math.log2(len(_pwd))
        return entropy

    def generate(self, length: int = 16) -> str:
        """
        Generate a random string of N characters.
        """
        _pwd = ''.join(choice(GenerateRandomNChar.alphabet) for _ in range(length))
        return _pwd

    def normalize(self, s: str, form: str = "NFKC") -> str:
        """
        Normalize a string to the specified Unicode normalization form.
        """
        assert form in ["NFKC", "NFKD", "NFC", "none"], "Invalid normalization form"
        s = unicodedata.normalize(form.upper(), s)
        return s.encode("utf-8")

    def main(self, length: int = 16):
        """
        Main function to generate random strings and display their entropy.
        """
        timestamp = super().encode_timestamp()
        _pwd = self.generate(length)
        _pwd = self.normalize(_pwd)
        print(f"Generated Password: {_pwd}")
        print(f"Entropy: {self.calc_entropy(length, _pwd)} bits")
        if self.__verbose__:
            self.help()
        f_w, _ = self.file_io(f"random_{length * 8}_char.char")
        f_w(_pwd, length * 8, ts=timestamp)
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
        print(f"Iteration: {_ + 1}")
        pw_gen.main(BIT_LEN)
        print()

    if args.verbose:
        pw_gen.help()

        PW = pw_gen.generate()
        print(PW)
        print(type(PW))

        print()
        pw_utf8 = pw_gen.normalize(PW, "NFKC")
        print(pw_utf8)
        print(type(pw_utf8))
