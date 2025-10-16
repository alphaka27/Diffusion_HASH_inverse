"""
MD5 Implementation
"""

import math
import numpy as np

class MD5Calc:
    """
    MD5 Calculation Class
    """
    def __init__(self, input_data: bytes):
        pass

    def bit_not(self, x):
        """
        Bitwise NOT operation
        """

    def bit_and(self, x, y):
        """
        Bitwise AND operation
        """

    def bit_or(self, x, y):
        """
        Bitwise OR operation
        """

    def bit_xor(self, x, y):
        """
        Bitwise XOR operation
        """

    def left_rotate(self, x, amount):
        """
        Left rotate operation
        """

    def add_mod_2_32(self, *args):
        """
        Addition modulo 2^32
        """


class MD5(MD5Calc):
    """
    MD5 Hashing Class
    """
    def __init__(self, verbose_flag=True, main_flag=False):
        print("MD5 Loadded")
        self.__verbose__ = verbose_flag
        print(f"Flags - Verbose: {verbose_flag}\n")

    # def 

    @staticmethod
    def main(*flags: bool, length: int = 512, iteration: int = 1):
        """
        Main function to execute the MD5 hashing process.
        """
        message, verbose, clear = flags
        assert length > 0, "Length must be a positive integer."
        assert iteration >= 0, "Iteration must be a non-negative integer."

        print(f"Flags - Verbose: {verbose}, Clear: {clear}")

if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser(description="MD5 Hash Generator")
    parser.add_argument('-l', '--length', type=int, default=argparse.SUPPRESS,
                        help='Length of random bits to generate (default: 512)')
    parser.add_argument('-e', '--exponentiation', type=int, default=argparse.SUPPRESS,
                        help='2 to the power of <exponentiation> (default: 9)')
    parser.add_argument('-i', '--iteration', type=int, default=1,
                        help='Running iterations (default: 1)')

    gv = parser.add_mutually_exclusive_group()
    gv.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                    help='Enable verbose output')
    gv.add_argument('-q', '--quiet', action='store_false', dest='verbose',
                    help='Suppress output')
    parser.set_defaults(verbose=True)

    gm = parser.add_mutually_exclusive_group()
    gm.add_argument('-m', '--message', action="store_true",
                    dest='message', help='Message input mode')
    gm.add_argument('-b', '--bit', action="store_false",
                    dest='message', help='Bit string input mode')
    parser.set_defaults(message=True)

    gc = parser.add_mutually_exclusive_group()
    gc.add_argument('-c', '--clear', action='store_true',
                    dest='clear', help='Clear generated files')
    gc.add_argument('-C', '--no-clear', action='store_true', dest='clear',
                    help='Do not clear generated files (default)')
    parser.set_defaults(clear=False)
    _args = parser.parse_args()

    TEST = "hello"
    md5 = hashlib.md5()
    md5.update(TEST.encode("UTF-8"))
    HM = md5.hexdigest()
    print(HM)


    LENGTH = None
    if hasattr(_args, 'length'):
        LENGTH = _args.length
    else:
        pass

    if hasattr(_args, 'exponentiation'):
        EXP = _args.exponentiation
        LENGTH = 2 ** EXP
    else:
        pass
    assert LENGTH >= 8, "Too Shot to make password"

    main(_args.message, _args.verbose, _args.clear, length=LENGTH, iteration=_args.iteration)
