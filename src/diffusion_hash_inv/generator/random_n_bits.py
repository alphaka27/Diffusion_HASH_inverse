"""
Generate 512-bit random number and save it as binary file
"""

from secrets import randbits
import argparse
import math

try:
    from diffusion_hash_inv.utils import FileIO
except ImportError as e:
    print(f"Error importing FileIO: {e}")

class GenerateRandom(FileIO):
    """
    Generate a random number of specified bit length.
    """
    def __init__(self, verbose_flag = True, main_flag = False):
        super().__init__(init_flag=main_flag, clear_flag=False,
                        verbose_flag=verbose_flag, start_time=super().encode_timestamp())
        print(f"Flags - Verbose: {verbose_flag}\n")
        self.__verbose__ = verbose_flag
        self.ts = super().encode_timestamp()

    @staticmethod
    def bytes_to_hex_block(b: bytes, *, word_bytes: int = 2, line_bytes: int = 16,
                    pad_last: bool = True) -> str:
        """
        Convert bytes to a formatted hexadecimal string.
        """
        out_lines = []
        for i in range(0, len(b), line_bytes):
            line = b[i:i+line_bytes]
            groups = []
            for j in range(0, len(line), word_bytes):
                chunk = line[j:j+word_bytes]
                hs = chunk.hex()
                if pad_last and len(chunk) < word_bytes:
                    hs = hs.zfill(word_bytes * 2)  # 마지막 덜 찬 그룹 0패딩
                groups.append(hs)
            out_lines.append(' '.join(groups))
        return '\n'.join(out_lines)

    @staticmethod
    def print_hex(msg, x):
        """
        Print the hexadecimal representation of the given bytes.
        """
        if msg.endswith("\n"):
            print(msg, end="")
        else:
            print(msg+":")
        print(GenerateRandom.bytes_to_hex_block(x))
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
        print(' '.join(f'{x:08b}' for x in data))
        print()

    def generate_random_bits(self, length=512) -> bytes:
        """
        Generate a random 512-bit number and return its hexadecimal and binary representations.
        """
        timestamp = super().encode_timestamp()
        _n = randbits(length)
        _length = math.ceil(length / 8)
        bytes_n = _n.to_bytes(_length, byteorder='big', signed=False)

        assert len(bytes_n) == _length, "Binary length does not match specified length."

        if self.__verbose__:
            print(f"Generated {length}-bit random number.")
            print(f"Generated at: {timestamp.decode('utf-8')}")
            self.print_bin("Data in Binary", bytes_n)
            print(f"Binary length in Bytes: \n{len(bytes_n)}\n") # type: str
        self.print_hex(f"Data in Hexadecimal (len: {len(bytes_n)}bytes)", bytes_n)

        f_w, _ = self.file_io(f"random_{length}_bits.bin")
        f_w(bytes_n, length, ts=timestamp)

        return bytes_n

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

    generator = GenerateRandom(args.clear, args.verbose, main_flag=True)

    for _ in range(args.iterations):
        print(f"Iteration: {_ + 1}")
        _ = generator.generate_random_bits(BIT_LEN)
        print()
