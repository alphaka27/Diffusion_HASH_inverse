"""
SHA-256 Implementation
"""

# TODO
# Time logging for performance measurement

import math
import numpy as np


# Constants start
# SHA-256 use sixty-four constant 32-bit words
K = np.array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
], dtype=np.uint32)

# Initial Hash value
INIT_HASH = np.array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
], dtype=np.uint32)
# Constants end

# Internal Operations of the SHA-256 algorithm start
class SHACalc:
    """
    Base class for SHA hash calculations.
    """
    def __init__(self):
        # Set Properties (bits)
        self.block_size = 512  # 64 bytes
        self.word_size = 32  # 4 bytes
        self.block_size_bytes = self.block_size // 8
        self.word_size_bytes = self.word_size // 8
        self.mask = np.uint32(0xFFFFFFFF)

    @staticmethod
    def add32(*ops):
        """
        Adds multiple np.uint32 numbers with modulo 2^32.
        """
        ops_arr = [np.asarray(op, dtype=np.uint32) for op in ops]
        ret = np.add.reduce(ops_arr, dtype=np.uint32)

        return ret

    def rotr(self, x:np.uint32, n:int):
        """
        Rotate right function for SHA-256.
        """

        assert isinstance(x, np.uint32), "Input must be a np.uint32 (rotr)."

        n = n % self.word_size  # Ensure n is within the word size
        right = x >> n
        left = (x << (self.word_size - n)) & self.mask
        ret = (left | right) & self.mask
        return ret

    def shr(self, x:np.uint32, n:int):
        """
        Shift right function for SHA-256.
        """

        assert isinstance(x, np.uint32), "Input must be a np.uint32 (shr)."

        n = n % self.word_size  # Ensure n is within the word size
        ret = (x >> n) & self.mask

        return ret

    def ch(self, x:np.uint32, y:np.uint32, z:np.uint32):
        """
        Ch function for SHA-256.
        """
        assert isinstance(x, np.uint32), "Input must be a np.uint32 (ch)."
        assert isinstance(y, np.uint32), "Input must be a np.uint32 (ch)."
        assert isinstance(z, np.uint32), "Input must be a np.uint32 (ch)."

        ret = ((x & y) ^ (~x & z)) & self.mask
        return ret

    def maj(self, x:np.uint32, y:np.uint32, z:np.uint32):
        """
        Maj function for SHA-256.
        """
        assert isinstance(x, np.uint32), "Input must be a np.uint32 (maj)."
        assert isinstance(y, np.uint32), "Input must be a np.uint32 (maj)."
        assert isinstance(z, np.uint32), "Input must be a np.uint32 (maj)."

        ret = ((x & y) ^ (x & z) ^ (y & z)) & self.mask
        return ret

    def sigma0(self, x:np.uint32):
        """
        Sigma0 function for SHA-256.
        """
        assert isinstance(x, np.uint32), "Input must be a np.uint32 (sigma0)."

        ret = (self.rotr(x, 7) ^ self.rotr(x, 18) ^ self.shr(x, 3)) & self.mask
        return ret

    def sigma1(self, x:np.uint32):
        """
        Sigma1 function for SHA-256.
        """
        assert isinstance(x, np.uint32), "Input must be a np.uint32 (sigma1)."

        ret = (self.rotr(x, 17) ^ self.rotr(x, 19) ^ self.shr(x, 10)) & self.mask
        return ret

    def cap_sigma0(self, x:np.uint32):
        """
        CapSigma0 function for SHA-256.
        """
        assert isinstance(x, np.uint32), "Input must be a np.uint32 (cap_sigma0)."

        ret = (self.rotr(x, 2) ^ self.rotr(x, 13) ^ self.rotr(x, 22)) & self.mask
        return ret

    def cap_sigma1(self, x:np.uint32):
        """
        CapSigma1 function for SHA-256.
        """
        assert isinstance(x, np.uint32), "Input must be a np.uint32 (cap_sigma1)."

        ret = (self.rotr(x, 6) ^ self.rotr(x, 11) ^ self.rotr(x, 25)) & self.mask
        return ret

    @staticmethod
    def to_hex32_scalar(x) -> str:
        """단일 32-bit 값 → 8자리 hex"""
        return "0x" + f"{int(x):08x}"

    @staticmethod
    def to_hex32_concat(seq) -> str:
        """시퀀스(8워드 등) → 64자리 hex"""
        return ''.join(f"{int(x):08x}" for x in seq)
# Internal Operations of the SHA-256 algorithm end

# Implementation of the SHA-256 algorithm start
class SHA256(SHACalc):
    """
    Implementation of the SHA-256 hash function.
    """
    def __init__(self, is_verbose = True, output_format = None):
        print("Load SHA-256")
        super().__init__()
        SHA256.verbose_flag = is_verbose

        self.prev_hash = INIT_HASH.copy()
        self.message = None # in bytes
        self.message_len = -1 # in bits
        self.hash = None

        self.message_block = []

        self.block_n = math.ceil((self.message_len + 1 + 64) / self.block_size)
        assert output_format is not None, "JSON Formatter is needed"
        self.res_out = output_format
        # if output_format is not None:
        #     self.res_out = output_format
        # else:
        #     self.res_out = OutputFormat()

    def reset(self):
        """각 해시 계산 시작 시 내부 상태 초기화"""
        print("Reset state of SHA256")
        self.prev_hash = INIT_HASH.copy()
        self.hash = None
        self.message_block = []
        self.block_n = 0

    # Implementation of the SHA-256 algorithm start

    @staticmethod
    def add32(*ops):
        """
        Adds multiple np.uint32 numbers with modulo 2^32.
        """
        ops_arr = [np.asarray(op, dtype=np.uint32) for op in ops]
        ret = np.add.reduce(ops_arr, dtype=np.uint32)

        return ret

    def pad(self):
        """
        SHA-256 padding (bytearray -> uint32 words, big-endian)
        self.message: bytearray
        self.message_len: 원본 비트 길이 l
        """

        l = len(self.message)
        self.message = bytearray(self.message)  # Ensure message is a bytearray
        self.message += b'\x80'  # Append the bit '1' (0x80 in hex)
        pad_zero_len = (56 - (l + 1) % 64) % 64
        self.message += b'\x00' * pad_zero_len

        # Append the original message length in bits (64 bits)
        self.message += self.message_len.to_bytes(8, byteorder='big')
        words_be = np.frombuffer(self.message, dtype='>u4')
        self.message = words_be.astype(np.uint32, copy=True)  # 워드 배열
        self.block_n = self.message.size // 16

    def parse(self):
        """
        Parsing function for SHA-256.
        """
        for i in range(self.block_n):
            self.message_block.append(self.message[i * 16:(i + 1) * 16])

    def preprocess(self):
        """
        Padding & Parsing for SHA-256.
        """
        self.message_block = []
        assert self.message is not None, "Message must be set before preprocessing."
        assert self.message_len > 0, "Message length must be positive."
        assert isinstance(self.message, (bytearray, bytes)), "Message must be a bytearray."

        self.pad()
        self.parse()

        block_dict = {}

        print("Preprocessing complete.")
        if self.verbose_flag:
            print("Message blocks: ")
            for i, block in enumerate(self.message_block):
                print(f"Block {i}:")
                for j, word in enumerate(block):
                    if j % 8 == 0 and j != 0:
                        print()
                    print(f"\\x{super().to_hex32_scalar(word)}", end=' ')
                print()
            print()

        for _i, block in enumerate(self.message_block):
            block_dict[f"Block {_i}"] = block
        self.res_out.add_preprocess(block_dict)
        return True

    def step1(self, iteration):
        """
        Step 1: Message Schedule Preparation
        """
        print("Step 1: Message Schedule Preparation")

        w_tmp = []
        for _i in range(64):
            if _i < 16:
                w_tmp.append(self.message_block[iteration][_i])
            else:
                s0 = super().sigma0(w_tmp[_i - 15])
                s1 = super().sigma1(w_tmp[_i - 2])
                wt_7 = w_tmp[_i - 7]
                wt_16 = w_tmp[_i - 16]
                _tmp = super().add32(s1, wt_7, s0, wt_16)
                w_tmp.append(_tmp)
        if self.verbose_flag:
            print(super().to_hex32_concat(w_tmp))
        self.res_out.add_step1(w_tmp)
        return w_tmp

    def step2(self, in_hash):
        """
        Step 2: Initialize working variables.
        """
        print("Step 2: Initialize working variables")
        a, b, c, d, e, f, g, h = in_hash
        ret = [a, b, c, d, e, f, g, h]

        ret_dict = {}
        for _i, val in enumerate(ret):
            ret_dict[chr(ord('a') + _i)] = super().to_hex32_scalar(val)
        self.res_out.add_step2(ret_dict)

        if self.verbose_flag:
            print(ret_dict)
        return ret

    #pylint: disable=too-many-locals
    def step3(self, w, in_hash):
        """
        Step 3: Main compression function loop.
        """
        print("Step 3: Main compression function loop")

        a, b, c, d, e, f, g, h = in_hash
        for _i in range(64):
            t1 = super().add32(h, self.cap_sigma1(e), super().ch(e, f, g), K[_i], w[_i])
            t2 = super().add32(self.cap_sigma0(a), super().maj(a, b, c))
            h = g
            g = f
            f = e
            e = super().add32(d, t1)
            d = c
            c = b
            b = a
            a = super().add32(t1, t2)
            ret_dict = {"a": a, "b": b, "c": c, "d": d, "e": e,
                        "f": f, "g": g, "h": h, "t1": t1, "t2": t2}

            for _k, _v in ret_dict.items():
                ret_dict[_k] = super().to_hex32_scalar(_v)

            self.res_out.add_step3_round(_i, ret_dict)
            _round_idx = _i + 1
            if _round_idx == 1:
                loop_m = f"{_round_idx:02}st loop"
            elif _round_idx == 2:
                loop_m = f"{_round_idx:02}nd loop"
            elif _round_idx == 3:
                loop_m = f"{_round_idx:02}rd loop"
            else:
                loop_m = f"{_round_idx:02}th loop"

            if self.verbose_flag:
                print(f"{loop_m} - {ret_dict}")

        return [a, b, c, d, e, f, g, h]
    #pylint: enable=too-many-locals

    def step4(self, work, in_hash):
        """
        Step 4: Finalize the hash value.
        """

        print("Step 4: Finalize the hash value")
        a,b,c,d,e,f,g,h = work
        res = [
            super().add32(a, in_hash[0]), super().add32(b, in_hash[1]),
            super().add32(c, in_hash[2]), super().add32(d, in_hash[3]),
            super().add32(e, in_hash[4]), super().add32(f, in_hash[5]),
            super().add32(g, in_hash[6]), super().add32(h, in_hash[7]),
        ]

        ret_dict = {}
        for _i, val in enumerate(res):
            ret_dict[chr(ord('a') + _i)] = super().to_hex32_scalar(val)
        self.res_out.add_step4(ret_dict)
        if self.verbose_flag:
            print(ret_dict)
        return res

    def compute_hash(self):
        """
        Compute the SHA-256 hash of the input message.
        """
        w = []
        try:
            for _i in range(self.block_n):
                # pylint: disable=line-too-long
                print(f"\nProcessing {_i + 1} block of {self.block_n} blocks ({_i / self.block_n * 100:.2f}%)")
                # pylint: enable=line-too-long
                w = self.step1(_i)
                print()

                self.prev_hash = self.step2(self.prev_hash)
                print()

                out = self.step3(w, self.prev_hash)
                print()

                self.hash = self.step4(out, self.prev_hash)
                print()

                self.prev_hash = self.hash
                self.res_out.add_round(_i)

            # breakpoint()
            return True

        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Error during hash computation: {e}")
            return False
        # pylint: enable=broad-exception-caught

    def finalize(self):
        """
        Finalize the hash computation and return the final hash value.
        """
        # breakpoint()

        a,b,c,d,e,f,g,h = (np.uint32(x) for x in self.hash)
        out = np.array([a,b,c,d,e,f,g,h], dtype=np.uint32)
        return out

    def digest(self, message = None, message_len = -1) -> bytearray:
        """
        Generate the SHA-256 hash for the given message.
        """
        assert message is not None, "Message must be provided."
        assert isinstance(message, (bytes, bytearray)), "Message must be bytes or bytearray."
        assert message_len > 0, "Message length must be positive."

        self.message = message # in binary string
        self.message_len = message_len

        preprocess_success = False
        compute_success = False

        preprocess_success = self.preprocess()
        print("Preprocessing successful")
        # breakpoint()
        compute_success = self.compute_hash()
        print("Computation successful")
        print()

        if preprocess_success and compute_success:
            result = self.finalize()
            return result

        raise RuntimeError("Hash computation failed.")
# Implementation of the SHA-256 algorithm end
