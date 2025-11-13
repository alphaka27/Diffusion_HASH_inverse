"""
MD5 implementation aligned with the diffusion_hash_inv codebase.
"""

from typing import Optional
import struct
import math

MASK = 0xFFFFFFFF
_S = [7,12,17,22]*4 + [5,9,14,20]*4 + [4,11,16,23]*4 + [6,10,15,21]*4
_K = [int((1 << 32) * abs(math.sin(i + 1))) & MASK for i in range(64)]

class MD5Calc:
    """
    MD5 Calculation Class
    """
    def __init__(self):
        self.block_size = 512
        self.word_size = 32
        self.block_bytes = self.block_size // 8
        self.word_bytes = self.word_size // 8

    def add32(self, x, y):
        """
        Perform addition modulo 2^32.
        """
        return (x + y) & MASK

    def rotl32(self, x, s: int):
        """
        Perform left rotation on a 32-bit integer.
        """
        return ((x << s) | (x >> (self.word_size - s))) & MASK

    def not32(self, x):
        """
        Perform bitwise NOT on a 32-bit integer.
        """
        return (~x) & MASK




class MD5(MD5Calc):
    """
    MD5 Hashing Class
    """
    def __init__(self):
        super().__init__()
        self.message: Optional[bytes] = None

    def __preprocess(self, data: bytes) -> bytes:
        """
        Preprocess the input data by padding and appending length.
        """
        STEP_INDEX = "Preprocess"
        original_byte_len = len(data)
        original_bit_len = original_byte_len * 8

        # Append the bit '1' to the message
        data += b'\x80'

        # Append '0' bits until message length in bits ≡ 448 (mod 512)
        while (len(data) * 8) % 512 != 448:
            data += b'\x00'

        # Append the original length as a 64-bit little-endian integer
        data += struct.pack('<Q', original_bit_len)

        return data

    def __step1(self, data: bytes) -> bytes:
        """
        Process input data in 64-byte blocks.
        """
        STEP_INDEX = "Step1"

    def __step2(self) -> bytes:
        """
        Finalize the MD5 hash computation.
        """
        STEP_INDEX = "Step2"

    def __step3(self) -> bytes:
        """
        Produce the final MD5 digest.
        """
        STEP_INDEX = "Step3"

    def __step4(self) -> bytes:
        """
        Produce the final MD5 digest in hexadecimal format.
        """
        STEP_INDEX = "Step4"

    def main(self) -> bytes:
        """
        Main method to compute MD5 hash.
        """


    def __update(self, data: bytes) -> bytes:
        """
        Update the MD5 hash with new data.
        """
        assert isinstance(data, bytes), "Input data must be bytes."
        self.message = data

    def digest(self, data: bytes) -> bytes:
        """
        Return the binary MD5 digest of the data.
        """
        self.__update(data)
        
        

    def hexdigest(self, data: bytes) -> str:
        """
        Return the hexadecimal MD5 digest of the data.
        """

if __name__ == "__main__":
    # 테스트 벡터
    test_vectors = {
        b"": "d41d8cd98f00b204e9800998ecf8427e",
        b"a": "0cc175b9c0f1b6a831c399e269772661",
        b"abc": "900150983cd24fb0d6963f7d28e17f72",
        b"message digest": "f96b697d7cb7938d525a2f31aaf161d0",
        b"abcdefghijklmnopqrstuvwxyz": "c3fcd3d76192e4007dfb496cca67e13b",
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
            "d174ab98d277d9f5a5611c2c9f419d9f",
        b"12345678901234567890123456789012345678901234567890123456789012345678901234567890":
            "57edf4a22be3c955ac49da2e2107b67a",
    }

    # for msg, expected in test_vectors.items():
    #     result = md5_hexdigest(msg)
    #     print(f"MD5('{msg.decode('utf-8', errors='ignore')}') = {result}  ", end='')
    #     print("OK" if result == expected else f"FAIL (expected {expected})")
