"""
MD5 implementation aligned with the diffusion_hash_inv codebase.
"""

from typing import Optional, Dict, Generator, Sequence, Any, List
import struct
import copy
from contextlib import contextmanager

from diffusion_hash_inv.core import BaseCalc
from diffusion_hash_inv.logger import (
    StepLogs,
    Logs,
    MD5RoundTrace,
    MD5Step4Trace,
    MD5Logger,
)
from diffusion_hash_inv.config import MainConfig, HashConfig

class MD5Calc(BaseCalc):
    """
    MD5 Calculation Class
    """
    def __init__(self, hash_config: Optional[HashConfig] = None):
        super().__init__(hash_config)

    def f_func(self, x: int, y: int, z: int) -> int:
        """
        MD5 F function.
        """
        assert isinstance(x, int) and isinstance(y, int) and isinstance(z, int), \
            "Inputs must be integers."
        assert 0 <= x <= self.mask and 0 <= y <= self.mask and 0 <= z <= self.mask, \
            "Inputs must be 32-bit unsigned integers."

        return self.modular_or(self.modular_and(x, y), self.modular_and(self.modular_not(x), z))

    def g_func(self, x: int, y: int, z: int) -> int:
        """
        MD5 G function.
        """
        assert isinstance(x, int) and isinstance(y, int) and isinstance(z, int), \
            "Inputs must be integers."
        assert 0 <= x <= self.mask and 0 <= y <= self.mask and 0 <= z <= self.mask, \
            "Inputs must be 32-bit unsigned integers."

        return self.modular_or(self.modular_and(x, z), self.modular_and(y, self.modular_not(z)))

    def h_func(self, x: int, y: int, z: int) -> int:
        """
        MD5 H function.
        """
        assert isinstance(x, int) and isinstance(y, int) and isinstance(z, int), \
            "Inputs must be integers."
        assert 0 <= x <= self.mask and 0 <= y <= self.mask and 0 <= z <= self.mask, \
            "Inputs must be 32-bit unsigned integers."

        return self.modular_xor(self.modular_xor(x, y), z)

    def i_func(self, x: int, y: int, z: int) -> int:
        """
        MD5 I function.
        """
        assert isinstance(x, int) and isinstance(y, int) and isinstance(z, int), \
            "Inputs must be integers."
        assert 0 <= x <= self.mask and 0 <= y <= self.mask and 0 <= z <= self.mask, \
            "Inputs must be 32-bit unsigned integers."

        return self.modular_xor(y, self.modular_or(x, self.modular_not(z)))

class MD5Logic(MD5Calc):
    """
    MD5 Logic Class
    """
    def __init__(self, hash_config: HashConfig):
        super().__init__(hash_config)
        self.hash_config = hash_config

        self.init_hash: Dict[str, bytes] = None
        self.k = copy.deepcopy(self.hash_config.constants.k)
        self.s = copy.deepcopy(self.hash_config.constants.s)
        self.initialize()
        assert self.init_hash is not None, "Initial hash must be set."

    def initialize(self):
        """
        Reset the MD5 logic instance for a new computation.
        """
        self.init_hash = copy.deepcopy(self.hash_config.init_hash)

    @staticmethod
    def _state(a: int, b: int, c: int, d: int) -> Dict[str, int]:
        """Build MD5 register snapshot."""
        return {"A": a, "B": b, "C": c, "D": d}

    @MD5Logger.step(step_index=1)
    def step1(self, data: bytes) -> bytes:
        """
        Step 1 of MD5 processing.  
        Append Padding bits  
        Parameters:
            data (bytes): Original data.
        Returns:
            padded (bytes): Padded data.
        """
        padded: bytes = data + b'\x80'
        while (len(padded) * 8) % 512 != 448:
            padded += b'\x00'

        return padded


    @MD5Logger.step(step_index=2)
    def step2(self, data: bytes, original_bit_len: int) -> Sequence[Sequence[bytes]]:
        """
        Step 2 of MD5 processing.

        Append 64 bits for Length in little-endian

        Parameters:
            data (bytes): Padded data.
            original_bit_len (int): Original length of data in bits.
        Returns:
            pre_processed_blocks (List[int]): List of 16-word blocks.
        """
        bs_byte = self.hash_config.bs_bits // 8
        ws_byte = self.hash_config.ws_bits // 8
        _pre_processed: bytes = data + struct.pack('<Q', original_bit_len)
        assert len(_pre_processed) % (bs_byte) == 0, \
            f"Pre-processed data length must be a multiple of {self.block_size} bits."

        # List of 16-words blocks [[16-words], [16-words], ...]
        pre_processed_blocks: Sequence[Sequence[bytes]]= []
        for i in range(0, len(_pre_processed), bs_byte):
            block = []
            for j in range(0, bs_byte, ws_byte):
                word = _pre_processed[i + j:i + j + (ws_byte)]
                block.append(word)
            pre_processed_blocks.append(block)
        return pre_processed_blocks


    @MD5Logger.step(step_index=3)
    def step3(self) -> Dict[str, int]:
        """
        Step 3 of MD5 processing.

        Initialize MD Buffer

        Returns:
            init_hash (Dict[str, bytes]): Initial hash values.
        """
        prev_hash = {}
        a = self.init_hash['A']
        b = self.init_hash['B']
        c = self.init_hash['C']
        d = self.init_hash['D']
        prev_hash = {'A': a, 'B': b, 'C': c, 'D': d}
        return prev_hash

    @contextmanager
    def _step4_outer(self, a: int, b: int, c: int, d: int) \
                        -> Generator[Dict[str, Any], None, None]:
        """
        Wrapper for the inner loop of step 4 for MD5 processing.
        Parameters:
            init_hash (Dict[str, int]): Initial hash values.
        Returns:

        """

        prev_a, prev_b, prev_c, prev_d = a, b, c, d

        loop_params = {'A': prev_a, 'B': prev_b, 'C': prev_c, 'D': prev_d,
                    "res_list": []}

        try:
            yield loop_params

        finally:
            # res_list = loop_params["res_list"]
            _a = loop_params["A"]
            _b = loop_params["B"]
            _c = loop_params["C"]
            _d = loop_params["D"]

            _a = self.modular_add(_a, prev_a)
            _b = self.modular_add(_b, prev_b)
            _c = self.modular_add(_c, prev_c)
            _d = self.modular_add(_d, prev_d)

            loop_params["A"] = _a
            loop_params["B"] = _b
            loop_params["C"] = _c
            loop_params["D"] = _d


    @MD5Logger.step(step_index=4)
    def step4(self, data: Sequence[Sequence[bytes]], init_hash: Dict[str, int]) \
        -> MD5Step4Trace:
        """
        Step 4 of MD5 processing.  
        Process Message in 16-Word Blocks
        """
        updated_hash: Dict[str, int] = dict(init_hash)
        rounds: List[MD5RoundTrace] = []

        for _ in data:
            prev_hash = dict(updated_hash)
            a = prev_hash["A"]
            b = prev_hash["B"]
            c = prev_hash["C"]
            d = prev_hash["D"]
            loop_start = self._state(a, b, c, d)
            loop_states: List[Dict[str, int]] = []

            for i in range(64):
                if 0 <= i <= 15:
                    f = self.f_func(b, c, d)
                    g = i
                elif 16 <= i <= 31:
                    f = self.g_func(b, c, d)
                    g = (5 * i + 1) % 16
                elif 32 <= i <= 47:
                    f = self.h_func(b, c, d)
                    g = (3 * i + 5) % 16
                else:
                    f = self.i_func(b, c, d)
                    g = (7 * i) % 16

                f = self.modular_add(f, a)
                f = self.modular_add(f, self.k[i])
                f = self.modular_add(f, self.word_to_int(_[g]))

                a, b, c, d = (
                    d,
                    self.modular_add(b, self.rotl(f, self.s[i])),
                    b,
                    c,
                )
                loop_states.append(self._state(a, b, c, d))

            a = self.modular_add(a, prev_hash["A"])
            b = self.modular_add(b, prev_hash["B"])
            c = self.modular_add(c, prev_hash["C"])
            d = self.modular_add(d, prev_hash["D"])

            updated_hash["A"] = a
            updated_hash["B"] = b
            updated_hash["C"] = c
            updated_hash["D"] = d

            rounds.append(
                MD5RoundTrace(
                    loop_start=loop_start,
                    loop_states=loop_states,
                    loop_end=self._state(a, b, c, d),
                )
            )

        return MD5Step4Trace(updated_hash=updated_hash, rounds=rounds)


    @MD5Logger.step(step_index=5, update_overflow=True)
    def step5(self, data: Dict[str, int]) -> bytes:
        """
        Step 5 of MD5 processing.  
        Output
        """
        digest_bytes: bytes = struct.pack('<4I', \
                        data['A'], data['B'], data['C'], data['D'])
        return digest_bytes


class MD5(MD5Logic):
    """
    MD5 Hashing Class
    """
    def __init__(self, main_config: MainConfig, hash_config: HashConfig, steplogs: StepLogs):

        hash_cfg = hash_config if hash_config is not None \
            else HashConfig(hash_alg="md5", length=256)

        super().__init__(hash_cfg)

        self.is_verbose = main_config.verbose_flag if main_config is not None else True
        self.logs: StepLogs = steplogs
        self.reset()

    def reset(self):
        """
        Reset the MD5 instance for a new computation.
        """
        super().initialize()

    def digest(self, data: bytes) -> bytes:
        """
        Return the binary MD5 digest of the data.  
        """
        self.clear_overflow()
        org_data_len = len(data) * 8
        if self.is_verbose:
            print(f"Original data length (bits): {org_data_len}")
            print(f"Original data (bytes): {data}")

        padded_data = super().step1(data)

        processed_data: Sequence[Sequence[bytes]] = super().step2(padded_data, org_data_len)

        init_hash: Dict[str, int] = super().step3()

        generated_hash: Dict[str, int] = super().step4(processed_data, init_hash=init_hash)

        digest_bytes: bytes = super().step5(generated_hash)

        return digest_bytes

    def hexdigest(self, data: bytes) -> str:
        """
        Return the hexadecimal MD5 digest of the data.  
        """
        digest_bytes: bytes = self.digest(data)
        return Logs.bytes_to_str(digest_bytes)

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
    _main_config = MainConfig(
        message_flag=False,
        verbose_flag=True,
        clean_flag=False,
        debug_flag=False,
        make_xlsx_flag=False
    )
    _hash_config = HashConfig(hash_alg="md5", \
                    length=len(b"abcdefghijklmnopqrstuvwxyz")*8)

    hash_alg = MD5(_main_config, _hash_config, StepLogs())
    _REX_HEX = hash_alg.hexdigest(b"abcdefghijklmnopqrstuvwxyz")
    print(_REX_HEX)
    # for msg, expected in test_vectors.items():
    #     result = md5_hexdigest(msg)
    #     print(f"MD5('{msg.decode('utf-8', errors='ignore')}') = {result}  ", end='')
    #     print("OK" if result == expected else f"FAIL (expected {expected})")
