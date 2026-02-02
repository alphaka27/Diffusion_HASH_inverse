"""
MD5 implementation aligned with the diffusion_hash_inv codebase.
"""

from typing import Optional, Dict, Generator, Sequence, cast
import struct
import copy

from diffusion_hash_inv.core import BaseCalc
from diffusion_hash_inv.core import StepLogs, Logs
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

class MD5(MD5Calc):
    """
    MD5 Hashing Class
    """
    def __init__(self, main_config: MainConfig, hash_config: HashConfig):

        hash_cfg = hash_config if hash_config is not None \
            else HashConfig(hash_alg="md5", length=256)

        super().__init__(hash_cfg)

        self.hash_config = hash_cfg
        self.is_verbose = main_config.is_verbose if main_config is not None else True
        self.step_logs: StepLogs = StepLogs(wordsize=self.hash_config.ws_bits, \
                                            byteorder=self.hash_config.byteorder, \
                                            hierarchy=self.hash_config.constants.hierarchy)

        self.init_hash: Dict[str, bytes] = copy.deepcopy(self.hash_config.init_hash)

        self.reset()

    def reset(self):
        """
        Reset the MD5 instance for a new computation.
        """
        Logs.clear(step_logs = self.step_logs)
        self.init_hash = copy.deepcopy(self.hash_config.init_hash)

    @Logs.steplogs_update(step_cat=1)
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

    @Logs.steplogs_update(step_cat=2)
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

    @Logs.steplogs_update(step_cat=3)
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

    # TODO: Merge step4 inner and outer loops to clean up code.

    @Logs.steplogs_update(step_cat=4)
    def _step4_loop(self, message_block: Sequence[int], **previous_hash: Dict[str, int]) \
        -> Generator[Dict[str, int], None, Dict[str, int]]:
        """
        inner loop of step 4 for MD5 processing.
        Parameters:
            message_block (Sequence[int]): 16-words block.
            previous_hash (Dict[str, int]): Previous hash values.
        Returns:
            loop_result (Dict[str, int]): Updated A, B, C, D values.
        """
        a: Optional[int] = previous_hash.pop('A', None)
        b: Optional[int] = previous_hash.pop('B', None)
        c: Optional[int] = previous_hash.pop('C', None)
        d: Optional[int] = previous_hash.pop('D', None)

        assert a is not None and b is not None and c is not None and d is not None, \
                "Previous hash must contain A, B, C, and D."

        assert isinstance(a, int) and isinstance(b, int) \
            and isinstance(c, int) and isinstance(d, int), "a, b, c, d must be integers."

        
        return state

    @Logs.steplogs_update(step_cat=4)
    def _step4_block_loop(self, message_block: Sequence[Sequence[int]], \
                          **previous_hash: Dict[str, int])\
                            ->Generator[Dict[str, int], None, Dict[str, int]]:
        """
        Wrapper for the inner loop of step 4 for MD5 processing.
        Parameters:
            message_block (Sequence[Sequence[int]]): 16-words blocks.
            previous_hash (Dict[str, int]): Previous hash values.
        Returns:

        """

        prev_a = previous_hash['A']
        prev_b = previous_hash['B']
        prev_c = previous_hash['C']
        prev_d = previous_hash['D']

        assert prev_a is not None and prev_b is not None \
            and prev_c is not None and prev_d is not None, \
                "Previous hash must contain A, B, C, and D."

        assert isinstance(prev_a, int) and isinstance(prev_b, int) \
            and isinstance(prev_c, int) and isinstance(prev_d, int), "a, b, c, d must be integers."

        for block in message_block:
            a, b, c, d = prev_a, prev_b, prev_c, prev_d
            for _i in range(64):
                self.set_variable("loop_overflow_count", 0)
                if 0 <= _i <= 15:
                    f = self.f_func(b, c, d)
                    g = _i
                elif 16 <= _i <= 31:
                    f = self.g_func(b, c, d)
                    g = (5 * _i + 1) % 16
                elif 32 <= _i <= 47:
                    f = self.h_func(b, c, d)
                    g = (3 * _i + 5) % 16
                else:
                    f = self.i_func(b, c, d)
                    g = (7 * _i) % 16

                f = self.modular_add(f, a)
                f = self.modular_add(f, self.hash_config.k[_i])
                f = self.modular_add(f, self.word_to_int(block[g]))
                a = d
                d = c
                c = b
                b = self.modular_add(b, self.rotl(f, self.hash_config.s[_i]))
                state = {'A': a, 'B': b, 'C': c, 'D': d}
            yield state

            a = state['A']
            b = state['B']
            c = state['C']
            d = state['D']

            a = self.modular_add(a, prev_a)
            b = self.modular_add(b, prev_b)
            c = self.modular_add(c, prev_c)
            d = self.modular_add(d, prev_d)

            yield {
                'Update_A': a,
                'Update_B': b,
                'Update_C': c,
                'Update_D': d
            }
        return {
            'Update_A': a,
            'Update_B': b,
            'Update_C': c,
            'Update_D': d
        }

    @Logs.steplogs_update(step_cat=4)
    def step4(self, data: Sequence[Sequence[int]], previous_hash: Dict[str, int]) \
        -> Dict[str, int]:
        """
        Step 4 of MD5 processing.  
        Process Message in 16-Word Blocks
        """
        generated_hash = self._step4_block_loop(data, **previous_hash)
        return generated_hash

    @Logs.steplogs_update(step_cat=5)
    def step5(self, data: Dict[str, int]) -> bytes:
        """
        Step 5 of MD5 processing.  
        Output
        """
        return struct.pack('<4I', \
                        data['Update_A'], data['Update_B'], data['Update_C'], data['Update_D'])

    def digest(self, data: bytes) -> bytes:
        """
        Return the binary MD5 digest of the data.  
        """
        self.clear_overflow()
        org_data_len = len(data) * 8
        padded_data = self.step1(data)
        # Logs.stdout_logs("After Step 1 (Padding bits):", padded_data, self.is_verbose)

        processed_data: Sequence[Sequence[bytes]] = self.step2(padded_data, org_data_len)
        # Logs.stdout_logs("After Step 2 (Pre-processing):", \
        #         f"{len(processed_data)} blocks of 16 words", self.is_verbose)
        # Logs.stdout_logs(processed_data, self.is_verbose)

        prev_hash: Dict[str, int] = self.step3()

        # Logs.stdout_logs("After Step 3 (Initialize MD Buffer):", prev_hash, self.is_verbose)
        generated_hash: Dict[str, int] = self.step4(processed_data, prev_hash)
        # Logs.stdout_logs("After Step 4 (Process Message in 16-Word Blocks
        #     ):", generated_hash, self.is_verbose)
        digest_bytes: bytes = self.step5(generated_hash)
        # Logs.stdout_logs("After Step 5 (Output):", digest_bytes, self.is_verbose)
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
    _hash_config = HashConfig(hash_alg="md5", length=256)

    hash_alg = MD5(_main_config, _hash_config)
    _res = hash_alg.digest(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    print(Logs.bytes_to_str(_res))
    print(_res)
    _REX_HEX = hash_alg.hexdigest(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    print(_REX_HEX)
    # for msg, expected in test_vectors.items():
    #     result = md5_hexdigest(msg)
    #     print(f"MD5('{msg.decode('utf-8', errors='ignore')}') = {result}  ", end='')
    #     print("OK" if result == expected else f"FAIL (expected {expected})")
