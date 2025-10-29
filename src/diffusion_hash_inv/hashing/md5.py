"""
MD5 implementation aligned with the diffusion_hash_inv codebase.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

# MD5 round-dependent left rotation amounts (RFC 1321 order)
_SHIFT_AMOUNTS = np.array(
    [7, 12, 17, 22] * 4
    + [5, 9, 14, 20] * 4
    + [4, 11, 16, 23] * 4
    + [6, 10, 15, 21] * 4,
    dtype=np.uint8,
)

# K[i] = floor(2^32 * |sin(i+1)|)
_K = np.array(
    [int((1 << 32) * abs(math.sin(i + 1))) & 0xFFFFFFFF for i in range(64)],
    dtype=np.uint32,
)

# Initial MD5 state (A, B, C, D)
_INIT_STATE = np.array(
    [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476],
    dtype=np.uint32,
)


class MD5Calc:
    """
    Shared helpers for the MD5 hash implementation.
    """

    def __init__(self) -> None:
        self.block_size = 512  # in bits (64 bytes)
        self.word_size = 32  # in bits (4 bytes)
        self.block_size_bytes = self.block_size // 8
        self.mask = np.uint32(0xFFFFFFFF)
        self.K = _K.copy()
        self.shift_amounts = _SHIFT_AMOUNTS

    @staticmethod
    def _to_uint32(value) -> np.uint32:
        return np.uint32(int(np.uint32(value)) & 0xFFFFFFFF)

    @staticmethod
    def to_hex32(value) -> str:
        """Return an 8-digit hex string with 0x prefix."""
        return "0x" + f"{int(np.uint32(value)) & 0xFFFFFFFF:08x}"

    def bit_not(self, x):
        return np.uint32(~int(np.uint32(x)) & 0xFFFFFFFF)

    def bit_and(self, x, y):
        return np.uint32(int(np.uint32(x)) & int(np.uint32(y)))

    def bit_or(self, x, y):
        return np.uint32(int(np.uint32(x)) | int(np.uint32(y)))

    def bit_xor(self, x, y):
        return np.uint32(int(np.uint32(x)) ^ int(np.uint32(y)))

    def rotl(self, x, amount: int) -> np.uint32:
        value = int(np.uint32(x)) & 0xFFFFFFFF
        amount = amount % self.word_size
        left = ((value << amount) & 0xFFFFFFFF)
        right = (value >> (self.word_size - amount))
        return np.uint32((left | right) & 0xFFFFFFFF)

    def add_mod_2_32(self, *ops) -> np.uint32:
        acc = 0
        for op in ops:
            acc = (acc + int(np.uint32(op))) & 0xFFFFFFFF
        return np.uint32(acc)

    def f(self, x, y, z):
        return self.bit_or(self.bit_and(x, y), self.bit_and(self.bit_not(x), z))

    def g(self, x, y, z):
        return self.bit_or(self.bit_and(x, z), self.bit_and(y, self.bit_not(z)))

    def h(self, x, y, z):
        return self.bit_xor(x, self.bit_xor(y, z))

    def i(self, x, y, z):
        return self.bit_xor(y, self.bit_or(x, self.bit_not(z)))


class MD5(MD5Calc):
    """
    MD5 hashing class that mirrors the SHA-256 helper structure.
    """

    def __init__(self, is_verbose: bool = True, output_format=None) -> None:
        super().__init__()
        self.verbose_flag = is_verbose
        assert output_format is not None, "JSON Format must be specified."
        self.res_out = output_format
        if self.verbose_flag:
            print("MD5 loaded")
        self.reset()

    def reset(self) -> None:
        """Reset internal state for a fresh hash computation."""
        self.state = _INIT_STATE.copy()
        self.message: bytes = b""
        self.message_len: int = 0  # in bits
        self.block_n: int = 0
        self.message_blocks: List[np.ndarray] = []
        self.hash: np.ndarray | None = None

    def _log(self, method_name: str, *args, **kwargs) -> None:
        """Safely call OutputFormat helpers without letting them break hashing."""
        method = getattr(self.res_out, method_name, None)
        if method is None:
            return
        try:
            method(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if self.verbose_flag:
                print(f"[WARN] Logging skipped for {method_name}: {exc}")

    def pad(self) -> None:
        """Apply MD5 padding (little-endian length)."""
        msg = bytearray(self.message)
        msg.append(0x80)
        while (len(msg) * 8) % self.block_size != 448:
            msg.append(0x00)
        msg.extend(int(self.message_len).to_bytes(8, byteorder="little"))
        self.message = bytes(msg)

    def parse(self) -> None:
        """Split padded message into 512-bit blocks of 16 little-endian words."""
        words_le = np.frombuffer(self.message, dtype="<u4")
        self.block_n = words_le.size // 16
        self.message_blocks = [
            words_le[i * 16:(i + 1) * 16].astype(np.uint32, copy=True)
            for i in range(self.block_n)
        ]

    def preprocess(self) -> bool:
        """
        Padding & parsing for MD5, with optional verbose logging and OutputFormat logs.
        """
        self.message_blocks = []
        self.pad()
        self.parse()

        print("Preprocessing complete.")
        block_dict = {}
        if self.verbose_flag:
            for idx, block in enumerate(self.message_blocks):
                print(f"Block {idx}:")
                for word_idx, word in enumerate(block):
                    print(self.to_hex32(word), end=" ")
                    if (word_idx + 1) % 8 == 0:
                        print()
                print()

        for idx, block in enumerate(self.message_blocks):
            block_dict[f"Block {idx}"] = block.copy()
        self._log("add_preprocess", block_dict)
        return True

    def step1(self, iteration: int) -> np.ndarray:
        """Step 1: Load 16-word block."""
        print("Step 1: Message words")
        block = self.message_blocks[iteration]
        self._log("add_step1", block)
        if self.verbose_flag:
            print("Words:", " ".join(self.to_hex32(word) for word in block))
        return block

    def step2(self, current_state: Sequence[int]) -> List[np.uint32]:
        """Step 2: Initialize working variables."""
        print("Step 2: Initialize working variables")
        a, b, c, d = (self._to_uint32(val) for val in current_state)
        ret = [a, b, c, d]
        ret_dict = {name: self.to_hex32(val) for name, val in zip("abcd", ret)}
        self._log("add_step2", ret_dict)
        if self.verbose_flag:
            print(ret_dict)
        return ret

    def step3(self, block_words: np.ndarray, working: Sequence[np.uint32]) -> List[np.uint32]:
        """Step 3: Main compression function loop."""
        print("Step 3: Main compression function loop")
        a, b, c, d = (self._to_uint32(val) for val in working)
        for idx in range(64):
            if idx < 16:
                f_val = self.f(b, c, d)
                g_idx = idx
            elif idx < 32:
                f_val = self.g(b, c, d)
                g_idx = (5 * idx + 1) % 16
            elif idx < 48:
                f_val = self.h(b, c, d)
                g_idx = (3 * idx + 5) % 16
            else:
                f_val = self.i(b, c, d)
                g_idx = (7 * idx) % 16

            temp = self.add_mod_2_32(a, f_val, self.K[idx], block_words[g_idx])
            rotated = self.rotl(temp, int(self.shift_amounts[idx]))
            new_b = self.add_mod_2_32(b, rotated)
            a, b, c, d = d, new_b, b, c

            log_dict = {
                "a": self.to_hex32(a),
                "b": self.to_hex32(b),
                "c": self.to_hex32(c),
                "d": self.to_hex32(d),
                "f": self.to_hex32(f_val),
                "g_idx": self.to_hex32(g_idx),
                "m_g": self.to_hex32(block_words[g_idx]),
                "rot": self.to_hex32(rotated),
            }
            self._log("add_step3_round", idx, log_dict)

            loop_idx = idx + 1
            if self.verbose_flag:
                if loop_idx == 1:
                    suffix = "st"
                elif loop_idx == 2:
                    suffix = "nd"
                elif loop_idx == 3:
                    suffix = "rd"
                else:
                    suffix = "th"
                print(f"{loop_idx:02}{suffix} loop - {log_dict}")

        return [a, b, c, d]

    def step4(self, working: Sequence[np.uint32], prev_state: Sequence[int]) -> List[np.uint32]:
        """Step 4: Feed-forward addition."""
        print("Step 4: Finalize the hash value")
        result = [
            self.add_mod_2_32(working[0], prev_state[0]),
            self.add_mod_2_32(working[1], prev_state[1]),
            self.add_mod_2_32(working[2], prev_state[2]),
            self.add_mod_2_32(working[3], prev_state[3]),
        ]
        ret_dict = {name: self.to_hex32(val) for name, val in zip("abcd", result)}
        self._log("add_step4", ret_dict)
        if self.verbose_flag:
            print(ret_dict)
        return result

    def compute_hash(self) -> bool:
        """Run all blocks through the MD5 compression function."""
        try:
            for block_idx in range(self.block_n):
                print(
                    f"\nProcessing {block_idx + 1} block of {self.block_n} blocks "
                    f"({block_idx / max(self.block_n, 1) * 100:.2f}%)"
                )
                block = self.step1(block_idx)
                working = self.step2(self.state.tolist())
                out_state = self.step3(block, working)
                self.state = np.array(self.step4(out_state, self.state.tolist()), dtype=np.uint32)
                self.hash = self.state.copy()
                self._log("add_round", block_idx)
            return True
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"Error during hash computation: {exc}")
            return False

    def finalize(self) -> np.ndarray:
        """Return the final MD5 digest words in big-endian order."""
        assert self.hash is not None, "No hash computed."
        little_state = np.array([np.uint32(x) for x in self.hash], dtype=np.uint32)
        digest_bytes = b"".join(int(word).to_bytes(4, "little") for word in little_state)
        words_be = np.frombuffer(digest_bytes, dtype=">u4").astype(np.uint32, copy=True)
        return words_be

    def digest(self, message: bytes | bytearray | None = None) -> np.ndarray:
        """Compute the MD5 digest for a single message."""
        assert message is not None, "Message must be provided."
        assert isinstance(message, (bytes, bytearray)), "Message must be bytes or bytearray."

        self.reset()
        try:
            self.res_out.reset(only_step=True)
            if hasattr(self.res_out, "generated_hash"):
                self.res_out.generated_hash = ""
            if hasattr(self.res_out, "correct_hash"):
                self.res_out.correct_hash = ""
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if self.verbose_flag:
                print(f"[WARN] Failed to reset output formatter: {exc}")

        self.message = bytes(message)
        self.message_len = len(self.message) * 8
        assert self.message_len > 0, "Message length must be positive."

        preprocess_success = self.preprocess()
        print("Preprocessing successful" if preprocess_success else "Preprocessing failed")
        compute_success = self.compute_hash()
        print("Computation successful\n" if compute_success else "Computation failed\n")

        if preprocess_success and compute_success:
            return self.finalize()

        raise RuntimeError("Hash computation failed.")

    def hexdigest(self, message: bytes | bytearray | None = None) -> str:
        """Return the hexadecimal MD5 digest for convenience."""
        state_words = self.digest(message)
        digest_bytes = b"".join(int(word).to_bytes(4, "big") for word in state_words)
        return digest_bytes.hex()
