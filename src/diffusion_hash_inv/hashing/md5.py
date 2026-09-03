"""MD5 execution-trace implementation."""

from __future__ import annotations

import struct

from .common import _MASK_32, _rotate_left, _word
from .constants import MD5_CONSTANTS, MD5_INITIAL_STATE, MD5_SHIFTS
from .trace import HashTracer, Trace


def _state(values: tuple[int, int, int, int]) -> dict[str, str]:
    return {name: _word(value) for name, value in zip(("A", "B", "C", "D"), values)}


class MD5Tracer(HashTracer):
    """MD5 implementation that records padding, blocks, and all 64 steps."""

    algorithm = "md5"

    def trace(self, message: bytes) -> Trace:
        bit_length = len(message) * 8
        padded = bytearray(message)
        padded.append(0x80)
        padded.extend(b"\x00" * ((56 - len(padded) % 64) % 64))
        padded.extend(struct.pack("<Q", bit_length & 0xFFFFFFFFFFFFFFFF))

        state = MD5_INITIAL_STATE
        blocks: list[dict[str, object]] = []
        for block_index in range(0, len(padded), 64):
            words = struct.unpack("<16I", padded[block_index:block_index + 64])
            a, b, c, d = state
            initial_working_state = (a, b, c, d)
            rounds: list[dict[str, object]] = []

            for round_index in range(4):
                steps: list[dict[str, object]] = []
                for step in range(round_index * 16, (round_index + 1) * 16):
                    state_before = (a, b, c, d)
                    if step < 16:
                        function_name = "F"
                        function_output = (b & c) | (~b & d)
                        word_index = step
                    elif step < 32:
                        function_name = "G"
                        function_output = (d & b) | (~d & c)
                        word_index = (5 * step + 1) % 16
                    elif step < 48:
                        function_name = "H"
                        function_output = b ^ c ^ d
                        word_index = (3 * step + 5) % 16
                    else:
                        function_name = "I"
                        function_output = c ^ (b | ~d)
                        word_index = (7 * step) % 16

                    function_output &= _MASK_32
                    total = (a + function_output + MD5_CONSTANTS[step] + words[word_index]) & _MASK_32
                    a, b, c, d = d, (b + _rotate_left(total, MD5_SHIFTS[step])) & _MASK_32, b, c
                    steps.append(
                        {
                            "step": step,
                            "function": function_name,
                            "message_word_index": word_index,
                            "message_word": _word(words[word_index]),
                            "constant": _word(MD5_CONSTANTS[step]),
                            "shift": MD5_SHIFTS[step],
                            "function_output": _word(function_output),
                            "sum_before_rotation": _word(total),
                            "state_before": _state(state_before),
                            "state_after": _state((a, b, c, d)),
                        }
                    )
                rounds.append({"round": round_index + 1, "steps": steps})

            state = tuple((left + right) & _MASK_32 for left, right in zip(state, (a, b, c, d)))
            blocks.append(
                {
                    "index": block_index // 64,
                    "words": [_word(word) for word in words],
                    "state_before": _state(initial_working_state),
                    "rounds": rounds,
                    "working_state_before_feed_forward": _state((a, b, c, d)),
                    "state_after": _state(state),
                }
            )

        digest = b"".join(word.to_bytes(4, "little") for word in state).hex()
        return {
            "schema_version": 1,
            "algorithm": self.algorithm,
            "preprocessing": {
                "endianness": "little",
                "original_bit_length": bit_length,
                "padded_message_hex": padded.hex(),
                "block_count": len(blocks),
            },
            "intermediate": {"initial_state": _state(MD5_INITIAL_STATE), "blocks": blocks},
            "digest": digest,
        }
