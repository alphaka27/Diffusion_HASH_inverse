"""SHA-256 execution-trace implementation."""

from __future__ import annotations

import struct

from .common import _MASK_32, _rotate_right, _word
from .constants import SHA256_CONSTANTS, SHA256_INITIAL_STATE
from .trace import HashTracer, Trace


def _working_state(values: tuple[int, ...]) -> dict[str, str]:
    return {name: _word(value) for name, value in zip("abcdefgh", values)}


def _hash_state(values: tuple[int, ...]) -> dict[str, str]:
    return {f"H{index}": _word(value) for index, value in enumerate(values)}


class SHA256Tracer(HashTracer):
    """SHA-256 implementation that records schedule expansion and 64 rounds."""

    algorithm = "sha256"

    def trace(self, message: bytes) -> Trace:
        bit_length = len(message) * 8
        padded = bytearray(message)
        padded.append(0x80)
        padded.extend(b"\x00" * ((56 - len(padded) % 64) % 64))
        padded.extend(struct.pack(">Q", bit_length & 0xFFFFFFFFFFFFFFFF))

        hash_state = SHA256_INITIAL_STATE
        blocks: list[dict[str, object]] = []
        for block_index in range(0, len(padded), 64):
            words = list(struct.unpack(">16I", padded[block_index:block_index + 64]))
            schedule_expansion: list[dict[str, object]] = []
            for index in range(16, 64):
                sigma0 = (
                    _rotate_right(words[index - 15], 7)
                    ^ _rotate_right(words[index - 15], 18)
                    ^ (words[index - 15] >> 3)
                )
                sigma1 = (
                    _rotate_right(words[index - 2], 17)
                    ^ _rotate_right(words[index - 2], 19)
                    ^ (words[index - 2] >> 10)
                )
                value = (words[index - 16] + sigma0 + words[index - 7] + sigma1) & _MASK_32
                words.append(value)
                schedule_expansion.append(
                    {
                        "word_index": index,
                        "sigma0": _word(sigma0),
                        "sigma1": _word(sigma1),
                        "word_minus_16": _word(words[index - 16]),
                        "word_minus_7": _word(words[index - 7]),
                        "value": _word(value),
                    }
                )

            a, b, c, d, e, f, g, h = hash_state
            initial_working_state = (a, b, c, d, e, f, g, h)
            compression_rounds: list[dict[str, object]] = []
            for round_index in range(64):
                state_before = (a, b, c, d, e, f, g, h)
                sigma1 = _rotate_right(e, 6) ^ _rotate_right(e, 11) ^ _rotate_right(e, 25)
                choice = (e & f) ^ (~e & g)
                temp1 = (h + sigma1 + choice + SHA256_CONSTANTS[round_index] + words[round_index]) & _MASK_32
                sigma0 = _rotate_right(a, 2) ^ _rotate_right(a, 13) ^ _rotate_right(a, 22)
                majority = (a & b) ^ (a & c) ^ (b & c)
                temp2 = (sigma0 + majority) & _MASK_32
                a, b, c, d, e, f, g, h = (
                    (temp1 + temp2) & _MASK_32,
                    a,
                    b,
                    c,
                    (d + temp1) & _MASK_32,
                    e,
                    f,
                    g,
                )
                compression_rounds.append(
                    {
                        "round": round_index,
                        "message_schedule_word": _word(words[round_index]),
                        "constant": _word(SHA256_CONSTANTS[round_index]),
                        "sigma0": _word(sigma0),
                        "sigma1": _word(sigma1),
                        "choice": _word(choice),
                        "majority": _word(majority),
                        "temp1": _word(temp1),
                        "temp2": _word(temp2),
                        "state_before": _working_state(state_before),
                        "state_after": _working_state((a, b, c, d, e, f, g, h)),
                    }
                )

            hash_state = tuple(
                (previous + current) & _MASK_32
                for previous, current in zip(hash_state, (a, b, c, d, e, f, g, h))
            )
            blocks.append(
                {
                    "index": block_index // 64,
                    "initial_words": [_word(word) for word in words[:16]],
                    "message_schedule_expansion": schedule_expansion,
                    "message_schedule": [_word(word) for word in words],
                    "state_before": _hash_state(initial_working_state),
                    "working_state_before_feed_forward": _working_state((a, b, c, d, e, f, g, h)),
                    "compression_rounds": compression_rounds,
                    "state_after": _hash_state(hash_state),
                }
            )

        digest = b"".join(word.to_bytes(4, "big") for word in hash_state).hex()
        return {
            "schema_version": 1,
            "algorithm": self.algorithm,
            "preprocessing": {
                "endianness": "big",
                "original_bit_length": bit_length,
                "padded_message_hex": padded.hex(),
                "block_count": len(blocks),
            },
            "intermediate": {"initial_state": _hash_state(SHA256_INITIAL_STATE), "blocks": blocks},
            "digest": digest,
        }
