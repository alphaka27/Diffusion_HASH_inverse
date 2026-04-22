"""
Implementation of a generic Sponge structure for hashing.
"""

from __future__ import annotations

from typing import Callable, Sequence


PermutationFn = Callable[[bytes], bytes | bytearray]
__all__ = ["PermutationFn", "Sponge", "toy_permutation"]


def _rotl8(value: int, shift: int) -> int:
    """Rotate an 8-bit value to the left."""
    shift %= 8
    return ((value << shift) | (value >> (8 - shift))) & 0xFF


def toy_permutation(state: bytes, rounds: int = 8) -> bytes:
    """
    Small permutation-like mixing function for demos and tests.

    This is not cryptographically secure. It only exists so that
    `sponge.py` can be exercised without a full Keccak permutation.
    """
    if not isinstance(state, bytes):
        raise TypeError("state must be bytes")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if len(state) == 0:
        raise ValueError("state must not be empty")

    mixed = bytearray(state)
    width = len(mixed)

    for rnd in range(rounds):
        shift = (rnd % width) or 1
        mixed = mixed[shift:] + mixed[:shift]

        prev = mixed[-1]
        for idx in range(width):
            current = mixed[idx]
            constant = (31 * idx + 17 * rnd + width) & 0xFF
            mixed[idx] = _rotl8(current ^ constant ^ prev, (idx + rnd) % 8)
            prev = current

        mixed[0] ^= (0xA5 + rnd) & 0xFF
        mixed[-1] = _rotl8(mixed[-1], (rnd % 7) + 1)

    return bytes(mixed)


class Sponge:
    """
    Generic byte-aligned Sponge construction.

    The provided `permutation` must accept and return exactly
    `rate_bytes + capacity_bytes` bytes. The security of the sponge
    depends entirely on that permutation.
    """

    def __init__(
        self,
        rate_bits: int,
        capacity_bits: int,
        output_bits: int,
        permutation: PermutationFn,
        delimited_suffix: int = 0x01,
    ) -> None:
        if rate_bits <= 0 or rate_bits % 8 != 0:
            raise ValueError("rate_bits must be a positive multiple of 8")
        if capacity_bits <= 0 or capacity_bits % 8 != 0:
            raise ValueError("capacity_bits must be a positive multiple of 8")
        if output_bits <= 0 or output_bits % 8 != 0:
            raise ValueError("output_bits must be a positive multiple of 8")
        if not callable(permutation):
            raise TypeError("permutation must be callable")
        if not 0 <= delimited_suffix <= 0xFF:
            raise ValueError("delimited_suffix must fit in one byte")

        self.rate_bits = rate_bits
        self.capacity_bits = capacity_bits
        self.output_bits = output_bits
        self.rate_bytes = rate_bits // 8
        self.capacity_bytes = capacity_bits // 8
        self.state_bytes = self.rate_bytes + self.capacity_bytes
        self.output_bytes = output_bits // 8
        self.permutation = permutation
        self.delimited_suffix = delimited_suffix

        self.state = bytearray(self.state_bytes)

    def reset(self) -> None:
        """Reset the internal state to all-zero."""
        self.state = bytearray(self.state_bytes)

    def _permute(self, state: bytes | bytearray) -> bytes:
        """Run the external permutation and validate its output shape."""
        result = self.permutation(bytes(state))
        if not isinstance(result, (bytes, bytearray)):
            raise TypeError("permutation must return bytes or bytearray")
        if len(result) != self.state_bytes:
            raise ValueError(
                "permutation output length must match the sponge state size"
            )
        return bytes(result)

    def step1(self, data: bytes) -> bytes:
        """
        Apply byte-aligned multi-rate padding.

        This is the usual "append delimiter, zero pad, set the last bit"
        form of `pad10*1`, restricted to byte-aligned inputs.
        """
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")

        padded = bytearray(data)
        padded.append(self.delimited_suffix)
        padded.extend(b"\x00" * ((-len(padded)) % self.rate_bytes))
        padded[-1] |= 0x80
        return bytes(padded)

    def step2(self, data: bytes) -> Sequence[bytes]:
        """Split padded input into `rate_bytes`-sized blocks."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if len(data) == 0 or len(data) % self.rate_bytes != 0:
            raise ValueError("padded data length must be a multiple of rate_bytes")

        return [
            data[index:index + self.rate_bytes]
            for index in range(0, len(data), self.rate_bytes)
        ]

    def step3(self) -> bytes:
        """Initialize and return an all-zero sponge state."""
        self.reset()
        return bytes(self.state)

    def step4(
        self,
        blocks: Sequence[bytes],
        state: bytes | bytearray | None = None,
    ) -> bytes:
        """
        Absorb each rate block into the state and permute after every block.
        """
        if state is None:
            working_state = bytearray(self.step3())
        else:
            if len(state) != self.state_bytes:
                raise ValueError("state length must match the sponge state size")
            working_state = bytearray(state)

        for block in blocks:
            if not isinstance(block, bytes):
                raise TypeError("each block must be bytes")
            if len(block) != self.rate_bytes:
                raise ValueError("each block length must match rate_bytes")

            for index, value in enumerate(block):
                working_state[index] ^= value

            working_state = bytearray(self._permute(working_state))

        self.state = working_state
        return bytes(self.state)

    def step5(
        self,
        state: bytes | bytearray | None = None,
        output_bytes: int | None = None,
    ) -> bytes:
        """
        Squeeze bytes from the state, permuting again when more output is needed.
        """
        if output_bytes is None:
            output_bytes = self.output_bytes
        if output_bytes <= 0:
            raise ValueError("output_bytes must be positive")

        if state is None:
            if len(self.state) != self.state_bytes:
                raise ValueError("internal state is not initialized")
            working_state = bytearray(self.state)
        else:
            if len(state) != self.state_bytes:
                raise ValueError("state length must match the sponge state size")
            working_state = bytearray(state)

        output = bytearray()
        while len(output) < output_bytes:
            take = min(self.rate_bytes, output_bytes - len(output))
            output.extend(working_state[:take])
            if len(output) < output_bytes:
                working_state = bytearray(self._permute(working_state))

        self.state = working_state
        return bytes(output)

    def digest(self, data: bytes) -> bytes:
        """Hash input data using the Sponge absorb/squeeze pipeline."""
        padded = self.step1(data)
        blocks = self.step2(padded)
        state = self.step3()
        absorbed_state = self.step4(blocks, state)
        return self.step5(absorbed_state)

    def hexdigest(self, data: bytes) -> str:
        """Return the sponge digest as a hexadecimal string."""
        return self.digest(data).hex()


if __name__ == "__main__":
    sponge = Sponge(
        rate_bits=64,
        capacity_bits=64,
        output_bits=256,
        permutation=toy_permutation,
    )

    message = b"abc"
    print("message:", message)
    print("digest :", sponge.hexdigest(message))
