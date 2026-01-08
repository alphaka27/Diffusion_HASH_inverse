from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class CubeCoordinate:
    """3D axis-aligned box in integer RGB space."""
    # inclusive integer bounds
    r_min: int
    r_max: int
    g_min: int
    g_max: int
    b_min: int
    b_max: int

    """
    Returns RGB Color Space coordinates as inclusive and half-open bounds, and size per axis.
    """
    def as_inclusive(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Returns the inclusive bounds as tuples.

        :return: A tuple containing the inclusive bounds for R, G, and B axes.
        :rtype: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
        """
        return (self.r_min, self.r_max), (self.g_min, self.g_max), (self.b_min, self.b_max)

    def as_half_open(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Returns the half-open bounds as tuples.

        :return: A tuple containing the half-open bounds for R, G, and B axes.
        :rtype: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
        """
        # [min, max+1)
        return (self.r_min, self.r_max + 1), (self.g_min, self.g_max + 1), \
            (self.b_min, self.b_max + 1)

    def size_per_axis(self) -> Tuple[int, int, int]:
        """
        Returns the size (number of integer points) along each axis.

        :return: A tuple containing the size along R, G, and B axes.
        :rtype: Tuple[int, int, int]
        """
        # number of integer points on each axis
        return (self.r_max - self.r_min + 1, self.g_max - self.g_min + 1, \
                self.b_max - self.b_min + 1)

class Byte2RGB:
    """
    A class to convert byte values(0x00 ~ 0xFF) to RGB color tuples.
    """

    def __init__(self):
        self.full_space_min: int = 0
        self.full_space_max: int = 255
        self.sub_len: int = 252
        self.split: int = 7

    def centered_subspace_1d(self) -> Tuple[int, int]:
        """
        full range is inclusive [full_min, full_max].
        returns centered inclusive [sub_min, sub_max] with exactly sub_len integer values.
        Returns: 
            Tuple[int, int]
            - sub_min: Minimum value of the centered subspace.
            - sub_max: Maximum value of the centered subspace.
        """
        full_len = self.full_space_max - self.full_space_min + 1
        if self.sub_len <= 0:
            raise ValueError("sub_len must be positive.")
        if self.sub_len > full_len:
            raise ValueError(f"sub_len({self.sub_len}) cannot exceed full length({full_len}).")

        # leftover points split equally on both sides if possible
        leftover = full_len - self.sub_len
        assert leftover >= 0, "Logic error: leftover should be non-negative."
        assert leftover % 1 == 0, "Logic error: leftover should be integer."
        assert leftover % 2 == 0, "Logic error: leftover should be even."
        left_pad = right_pad = leftover // 2

        sub_min = self.full_space_min + left_pad
        sub_max = self.full_space_max - right_pad
        # sanity
        if sub_max - sub_min + 1 != self.sub_len:
            raise RuntimeError("Internal error: computed subspace length mismatch.")
        return sub_min, sub_max


    def split_1d_inclusive(self, start: int, end: int, parts: int) -> List[Tuple[int, int]]:
        """
        Split inclusive [start, end] into `parts` equal-size inclusive chunks.
        Requires divisibility in integer-point counts.
        """
        if parts <= 0:
            raise ValueError("parts must be positive.")
        length = end - start + 1
        if length % parts != 0:
            raise ValueError(f"Cannot split length {length} into {parts} equal parts.")
        step = length // parts  # points per chunk
        chunks = []
        for i in range(parts):
            a = start + step * i
            b = a + step - 1
            chunks.append((a, b))
        # sanity
        if chunks[0][0] != start or chunks[-1][1] != end:
            raise RuntimeError("Internal error: split does not cover full range.")
        return chunks


    def rgb_subcubes(
        sub_len: int,
        split: int,
        full_min: int = 0,
        full_max: int = 255,
    ) -> Dict[str, object]:
        """
        Returns:
        - 'subspace_inclusive': (r_min,r_max),(g_min,g_max),(b_min,b_max)
        - 'subspace_half_open': [r_min,r_max), ...
        - 'cube_edge_len': edge length in integer points (per axis)
        - 'cubes': list of dicts with index (i,j,k) and Box3D bounds
        """
        r0, r1 = centered_subspace_1d(full_min, full_max, sub_len)
        g0, g1 = centered_subspace_1d(full_min, full_max, sub_len)
        b0, b1 = centered_subspace_1d(full_min, full_max, sub_len)

        r_chunks = split_1d_inclusive(r0, r1, split)
        g_chunks = split_1d_inclusive(g0, g1, split)
        b_chunks = split_1d_inclusive(b0, b1, split)

        edge_len = r_chunks[0][1] - r_chunks[0][0] + 1  # same for all axes

        cubes: List[Dict[str, object]] = []
        for i, (ra, rb) in enumerate(r_chunks):
            for j, (ga, gb) in enumerate(g_chunks):
                for k, (ba, bb) in enumerate(b_chunks):
                    box = CubeCoordinate(ra, rb, ga, gb, ba, bb)
                    cubes.append(
                        {
                            "idx": (i, j, k),
                            "inclusive": box.as_inclusive(),
                            "half_open": box.as_half_open(),
                        }
                    )

        return {
            "subspace_inclusive": ((r0, r1), (g0, g1), (b0, b1)),
            "subspace_half_open": ((r0, r1 + 1), (g0, g1 + 1), (b0, b1 + 1)),
            "cube_edge_len": edge_len,
            "num_cubes": len(cubes),
            "cubes": cubes,
        }


if __name__ == "__main__":
    # 예시: 서브 길이 252, 각 축 7등분 -> 7^3 = 343개
    info = rgb_subcubes(sub_len=252, split=7)

    print("Subspace inclusive:", info["subspace_inclusive"])   # ((2,253),(2,253),(2,253))
    print("Subspace half-open:", info["subspace_half_open"])   # ((2,254),(2,254),(2,254))
    print("Cube edge len:", info["cube_edge_len"])             # 36
    print("Num cubes:", info["num_cubes"])                     # 343

    # 첫 번째/마지막 큐브 확인
    print("First cube:", info["cubes"][0])
    print("Last cube:", info["cubes"][-1])