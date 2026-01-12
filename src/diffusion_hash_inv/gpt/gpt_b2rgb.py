"""
Defines RGB color space subcubes and provides utilities to convert byte values to RGB tuples.
Each subcube represents a partition of the RGB color space.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from typing import NamedTuple

class Chunk1D(NamedTuple):
    """
    Represents a 1D chunk with start and end values.
    """

    start: int
    end: int

class RGBChunk(NamedTuple):
    """
    Represents an RGB color with red, green, and blue chunk components.

    Details:
        - r: Chunk1D representing the red component range.
        - g: Chunk1D representing the green component range.
        - b: Chunk1D representing the blue component range.
    """

    r: Chunk1D
    g: Chunk1D
    b: Chunk1D

class RGB(NamedTuple):
    """
    Represents an RGB color with red, green, and blue coordinate components.
    """

    coord: Tuple[int, int, int]

class RGBVertex(NamedTuple):
    """
    Represents the 8 corner vertices of a cube in RGB space.
    """

    p0: RGBChunk
    p1: RGBChunk
    p2: RGBChunk
    p3: RGBChunk
    p4: RGBChunk
    p5: RGBChunk
    p6: RGBChunk
    p7: RGBChunk

class RGB3D(NamedTuple):
    """
    Represents a 3D RGB space with red, green, and blue chunk components.
    """

    coord: tuple[RGB, ...]

class CubeBounds:
    """
    Information about RGB subcubes in a reduced color space.
    """

    @staticmethod
    def _input_handling_1d(start: Optional[int], end: Optional[int]) \
        -> Chunk1D:
        if start is not None:
            _start = start
        else:
            raise ValueError("start must be provided.")
        if end is not None:
            _end = end
        else:
            raise ValueError("end must be provided.")
        return Chunk1D(_start, _end)

    @staticmethod
    def as_inclusive_1d(start: Optional[int] = None, end: Optional[int] = None) \
        -> Chunk1D:
        """
        Returns the inclusive bounds of the cube.

        :return: A tuple containing the inclusive bounds (min, max).
        :rtype: Tuple[int, int]
        """
        _start, _end = CubeBounds._input_handling_1d(start, end)

        return Chunk1D(_start, _end)

    @staticmethod
    def as_half_open_1d(start: Optional[int] = None, end: Optional[int] = None) \
        -> Chunk1D:
        """
        Returns the half-open bounds of the cube.
        :return: A tuple containing the half-open bounds [min, max+1).
        :rtype: Tuple[int, int]
        """
        _start, _end = CubeBounds._input_handling_1d(start, end)

        return Chunk1D(_start, _end + 1)

    @staticmethod
    def size_1d(start: Optional[int] = None, end: Optional[int] = None) -> int:
        """
        Returns the size (number of integer points) of the cube.

        :return: The size of the cube.
        :rtype: int
        """
        _start, _end = CubeBounds._input_handling_1d(start, end)

        return _end - _start + 1

    @staticmethod
    def as_inclusive_3d(
        r_bounds: Tuple[int, int],
        g_bounds: Tuple[int, int],
        b_bounds: Tuple[int, int]
    ) -> RGBChunk:
        """
        Returns the inclusive bounds of the cube in 3D RGB space.

        :return: A tuple containing the inclusive bounds for R, G, and B axes.
        :rtype: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
        """

        # [min, max]
        return RGBChunk(CubeBounds.as_inclusive_1d(*r_bounds),
                        CubeBounds.as_inclusive_1d(*g_bounds),
                        CubeBounds.as_inclusive_1d(*b_bounds))

    @staticmethod
    def as_half_open_3d(
        r_bounds: Tuple[int, int],
        g_bounds: Tuple[int, int],
        b_bounds: Tuple[int, int],
    ) -> RGBChunk:
        """
        Returns the half-open bounds of the cube in 3D RGB space.

        :return: A tuple containing the half-open bounds for R, G, and B axes.
        :rtype: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
        """

        # [min, max+1)
        return RGBChunk(CubeBounds.as_half_open_1d(*r_bounds),
                        CubeBounds.as_half_open_1d(*g_bounds),
                        CubeBounds.as_half_open_1d(*b_bounds))

    @staticmethod
    def size_3d(
        r_bounds: Chunk1D,
        g_bounds: Chunk1D,
        b_bounds: Chunk1D,
    ) -> Tuple[int, int, int]:
        """
        Returns the size (number of integer points) along each axis in 3D RGB space.

        :return: A tuple containing the size along R, G, and B axes.
        :rtype: Tuple[int, int, int]
        """

        # number of integer points on each axis
        return (r_bounds.start - r_bounds.end + 1, g_bounds.start - g_bounds.end + 1, \
                b_bounds.start - b_bounds.end + 1)

    @staticmethod
    def cube_coords(r_chunk: Chunk1D,
                    g_chunk: Chunk1D,
                    b_chunk: Chunk1D) \
                    -> Tuple[RGB, RGB]:
        """
        Returns the 8 corner coordinates of the subspace in RGB space.

        :return: A tuple containing the RGB start and end tuples.
        :rtype: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        """

        if len(r_chunk) != 2 or len(g_chunk) != 2 or len(b_chunk) != 2:
            raise ValueError("Each chunk must be containing exactly 2 elements (start, end).")

        r_s, r_e = r_chunk
        g_s, g_e = g_chunk
        b_s, b_e = b_chunk

        coords = tuple(
            (r, g, b)
            for r in (r_s, r_e)
            for g in (g_s, g_e)
            for b in (b_s, b_e)
        )

        assert len(coords) == 8, "There should be exactly 8 corner coordinates."

        return coords

    def coord_valid(self, rgb_coord: Tuple[int, int, int]) -> bool:
        """
        Validates the coordinate values.
        """



class Byte2RGB(CubeBounds):
    """
    A class to convert byte values(0x00 ~ 0xFF) to RGB tuples in  RGB color space.
    """

    def __init__(self, full_space_min: int = 0, full_space_max: int = 255, \
                sub_len: int = 36, split: int = 7):
        self.full_space_min: int = full_space_min
        self.full_space_max: int = full_space_max
        self.sub_len: int = sub_len
        self.split: int = split

        self.full_len = self.size_1d(self.full_space_min, self.full_space_max)
        self.tot_sub_len = self.sub_len * self.split
        if self.tot_sub_len > self.full_len:
            _t = self.tot_sub_len
            _f = self.full_len
            raise ValueError(f"Requested subspace length({_t}) exceeds full space length({_f}).")

    def centered_subspace_1d(self) -> Tuple[int, int]:
        """
        full range is inclusive [full_min, full_max].
        returns centered inclusive [sub_min, sub_max] with exactly sub_len integer values.

        Returns:
            - sub_min: Minimum value of the centered subspace.
            - sub_max: Maximum value of the centered subspace.
        """

        if self.sub_len <= 0:
            raise ValueError("space_len must be positive.")

        # leftover points split equally on both sides if possible
        leftover = self.full_len - self.tot_sub_len

        if leftover < 0:
            raise RuntimeError("Internal error: leftover should be non-negative.")
        if leftover % 2 != 0:
            raise RuntimeError("Internal error: leftover should be even.")
        if leftover % 1 != 0:
            raise RuntimeError("Internal error: leftover should be integer.")

        left_pad = right_pad = leftover // 2

        sub_min = self.full_space_min + left_pad
        sub_max = self.full_space_max - right_pad
        _sub_len = self.size_1d(sub_min, sub_max)

        if _sub_len != self.sub_len * self.split:
            _t = self.sub_len * self.split
            _s = _sub_len
            raise RuntimeError\
                (f"Internal error: computed subspace length({_s}) mismatch({_t}).")
        return sub_min, sub_max

    def split_1d_inclusive(self) -> List[Tuple[int, int]]:
        """
        Split inclusive [start, end] into `parts` equal-size inclusive chunks.
        Requires divisibility in integer-point counts.
        """

        start, end = self.centered_subspace_1d()

        if self.tot_sub_len % self.split != 0:
            raise ValueError("Cannot evenly split the range into the specified parts.")

        chunk_len = self.tot_sub_len // self.split

        chunks: List[Tuple[int, int]] = []
        for i in range(self.split):
            chunk_start = start + i * chunk_len
            chunk_end = chunk_start + chunk_len - 1  # inclusive
            chunks.append((chunk_start, chunk_end))

        if len(chunks) != self.split:
            raise RuntimeError("Internal error: number of chunks mismatch.")
        if chunks[0][0] != start or chunks[-1][1] != end:
            raise RuntimeError("Internal error: chunk boundaries mismatch.")

        return chunks

    def rgb_subcubes(self) -> Dict[str, object]:
        """
        Returns:
            A dictionary containing information about RGB subcubes.
        """
        r_chunks = self.split_1d_inclusive()
        g_chunks = self.split_1d_inclusive()
        b_chunks = self.split_1d_inclusive()

        cubes: List[Dict[str, object]] = []
        for i, (ra, rb) in enumerate(r_chunks):
            for j, (ga, gb) in enumerate(g_chunks):
                for k, (ba, bb) in enumerate(b_chunks):

                    cubes.append(
                        {
                            "idx": (i, j, k),
                            "inclusive": self.as_inclusive_3d(
                                (ra, rb), (ga, gb), (ba, bb)
                            ),
                            "half_open": self.as_half_open_3d(
                                (ra, rb), (ga, gb), (ba, bb)
                            )
                        }
                    )

        return {
            "num_cubes": len(cubes),
            "cubes": cubes,
        }

    def exclude_cubes(self, rgb_cube: Optional[Dict[str, object]] = None):
        """
        Placeholder for excluding a box from the RGB space.
        """
        cubes = self.rgb_subcubes()["cubes"] if rgb_cube is None else [rgb_cube]
        cubes_idx = [cubes[i]["idx"] for i in range(len(cubes))]
        excluded_idx: List[Dict[str, object]] = []

        # Exclude each cubes following edge cases & vertex cases
        for cube in cubes:
            pass

        # Exclude each cube following specific criteria
        for cube in cubes:
            pass




if __name__ == "__main__":
    # Example usage
    # Each Subspace: size 36^3 Cubes, Total 7x7x7=343 Cubes
    byte2rgb = Byte2RGB()
    info = byte2rgb.rgb_subcubes()

    print("Num cubes:", info["num_cubes"]) # 343

    # 첫 번째/마지막 큐브 확인
    print("First cube:", info["cubes"][0])
    print("Last cube:", info["cubes"][-1])
