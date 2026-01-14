"""
Type hint for RGB-related data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Tuple

@dataclass
class Chunk1D:
    """
    Represents a 1D chunk with start and end values.  
    Working with inclusive intervals [start, end].
    """
    start: int
    end: int

    def __post_init__(self):
        _min = 0
        _max = 255
        if self.end <= self.start:
            raise ValueError("End must be greater than start for a valid chunk.")

        if self.start < _min or self.end > _max:
            raise ValueError("Chunk values must be in the range [0, 255].")

        if self.start >= _max or self.end <= _min:
            raise ValueError("Chunk values must be in the range [0, 255].")

    @property
    def length(self) -> int:
        """
        Returns the size of the chunk.
        """
        return self.end - self.start + 1

    @property
    def as_inclusive(self) -> Tuple[int, int]:
        """
        Returns the chunk as an inclusive range [start, end].
        """
        return (self.start, self.end)

    @property
    def to_tuple(self) -> Tuple[int, int]:
        """
        Returns the chunk as a tuple (start, end).
        """
        return (self.start, self.end)

    @property
    def as_half_open(self) -> Tuple[int, int]:
        """
        Returns the chunk as a half-open range [start, end).
        """
        return (self.start, self.end + 1)

    @staticmethod
    def from_tuple(chunk_tuple: Tuple[int, int]) -> Chunk1D:
        """
        Creates a Chunk1D instance from a tuple.
        """
        return Chunk1D(start=chunk_tuple[0], end=chunk_tuple[1])

    @staticmethod
    def from_ints(start: int, end: int) -> Chunk1D:
        """
        Creates a Chunk1D instance from start and end integers.
        """
        return Chunk1D(start=start, end=end)

@dataclass
class RGBChunks:
    """
    Represents a chunk in RGB space with separate chunks for R, G, and B axes.

    Working with inclusive intervals [start, end].
    """
    r_chunk: Chunk1D
    g_chunk: Chunk1D
    b_chunk: Chunk1D

    @property
    def length(self) -> Tuple[int, int, int]:
        """
        Returns the size (number of integer points) along each axis in RGB space.
        """
        return (self.r_chunk.length, self.g_chunk.length, self.b_chunk.length)

    @property
    def volume(self) -> int:
        """
        Returns the total number of integer points in the RGB chunk.
        """
        r_len, g_len, b_len = self.length
        return r_len * g_len * b_len

    @property
    def as_half_open(self) -> Tuple[Chunk1D, Chunk1D, Chunk1D]:
        """
        Returns the RGB chunk as half-open intervals for each axis.
        """
        return (self.r_chunk.as_half_open,
                self.g_chunk.as_half_open,
                self.b_chunk.as_half_open)

    @property
    def as_inclusive(self) -> Tuple[Chunk1D, Chunk1D, Chunk1D]:
        """
        Returns the RGB chunk as inclusive intervals for each axis.
        """
        return (self.r_chunk.as_inclusive,
                self.g_chunk.as_inclusive,
                self.b_chunk.as_inclusive)

    @staticmethod
    def to_rgbchunks(start: RGB, end: RGB) -> Tuple[Chunk1D, Chunk1D, Chunk1D]:
        """
        Creates a Chunk1D instance from RGB start and end values.
        """
        r_chunk = Chunk1D(start=start.r, end=end.r)
        g_chunk = Chunk1D(start=start.g, end=end.g)
        b_chunk = Chunk1D(start=start.b, end=end.b)
        return r_chunk, g_chunk, b_chunk

@dataclass
class RGB:
    """
    Coordinates of a point in RGB space.
    """
    r: int
    g: int
    b: int

    @property
    def as_tuple(self) -> Tuple[int, int, int]:
        """
        Returns the RGB coordinates as a tuple.
        """
        return (self.r, self.g, self.b)

    @staticmethod
    def from_tuple(rgb_tuple: Tuple[int, int, int]) -> RGB:
        """
        Creates an RGB instance from a tuple.
        """
        return RGB(r=rgb_tuple[0], g=rgb_tuple[1], b=rgb_tuple[2])

    @staticmethod
    def to_rgb(
        r_chunk: Chunk1D,
        g_chunk: Chunk1D,
        b_chunk: Chunk1D) \
            -> Tuple[RGB, RGB]:
        """
        Creates an RGB instances from Chunk1D instances.

        Working with half-open intervals [start, end).
        """
        _start = RGB(r=r_chunk.start, g=g_chunk.start, b=b_chunk.start)
        _end = RGB(r=r_chunk.end, g=g_chunk.end, b=b_chunk.end)
        return _start, _end

    @staticmethod
    def cube_coord(
        r_chunk: Chunk1D,
        g_chunk: Chunk1D,
        b_chunk: Chunk1D) \
            -> Tuple[RGB, RGB, RGB, RGB, RGB, RGB, RGB, RGB]:
        """
        Returns the 8 corner coordinates of the subspace in RGB space.

        Working with inclusive intervals [start, end].
        """

        if isinstance(r_chunk, (tuple, list)):
            r_chunk = Chunk1D.from_tuple(r_chunk)
        if isinstance(g_chunk, (tuple, list)):
            g_chunk = Chunk1D.from_tuple(g_chunk)
        if isinstance(b_chunk, (tuple, list)):
            b_chunk = Chunk1D.from_tuple(b_chunk)

        assert isinstance(r_chunk, Chunk1D), "r_chunk must be of type Chunk1D."
        assert isinstance(g_chunk, Chunk1D), "g_chunk must be of type Chunk1D."
        assert isinstance(b_chunk, Chunk1D), "b_chunk must be of type Chunk1D."

        r_s, r_e = r_chunk.as_inclusive
        g_s, g_e = g_chunk.as_inclusive
        b_s, b_e = b_chunk.as_inclusive

        return (RGB(r=r_s, g=g_s, b=b_s),
                RGB(r=r_s, g=g_s, b=b_e),
                RGB(r=r_s, g=g_e, b=b_s),
                RGB(r=r_s, g=g_e, b=b_e),
                RGB(r=r_e, g=g_s, b=b_s),
                RGB(r=r_e, g=g_s, b=b_e),
                RGB(r=r_e, g=g_e, b=b_s),
                RGB(r=r_e, g=g_e, b=b_e))

@dataclass
class ReducedRGBChunks:
    """
    Reduced RGB coordinates with fewer bits per channel.
    """
    reduced_r_chunk: Chunk1D
    reduced_g_chunk: Chunk1D
    reduced_b_chunk: Chunk1D

    
