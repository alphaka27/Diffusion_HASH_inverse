"""
Type hint for RGB-related data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Tuple, List, Dict, ClassVar, Optional

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
    def as_inclusive(self) -> Chunk1D:
        """
        Returns the chunk as an inclusive range [start, end].
        """
        return Chunk1D(self.start, self.end)

    @property
    def inclusive_tuple(self) -> Tuple[int, int]:
        """
        Returns the chunk as a tuple (start, end).
        """
        return (self.start, self.end)

    @property
    def as_half_open(self) -> Chunk1D:
        """
        Returns the chunk as a half-open range [start, end).
        """
        return Chunk1D(self.start, self.end + 1)

    @property
    def half_open_tuple(self) -> Tuple[int, int]:
        """
        Returns the chunk as a tuple (start, end) in half-open format.
        """
        return (self.start, self.end + 1)

    @staticmethod
    def from_tuple(chunk_tuple: Tuple[int, int]) -> Chunk1D:
        """
        Creates a Chunk1D instance from a tuple.
        """
        return Chunk1D(start=chunk_tuple[0], end=chunk_tuple[1])

    def is_in_chunk(self, value: int) -> bool:
        """
        Checks if a value is within the chunk (inclusive).
        """
        return self.start <= value <= self.end

@dataclass
class RGBChunk:
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

    @property
    def as_chunks(self) -> Tuple[Chunk1D, Chunk1D, Chunk1D]:
        """
        Extracts the Chunk1D instances from an RGBChunk.
        """
        return (self.r_chunk, self.g_chunk, self.b_chunk)

    @staticmethod
    def rgb_to_chunks(start: RGB, end: RGB) -> Tuple[Chunk1D, Chunk1D, Chunk1D]:
        """
        Creates a Chunk1D instance from RGB start and end values.
        """
        r_chunk = Chunk1D(start=start.r, end=end.r)
        g_chunk = Chunk1D(start=start.g, end=end.g)
        b_chunk = Chunk1D(start=start.b, end=end.b)
        return r_chunk, g_chunk, b_chunk

    def is_in_rgbchunk(self, value: RGB) -> bool:
        """
        Checks if an RGB value is within the RGB chunk (inclusive).
        """
        return (self.r_chunk.is_in_chunk(value.r) and
                self.g_chunk.is_in_chunk(value.g) and
                self.b_chunk.is_in_chunk(value.b))

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

# pylint: disable=too-many-instance-attributes
@dataclass
class RGBCube:
    """
    Represents a cube in RGB space defined by its minimum and maximum RGB coordinates.
    """
    p0: RGB
    p1: RGB
    p2: RGB
    p3: RGB
    p7: RGB
    p4: RGB
    p5: RGB
    p6: RGB

    def __post_init__(self):
        pass

    @property
    def as_tuple(self) -> Tuple[RGB, RGB, RGB, RGB, RGB, RGB, RGB, RGB]:
        """
        Returns the cube corners as a tuple of RGB points.
        """
        if None in (self.p0, self.p1, self.p2, self.p3,
                    self.p4, self.p5, self.p6, self.p7):
            raise ValueError("All corner points must be defined.")

        return (self.p0, self.p1, self.p2, self.p3,
                self.p4, self.p5, self.p6, self.p7)

    @staticmethod
    def from_chunks(
            r_chunk: Chunk1D,
            g_chunk: Chunk1D,
            b_chunk: Chunk1D) \
                -> RGBCube:
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

        return RGBCube(
                p0=RGB(r=r_s, g=g_s, b=b_s),
                p1=RGB(r=r_s, g=g_s, b=b_e),
                p2=RGB(r=r_s, g=g_e, b=b_s),
                p3=RGB(r=r_s, g=g_e, b=b_e),
                p4=RGB(r=r_e, g=g_s, b=b_s),
                p5=RGB(r=r_e, g=g_s, b=b_e),
                p6=RGB(r=r_e, g=g_e, b=b_s),
                p7=RGB(r=r_e, g=g_e, b=b_e))

    @staticmethod
    def from_rgbchunk(rgb_chunk: RGBChunk) \
            -> RGBCube:
        """
        Returns the 8 corner coordinates of the subspace in RGB space.

        Working with inclusive intervals [start, end].
        """
        r_chunk, g_chunk, b_chunk = rgb_chunk.as_chunks
        return RGBCube.from_chunks(r_chunk, g_chunk, b_chunk)

    @staticmethod
    def to_rgbchunk(rgb_cube: RGBCube) -> RGBChunk:
        """
        Creates an RGBChunk from the 8 corner coordinates of the subspace in RGB space.

        Working with inclusive intervals [start, end].
        """
        r_values = [rgb.r for rgb in rgb_cube.as_tuple]
        g_values = [rgb.g for rgb in rgb_cube.as_tuple]
        b_values = [rgb.b for rgb in rgb_cube.as_tuple]

        r_chunk = Chunk1D(start=min(r_values), end=max(r_values))
        g_chunk = Chunk1D(start=min(g_values), end=max(g_values))
        b_chunk = Chunk1D(start=min(b_values), end=max(b_values))

        return RGBChunk(r_chunk=r_chunk, g_chunk=g_chunk, b_chunk=b_chunk)

    @staticmethod
    def to_chunks(rgb_cube: RGBCube) \
            -> Tuple[Chunk1D, Chunk1D, Chunk1D]:
        """
        Creates Chunk1D instances from the 8 corner coordinates of the subspace in RGB space.

        Working with inclusive intervals [start, end].
        """
        rgb_chunk = RGBCube.to_rgbchunk(rgb_cube)
        return (rgb_chunk.r_chunk,
                rgb_chunk.g_chunk,
                rgb_chunk.b_chunk)

    def is_in_rgbcube(self, value: RGB) -> bool:
        """
        Checks if an RGB value is within the RGB cube (inclusive).
        """
        r_chunk, g_chunk, b_chunk = self.to_chunks(self)
        rgb_chunk = RGBChunk(r_chunk=r_chunk, g_chunk=g_chunk, b_chunk=b_chunk)
        return rgb_chunk.is_in_rgbchunk(value)

@dataclass
class RGBBin:
    """
    Data structure for RGB binning information.
    """
    bin_idx: Tuple[int, int, int]
    r_chunk: Chunk1D
    g_chunk: Chunk1D
    b_chunk: Chunk1D

    @property
    def as_rgbchunk(self) -> RGBChunk:
        """
        Returns the RGBChunk representation of the RGB bin.
        """
        return RGBChunk(
            r_chunk=self.r_chunk,
            g_chunk=self.g_chunk,
            b_chunk=self.b_chunk
        )

    def as_rgbcube(self) -> RGBCube:
        """
        Returns the RGBCube representation of the RGB bin.
        """
        return RGBCube.from_chunks(r_chunk=self.r_chunk,
                                    g_chunk=self.g_chunk,
                                    b_chunk=self.b_chunk)

class FreezeClassVar(type):
    """
    Metaclass to freeze class variables after initialization.
    """
    _is_locked: bool = False

    def __setattr__(cls, key, value):
        if key == "_is_locked":
            return super().__setattr__(key, value)

        if cls._is_locked:
            raise AttributeError("Cannot modify class variable after initialization.")
        return super().__setattr__(key, value)

    def __delattr__(cls, name):
        if cls._is_locked:
            raise AttributeError("Cannot delete class variable after initialization.")
        return super().__delattr__(name)

    def lock(cls):
        """
        Lock the class to prevent further modifications to class variables.
        """
        cls._is_locked = True

    def unlock(cls):
        """
        Unlock the class to allow modifications to class variables.
        """
        cls._is_locked = False

@dataclass(frozen=True)
class RGBBinning(metaclass=FreezeClassVar):
    """
    Data structure for intermediate conversion space between bytes and RGB values.
    """
    bin_num: ClassVar[int] = 7
    bin_width: ClassVar[int] = 36

    fr_min: ClassVar[int] = 0
    fr_max: ClassVar[int] = 255

    full_range: ClassVar[Chunk1D] = field(default=Chunk1D(start=fr_min, end=fr_max), init=False)
    tot_bins_width: ClassVar[int] = field(default=bin_num * bin_width, init=False)

    Encoding_space: ClassVar[List[Dict[str, object]]] = field(default=None, init=False)

    def __post_init__(self):
        if self.bin_num <= 0 or self.bin_width <= 0:
            raise ValueError("bin_num and bin_width must be positive integers.")
        if self.fr_min < 0 or self.fr_max > 255:
            raise ValueError("fr_min and fr_max must be in the range [0, 255].")
        if self.fr_min >= self.fr_max:
            raise ValueError("fr_min must be less than fr_max.")
        if self.full_range.length < self.tot_bins_width:
            raise ValueError("Total bin width exceeds the full range.")
        assert self.bin_num % 2 == 1, "bin_num must be an odd integer."

    def __call__(self, *args, **kwargs):
        return self.quantization(*args, **kwargs)

    @classmethod
    def config(cls, bin_num: int, bin_width: int, fr_min: int = 0, fr_max: int = 255):
        """
        Configure the RGBBinning class with custom
        bin_num, bin_width, fr_min, and fr_max.
        """
        cls.unlock()
        cls.bin_num = bin_num
        cls.bin_width = bin_width
        cls.fr_min = fr_min
        cls.fr_max = fr_max
        cls.full_range = Chunk1D(start=fr_min, end=fr_max)
        cls.tot_bins_width = bin_num * bin_width
        cls.lock()

    def alignment(self) -> Chunk1D:
        """
        Center alignment of RGB bins within the full RGB range.

        Returns:
            bins (Chunk1D): Center aligned RGB bins.
        """
        total_padding = self.full_range.length - self.tot_bins_width
        padding_per_side = total_padding // 2

        aligned_start = self.fr_min + padding_per_side
        aligned_end = aligned_start + self.tot_bins_width - 1

        bins = Chunk1D(start=aligned_start, end=aligned_end)
        return bins

    def binning1d(self) -> List[Chunk1D]:
        """
        Binning of RGB values into specified number of bins.

        Returns:
            bins (List[Chunk1D]): List of 1D chunks representing the bins.
        """

        bins: List[Chunk1D] = []
        aligned_bins = self.alignment()

        for i in range(self.bin_num):
            bin_start = aligned_bins.start + i * self.bin_width
            bin_end = bin_start + self.bin_width - 1

            bin_chunk = Chunk1D(start=bin_start, end=bin_end)
            bins.append(bin_chunk)

        return bins

    def binning3d(self) -> List[RGBBin]:
        """
        3D Binning of RGB values into specified number of bins along each axis.

        Returns:
            bins_3d (List[RGBBin]): List of RGBBin representing the 3D bins.
        """

        bins_1d = self.binning1d()
        bins_3d: List[RGBBin] = []
        for r_bin in bins_1d:
            for g_bin in bins_1d:
                for b_bin in bins_1d:
                    bin_dict = {
                        "bin_idx": \
                            (bins_1d.index(r_bin), bins_1d.index(g_bin), bins_1d.index(b_bin)),
                        "r_chunk": r_bin,
                        "g_chunk": g_bin,
                        "b_chunk": b_bin
                    }
                    bins_3d.append(RGBBin(**bin_dict))

        return bins_3d

    @staticmethod
    def rgb_codition(idx: Tuple[int, int, int], \
                    condition: List | Tuple | int) -> Tuple[bool, bool, bool]:
        """
        Check if the RGB indices meet the specified condition.

        Args:
            idx (Tuple[int, int, int]): The RGB indices to check.
            condition (List | Tuple | int): The condition(s) to check against.

        Returns:
            Tuple[bool, bool, bool]: A tuple indicating whether each RGB index meets the condition.
        """
        r_idx, g_idx, b_idx = idx

        if isinstance(condition, (int, List, Tuple)):
            r_meet = r_idx in condition \
                if isinstance(condition, (list, tuple)) else (r_idx == condition)
            g_meet = g_idx in condition \
                if isinstance(condition, (list, tuple)) else (g_idx == condition)
            b_meet = b_idx in condition \
                if isinstance(condition, (list, tuple)) else (b_idx == condition)
        else:
            raise ValueError("Condition must be an int or a list/tuple of length 3.")

        return (r_meet, g_meet, b_meet)

    def ex_idx(self, \
            rgb_cube: Optional[List[Dict[str, RGBChunk | Tuple[int, int, int]]]] = None) \
                -> List[Tuple[int, int, int]]:
        """
        Select indices to exclude from the RGB space.
        """

        cubes = rgb_cube if rgb_cube is not None else self.binning3d()

        assert len(cubes) == self.bin_num ** 3, \
            f"The number of cubes ({len(cubes)}) must equal {self.bin_num ** 3}."

        cubes_idx = [cube.bin_idx for cube in cubes]
        excluded_idx: List[Tuple[int, int, int]] = []

        for _idx in cubes_idx:
            condition = (0, self.bin_num - 1)
            r_meet, g_meet, b_meet = self.rgb_codition(_idx, condition)

            condtion_meet = sum([r_meet, g_meet, b_meet])
            if condtion_meet >= 2:
                excluded_idx.append(_idx)

            condition = self.bin_num // 2
            r_meet, g_meet, b_meet = self.rgb_codition(_idx, condition)
            condtion_meet = sum([r_meet, g_meet, b_meet])
            if condtion_meet >= 2:
                excluded_idx.append(_idx)

        return excluded_idx

    def inc_bins(self, \
            bins3d: Optional[List[RGBBin]] = None, \
            excluded_idx: Optional[List[Tuple[int, int, int]]] = None) \
                -> List[RGBBin]:
        """
        Include bins by excluding specified indices from all bins.
        """

        all_bins3d = bins3d if bins3d is not None else self.binning3d()
        ex_idx = excluded_idx if excluded_idx is not None else self.ex_idx()

        included_bins3d: List[RGBBin] = []
        for bin3d in all_bins3d:
            assert isinstance(bin3d, RGBBin), "Each item in bins3d must be of type RGBBin."
            assert hasattr(bin3d, 'bin_idx'), "Each RGBBin must have a 'bin_idx' attribute."
            assert isinstance(bin3d.bin_idx, tuple) and len(bin3d.bin_idx) == 3, \
                "bin_idx must be a tuple of length 3."
            assert all(isinstance(i, int) for i in bin3d.bin_idx), \
                "Each element in bin_idx must be an integer."
            assert all(0 <= i < self.bin_num for i in bin3d.bin_idx), \
                f"Each element in bin_idx must be in the range [0, {self.bin_num - 1}]."

            if bin3d.bin_idx not in ex_idx:
                included_bins3d.append(bin3d)

        type(self).unlock()
        type(self).Encoding_space = tuple(included_bins3d)
        type(self).lock()

        return included_bins3d

    def quantization(self) -> List[RGBBin]:
        """
        Perform RGB binning and return the included bins after exclusion.
        """
        bins3d = self.binning3d()
        excluded_idx = self.ex_idx()
        included_bins3d = self.inc_bins(bins3d=bins3d, excluded_idx=excluded_idx)
        return included_bins3d


if __name__ == "__main__":
    _test = RGBBinning()
    quantized_bins = _test.inc_bins()
    print(f"Total included bins: {len(quantized_bins)}")
    print("Sample included bins:")
    for _i, _bin in enumerate(quantized_bins):
        print(f"{_i} Bin Index: {_bin.bin_idx}, R Chunk: {_bin.r_chunk.as_inclusive}, "
            f"G Chunk: {_bin.g_chunk.as_inclusive}, B Chunk: {_bin.b_chunk.as_inclusive}")

    print("Class variable Encoding_space is now locked and cannot be modified.")
    print(f"Encoding_space has {len(RGBBinning.Encoding_space)} bins.")
