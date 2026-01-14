"""
Defines RGB color space subcubes and provides utilities to convert byte values to RGB tuples.
Each subcube represents a partition of the RGB color space.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from diffusion_hash_inv.utils import RGB, RGBChunks, Chunk1D

class Byte2RGB:
    """
    A class to convert byte values(0x00 ~ 0xFF) to RGB tuples in  RGB color space.
    """

    def __init__(self, full_space_min: int = 0, full_space_max: int = 255, \
                sub_len: int = 36, split: int = 7):

        self.sub_len: int = sub_len
        self.split: int = split

        self.full_chunk: Chunk1D = Chunk1D.from_ints(full_space_min, full_space_max)

        self.tot_sub_len = self.sub_len * self.split

        if self.tot_sub_len > self.full_chunk.length:
            _t = self.tot_sub_len
            _f = self.full_chunk.length
            raise ValueError(f"Requested subspace length({_t}) exceeds full space length({_f}).")

    def centered_subspace_1d(self) -> Chunk1D:
        """
        full range is inclusive [full_min, full_max].
        returns centered inclusive [sub_min, sub_max] with exactly sub_len integer values.

        Returns:
            Chunk1D: The centered subspace as a Chunk1D object.
        """

        if self.sub_len <= 0:
            raise ValueError("space_len must be positive.")

        leftover = self.full_chunk.length - self.tot_sub_len

        _full_len = self.full_chunk.length
        _tot_sub_len = self.tot_sub_len
        print(f"full_len: {_full_len}, tot_sub_len: {_tot_sub_len}, leftover: {leftover}")

        if leftover < 0:
            raise RuntimeError("Internal error: leftover should be non-negative.")
        if leftover % 2 != 0:
            raise RuntimeError("Internal error: leftover should be even.")
        if leftover % 1 != 0:
            raise RuntimeError("Internal error: leftover should be integer.")

        left_pad = right_pad = leftover // 2

        print(f"left_pad: {left_pad}, right_pad: {right_pad}")

        sub_min = self.full_chunk.start + left_pad
        sub_max = self.full_chunk.end - right_pad
        _sub_len = sub_max - sub_min + 1

        if _sub_len != self.sub_len * self.split:
            _t = self.sub_len * self.split
            _s = _sub_len
            raise RuntimeError\
                (f"Internal error: computed subspace length({_s}) mismatch({_t}).")
        return Chunk1D.from_ints(sub_min, sub_max)

    def split_1d_inclusive(self) -> List[Chunk1D]:
        """
        Split inclusive [start, end] into `parts` equal-size inclusive chunks.
        Requires divisibility in integer-point counts.
        """

        start, end = self.centered_subspace_1d().to_tuple

        chunks: List[Chunk1D] = []

        for i in range(self.split):
            chunk_start = start + i * self.sub_len
            chunk_end = chunk_start + self.sub_len - 1  # inclusive
            chunks.append(Chunk1D.from_ints(chunk_start, chunk_end))

        if len(chunks) != self.split:
            raise RuntimeError("Internal error: number of chunks mismatch.")
        if chunks[0].start != start or chunks[-1].end != end:
            raise RuntimeError("Internal error: chunk boundaries mismatch.")

        return chunks

    def rgb_subcubes(self) -> Dict[str, object]:
        """
        Returns:
            A dictionary containing information about RGB subcubes.
        """
        print("Splitting RGB space into subcubes...")
        print("Processing R channel...")
        r_chunks = self.split_1d_inclusive()
        print("Processing G channel...")
        g_chunks = self.split_1d_inclusive()
        print("Processing B channel...")
        b_chunks = self.split_1d_inclusive()

        cubes: List[Dict[str, RGBChunks]] = []
        for i, r in enumerate(r_chunks):
            for j, g in enumerate(g_chunks):
                for k, b in enumerate(b_chunks):

                    cubes.append(
                        {
                            "idx": (i, j, k),
                            "inclusive": RGBChunks(r, g, b),
                            "half_open": RGBChunks(
                                Chunk1D.from_ints(r.start, r.end + 1),
                                Chunk1D.from_ints(g.start, g.end + 1),
                                Chunk1D.from_ints(b.start, b.end + 1)
                            )
                        }
                    )

        return {
            "num_cubes": len(cubes),
            "cubes": cubes,
        }

    @staticmethod
    def rgb_condition(idx: Tuple[int, int, int], \
                    condition: List | Tuple | int) -> Tuple[bool, bool, bool]:
        """
        Check if the RGB cube index meets the specified conditions for each channel.

        Args:
            idx (Tuple[int, int, int]): The RGB cube index as a tuple (r_idx, g_idx, b_idx).
            condition (List, Tuple, int): The condition(s) for the R channel.

        Returns:
            Tuple[bool, bool, bool]: A tuple indicating whether each channel meets the condition.
        """

        r_idx, g_idx, b_idx = idx

        r_meets = r_idx in condition \
            if isinstance(condition, (list, tuple)) else r_idx == condition
        g_meets = g_idx in condition \
            if isinstance(condition, (list, tuple)) else g_idx == condition
        b_meets = b_idx in condition \
            if isinstance(condition, (list, tuple)) else b_idx == condition

        return (r_meets, g_meets, b_meets)

    def exclude_cubes(self, rgb_cube: Optional[Dict[str, object]] = None):
        """
        Placeholder for excluding a box from the RGB space.
        """
        cubes = self.rgb_subcubes()["cubes"] if rgb_cube is None else rgb_cube["cubes"]

        assert len(cubes) == self.split ** 3, \
            f"Cube count: {len(cubes)} mismatch {self.split ** 3}."

        cubes_idx:List[Tuple[int, int, int]] = [cubes[i]["idx"] for i in range(len(cubes))]
        excluded_idx: List[Tuple[int, int, int]] = []
        print("Total cubes to process:", len(cubes_idx))
        print(" 000. Processing cube idx: (r, g, b)")

        for _i, _idx in enumerate(cubes_idx):
            print(f" {_i+1:03}. Processing cube idx: {_idx}")

            condition = [0, self.split -1]
            r_cond, g_cond, b_cond = \
                self.rgb_condition(idx=_idx, condition=condition)

            print("     Checking edge and vertex cases...")
            print(f"     Condition for edge/vertex check: {condition}")
            print(f"     r_cond: {r_cond}, g_cond: {g_cond}, b_cond: {b_cond}")

            # Exclude each cubes following edge cases & vertex cases
            is_edge_cube = \
                (r_cond + g_cond + b_cond) == 2

            is_vertex_cube = \
                (r_cond + g_cond + b_cond) == 3

            # Exclude each cube following specific criteria
            assert self.split % 2 == 1, "Specific criteria requires odd split value."
            condition = self.split // 2
            r_cond, g_cond, b_cond = \
                self.rgb_condition(idx=_idx, condition=condition)

            print("     Checking specific criteria for exclusion...")
            print(f"     Condition for specific criteria: {condition}")
            print(f"     r_cond: {r_cond}, g_cond: {g_cond}, b_cond: {b_cond}")

            is_specific = (r_cond + g_cond + b_cond) >= 2

            print()
            print(f"     is_edge: {is_edge_cube}")
            print(f"     is_vertex: {is_vertex_cube}")
            print(f"     is_specific: {is_specific}")

            if is_specific or is_edge_cube or is_vertex_cube:
                if _idx not in excluded_idx:
                    excluded_idx.append(_idx)
                    # breakpoint()

            print("--------------------------------------------------")
            print()


        print("Exclusion completed.")
        print(f"Total excluded cubes: {len(excluded_idx)}")
        print(f"Excluded cube indices: {excluded_idx}")

        return excluded_idx






if __name__ == "__main__":
    # Example usage
    # Each Subspace: size 36^3 Cubes, Total 7x7x7=343 Cubes
    byte2rgb = Byte2RGB()
    all_cubes = byte2rgb.rgb_subcubes()

    print("Num cubes:", all_cubes["num_cubes"]) # 343

    # 첫 번째/마지막 큐브 확인
    print("First cube:", all_cubes["cubes"][0])
    print("Last cube:", all_cubes["cubes"][-1])

    print()

    print("Excluding cubes...")
    byte2rgb.exclude_cubes(all_cubes)
