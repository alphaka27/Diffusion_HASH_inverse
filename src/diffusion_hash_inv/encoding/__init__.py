"""
__init__.py for the encoding module of the diffusion hash inversion model.
"""

from diffusion_hash_inv.encoding.rgb_grid import RGBGridEncoder
# from diffusion_hash_inv.encoding.rgb_cube import RGBCubeEncoder
# from diffusion_hash_inv.encoding.rgb_cuboid import RGBCuboidEncoder

__all__ = [
    "RGBGridEncoder",
    # "RGBCubeEncoder",
    # "RGBCuboidEncoder",
]
# EOF
