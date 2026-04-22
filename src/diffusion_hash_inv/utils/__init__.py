"""
Utility exports for diffusion_hash_inv.

Avoid importing optional heavy dependencies until the corresponding symbol is
actually requested.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .formatter import JSONFormat

if TYPE_CHECKING:
    from diffusion_hash_inv.legacy.deprecated.json_to_xlsx import (  # noqa: F401
        JSONToXLSXConverter,
    )
    from .byte2rgb import Byte2RGB  # noqa: F401
    from .file_io import FileIO, Reader, Writer  # noqa: F401
    from .image_writer import RGBImgMaker  # noqa: F401

__all__ = [
    "Byte2RGB",
    "FileIO",
    "JSONFormat",
    "JSONToXLSXConverter",
    "RGBImgMaker",
    "Reader",
    "Writer",
]

_LAZY_EXPORTS = {
    "Byte2RGB": ("diffusion_hash_inv.utils.byte2rgb", "Byte2RGB"),
    "FileIO": ("diffusion_hash_inv.utils.file_io", "FileIO"),
    "Reader": ("diffusion_hash_inv.utils.file_io", "Reader"),
    "Writer": ("diffusion_hash_inv.utils.file_io", "Writer"),
    "JSONToXLSXConverter": (
        "diffusion_hash_inv.legacy.deprecated.json_to_xlsx",
        "JSONToXLSXConverter",
    ),
    "RGBImgMaker": ("diffusion_hash_inv.utils.image_writer", "RGBImgMaker"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    raise AttributeError(f"module 'diffusion_hash_inv.utils' has no attribute '{name}'")
