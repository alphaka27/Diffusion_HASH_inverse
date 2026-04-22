"""
Package exports for diffusion_hash_inv.

Keep imports light at package import time so tooling such as coverage can
import submodules without triggering heavyweight optional dependencies.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from diffusion_hash_inv.config import Byte2RGBConfig, HashConfig, MainConfig
from diffusion_hash_inv.core import BaseCalc, RGB, RGBBinning
from diffusion_hash_inv.logger import BaseLogs, Logs, Metadata, StepLogs

if TYPE_CHECKING:
    from diffusion_hash_inv.hashing import MD5  # noqa: F401
    from diffusion_hash_inv.utils.byte2rgb import Byte2RGB  # noqa: F401
    from diffusion_hash_inv.utils.file_io import FileIO  # noqa: F401
    from diffusion_hash_inv.utils.formatter import JSONFormat  # noqa: F401

__all__ = [
    "BaseCalc",
    "Byte2RGB",
    "Byte2RGBConfig",
    "FileIO",
    "HashConfig",
    "JSONFormat",
    "Logs",
    "MD5",
    "MainConfig",
    "Metadata",
    "BaseLogs",
    "RGB",
    "RGBBinning",
    "StepLogs",
]

_LAZY_EXPORTS = {
    "Byte2RGB": ("diffusion_hash_inv.utils.byte2rgb", "Byte2RGB"),
    "FileIO": ("diffusion_hash_inv.utils.file_io", "FileIO"),
    "JSONFormat": ("diffusion_hash_inv.utils.formatter", "JSONFormat"),
    "MD5": ("diffusion_hash_inv.hashing", "MD5"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    raise AttributeError(f"module 'diffusion_hash_inv' has no attribute '{name}'")


try:
    __version__ = version("diffusion-hash-inverse")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"
