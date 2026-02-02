"""
Configuration module for diffusion_hash_inv core components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

@dataclass(frozen=True)
class MainConfig:
    """
    Command line flags
    """
    message_flag: bool
    verbose_flag: bool
    clean_flag: bool
    debug_flag: bool
    make_xlsx_flag: bool

    def __getattribute__(self, name):
        if not hasattr(self, name):
            raise ValueError(f"MainConfig has no attribute '{name}'.")

        ret = super().__getattribute__(name)
        if ret is None:
            raise ValueError(f"MainConfig attribute '{name}' is not initialized.")
        return ret

    def reset_clean_flag(self):
        """
        Reset the clean_flag to False after cleaning.
        """
        object.__setattr__(self, "clean_flag", False)

@dataclass(frozen=True)
class HeaderConstants:
    """
    Configuration for fixed header settings.
    """
    timestamp_length: int = 32  # 32 bytes UTF-8 timestamp
    bits_length: int = 8  # 64 bits
    difftime_length: int = 8  # 64 bits
    padding_length: int = 16 - bits_length - difftime_length
    header_length: int = \
        timestamp_length + bits_length + difftime_length + padding_length  # 48 bytes = 16 * 3

@dataclass(frozen=True)
class OutputConfig:
    """
    Configuration for output settings.
    """
    root_dir: Optional[Path] = None
    data_dir: Path = field(init=False, default=None)
    output_dir: Path = field(init=False, default=None)
    encoding: str = "utf-8"

    def __post_init__(self):
        _root_dir = self.get_project_root()

        if self.root_dir is None:
            object.__setattr__(self, "root_dir", _root_dir)

        object.__setattr__(self, "data_dir", _root_dir / "data")
        object.__setattr__(self, "output_dir", _root_dir / "output")

    def __getattribute__(self, name):
        if not hasattr(self, name):
            raise ValueError(f"OutputConfig has no attribute '{name}'.")

        ret = super().__getattribute__(name)
        if ret is None:
            raise ValueError(f"OutputConfig attribute '{name}' is not initialized.")
        return ret

    @staticmethod
    def get_project_root(marker_files=("pyproject.toml", ".git")) -> Path:
        """
        Jupyter/Script 어디서 실행해도 프로젝트 루트를 찾아줌.
        marker_files 중 하나라도 있으면 거기를 루트로 간주.
        """
        current = Path.cwd().resolve()  # notebook에서는 cwd 기준
        for parent in [current, *current.parents]:
            if any((parent / marker).exists() for marker in marker_files):
                return parent
        raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")
