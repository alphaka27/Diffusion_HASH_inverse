"""
Output Configuration module for diffusion_hash_inv core components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

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
    emnist_dir: Path = field(init=False, default=None)
    encoding: str = "utf-8"

    def __post_init__(self):
        configured_root = object.__getattribute__(self, "root_dir")
        resolved_root = configured_root if configured_root is not None else self.get_project_root()
        resolved_root = Path(resolved_root).resolve()
        object.__setattr__(self, "root_dir", resolved_root)
        object.__setattr__(self, "data_dir", resolved_root / "data")
        object.__setattr__(self, "output_dir", resolved_root / "output")
        object.__setattr__(self, "emnist_dir", resolved_root / "EMNIST")

    def __getattribute__(self, name):
        try:
            ret = super().__getattribute__(name)
        except AttributeError as exc:
            raise ValueError(f"OutputConfig has no attribute '{name}'.") from exc

        fields = object.__getattribute__(self, "__dataclass_fields__")
        if name in fields and ret is None:
            raise ValueError(f"OutputConfig attribute '{name}' is not initialized.")
        return ret

    def __repr__(self):
        return ("OutputConfig\n"
            f"  Root Directory: {self.root_dir},\n"
            f"  Data Directory: {self.data_dir},\n"
            f"  Output Directory: {self.output_dir},\n"
            f"  EMNIST Directory: {self.emnist_dir},\n"
            f"  Encoding: '{self.encoding}'\n")

    @staticmethod
    def help() -> str:
        """
        Return a help string describing the OutputConfig fields.
        """
        return (
            "OutputConfig\n"
            "  root_dir: Optional custom root directory. If None, the project root will be used.\n"
            "  data_dir: Directory for input data, set to root_dir/data.\n"
            "  output_dir: Directory for output results, set to root_dir/output.\n"
            "  encoding: Encoding format for text files (default 'utf-8').\n")

    @staticmethod
    def get_project_root(marker_files=("pyproject.toml", ".git")) -> Path:
        """
        Jupyter/Script 어디서 실행해도 프로젝트 루트를 찾아줌.
        marker_files 중 하나라도 있으면 거기를 루트로 간주.
        """
        candidates = []
        try:
            candidates.append(Path.cwd().resolve())  # notebook에서는 cwd 기준
        except FileNotFoundError:
            # A long-lived Jupyter kernel can keep a cwd that was deleted or moved.
            # Fall back to this module's location so OutputConfig() can still boot.
            pass

        candidates.append(Path(__file__).resolve().parent)

        for current in candidates:
            for parent in [current, *current.parents]:
                if any((parent / marker).exists() for marker in marker_files):
                    return parent
        raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")
