"""
Configuration module for diffusion_hash_inv core components.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional
import secrets
from pathlib import Path

@dataclass(frozen=True)
class MainConfig:
    """
    Command line flags
    """
    verbose_flag: bool
    clean_flag: bool
    debug_flag: bool
    make_image_flag: bool

    def __getattribute__(self, name):
        try:
            ret = super().__getattribute__(name)
        except AttributeError as exc:
            raise ValueError(f"MainConfig has no attribute '{name}'.") from exc

        fields = object.__getattribute__(self, "__dataclass_fields__")
        if name in fields and ret is None:
            raise ValueError(f"MainConfig attribute '{name}' is not initialized.")
        return ret

    def __repr__(self):
        return (
            "MainConfig\n"
            f"  verbose_flag: {self.verbose_flag},\n"
            f"  clean_flag: {self.clean_flag},\n"
            f"  debug_flag: {self.debug_flag},\n"
            f"  make_image_flag: {self.make_image_flag},\n"
            )

    def reset_clean_flag(self):
        """
        Reset the clean_flag to False after cleaning.
        """
        object.__setattr__(self, "clean_flag", False)

    @staticmethod
    def help() -> str:
        """
        Return a help string describing the MainConfig fields.
        """
        return (
            "MainConfig\n"
            "  verbose_flag: Enable verbose output.\n"
            "  clean_flag: Perform cleaning operations (e.g., remove old outputs).\n"
            "  debug_flag: Enable debug mode with additional checks and logging.\n"
            "  make_image_flag: Generate images during processing.\n")

@dataclass(frozen=True)
class MessageConfig:
    """
    Configuration for message generation.
    """
    # True: generate message, False: generate bits
    message_flag: bool = field(default=True)

    length: int = field(default=0) # Length in bits, must be a positive multiple of 8

    # True: generate random message/bits
    # False: use input_seed to generate deterministic message/bits (currently unavailable)
    random_flag: bool = field(default=True)

    # True: set random seed, False: use input_seed
    seed_flag: bool = field(default=True)

    # Seed value if seed_flag is False
    input_seed: Optional[int] = field(default=None)

    # Random seed value if seed_flag is True, set in __post_init__
    seed: Optional[int] = field(default=None)

    def __post_init__(self):
        if self.length <= 0 or self.length % 8 != 0:
            raise ValueError("length must be a positive multiple of 8")
        if self.seed_flag:
            object.__setattr__(self, "seed", secrets.randbits(32))
        else:
            assert self.input_seed is not None, "input_seed must be provided if seed_flag is False"
            object.__setattr__(self, "seed", 0 if self.input_seed is None else self.input_seed)

    def __getattribute__(self, name):
        try:
            ret = super().__getattribute__(name)
        except AttributeError as exc:
            raise ValueError(f"MessageConfig has no attribute '{name}'.") from exc

        fields = object.__getattribute__(self, "__dataclass_fields__")
        _allowed_uninitialized = {"input_seed", "seed"} # seed 관련 필드는 상황에 따라 None일 수 있음
        if name in fields and ret is None and name not in _allowed_uninitialized:
            raise ValueError(f"MessageConfig attribute '{name}' is not initialized.")
        return ret

    def __repr__(self):
        return (
            "MessageConfig\n"
            f"  message_flag: {self.message_flag},\n"
            f"  length: {self.length},\n"
            f"  random_flag: {self.random_flag},\n"
            f"  seed_flag: {self.seed_flag},\n"
            f"  input_seed: {self.input_seed if self.input_seed is not None else 'None'},\n"
            f"  seed: {self.seed}\n")

    @staticmethod
    def help() -> str:
        """
        Return a help string describing the MessageConfig fields.
        """
        return (
            "MessageConfig\n"
            "  message_flag: True to generate message, False to generate bits.\n"
            "  length: Length of message/bits in bits (must be a positive multiple of 8).\n"
            "  random_flag: True to generate random message/bits, False to use input_seed.\n"
            "  seed_flag: True to set random seed, False to use input_seed.\n"
            "  input_seed: Seed value if seed_flag is False (currently unavailable).\n"
            "  seed: Random seed value if seed_flag is True, set in __post_init__.\n")

    def update(self, **kwargs) -> MessageConfig:
        """
        Return a new MessageConfig with updated fields.
        """
        updated = replace(self, **kwargs)
        return updated

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
        configured_root = object.__getattribute__(self, "root_dir")
        resolved_root = configured_root if configured_root is not None else self.get_project_root()
        object.__setattr__(self, "root_dir", resolved_root)
        object.__setattr__(self, "data_dir", resolved_root / "data")
        object.__setattr__(self, "output_dir", resolved_root / "output")

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
        current = Path.cwd().resolve()  # notebook에서는 cwd 기준
        for parent in [current, *current.parents]:
            if any((parent / marker).exists() for marker in marker_files):
                return parent
        raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")
