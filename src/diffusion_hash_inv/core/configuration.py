"""
Configuration module for diffusion_hash_inv core components.
"""

from dataclasses import dataclass

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
    hash_alg: str

    @property
    def is_message(self) -> bool:
        """
        Check if message generation flag is set.
        """
        return self.message_flag

    @property
    def is_verbose(self) -> bool:
        """
        Check if verbose flag is set.
        """
        return self.verbose_flag

    @property
    def is_clean(self) -> bool:
        """
        Check if clean flag is set.
        """
        return self.clean_flag

    @property
    def is_debug(self) -> bool:
        """
        Check if debug flag is set.
        """
        return self.debug_flag

    @property
    def is_make_xlsx(self) -> bool:
        """
        Check if make_xlsx flag is set.
        """
        return self.make_xlsx_flag

    @property
    def hash_algorithm(self) -> str:
        """
        Get the specified hash algorithm.
        """
        return self.hash_alg

    def get_flags(self) -> dict:
        """
        Get flags as a dictionary.
        """
        return {
            "is_message": self.is_message,
            "is_verbose": self.is_verbose,
            "is_clean": self.is_clean,
            "is_debug": self.is_debug,
            "is_make_xlsx": self.is_make_xlsx,
        }

@dataclass(frozen=True)
class HashConfig:
    """
    Configuration for hash generation and validation.
    """


@dataclass
class Byte2RGBConfig:
    """
    Configuration for Byte to RGB conversion.
    """

    full_space_min: int = 0
    full_space_max: int = 255
    sub_len: int = 36
    split: int = 7
