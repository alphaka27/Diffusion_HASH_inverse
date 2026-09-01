"""
Main Configuration module for diffusion_hash_inv core components.
"""

from __future__ import annotations

from dataclasses import dataclass

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
