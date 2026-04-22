"""
Validation module for diffusion_hash_inv package.
    - validate: Function to validate hash outputs against expected values.
"""

from .hash_validation import validate
from .cfg_validation import config_validate

__all__ = [
    "validate",
    "encoding_validate",
    "config_validate"
]

def __getattr__(name):
    if name == "encoding_validate":
        from .encoding_validation import encoding_validate
        return encoding_validate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# EOF
