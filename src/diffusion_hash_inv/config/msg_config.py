"""
Message Configuration module for diffusion_hash_inv core components.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field, replace
from typing import Optional, List
import secrets

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

    # Character candidate type (e.g., "digit", "lower", "upper", "ascii", "alphanumeric", "all")
    candidate: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.length <= 0 or self.length % 8 != 0:
            raise ValueError("length must be a positive multiple of 8")
        if self.seed_flag:
            object.__setattr__(self, "seed", secrets.randbits(32))
        else:
            assert self.input_seed is not None, "input_seed must be provided if seed_flag is False"
            object.__setattr__(self, "seed", 0 if self.input_seed is None else self.input_seed)

        if self.candidate is None:
            raise ValueError("Character candidate must be specified")
        if self.message_flag:
            self.select_candidate_list()

    def __getattribute__(self, name):
        try:
            ret = super().__getattribute__(name)
        except AttributeError as exc:
            raise ValueError(f"MessageConfig has no attribute '{name}'.") from exc

        fields = object.__getattribute__(self, "__dataclass_fields__")
        _allowed_uninitialized = {"input_seed", "candidate"}
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
            f"  seed: {self.seed},\n"
            f"  candidate: {self.candidate},\n"
            f"  candidate_list: {self.candidate_list},\n"
            f"  candidate_list length: {len(self.candidate_list)}\n")

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
            "  seed: Random seed value if seed_flag is True, set in __post_init__.\n"
            "  candidate: Character candidate type.\n"
            "  candidate_list: List of characters based on the specified candidate type.\n")

    def update(self, **kwargs) -> MessageConfig:
        """
        Return a new MessageConfig with updated fields.
        """
        updated = replace(self, **kwargs)
        return updated

    def select_candidate_list(self, candidate_type: Optional[str] = None) -> List[str]:
        """
        Get the candidate list based on the specified candidate type.
        """

        if candidate_type is not None:
            object.__setattr__(self, "candidate", candidate_type)

        if self.candidate is None:
            raise ValueError("candidate is not set.")

        candidate_lower = self.candidate.lower()
        if candidate_lower == "digit": # 0-9
            object.__setattr__(self, "candidate_list", \
                            list(string.digits)) # bypass frozen
        elif candidate_lower == "lower": # a-z
            object.__setattr__(self, "candidate_list", \
                            list(string.ascii_lowercase)) # bypass frozen
        elif candidate_lower == "upper": # A-Z
            object.__setattr__(self, "candidate_list", \
                            list(string.ascii_uppercase)) # bypass frozen
        elif candidate_lower == "ascii": # a-zA-Z
            object.__setattr__(self, "candidate_list", \
                            list(string.ascii_letters)) # bypass frozen
        elif candidate_lower == "alphanumeric": # 0-9a-zA-Z
            object.__setattr__(self, "candidate_list", \
                            list(string.ascii_letters + string.digits)) # bypass frozen
        elif candidate_lower == "all": # 0-9a-zA-Z and punctuation, excluding whitespace
            object.__setattr__(self, "candidate_list", \
                            list(string.printable.strip())) # bypass frozen
        else:
            raise ValueError(f"Unsupported candidate type: {self.candidate}."
                            "Use 'digit', 'lower', 'upper', 'ascii', 'alphanumeric', or 'all'.")

if __name__ == "__main__":
    # Example usage
    msg_config = MessageConfig(length=128, candidate="alphanumeric")
    msg_config.select_candidate_list("all")
    print(msg_config)
