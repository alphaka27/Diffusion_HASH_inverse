"""
Validation Value of Configuration
"""

from typing import List
from dataclasses import asdict

def config_validate(config1: dict, config2: dict, key: List[str]) -> None:
    """
    Validate the configuration
    """
    _cfg1 = asdict(config1) if not isinstance(config1, dict) else config1
    _cfg2 = asdict(config2) if not isinstance(config2, dict) else config2
    for k in key:
        if k not in _cfg1:
            raise KeyError(f"Key '{k}' not found in {type(config1).__name__}")

        if k not in _cfg2:
            raise KeyError(f"Key '{k}' not found in {type(config2).__name__}")

        if _cfg1[k] != _cfg2[k]:
            err_msg = (f"Configuration mismatch for key '{k}' of "
                    f"{type(config1).__name__} & {type(config2).__name__}: "
                    f"{_cfg1[k]} != {_cfg2[k]}")
            raise ValueError(err_msg)
