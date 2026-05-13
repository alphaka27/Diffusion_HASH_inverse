"""
Validation Value of Configuration
"""

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping


def _as_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    if is_dataclass(config):
        return asdict(config)
    raise TypeError(f"config must be a mapping or dataclass, got {type(config).__name__}")


def config_validate(config1: Any, config2: Any, key: Iterable[str]) -> None:
    """
    Validate the configuration
    """
    _cfg1 = _as_mapping(config1)
    _cfg2 = _as_mapping(config2)
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
