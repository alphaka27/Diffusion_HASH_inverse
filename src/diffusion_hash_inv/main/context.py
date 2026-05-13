"""
Runtime context for logging and configuration management
Contains metadata, base logs, step logs
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict, field
from typing import Any, Optional, List

from diffusion_hash_inv.logger import Metadata, BaseLogs, StepLogs
from diffusion_hash_inv.config import \
    (MainConfig, HashConfig, MessageConfig, OutputConfig, Byte2RGBConfig)
from diffusion_hash_inv.validation import config_validate

@dataclass
class RuntimeState:
    """
    Runtime context for logging
    Contains metadata, base logs, step logs
    """
    metadata: Optional[Metadata] = None
    baselogs: Optional[BaseLogs] = None
    steplogs: Optional[StepLogs] = None
    algo: Optional[Any] = None

    def validate(self) -> RuntimeState:
        """
        Validate the runtime state before logging
        """
        assert self.metadata is not None, "Metadata is not set."
        assert self.baselogs is not None, "BaseLogs is not set."
        assert self.steplogs is not None, "StepLogs is not set."
        return self

    def with_updates(self, metadata: Optional[Metadata] = None, baselogs: Optional[BaseLogs] = None,
            steplogs: Optional[StepLogs] = None, algo: Optional[Any] = None) -> RuntimeState:
        """
        Update the runtime state with new values
        """
        updated = replace(self,
                        metadata=metadata if metadata is not None else self.metadata,
                        baselogs=baselogs if baselogs is not None else self.baselogs,
                        steplogs=steplogs if steplogs is not None else self.steplogs,
                        algo=algo if algo is not None else self.algo)

        return updated

    def copy(self) -> RuntimeState:
        """
        Create a copy of the runtime state
        """
        return replace(self)

@dataclass(frozen=True)
class RuntimeConfig:
    """
    Configuration context for hash generation
    Contains main configuration, message configuration, and hash configuration
    """
    main: MainConfig
    message: MessageConfig
    hash: HashConfig
    output: OutputConfig
    rgb: Byte2RGBConfig
    checklist: List[str] = field(init=False, \
                                default_factory=lambda: ["length"])

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        """
        Validate cross-config constraints.
        """
        config_validate(asdict(self.message),
                asdict(self.hash),
                key=self.checklist)
        if self.message.length % 8 != 0:
            raise ValueError("MessageConfig length must be a positive multiple of 8")
        if self.hash.length % 8 != 0:
            raise ValueError("HashConfig length must be a positive multiple of 8")

    def __repr__(self):
        def indent_block(value: Any, spaces: int = 2) -> str:
            prefix = " " * spaces
            return "\n".join(
                f"{prefix}{line}" for line in repr(value).rstrip().splitlines()
            )

        return "\n".join((
            "RuntimeConfig",
            indent_block(self.main),
            indent_block(self.message),
            indent_block(self.hash),
            indent_block(self.output),
            indent_block(self.rgb),
        ))

    @classmethod
    def set_default(cls) -> RuntimeConfig:
        """
        Return a default runtime configuration
        """
        return cls(
            main=MainConfig(
                verbose_flag=False,
                clean_flag=False,
                debug_flag=False,
                make_image_flag=True
            ),
            message=MessageConfig(length=128),
            hash=HashConfig(hash_alg="MD5", length=128),
            output=OutputConfig(),
            rgb=Byte2RGBConfig()
        )

    def main_update(self, main_config: MainConfig) -> RuntimeConfig:
        """
        Update the main configuration in the context
        """
        object.__setattr__(self, "main", main_config) # bypass frozen
        self._validate()
        return self


    def message_update(self, message_config: MessageConfig) -> RuntimeConfig:
        """
        Update the message configuration in the context
        """
        object.__setattr__(self, "message", message_config) # bypass frozen
        self._validate()
        return self

    def hash_update(self, hash_config: HashConfig) -> RuntimeConfig:
        """
        Update the hash configuration in the context
        """
        object.__setattr__(self, "hash", hash_config) # bypass frozen
        self._validate()
        return self

    def full_update(
        self,
        main_config: Optional[MainConfig] = None,
        message_config: Optional[MessageConfig] = None,
        hash_config: Optional[HashConfig] = None) -> RuntimeConfig:
        """
        Update the configuration context with new configurations
        """
        if main_config is not None:
            object.__setattr__(self, "main", main_config)
        if message_config is not None:
            object.__setattr__(self, "message", message_config)
        if hash_config is not None:
            object.__setattr__(self, "hash", hash_config)

        self._validate()
        return self
