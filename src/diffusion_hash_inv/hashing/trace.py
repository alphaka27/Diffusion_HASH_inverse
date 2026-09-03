"""Shared hash-trace entry point and tracer registration."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterable, Mapping, TypeAlias


Trace: TypeAlias = dict[str, object]
TracerFactory: TypeAlias = Callable[[], "HashTracer"]


class HashTracer(ABC):
    """Algorithm abstraction used by :func:`trace_hash`.

    A custom implementation must return a dictionary containing only JSON
    values.  The entry point adds the input metadata and optionally persists
    the final dictionary as JSON.
    """

    algorithm: str

    @abstractmethod
    def trace(self, message: bytes) -> Trace:
        """Return the complete, JSON-serialisable trace for ``message``."""


_TRACERS: dict[str, TracerFactory] = {}


def _normalise_algorithm(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("algorithm must be a non-empty string")
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")


def register_hash_function(
    name: str,
    factory: TracerFactory,
    *,
    aliases: Iterable[str] = (),
) -> None:
    """Register a :class:`HashTracer` factory for use by ``trace_hash``.

    Factories are used instead of instances so every call starts with fresh
    algorithm state.  Registering a name twice deliberately replaces the old
    implementation, making versioned or experimental implementations simple.
    """
    if not callable(factory):
        raise TypeError("factory must be callable and return a HashTracer")
    for registered_name in (name, *aliases):
        _TRACERS[_normalise_algorithm(registered_name)] = factory


def _write_json(trace: Mapping[str, object], output_path: str | Path) -> Path:
    """Atomically write a trace so callers never observe a partial JSON file."""
    target = Path(output_path)
    if target.exists() and target.is_dir():
        raise IsADirectoryError(f"output_path is a directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
            json.dump(trace, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        os.replace(temporary_path, target)
    except Exception:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
        raise
    return target


def trace_hash(
    algorithm: str,
    message: bytes | bytearray | memoryview | str,
    output_path: str | Path | None = None,
    *,
    encoding: str = "utf-8",
) -> Trace:
    """Hash ``message`` and return a detailed JSON-serialisable trace.

    This is the single execution entry point.  ``algorithm`` selects a
    registered implementation (built-ins: ``md5`` and ``sha256``).  When
    ``output_path`` is supplied, the exact returned trace is also stored there
    as UTF-8 JSON.  Text messages are encoded with ``encoding``; binary input
    is preserved exactly.
    """
    input_type: str
    if isinstance(message, str):
        message_bytes = message.encode(encoding)
        input_type = "text"
    elif isinstance(message, (bytes, bytearray, memoryview)):
        message_bytes = bytes(message)
        input_type = "bytes"
    else:
        raise TypeError("message must be str, bytes, bytearray, or memoryview")

    algorithm_key = _normalise_algorithm(algorithm)
    try:
        tracer = _TRACERS[algorithm_key]()
    except KeyError as error:
        supported = ", ".join(sorted(_TRACERS))
        raise ValueError(f"unsupported hash algorithm: {algorithm!r}. Supported: {supported}") from error
    if not isinstance(tracer, HashTracer):
        raise TypeError("registered factory must return a HashTracer instance")

    trace = tracer.trace(message_bytes)
    trace["input"] = {
        "type": input_type,
        "encoding": encoding if input_type == "text" else None,
        "byte_length": len(message_bytes),
        "hex": message_bytes.hex(),
    }
    if output_path is not None:
        _write_json(trace, output_path)
    return trace


from .md5 import MD5Tracer
from .sha256 import SHA256Tracer


register_hash_function("md5", MD5Tracer)
register_hash_function("sha256", SHA256Tracer, aliases=("sha-256",))


__all__ = ["HashTracer", "MD5Tracer", "SHA256Tracer", "register_hash_function", "trace_hash"]
