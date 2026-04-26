"""
Logging utilities for Diffusion Hash Inversion
"""

from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from datetime import datetime
from collections.abc import Hashable, MutableMapping
from dataclasses import dataclass, field
import time
from pathlib import Path

@dataclass
class Metadata:
    """
    Metadata container  
    Contains hash algorithm, input bits length, execution start time,\
        elapsed time, entropy, strength
    """
    hash_alg: str
    is_message: bool = field(default_factory=bool)

    started_at: str = field(default_factory=str)
    input_bits_len: int = field(default_factory=int)

    byteorder: str = field(default_factory=str, init=False)
    hierarchy: Sequence[Hashable] = field(default_factory=tuple, init=False)
    elapsed_time:str = field(default_factory=str, init=False)

    def clear(self)->None:
        """Clear all metadata"""
        self.started_at = ""
        self.elapsed_time = ""

    def hash_property(self, byteorder: str, hierarchy: Sequence[Hashable])->None:
        """
        Set hash properties
        Parameters:
            byteorder: str
                Byte order
            hierarchy: Sequence[Hashable]
                Hierarchy levels
        """
        self.byteorder = byteorder
        self.hierarchy = hierarchy

    def time_logger(self, elapsed_time: int)->None:
        """
        Update elapsed time

        Parameters:
            elapsed_time: int
                Elapsed time in nanoseconds
        """
        self.elapsed_time = Logs.perftimer_str(elapsed_time)

    def get_dict(self)->Dict[str, Any]:
        """
        Get Metadata as dictionary
        Returns:
            Dict[str, Any]: Metadata dictionary
        """
        return {
            "Hash Algorithm": self.hash_alg,
            "Hierarchy": self.hierarchy,
            "Input bits": self.input_bits_len,
            "Message mode": "Message" if self.is_message else "Bit string",
            "Program started at": self.started_at,
            "Program elapsed time": self.elapsed_time,
            "Byte order": self.byteorder
        }

    def getter(self) -> Dict[str, Any]:
        """Compatibility alias for metadata serialization."""
        return self.get_dict()

@dataclass
class BaseLogs:
    """
    Base logs container
    Contains Message(String and Hex), Generated hash, Correct hash
    """
    message: Dict[str, Any] = field(default_factory=dict, init=False)
    generated_hash: bytes = field(default_factory=bytes, init=False)
    correct_hash: bytes = field(default_factory=bytes, init=False)

    def clear(self):
        """Clear all base logs"""
        self.message = {}
        self.generated_hash = b""
        self.correct_hash = b""

    def set_message(self, message_bytes: bytes, message_mode: bool):
        """Set message"""
        self.message["Hex"] = Logs.bytes_to_str(message_bytes)
        if message_mode:
            try:
                self.message["Bit String"] = message_bytes.decode("utf-8")
            except UnicodeDecodeError:
                self.message["Bit String"] = message_bytes.decode("utf-8", errors="replace")
        else:
            self.message.pop("Bit String", None)

    def set_hashes(self, generated_hash: str, correct_hash: str):
        """Set hash result"""
        self.generated_hash = Logs.bytes_to_str(generated_hash)
        self.correct_hash = Logs.bytes_to_str(correct_hash)

    def update(self, **data) -> None:
        """
        Update base logs
        Parameters:
            message: bytes
                Message in utf-8 encoded or raw bytes
            is_message: bool
                Whether the message is in string mode
            generated_hash: bytes
                Generated hash bytes
            correct_hash: bytes
                Correct hash bytes
        """
        message = data.get("message", None)
        assert message is not None, "message must be provided"
        is_message = data.get("is_message", True)
        assert isinstance(is_message, bool), "is_message must be a boolean"
        generated_hash = data.get("generated_hash", None)
        assert generated_hash is not None, "generated_hash must be provided"
        correct_hash = data.get("correct_hash", None)
        assert correct_hash is not None, "correct_hash must be provided"

        if message is not None:
            self.set_message(message, is_message)
        if generated_hash is not None and correct_hash is not None:
            self.set_hashes(generated_hash, correct_hash)

    @staticmethod
    def keys() -> List[str]:
        """Get base logs keys"""
        return ["Message", "Generated hash", "Correct   hash"]

    def getter(self)->Dict[str, Any]:
        """Get base logs"""
        return {
            "Message": self.message,
            "Generated hash": self.generated_hash,
            "Correct   hash": self.correct_hash
        }

@dataclass
class StepLogs:
    """
    Step logs container that behaves like a dict while keeping per-step structure.
    """
    wordsize: Optional[int] = None
    byteorder: Optional[str] = None
    hierarchy: Sequence[Hashable] = field(default_factory=tuple)

    logs: Dict[str, Any] = field(default_factory=dict, init=False)

    overflow_log: Dict[str, Any] = field(default_factory=dict, init=False)
    step_metadata: Dict[str, Any] = field(default_factory=dict, init=False)
    overflow: int = field(default_factory=int, init=False)

    @staticmethod
    def _ordinal(n: int) -> str:
        """Return English ordinal text such as 1st, 2nd, 3rd, 4th."""
        assert n > 0, "index must be positive"
        if 10 <= (n % 100) <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    @classmethod
    def index_label(cls, index: int, label: str) -> str:
        """Build a stable key for Step/Round/Loop/Block."""
        return f"{cls._ordinal(index)} {label}"

    def clear(self) -> None:
        """Clear all step logs."""
        self.logs.clear()
        self.overflow_log.clear()
        self.step_metadata.clear()
        self.overflow = 0

    @staticmethod
    def _ensure_nested_dict(
        root: MutableMapping[Hashable, Any],
        path: Sequence[Hashable],
    ) -> MutableMapping[Hashable, Any]:
        """Create missing dictionaries for path and return the deepest dict."""
        cur: MutableMapping[Hashable, Any] = root
        for key in path:
            if key is None:
                continue
            nxt = cur.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[key] = nxt
            cur = nxt
        return cur

    def _int_to_hex(self, value: int) -> str:
        """Convert integer to hex string using configured word size/byte order."""
        if value < 0:
            raise ValueError("negative integers are not supported in step logs")
        if self.wordsize is not None and self.byteorder in ("big", "little"):
            masked = value & ((1 << self.wordsize) - 1)
            return Logs.bytes_to_str(
                LogHelper.int_to_bytes(masked, self.wordsize, byteorder=self.byteorder)
            )

        width = max(1, (value.bit_length() + 7) // 8)
        return Logs.bytes_to_str(value.to_bytes(width, byteorder="big", signed=False))

    def _normalize_value(self, value: Any) -> Any:
        """Normalize value for JSON logging."""
        if isinstance(value, (bytes, bytearray)):
            return Logs.bytes_to_str(bytes(value))

        if isinstance(value, int):
            return self._int_to_hex(value)

        if isinstance(value, dict):
            return {k: self._normalize_value(v) for k, v in value.items()}

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            blocks: Dict[str, Any] = {}
            for idx, item in enumerate(value, start=1):
                block_key = self.index_label(idx, "Block")
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                    blocks[block_key] = [self._normalize_value(sub) for sub in item]
                else:
                    blocks[block_key] = self._normalize_value(item)
            return blocks

        return value

    def set_value(self, path: Sequence[Hashable], value: Any, normalize: bool = True) -> None:
        """Set value at a nested path under step logs."""
        if not path:
            raise ValueError("path must not be empty")
        *parents, last = path
        parent_dict = self._ensure_nested_dict(self.logs, parents)
        parent_dict[last] = self._normalize_value(value) if normalize else value

    def update(
        self,
        *,
        step_index: int,
        step_result: Any,
        round_idx: Optional[int] = None,
    ) -> None:
        """Update step-level or round-level value."""
        step_key = self.index_label(step_index, "Step")
        if round_idx is None:
            self.set_value((step_key,), step_result)
            return
        round_key = self.index_label(round_idx, "Round")
        self.set_value((step_key, round_key), step_result)

    def update_loop(
        self,
        *,
        step_index: int,
        round_idx: int,
        loop_index: int,
        step_result: Any,
    ) -> None:
        """Update a loop-level value under a round."""
        step_key = self.index_label(step_index, "Step")
        round_key = self.index_label(round_idx, "Round")
        loop_key = self.index_label(loop_index, "Loop")
        self.set_value((step_key, round_key, loop_key), step_result)

    def update_step(
        self,
        step_idx: int | str,
        step_result: Any = None,
        *,
        level2: Optional[Tuple[str, int | str]] = None,
        level3: Optional[Tuple[str, int | str]] = None,
    ) -> None:
        """
        Compatibility updater used by legacy/local-variable tracer.
        """
        if step_result is None:
            return

        step_key = step_idx if isinstance(step_idx, str) else self.index_label(step_idx, "Step")
        path: List[Hashable] = [step_key]

        if level2 is not None:
            _, level2_idx = level2
            if isinstance(level2_idx, int):
                path.append(self.index_label(level2_idx, "Round"))
            elif level2_idx is not None:
                path.append(str(level2_idx))

        if level3 is not None:
            _, level3_idx = level3
            if isinstance(level3_idx, int):
                path.append(self.index_label(level3_idx, "Loop"))
            elif level3_idx is not None:
                path.append(str(level3_idx))

        self.set_value(tuple(path), step_result)

    def update_overflow(self, step: int) -> None:
        """Store overflow count in metadata."""
        assert isinstance(step, int), "overflow count must be integer"
        self.overflow = step

    def getter(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get logs and metadata in dump-ready form."""
        self.step_metadata = {
            "word_size": self.wordsize,
            "byteorder": self.byteorder,
            "hierarchy": list(self.hierarchy),
            "overflow_count": self.overflow,
        }
        return self.logs, self.step_metadata

class TimeHelper:
    """
    Helper class for time-related functions
    """

    @staticmethod
    def get_current_timestamp() -> str:
        """Get current timestamp as string"""
        dt = datetime.now().astimezone()
        tz = dt.strftime("%z")  # +0900
        tz = f"{tz[:3]}:{tz[3:]}"  # +09:00
        s = f"{dt:%Y-%m-%d %H:%M:%S.%f}{tz}"
        return s

    @staticmethod
    def perftimer_start() -> int:
        """Calculate performance time in seconds"""
        return time.perf_counter_ns()

    @staticmethod
    def perftimer_end(start_time: int) -> int:
        """Calculate performance time in nanoseconds"""
        end_time = time.perf_counter_ns()
        return end_time - start_time

    @staticmethod
    def perftimer_str(elapsed_time: int) -> str:
        """Get elapsed time as formatted string"""
        sec, rem = divmod(elapsed_time, 1_000_000_000)
        ms, rem = divmod(rem, 1_000_000)
        us, ns = divmod(rem, 1_000)

        parts = []
        if sec > 0:
            parts.append(f"{sec} s")
        if ms > 0:
            parts.append(f"{ms} ms")
        if us > 0:
            parts.append(f"{us} us")
        if ns > 0 or not parts:
            parts.append(f"{ns} ns")

        return ", ".join(parts)

class LogHelper:
    """
    Helper class for logging
    """

    @staticmethod
    def str_strip(s: str) -> str:
        """Strip hex string"""
        return s[2:] if s.startswith("0x") else s

    @staticmethod
    def str_to_bytes(s: str) -> bytes:
        """Convert hex string to bytes"""
        _s = s[2:] if s.startswith("0x") else s
        return bytes.fromhex(_s)

    @staticmethod
    def bytes_to_str(b: bytes) -> str:
        """
        Convert bytes to hex string

        Example:
            b'\\x12\\x34\\x56' -> '0x123456'
        
        Parameters:
            b (bytes):
                Input bytes
        Returns:
            hex_str (str): 
                Hex string representation
        """
        return f"0x{b.hex()}"

    @staticmethod
    def iter_to_bytes(t: Tuple[int] | List[int], byteorder: Optional[str] = None) -> bytes:
        """Convert Tuple or List of int to bytes"""
        assert all(0 <= _i < 256 for _i in t), "All integers must be in range 0-255"
        assert all(isinstance(_i, int) for _i in t), "All elements must be integers"
        assert byteorder in ("big", "little"), "byteorder must be 'big' or 'little'"

        byte_chunks = bytearray()
        if byteorder == "big":
            for _i in reversed(t):
                byte_chunks.append(_i)
        else:
            for _i in t:
                byte_chunks.append(_i)

        return bytes(byte_chunks)

    @staticmethod
    def bytes_to_int(b: bytes, byteorder: Optional[str] = None) -> tuple[int]:
        """Convert bytes to int"""
        assert byteorder in ("big", "little"), "byteorder must be specified"
        res = []
        if byteorder == "big":
            for _i in range(len(b)-1, -1, -1):
                res.append(int.from_bytes(b[_i:_i+1], byteorder=byteorder))
        else:
            for _i in range(len(b)):
                res.append(int.from_bytes(b[_i:_i+1], byteorder=byteorder))

        return tuple(res)

    @staticmethod
    def byte_to_int(b: bytes) -> int:
        """
        Convert bytes to integer using the specified byteorder.
        Parameters:
            b (bytes): Input byte.
        Returns:
            int: Converted integer.
        """
        assert isinstance(b, bytes), "Input must be bytes."
        assert len(b) == 1, "Input bytes length must be 1."

        return int.from_bytes(b, byteorder="big")

    @staticmethod
    def int_to_bytes(integer: int, length: int, byteorder: Optional[str] = None) -> bytes:
        """
        Convert int to bytes

        Parameters:
            integer (int):
                Input integer
            length (int):
                Length in bits
            byteorder (str):
                Byte order ('big' or 'little')

        Returns:
            bytes_repr (bytes):
                Byte representation of the integer
        """
        assert byteorder is not None, "byteorder must be specified"
        assert length >= 0, "length must be greater than or equal 0"

        return integer.to_bytes(length // 8, byteorder=byteorder)

    @staticmethod
    def idx_setter(index: Optional[int], prefix: str) -> Optional[str]:
        """Convert index to string"""
        # TODO
        # Add description for logs

        if index is None:
            return None

        if "Step" not in prefix:
            index += 1

        return f"{prefix} {index}"

    @staticmethod
    def json_file_namer(hash_alg: str, \
                        length: int, \
                        start_time: str, \
                        iteration: int, \
                        iteration_max: int) -> str:
        """Generate JSON file name"""
        width = len(str(abs(iteration_max)))
        ret = f"{hash_alg}_{length}_{start_time[:19]}_{str(iteration).zfill(width)}.json"

        return ret

    @staticmethod
    def iter_logs(io_controller,
                hash_cfg,
                main_cfg) -> Iterator[Dict]:
        """
        Stream Logs data from files one at a time.
        """
        latest_logs = \
            io_controller.get_latest_files_by_date(hash_cfg.hash_alg, \
                                                        hash_cfg.length)
        latest_logs.sort()

        assert len(latest_logs) > 0, "No Logs files found."

        for log_file in latest_logs:
            if main_cfg.verbose_flag:
                print(f"Loading Logs from file: {log_file}")

            log = io_controller.file_reader(log_file)

            yield {log_file.stem: log}

    @staticmethod
    def get_logs(io_controller,
                hash_cfg,
                main_cfg) -> List[Dict]:
        """
        Get Logs data from file.
        """
        return list(LogHelper.iter_logs(io_controller, hash_cfg, main_cfg))

    @staticmethod
    def iter_logs_with_hierarchy(io_controller,
                            hierarchy: Optional[List[str]] = None,
                            logs_path: Optional[List[Path]] = None) -> Iterator[Dict]:
        """
        Get Logs data from file with hierarchy information.
        """
        assert logs_path is not None, "logs_path must be specified"
        assert len(logs_path) > 0, "logs_path must not be empty"
        assert hierarchy is not None, "hierarchy must be specified"

        logs_path.sort()

        for log_file in logs_path:
            log = io_controller.file_reader(log_file)
            _hierarchy = log.get("Hierarchy", None)
            assert _hierarchy is not None, "No Hierarchy found in Logs."
            if isinstance(hierarchy, list):
                if len(hierarchy) == 0:
                    hierarchy.extend(_hierarchy)
                if "Block" not in hierarchy:
                    hierarchy.append("Block")
            yield {log_file.stem: log}



class Logs(LogHelper, TimeHelper):
    """
    Logging utilities
    """

    @staticmethod
    def clear(**kwargs) -> None:
        """
        Clear all logs
        Parameters:
            metadata: Metadata
            base_logs: BaseLogs
            step_logs: StepLogs
        """
        _metadata: Optional[Metadata] = None
        _base_logs: Optional[BaseLogs] = None
        _step_logs: Optional[StepLogs] = None

        if "metadata" in kwargs:
            _metadata: Optional[Metadata] = kwargs.get("metadata")
        if "base_logs" in kwargs:
            _base_logs: Optional[BaseLogs] = kwargs.get("base_logs")
        if "step_logs" in kwargs:
            _step_logs: Optional[StepLogs] = kwargs.get("step_logs")

        if _metadata is not None:
            _metadata.clear()
        if _base_logs is not None:
            _base_logs.clear()
        if _step_logs is not None:
            _step_logs.clear()
