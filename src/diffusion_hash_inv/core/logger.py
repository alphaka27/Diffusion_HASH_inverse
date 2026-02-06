"""
Logging utilities for Diffusion Hash Inversion
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, ClassVar

from datetime import datetime
from collections.abc import Hashable
from dataclasses import dataclass, field
import time
import inspect
import types
from enum import Enum
import sys
from functools import wraps
import threading

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
            "Hash function": self.hash_alg,
            "Hierarchy": self.hierarchy,
            "Input bits": self.input_bits_len,
            "Message mode": "Message" if self.is_message else "Bit string",
            "Program started at": self.started_at,
            "Program elapsed time": self.elapsed_time,
            "Byte order": self.byteorder
        }

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

    def clear(self) -> None:
        """Clear all step logs"""
        self.logs.clear()
        self.overflow_log.clear()
        self.step_metadata.clear()
        self.overflow = -1

    def update_step(self):
        """Update step logs"""

    def update_overflow(self, step) -> None:
        """
        Update overflow logs

        Parameters:
            step: Any
                Step information
        """


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

class MemberKind(Enum):
    """
    Enum for classifying member kinds
    """
    # class attributes
    CLASS_STATICMETHOD = "class: staticmethod"
    CLASS_CLASSMETHOD = "class: classmethod"
    CLASS_FUNCTION = "class: function (binds as instance method)"
    CLASS_PROPERTY = "class: property"
    CLASS_OTHER = "class: other"

    # module attributes
    MODULE_FUNCTION = "module: function"
    MODULE_CLASS = "module: class"
    MODULE_MODULE = "module: module"
    MODULE_BUILTIN = "module: builtin"          # e.g., math.sin
    MODULE_OTHER = "module: other"

    # fallback
    OWNER_UNSUPPORTED = "owner: unsupported"

    @property
    def description(self) -> str:
        """Get description of the member kind"""
        return self.value

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
        assert byteorder is not None, "byteorder must be specified"
        assert all(0 <= _i < 256 for _i in t), "All integers must be in range 0-255"
        assert all(isinstance(_i, int) for _i in t), "All elements must be integers"

        byte_chunks = bytearray()
        for _i in t:
            byte_chunks.append(_i)

        return bytes(byte_chunks)

    @staticmethod
    def bytes_to_int(b: bytes, byteorder: Optional[str] = None) -> tuple[int]:
        """Convert bytes to int"""
        assert byteorder is not None, "byteorder must be specified"
        res = []
        for _i in range(len(b)):
            res.append(int.from_bytes(b[_i:_i+1], byteorder=byteorder))
        return tuple(res)

    @staticmethod
    def byte_to_int(b: bytes, byteorder: Optional[str] = None) -> int:
        """
        Convert bytes to integer using the specified byteorder.
        Parameters:
            b (bytes): Input byte.
        Returns:
            int: Converted integer.
        """
        assert isinstance(b, bytes), "Input must be bytes."
        assert byteorder in ('big', 'little'), "Byteorder must be 'big' or 'little'."

        return int.from_bytes(b, byteorder)

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
    def stdout_logs(*args, verbose: bool = True, message: bool = False, **kwargs) -> None:
        """
        Custom print function for logging

        Parameters:
            verbose (bool):
                If True, print the message. Default is **True**.
            message (bool):
                If True, decode bytes to string before printing. Default is **False**.
            sep (str):
                Separator between arguments. Default is space.
            end (str):
                End character. Default is newline.
            flush (bool):
                Whether to flush the output. Default is False.
        """
        if not verbose:
            return
        processed_args: List[str] = []
        _word_size: Optional[int] = kwargs.pop("word_size", None)
        _byteorder: Optional[str] = kwargs.pop("byteorder", None)
        assert _word_size is not None, "word_size must be specified"
        assert _byteorder is not None, "byteorder must be specified"

        for arg in args:
            if isinstance(arg, (bytes, bytearray)):
                if message:
                    try:
                        decoded_msg = arg.decode("UTF-8")
                    except UnicodeDecodeError:
                        decoded_msg = arg.decode("UTF-8", errors="replace")
                else:
                    decoded_msg = Logs.bytes_to_str(arg)
                processed_args.append(decoded_msg)
            elif isinstance(arg, Dict):
                arg_key = list(arg.keys())
                arg_val = list(arg.values())

                for _val in arg_val:
                    _val = LogHelper.bytes_to_str(\
                        LogHelper.int_to_bytes(_val, _word_size, byteorder=_byteorder))


            else:
                processed_args.append(arg)

        print(*processed_args, **kwargs)

    @staticmethod
    def stdout_result(*args, **kwargs) -> None:
        """
        Custom print function for result logging

        Parameters:
            message: bool - If True, print the message. Default is **True**.
        """
        message_flag: bool = kwargs.pop("message", True)

        iteration_index: Optional[int] = kwargs.pop("iteration_index", None)
        assert iteration_index is not None, "iteration_index must be specified"
        hash_alg: str = kwargs.pop("hash_alg", None)
        assert hash_alg is not None, "hash_alg must be specified"

        _metadata = kwargs.pop("metadata", None)
        assert _metadata is not None, "metadata must be provided"
        _base_logs = kwargs.pop("base_logs", None)
        assert _base_logs is not None, "base_logs must be provided"
        _step_logs = kwargs.pop("step_logs", None)
        assert _step_logs is not None, "step_logs must be provided"


        generated_hash, valid, correct_hash = args
        if message_flag:
            decoded_msg = msg.decode("UTF-8")
            str_len_bits = f"{len(decoded_msg) * 8} bits"

        else:
            decoded_msg = msg
            str_len_bits = f"{len(decoded_msg)} bits"

        # str_entropy = f"Entropy = {entropy}"
        # str_strength = f"Strength = {strength}"

        # print(f"Input ({str_len_bits}, {str_entropy}, {str_strength}):")
        print(f"Input ({str_len_bits}):")
        print(f"{decoded_msg}\n")
        print(f"----------------Result for iteration ({iteration_index + 1})----------------")
        print(f"Generated {hash_alg.upper()} Hash: ")
        hex_str = ''.join(f"{x:08x}" for x in generated_hash)
        print(f"{hex_str}\n")

        print(f"--------------Validation for iteration ({iteration_index + 1})--------------")
        print(f"Correct {hash_alg.upper()} Hash: ")
        print(f"{correct_hash}\n")

        print("--------------Validation result--------------")
        valid_message = "Fail" if not valid else "Success"
        print(f"Validation: {valid_message}")

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
    def _classify_class_attr(raw) -> str:
        """Classify class attribute type"""
        if isinstance(raw, staticmethod):
            return MemberKind.CLASS_STATICMETHOD.name
        if isinstance(raw, classmethod):
            return MemberKind.CLASS_CLASSMETHOD.name
        if inspect.isfunction(raw):
            return MemberKind.CLASS_FUNCTION.name
        if isinstance(raw, property):
            return MemberKind.CLASS_PROPERTY.name
        return MemberKind.CLASS_OTHER.name

    @staticmethod
    def _classify_module_attr(raw) -> str:
        """Classify module attribute type"""
        if inspect.isfunction(raw):
            return MemberKind.MODULE_FUNCTION.name
        if inspect.isclass(raw):
            return MemberKind.MODULE_CLASS.name
        if inspect.ismodule(raw):
            return MemberKind.MODULE_MODULE.name
        if inspect.isbuiltin(raw):
            return MemberKind.MODULE_BUILTIN.name
        return MemberKind.MODULE_OTHER.name

    @staticmethod
    def classify_member(owner, name: str) -> str:
        """
        Classify member type
        Parameters:
            owner: class 또는 module
            name: 그 안의 속성 이름
        """
        raw = inspect.getattr_static(owner, name)

        # class에 붙은 경우: staticmethod/classmethod/일반 메서드 후보 구분 가능
        if isinstance(owner, type):
            return LogHelper._classify_class_attr(raw)

        # module에 붙은 경우: 모듈 레벨 함수/클래스/변수 구분
        if isinstance(owner, types.ModuleType):
            return LogHelper._classify_module_attr(raw)

        return f"owner type not supported: {type(owner).__name__}"


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

    @staticmethod
    def is_dunder(name: str) -> bool:
        """Check if the name is a dunder method"""
        return name.startswith("__") and name.endswith("__")

    @staticmethod
    def trace_wrapper(fn, watch_vars, output, **kwargs):
        """
        Trace wrapper for logging
        Parameters:
            fn: Callable
                Target function to trace
            output: Any
        """
        tl = threading.local()
        if "show_return" in kwargs:
            tl.show_return = kwargs["show_return"]
        else:
            tl.show_return = True

        def tracer(frame, event, arg):
            """
            Trace function for logging
            Parameters:
                fn: Callable
                    Target function to trace
                frame: FrameType
                event: str
                arg: Any
            """
            nonlocal tl
            nonlocal output

            tl.target_code = fn.__code__
            tl.active = False
            tl.depth = 0
            tl.local_vars = dict(frame.f_locals)
            tl.loop_tracker_j = 0
            tl.loop_tracker_i = 0


            if not tl.active:
                if event == "call" and frame.f_code is tl.target_code:
                    tl.active = True
                    tl.depth = 1
                    print("  " * (tl.depth-1) + f"→ {fn.__name__}:{frame.f_lineno}")
                    return tracer
                return None

            if event == "call":
                tl.depth += 1
                if not Logs.is_dunder(fn.__name__):
                    print("  " * (tl.depth-1) + f"→ {fn.__name__}:{frame.f_lineno}")
                return tracer

            if event == "line":
                if not Logs.is_dunder(fn.__name__):
                    print("  " * (tl.depth-1) + f"↷ {fn.__name__}:{frame.f_lineno}")
                    print("  " * (tl.depth) + f"At line {frame.f_lineno}:")
                    print("  " * (tl.depth) + "All local variables:")
                    for var_name, var_value in tl.local_vars.items():
                        print("  " * (tl.depth) + f"{var_name} = {var_value!r}")
                    print("  " * (tl.depth) + "Watching variables:")

            if event == "return":
                if tl.show_return:
                    msg = f"← {fn.__name__}:{frame.f_lineno}"
                    print("  " * (tl.depth-1) + msg)
                    print("  " * (tl.depth-1) + f"  ret={arg!r}\n")

                tl.depth -= 1
                if tl.depth == 0:
                    tl.active = False
                return tracer

    @classmethod
    def logger(cls, step: int, watch_var: Tuple[str, ...], logs_save: str, show_ret = True) -> Any:
        """
        Update all logs
        Parameters:
            step_cat (Optional[str]):
                Step category
        """
        step_cat: str = Logs.idx_setter(step, "Step")

        def deco(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                self = args[0]

                print(f"{self.__class__.__name__}.{fn.__name__} called.")

                assert hasattr(self, logs_save), \
                    f"{logs_save} is not an attribute of {type(self).__name__}."

                logs = getattr(self, logs_save)
                old = sys.gettrace()
                sys.settrace(Logs.trace_wrapper(fn, \
                                                watch_vars=watch_var, output=logs, show_return=show_ret))

                try:
                    return fn(*args, **kwargs)
                finally:
                    sys.settrace(old)

            return wrapper
        return deco
