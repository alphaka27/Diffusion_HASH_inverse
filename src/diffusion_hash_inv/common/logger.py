"""
Logging utilities for Diffusion Hash Inversion
"""
# pylint: disable=fixme
from __future__ import annotations
from typing import Any, Dict, List, Optional, Generator, Sequence, MutableMapping, Tuple
from datetime import datetime
from collections.abc import Hashable
from dataclasses import dataclass, field
import time
import math

from functools import wraps

@dataclass
class Metadata():
    """
    Metadata container  
    Contains hash algorithm, input bits length, execution start time,\
        elapsed time, entropy, strength
    """
    hash_alg: str
    is_message: bool

    entropy: float = field(default_factory=float, init=False)
    strength: str = field(default_factory=str, init=False)
    elapsed_time:float = field(default_factory=float, init=False)

    input_bits_len: int = field(default_factory=int, init=False)
    exec_start:str = field(default_factory=str, init=False)

    @staticmethod
    def calc_entropy(char_len: int, _pwd: str) -> float:
        """
        Calculate the entropy of the generated password.
        """
        entropy = char_len * math.log2(len(_pwd))
        return entropy

    def _set_strength(self)->None:
        """Set strength based on entropy"""
        if self.entropy < 28:
            self.strength = "Very Weak"
        elif 28 <= self.entropy < 36:
            self.strength = "Weak"
        elif 36 <= self.entropy < 60:
            self.strength = "Reasonable"
        elif 60 <= self.entropy < 128:
            self.strength = "Strong"
        else:
            self.strength = "Very Strong"

    def clear(self)->None:
        """Clear all metadata"""
        self.exec_start = ""
        # self.elapsed_time = 0.0

    def setter(self, input_length: int, exec_start: str)->None:
        """set metadata"""
        self.input_bits_len = input_length
        self.exec_start = exec_start

    def update(self, entropy: float, elapsed_time: int)->None:
        """
        Update metadata
        Parameters:
            entropy: float
                Entropy of the generated input
            strength: str
                Strength of the generated input
            elapsed_time: float
                Elapsed time for generation
        """
        self.entropy = entropy
        self._set_strength()
        self.elapsed_time = f"{elapsed_time} ns"

    def getter(self)->Dict[str, Any]:
        """
        Get Metadata
        """
        return {
            "Hash function": self.hash_alg,
            "Input bits": self.input_bits_len,
            "Program started at": self.exec_start,
            "Message mode": self.is_message,
            "Entropy": self.entropy,
            "Strength": self.strength,
            "Elapsed time": self.elapsed_time
        }

@dataclass
class BaseLogs():
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
                self.message["String"] = message_bytes.decode("utf-8")
            except UnicodeDecodeError:
                self.message["String"] = message_bytes.decode("utf-8", errors="replace")
        else:
            self.message.pop("String", None)

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
        self.step_metadata.clear()
        self.overflow = -1

    @staticmethod
    def _ensure_nested_dict(root: MutableMapping, path: Sequence[Hashable]) -> MutableMapping:
        """
        path[:-1]까지는 dict로 내려가고, 없으면 {}로 생성해서 내려가요.
        마지막 key 직전의 dict를 반환해요.
        """
        cur: MutableMapping = root
        for key in path:
            if key is None:
                pass
            cur = cur.setdefault(key, {})
        return cur

    # 1) 그냥 값 통째로 넣기 (bytes, dict, list 등 어떤 타입이든)
    def set_value(self, path: Sequence[Hashable], value: Any) -> None:
        """
        path 위치에 value를 그대로 넣어요.
        예: ("Step1",) -> bytes
            ("Step2",) -> 전체 list
            ("Step3",) -> 전체 dict
        """
        if not path:
            raise ValueError("path는 비어 있을 수 없어요.")
        *parents, last = path
        parent_dict = self._ensure_nested_dict(self.logs, parents)
        parent_dict[last] = value

    def dict_updater(self, **data) -> Dict[str, Any]:
        """Update dictionary type step logs"""
        temp_dict = {}
        for key, value in data.items():
            if isinstance(value, (bytearray, bytes)):
                temp_dict[key] = Logs.bytes_to_str(value)

            elif isinstance(value, int):
                value = LogHelper.int_to_bytes(value, self.wordsize, self.byteorder)
                temp_dict[key] = Logs.bytes_to_str(value)

            else:
                raise NotImplementedError("Unsupported type in dict for step log update")

            temp_dict[key] = Logs.bytes_to_str(value)
        return temp_dict

    def list_updater(self, data) -> Dict[str, Any]:
        """Update step logs list directly"""
        temp_dict = {}
        for idx, item in enumerate(data):
            idx_str = Logs.idx_setter(idx, " Block")

            if isinstance(item, (bytearray, bytes)):
                temp_dict[idx_str] = Logs.bytes_to_str(item)

            elif isinstance(item, int):
                item = LogHelper.int_to_bytes(item, self.wordsize, self.byteorder)
                temp_dict[idx_str] = Logs.bytes_to_str(item)

            elif isinstance(item, Sequence) and \
                not isinstance(item, (str, bytes, bytearray)):
                # Nested sequence handling
                _nested_list = []
                for _item in item:
                    if isinstance(_item, (bytearray, bytes)):
                        _item = Logs.bytes_to_str(_item)

                    elif isinstance(_item, int):
                        _item = LogHelper.int_to_bytes(_item, self.wordsize, self.byteorder)
                        _item = Logs.bytes_to_str(_item)

                    else:
                        raise NotImplementedError\
                            ("Unsupported type in nested sequence for step log update")

                    _nested_list.append(_item)
                temp_dict[idx_str] = _nested_list
            else:
                raise NotImplementedError("Unsupported type in sequence for step log update")
        return temp_dict

    def update(self, **kwargs) -> Dict[str, Any]:
        """
        Update step logs
        """
        _hier_path = []
        if self.hierarchy:
            for level in self.hierarchy:
                round_idx: Optional[str] = kwargs.get("round_idx", None)
                if round_idx is not None and level == "Round":
                    _hier_path.append(round_idx)

                step_index: Optional[str] = kwargs.get("step_index", None)
                if step_index is not None and level == "Step":
                    _hier_path.append(step_index)

        step_result: Dict | Sequence | bytes | bytearray = kwargs.get("step_result", None)

        assert step_result is not None, "step_result must be provided"
        assert isinstance(step_result, (Dict, Sequence, bytes, bytearray)), \
            "step_result must be Dict, Sequence, or bytes-like"

        _logs: Dict[str, Any] | str = None
        if isinstance(step_result, Dict):
            _logs = self.dict_updater(**step_result)

        elif isinstance(step_result, Sequence) and \
        not isinstance(step_result, (str, bytes, bytearray)):
            _logs = self.list_updater(step_result)

        elif isinstance(step_result, (bytes, bytearray)):
            # For bytes results, store as hex string
            _logs = Logs.bytes_to_str(step_result)

        else:
            raise NotImplementedError("Unsupported type for step log update")
        self.set_value(_hier_path, _logs)

    def update_loop(self, **kwargs) -> Dict[str, Any]:
        """Update step logs in loop"""
        _hier_path = []
        if self.hierarchy:
            for level in self.hierarchy:
                round_idx: Optional[str] = kwargs.get("round_idx", None)
                if round_idx is not None and level == "Round":
                    _hier_path.append(round_idx)

                step_index: Optional[str] = kwargs.get("step_index", None)
                if step_index is not None and level == "Step":
                    _hier_path.append(step_index)

                loop_index: Optional[int] = kwargs.get("loop_index", None)
                loop_idx_str = Logs.idx_setter(loop_index, " Loop")
                if loop_index is not None and level == "Loop":
                    _hier_path.append(loop_idx_str)

        loop_result: Any = kwargs.get("step_result", None)
        assert loop_result is not None, "step_result must be provided"
        assert isinstance(loop_result, (Dict, Sequence, bytes, bytearray)), \
            "loop_result must be Dict, Sequence, or bytes-like"

        if isinstance(loop_result, Dict):
            # For dict results, store directly
            _logs = self.dict_updater(**loop_result)

        else:
            raise NotImplementedError("Unsupported type for step log update")
        self.set_value(_hier_path, _logs)

    def getter(self) -> Dict[str, Any]:
        """Get step logs"""
        self.step_metadata.update({"word_size": self.wordsize, "byteorder": self.byteorder})
        self.step_metadata.update({"hierarchy": self.hierarchy})
        self.step_metadata.update({"overflow_count": self.overflow})
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
    def perftimer(elapsed_time: int) -> str:
        """Calculate elapsed performance time in nanoseconds"""
        sec, rem = divmod(elapsed_time, 1_000_000_000)
        ms, rem = divmod(rem, 1_000_000)
        us, ns = divmod(rem, 1_000)

        return f"{elapsed_time} ns ({sec} s {ms} ms, {us} us, {ns} ns)"


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
    def idx_setter(index: Optional[int], suffix: str, **desc) -> Optional[str]:
        """Convert index to string"""
        # TODO
        # Add description for logs
        _desc = desc.get("description", None)
        suffix = f"{suffix}" if _desc is None else f"{suffix}-{_desc}"
        if index is None:
            return None

        if "Step" not in suffix:
            index += 1

        if index == 1:
            return "1st" + suffix
        if index == 2:
            return "2nd" + suffix
        if index == 3:
            return "3rd" + suffix
        return f"{index}th" + suffix


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
        _metadata: Optional[Metadata] = kwargs.get("metadata")
        _base_logs: Optional[BaseLogs] = kwargs.get("base_logs")
        _step_logs: Optional[StepLogs] = kwargs.get("step_logs")

        if _metadata is not None:
            _metadata.clear()
        if _base_logs is not None:
            _base_logs.clear()
        if _step_logs is not None:
            _step_logs.clear()

    #decorator
    @staticmethod
    def steplogs_update(step_cat: Optional[str] = None, is_loop: bool = False, **desc) -> Any:
        """
        Update all logs
        Parameters:
            step_cat (Optional[str]):
                Step category
            is_loop (bool):
                Whether the decorated function is a loop
            desc (Dict[str, Any]):
                Additional description for logs in the form of kwargs
        """

        step_idx = Logs.idx_setter(step_cat, " Step", **desc)

        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                # Get block iteration and set description if exists
                round_iteration: Optional[int] = kwargs.pop("Round_Index", None)
                round_idx: Optional[str] = Logs.idx_setter(round_iteration, " Round")

                # Ensure the class has step_logs attribute
                assert hasattr(self, 'step_logs'), \
                    "The class must have 'step_logs' attribute."
                assert isinstance(self.step_logs, StepLogs), \
                    "'step_logs' must be an instance of StepLogs."
                assert hasattr(self, 'word_size'), \
                    "The class must have 'word_size' attribute."
                assert hasattr(self, 'byteorder'), \
                    "The class must have 'byteorder' attribute."

                # Call the original function
                org: bytes | Sequence | Generator | Dict = func(self, *args, **kwargs)

                assert hasattr(self, 'total_overflow_count'), \
                    "The class must have 'overflow' attribute."
                assert hasattr(self, 'get_variable'), \
                    "The class must have 'get_variable' method."
                # Look base_calc.py if assertion fails

                _overflow = self.get_variable("total_overflow_count")
                _overflow = max(self.step_logs.overflow, _overflow)
                self.step_logs.overflow = _overflow
                _is_overflow = self.get_variable("overflow_boolean")
                self.set_variable("overflow_boolean", False)
                print(f"Total overflow status: {_overflow}")

                _result = None
                # Validate the result type based on is_loop
                assert is_loop and isinstance(org, Generator) or not is_loop, \
                    "If is_loop is True, the result must be a Generator."

                if is_loop:
                    _i = 0
                    print(step_idx)
                    while True:
                        try:
                            loop_result = next(org)
                            StepLogs.update_loop(self.step_logs,
                                round_idx=round_idx,
                                step_index=step_idx,
                                loop_index=_i,
                                step_result=loop_result,
                                wordsize=self.word_size,
                                byteorder=self.byteorder,
                                is_overflow=_is_overflow)
                            _i += 1
                        except StopIteration as e:
                            _result = e.value
                            break

                else:
                    _result = org
                    print(step_idx)
                    StepLogs.update(self.step_logs,
                        round_idx=round_idx,
                        step_index=step_idx,
                        step_result=org,
                        wordsize=self.word_size,
                        byteorder=self.byteorder,
                        is_overflow=_is_overflow)
                assert _result is not None, "Log update failed in loop."
                return _result
            return wrapper
        return decorator
