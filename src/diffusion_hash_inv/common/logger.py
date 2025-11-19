"""
Logging utilities for Diffusion Hash Inversion
"""
# pylint: disable=fixme
from __future__ import annotations
from typing import Any, Dict, List, Optional, Generator
from datetime import datetime
from dataclasses import dataclass, field

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

    # TODO
    # entropy: float = field(default_factory=float)
    # strength: str = field(default_factory=str)
    # elapsed_time:float = field(default_factory=float)

    input_bits_len: int = field(default_factory=int)
    exec_start:str = field(default_factory=str)


    # TODO
    '''
    @staticmethod
    def calc_entropy(char_len: int, _pwd: str) -> float:
        """
        Calculate the entropy of the generated password.
        """
        entropy = char_len * math.log2(len(_pwd))
        return entropy

    def __set_strength(self)->None:
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
    '''

    def clear(self)->None:
        """Clear all metadata"""
        self.exec_start = ""
        # self.elapsed_time = 0.0

    def setter(self, hash_alg: str, is_message: bool)->None:
        """set metadata"""
        self.hash_alg = hash_alg
        self.is_message = is_message

    def update(self, message: bytes, decode: str = "UTF-8")->None:
        """update metadata"""
        if self.is_message:
            self.input_bits_len = len(message.decode(decode)) * 8
        else:
            self.input_bits_len = len(message) * 8

    def getter(self)->Dict[str, Any]:
        """
        Get Metadata
        """
        return {
            "Hash function": self.hash_alg,
            "Input bits": self.input_bits_len,
            "Program started at": self.exec_start,
            "Message mode": self.is_message
        }

@dataclass
class BaseLogs():
    """
    Base logs container
    Contains Message(String and Hex), Generated hash, Correct hash
    """
    message: Dict[str, Any] = field(default_factory=dict)
    generated_hash: bytes = field(default_factory=bytes)
    correct_hash: bytes = field(default_factory=bytes)

    def clear(self):
        """Clear all base logs"""
        self.message = {}
        self.generated_hash = b""
        self.correct_hash = b""

    def set_message(self, message_bytes: bytes, message_mode: bool):
        """Set message"""
        self.message["Hex"] = message_bytes.hex()
        if message_mode:
            try:
                self.message["String"] = message_bytes.decode("utf-8")
            except UnicodeDecodeError:
                self.message["String"] = message_bytes.decode("utf-8", errors="replace")
        else:
            self.message.pop("String", None)

    def set_hashes(self, generated_hash: str, correct_hash: str):
        """Set hash result"""
        self.generated_hash = generated_hash
        self.correct_hash = correct_hash

    def update(self, **data) -> None:
        """
        Update base logs
        Parameters:
            message: bytes
                Message bytes
            is_message: bool
                Whether the message is in string mode
            generated_hash: bytes
                Generated hash bytes
            correct_hash: bytes
                Correct hash bytes
        """
        message = data.get("message", None)
        is_message = data.get("is_message", True)
        generated_hash = data.get("generated_hash", None)
        correct_hash = data.get("correct_hash", None)

        if message is not None:
            self.set_message(message, is_message)
        if generated_hash is not None and correct_hash is not None:
            self.set_hashes(generated_hash, correct_hash)

    def getter(self)->Dict[str, Any]:
        """Get base logs"""
        return {
            "Message": self.message,
            "Generated hash": self.generated_hash,
            "Correct   hash": self.correct_hash
        }

@dataclass
class StepLogs():
    """
    Step logs container that behaves like a dict while keeping per-step structure.
    """
    round_logs: Dict[str, Dict] = field(default_factory=dict)
    step_logs: Dict[str, Any] = field(default_factory=dict)
    value: Any = None
    step_metadata: Dict[str, Any] = field(default_factory=dict)

    def clear(self) -> None:
        """Clear all step logs"""
        self.round_logs.clear()
        self.step_logs.clear()
        self.value = None

    def update(self, **kwargs) -> Dict[str, bytes]:
        """
        Update step logs
        """
        round_idx: Optional[str] = kwargs.get("round_idx", None)
        step_index: Optional[str] = kwargs.get("step_index", None)
        step_result: Any = kwargs.get("step_result", None)


    def update_loop(self) -> List[bytes]:
        """Update step logs in loop"""

    def dict_maker(self) -> Dict[str, Any]:
        """Make dict from step logs"""

class LogHelper:
    """
    Helper class for logging
    """

    @staticmethod
    def now_time() -> str:
        """Get current time as string"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    def int_to_bytes(i: int, length: int, byteorder: Optional[str] = None) -> bytes:
        """Convert int to bytes"""
        assert byteorder is not None, "byteorder must be specified"

        return i.to_bytes(length, byteorder=byteorder)

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
                processed_args.append(str(arg))
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




class Logs(LogHelper):
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

    @staticmethod
    def idx_setter(index: Optional[int], suffix: str, **desc) -> Optional[str]:
        """Convert index to string"""
        # TODO
        # Add description for logs
        _ = desc

        if index is None:
            return None
        if index == 0:
            return "1st" + suffix
        if index == 1:
            return "2nd" + suffix
        if index == 2:
            return "3rd" + suffix
        return f"{index}th" + suffix

    @staticmethod
    def steplogs_update(step_cat: Optional[str] = None, is_loop: bool = False, **desc):
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
                block_iteration: Optional[int] = kwargs.pop("Block_Index", None)
                block_idx: Optional[str] = None
                if block_iteration is not None:
                    block_idx = Logs.idx_setter(block_iteration, " Block", **desc)

                assert hasattr(self, 'step_logs'), \
                    "The class must have 'step_logs' attribute."
                assert isinstance(self.step_logs, StepLogs), \
                    "'step_logs' must be an instance of StepLogs."

                result = func(self, *args, **kwargs)

                assert is_loop and isinstance(result, Generator) or not is_loop, \
                    "If is_loop is True, the result must be a Generator."
                _result = None

                if is_loop:
                    _result = self.step_logs.update_loop()
                else:
                    _result = self.step_logs.update(round_idx=block_idx,
                                                    step_index=step_idx,
                                                    step_result=result)
                assert _result is not None, "Log update failed."

                return _result
            return wrapper
        return decorator

    @staticmethod
    def get_all(metadata: Metadata,
                base_logs: BaseLogs,
                step_logs: StepLogs) -> Dict[str, Any]:
        """Get all logs as dict"""
        return {
            "Metadata": metadata.getter(),
            "Base Logs": base_logs.getter(),
            "Step Logs": step_logs.dict_getter(),
        }
