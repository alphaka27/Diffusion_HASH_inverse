"""
JSON-safe formatter
"""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import json
import numpy as np

@dataclass
class Metadata():
    """
    Metadata container  
    Contains hash algorithm, input bits length, execution start time,\
        elapsed time, entropy, strength
    """
    def __init__(self, hash_alg:str, input_bits_len:int, entropy:float)->None:
        Metadata.hash_alg:str = hash_alg
        Metadata.input_bits_len:int = input_bits_len
        self.exec_start:str = ""
        self.elapsed_time:float = 0.0
        Metadata.entropy:float = entropy
        Metadata.strength:str = ""

    def clear(self)->None:
        """Clear all metadata"""
        self.exec_start = ""
        self.elapsed_time = 0.0

    def setter(self, exec_start:str, elapsed_time:Optional[float])->None:
        """Set Metadata"""
        self.exec_start = exec_start
        self.elapsed_time = elapsed_time
        if self.entropy < 28:
            self.strength = "Very Weak"
        elif self.entropy < 36:
            self.strength = "Weak"
        elif self.entropy < 60:
            self.strength = "Reasonable"
        elif self.entropy < 128:
            self.strength = "Strong"
        else:
            self.strength = "Very Strong"

    def update(self):
        """
        Update Metadata
        """

    def getter(self)->Dict[str, Any]:
        """
        Get Metadata
        """
        return {
            "Hash function": self.hash_alg,
            "Input bits": self.input_bits_len,
            "Program started at": self.exec_start,
            "Entropy": self.entropy,
            "Strength": self.strength
        }

@dataclass
class BaseLogs():
    """
    Base logs container
    Contains Message(String and Hex), Generated hash, Correct hash
    """

    def __init__(self):
        self.message = {"String": "", "Hex": ""}
        self.generated_hash = ""
        self.correct_hash = ""

    def clear(self):
        """Clear all base logs"""
        self.message = {"String": "", "Hex": ""}
        self.generated_hash = ""
        self.correct_hash = ""

    def set_message(self, message_bytes: bytes, is_message_mode: bool):
        """Set message"""
        if is_message_mode:
            try:
                self.message["Hex"] = message_bytes
                self.message["String"] = message_bytes.decode("utf-8")
            except UnicodeDecodeError:
                self.message["Hex"] = message_bytes
                self.message["String"] = message_bytes.decode("utf-8", errors="replace")

        else:
            self.message["Hex"] = message_bytes.hex()
            self.message["String"] = "Input is in BYTE mode."

    def set_hashes(self, generated_hash: str, correct_hash: str):
        """Set hash result"""
        self.generated_hash = generated_hash
        self.correct_hash = correct_hash

    def update(self):
        """
        Update base logs
        """

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

    def __init__(self):
        super().__init__()
        self.key_list: List[str] = field(default_factory=list)
        self.val_list: List[Any] = field(default_factory=list)

    def clear(self):
        """Clear all step logs"""
        self.key_list = []
        self.val_list = []

    def key_setter(self, key:List[str])->None:
        """Dict key setter"""
        self.key_list.append(key)

    def val_setter(self, val:List[str])->None:
        """Dict value setter"""
        self.val_list.append(val)

    def update(self):
        """
        Update step logs
        """

    def dict_getter(self)->Dict[str, Any]:
        """Get step logs as dict"""


class OutputFormat:
    """
    Class to handle output formatting for hash results.
    """
    def __init__(self):
        self.metadata = Metadata("", 0, 0.0)
        self.base_logs = BaseLogs()
        self.step_logs = StepLogs()

    @staticmethod
    def json_safe(o):
        """JSON-safe """
        if isinstance(o, np.ndarray):
            return [OutputFormat.json_safe(v) for v in o.tolist()]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, (list, tuple)):
            return [OutputFormat.json_safe(v) for v in o]
        if isinstance(o, dict):
            return {k: OutputFormat.json_safe(v) for k, v in o.items()}
        return o

    def dumps(self, indent=4, data=None):
        """Make JSON dump"""
        ret = OutputFormat.json_safe(self.to_dict()) \
            if data is None else OutputFormat.json_safe(data)
        return json.dumps(ret, ensure_ascii=False, indent=indent)
