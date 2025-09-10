"""
JSON-safe formatter
"""
from typing import Any, Dict
import json
import numpy as np

class OutputFormat:
    """
    Class to handle output formatting for SHA-256 hash results.

    Output Format
    """
    def __init__(self):
        self.metadata: Dict[str, Any] = {}
        self.message: str = ""
        self.generated_hash: str = ""
        self.correct_hash: str = ""
        self.rounds: Dict[str, Any] = {}
        self.step_logs: Dict[str, Any] = {
            "Message Schedule(Step1)": [],
            "Initialize working variables(Step2)": [],
            "Main Compute Function loops(Step3)": {},
            "Finalize the hash value(Step4)": []
        }

    def _ret_dict_key(self, in_key: str):
        dict_keys = {
            "Step1": "Message Schedule(Step1)",
            "Step2": "Initialize working variables(Step2)",
            "Step3": "Main Compute Function loops(Step3)",
            "Step4": "Finalize the hash value(Step4)"}

        return dict_keys[in_key.capitalize()]

    def reset(self, only_step = False):
        """
        Reset variables in class
        """
        if not only_step:
            self.metadata.clear()
            self.message = ""
            self.generated_hash = ""
            self.correct_hash = ""

        self.step_logs = ({
            "Message Schedule(Step1)": [],
            "Initialize working variables(Step2)": [],
            "Main Compute Function loops(Step3)": {},
            "Finalize the hash value(Step4)": []
        })


    @staticmethod
    def to_hex32_scalar(x) -> str:
        """단일 32-bit 값 → 8자리 hex"""
        return "0x" + f"{int(x):08x}"

    @staticmethod
    def to_hex32_concat(seq) -> str:
        """시퀀스(8워드 등) → 64자리 hex"""
        return ''.join(f"{int(x):08x}" for x in seq)

    def set_metadata(self, input_bits_len:int, exec_start:str, elapsed_time:float, entropy:float):
        """Set Metadata"""
        strength = ""
        if entropy < 28:
            strength = "Very Weak"
        elif entropy < 36:
            strength = "Weak"
        elif entropy < 60:
            strength = "Reasonable"
        elif entropy < 128:
            strength = "Strong"
        else:
            strength = "Very Strong"

        self.metadata = {
            "Input bits": input_bits_len,
            "Program started at": exec_start,
            "Elapsed time": elapsed_time,
            "Entropy": entropy,
            "Strength": strength
        }

        return strength

    def set_message(self, message_bytes: bytes, is_message_mode: bool):
        """Set message"""
        if is_message_mode:
            try:
                self.message = message_bytes.decode("utf-8")
            except UnicodeDecodeError:
                self.message = message_bytes.decode("utf-8", errors="replace")
        else:
            self.message = message_bytes.hex()

    def set_hashes(self, generated_words8, correct_hex: str):
        """Set hash result"""
        self.generated_hash = self.to_hex32_concat(generated_words8)
        self.correct_hash = correct_hex

    def add_preprocess(self, val):
        """Add preprocess log"""
        for _k, _v in val.items():
            tmp = []
            for _tmp in _v:
                tmp.append(OutputFormat.to_hex32_scalar(_tmp))
            val[_k] = tmp
        self.rounds["Preprocess"] = val

    def add_step1(self, w64):
        """Add step1 log"""
        self.step_logs[self._ret_dict_key("step1")] \
            = [self.to_hex32_scalar(w) for w in w64]

    def add_step2(self, val):
        """Add step2 log"""
        self.step_logs[self._ret_dict_key("step2")] \
            = val

    def add_step3_round(self, round_idx:int, val):
        """Add step3 log"""
        _round_idx = round_idx + 1
        if _round_idx == 1:
            loop_m = f"{_round_idx}st loop"
        elif _round_idx == 2:
            loop_m = f"{_round_idx}nd loop"
        elif _round_idx == 3:
            loop_m = f"{_round_idx}rd loop"
        else:
            loop_m = f"{_round_idx}th loop"
        self.step_logs[self._ret_dict_key("step3")][loop_m] \
            = val

    def add_step4(self, out_words8):
        """Add step4 log"""
        self.step_logs[self._ret_dict_key("step4")] = out_words8

    def add_round(self, round_idx):
        """Add round for long message"""
        _round_idx = round_idx + 1
        if _round_idx  == 1:
            _idx = f"{_round_idx}st round"
        elif _round_idx == 2:
            _idx = f"{_round_idx}nd round"
        elif _round_idx == 3:
            _idx = f"{_round_idx}rd round"
        else:
            _idx = f"{_round_idx}th round"

        self.rounds[_idx] = self.step_logs
        self.reset(only_step=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serializatoin"""
        ret_dict = {
            "Metadata": self.metadata,
            "Message": self.message,
            "Generated hash": self.generated_hash,
            "Correct   hash": self.correct_hash,
            "Rounds logs": self.rounds
        }
        return ret_dict

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
