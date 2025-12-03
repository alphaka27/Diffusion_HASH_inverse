"""
File Output Formatter for Diffusion HASH Inversion
Handles formatting of logs into various file formats such as XLSX, JSON, etc.
"""
from __future__ import annotations
from typing import Dict, Any
import json
import copy

from diffusion_hash_inv.common import Metadata, BaseLogs, StepLogs

class JSONFormat:
    """
    Class to handle output formatting for hash results.
    """
    @staticmethod
    def loads(json_string: str):
        """Make JSON load"""
        data = json.loads(json_string) if isinstance(json_string, (str, bytes)) \
            else dict(json_string)

        meta_raw = data.get("Metadata")
        if meta_raw is None:
            raise ValueError("Metadata not found in JSON.")
        metadata = Metadata(hash_alg=meta_raw.get("Hash function"), \
                            is_message=meta_raw.get("Message mode"))
        metadata.input_bits_len = meta_raw.get("Input bits", 0)
        metadata.exec_start = meta_raw.get("Program started at", "")
        metadata.entropy = meta_raw.get("Entropy", 0.0)
        metadata.strength = meta_raw.get("Strength", "")
        metadata.elapsed_time = meta_raw.get("Elapsed time", "")
        ret_meta: Dict[str, Any] = {"Metadata": copy.deepcopy(metadata.getter())}

        baselog_key_map = {
            "Message": "message",
            "Generated hash": "generated_hash",
            "Correct   hash": "correct_hash",
        }
        base_logs = BaseLogs()
        for json_key, attr in baselog_key_map.items():
            value = data.get(json_key)
            if value is None:
                raise ValueError(f"Key {json_key} not found in JSON.")
            setattr(base_logs, attr, value)
        ret_base: Dict[str, Any] = {"BaseLogs": copy.deepcopy(base_logs.getter())}

        step_raw = data.get("Logs")
        step_meta = data.get("Step Metadata")

        if step_raw is None:
            raise ValueError("Step Logs not found in JSON.")
        if step_meta is None:
            raise ValueError("Step Metadata not found in JSON.")

        step_logs: Dict[str, Any] = {"Logs": step_raw, "Step Metadata": step_meta}
        ret = {**ret_meta, **ret_base, **step_logs}

        return ret

    @staticmethod
    def dumps(indent=4, **data):
        """Make JSON dump"""
        metadata = data.get("metadata", None)
        baselogs = data.get("baselogs", None)
        steplogs = data.get("steplogs", None)
        assert isinstance(metadata, Metadata), "metadata must be an instance of Metadata"
        assert isinstance(baselogs, BaseLogs), "baselogs must be an instance of BaseLogs"
        assert isinstance(steplogs, StepLogs), \
            f"steplogs must be an instance of StepLogs, {type(steplogs)} given"
        _all_data = {"Metadata": metadata.getter()}
        _all_data.update(dict(baselogs.getter().items()))
        step_log, step_meta = steplogs.getter()
        _all_data.update({"Logs": step_log, "Step Metadata": step_meta})

        return json.dumps(_all_data, ensure_ascii=False, indent=indent)
