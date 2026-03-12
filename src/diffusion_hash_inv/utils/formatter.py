"""
File Output Formatter for Diffusion HASH Inversion
Handles formatting of logs into various file formats such as XLSX, JSON, etc.
"""
from __future__ import annotations
from typing import Dict, Any
import json

from diffusion_hash_inv.logger import Metadata, BaseLogs, StepLogs

class JSONFormat:
    """
    Class to handle output formatting for hash results.
    """
    @staticmethod
    def loads(json_string: str):
        """Make JSON load"""
        data = json.loads(json_string)
        ret: Dict[str, Any] = {}

        metadata = data.get("Metadata", None)
        message = data.get("Message", None)
        message = message.get("Hex", None) if message is not None else None
        step_logs = data.get("Logs", None)
        assert metadata is not None, "Metadata must be present in JSON data"
        assert message is not None, "Message must be present in JSON data"
        assert step_logs is not None, "Logs must be present in JSON data"

        hash_alg = metadata.get("Hash Algorithm", None)
        byteorder = metadata.get("Byte order", None)
        hierarchy = metadata.get("Hierarchy", None)
        length = metadata.get("Input bits", None)
        assert hash_alg is not None, "Hash Algorithm must be specified in Metadata"
        assert byteorder is not None, "Byte order must be specified in Metadata"
        assert hierarchy is not None, "Hierarchy must be specified in Metadata"

        ret["Hash Algorithm"] = hash_alg
        ret["Byte order"] = byteorder
        ret["Hierarchy"] = hierarchy
        ret["Length"] = length
        ret["Message"] = message
        ret["Logs"] = step_logs

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
