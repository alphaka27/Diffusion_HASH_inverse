"""
File Output Formatter for Diffusion HASH Inversion
Handles formatting of logs into various file formats such as XLSX, JSON, etc.
"""
from __future__ import annotations
from typing import Dict, Any
import json

from diffusion_hash_inv.logger import Metadata, BaseLogs, StepLogs


def bytes_to_hex_block(
    data: bytes,
    *,
    word_bytes: int = 2,
    line_bytes: int = 16,
    pad_last: bool = True,
) -> str:
    """
    Format bytes as grouped hexadecimal text.
    """
    out_lines: list[str] = []
    for i in range(0, len(data), line_bytes):
        line = data[i:i + line_bytes]
        groups: list[str] = []
        for j in range(0, len(line), word_bytes):
            chunk = line[j:j + word_bytes]
            hex_string = chunk.hex()
            if pad_last and len(chunk) < word_bytes:
                hex_string = hex_string.zfill(word_bytes * 2)
            groups.append(hex_string)
        out_lines.append(" ".join(groups))
    return "\n".join(out_lines)


def bytes_to_binary_block(data: bytes) -> str:
    """
    Format bytes as space-separated binary octets.
    """
    return " ".join(f"{byte:08b}" for byte in data)


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
