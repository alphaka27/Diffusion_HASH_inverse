"""
File Output Formatter for Diffusion HASH Inversion
Handles formatting of logs into various file formats such as XLSX, JSON, etc.
"""
from __future__ import annotations
import json

from diffusion_hash_inv.common import Metadata, BaseLogs, StepLogs

class JSONFormat:
    """
    Class to handle output formatting for hash results.
    """
    def loads(self, json_string: str):
        """Make JSON load"""
        json_string = json.loads(json_string)
        metadata = json_string.get("Metadata", None)
        assert metadata is not None, "Metadata not found in JSON."
        baselogs = {}
        for key in BaseLogs.keys():
            baselogs[key] = json_string.get(key, None)
            assert baselogs[key] is not None, f"Key {key} not found in JSON."
        steplogs = json_string.get("Step Logs", None)
        assert steplogs is not None, "Step Logs not found in JSON."
        return Metadata(**metadata), BaseLogs(**baselogs), StepLogs(**steplogs)

    @staticmethod
    def dumps(indent=4, **data):
        """Make JSON dump"""
        metadat = data.get("metadata", None)
        baselogs = data.get("baselogs", None)
        steplogs = data.get("steplogs", None)
        assert isinstance(metadat, Metadata), "metadata must be an instance of Metadata"
        assert isinstance(baselogs, BaseLogs), "baselogs must be an instance of BaseLogs"
        assert isinstance(steplogs, StepLogs), \
            f"steplogs must be an instance of StepLogs, {type(steplogs)} given"
        _all_data = {"Metadata": metadat.getter()}
        _all_data.update(dict(baselogs.getter().items()))
        _all_data.update({"Step Logs": steplogs.getter()})

        return json.dumps(_all_data, ensure_ascii=False, indent=indent)

class XLSXFormat:
    """
    Class to handle XLSX formatting for hash results.
    """
    pass
