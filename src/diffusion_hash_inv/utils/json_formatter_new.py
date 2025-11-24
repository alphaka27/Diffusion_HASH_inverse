"""
JSON-safe formatter
"""
from __future__ import annotations
import json

class JSONFormat:
    """
    Class to handle output formatting for hash results.
    """
    def loads(self, json_string: str):
        """Make JSON load"""
        return json_string

    def dumps(self, data, indent=4):
        """Make JSON dump"""
        return json.dumps(data, ensure_ascii=False, indent=indent)
