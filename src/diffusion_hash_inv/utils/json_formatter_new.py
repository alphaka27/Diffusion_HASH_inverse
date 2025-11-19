"""
JSON-safe formatter
"""
from __future__ import annotations
import json
import numpy as np

class JSONFormat:
    """
    Class to handle output formatting for hash results.
    """
    def __init__(self):
        pass

    @staticmethod
    def json_safe(o):
        """JSON-safe """
        if isinstance(o, np.ndarray):
            return [JSONFormat.json_safe(v) for v in o.tolist()]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, (list, tuple)):
            return [JSONFormat.json_safe(v) for v in o]
        if isinstance(o, dict):
            return {k: JSONFormat.json_safe(v) for k, v in o.items()}
        return o

    def dumps(self, indent=4, data=None):
        """Make JSON dump"""
        ret = JSONFormat.json_safe(self.to_dict()) \
            if data is None else JSONFormat.json_safe(data)
        return json.dumps(ret, ensure_ascii=False, indent=indent)
