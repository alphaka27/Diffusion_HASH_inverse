"""
Scheduling Approach
Approach 1: Rescaling Multiply SN with Beta and Rescaling to Original Beta Range
1. Multiply the mean SN values with the original betas to get raw candidate betas.
2. Rescale the raw candidate betas to the original beta range using min-max normalization.

Approach 2: Using Linear Equation to Map SN to Beta
1. Determine the linear relationship between the mean SN values and the original betas.
2. Use the linear equation to calculate candidate betas from the mean SN values.
"""

import mlx.core as mx
import numpy as np
from typing import List, Optional

from diffusion_hash_inv.logger import Logs


class BetaScheduler:
    def __init__(self, beta_min, beta_max, sn_array):
        self.sn_array = sn_array
        self.beta_min = beta_min
        self.beta_max = beta_max

        @staticmethod
        def get_step4(io_contoller, runtime_cfg):
            """
            Get Step 4 from Hash logs
            """
            logs = Logs.get_logs(io_contoller, runtime_cfg.hash, runtime_cfg.main)
            _step4_logs = []
            for _log in logs:
                _tmp = list(_log.values())
                assert len(_tmp) == 1, "Each log entry should contain exactly one key-value pair."
                log_dict = list(_log.values())[0]
                if "Logs" in log_dict and "4th Step" in log_dict["Logs"]:
                    _step4_logs.append(log_dict["Logs"]["4th Step"])
            return _step4_logs
