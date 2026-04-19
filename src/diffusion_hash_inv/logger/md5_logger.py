"""
MD5-specific logging utilities
Including data structures for trace payloads and decorators for logging steps and rounds.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, TypeVar, Any, cast
from functools import wraps

from diffusion_hash_inv.logger import StepLogs

@dataclass
class MD5RoundTrace:
    """Snapshot bundle for one MD5 message block round."""
    loop_start: Dict[str, int]
    loop_states: List[Dict[str, int]]
    loop_end: Dict[str, int]


@dataclass
class MD5Step4Trace:
    """Step4 return payload used by decorator-driven logging."""
    updated_hash: Dict[str, int]
    rounds: List[MD5RoundTrace]


F = TypeVar("F", bound=Callable[..., Any])


class MD5Logger:
    """
    Centralized logging decorators.
    """

    @staticmethod
    def step(step_index: int, update_overflow: bool = False) -> Callable[[F], F]:
        """
        Log a step return value into `self.logs`.
        """
        assert step_index > 0, "step_index must be positive"
        if step_index == 4:
            def deco_step4(fn: F) -> F:
                @wraps(fn)
                def wrapper(self, *args, **kwargs):
                    trace: MD5Step4Trace = fn(self, *args, **kwargs)
                    logs = getattr(self, "logs", None)
                    if isinstance(logs, StepLogs):
                        step_key = logs.index_label(step_index, "Step")
                        for round_idx, round_trace in enumerate(trace.rounds, start=1):
                            round_key = logs.index_label(round_idx, "Round")
                            round_payload: Dict[str, Any] = {"Loop Start": round_trace.loop_start}
                            for loop_idx, state in enumerate(round_trace.loop_states, start=1):
                                loop_key = logs.index_label(loop_idx, "Loop")
                                round_payload[loop_key] = state
                            round_payload["Loop End"] = round_trace.loop_end
                            logs.set_value((step_key, round_key), round_payload)
                    return trace.updated_hash

                return cast(F, wrapper)

            return deco_step4

        def deco(fn: F) -> F:
            @wraps(fn)
            def wrapper(self, *args, **kwargs):
                result = fn(self, *args, **kwargs)
                logs = getattr(self, "logs", None)
                if isinstance(logs, StepLogs):
                    logs.update(step_index=step_index, step_result=result)
                    if update_overflow:
                        logs.update_overflow(getattr(self, "total_overflow_count", 0))
                return result

            return cast(F, wrapper)

        return deco
