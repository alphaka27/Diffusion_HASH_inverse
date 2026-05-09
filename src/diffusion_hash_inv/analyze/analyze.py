"""
Streaming helpers for analyzing hash intermediate process logs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

from diffusion_hash_inv.config import OutputConfig
from diffusion_hash_inv.logger import Logs


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class BetaScheduleSummary:
    """
    Online aggregate statistics for beta schedules.
    """

    count: int
    length: int
    minimum: np.ndarray
    maximum: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    std: np.ndarray
    first: np.ndarray


def _unwrap_log(log_dict: JsonDict) -> JsonDict:
    """
    Accept either a raw log or {filename: log} and return the actual log body.
    """
    if "Logs" in log_dict:
        return log_dict

    if len(log_dict) != 1:
        raise ValueError("Each wrapped log must contain exactly one file key.")

    full_log = next(iter(log_dict.values()))
    if not isinstance(full_log, dict) or "Logs" not in full_log:
        raise ValueError("Log body must be a dictionary containing a 'Logs' field.")

    return full_log


def cumulative_block(values: List[int], block: bytes) -> None:
    """
    Append byte-wise cumulative values to an existing schedule.
    """
    running = values[-1] if values else 0
    for byte_value in block:
        running += byte_value
        values.append(running)


def make_beta_schedule(step4_logs: JsonDict) -> List[int]:
    """
    Build one cumulative byte schedule from the MD5 4th Step log.
    """
    beta_schedule: List[int] = []

    for round_log in step4_logs.values():
        if not isinstance(round_log, dict):
            raise ValueError(f"Round log must be a dictionary, got {type(round_log)}")

        for state in round_log.values():
            if not isinstance(state, dict):
                raise ValueError(f"Loop state must be a dictionary, got {type(state)}")

            for value in state.values():
                if not isinstance(value, str):
                    raise ValueError(f"Loop state value must be a hex string, got {type(value)}")
                cumulative_block(beta_schedule, Logs.str_to_bytes(value))

    return beta_schedule


def iter_step_logs(logs: Iterable[JsonDict], step_name: str = "4th Step") -> Iterator[JsonDict]:
    """
    Yield one named step log at a time from a raw or wrapped log stream.
    """
    for log_dict in logs:
        full_log = _unwrap_log(log_dict)
        step_logs = full_log.get("Logs")
        if not isinstance(step_logs, dict):
            raise ValueError("Log body must contain a dictionary 'Logs' field.")

        step_log = step_logs.get(step_name)
        if step_log is None:
            continue
        if not isinstance(step_log, dict):
            raise ValueError(f"{step_name} must be a dictionary.")

        yield step_log


def iter_beta_schedules(
    logs: Iterable[JsonDict],
    step_name: str = "4th Step",
) -> Iterator[List[int]]:
    """
    Yield beta schedules without retaining all logs or schedules in memory.
    """
    for step_log in iter_step_logs(logs, step_name=step_name):
        yield make_beta_schedule(step_log)


def summarize_beta_schedules(
    schedules: Iterable[Sequence[int]],
    *,
    dtype: np.dtype | type = np.float64,
) -> BetaScheduleSummary:
    """
    Compute min, max, mean, variance, and std in one pass.

    The variance matches numpy's default population variance (ddof=0), which is
    what the notebook previously used after materializing the full array.
    """
    count = 0
    length = 0
    first: Optional[np.ndarray] = None
    minimum: Optional[np.ndarray] = None
    maximum: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None

    for schedule in schedules:
        arr = np.asarray(schedule, dtype=dtype)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("Each beta schedule must be a non-empty one-dimensional sequence.")

        if count == 0:
            count = 1
            length = int(arr.size)
            first = arr.copy()
            minimum = arr.copy()
            maximum = arr.copy()
            mean = arr.copy()
            m2 = np.zeros_like(arr, dtype=dtype)
            continue

        if arr.size != length:
            raise ValueError(f"Schedule length mismatch: expected {length}, got {arr.size}.")

        count += 1
        assert minimum is not None
        assert maximum is not None
        assert mean is not None
        assert m2 is not None

        minimum = np.minimum(minimum, arr)
        maximum = np.maximum(maximum, arr)
        delta = arr - mean
        mean = mean + delta / count
        delta2 = arr - mean
        m2 = m2 + delta * delta2

    if count == 0:
        raise ValueError("No beta schedules were provided.")

    assert first is not None
    assert minimum is not None
    assert maximum is not None
    assert mean is not None
    assert m2 is not None

    variance = m2 / count
    return BetaScheduleSummary(
        count=count,
        length=length,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        variance=variance,
        std=np.sqrt(variance),
        first=first,
    )


class Analyze:
    """
    SHA256/MD5 intermediate-process analyzer.

    By default this class streams JSON files. Use load() or preload=True only
    when the full dataset is intentionally small enough to keep in memory.
    """

    def __init__(self, data_path: str | Path, is_verbose: bool = False, preload: bool = False):
        self.data_path = Path(data_path)
        self.is_verbose = is_verbose
        self.data: List[JsonDict] = []

        if preload:
            self.load()

    def iter_files(self) -> Iterator[Path]:
        """
        Yield JSON files below data_path in deterministic order.
        """
        if self.data_path.is_file():
            if self.data_path.suffix == ".json":
                yield self.data_path
            return

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        if not self.data_path.is_dir():
            raise NotADirectoryError(f"Data path is not a directory: {self.data_path}")

        yield from sorted(path for path in self.data_path.rglob("*.json") if path.is_file())

    def iter_logs(self) -> Iterator[JsonDict]:
        """
        Stream raw JSON logs one file at a time.
        """
        for log_path in self.iter_files():
            if self.is_verbose:
                print(f"Loading log: {log_path}")
            with log_path.open("r", encoding="utf-8") as rf:
                yield json.load(rf)

    def load(self) -> List[JsonDict]:
        """
        Load all logs into memory for small datasets.
        """
        self.data = list(self.iter_logs())
        if self.is_verbose:
            print(f"Loaded logs: {len(self.data)}")
            if self.data:
                print(self.data[0])
        return self.data

    def get(self, name: Optional[str] = None, iteration: int = 0) -> Any:
        """
        Compatibility accessor for preloaded data.
        """
        if not self.data:
            self.load()
        if name is None:
            return self.data
        return self.data[iteration][name]

    def avalanche(self) -> None:
        """
        Avalanche rate analysis placeholder.
        """

    def haming_distnace(self) -> None:
        """
        Hamming distance analysis placeholder.
        """

    def subtract(self) -> None:
        """
        Step subtraction analysis placeholder.
        """


def main(file_path: str | Path, is_verbose: bool) -> None:
    """
    Stream logs and report how many JSON records are available.
    """
    analyzer = Analyze(file_path, is_verbose)
    count = sum(1 for _ in analyzer.iter_logs())
    print(f"Found {count} JSON log files.")


if __name__ == "__main__":
    project_root = OutputConfig.get_project_root()
    _data_path = project_root / "output" / "json"

    parser = argparse.ArgumentParser(description="Hash intermediate-process analyzer")
    parser.add_argument("-f", "--file_path", type=str, default=str(_data_path),
                        help="Set output JSON directory or file")

    gv = parser.add_mutually_exclusive_group()
    gv.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                    help="Enable verbose output")
    gv.add_argument("-q", "--quiet", action="store_false", dest="verbose",
                    help="Suppress output")
    parser.set_defaults(verbose=False)

    _args = parser.parse_args()
    main(_args.file_path, _args.verbose)
