"""
Class-based streaming analyzers for hash intermediate process logs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

from diffusion_hash_inv.config import OutputConfig
from diffusion_hash_inv.logger import LogStream, Logs, StepLogParser


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


class BetaScheduleAnalyzer:
    """
    Build and summarize beta schedules from step logs.
    """

    def __init__(self, step_name: str = "4th Step") -> None:
        self.step_name = step_name

    @staticmethod
    def append_cumulative_block(values: List[int], block: bytes) -> None:
        """
        Append byte-wise cumulative values to an existing schedule.
        """
        running = values[-1] if values else 0
        for byte_value in block:
            running += byte_value
            values.append(running)

    @classmethod
    def make_schedule(cls, step_logs: JsonDict) -> List[int]:
        """
        Build one cumulative byte schedule from a step log.
        """
        beta_schedule: List[int] = []

        for value in Logs.iter_leaf_values(step_logs):
            if not isinstance(value, str):
                raise ValueError(f"Step log leaf value must be a hex string, got {type(value)}")
            cls.append_cumulative_block(beta_schedule, Logs.str_to_bytes(value))

        return beta_schedule

    def iter_schedules(self, logs: Iterable[JsonDict]) -> Iterator[List[int]]:
        """
        Yield schedules without retaining all logs or schedules in memory.
        """
        for step_log in Logs.iter_step_logs(logs, step_name=self.step_name):
            yield self.make_schedule(step_log)

    @staticmethod
    def summarize(
        schedules: Iterable[Sequence[int]],
        *,
        dtype: np.dtype | type = np.float64,
    ) -> BetaScheduleSummary:
        """
        Compute min, max, mean, variance, and std in one pass.

        The variance matches numpy's default population variance (ddof=0), which
        is what the notebook previously used after materializing the full array.
        """
        count = 0
        length = 0
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
        )


class Analyze:
    """
    SHA256/MD5 intermediate-process analyzer.

    By default this class streams JSON files. Use load() or preload=True only
    when the full dataset is intentionally small enough to keep in memory.
    """

    def __init__(
        self,
        data_path: str | Path,
        is_verbose: bool = False,
        preload: bool = False,
        step_name: str = "4th Step",
    ) -> None:
        self.stream = LogStream(data_path, is_verbose=is_verbose)
        self.beta_schedule = BetaScheduleAnalyzer(step_name=step_name)
        self.data: List[JsonDict] = []

        if preload:
            self.load()

    @property
    def data_path(self) -> Path:
        """
        Return the configured log path.
        """
        return self.stream.data_path

    @property
    def is_verbose(self) -> bool:
        """
        Return whether verbose streaming is enabled.
        """
        return self.stream.is_verbose

    def iter_files(self) -> Iterator[Path]:
        """
        Yield JSON files below data_path in deterministic order.
        """
        return self.stream.iter_files()

    def iter_logs(self) -> Iterator[JsonDict]:
        """
        Stream raw JSON logs one file at a time.
        """
        return self.stream.iter_logs()

    def iter_step_logs(self, step_name: Optional[str] = None) -> Iterator[JsonDict]:
        """
        Stream selected step logs from this analyzer's input logs.
        """
        selected_step = step_name if step_name is not None else self.beta_schedule.step_name
        return StepLogParser.iter_step_logs(self.iter_logs(), step_name=selected_step)

    def iter_beta_schedules(self, step_name: Optional[str] = None) -> Iterator[List[int]]:
        """
        Stream beta schedules from this analyzer's input logs.
        """
        selected_step = step_name if step_name is not None else self.beta_schedule.step_name
        analyzer = self.beta_schedule if selected_step == self.beta_schedule.step_name \
            else BetaScheduleAnalyzer(step_name=selected_step)
        return analyzer.iter_schedules(self.iter_logs())

    def summarize_beta_schedules(
        self,
        step_name: Optional[str] = None,
        *,
        dtype: np.dtype | type = np.float64,
    ) -> BetaScheduleSummary:
        """
        Stream schedules and return online aggregate statistics.
        """
        return BetaScheduleAnalyzer.summarize(
            self.iter_beta_schedules(step_name=step_name),
            dtype=dtype,
        )

    def load(self) -> List[JsonDict]:
        """
        Load all logs into memory for small datasets.
        """
        self.data = self.stream.load()
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
