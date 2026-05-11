import json
from pathlib import Path

import numpy as np

from diffusion_hash_inv.analyze.analyze import (
    Analyze,
    BetaScheduleAnalyzer,
)
from diffusion_hash_inv.logger import Logs, StepLogParser


def _step4_log(offset: int = 0) -> dict:
    return {
        "1st Round": {
            "Loop Start": {
                "A": f"0x{1 + offset:02x}",
                "B": f"0x{2 + offset:02x}",
            },
            "1st Loop": {
                "A": f"0x{3 + offset:02x}",
                "B": f"0x{4 + offset:02x}",
            },
        },
    }


def _raw_log(offset: int = 0) -> dict:
    return {
        "Message": {"Hex": "0x00"},
        "Logs": {
            "4th Step": _step4_log(offset),
        },
    }


def test_append_cumulative_block_matches_notebook_cumulative_logic() -> None:
    schedule = []
    BetaScheduleAnalyzer.append_cumulative_block(schedule, b"\x01\x02")
    BetaScheduleAnalyzer.append_cumulative_block(schedule, b"\x03\x04")

    assert schedule == [1, 3, 6, 10]


def test_beta_schedule_analyzer_class_matches_notebook_cumulative_logic() -> None:
    schedule = BetaScheduleAnalyzer.make_schedule(_step4_log())

    assert schedule == [1, 3, 6, 10]


def test_iter_beta_schedules_accepts_wrapped_and_raw_logs() -> None:
    logs = [
        {"first": _raw_log(0)},
        _raw_log(1),
    ]

    schedules = list(BetaScheduleAnalyzer().iter_schedules(logs))

    assert schedules == [
        [1, 3, 6, 10],
        [2, 5, 9, 14],
    ]


def test_step_log_parser_class_accepts_wrapped_and_raw_logs() -> None:
    logs = [
        {"first": _raw_log(0)},
        _raw_log(1),
    ]

    step_logs = list(StepLogParser.iter_step_logs(logs))

    assert step_logs == [_step4_log(0), _step4_log(1)]


def test_logs_streaming_helpers_accept_nested_log_files(tmp_path: Path) -> None:
    log_dir = tmp_path / "output" / "json" / "2026-05-09 12-00-00"
    log_dir.mkdir(parents=True)
    (log_dir / "MD5_16_2026-05-09 12-00-00_0.json").write_text(
        json.dumps(_raw_log()),
        encoding="utf-8",
    )

    logs = list(Logs.iter_logs_from_path(tmp_path / "output" / "json"))
    step_logs = list(Logs.iter_step_logs(logs))
    leaf_values = list(Logs.iter_leaf_values(step_logs[0]))

    assert len(logs) == 1
    assert step_logs == [_step4_log()]
    assert leaf_values == ["0x01", "0x02", "0x03", "0x04"]


def test_summarize_beta_schedules_streams_numpy_equivalent_stats() -> None:
    schedules = [
        [1, 3, 6, 10],
        [2, 5, 9, 14],
        [3, 7, 12, 18],
    ]

    summary = BetaScheduleAnalyzer.summarize(iter(schedules))
    arr = np.asarray(schedules, dtype=np.float64)

    assert summary.count == 3
    assert summary.length == 4
    np.testing.assert_array_equal(summary.minimum, np.min(arr, axis=0))
    np.testing.assert_array_equal(summary.maximum, np.max(arr, axis=0))
    np.testing.assert_allclose(summary.mean, np.mean(arr, axis=0))
    np.testing.assert_allclose(summary.variance, np.var(arr, axis=0))
    np.testing.assert_allclose(summary.std, np.std(arr, axis=0))


def test_analyze_streams_nested_json_files_without_preloading(tmp_path: Path) -> None:
    log_dir = tmp_path / "output" / "json" / "2026-05-09 12-00-00"
    log_dir.mkdir(parents=True)
    (log_dir / "MD5_16_2026-05-09 12-00-00_0.json").write_text(
        json.dumps(_raw_log()),
        encoding="utf-8",
    )

    analyzer = Analyze(tmp_path / "output" / "json")

    assert analyzer.data == []
    step_logs = list(analyzer.iter_step_logs())

    assert step_logs == [_step4_log()]
    assert analyzer.data == []


def test_analyze_class_summarizes_streamed_beta_schedules(tmp_path: Path) -> None:
    log_dir = tmp_path / "output" / "json" / "2026-05-09 12-00-00"
    log_dir.mkdir(parents=True)
    for idx, offset in enumerate((0, 1, 2)):
        (log_dir / f"MD5_16_2026-05-09 12-00-00_{idx}.json").write_text(
            json.dumps(_raw_log(offset)),
            encoding="utf-8",
        )

    analyzer = Analyze(tmp_path / "output" / "json")
    summary = analyzer.summarize_beta_schedules()

    assert summary.count == 3
    assert summary.length == 4
    np.testing.assert_allclose(summary.mean, [2, 5, 9, 14])
    assert analyzer.data == []
