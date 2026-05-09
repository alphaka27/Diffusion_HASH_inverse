import json
from pathlib import Path

import numpy as np

from diffusion_hash_inv.analyze.analyze import (
    Analyze,
    iter_beta_schedules,
    iter_step_logs,
    make_beta_schedule,
    summarize_beta_schedules,
)


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


def test_make_beta_schedule_matches_notebook_cumulative_logic() -> None:
    schedule = make_beta_schedule(_step4_log())

    assert schedule == [1, 3, 6, 10]


def test_iter_beta_schedules_accepts_wrapped_and_raw_logs() -> None:
    logs = [
        {"first": _raw_log(0)},
        _raw_log(1),
    ]

    schedules = list(iter_beta_schedules(logs))

    assert schedules == [
        [1, 3, 6, 10],
        [2, 5, 9, 14],
    ]


def test_summarize_beta_schedules_streams_numpy_equivalent_stats() -> None:
    schedules = [
        [1, 3, 6, 10],
        [2, 5, 9, 14],
        [3, 7, 12, 18],
    ]

    summary = summarize_beta_schedules(iter(schedules))
    arr = np.asarray(schedules, dtype=np.float64)

    assert summary.count == 3
    assert summary.length == 4
    np.testing.assert_array_equal(summary.first, arr[0])
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
    step_logs = list(iter_step_logs(analyzer.iter_logs()))

    assert step_logs == [_step4_log()]
    assert analyzer.data == []
