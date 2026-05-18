from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from diffusion_hash_inv.models.diffusion_with_mlx import build_arg_parser, run_demo


def test_mlx_toy_can_save_forward_and_reverse_process_traces(tmp_path: Path) -> None:
    output_path = tmp_path / "samples.png"
    trace_dir = tmp_path / "process_traces"
    args = build_arg_parser().parse_args(
        [
            "--device",
            "cpu",
            "--seed",
            "0",
            "--image-size",
            "8",
            "--num-classes",
            "2",
            "--timesteps",
            "3",
            "--time-dim",
            "8",
            "--hidden-dim",
            "16",
            "--batch-size",
            "2",
            "--train-steps",
            "1",
            "--samples",
            "2",
            "--columns",
            "2",
            "--output",
            str(output_path),
            "--save-process-traces",
            "--trace-sample-count",
            "2",
            "--process-traces-dir",
            str(trace_dir),
        ]
    )

    result = run_demo(args)

    assert result == output_path
    assert output_path.is_file()
    assert (trace_dir / "forward" / "png" / "x0.png").is_file()
    assert (trace_dir / "forward" / "png" / "t_000000.png").is_file()
    assert (trace_dir / "forward" / "png" / "t_000001.png").is_file()
    assert (trace_dir / "forward" / "png" / "t_000002.png").is_file()
    assert (trace_dir / "forward" / "json" / "x0.labels.json").is_file()
    assert (trace_dir / "forward" / "json" / "t_000000.labels.json").is_file()
    assert (trace_dir / "forward" / "json" / "t_000001.labels.json").is_file()
    assert (trace_dir / "forward" / "json" / "t_000002.labels.json").is_file()
    assert (trace_dir / "reverse" / "png" / "xT_noise.png").is_file()
    assert (trace_dir / "reverse" / "png" / "t_000002.png").is_file()
    assert (trace_dir / "reverse" / "png" / "t_000001.png").is_file()
    assert (trace_dir / "reverse" / "png" / "t_000000.png").is_file()
    assert (trace_dir / "reverse" / "json" / "xT_noise.labels.json").is_file()
    assert (trace_dir / "reverse" / "json" / "t_000002.labels.json").is_file()
    assert (trace_dir / "reverse" / "json" / "t_000001.labels.json").is_file()
    assert (trace_dir / "reverse" / "json" / "t_000000.labels.json").is_file()
    assert len(list((trace_dir / "forward" / "png").glob("t_*.png"))) == 3
    assert len(list((trace_dir / "forward" / "json").glob("t_*.labels.json"))) == 3
    assert len(list((trace_dir / "reverse" / "png").glob("t_*.png"))) == 3
    assert len(list((trace_dir / "reverse" / "json").glob("t_*.labels.json"))) == 3
