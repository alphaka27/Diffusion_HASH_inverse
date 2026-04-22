import os
from pathlib import Path

from diffusion_hash_inv.config import OutputConfig


def test_get_project_root_falls_back_when_cwd_was_deleted(tmp_path) -> None:
    original_cwd = Path.cwd()
    deleted_cwd = tmp_path / "deleted-cwd"
    deleted_cwd.mkdir()

    os.chdir(deleted_cwd)
    deleted_cwd.rmdir()
    try:
        project_root = OutputConfig.get_project_root()
    finally:
        os.chdir(original_cwd)

    assert project_root == Path(__file__).resolve().parents[1]
