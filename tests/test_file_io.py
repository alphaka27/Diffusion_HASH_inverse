from pathlib import Path

from PIL import Image

from diffusion_hash_inv.config import MainConfig, OutputConfig
from diffusion_hash_inv.utils.file_io import FileIO


def _file_io(tmp_path: Path) -> FileIO:
    return FileIO(
        MainConfig(
            verbose_flag=False,
            clean_flag=False,
            debug_flag=False,
            make_image_flag=False,
        ),
        OutputConfig(root_dir=tmp_path),
    )


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def test_get_latest_files_by_date_uses_current_timestamp_dirs(tmp_path: Path) -> None:
    io = _file_io(tmp_path)
    json_root = io.out_dir / "json"

    _touch(json_root / "2026-04-25 20-36-06" / "MD5_16_2026-04-25 20-36-06_00000.json")
    _touch(json_root / "2026-04-25 20-36-06" / "MD5_16_2026-04-25 20-36-06_00001.json")
    _touch(json_root / "2026-04-25 20-40-00" / "MD5_16_2026-04-25 20-40-00_00000.json")
    _touch(json_root / "2026-04-25 20-40-00" / "SHA256_16_2026-04-25 20-40-00_00000.json")
    _touch(json_root / "2026-04-25 20-40-00" / "MD5_32_2026-04-25 20-40-00_00000.json")

    latest = io.get_latest_files_by_date("md5", 16)

    assert [path.name for path in latest] == ["MD5_16_2026-04-25 20-40-00_00000.json"]


def test_get_latest_files_by_date_accepts_legacy_colon_timestamp_dirs(tmp_path: Path) -> None:
    io = _file_io(tmp_path)
    json_root = io.out_dir / "json"

    _touch(json_root / "2026-04-25 20:36:06" / "MD5_16_2026-04-25 20-36-06_00000.json")
    _touch(json_root / "2026-04-25 20:40:00" / "MD5_16_2026-04-25 20-40-00_00000.json")

    latest = io.get_latest_files_by_date("md5", 16)

    assert [path.name for path in latest] == ["MD5_16_2026-04-25 20-40-00_00000.json"]


def test_get_latest_files_by_date_keeps_filename_timestamp_fallback(tmp_path: Path) -> None:
    io = _file_io(tmp_path)
    legacy_root = io.out_dir / "json" / "16"

    _touch(legacy_root / "MD5_16_2026-04-25 20-36-06_00000.json")
    _touch(legacy_root / "MD5_16_2026-04-25 20-40-00_00000.json")
    _touch(legacy_root / "MD5_16_2026-04-25 20-40-00_00001.json")

    latest = io.get_latest_files_by_date("MD5", 16, dir_path=legacy_root)

    assert [path.name for path in latest] == [
        "MD5_16_2026-04-25 20-40-00_00000.json",
        "MD5_16_2026-04-25 20-40-00_00001.json",
    ]


def test_select_dir_sanitizes_json_timestamp_dir(tmp_path: Path) -> None:
    io = _file_io(tmp_path)

    path = io.select_dir("json", path_infix="2026-04-27 02:12:47.123456+09:00/0")

    assert path.name == "2026-04-27 02-12-47"
    assert ":" not in path.name


def test_file_writer_sanitizes_parent_dir_path_components(tmp_path: Path) -> None:
    io = _file_io(tmp_path)
    image = Image.new("RGB", (1, 1))

    io.file_writer(
        "pixel.png",
        image,
        parent_dir=Path("2026-04-27 02:12:47") / "Step: 1",
        data_type="data",
    )

    written = io.data_dir / "images" / "2026-04-27 02-12-47" / "Step- 1" / "pixel.png"
    assert written.is_file()
