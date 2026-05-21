import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from diffusion_hash_inv.config import (
    Byte2RGBConfig,
    HashConfig,
    MainConfig,
    MessageConfig,
    OutputConfig,
)
from diffusion_hash_inv.main import RuntimeConfig
from diffusion_hash_inv.utils import (
    FileIO,
    HDF5Maker,
    HDF5TensorDataset,
    RGBImgMaker,
    create_hdf5_tensor_dataloader,
)


def _runtime(tmp_path: Path) -> tuple[RuntimeConfig, FileIO, Byte2RGBConfig]:
    main_cfg = MainConfig(
        verbose_flag=False,
        clean_flag=False,
        debug_flag=False,
        make_image_flag=False,
    )
    hash_cfg = HashConfig(hash_alg="md5", length=32)
    message_cfg = MessageConfig(
        message_flag=False,
        length=32,
        random_flag=True,
        seed_flag=True,
    )
    output_cfg = OutputConfig(root_dir=tmp_path)
    rgb_cfg = Byte2RGBConfig(seed_flag=False, input_seed=42)
    runtime_cfg = RuntimeConfig(
        main=main_cfg,
        message=message_cfg,
        hash=hash_cfg,
        output=output_cfg,
        rgb=rgb_cfg,
    )
    return runtime_cfg, FileIO(main_cfg, output_cfg), rgb_cfg


def _write_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "Hierarchy": ["Step"],
                "Metadata": {
                    "Hash Algorithm": "MD5",
                    "Hierarchy": ["Step"],
                    "Input bits": 32,
                    "Message mode": "Bit string",
                    "Program started at": "2026-05-14 00:00:00+09:00",
                    "Program elapsed time": "0 ns",
                    "Byte order": "little",
                },
                "Message": {"Hex": "0x00000000"},
                "Generated hash": "0x00000000",
                "Correct   hash": "0x00000000",
                "Logs": {"Only Step": "0x01020304"},
                "Step Metadata": {
                    "word_size": 32,
                    "byteorder": "little",
                    "hierarchy": ["Step"],
                    "overflow_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_rgb_image_maker_accepts_explicit_log_paths(tmp_path: Path) -> None:
    runtime_cfg, io_controller, rgb_cfg = _runtime(tmp_path)
    older_log = (
        io_controller.out_dir
        / "json"
        / "2026-05-14 00-00-00"
        / "MD5_32_2026-05-14 00-00-00_0.json"
    )
    latest_log = (
        io_controller.out_dir
        / "json"
        / "2026-05-14 00-01-00"
        / "MD5_32_2026-05-14 00-01-00_0.json"
    )
    _write_log(older_log)
    _write_log(latest_log)

    maker = RGBImgMaker(runtime_cfg, io_controller, rgb_cfg)
    images_written = maker.main(logs=[older_log])

    older_image_dir = io_controller.data_dir / "images" / older_log.stem
    latest_image_dir = io_controller.data_dir / "images" / latest_log.stem
    assert images_written == 2
    assert (older_image_dir / "message.png").is_file()
    assert (older_image_dir / "Only Step.png").is_file()
    assert not latest_image_dir.exists()


def test_rgb_image_maker_validates_saved_png_by_decoding(tmp_path: Path) -> None:
    runtime_cfg, io_controller, rgb_cfg = _runtime(tmp_path)
    log_path = (
        io_controller.out_dir
        / "json"
        / "2026-05-14 00-00-00"
        / "MD5_32_2026-05-14 00-00-00_0.json"
    )
    _write_log(log_path)

    maker = RGBImgMaker(runtime_cfg, io_controller, rgb_cfg)
    maker.main(logs=[log_path])

    message_path = io_controller.data_dir / "images" / log_path.stem / "message.png"
    maker._validate_saved_png(message_path, "0x00000000")

    Image.new("RGB", (112, 28), color=(255, 255, 255)).save(message_path)
    with pytest.raises(RuntimeError, match="Saved PNG validation failed"):
        maker._validate_saved_png(message_path, "0x00000000")


def test_rgb_image_maker_writes_png_in_parallel(tmp_path: Path) -> None:
    runtime_cfg, io_controller, rgb_cfg = _runtime(tmp_path)
    log_paths = []
    for idx in range(2):
        log_path = (
            io_controller.out_dir
            / "json"
            / f"2026-05-14 00-00-0{idx}"
            / f"MD5_32_2026-05-14 00-00-0{idx}_0.json"
        )
        _write_log(log_path)
        log_paths.append(log_path)

    maker = RGBImgMaker(runtime_cfg, io_controller, rgb_cfg)
    images_written = maker.main(
        logs=log_paths,
        workers=2,
        parallel_chunk_size=1,
    )

    assert images_written == 4
    for log_path in log_paths:
        png_dir = io_controller.data_dir / "images" / log_path.stem
        assert (png_dir / "message.png").is_file()
        assert (png_dir / "Only Step.png").is_file()


def test_hdf5_maker_writes_tensor_shards_in_parallel(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    runtime_cfg, io_controller, rgb_cfg = _runtime(tmp_path)
    log_paths = []
    for idx in range(2):
        log_path = (
            io_controller.out_dir
            / "json"
            / f"2026-05-14 00-00-0{idx}"
            / f"MD5_32_2026-05-14 00-00-0{idx}_0.json"
        )
        _write_log(log_path)
        log_paths.append(log_path)

    maker = HDF5Maker(runtime_cfg, io_controller)
    shard_paths = maker.main(
        logs=log_paths,
        rgb_config=rgb_cfg,
        workers=2,
        shard_size=1,
        output_name="test_tensors",
        include_paths=("message.png",),
        compression=None,
    )

    dataset_dir = io_controller.data_dir / "tensor_datasets" / "test_tensors"
    assert len(shard_paths) == 2
    assert (dataset_dir / "manifest.json").is_file()

    with h5py.File(shard_paths[0], "r") as h5_file:
        assert h5_file.attrs["log_count"] == 1
        assert h5_file.attrs["tensor_count"] == 1
        tensor = h5_file["records"]["00000000"]["tensor"]
        assert tensor.shape == (3, 28, 112)
        assert tensor.dtype == np.dtype("float32")
        path_value = h5_file["paths"][0]
        if isinstance(path_value, bytes):
            path_value = path_value.decode("utf-8")
        assert path_value == "message.png"


def test_hdf5_maker_writes_all_image_tensors_by_default(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    runtime_cfg, io_controller, rgb_cfg = _runtime(tmp_path)
    log_path = (
        io_controller.out_dir
        / "json"
        / "2026-05-14 00-00-00"
        / "MD5_32_2026-05-14 00-00-00_0.json"
    )
    _write_log(log_path)

    maker = HDF5Maker(runtime_cfg, io_controller)
    shard_paths = maker.main(
        logs=[log_path],
        rgb_config=rgb_cfg,
        workers=1,
        shard_size=1,
        output_name="all_tensors",
        compression=None,
    )

    with h5py.File(shard_paths[0], "r") as h5_file:
        assert h5_file.attrs["tensor_count"] == 2
        paths = []
        for value in h5_file["paths"]:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            paths.append(value)
        assert paths == ["message.png", "Only Step.png"]


def test_hdf5_maker_preserves_log_hierarchy(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    runtime_cfg, io_controller, rgb_cfg = _runtime(tmp_path)
    log_path = (
        io_controller.out_dir
        / "json"
        / "2026-05-14 00-00-00"
        / "MD5_32_2026-05-14 00-00-00_0.json"
    )
    _write_log(log_path)

    maker = HDF5Maker(runtime_cfg, io_controller)
    shard_paths = maker.main(
        logs=[log_path],
        rgb_config=rgb_cfg,
        workers=1,
        shard_size=1,
        output_name="hierarchy_tensors",
        compression=None,
    )

    with h5py.File(shard_paths[0], "r") as h5_file:
        source_group = h5_file["logs"][log_path.stem]
        message_tensor = source_group["Message"]["Hex"]["tensor"]
        log_tensor = source_group["Logs"]["Only Step"]["tensor"]
        assert message_tensor.shape == (3, 28, 112)
        assert log_tensor.shape == (3, 28, 112)
        assert source_group["Message"]["Hex"].attrs["path"] == "message.png"
        assert source_group["Logs"]["Only Step"].attrs["path"] == "Only Step.png"
        assert message_tensor.id == h5_file["records"]["00000000"]["tensor"].id
        assert log_tensor.id == h5_file["records"]["00000001"]["tensor"].id


def test_hdf5_tensor_dataset_reads_shards(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    runtime_cfg, io_controller, rgb_cfg = _runtime(tmp_path)
    log_paths = []
    for idx in range(2):
        log_path = (
            io_controller.out_dir
            / "json"
            / f"2026-05-14 00-00-0{idx}"
            / f"MD5_32_2026-05-14 00-00-0{idx}_0.json"
        )
        _write_log(log_path)
        log_paths.append(log_path)

    maker = HDF5Maker(runtime_cfg, io_controller)
    maker.main(
        logs=log_paths,
        rgb_config=rgb_cfg,
        workers=2,
        shard_size=1,
        output_name="loader_tensors",
        include_paths=("message.png",),
        compression=None,
    )

    dataset_dir = io_controller.data_dir / "tensor_datasets" / "loader_tensors"
    dataset = HDF5TensorDataset(
        dataset_dir,
        include_paths=("message.png",),
        require_same_shape=True,
    )
    try:
        assert len(dataset) == 2
        assert dataset.tensor_shape == (3, 28, 112)
        tensor, index = dataset[0]
        assert tensor.shape == (3, 28, 112)
        assert tensor.dtype == torch.float32
        assert int(index) == 0
        assert dataset.metadata(0)["path"] == "message.png"
    finally:
        dataset.close()


def test_hdf5_tensor_dataloader_batches_shards(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    runtime_cfg, io_controller, rgb_cfg = _runtime(tmp_path)
    log_paths = []
    for idx in range(2):
        log_path = (
            io_controller.out_dir
            / "json"
            / f"2026-05-14 00-01-0{idx}"
            / f"MD5_32_2026-05-14 00-01-0{idx}_0.json"
        )
        _write_log(log_path)
        log_paths.append(log_path)

    maker = HDF5Maker(runtime_cfg, io_controller)
    maker.main(
        logs=log_paths,
        rgb_config=rgb_cfg,
        workers=2,
        shard_size=1,
        output_name="dataloader_tensors",
        include_paths=("message.png",),
        compression=None,
    )

    dataset_dir = io_controller.data_dir / "tensor_datasets" / "dataloader_tensors"
    dataloader = create_hdf5_tensor_dataloader(
        dataset_dir,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        include_paths=("message.png",),
    )
    batch, indices = next(iter(dataloader))
    assert batch.shape == (2, 3, 28, 112)
    assert indices.tolist() == [0, 1]
