"""
PyTorch Dataset/DataLoader helpers for sharded HDF5 tensor datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def _import_h5py():
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to read HDF5 tensor datasets. "
            "Install it with `pip install h5py` or `pip install -e '.[hdf5]'`."
        ) from exc
    return h5py


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


@dataclass(frozen=True)
class HDF5TensorRecord:
    """One tensor record inside an HDF5 shard."""

    shard_path: Path
    record_key: str
    source_log: str
    path: str
    shape: tuple[int, ...]


class HDF5TensorDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    Dataset for HDF5 tensor shards written by ``HDF5Maker``.

    The dataset keeps no open HDF5 handles when pickled. Each DataLoader worker
    lazily opens the shard files it needs, so ``num_workers > 0`` can be used.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        include_paths: Optional[Sequence[str]] = None,
        max_items: int | None = None,
        require_same_shape: bool = False,
        as_float32: bool = True,
        return_metadata: bool = False,
    ) -> None:
        self.root = Path(root)
        self.manifest_path = self._resolve_manifest_path(self.root)
        self.dataset_dir = self.manifest_path.parent
        self.include_paths = tuple(include_paths) if include_paths is not None else None
        self.max_items = max_items
        self.require_same_shape = require_same_shape
        self.as_float32 = as_float32
        self.return_metadata = return_metadata
        self.manifest = self._read_manifest(self.manifest_path)
        self.records = self._load_records()
        self._files: dict[Path, Any] = {}

        if max_items is not None:
            if max_items <= 0:
                raise ValueError("max_items must be positive when provided")
            self.records = self.records[:max_items]

        if not self.records:
            raise ValueError(f"No HDF5 tensor records found in: {self.manifest_path}")
        if require_same_shape:
            self._validate_same_shape()

    @staticmethod
    def _resolve_manifest_path(root: Path) -> Path:
        if root.is_file():
            if root.name != "manifest.json":
                raise ValueError(f"HDF5 dataset file must be manifest.json: {root}")
            return root
        if not root.exists():
            raise FileNotFoundError(f"HDF5 tensor dataset root does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"HDF5 tensor dataset root must be a directory: {root}")
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"HDF5 tensor manifest not found: {manifest_path}")
        return manifest_path

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"HDF5 tensor manifest must be a JSON object: {path}")
        if payload.get("format") != "diffusion_hash_inv.hdf5_tensor_manifest":
            raise ValueError(f"Unsupported HDF5 tensor manifest format: {path}")
        return payload

    def _load_records(self) -> list[HDF5TensorRecord]:
        h5py = _import_h5py()
        include_path_set = set(self.include_paths) if self.include_paths is not None else None
        records: list[HDF5TensorRecord] = []

        for shard in self.manifest.get("shards", []):
            if not isinstance(shard, dict) or "file" not in shard:
                raise ValueError(f"Invalid shard entry in manifest: {shard!r}")
            shard_path = self.dataset_dir / str(shard["file"])
            if not shard_path.is_file():
                raise FileNotFoundError(f"HDF5 shard not found: {shard_path}")

            with h5py.File(shard_path, "r") as h5_file:
                source_logs = [_decode_text(value) for value in h5_file["source_logs"][()]]
                paths = [_decode_text(value) for value in h5_file["paths"][()]]
                record_keys = [_decode_text(value) for value in h5_file["record_keys"][()]]
                if not (len(source_logs) == len(paths) == len(record_keys)):
                    raise ValueError(f"HDF5 shard index lengths do not match: {shard_path}")

                for source_log, path, record_key in zip(source_logs, paths, record_keys):
                    if include_path_set is not None and path not in include_path_set:
                        continue
                    tensor_path = f"records/{record_key}/tensor"
                    if tensor_path not in h5_file:
                        raise KeyError(f"Tensor record missing in {shard_path}: {tensor_path}")
                    shape = tuple(int(dim) for dim in h5_file[tensor_path].shape)
                    records.append(
                        HDF5TensorRecord(
                            shard_path=shard_path,
                            record_key=record_key,
                            source_log=source_log,
                            path=path,
                            shape=shape,
                        )
                    )
        return records

    def _validate_same_shape(self) -> None:
        expected = self.records[0].shape
        for record in self.records:
            if record.shape != expected:
                raise ValueError(
                    "HDF5 tensor records have different shapes. "
                    "Use include_paths=(...) to select one image type, or set "
                    "require_same_shape=False for unbatched inspection. "
                    f"Expected {expected}, got {record.shape} at {record.path}."
                )

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        return self.records[0].shape

    @property
    def paths(self) -> list[str]:
        return [record.path for record in self.records]

    @property
    def source_logs(self) -> list[str]:
        return [record.source_log for record in self.records]

    def metadata(self, index: int) -> dict[str, str]:
        record = self.records[index]
        return {
            "shard": str(record.shard_path),
            "record_key": record.record_key,
            "source_log": record.source_log,
            "path": record.path,
        }

    def _file(self, path: Path):
        h5py = _import_h5py()
        resolved = path.resolve()
        if resolved not in self._files:
            self._files[resolved] = h5py.File(resolved, "r")
        return self._files[resolved]

    def close(self) -> None:
        for h5_file in self._files.values():
            h5_file.close()
        self._files.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def __enter__(self) -> "HDF5TensorDataset":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        h5_file = self._file(record.shard_path)
        array = h5_file[f"records/{record.record_key}/tensor"][()]
        tensor = torch.from_numpy(np.asarray(array))
        if self.as_float32 and tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
        index_tensor = torch.tensor(index, dtype=torch.long)
        if self.return_metadata:
            return tensor, index_tensor, self.metadata(index)
        return tensor, index_tensor


def create_hdf5_tensor_dataloader(
    root: Path | str,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    include_paths: Optional[Sequence[str]] = None,
    max_items: int | None = None,
    require_same_shape: bool = True,
    as_float32: bool = True,
    persistent_workers: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for HDF5 tensor shards.

    ``require_same_shape=True`` is the default because PyTorch's default collate
    function needs tensors in a batch to have the same shape.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be greater than or equal to 0")

    dataset = HDF5TensorDataset(
        root,
        include_paths=include_paths,
        max_items=max_items,
        require_same_shape=require_same_shape,
        as_float32=as_float32,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers and num_workers > 0,
    )

