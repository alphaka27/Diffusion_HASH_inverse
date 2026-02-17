"""
File I/O Utilities
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Any
import re
from PIL import Image

import pandas as pd
from torch.utils.data import Dataset

from diffusion_hash_inv.config import MainConfig, OutputConfig, HeaderConstants as HC
from diffusion_hash_inv.utils import JSONFormat

# Instantiate configuration once so property fields are resolved instead of accessing the class.
# TODO: Change to use OutputConfig constants instead of using global constants here
TS_LEN = HC.timestamp_length  # 32 bytes
BITLEN_LEN = HC.bits_length # 8 bytes
DIFFTIME_LEN = HC.difftime_length  # 8 bytes
PAD_LEN = HC.padding_length  # 0 bytes
HEADER_LEN = HC.header_length  # 48 bytes

@dataclass
class Header:
    """
    Represents the header of a binary file.
    """
    timestamp: str   # 32 Bytes
    time_diff: int   # int64
    bit_length: int  # uint64
    byteorder: str

    def encode_timestamp(self) -> bytes:
        """
        Encode the current timestamp as bytes.
        """
        s = self.timestamp
        if len(s) != HC.timestamp_length:
            raise ValueError(f"timestamp length != {HC.timestamp_length}: {len(s)}")
        return s.encode(OutputConfig.encoding)

    def decode_timestamp(self, b: bytes) -> datetime:
        """
        Decode the timestamp from bytes.
        """
        return datetime.fromisoformat(b.decode(OutputConfig.encoding))

    def encode_bit_length(self) -> bytes:
        """
        Encode the bit length as bytes.
        """
        return self.bit_length.to_bytes(BITLEN_LEN, self.byteorder, signed=False)

    def decode_bit_length(self, b: bytes) -> int:
        """
        Decode the bit length from bytes.
        """
        return int.from_bytes(b, self.byteorder, signed=False)

    def encode_timediff(self) -> bytes:
        """
        Encode the time difference as bytes.
        """
        return self.time_diff.to_bytes(DIFFTIME_LEN, self.byteorder, signed=False)

    def decode_timediff(self, b: bytes) -> int:
        """
        Decode the time difference from bytes.
        """
        return int.from_bytes(b, self.byteorder, signed=False)

    def encode(self) -> bytes:
        """
        Encode the header to bytes.
        """
        header_bytes = bytearray()
        header_bytes.extend(self.encode_timestamp())
        header_bytes.extend(self.encode_timediff())
        header_bytes.extend(self.encode_bit_length())
        header_bytes.extend(b'\x00' * HC.padding_length)  # Padding
        assert len(header_bytes) == HC.header_length, \
            f"Header length != {HC.header_length}: {len(header_bytes)}"
        return bytes(header_bytes)

    def decode(self, b: bytes, encoding: Optional[str] = None, \
            byteorder: Optional[str] = None) -> Header:
        """
        Decode the header from bytes.
        """
        assert encoding is not None, "encoding must be specified"
        assert byteorder is not None, "byteorder must be specified"
        if len(b) != HC.header_length:
            raise ValueError(f"Header length != {HC.header_length}: {len(b)}")
        timestamp_bytes = b[0:HC.timestamp_length]
        time_diff_bytes = b[HC.timestamp_length:HC.timestamp_length + HC.difftime_length]
        bit_length_bytes = b[HC.timestamp_length + HC.difftime_length:\
                            HC.timestamp_length + HC.difftime_length + HC.bits_length]
        timestamp = self.decode_timestamp(timestamp_bytes)
        time_diff = self.decode_timediff(time_diff_bytes)
        bit_length = self.decode_bit_length(bit_length_bytes)

        return Header(timestamp=timestamp.isoformat(), \
                    time_diff=time_diff, bit_length=bit_length, byteorder=byteorder)


class Writer:
    """
    File Writer Utilities
    """
    def __init__(self):
        pass

    @staticmethod
    def write_json(path: Path, content: str):
        """
        Write the JSON content to a file.
        """
        with open(path, "w", encoding=OutputConfig.encoding, newline="\n") as j:
            j.write(JSONFormat.dumps(indent=4, **content))

    @staticmethod
    def write_xlsx(path: Path, df: pd.DataFrame):
        """
        Write the DataFrame to an Excel file.
        """
        df.to_excel(path, engine="openpyxl", index=True)

    @staticmethod
    def write_binary(path: Path, content: Optional[bytes] = None, **kwargs):
        """
        Write the binary content to a file.
        """
        assert content is not None and isinstance(content, bytes), "content must be bytes"
        length = kwargs.pop("length", None)
        header = None
        header_bytes = None
        start_timestamp = kwargs.pop("timestamp", None)
        elapsed_time = kwargs.pop("elapsed_time", None)
        byteorder = kwargs.pop("byteorder", None)
        if start_timestamp is not None and elapsed_time is not None:
            header = Header(start_timestamp, elapsed_time, length, byteorder)
            header_bytes = header.encode()
        content = header_bytes + content

        with open(path, "ab") as f:
            f.write(content)

    @staticmethod
    def image_writer(path: Path, content: Image.Image):
        """
        Write the image content to a file.
        """
        par, child = path.parent, path.name
        par.mkdir(parents=True, exist_ok=True)
        print(f"Saving image to: {path}")
        content.save(par / child)

class Reader:
    """
    File Reader Utilities
    """
    def __init__(self):
        pass

    @staticmethod
    def read_json(path: Path) -> str:
        """
        Read the JSON content from a file.
        """
        with open(path, "r", encoding=OutputConfig.encoding) as j:
            temp = j.read()
            return JSONFormat.loads(temp)

    @staticmethod
    def read_xlsx(path: Path) -> pd.DataFrame:
        """
        Read the Excel content from a file.
        """
        return pd.read_excel(path, engine="openpyxl")

    @staticmethod
    def read_binary(path: Path, byteorder: Optional[str] = None) -> bytes:
        """
        Read the binary content from a file.
        """
        assert byteorder in ('big', 'little'), "byteorder must be 'big', 'little'"
        with open(path, "rb") as f:
            return f.read()

    @staticmethod
    def read_image(path: Path) -> Image.Image:
        """
        Read the image content from a file.
        """
        return Image.open(path)


class FileIO:
    """
    File I/O Utilities
    """
    def __init__(self, main_config: MainConfig, output_cfg: OutputConfig) -> None:
        self.main_config = main_config
        self.root_dir = output_cfg.root_dir
        self.data_dir = output_cfg.data_dir
        self.out_dir = output_cfg.output_dir
        self.allow_extensions = (".bin", ".char", ".json", ".xlsx", ".png")

        if self.main_config.verbose_flag:
            print(f"Data Directory: {self.data_dir}")
            print(f"Output Directory: {self.out_dir}")

        if self.main_config.clean_flag:
            self.file_clean()

    def _dfs_remove(self, cur: Path, depth: int) -> None:
        if cur.is_symlink():
            print("\t"*depth + cur.name+"@")
            cur.unlink()
            return

        if cur.is_dir():
            print("\t"*depth + cur.name+"/")
            children = sorted(cur.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            for child in children:
                self._dfs_remove(child, depth + 1)
            cur.rmdir()

        else:
            print("\t"*depth + cur.name)
            cur.unlink()

    def file_clean(self):
        """
        Clean up the generated files.
        """

        if not self.main_config.clean_flag:
            return

        print("Clearing generated files...")

        targets = [self.data_dir, self.out_dir]
        root_dir: Path = self.root_dir
        _root = root_dir.resolve()

        if self.main_config.verbose_flag:
            print(f"Root Directory: {_root}")

        for target in targets:
            _target = (_root / target).resolve()
            _target.relative_to(_root)  # Ensure _target is within _root
            if self.main_config.verbose_flag:
                print(f"Removing Directory: {_target}")

            if not _target.exists():
                raise FileNotFoundError(f"Directory not found: {_target}")
            if not _target.is_dir():
                raise NotADirectoryError(f"Not a directory: {_target}")

            self._dfs_remove(_target, 0)

        self.main_config.reset_clean_flag()

    def get_latest_files_by_date(self, hash_alg: str, length: int, dir_path: Path = None) \
        -> List[Path]:
        """
        Parse the filename to extract base name without extension.
        """
        pattern = re.compile(r'(\d{4}-\d{2}-\d{2}) (\d{2}-\d{2}-\d{2})')
        base = dir_path if dir_path is not None else self.out_dir / "json" / f"{length}"
        candidates = list(base.glob(f"{hash_alg.upper()}_{length}_*.json"))
        latest_dt: Optional[datetime] = None
        latest_files: List[Path] = []

        for p in candidates:
            m = pattern.search(p.name)
            if not m:
                continue
            date_str, time_str = m.groups()
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H-%M-%S")
            if latest_dt is None or dt > latest_dt:
                latest_dt = dt
                latest_files = [p]
            elif dt == latest_dt:
                latest_files.append(p)
        return latest_files

    def select_dir(self, filepath: Path | str, **kwargs) -> Path:
        """
        Decide subdirectory by extension and ensure it exists.
        """
        if isinstance(filepath, str):
            if not any(filepath.endswith(ext) for ext in self.allow_extensions):
                raise ValueError(f"Invalid file extension. Use {', '.join(self.allow_extensions)}")
        length: int = kwargs.pop("length", None)
        data_type: str = kwargs.pop("data_type", None)

        if isinstance(filepath, Path):
            filepath = str(filepath.name)

        if filepath.endswith(".bin") or filepath == "bin":
            base = self.data_dir / "binary"

        elif filepath.endswith(".char") or filepath == "char":
            base = self.data_dir / "character"

        elif filepath.endswith(".json") or filepath == "json":
            assert length is not None, "length must be specified for JSON files"
            base = self.out_dir / "json" / f"{length}"

        elif filepath.endswith(".xlsx") or filepath == "xlsx":
            assert length is not None, "length must be specified for XLSX files"
            base = self.out_dir / "xlsx" / f"{length}"

        elif filepath.endswith(".png") or filepath in ("png", "image"):
            assert data_type is not None, "data_type must be specified for image files"
            if data_type == "data":
                base = self.data_dir / "images"
            elif data_type == "output":
                base = self.out_dir / "images"
        else:
            raise ValueError(f"Invalid file extension. Use {', '.join(self.allow_extensions)}")

        base.mkdir(parents=True, exist_ok=True)  # ← 실제 타깃 디렉터리 생성

        return base

    def _sanitize_filename(self, name: str) -> str:
        # 윈도우/범용 안전: 콜론, 슬래시, 역슬래시 등 치환
        return name.replace(":", "-").replace("/", "_").replace("\\", "_")

    def file_writer(self, filename: Path | str, content: Any, **kwargs) -> None:
        """
        Get the full path for writing a file, ensuring directories exist.
        """
        length = kwargs.get("length", None)
        data_type = kwargs.pop("data_type", None)
        parent_dir = kwargs.pop("parent_dir", None)
        if parent_dir is not None:
            parent_dir = Path(parent_dir)
            filename = parent_dir / Path(filename)

        base: Path

        if isinstance(filename, str):
            base = self.select_dir(filename, length=length, data_type=data_type)
            safe_name = self._sanitize_filename(filename)
            full_path = base / safe_name
        elif isinstance(filename, Path):
            if not filename.is_dir():
                base = self.select_dir(filename, length=length, data_type=data_type)
            full_path = base / filename
        else:
            raise ValueError(f"filename must be a string or Path, got {type(filename)}")

        print(full_path)

        if full_path.suffix == ".json":
            Writer.write_json(full_path, content)
        elif full_path.suffix == ".xlsx":
            Writer.write_xlsx(full_path, content)
        elif full_path.suffix in (".bin", ".char"):
            Writer.write_binary(full_path, content=content, **kwargs)
        elif full_path.suffix == ".png" or isinstance(content, Dataset):
            Writer.image_writer(full_path, content)
        else:
            raise ValueError(f"Invalid file extension. Use {', '.join(self.allow_extensions)}")

    def file_reader(self, filename: Path | str, length: int) -> Any:
        """
        Get the full path for reading a file.
        """
        base = self.select_dir(filename, length = length)
        if isinstance(filename, str):
            safe_name = self._sanitize_filename(filename)
            full_path = base / safe_name
        else:
            full_path = base / filename.name
        if full_path.suffix == ".json":
            return Reader.read_json(full_path)
        if full_path.suffix == ".xlsx":
            return Reader.read_xlsx(full_path)
        if full_path.suffix in (".bin", ".char"):
            return Reader.read_binary(full_path)
        raise ValueError("Invalid file extension. Use .bin, .char, .json, or .xlsx")
