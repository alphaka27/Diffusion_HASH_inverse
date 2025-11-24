"""
File I/O Utilities
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
import os
import re

import pandas as pd
from openpyxl import load_workbook
from pandas import ExcelWriter

try:
    from diffusion_hash_inv.utils.project_root import add_root_to_path
    ROOT_DIR = add_root_to_path()
except ImportError as _e:
    print(f"Error importing project root: {_e}")
    raise _e
from diffusion_hash_inv.common import Logs

# try:
#     from diffusion_hash_inv.utils import CSVFormat
# except ImportError as _e:
#     print(f"Error importing CSVFormat: {_e}")
#     raise _e

# 고정 헤더 길이
TS_LEN = 32 # 32 bytes UTF-8 timestamp
BITLEN_LEN = 8 # 64 bits
DIFFTIME_LEN = 8 # 64 bits
PAD_LEN = 16 - BITLEN_LEN - DIFFTIME_LEN  # 8 bytes padding to make total 48 bytes
HEADER_LEN = TS_LEN + BITLEN_LEN + PAD_LEN  # 48 bytes

@dataclass
class Header:
    """
    Represents the header of a binary file.
    """
    timestamp_utf8: bytes  # 32 Bytes
    time_diff: bytes   # float
    bit_length: int        # uint64

class Reader:
    """
    File Reader Utilities
    """
    def __init__(self, byteorder: str):
        self.byteorder = byteorder

    def read_json(self, path: Path) -> str:
        """
        Read the JSON content from a file.
        """
        with open(path, "r", encoding="UTF-8") as j:
            return j.read()

    def read_xlsx(self, path: Path) -> pd.DataFrame:
        """
        Read the Excel content from a file.
        """
        return pd.read_excel(path, engine="openpyxl")

    def read_binary(self, path: Path) -> bytes:
        """
        Read the binary content from a file.
        """
        with open(path, "rb") as f:
            return f.read()

class Writer:
    """
    File Writer Utilities
    """
    def __init__(self, byteorder: str):
        self.byteorder = byteorder

    def write_json(self, path: Path, content: str):
        """
        Write the JSON content to a file.
        """
        with open(path, "w", encoding="UTF-8", newline="\n") as j:
            j.write(content)

    def write_xlsx(self, path: Path, df: pd.DataFrame):
        """
        Write the DataFrame to an Excel file.
        """
        df.to_excel(path, engine="openpyxl", index=True)

    def write_binary(self, path: Path, content: bytes):
        """
        Write the binary content to a file.
        """
        with open(path, "wb") as f:
            f.write(content)

class FileIO(Writer, Reader):
    """
    File I/O Utilities
    """
    def __init__(self, clear_flag, verbose_flag, \
                byteorder: Optional[str] = None, length: Optional[int] = None):
        super().__init__(byteorder=byteorder)

        self.data_dir = ROOT_DIR / "data"
        self.out_dir = ROOT_DIR / "output"
        assert byteorder in ('big', 'little'), "byteorder must be 'big', 'little'"
        self.byteorder = byteorder
        assert length is not None and length > 0, "length must be a positive integer"
        self.length = length

        self.is_verbose = verbose_flag

        if clear_flag:
            print("Clearing generated files...")
            self.file_clean(clear_flag=clear_flag, verbose_flag=verbose_flag)

    #pylint: disable=broad-exception-caught, too-many-branches, too-many-nested-blocks
    def file_clean(self, clear_flag = False, verbose_flag = True):
        """
        Clean up the generated files.
        """
        if not clear_flag:
            return

        targets = [self.data_dir, self.out_dir]
        for root_dir in targets:
            if not root_dir:
                continue
            if not os.path.isdir(root_dir):
                if verbose_flag:
                    print(f"[SKIP] 디렉터리가 아님: {root_dir}")
                continue

            # 하위부터 올라오며 파일/디렉터리 삭제
            for cur, dirs, files in os.walk(root_dir, topdown=False, followlinks=False):
                # 파일 삭제
                for name in files:
                    p = os.path.join(cur, name)
                    try:
                        os.unlink(p)  # 파일/하드링크/심볼릭링크 모두 처리
                        if verbose_flag:
                            print(f"Remove file: {p}")
                    except Exception as e:
                        print(f"[SKIP] {p}: {e}")

                # 디렉터리(or 디렉터리 링크) 삭제
                for name in dirs:
                    p = os.path.join(cur, name)
                    try:
                        if os.path.islink(p):
                            os.unlink(p)      # 디렉터리 링크는 unlink
                            if verbose_flag:
                                print(f"Remove symlink dir: {p}")
                        else:
                            if verbose_flag:
                                print(f"Remove directory: {p}")
                            os.rmdir(p)       # 하위가 이미 지워져서 비어 있음
                    except Exception as e:
                        print(f"[SKIP] {p}: {e}")
    #pylint: enable=broad-exception-caught, too-many-branches, too-many-nested-blocks

    @staticmethod
    def _pad16(f):
        pos = f.tell()
        if pos % 16 != 0:
            f.write(b'\x00' * (16 - (pos % 16)))

    @staticmethod
    def encode_timestamp() -> bytes:
        """
        Encode the current timestamp as bytes.
        """
        s = Logs.timestamp()
        if len(s) != TS_LEN:
            raise ValueError(f"timestamp length != {TS_LEN}: {len(s)}")
        return s.encode("utf-8")

    @staticmethod
    def decode_timestamp(b: bytes) -> datetime:
        """
        Decode the timestamp from bytes.
        """
        return datetime.fromisoformat(b.decode('utf-8'))

    def encode_bit_length(self, bit_length: int) -> bytes:
        """
        Encode the bit length as bytes.
        """
        if bit_length < 0:
            raise ValueError("bit_length must be >= 0")
        return bit_length.to_bytes(8, self.byteorder, signed=False)

    def decode_bit_length(self, b: bytes) -> int:
        """
        Decode the bit length from bytes.
        """
        return int.from_bytes(b, self.byteorder, signed=False)

    def _select_data_dir(self, filename: str, length: int) -> Path:
        """
        Decide subdirectory by extension and ensure it exists.
        """

        if filename.endswith(".bin"):
            base = self.data_dir / "binary"

        elif filename.endswith(".char"):
            base = self.data_dir / "character"

        elif filename.endswith(".json"):
            base = self.out_dir / "json" / f"{length}"

        elif filename.endswith(".xlsx"):
            base = self.out_dir / "xlsx" / f"{length}"

        else:
            raise ValueError("Invalid file extension. Use .bin, .char, .json, or .xlsx")

        base.mkdir(parents=True, exist_ok=True)  # ← 실제 타깃 디렉터리 생성
        return base

    def _sanitize_filename(self, name: str) -> str:
        # 윈도우/범용 안전: 콜론, 슬래시, 역슬래시 등 치환
        return name.replace(":", "-").replace("/", "_").replace("\\", "_")

    def file_writer(self, filename: str, content: Any, length: int) -> None:
        """
        Get the full path for writing a file, ensuring directories exist.
        """
        base = self._select_data_dir(filename, length)
        safe_name = self._sanitize_filename(filename)
        full_path = base / safe_name
        if full_path.suffix == ".json":
            self.write_json(full_path, content)
        if full_path.suffix == ".xlsx":
            self.write_xlsx(full_path, content)
        if full_path.suffix in (".bin", ".char"):
            self.write_binary(full_path, content)

        raise ValueError("Invalid file extension. Use .bin, .char, .json, or .xlsx")

    def file_reader(self, filename: str, length: int) -> Any:
        """
        Get the full path for reading a file.
        """
        base = self._select_data_dir(filename, length)
        safe_name = self._sanitize_filename(filename)
        full_path = base / safe_name
        if full_path.suffix == ".json":
            return self.read_json(full_path)
        if full_path.suffix == ".xlsx":
            return self.read_xlsx(full_path)
        if full_path.suffix in (".bin", ".char"):
            return self.read_binary(full_path)

        raise ValueError("Invalid file extension. Use .bin, .char, .json, or .xlsx")
