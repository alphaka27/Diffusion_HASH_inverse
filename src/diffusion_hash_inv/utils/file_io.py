"""
File I/O Utilities
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Any
import os
import re

import pandas as pd

from diffusion_hash_inv.utils import JSONFormat
from diffusion_hash_inv.utils.project_root import add_root_to_path
ROOT_DIR = add_root_to_path()



# 고정 헤더 길이
TS_LEN = 32 # 32 bytes UTF-8 timestamp
BITLEN_LEN = 8 # 64 bits
DIFFTIME_LEN = 8 # 64 bits
PAD_LEN = 16 - BITLEN_LEN - DIFFTIME_LEN  # 8 bytes padding to make total 48 bytes
HEADER_LEN = TS_LEN + DIFFTIME_LEN + BITLEN_LEN + PAD_LEN  # 48 bytes

@dataclass
class Header:
    """
    Represents the header of a binary file.
    """
    timestamp: str   # 32 Bytes
    time_diff: int   # int
    bit_length: int  # uint64

    def encode_timestamp(self, encoding: str) -> bytes:
        """
        Encode the current timestamp as bytes.
        """
        s = self.timestamp
        if len(s) != TS_LEN:
            raise ValueError(f"timestamp length != {TS_LEN}: {len(s)}")
        return s.encode(encoding)

    def decode_timestamp(self, b: bytes, encoding: str) -> datetime:
        """
        Decode the timestamp from bytes.
        """
        return datetime.fromisoformat(b.decode(encoding))

    def encode_bit_length(self, bit_length: int, byteorder: str) -> bytes:
        """
        Encode the bit length as bytes.
        """
        if bit_length < 0:
            raise ValueError("bit_length must be >= 0")
        return bit_length.to_bytes(BITLEN_LEN, byteorder, signed=False)

    def decode_bit_length(self, b: bytes, byteorder: str) -> int:
        """
        Decode the bit length from bytes.
        """
        return int.from_bytes(b, byteorder, signed=False)

    def encode_timediff(self, byteorder: str) -> bytes:
        """
        Encode the time difference as bytes.
        """
        if self.time_diff < 0:
            raise ValueError("time_diff must be >= 0")
        return self.time_diff.to_bytes(DIFFTIME_LEN, byteorder, signed=False)

    def decode_timediff(self, b: bytes, byteorder: str) -> int:
        """
        Decode the time difference from bytes.
        """
        return int.from_bytes(b, byteorder, signed=False)

    def encode(self, encoding: Optional[str] = None, byteorder: Optional[str] = None) -> bytes:
        """
        Encode the header to bytes.
        """
        assert encoding is not None, "encoding must be specified"
        assert byteorder is not None, "byteorder must be specified"
        header_bytes = bytearray()
        header_bytes.extend(self.encode_timestamp(encoding))
        header_bytes.extend(self.encode_timediff(byteorder))
        header_bytes.extend(self.encode_bit_length(self.bit_length, byteorder))
        header_bytes.extend(b'\x00' * PAD_LEN)  # Padding
        assert len(header_bytes) == HEADER_LEN, \
            f"Header length != {HEADER_LEN}: {len(header_bytes)}"
        return bytes(header_bytes)

    def decode(self, b: bytes, encoding: Optional[str] = None, \
            byteorder: Optional[str] = None) -> Header:
        """
        Decode the header from bytes.
        """
        assert encoding is not None, "encoding must be specified"
        assert byteorder is not None, "byteorder must be specified"
        if len(b) != HEADER_LEN:
            raise ValueError(f"Header length != {HEADER_LEN}: {len(b)}")
        timestamp_bytes = b[0:TS_LEN]
        time_diff_bytes = b[TS_LEN:TS_LEN + DIFFTIME_LEN]
        bit_length_bytes = b[TS_LEN + DIFFTIME_LEN:TS_LEN + DIFFTIME_LEN + BITLEN_LEN]

        timestamp = self.decode_timestamp(timestamp_bytes, encoding)
        time_diff = self.decode_timediff(time_diff_bytes, byteorder)
        bit_length = self.decode_bit_length(bit_length_bytes, byteorder)

        return Header(timestamp=timestamp.isoformat(), time_diff=time_diff, bit_length=bit_length)

class Writer:
    """
    File Writer Utilities
    """
    def __init__(self):
        pass

    def write_json(self, path: Path, content: str):
        """
        Write the JSON content to a file.
        """
        with open(path, "w", encoding="UTF-8", newline="\n") as j:
            j.write(JSONFormat.dumps(indent=4, **content))

    def write_xlsx(self, path: Path, df: pd.DataFrame):
        """
        Write the DataFrame to an Excel file.
        """
        df.to_excel(path, engine="openpyxl", index=True)

    def write_binary(self, path: Path, header: Optional[bytes] = None, \
                    content: Optional[bytes] = None, byteorder: Optional[str] = None):
        """
        Write the binary content to a file.
        """
        assert byteorder in ('big', 'little'), "byteorder must be 'big', 'little'"
        assert content is not None and isinstance(content, bytes), "content must be bytes"
        assert header is not None and isinstance(header, bytes), "header must be bytes"
        content = header + content

        with open(path, "ab") as f:
            f.write(content)

class Reader:
    """
    File Reader Utilities
    """
    def __init__(self):
        pass

    def read_json(self, path: Path) -> str:
        """
        Read the JSON content from a file.
        """
        with open(path, "r", encoding="UTF-8") as j:
            temp = j.read()
            return JSONFormat.loads(temp)

    def read_xlsx(self, path: Path) -> pd.DataFrame:
        """
        Read the Excel content from a file.
        """
        return pd.read_excel(path, engine="openpyxl")

    def read_binary(self, path: Path, byteorder: Optional[str] = None) -> bytes:
        """
        Read the binary content from a file.
        """
        assert byteorder in ('big', 'little'), "byteorder must be 'big', 'little'"
        with open(path, "rb") as f:
            return f.read()

class FileIO(Writer, Reader):
    """
    File I/O Utilities
    """
    def __init__(self, verbose_flag=True):
        super().__init__()
        self.data_dir = ROOT_DIR / "data"
        self.out_dir = ROOT_DIR / "output"
        self.is_verbose = verbose_flag

    #pylint: disable=broad-exception-caught, too-many-branches, too-many-nested-blocks
    def file_clean(self, clear_flag = False, verbose_flag = True):
        """
        Clean up the generated files.
        """
        print("Clearing generated files...")
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

    def get_latest_file_by_date(self, dir_path: str, hash_alg: str, length: int) -> str:
        """
        Parse the filename to extract base name without extension.
        """
        pattern = re.compile(r'(\d{4}-\d{2}-\d{2}) (\d{2}-\d{2}-\d{2})')
        base = dir_path
        candidates = list(base.glob(f"{hash_alg}_{length}_*.json"))
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

    def select_data_dir(self, filetype: str, length: int) -> Path:
        """
        Decide subdirectory by extension and ensure it exists.
        """

        if filetype.endswith(".bin") or filetype == "bin":
            base = self.data_dir / "binary"

        elif filetype.endswith(".char") or filetype == "char":
            base = self.data_dir / "character"

        elif filetype.endswith(".json") or filetype == "json":
            base = self.out_dir / "json" / f"{length}"

        elif filetype.endswith(".xlsx") or filetype == "xlsx":
            base = self.out_dir / "xlsx" / f"{length}"

        elif filetype.endswith(".png") or filetype == "png":
            base = self.out_dir / "images" / f"{length}"

        else:
            raise ValueError("Invalid file extension. Use .bin, .char, .json, .xlsx, or .png")

        base.mkdir(parents=True, exist_ok=True)  # ← 실제 타깃 디렉터리 생성
        return base

    def _sanitize_filename(self, name: str) -> str:
        # 윈도우/범용 안전: 콜론, 슬래시, 역슬래시 등 치환
        return name.replace(":", "-").replace("/", "_").replace("\\", "_")

    def file_writer(self, filename: str, content: Any, length: int, **kwargs) -> None:
        """
        Get the full path for writing a file, ensuring directories exist.
        """
        header = None
        header_bytes = None
        start_timestamp = kwargs.get("timestamp", None)
        elapsed_time = kwargs.get("elapsed_time", None)
        byteorder = kwargs.get("byteorder", None)
        if start_timestamp is not None and elapsed_time is not None:
            header = Header(start_timestamp, elapsed_time, length)
            header_bytes = header.encode("utf-8", byteorder)

        base = self.select_data_dir(filename, length)
        safe_name = self._sanitize_filename(filename)
        full_path = base / safe_name
        if full_path.suffix == ".json":
            self.write_json(full_path, content)
        elif full_path.suffix == ".xlsx":
            self.write_xlsx(full_path, content)
        elif full_path.suffix in (".bin", ".char"):
            self.write_binary(full_path, header=header_bytes, content=content, \
                            byteorder=byteorder)
        else:
            raise ValueError("Invalid file extension. Use .bin, .char, .json, or .xlsx")

    def file_reader(self, filename: str, length: int) -> Any:
        """
        Get the full path for reading a file.
        """
        base = self.select_data_dir(filename, length)
        safe_name = self._sanitize_filename(filename)
        full_path = base / safe_name
        if full_path.suffix == ".json":
            return self.read_json(full_path)
        if full_path.suffix == ".xlsx":
            return self.read_xlsx(full_path)
        if full_path.suffix in (".bin", ".char"):
            return self.read_binary(full_path)

        raise ValueError("Invalid file extension. Use .bin, .char, .json, or .xlsx")
