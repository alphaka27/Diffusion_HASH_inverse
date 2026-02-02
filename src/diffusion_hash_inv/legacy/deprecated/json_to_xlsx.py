"""
This script converts JSON files to XLSX format.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Mapping, Optional
import argparse

import pandas as pd

from diffusion_hash_inv.utils import FileIO

class JSONToXLSXConverter:
    """
    Converts JSON files to XLSX format.
    """
    def __init__(self, standalone: bool = False, **kwargs):
        self.standalone = standalone

        self.is_verbose: bool = kwargs.pop("is_verbose", not self.standalone)
        self.length = kwargs.pop("length", 256)

        self.file_io = FileIO(verbose_flag=self.is_verbose)

        self.json_path_arg = kwargs.pop("json_path", None)
        self.xlsx_path_arg = kwargs.pop("xlsx_path", None)

        self.hash_alg = kwargs.pop("hash_alg", None)

    def walk(self, path: List[str], node: object, records: List[dict]) -> List[dict]:
        """
        Recursively walks through a nested dictionary to extract keys.
        """

        if isinstance(node, dict):
            for key, value in node.items():
                if key == "String" and isinstance(value, str):
                    value = "'" + value
                if key == "Overflow" and isinstance(value, bool):
                    pass
                self.walk(path + [key], value, records)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for i, item in enumerate(node):
                if isinstance(item, Mapping) or (isinstance(item, Sequence) \
                    and not isinstance(item, (str, bytes, bytearray))):
                    self.walk(path + [f"[{i}]"], item, records)
                else:
                    records.append({"path": path, "list_index": i, "value": item})
        else:
            records.append({"path": path, "list_index": "", "value": node})

    def pad_path(self, p: list[str], depth: int) -> list[str]:
        """
        Pads the path to ensure uniform depth.
        """
        return [""] * (depth - len(p)) + p

    def _index_to_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts DataFrame index to columns.
        """
        paths = list(df.index)  # [(Level_1, Level_2, ...), ...]
        list_idxs = df["list_index"].tolist()
        values = df["value"].tolist()

        # 컬럼 MultiIndex의 레벨 이름: 기존 인덱스 이름 + list_index
        col_level_names = [*(df.index.names or []), "list_index"]

        # 각 컬럼의 키: (Level_1, ..., Level_n, list_index)
        col_keys = [p + (li,) for p, li in zip(paths, list_idxs)]

        columns = pd.MultiIndex.from_tuples(col_keys, names=col_level_names)

        # 행은 'value' 한 줄만 두기
        df_top = pd.DataFrame([values], index=["value"], columns=columns)
        return df_top

    def _convert_single_file(self, json_file: str) -> pd.DataFrame:
        """
        Converts a single JSON content dictionary to an XLSX DataFrame.
        """
        content = self.file_io.file_reader(json_file, length=self.length)

        metadata = content.get("Metadata", None)
        assert metadata is not None, "Metadata is missing in the JSON data."

        baselogs = content.get("BaseLogs", None)
        assert baselogs is not None, "BaseLogs is missing in the JSON data."

        steplogs = content.get("Logs", None)
        assert steplogs is not None, "Logs are missing in the JSON data."

        program_started_at = metadata["Program started at"][:19].replace(":", "-").replace(" ", "_")
        xlsx_name = \
            f"{metadata['Hash function']}_{metadata['Input bits']}_{program_started_at}.xlsx"

        records = []
        self.walk([], baselogs, records=records)
        self.walk([], steplogs, records=records)
        if not records:
            raise ValueError(f"No records found in {json_file}, skipping.")

        max_depth = max(len(r["path"]) for r in records)

        index_tuples = [tuple(self.pad_path(r["path"], max_depth)) for r in records]
        level_names = [f"Level_{i+1}" for i in range(max_depth)]

        index = pd.MultiIndex.from_tuples(index_tuples, names=level_names)
        df = pd.DataFrame({
            "list_index": [r["list_index"] for r in records],
            "value": [r["value"] for r in records],
        }, index=index)

        df_top = self._index_to_column(df)
        return df_top, xlsx_name

    def update_dataframe(self, df: pd.DataFrame | None, data: object) -> pd.DataFrame:
        """
        Updates the DataFrame with new data.
        """
        if df is None:
            df = pd.DataFrame()
        df_base = pd.concat([df, data])
        # df_base = df_base[~df_base.index.duplicated(keep="last")]

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        return df_base

    def convert_to_xlsx(self, hash_alg: Optional[str] = None) -> None:
        """
        Converts the given data dictionary to XLSX format and saves it.
        """

        assert hash_alg is not None or self.hash_alg is not None, \
            "Hash algorithm must be specified."
        _hash_alg = hash_alg if hash_alg is not None else self.hash_alg

        json_path: Path = Path(self.json_path_arg) if self.json_path_arg \
            else self.file_io.select_dir(filetype="json", length=self.length)
        xlsx_path: Path = Path(self.xlsx_path_arg) if self.xlsx_path_arg \
            else self.file_io.select_dir(filetype="xlsx", length=self.length)

        json_list = self.file_io.get_latest_files_by_date(json_path, _hash_alg, self.length)

        json_list.sort()
        print(f"Found {len(json_list)} JSON files.")

        df = None
        for json_file in json_list:
            data, xlsx_name = self._convert_single_file(json_path / json_file)
            df = self.update_dataframe(df, data)
            print(f"Processed {json_file} into DataFrame.")
            if self.is_verbose:
                print(data)
                print(f"Updated DataFrame:\n{df}")

        assert df is not None and xlsx_name is not None, "DataFrame or XLSX name is not set."
        self.file_io.file_writer(xlsx_path / xlsx_name, df, length=self.length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to XLSX")

    parser.add_argument("--json_path", type=str, help="Path to the input JSON file")
    parser.add_argument("--xlsx_path", type=str, help="Path to the output XLSX file")
    parser.add_argument("-v", "--is_verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--length", type=int, default=256, help="Length of the output text")
    parser.add_argument("--standalone", action="store_true", help="Run as standalone script")
    parser.add_argument("--hash_alg", type=str, required=True, \
                        help="Hash algorithm to filter JSON files")

    args = parser.parse_args()

    converter = JSONToXLSXConverter(
        json_path=args.json_path,
        xlsx_path=args.xlsx_path,
        standalone=args.standalone,
        length=args.length,
        is_verbose=args.is_verbose,
        hash_alg=args.hash_alg,
    )

    converter.convert_to_xlsx()
