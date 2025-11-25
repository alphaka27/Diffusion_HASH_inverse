import json
from collections.abc import Mapping, Sequence
import pandas as pd


def json_to_multiindex_df_same_depth(data: object) -> pd.DataFrame:
    """
    dict의 키 경로만 MultiIndex로 사용하고,
    list는 인덱스 깊이를 늘리지 않고 같은 깊이에서
    list_index 컬럼으로 풀어내는 버전이에요.
    """
    records: list[dict[str, object]] = []

    def walk(node: object, path: list[str]) -> None:
        # dict: 키를 path에 추가
        if isinstance(node, Mapping):
            for k, v in node.items():
                walk(v, path + [str(k)])

        # list/tuple: 인덱스는 별도 컬럼(list_index)로 처리
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for i, v in enumerate(node):
                if isinstance(v, Mapping) or (
                    isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray))
                ):
                    walk(v, path + [str(i)])
                else:
                    records.append(
                        {
                            "path": path,
                            "list_index": i,
                            "value": v,
                        }
                    )
        else:
            # 스칼라 값
            records.append(
                {
                    "path": path,
                    "list_index": "",
                    "value": node,
                }
            )

    walk(data, [])
    # print(records)

    if not records:
        return pd.DataFrame(columns=["list_index", "value"])

    # dict 경로(path)만 MultiIndex로 사용
    max_depth = max(len(r["path"]) for r in records)

    def pad_path(p: list[str], depth: int) -> list[str]:
        return [""] * (depth - len(p)) + p

    index_tuples = [tuple(pad_path(r["path"], max_depth)) for r in records]
    level_names = [f"Level_{i+1}" for i in range(max_depth)]

    index = pd.MultiIndex.from_tuples(index_tuples, names=level_names)

    df = pd.DataFrame(
        {
            "list_index": [r["list_index"] for r in records],
            "value": [r["value"] for r in records],
        },
        index=index,
    )
    return df


def index_to_column_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    행 MultiIndex(= df.index)와 list_index를 합쳐서
    컬럼 MultiIndex로 올리고,
    value 값만 한 줄짜리 행으로 두는 함수예요.

    결과:
        - columns: MultiIndex (Level_1, Level_2, ..., Level_n, list_index)
        - index: ['value']
        - 값: 각 (path, list_index)에 해당하는 value
    """
    # 원래 행 인덱스(멀티인덱스)의 튜플 목록
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


if __name__ == "__main__":
    # 위에 주신 data 그대로 사용
    data = {
    "Metadata": {
        "Hash function": "md5",
        "Input bits": 256,
        "Program started at": "2025-11-25 01:59:47.151429+09:00",
        "Message mode": True,
        "Entropy": 1280.0,
        "Strength": "Very Strong",
        "Elapsed time": "1238791 ns"
    },
    "Message": {
        "Hex": "0x463a7755226f51732b4c2e61225a4b245e4a5a453c22756b243a682a5033672b",
        "String": "F:wU\"oQs+L.a\"ZK$^JZE<\"uk$:h*P3g+"
    },
    "Generated hash": "0x27379f5301ce4a016b812e4ae9924414",
    "Correct   hash": "0x27379f5301ce4a016b812e4ae9924414",
    "Logs": {
        "1st Step": "0x463a7755226f51732b4c2e61225a4b245e4a5a453c22756b243a682a5033672b800000000000000000000000000000000000000000000000",
        "2nd Step": {
            "1st Block": [
                "0x463a7755",
                "0x226f5173",
                "0x2b4c2e61",
                "0x225a4b24",
                "0x5e4a5a45",
                "0x3c22756b",
                "0x243a682a",
                "0x5033672b",
                "0x80000000",
                "0x00000000",
                "0x00000000",
                "0x00000000",
                "0x00000000",
                "0x00000000",
                "0x00010000",
                "0x00000000"
            ]
        },
        "3rd Step": {
            "A": "0x01234567",
            "B": "0x89abcdef",
            "C": "0xfedcba98",
            "D": "0x76543210"
        },
        "4th Step": {
            "1st Round": {
                "1st Loop": {
                    "A": "0x76543210",
                    "B": "0x1f0abd60",
                    "C": "0x89abcdef",
                    "D": "0xfedcba98"
                },
                "2nd Loop": {
                    "A": "0xfedcba98",
                    "B": "0x6c805a16",
                    "C": "0x1f0abd60",
                    "D": "0x89abcdef"
                },
                "3rd Loop": {
                    "A": "0x89abcdef",
                    "B": "0xb98f7ca1",
                    "C": "0x6c805a16",
                    "D": "0x1f0abd60"
                },
                "4th Loop": {
                    "A": "0x1f0abd60",
                    "B": "0xce3b42d3",
                    "C": "0xb98f7ca1",
                    "D": "0x6c805a16"
                },
                "5th Loop": {
                    "A": "0x6c805a16",
                    "B": "0xde253a49",
                    "C": "0xce3b42d3",
                    "D": "0xb98f7ca1"
                },
                "6th Loop": {
                    "A": "0xb98f7ca1",
                    "B": "0x8740862a",
                    "C": "0xde253a49",
                    "D": "0xce3b42d3"
                },
                "7th Loop": {
                    "A": "0xce3b42d3",
                    "B": "0x35db02c2",
                    "C": "0x8740862a",
                    "D": "0xde253a49"
                },
                "8th Loop": {
                    "A": "0xde253a49",
                    "B": "0xcfa584fd",
                    "C": "0x35db02c2",
                    "D": "0x8740862a"
                },
                "9th Loop": {
                    "A": "0x8740862a",
                    "B": "0x89c3445c",
                    "C": "0xcfa584fd",
                    "D": "0x35db02c2"
                },
                "10th Loop": {
                    "A": "0x35db02c2",
                    "B": "0xc6fc6379",
                    "C": "0x89c3445c",
                    "D": "0xcfa584fd"
                },
                "11th Loop": {
                    "A": "0xcfa584fd",
                    "B": "0x533a436a",
                    "C": "0xc6fc6379",
                    "D": "0x89c3445c"
                },
                "12th Loop": {
                    "A": "0x89c3445c",
                    "B": "0xb0040440",
                    "C": "0x533a436a",
                    "D": "0xc6fc6379"
                },
                "13th Loop": {
                    "A": "0xc6fc6379",
                    "B": "0x50856adc",
                    "C": "0xb0040440",
                    "D": "0x533a436a"
                },
                "14th Loop": {
                    "A": "0x533a436a",
                    "B": "0xdf5231b7",
                    "C": "0x50856adc",
                    "D": "0xb0040440"
                },
                "15th Loop": {
                    "A": "0xb0040440",
                    "B": "0xa01cd5bd",
                    "C": "0xdf5231b7",
                    "D": "0x50856adc"
                },
                "16th Loop": {
                    "A": "0x50856adc",
                    "B": "0x87d93466",
                    "C": "0xa01cd5bd",
                    "D": "0xdf5231b7"
                },
                "17th Loop": {
                    "A": "0xdf5231b7",
                    "B": "0xf5a80300",
                    "C": "0x87d93466",
                    "D": "0xa01cd5bd"
                },
                "18th Loop": {
                    "A": "0xa01cd5bd",
                    "B": "0xbc7e17f8",
                    "C": "0xf5a80300",
                    "D": "0x87d93466"
                },
                "19th Loop": {
                    "A": "0x87d93466",
                    "B": "0xcecf10b4",
                    "C": "0xbc7e17f8",
                    "D": "0xf5a80300"
                },
                "20th Loop": {
                    "A": "0xf5a80300",
                    "B": "0x49a74a58",
                    "C": "0xcecf10b4",
                    "D": "0xbc7e17f8"
                },
                "21th Loop": {
                    "A": "0xbc7e17f8",
                    "B": "0x5c029b0d",
                    "C": "0x49a74a58",
                    "D": "0xcecf10b4"
                },
                "22th Loop": {
                    "A": "0xcecf10b4",
                    "B": "0xe9ba0579",
                    "C": "0x5c029b0d",
                    "D": "0x49a74a58"
                },
                "23th Loop": {
                    "A": "0x49a74a58",
                    "B": "0x7af5308f",
                    "C": "0xe9ba0579",
                    "D": "0x5c029b0d"
                },
                "24th Loop": {
                    "A": "0x5c029b0d",
                    "B": "0x541eb1f5",
                    "C": "0x7af5308f",
                    "D": "0xe9ba0579"
                },
                "25th Loop": {
                    "A": "0xe9ba0579",
                    "B": "0xd81077cb",
                    "C": "0x541eb1f5",
                    "D": "0x7af5308f"
                },
                "26th Loop": {
                    "A": "0x7af5308f",
                    "B": "0x474733a7",
                    "C": "0xd81077cb",
                    "D": "0x541eb1f5"
                },
                "27th Loop": {
                    "A": "0x541eb1f5",
                    "B": "0x391d2fc0",
                    "C": "0x474733a7",
                    "D": "0xd81077cb"
                },
                "28th Loop": {
                    "A": "0xd81077cb",
                    "B": "0x6110316e",
                    "C": "0x391d2fc0",
                    "D": "0x474733a7"
                },
                "29th Loop": {
                    "A": "0x474733a7",
                    "B": "0x3c5b33e1",
                    "C": "0x6110316e",
                    "D": "0x391d2fc0"
                },
                "30th Loop": {
                    "A": "0x391d2fc0",
                    "B": "0x2321d5e9",
                    "C": "0x3c5b33e1",
                    "D": "0x6110316e"
                },
                "31th Loop": {
                    "A": "0x6110316e",
                    "B": "0x29f07c11",
                    "C": "0x2321d5e9",
                    "D": "0x3c5b33e1"
                },
                "32th Loop": {
                    "A": "0x3c5b33e1",
                    "B": "0x2545dde2",
                    "C": "0x29f07c11",
                    "D": "0x2321d5e9"
                },
                "33th Loop": {
                    "A": "0x2321d5e9",
                    "B": "0xbbf35154",
                    "C": "0x2545dde2",
                    "D": "0x29f07c11"
                },
                "34th Loop": {
                    "A": "0x29f07c11",
                    "B": "0x84cc480f",
                    "C": "0xbbf35154",
                    "D": "0x2545dde2"
                },
                "35th Loop": {
                    "A": "0x2545dde2",
                    "B": "0x6205aeda",
                    "C": "0x84cc480f",
                    "D": "0xbbf35154"
                },
                "36th Loop": {
                    "A": "0xbbf35154",
                    "B": "0x3e42df21",
                    "C": "0x6205aeda",
                    "D": "0x84cc480f"
                },
                "37th Loop": {
                    "A": "0x84cc480f",
                    "B": "0xd4d19c2b",
                    "C": "0x3e42df21",
                    "D": "0x6205aeda"
                },
                "38th Loop": {
                    "A": "0x6205aeda",
                    "B": "0x5f6d85a7",
                    "C": "0xd4d19c2b",
                    "D": "0x3e42df21"
                },
                "39th Loop": {
                    "A": "0x3e42df21",
                    "B": "0xf6174d2a",
                    "C": "0x5f6d85a7",
                    "D": "0xd4d19c2b"
                },
                "40th Loop": {
                    "A": "0xd4d19c2b",
                    "B": "0xcb911040",
                    "C": "0xf6174d2a",
                    "D": "0x5f6d85a7"
                },
                "41th Loop": {
                    "A": "0x5f6d85a7",
                    "B": "0x8d512461",
                    "C": "0xcb911040",
                    "D": "0xf6174d2a"
                },
                "42th Loop": {
                    "A": "0xf6174d2a",
                    "B": "0x25d15e1e",
                    "C": "0x8d512461",
                    "D": "0xcb911040"
                },
                "43th Loop": {
                    "A": "0xcb911040",
                    "B": "0x16345fd2",
                    "C": "0x25d15e1e",
                    "D": "0x8d512461"
                },
                "44th Loop": {
                    "A": "0x8d512461",
                    "B": "0x64476dab",
                    "C": "0x16345fd2",
                    "D": "0x25d15e1e"
                },
                "45th Loop": {
                    "A": "0x25d15e1e",
                    "B": "0x3e89c9d1",
                    "C": "0x64476dab",
                    "D": "0x16345fd2"
                },
                "46th Loop": {
                    "A": "0x16345fd2",
                    "B": "0xaf3ef484",
                    "C": "0x3e89c9d1",
                    "D": "0x64476dab"
                },
                "47th Loop": {
                    "A": "0x64476dab",
                    "B": "0x012ff826",
                    "C": "0xaf3ef484",
                    "D": "0x3e89c9d1"
                },
                "48th Loop": {
                    "A": "0x3e89c9d1",
                    "B": "0xc2b51a69",
                    "C": "0x012ff826",
                    "D": "0xaf3ef484"
                },
                "49th Loop": {
                    "A": "0xaf3ef484",
                    "B": "0xa0dc8abc",
                    "C": "0xc2b51a69",
                    "D": "0x012ff826"
                },
                "50th Loop": {
                    "A": "0x012ff826",
                    "B": "0xc026f62b",
                    "C": "0xa0dc8abc",
                    "D": "0xc2b51a69"
                },
                "51th Loop": {
                    "A": "0xc2b51a69",
                    "B": "0xc511f92e",
                    "C": "0xc026f62b",
                    "D": "0xa0dc8abc"
                },
                "52th Loop": {
                    "A": "0xa0dc8abc",
                    "B": "0x96b7bbd9",
                    "C": "0xc511f92e",
                    "D": "0xc026f62b"
                },
                "53th Loop": {
                    "A": "0xc026f62b",
                    "B": "0xdbee0424",
                    "C": "0x96b7bbd9",
                    "D": "0xc511f92e"
                },
                "54th Loop": {
                    "A": "0xc511f92e",
                    "B": "0x0b73a052",
                    "C": "0xdbee0424",
                    "D": "0x96b7bbd9"
                },
                "55th Loop": {
                    "A": "0x96b7bbd9",
                    "B": "0xefb39920",
                    "C": "0x0b73a052",
                    "D": "0xdbee0424"
                },
                "56th Loop": {
                    "A": "0xdbee0424",
                    "B": "0x973149ce",
                    "C": "0xefb39920",
                    "D": "0x0b73a052"
                },
                "57th Loop": {
                    "A": "0x0b73a052",
                    "B": "0x2f2228ab",
                    "C": "0x973149ce",
                    "D": "0xefb39920"
                },
                "58th Loop": {
                    "A": "0xefb39920",
                    "B": "0x36700e7e",
                    "C": "0x2f2228ab",
                    "D": "0x973149ce"
                },
                "59th Loop": {
                    "A": "0x973149ce",
                    "B": "0x83d1ca84",
                    "C": "0x36700e7e",
                    "D": "0x2f2228ab"
                },
                "60th Loop": {
                    "A": "0x2f2228ab",
                    "B": "0xe1b583a8",
                    "C": "0x83d1ca84",
                    "D": "0x36700e7e"
                },
                "61th Loop": {
                    "A": "0x36700e7e",
                    "B": "0x26145aec",
                    "C": "0xe1b583a8",
                    "D": "0x83d1ca84"
                },
                "62th Loop": {
                    "A": "0x83d1ca84",
                    "B": "0x733e1204",
                    "C": "0x26145aec",
                    "D": "0xe1b583a8"
                },
                "63th Loop": {
                    "A": "0xe1b583a8",
                    "B": "0x6da473b1",
                    "C": "0x733e1204",
                    "D": "0x26145aec"
                },
                "64th Loop": {
                    "A": "0x26145aec",
                    "B": "0x78227d11",
                    "C": "0x6da473b1",
                    "D": "0x733e1204"
                }
            },
            "1st Loop": {
                "A": "0x27379f53",
                "B": "0x01ce4a01",
                "C": "0x6b812e4a",
                "D": "0xe9924414"
            }
        },
        "4th Step Final": {
            "Update_A": "0x27379f53",
            "Update_B": "0x01ce4a01",
            "Update_C": "0x6b812e4a",
            "Update_D": "0xe9924414"
        },
        "5th Step": "0x27379f5301ce4a016b812e4ae9924414"
    }
}

    df = json_to_multiindex_df_same_depth(data)

    # ✅ 인덱스를 위쪽 MultiIndex로 올린 버전
    df_top = index_to_column_multiindex(df)

    # 엑셀로 저장
    df_top.to_excel("md5_log_same_depth_top_index.xlsx")