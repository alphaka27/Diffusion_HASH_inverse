"""
CSV file formatter
"""
import re
import pandas as pd
import numpy as np

try:
    from diffusion_hash_inv.utils import OutputFormat
except ImportError as e:
    print(f"Error importing OutputFormat: {e}")

K = np.array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
], dtype=np.uint32)

# Initial Hash value
INIT_HASH = np.array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
], dtype=np.uint32)

def _ordinal_to_index(s: str) -> int:
    """
    "1st round" → 0, "12th loop" → 11
    """
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) - 1 if m else 0

# pylint: disable=missing-class-docstring, missing-function-docstring, too-many-locals
class CSVFormat:
    def __init__(self):
        pass

    # (A) OutputFormat.to_dict() → csv_dict (라운드/스텝만 추림)
    def csv_dict(self, step_logs: dict) -> dict:
        # Step1: 리스트 64개(이미 hex 문자열)  # ok
        # Step2: dict {a..h: hex}               # ok
        # Step3: dict {"1st loop": {...}, ...}  # ok
        # Step4: dict {a..h: hex}               # ok
        out = {}
        out["input"] = step_logs["Message"]

        # 초기 해시(H0)
        out["pre_hash"] = {f"H0_{i}": OutputFormat.to_hex32_scalar(np.uint32(h))
                        for i, h in enumerate(INIT_HASH)}

        # 상수 K
        out["constant"] = {f"K{i}": OutputFormat.to_hex32_scalar(np.uint32(k))
                        for i, k in enumerate(K)}

        # Preprocess (이미 hex 문자열로 저장되어 있음)
        # step_logs["Rounds logs"]["Preprocess"] = {"Block 0":[hex..]*16, ...}
        preproc = step_logs["Rounds logs"].get("Preprocess", {})
        out["pre_message"] = {}
        for blk_name, words in preproc.items():
            out["pre_message"][blk_name] = {f"M{i}": words[i] for i in range(min(16, len(words)))}

        # 각 라운드: "1st round", "2nd round", ...
        rounds = {k: v for k, v in step_logs["Rounds logs"].items() if k != "Preprocess"}
        # 인덱스 정렬
        rk_sorted = sorted(rounds.keys(), key=_ordinal_to_index)

        for rk in rk_sorted:
            rlog = rounds[rk]
            rkey_norm = f"Round {_ordinal_to_index(rk)}"   # Round 0, Round 1, ...
            out[rkey_norm] = {}

            # Step1: 리스트 64개(이미 hex 문자열)
            s1 = rlog["Message Schedule(Step1)"]
            out[rkey_norm]["step1"] = s1

            # Step2: dict {a..h: hex}
            out[rkey_norm]["step2"] = rlog["Initialize working variables(Step2)"]

            # Step3: dict {"1st loop": {...}, ...}
            s3 = rlog["Main Compute Function loops(Step3)"]
            # 정렬된 (0..63) 인덱스로 변환
            items = sorted(s3.items(), key=lambda kv: _ordinal_to_index(kv[0]))
            # 표준화된 인덱스 문자열로 저장: {"0": {...}, "1": {...}}
            s3_std = {str(i): d for i, (_, d) in enumerate(items)}
            out[rkey_norm]["step3"] = s3_std

            # Step4: dict {a..h: hex}
            out[rkey_norm]["step4"] = rlog["Finalize the hash value(Step4)"]

        return out

    # (B) 3-레벨 컬럼(공통 + 라운드별)
    def make_columns_roundwise(self, num_rounds: int) -> pd.MultiIndex:
        cols = []
        # 공통(라운드 무관)
        cols += [
            ("", "meta", "nbr"),
            ("", "meta", "input"),
        ]
        for i in range(16):
            cols.append(("", "pre_message", f"M{i}"))

        for i in range(8):
            cols.append(("", "pre_hash",   f"H0_{i}"))

        for i in range(64):
            cols.append(("", "constant",   f"K{i}"))

        # 라운드별
        for r in range(num_rounds):
            _r = f"Round {r}"
            # cols.append((_r, "meta", "id"))
            for i in range(64):
                cols.append((_r, "step1", f"W{i}"))
            for ch in "abcdefgh":
                cols.append((_r, "step2", ch))
            for i in range(64):
                base = f"step3_iter{i}"
                cols += [(_r, base, "t1"), (_r, base, "t2")]
                for ch in "abcdefgh":
                    cols.append((_r, base, ch))
            for j in range(8):
                cols.append((_r, "step4", f"H1_{j}"))

        return pd.MultiIndex.from_tuples(cols, names=["round", "stage", "field"])

    # (C) csv_dict → 1행(dict) 평탄화
    def row_from_csv_dict(self, csv_result: dict, nbr: int) -> tuple[dict, int]:
        row = {("", "meta", "nbr"): nbr, ("", "meta", "input"): csv_result.get("input", "")}

        # pre_message: Block 0만 공통 영역으로
        pm = csv_result.get("pre_message", {}).get("Block 0", {})
        for i in range(16):
            row[("", "pre_message", f"M{i}")] = pm.get(f"M{i}", "")

        # pre_hash / constant
        ph = csv_result.get("pre_hash", {})
        for i in range(8):
            row[("", "pre_hash", f"H0_{i}")] = ph.get(f"H0_{i}", "")
        consts = csv_result.get("constant", {})
        for i in range(64):
            row[("", "constant", f"K{i}")] = consts.get(f"K{i}", consts.get(f"k{i}", ""))

        # 라운드 목록
        rkeys = sorted([k for k in csv_result.keys() if k.startswith("Round ")],
                    key=lambda s: int(s.split()[-1]))
        for ridx, rk in enumerate(rkeys):
            _r = f"Round {ridx}"
            rlog = csv_result[rk]
            # row[(_r, "meta", "id")] = ridx

            # step1
            for i, w in enumerate(rlog.get("step1", [])[:64]):
                row[(_r, "step1", f"W{i}")] = w

            # step2
            s2 = rlog.get("step2", {})
            for ch in "abcdefgh":
                row[(_r, "step2", ch)] = s2.get(ch, "")

            # step3
            s3 = rlog.get("step3", {})
            items = sorted(((int(k), v) for k, v in s3.items()), key=lambda x: x[0])
            for i, d in items[:64]:
                base = (_r, f"step3_iter{i}")
                row[base + ("t1",)] = d.get("t1", "")
                row[base + ("t2",)] = d.get("t2", "")
                for ch in "abcdefgh":
                    row[base + (ch,)] = d.get(ch, "")

            # step4
            s4 = rlog.get("step4", {})
            for j, ch in enumerate("abcdefgh"):
                row[(_r, "step4", f"H1_{j}")] = s4.get(ch, "")

        return row, len(rkeys)

    # (D) 맨 위 요약 행
    def build_top_round_row(self, num_rounds: int) -> dict:
        top = {("", "meta", "nbr"): "round_meta", ("", "meta", "input"): ""}
        for i in range(16):
            top[("", "pre_message", f"M{i}")] = ""
        for i in range(8):
            top[("", "pre_hash",   f"H0_{i}")] = ""
        for i in range(64):
            top[("", "constant",   f"K{i}")] = ""
        # for r in range(num_rounds):
        #     top[(f"Round {r}", "meta", "id")] = r
        return top

    def df_accumulate(self, df, logs: dict, iteration: int) -> pd.DataFrame:
        """
        df: 기존 DataFrame 또는 None
        logs: OutputFormat.to_dict() 결과(한 번의 실행 로그)
        iteration: 0-based 혹은 1-based? -> 여기서는 0-based 가정, 저장은 +1로 nbr 표기
        csv_fmt: CSVFormat 인스턴스

        반환: 갱신된 DataFrame
        """
        # 1) logs -> csv_dict -> 단일 행(row) + 라운드 개수(rn)
        csv_result = self.csv_dict(step_logs=logs)
        row, rn = self.row_from_csv_dict(csv_result, iteration + 1)  # nbr=iteration+1

        # 2) 최초 호출이면 df 생성 + top row 삽입
        if df is None:
            cols = self.make_columns_roundwise(rn)
            df = pd.DataFrame(columns=cols)


        existing_rounds = sorted(
            {int(str(r).split()[-1]) for r, _, _ in df.columns if str(r).startswith("Round ")},
            key=int
        )
        current_rn = (existing_rounds[-1] + 1) if existing_rounds else 0

        if rn > current_rn:
            new_cols = self.make_columns_roundwise(rn)
            df = df.reindex(columns=new_cols)

        # 4) 새 행 추가
        df.loc[len(df)] = [row.get(c, "") for c in df.columns]
        return df
