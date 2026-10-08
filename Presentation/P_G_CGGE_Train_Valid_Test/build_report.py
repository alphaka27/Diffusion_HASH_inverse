"""Build a read-only P-G-CGGE report from the completed V6 archive."""
import base64
import hashlib
import html
import json
from pathlib import Path
import struct
import zlib

import numpy as np

from dhi_v6 import codecs, data
from dhi_v6.protocol import canonical, file_hash, read_json


OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
ROOT = REPO / "local_experiment_archive/runs/v6-study-r2"
PIPELINE = "P-G-CGGE"
RECORD = np.dtype([("payload", "u1", (31,)), ("length", "u1"), ("flags", "u1"),
                   ("margin", "<f2"), ("reserved", "u1")], align=False)
assert RECORD.itemsize == 36
sources = {}
checks = []
samples = []
training = []
evaluations = []
diagnostics = []


def record_source(path, expected=None):
    path = Path(path)
    digest = file_hash(path)
    if expected is not None:
        assert digest == expected, f"Source checksum mismatch: {path}"
    sources[str(path.relative_to(REPO))] = digest


def read(path, expected=None):
    record_source(path, expected)
    return read_json(path)


def write_png(path, channel):
    pixels = np.rint((channel + 1) * 127.5).astype(np.uint8)
    pixels = pixels.repeat(5, axis=0).repeat(5, axis=1)
    height, width = pixels.shape
    raw = b"".join(b"\0" + row.tobytes() for row in pixels)
    def chunk(kind, value):
        return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value))
    encoded = zlib.compress(raw)
    assert zlib.decompress(encoded) == raw
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", encoded) + chunk(b"IEND", b""))


def add_sample(stage, split, seed, message, target, provenance, stored=None):
    identifier = f"{stage}_{split}_s{seed}_{len(samples):02d}"
    payload = np.zeros((1, 31), dtype=np.uint8)
    payload[0, :len(message)] = np.frombuffer(message, dtype=np.uint8)
    lengths = np.array([len(message)], dtype=np.int32)
    encoded = codecs.encode(payload, lengths, "cgge", "P")
    assert encoded.shape == (1, 2, 32, 64)
    assert codecs.decode(encoded, lengths, "cgge", "P")[0] == [message]
    assert codecs.strict_decode(encoded, lengths, "cgge", "P") == [message]
    digest = hashlib.md5(message).hexdigest() if stage == "C" else data.digest_reference(message, 4).hex()
    window = "W3" if stage == "C" else "W1"
    actual = data.window_value(bytes.fromhex(digest), window)
    if split != "Test":
        assert actual == target
    else:
        assert bool(stored["hit"]) == (actual == target)
    paths = []
    for channel, name in enumerate(["data", "mask"]):
        path = OUT / "images" / f"{identifier}_{name}.png"
        write_png(path, encoded[0, channel])
        paths.append(str(path.relative_to(OUT)))
    samples.append({
        "id": identifier, "stage": stage, "split": split, "seed": seed, "method": "Main",
        "target_hex": f"{target:03X}", "window": window, "rung": 64 if stage == "C" else 4,
        "original_message": message.decode("ascii") if split != "Test" else None,
        "original_digest_hex": digest if split != "Test" else None,
        "model_image": None,
        "decoded_message": message.decode("ascii") if split == "Test" else None,
        "decoded_digest_hex": digest if split == "Test" else None,
        "message_hex": message.hex(), "length": len(message), "actual_window_hex": f"{actual:03X}",
        "image_kind": "decoded_message_reencoded" if split == "Test" else "reconstructed_input_encoded",
        "data_image": paths[0], "mask_image": paths[1], "provenance": provenance, "stored": stored,
    })


(OUT / "images").mkdir(exist_ok=True)
frozen = read(ROOT / "protocol.frozen.json")
for name in ["__init__.py", "data.py", "codecs.py", "protocol.py", "runtime.py"]:
    record_source(REPO / "src/dhi_v6" / name, frozen["source"][name])
assert np.__version__ == frozen["environment"]["packages"]["numpy"]
final_report = read(ROOT / "decision.json")
assert final_report["status"] == "TERMINAL"
decision = final_report["pipelines"][PIPELINE]
c_result = read(ROOT / "C.json")
p_result = read(ROOT / "P.json")["pipelines"][PIPELINE]
assert not c_result["budget_stop"]
checks.append("입력 재구성에 사용한 소스 5개와 NumPy 버전이 동결 기록과 일치")

for stage, window, rung, seeds in [("A-Q", "W1", 64, range(3)), ("C", "W3", 64, range(3)), ("P", "W1", 4, range(1))]:
    groups = data.synthetic_split() if stage == "A-Q" else read(ROOT / stage / f"groups-{window}-r{rung}.json")
    assert set(groups["train"]).isdisjoint(groups["validation"])
    assert set(groups["test"]).isdisjoint(groups["train"] + groups["validation"])
    for seed in seeds:
        folder = ROOT / stage / "runs" / PIPELINE / f"Main-{seed}/u40000"
        contract = read(folder / "contract.json")
        complete = read(folder / "complete.json")
        assert contract["groups_sha256"] == hashlib.sha256(canonical(groups)).hexdigest()
        losses = []
        updates = []
        for name, digest in sorted(complete["segments"].items()):
            record_source(folder / name, digest)
            with np.load(folder / name, allow_pickle=False) as segment:
                losses.extend(segment["loss"].tolist())
                updates.extend(segment["update_id"].tolist())
        assert updates == list(range(1, 40001)) and np.isfinite(losses).all()
        values = []
        for name, digest in sorted(complete["diagnostics"].items()):
            diag = read(folder / name, digest)
            assert len(diag["clp_differences"]) == contract["diagnostic_pairs"] == 256
            row = {"stage": stage, "seed": seed, "update": diag["update"],
                   "train_segment_mean": float(np.mean(losses[diag["update"] - 4000:diag["update"]])),
                   "validation_loss": diag["validation_loss"],
                   "validation_clp_mean": float(np.mean(diag["clp_differences"]))}
            diagnostics.append(row)
            values.append(row)
        training.append({"stage": stage, "seed": seed, "updates": complete["updates"],
                         "pairs": complete["work"]["training_pairs"],
                         "train_first1000": float(np.mean(losses[:1000])),
                         "train_last4000": float(np.mean(losses[-4000:])),
                         "train_last": losses[-1], "valid_first": values[0]["validation_loss"],
                         "valid_last": values[-1]["validation_loss"],
                         "valid_clp_last": values[-1]["validation_clp_mean"]})
        if stage == "A-Q":
            qualification = read(folder / "qualification.json")
            evaluations.append({"stage": stage, "seed": seed, "qualification": qualification})
            continue
        payload, lengths, labels, _ = data.fresh_batch((stage, "P", seed), 39999, 256, groups["train"],
                                                       task="md5", window=window, rung=rung)
        with np.load(folder / "segments/seg-00040000.npz", allow_pickle=False) as segment:
            digest = hashlib.sha256(payload.tobytes() + lengths.tobytes() + labels.tobytes()).digest()
            assert digest == segment["data_sha256"][-1].tobytes()
        checks.append(f"{stage} seed {seed}: 마지막 Train 배치 256건의 data_sha256 일치")
        for index, message in enumerate(data.messages(payload, lengths)[:3]):
            add_sample(stage, "Train", seed, message, int(labels[index]),
                       {"update": 40000, "batch_row_1based": index + 1, "batch_sha256_verified": True,
                        "source": str((folder / "segments/seg-00040000.npz").relative_to(REPO))})
        payload, lengths, labels, _ = data.fresh_batch(("clp", stage, "validation", "P", seed),
            0, 512, groups["validation"], task="md5", window=window, rung=rung)
        assert set(labels.tolist()).issubset(groups["validation"])
        for index, message in enumerate(data.messages(payload, lengths)[:3]):
            add_sample(stage, "Valid", seed, message, int(labels[index]),
                       {"update": 40000, "probe_offset": 0, "probe_batch_size": 512,
                        "batch_row_1based": index + 1, "batch_sha256_verified": False,
                        "source": str((folder / "diagnostics/diag-00040000.json").relative_to(REPO)),
                        "note": "동결 소스와 namespace로 재구성. 원 입력 체크섬은 저장되지 않음."})
        targets = read(ROOT / stage / "trials.json")["targets"]
        eval_folder = ROOT / stage / "eval" / PIPELINE / f"Main-{seed}"
        block_results = []
        for block in ([1, 2] if stage == "C" else [1]):
            result = read(eval_folder / f"block-{block}.json")
            for name, digest in result["artifacts"].items():
                record_source(eval_folder / name, digest)
            trials = np.load(eval_folder / f"block-{block}.trials.npy", allow_pickle=False)
            assert len(trials) * 100 == result["rows"]
            for name, field in [("success_at_1", "at1"), ("success_at_10", "at10"), ("success_at_100", "at100"), ("hits", "hits")]:
                assert int(trials[field].sum()) == result[name]
            block_results.append(result)
        keys = ["rows", "hits", "success_at_1", "success_at_10", "success_at_100", "valid", "strict_valid", "duplicates", "training_matches"]
        evaluation = {"stage": stage, "seed": seed, **{k: sum(b[k] for b in block_results) for k in keys}}
        evaluation["trials"] = evaluation["rows"] // 100
        evaluations.append(evaluation)
        ledger_path = eval_folder / "block-1.bin"
        ledger = np.memmap(ledger_path, dtype=RECORD, mode="r")
        first_hit = int(np.flatnonzero(ledger["flags"] & 2)[0])
        trial_start = first_hit // 100 * 100
        first_miss = trial_start + int(np.flatnonzero((ledger["flags"][trial_start:trial_start + 100] & 2) == 0)[0])
        for index in [first_miss, first_hit]:
            row = ledger[index]
            flags = int(row["flags"])
            message = row["payload"][:int(row["length"])].tobytes()
            assert flags & 1
            add_sample(stage, "Test", seed, message, targets[index // 100],
                       {"block": 1, "ledger_row_1based": index + 1, "trial_1based": index // 100 + 1,
                        "candidate_1based": index % 100 + 1, "source": str(ledger_path.relative_to(REPO)),
                        "selection": "block 1의 첫 성공이 있는 시행에서 첫 실패 후보와 첫 성공 후보"},
                       {"hit": bool(flags & 2), "strict_valid": bool(flags & 8),
                        "training_match": bool(flags & 4), "margin_float16": float(row["margin"])})

c_eval = [e for e in evaluations if e["stage"] == "C"]
assert sum(e["rows"] for e in c_eval) == c_result["metrics"][PIPELINE]["rows"]
assert sum(e["hits"] for e in c_eval) == c_result["metrics"][PIPELINE]["hits"]
assert sum(e["success_at_100"] for e in c_eval) == sum(e["main"] for e in decision["seed_estimates"]["Random"])
checks += ["읽은 segment·진단 JSON·원장·trial 요약의 기록된 파일 체크섬 일치",
           "저장된 trial 요약의 합계와 block JSON 및 C 최종 집계 일치",
           f"표본 {len(samples)}건의 CGGE 인코딩/디코딩 round-trip 일치; 원문 또는 후보 해시 직접 계산",
           "Test 후보 재생성·모델 추론·전체 원장 재평가 없이 기존 기록만 추출"]
result = {"pipeline": PIPELINE, "run": "v6-study-r2", "scope": "Main; C 본문, P 및 A-Q 보조 결과",
          "training": training, "diagnostics": diagnostics, "evaluations": evaluations, "samples": samples,
          "checks": checks, "sources_sha256": sources}
(OUT / "records.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

md = []
parts = []


def section(title, text):
    md.extend([f"## {title}", "", text, ""])
    parts.append(f"<section><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p></section>")


def table(headers, rows):
    md.extend(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)])
    md.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    md.append("")
    parts.append("<div class='scroll'><table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
                 + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows)
                 + "</tbody></table></div>")


def pct(n, d):
    return f"{n / d * 100:.4f}%"


title = "P-G-CGGE · Train / Valid / Test 실험 결과"
md.extend([f"# {title}", "", "작성일: 2026-10-08 · 완료 실행: v6-study-r2 · Main 모델만 발췌", ""])
parts.append(f"<header><div class='eyebrow'>V6 / P-G-CGGE / 2026-10-08</div><h1>{title}</h1><p>완료 실행 v6-study-r2 · G3-U · Printable ASCII 33–126 · CGGE</p></header>")
section("결과 요약", "주 실험 C(W3, 정규 MD5 64단계)의 Success@100은 1,224 / 49,152 = 2.4902%다. "
        "Train 손실은 감소했지만 Test의 조건 활용 신호는 확인되지 않았다(CLP z=0.1215). "
        "등록된 +0.5%p 이상의 성공률 개선은 배제되었다(REJECTED_BOUNDED). 이 실험은 원문 복원이나 전체 MD5 128비트 일치를 측정하지 않는다.")
section("요청 항목의 보존 상태", "원본 메시지 → CGGE 이미지 → 원본 해시 → 모델 추론 이미지 → 디코딩 메시지 → 디코딩 해시 순서로 확인했다. "
        "모든 단계의 추론 원본 이미지는 미저장 상태다. 없는 결과를 복원 결과로 만들지 않았다.")
table(["항목", "Train", "Valid", "Test"], [
    ["원본 메시지", "결정적 배치 재구성 + 체크섬 검증", "결정적 진단 입력 재구성", "해당 없음: 목표 12비트를 직접 추출"],
    ["원본 인코딩 이미지", "재구성 원문을 실제 인코더로 변환", "재구성 원문을 실제 인코더로 변환", "대응 원문이 없어 해당 없음"],
    ["원본 메시지 해시", "직접 계산", "직접 계산", "대응 원문이 없어 해당 없음"],
    ["모델 추론 원본 이미지", "미저장", "미저장", "미저장"],
    ["모델 출력의 디코딩 메시지", "미저장", "미저장", "기존 36-byte 원장에서 추출"],
    ["디코딩 메시지 해시", "계산할 출력 메시지 없음", "계산할 출력 메시지 없음", "원장의 기존 메시지로 직접 계산"],
])
section("1. Train · Valid — 주 실험 C", "해시 그룹 수는 Train 2,816 / Valid 256 / Test 1,024이며 서로 겹치지 않는다. "
        "seed마다 40,000 updates × batch 256 = 10,240,000 학습 메시지를 사용했다. "
        "Valid는 4,000 update마다 256쌍(512개 메시지)을 같은 결정적 입력으로 진단했다. "
        "손실은 길이 예측 cross-entropy와 활성 픽셀 MSE의 합이다. Train은 한 번의 잡음 추출, Valid는 8회 평균이므로 단순한 원문 복원 정확도가 아니다.")
table(["seed", "Train 처음 1,000 평균", "Train 마지막 4,000 평균", "Valid @4,000", "Valid @40,000", "Valid CLP 평균 @40,000"],
      [[r["seed"], f'{r["train_first1000"]:.6f}', f'{r["train_last4000"]:.6f}', f'{r["valid_first"]:.6f}',
        f'{r["valid_last"]:.6f}', f'{r["valid_clp_last"]:+.6f}'] for r in training if r["stage"] == "C"])
section("2. Test — 주 실험 C", "각 seed는 16,384 시행, 시행당 후보 100개를 평가했다. 이미지 생성은 25단계이며 평가 batch는 256이다. "
        "Success@k는 처음 k개 후보 안에서 목표 W3와 일치한 메시지를 하나 이상 찾은 시행 비율이다. 성공 후보 수와 성공 시행 수는 다르다.")
table(["seed", "시행 수", "후보 수", "Success@1", "Success@10", "Success@100", "성공 후보"],
      [[r["seed"], f'{r["trials"]:,}', f'{r["rows"]:,}', pct(r["success_at_1"], r["trials"]),
        pct(r["success_at_10"], r["trials"]), f'{r["success_at_100"]:,} / {r["trials"]:,} ({pct(r["success_at_100"],r["trials"])})',
        f'{r["hits"]:,}'] for r in c_eval])
total = {k: sum(e[k] for e in c_eval) for k in ["rows", "trials", "hits", "success_at_1", "success_at_10", "success_at_100", "valid", "strict_valid", "training_matches", "duplicates"]}
table(["집계 항목", "결과"], [["총 후보 / 시행", f'{total["rows"]:,} / {total["trials"]:,}'],
    ["Success@1 / @10 / @100", " / ".join(pct(total[k], total["trials"]) for k in ["success_at_1", "success_at_10", "success_at_100"])],
    ["성공 후보 / 성공 시행", f'{total["hits"]:,} / {total["success_at_100"]:,}'],
    ["Prototype decode 유효율", pct(total["valid"], total["rows"])],
    ["Strict decode 유효율 (저장된 진단 flag)", pct(total["strict_valid"], total["rows"])],
    ["학습 메시지와 일치한 후보", f'{total["training_matches"]:,} ({pct(total["training_matches"], total["rows"])})'],
    ["시행 안 중복 후보", str(total["duplicates"])],
    ["Test CLP", "196,608쌍; z=0.121517; INFO_64=false"],
    ["대조군 대비 개선의 상한", "Random 대비 +0.305097%p; Shuffled 대비 +0.363590%p"],
    ["최종 판정", "REJECTED_BOUNDED; +0.5%p 이상 개선 배제, 효과가 정확히 0이라는 뜻은 아님"],
])
section("3. 메시지 · 이미지 · 해시 대응표", "Train은 마지막 학습 배치의 첫 3건, Valid는 마지막 진단 입력의 첫 3건을 seed마다 제시한다. "
        "Test는 block 1의 첫 성공 시행에서 실패 후보와 성공 후보를 하나씩 추출했다. "
        "Test 사례는 성공과 실패를 보여주기 위한 선택 예시이므로 성공률 추정용 표본이 아니다. "
        "검정=-1, 흰색=+1이며 왼쪽은 내용 채널, 오른쪽은 활성 위치 mask다. 원 배열은 2×32×64이고 보간 없이 확대 표시했다. "
        "Test 그림은 저장된 디코딩 메시지의 재인코딩 결과이며 당시 모델 추론 이미지가 아니다. "
        "문자열은 JSON 표기로 표시하며, 이스케이프 없이 실제 바이트를 확인할 수 있도록 hex도 병기했다.")


def sample_cards(stage):
    for split in ["Train", "Valid", "Test"]:
        subset = [s for s in samples if s["stage"] == stage and s["split"] == split]
        parts.append(f"<details {'open' if stage == 'C' else ''}><summary>{stage} · {split} — {len(subset)}개 사례</summary>")
        for s in subset:
            test = split == "Test"
            message = s["decoded_message"] if test else s["original_message"]
            digest = s["decoded_digest_hex"] if test else s["original_digest_hex"]
            image_label = "디코딩 메시지의 재인코딩 — 추론 원본 아님" if test else "원본 메시지의 CGGE 인코딩 — 입력 재구성"
            fields = [("원본 메시지", "해당 없음: 목표 해시를 직접 추출한 시행" if test else json.dumps(message, ensure_ascii=False)),
                      ("원본 메시지 해시", "해당 없음" if test else digest),
                      ("모델 추론 원본 이미지", "미저장"),
                      ("디코딩 메시지", json.dumps(message, ensure_ascii=False) if test else "미저장"),
                      ("디코딩 메시지 해시", digest if test else "계산할 출력 메시지 없음"),
                      ("조건 / 메시지의 window 값", f'{s["window"]}: 목표 {s["target_hex"]} / 메시지 {s["actual_window_hex"]}'),
                      ("바이트 길이 / hex", f'{s["length"]} / {s["message_hex"]}')]
            if test:
                fields += [("원장 기록", f'시행 {s["provenance"]["trial_1based"]}, 후보 {s["provenance"]["candidate_1based"]}; '
                            f'{"성공" if s["stored"]["hit"] else "실패"}; strict={s["stored"]["strict_valid"]}')]
            else:
                fields += [("입력 위치", f'update 40,000, 배치 {s["provenance"]["batch_row_1based"]}번째'),
                           ("재구성 확인", "배치 SHA-256 일치" if split == "Train" else "동결 코드·난수 namespace 일치; 원 입력 체크섬 미저장")]
            heading = f'{stage} · {split} · seed {s["seed"]} · {s["id"]}'
            md.extend([f"### {heading}", ""] + [f"- {k}: <code>{html.escape(v)}</code>" for k, v in fields] + ["", image_label, "",
                f'![내용 채널]({s["data_image"]})', f'![활성 위치 mask]({s["mask_image"]})', "",
                f'근거: `{s["provenance"]["source"]}`', ""])
            parts.append(f"<article><h3>{html.escape(heading)}</h3><dl>" + "".join(f"<dt>{html.escape(k)}</dt><dd><code>{html.escape(v)}</code></dd>" for k, v in fields)
                         + f"</dl><div class='image-label'>{html.escape(image_label)}</div><div class='images'>")
            for key, label in [("data_image", "내용 채널"), ("mask_image", "활성 위치 mask")]:
                encoded = base64.b64encode((OUT / s[key]).read_bytes()).decode()
                parts.append(f"<figure><img width='320' height='160' alt='{label}' src='data:image/png;base64,{encoded}'><figcaption>{label} · 원본 64×32</figcaption></figure>")
            parts.append(f"</div><p class='source'>근거: {html.escape(s['provenance']['source'])}</p></article>")
        parts.append("</details>")


sample_cards("C")
section("4. 보조 실험 — P 및 A-Q", "아래 결과도 P-G-CGGE만 발췌했다. P는 4단계로 약화한 MD5(W1)이고, A-Q는 메시지 첫 3개의 대문자 16진수 문자를 조건으로 삼는 synthetic 과제다. "
        "세 과제는 서로 다른 모델을 학습했으며 C의 정규 MD5 결과와 합산하지 않는다. P 사례의 32자리 해시는 4단계 함수의 출력으로 표준 MD5와 다르다.")
table(["과제 / seed", "Train 마지막 4,000 평균", "Valid @40,000", "Valid CLP 평균 @40,000"],
      [[f'{r["stage"]} / {r["seed"]}', f'{r["train_last4000"]:.6f}', f'{r["valid_last"]:.6f}', f'{r["valid_clp_last"]:+.6f}'] for r in training if r["stage"] != "C"])
p_eval = next(r for r in evaluations if r["stage"] == "P")
table(["과제", "평가 결과"], [
    ["P / W1 / r=4", f'Success@100: {p_eval["success_at_100"]:,} / {p_eval["trials"]:,} = {pct(p_eval["success_at_100"],p_eval["trials"])}; 후보 {p_eval["rows"]:,}개'],
    ["P 조건 신호", f'CLP z={p_result["clp"]["z"]:.6f}; GEN_4=true; INFO_4=true'],
    ["A-Q synthetic / 각 seed", "정상 조건 512/512, 반전 조건 512/512; 기존 조건으로의 오성공 0"],
    ["A-Q CLP z / seed 0, 1, 2", ", ".join(f'{e["qualification"]["clp"]["z"]:.6f}' for e in evaluations if e["stage"] == "A-Q")],
])
sample_cards("P")
section("5. 검증 · 해석 범위", "본문 수치는 저장된 집계를 사용했다. 예시 해시는 표시된 메시지에서 직접 계산했다. "
        "추론 이미지 미저장은 명세 §5.5와 runtime의 저장 형식에서 확인했다. "
        "특히 Test의 원문은 원래 실험 설계에 없으므로 원문/추론 이미지의 쌍별 비교를 만들 수 없다. "
        "Valid 입력은 재구성했으나 메시지 자체의 저장 체크섬이 없어 Train과 같은 수준의 대조 검증을 주장하지 않는다.")
for check in checks:
    md.append(f"- {check}")
parts.append("<ul>" + "".join(f"<li>{html.escape(c)}</li>" for c in checks) + "</ul>")
md += ["", "데이터: `records.json` (표본 32건, checkpoint별 Train/Valid 진단 70행, 출처 파일 SHA-256 포함).", "",
       "원본 archive·실험 코드·기존 자료는 수정하지 않았다. 모델 추론, 재학습, Metal 테스트, 금지된 Test 재평가는 실행하지 않았다.", ""]
parts.append("<footer>표본·진단·출처 파일 SHA-256: records.json · 빌드: build_report.py<br>모델 추론·재학습·Metal 테스트·전체 Test 재평가 미실행. 원본 archive 수정 없음.</footer>")
(OUT / "REPORT_KO.md").write_text("\n".join(md))
style = """
*{box-sizing:border-box}body{font:16px/1.7 -apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo',sans-serif;margin:0;color:#243149;background:#f4f6f9}
main{max-width:1180px;margin:auto;padding:48px 32px}header{background:#172c48;color:white;border-radius:18px;padding:36px;margin-bottom:36px}
h1{font-size:34px;line-height:1.35;margin:8px 0 16px}h2{font-size:25px;margin-top:42px}h3{font-size:18px;margin:0 0 18px}.eyebrow{letter-spacing:.12em;font-size:13px;color:#acd9e0}
table{border-collapse:collapse;background:white;width:100%;font-size:14px;margin:18px 0 26px}th,td{padding:12px 14px;border:1px solid #dce2eb;text-align:left}th{background:#e8eef5;white-space:nowrap}.scroll{overflow:auto}
article{padding:26px;background:#fff;border:1px solid #dae1eb;border-radius:12px;margin:20px 0}details{margin:18px 0}summary{cursor:pointer;font-size:20px;font-weight:650;padding:14px;background:#e7edf4;border-radius:8px}
dl{display:grid;grid-template-columns:225px 1fr;gap:7px 18px;margin:0 0 20px}dt{font-weight:600}dd{margin:0;overflow-wrap:anywhere}code{font:14px/1.6 ui-monospace,SFMono-Regular,monospace;white-space:pre-wrap}
.images{display:flex;flex-wrap:wrap;gap:28px}figure{margin:12px 0}img{image-rendering:pixelated;max-width:100%;height:auto;border:1px solid #b6c1cf}figcaption,.source,footer{font-size:13px;color:#60718a}.source{overflow-wrap:anywhere}
.image-label{font-weight:650;color:#a25215;padding-top:8px}footer{border-top:1px solid #ccd5e2;margin-top:40px;padding-top:22px}
@media(max-width:680px){main{padding:20px 14px}header{padding:24px}h1{font-size:27px}dl{grid-template-columns:1fr;gap:3px}dd{margin-bottom:12px}}
@media print{body{background:white}main{padding:0}article{break-inside:avoid}summary{background:#eee}}
"""
(OUT / "REPORT_KO.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
                                   + f"<title>{title}</title><style>{style}</style><main>" + "".join(parts) + "</main></html>")
assert len(samples) == 32 and len(diagnostics) == 70
assert len(list((OUT / "images").glob("*.png"))) == 64
print(json.dumps({"output": str(OUT), "samples": len(samples), "diagnostics": len(diagnostics),
                  "source_files": len(sources), "checks": checks, "C": total}, ensure_ascii=False, indent=2))
