# V6 구현 세부 명세

**Protocol:** `dhi-v6-20260929` · **작성일:** 2026-09-29 KST · **상태:** `SPEC_DRAFT` — 구현 전 명세다. V6 코드와 실행 산출물은 아직 없다.

- **기준 문서.** [RESEARCH_PLAN_V6.md](RESEARCH_PLAN_V6.md)가 과학적 설정(질문, 대조군, δ, 판정 규칙, 자원)을 정한다. 이 문서는 그 설정을 코드로 옮기는 데 필요한 세부를 모두 정하고, 계획과 현재 구현을 비교할 때 드러난 명세 공백을 해결한다. 충돌하면 과학적 설정은 계획이, 구현 세부는 이 문서가 우선한다.
- **등록값.** [examples/v6-protocol.json](examples/v6-protocol.json)(정규화 JSON, SHA-256 `d6d2d518dbc3488d4d0b0520e9db48ece49376a269e0ab910d259d3f2cdff001`). 코드의 `registration()`은 이 JSON과 같아야 하며 테스트로 강제한다. 계획 인계 안내의 사람 결정 항목 1–5가 권장값과 다르게 정해지면 구현 전에 JSON과 hash를 먼저 갱신한다.

---

## 0. 이 문서가 확정한 것

### 0.1 명세 공백 해결

| # | 공백 | 결정 | 근거 |
|---|---|---|---|
| 1 | R-DISC token ID | payload 0–255, **PAD 256, EOS 257, MASK 258**(vocab 259). 출력층은 32×258이며 앞 256개만 sampling한다 | V5 규칙(PAD = S, EOS = S+1, MASK = S+2)과 같게 두면 D1-S 코드를 상태 수 S만 바꿔 재사용할 수 있다. v3.1의 순서(EOS 256, PAD 257)는 쓰지 않는다 |
| 2 | G3 CLP의 t 분포 | 쌍마다 draw d = 0..7에서 **t ~ U{0,…,999}**(key draw 2d), **ε ~ N(0, I)**(draw 2d+1). 한 쌍의 네 점수가 같은 (t, ε)를 쓴다 | 학습 loss의 t 분포와 같으므로, 점수가 학습 목적함수의 음수에 대한 추정치가 된다 |
| 3 | 보완 이어 학습의 봉인 | 새 run 폴더 `u80000/`의 contract에 `resume_from`(40k checkpoint 파일 hash, update 40,000)을 기록한다. 봉인된 40k checkpoint(모델과 Adam 상태)를 hash 검증 후 불러 update 40,000–79,999를 같은 stream으로 실행한다. C·P에서는 80,000 update를 처음부터 학습한다. 두 방식이 bitwise로 같은지 A-impl에서 검사한다 | 데이터, corruption, lr이 모두 update 번호만의 함수다. G3 학습이 bitwise 결정적임도 실측했다(§0.2) |
| 4 | Gaussian 원장의 batch 구성 | 생성 batch는 **64의 배수**만 허용한다. {256, 1,024, 2,048} 중 하나를 A-prof-1에서 봉인한다. G3 sampler 출력이 batch ≥ 64에서 batch 구성과 무관하게 bitwise 같은지를 A-impl 게이트로 검사한다. Scalar 참조(batch 1)는 허용오차로 비교한다 | 실측 결과, 순전파가 batch 64·256·1,024 사이에서 bitwise 같고 batch 1만 최대 1.8×10⁻⁷ 다르다 |

### 0.2 명세를 정하려고 새로 잰 값 (2026-09-29, MLX 0.32.2, Apple M3 Max, 임의 가중치, MD5 0회)

| 항목 | 결과 | 자료 |
|---|---|---|
| G3 U-Net 5 update를 같은 상태에서 두 번 학습(BGV, CGGE) | 파라미터 bitwise 동일 | `determinism_probe.json`(로컬) |
| G3 순전파, batch 64·256 대 1,024 | bitwise 동일 | 같음 |
| G3 순전파, batch 1 대 1,024 | 최대 절대차 1.8×10⁻⁷ | 같음 |
| 같은 batch 반복 순전파 | bitwise 동일 | 같음 |
| V5 학습 메시지 SQLite 조회 / NumPy `searchsorted` | 건당 6.3–8.3 µs / 약 13 ns | 계획 비교 작업 중 측정(파일 미보존) |
| V5 방식의 update당 SQLite 동기 commit | 3.9 ms(빈 DB) | 같음 |
| CGGE 글꼴 표 SHA-256 | `6ef6d0bf…ed50a`(v3.1 등록값과 일치) | v3.1 `encoding/cgge.py` |
| Prototype 최소 간격 | BGV(Printable) 1 bit, CGGE MSE 1/64('I'와 'l', 1픽셀) | 같음 |
| V5 A-dev가 고른 D1-T lr | 1e-3. 따라서 D1-T-L은 1e-3 × 192/256 = 7.5e-4 | V5 `A-dev.json` |

### 0.3 계획 대비 구현 세부 보완 (과학적 설정은 바꾸지 않음)

| 항목 | 계획 | 이 명세 |
|---|---|---|
| A-prof 순서 | A-prof 1회 | **A-prof-1**(A-Q 전: 학습 속도, burst 생성, batch 선택)과 **A-prof-2**(A-dev 후: 선택된 S_G로 25분 지속 측정)로 나눈다. 예산 봉인은 A-prof-2 값을 쓴다 |
| 보완 후 S_G | 명시 없음 | 보완된 Gaussian 파이프라인은 80k seed-0 checkpoint로 A-dev 규칙을 다시 적용한다 |
| Gaussian batch 게이트 | decode 99.9% 일치 | batch ≥ 64에서 bitwise 일치로 강화한다. Batch 1 참조만 허용오차를 둔다 |
| 후보 원장 | chunk binary | 36-byte 고정 레코드, 블록 파일, commit JSON(§10) |
| 학습 digest | 메모리 정렬 배열 | stream 단위 digest segment 파일. 같은 source 파이프라인이 공유하고 교차 검증한다(§7.6) |
| 재생성 감사 | 무작위 1% 후보 | 결정적으로 고른 1% 후보를 64개씩 묶어 재생성하고 bitwise로 비교한다(§10.5) |
| 중복 정의 | trial 내 duplicate | 같은 trial의 앞선 attempt와 (길이, payload)가 같은 후보(§10.4) |
| 사람 cap 결정 | 멈추고 묻는다 | `halt.json`과 `approve-caps` 명령. MD5 조건 데이터 생성 전에만 허용한다(§13.3) |
| 다음 블록 사전 점검 | cap 초과 시 마지막 look | 직전 블록 실측 시간 × 1.2가 남은 C cap보다 크면 다음 블록을 시작하지 않는다(§12.10) |
| Stage S lr | 명시 없음 | 7.5e-4(V5 규칙) |
| 블라인드 | 계속/중단만 표시 | look 파일은 `C/looks/`에 봉인한다. CLI와 `status`는 진행과 자원만 출력한다. 절차적 블라인드이며 보고서에 기록한다 |

---

## 1. 패키지 구조와 의존 규칙

| 모듈 | 책임 |
|---|---|
| `src/dhi_v6/__init__.py` | `PROTOCOL = "dhi-v6-20260929"`, `MASTER_SEED = 2026092907` |
| `protocol.py` | `registration()`, 정규화 JSON·봉인(V5 `canonical`, `atomic_json`, `sealed_json`, `read_json`), 동결·검증, 노출 감사 v6, `Budget`, 예산 계획 |
| `data.py` | 난수 identity, source P·R, token codec, H12 해시(W1–W4), split, 학습 stream, synthetic 과제, prior 후보, key 유도, permutation, derangement |
| `codecs.py` | BGV·CGGE NumPy 인코더, G3 고정 구조, prototype decoder, 진단용 strict decoder |
| `models.py` | D1-S(P·R), D1-T-L, G3-U. 학습 step, 벡터화 sampler, scalar 참조 sampler, CLP 점수 |
| `runtime.py` | 학습 루프, checkpoint, segment 원장, 학습 digest 저장소, 평가 stream, 후보 원장, 독립 검증기, 재생성 감사, trial schedule |
| `statistics.py` | 구간, CP 한계, 순차 판정 엔진, 재현, P 검정, 대비, C4, 종합 판정, production calibration |
| `checks.py` | A-impl 게이트 |
| `study.py` | 단계 실행기, 보고서, CLI |
| `tests/test_study_v6.py` | §17의 테스트 |

- `src/dhi_v6/`는 `dhi_v5`, `diffusion_hash_inv`, `torch`를 import하지 않는다(AST 테스트). 필요한 코드는 복사해서 고친다. `dhi_v5`는 V5 봉인에 소스 hash가 기록되어 있으므로 수정하지 않는다.
- 테스트는 등가 검사를 위해 `dhi_v5`와 `diffusion_hash_inv`를 읽기 전용으로 import할 수 있다. Metal이 필요한 검사는 V5처럼 subprocess로 실행한다.
- `pyproject.toml`에 console script `hash-inverse-v6 = "dhi_v6.study:main"`을 추가한다.

## 2. 등록값

`registration()`은 [examples/v6-protocol.json](examples/v6-protocol.json)을 그대로 반환하는 dict다(키 정렬, 구분자 `,`·`:`, 끝 줄바꿈 1개). 자주 참조하는 값은 다음과 같다.

| 분류 | 값 |
|---|---|
| 공통 | K = 100, temperature 1, float32, MLX |
| Source | P: byte 33–126(94 상태), R: 0–255(256 상태), 길이 4–31 |
| Token | P: PAD 94, EOS 95, MASK 96. R: PAD 256, EOS 257, MASK 258 |
| Window | W1 `>>116`, W2 `>>0`(금지), W3 `>>52`(주), W4 `>>84`(재현) |
| Group | test 1,024 / validation 256 / train 2,816 |
| 학습 | 40,000 updates × 256, 보완 80,000, S 160,000, checkpoint 4,000마다, 진단 CLP 256쌍 |
| D1-S | lr 1e-3, warmup 없음, clip 없음, 32 intervals, NFE 33 |
| D1-T-L | lr 7.5e-4, warmup 1,000, clip 1.0 |
| G3-U | width 32, 1,000 step, β 1e-4→0.02, x0-prediction, DDIM, S_G ∈ {25, 50, 100}, lr 1e-3, warmup 1,000, clip 1.0 |
| A-Q | seeds 0–2, acceptance 512, joint ≥ 461, 오성공 ≤ 25, CLP 4,096쌍 z > 3.26 |
| A-dev | joint ≥ 231/256, 중복 허용 +0.005, 기본 100 step |
| C | seeds 0–2, 블록 8,192(fallback 6,144), look 3, α 0.05, 비율 0.1/0.4/0.5, δ 0.005, CLP 65,536쌍/seed |
| R / P / S | 16,384 trials, one-sided 0.025 / 4,096(fallback 2,048) trials, α 0.01, CLP 16,384 / D1-T-L, CLP 65,536, 4,096 trials, one-sided 0.0005 |
| 원장 | 레코드 36 B, commit 8 batch 또는 30초, 재생성 감사 1%(batch 64) |
| 측정 | burst 60초, 지속 워밍업 600초 + 측정 900초, 마지막 300초 구간 |
| 예산 | look 2 안전계수 1.5, 다음 블록 안전계수 1.2 |
| Cap(h) | A 24, A_repair 12, C 80, R 36, P 10, S 10, 필수 경로 114(보완 시 126) |
| Cap(GiB) | 저장·RSS·GPU 각 64, 최소 여유 디스크 20 |

---

## 3. 난수 identity와 namespace

- **기본 함수는 V5와 같다.** `identity(*parts) = sha256(canonical_json([PROTOCOL, MASTER_SEED, *parts]))`, `seed(*parts) = int(identity[:16], 16)`, `rng(*parts) = numpy.random.default_rng(seed(*parts))`.
- **후보 key.** `key_words(namespace, idx)`는 SplitMix64(idx + seed("candidate", *namespace) + 0x9E3779B97F4A7C15)의 (상위 32 bit, 하위 32 bit) uint32 쌍이다. 이것이 MLX key가 된다.
- **Draw 분리.** `draw_key(key, d) = key XOR ((0x9E3779B9·(d+1)) mod 2³², (0x85EBCA6B·(d+1)) mod 2³²)`다. V5와 같다.
- **idx**는 stream 안의 전역 후보 번호 `trial × 100 + attempt`다. stage는 `A-prof`, `A-Q`, `A-dev`, `C`, `R`, `P`, `S` 중 하나다.

| 용도 | namespace | 공유 범위 |
|---|---|---|
| MD5 과제 split | `rng("ownership", window, rung)`. 노출 group 제외 | 모든 source·파이프라인·stage (C와 S는 같은 W3 split) |
| Synthetic split | `rng("synthetic-ownership")` | P·R 공통 |
| 학습 메시지 stream | `rng("fresh", stage, source, seed_id, task, window, rung, update)` | 같은 source의 모든 파이프라인, Main·Shuffled |
| Shuffled permutation | `rng("shuffle", stage, source, seed_id, update)` | 같은 source의 Shuffled 모델 |
| 가중치 초기화 | `mx.random.seed(seed("weights", stage, pipeline, seed_id) & 0xFFFFFFFF)` 뒤 모델 생성 | 같은 파이프라인·seed의 Main·Shuffled |
| 학습 corruption | `key_words(("train-corruption", stage, pipeline, seed_id, update), rows)` | Main·Shuffled |
| Trial schedule | `rng("trials", stage, window, rung)` | 해당 stage의 모든 파이프라인 |
| 학습 모델 후보 | `key_words((stage, pipeline, window, rung, method, seed_id), idx)` | — |
| Random 후보 | `prior_candidates((stage, source, window, rung, "Random", seed_id), idx)` | 같은 source의 모든 파이프라인 |
| MC donor | `rng("mc-derangement", stage, pipeline, window, rung, seed_id)` | — |
| CLP 쌍 메시지 | 학습 stream 함수에 namespace `("clp", stage, purpose, source, seed_id)`, 대상 group | 같은 source |
| CLP corruption | `key_words(("clp-corruption", stage, purpose, pipeline, seed_id), pair_rows)` | — |
| A-Q·A-dev 후보 | `key_words((stage, pipeline, variant, seed_id), idx)`. S_G 선택지끼리 공통 | — |
| A-prof-2 target | `rng("A-prof-2", pipeline)` | — |
| 재생성 감사 선택 | `key_words(("regen-audit", stream_id, block), idx)`의 하위 word mod 100 = 0 | — |
| Production calibration | `rng("production-calibration", scenario)` | — |

---

## 4. 데이터

### 4.1 Source

- `source(generator, count, src)`: 길이는 `integers(4, 32)`, payload는 `integers(byte_min, byte_max + 1, (count, 31))`로 뽑고 길이 밖 위치를 0으로 채운다.
- `prior_candidates(namespace, idx, src)`: V5 방식이다. 위치 0은 bound 28(+4), 위치 1–31은 bound = 상태 수(+byte_min)로 위치별 rejection을 해서 modulo bias를 없앤다. R은 2³²가 256의 배수라 rejection이 일어나지 않는다. 결과는 batch 구성과 무관하다.
- `valid(message, src)`: 4 ≤ 길이 ≤ 31이어야 하고, P는 모든 byte가 33–126이어야 한다.

### 4.2 Token codec

- **encode.** 위치 < L은 byte − byte_min, 위치 L은 EOS, 위치 > L은 PAD로 채운 32칸 배열이다.
- **decode.** EOS가 정확히 1개이고 그 위치 n이 4–31이며, 앞은 모두 상태 < S, 뒤는 모두 PAD여야 한다. 그러면 `bytes(tokens[:n] + byte_min)`을 반환하고, 아니면 `None`이다.

### 4.3 해시와 window

- `digest_batch`(NumPy 벡터화)와 `digest_reference`(scalar RFC 1321)를 V5에서 복사한다. 두 source의 임의 byte를 허용한다.
- Window 추출은 다음과 같다(d[0..15]는 표준 digest byte).

| Window | 벡터 추출 | `window_value` shift |
|---|---|---|
| W1 | `(d0 << 4) | (d1 >> 4)` | 116 |
| W2 | `((d14 & 15) << 8) | d15` | 0 |
| W3 | `(d8 << 4) | (d9 >> 4)` | 52 |
| W4 | `(d4 << 4) | (d5 >> 4)` | 84 |

- W2로 split이나 학습 stream을 만들려 하면 예외를 낸다. 단 A-impl의 해시 검사는 네 window를 모두 검사한다.

### 4.4 Split

- **MD5 과제.** V5 `split`과 같은 알고리즘에서 namespace만 바꾼다. `order = rng("ownership", window, rung).permutation(4096)`로 섞고, 노출 group을 뺀 앞 1,024개를 test로 둔다. 나머지를 원래 순서대로 앞 256개 validation, 그 뒤를 train으로 둔다. 과제는 C·S = (W3, 64), R = (W4, 64), P = (W1, 4)다.
- **Synthetic.** V5 `synthetic_split`과 같다(complement 쌍 유지). Acceptance 256쌍(512 조건), dev = validation 128쌍(256 조건), train 1,408쌍(2,816 조건).

### 4.5 학습 stream과 synthetic 과제

- `fresh_batch(namespace, update, 256, train_groups, task, window, rung)`는 V5의 결정적 청크 rejection 알고리즘이다. namespace는 `(stage, source, seed_id)`이고, payload, lengths, labels, md5_calls를 반환한다.
- **Synthetic P.** `payload[:, :3] = b"0123456789ABCDEF"[nibbles]`.
- **Synthetic R.** `payload[:, :3] = nibbles`(0x00–0x0F). 두 경우 모두 labels는 `generator.choice(groups, count)`다.
- **`synthetic_label(message, src)`.** P는 앞 3 byte가 모두 대문자 hex이면 그 16진수 값이다. R은 앞 3 byte가 모두 16 미만이면 `(b0<<8)|(b1<<4)|b2`다. 그 외는 −1이다.

---

## 5. 이미지 codec

### 5.1 BGV 인코더 (2×32×128, NumPy 벡터화)

- 슬롯 s(0–31)는 행 s//8, 열 s%8이고, 셀은 8×16이며 좌상단은 (8·(s//8), 16·(s%8))이다.
- 슬롯 byte: 0번은 L, 1..L번은 payload, 나머지는 0이다.
- Bit b(0–7, MSB 먼저)는 셀 안 논리 위치 (b//4, b%4)의 4×4 픽셀 block에 들어간다.
- 채널 0은 bit 값, 채널 1은 슬롯 ≤ L이면 1이다. 최종 x = 2v − 1(float32)이다.
- 모든 byte와 길이에서 v3.1 `BGVEncoder`와 같은 텐서를 만들어야 한다(테스트).

### 5.2 CGGE 인코더 (2×32×64)

- 슬롯 s는 8×8 셀(8·(s//8), 8·(s%8))이며, 0..L−1번 슬롯이 payload 문자다.
- Glyph는 v3.1 `_GLYPH_BYTES`(94×8 byte, SHA-256 `6ef6d0bf…`)를 그대로 내장한다. 행 r, 열 c의 픽셀은 `(row_byte_r >> c) & 1`이다.
- 채널 1은 슬롯 < L이면 1이다. x = 2v − 1.

### 5.3 G3 고정 구조

v3.1 `LengthGaussianDiffusion.structure`와 같다.

- first = 1(BGV) 또는 0(CGGE). active = slot < L + first, payload = active ∧ slot ≥ first.
- 고정값: 채널 0은 −1이고, BGV 슬롯 0에는 L의 bit 패턴(±1)이 들어간다. 채널 1은 active이면 +1, 아니면 −1이다.
- 학습 잡음과 생성은 payload 영역(채널 0)만 확산하고, 나머지는 매 step 고정값으로 덮어쓴다.

### 5.4 Prototype decoder (등록 decoder)

입력은 최종 이미지 x(float32, 값 범위 −1–1)와 길이 L이다. 계산은 NumPy float64로 한다.

- **BGV.** 슬롯 s ∈ [1, L]마다 4×4 block 평균 m_b(b = 0..7)를 구하고 u_b = (m_b + 1)/2로 바꾼다. 허용 byte v에 대해 `D(v) = Σ_b (u_b − bit_b(v))²`를 계산하고, v* = argmin(동점이면 작은 v)을 고른다. 슬롯 margin은 D(v*)다.
- **CGGE.** 슬롯 s ∈ [0, L)마다 u = (glyph + 1)/2(8×8)로 바꾸고, 94개 prototype P_c에 대해 `D(c) = mean((u − P_c)²)`를 계산한다. c* = argmin(동점이면 작은 code)을 고르고, 슬롯 margin은 D(c*)다.
- **후보 margin**은 슬롯 margin의 최댓값이며 원장에 float16으로 기록한다.
- 깨끗한 인코딩 이미지는 정확히 복원되어야 한다(round-trip 100%).

### 5.5 Strict decoder (진단, v3.1 의미)

- **BGV.** Block 평균 ≥ 0.5이면 bit 1로 읽는다. 모든 payload byte가 source alphabet 안에 있으면 strict_valid다. R은 항상 strict_valid다.
- **CGGE.** 모든 활성 슬롯의 최근접 MSE가 0.1 이하이면 strict_valid다.
- 원장 flags의 bit 3에 기록하지만 판정에는 쓰지 않는다. 원 이미지를 보존하지 않으므로 검증기가 재계산할 수 없는 항목으로 표시한다.

---

## 6. 모델

### 6.1 D1-S (P, R)

V5 D1-S 구조에서 상태 수 S만 파라미터로 둔다.

- Embedding: (S+3)×16, one-hot 행렬곱으로 계산한다.
- Hidden: Linear(32·16+14 → 128) → SiLU.
- 출력: Linear(128 → 32·(S+2)). 잔차: Linear(13 → 32·(S+2)). Length head: Linear(12 → 28).
- 최종 logits는 `[..., :S]`만 쓴다.
- 파라미터 수는 **P 508,668, R 1,252,572**다.
- P 모델은 같은 가중치와 key에서 `dhi_v5.models`와 bitwise 같은 후보를 내야 한다(테스트).

### 6.2 D1-T-L (Stage S)

V5 D1-T-L을 그대로 쓴다. Pre-LN Transformer 8층 × d=256, 8 heads, FFN 1,024, position embedding, context Linear(14→256), 위치 > L을 가리는 attention mask. 파라미터 수는 **6,415,308**이다.

### 6.3 G3-U

v3.1 `ImageUNet(shape, 32, coordinates=True, condition_output=True, factorized_length=True)`를 그대로 복사한다.

- **구조.** 입력 conv(4→32, 3×3; 좌표 2채널 추가), down conv(32→64, 4×4, stride 2), middle conv(64→64, 3×3), up convT(64→32, 4×4, stride 2), skip과 concat한 뒤 output conv(64→2, 3×3).
- **조건 경로.** Linear(14→96) → SiLU → Linear(96→96)의 결과를 두 해상도에 채널별 bias로 더한다. 공간 condition-output Linear(96 → 2·32·W)를 출력에 더한다. Length head는 Linear(12→28)다.
- **초기화.** v3.1 `_initialize`와 같다. 가중치와 bias 모두 U(±1/√fan_in)이고, 전역 RNG를 weights namespace로 seed한 뒤 생성한다.
- **파라미터 수.** BGV 910,638, CGGE 513,326.
- **입력 규약.** NCHW(내부에서 NHWC로 변환), time = t/999, 조건 = (12 bits, L/31).
- 같은 가중치에서 v3.1 `ImageUNet`과 bitwise 같은 출력을 내야 한다(테스트).

---

## 7. 학습

### 7.1 공통

- Batch 256, updates 40,000(보완 80,000, S 160,000). Adam(β 0.9/0.999, ε 1e-8, bias correction, weight decay 0).
- 매 update 순서: `fresh_batch` → (Shuffled면 labels에 permutation 적용) → 표현 인코딩 → `train_step` → `charge_work` → 메모리에 기록.
- 4,000 update마다, 그리고 마지막 update에서: validation 진단(validation group에서 CLP 256쌍) → segment 파일 기록 → checkpoint 저장(모델과 Adam 상태, safetensors, 원자적 pointer).
- 평가에는 최종 checkpoint만 쓴다.

### 7.2 D1-S step

V5 `train_step`을 그대로 쓴다. Corruption은 t ~ U(0, 1)(draw 0)과 mask uniform(draw 1)이며 위치 < L만 가린다. Loss는 CE_len + 가려진 payload CE의 평균이다. lr 1e-3 고정, clip 없음.

### 7.3 G3-U step

- 행마다 key에서 `t = randint(0, 1000)`(draw 0)과 `ε = normal(2×32×W)`(draw 1)을 `vmap`으로 뽑는다.
- `x_t = where(payload, √ᾱ_t·x₀ + √(1−ᾱ_t)·ε, fixed)`, `out = model(x_t, t/999, (y, L/31))`.
- `loss_row = CE(length_head(y), L−4) + Σ(payload₀ ⊙ (out₀ − x₀₀)²) / max(Σ payload₀, 1)`. 채널 0의 활성 payload 픽셀을 균일하게 평균한다.
- `lr_u = 1e-3 × min((u+1)/1000, 1)`, grad-norm clip 1.0. `ᾱ = cumprod(1 − linspace(1e-4, 0.02, 1000))`(float32).
- Loss나 gradient가 유한하지 않으면 `FloatingPointError`를 내고 checkpoint를 전진시키지 않는다.

### 7.4 Shuffled

`permutation = rng("shuffle", stage, source, seed_id, update).permutation(256)`. `labels[permutation]`을 length head와 payload 조건 양쪽에 쓴다. L/31은 메시지의 실제 길이를 쓴다.

### 7.5 Run 폴더와 contract

- 경로는 `<root>/<stage>/runs/<pipeline>/<method>-<seed>/u<updates>/`다.
- `contract.json`(봉인)에는 protocol, pipeline, model, stage, task, window, rung, method, seed_id, updates, lr 규칙, batch, 모든 namespace, groups sha256, `resume_from`(null 또는 `{path, checkpoint_sha256, update}`)을 기록한다.
- 재시도는 V5 `attempt.json` 규칙을 따른다(재개 1회까지).

### 7.6 Segment 원장과 학습 digest 저장소

- **Segment 파일.** `segments/seg-<end:08d>.npz`에 update_id(u32), loss(f32), md5_calls(u32), data_sha256(32 B; payload‖lengths‖permutation 적용 전 labels), perm_sha256(32 B)를 담는다.
- **기록 순서.** Segment 파일(원자적, fsync) → checkpoint → pointer. 재개할 때는 pointer의 update보다 뒤인 segment를 지운다. V5의 update별 SQLite commit을 대체한다.
- **학습 digest.** MD5 과제에서는 메시지마다 SHA-256 앞 16 byte를 저장한다. 저장소는 `<root>/<stage>/streams/<source>-seed<s>/digests-<end:08d>.npy`(segment 단위, 정렬)다. 처음 만든 run이 기록하고, 같은 stream을 쓰는 이후 run은 내용이 같은지 hash로 확인한다. 공유 stream의 교차 검증이며, 불일치하면 무결성 실패다.
- **작업량.** `work.json`은 V5 `charge_work`를 그대로 쓴다(update마다 원자적 JSON, 약 0.14 ms).
- **완료.** `complete.json`(봉인)에 updates, checkpoint pointer, segment manifest hash, md5_calls, digest segment hash 목록을 기록한다.

### 7.7 보완 이어 학습 (공백 3)

- A-Q에 실패한 파이프라인의 seed 0–2마다 `u80000/` run을 만든다. `contract.resume_from`은 `u40000`의 `complete.json`이 가리키는 checkpoint다(sha256 확인).
- `u40000` checkpoint를 불러 update 40,000부터 79,999까지 실행한다. Segment와 digest는 41–80번째 구간만 새로 쓴다.
- **등가성.** 같은 설정으로 처음부터 80,000 update를 학습한 결과와 bitwise 같아야 한다. A-impl에서 4 → 8 update 축소판으로 검사한다. C와 P에서는 80,000 update를 처음부터 학습한다.

---

## 8. Sampler

### 8.1 D1 (V5와 같음, 상태 수 S만 파라미터)

- `L = 4 + categorical(length_head(y), draw 0)`, `tokens = [MASK×L, EOS, PAD…]`.
- r = 32부터 1까지: `logits = model(tokens, r/32, y, L)`, `choice = categorical(logits, draw 2r−1)`, `u = uniform(32, draw 2r)`, `reveal = (token == MASK) ∧ (u < 1/r)`.
- NFE는 33이다. Scalar 참조는 V5 `sample_reference`다.

### 8.2 G3-U (DDIM, eta 0)

1. `L = 4 + categorical(length_head(y), draw 0)`. `fixed, payload = structure(L)`.
2. `x = where(payload, normal(2×32×W, draw 1), fixed)`.
3. `times = rint(linspace(999, 0, S_G, float32)).astype(int)`. 각 step p의 t에 대해:
   - `out = model(x, t/999, (y, L/31))`, `x̂₀ = clip(out, −1, 1)`
   - `ε̂ = (x − √ᾱ_t·x̂₀)/√(1−ᾱ_t)`
   - `ᾱ′ = ᾱ_{t_{p+1}}`(마지막 step은 1)
   - `x = where(payload, √ᾱ′·x̂₀ + √(1−ᾱ′)·ε̂, fixed)`
4. 최종 `x = clip(x, −1, 1)`을 prototype decoder로 복원한다. NFE = S_G + 1이다.
5. 유한하지 않은 값이 나오면 `FloatingPointError`를 낸다.

### 8.3 Batch 규칙 (공백 4)

- 생성과 재생성 batch 크기는 64의 배수(64 이상)여야 한다. A-prof-1에서 파이프라인·S_G별로 B ∈ {256, 1,024, 2,048}를 봉인한다.
- Stream 길이(블록 819,200 또는 614,400 후보, R 1,638,400, P 409,600)는 모든 B의 배수이므로 batch가 블록 경계를 넘지 않는다.
- **A-impl 게이트.** 같은 후보 집합을 batch 64, 256, 1,024, 2,048로 생성했을 때 최종 이미지와 후보가 bitwise 같아야 한다(5개 파이프라인, S_G 셋). 실패하면 그 파이프라인은 봉인된 B로만 생성·재생성하고, 재생성 감사도 원래 batch 경계로 한다.
- **Scalar 참조(G3).** 후보 256개를 batch 1로 생성해 벡터화 결과와 비교한다. Decode 결과가 255/256개 이상 일치하고, 최종 이미지의 최대 절대차가 1e-3 이하여야 한다. 모든 차이를 기록한다.

---

## 9. CLP

- **쌍 구성.** Stream 함수로 뽑은 held-out 메시지를 두 개씩 묶는다(행 2i, 2i+1). `D_i = [s(x₂ᵢ,y₂ᵢ) − s(x₂ᵢ,y₂ᵢ₊₁)] + [s(x₂ᵢ₊₁,y₂ᵢ₊₁) − s(x₂ᵢ₊₁,y₂ᵢ)]`.
- **D1 점수.** V5 `score`와 같다. Draw d의 corruption은 t = draw 2d, mask = draw 2d+1이며 8 draw를 평균한다.
- **G3 점수(공백 2).** `s = −[CE_len + (1/8) Σ_d payload-MSE(model(x_t^(d), t_d/999, (y, L/31)), x₀)]`. t_d = `randint(0, 1000)`(draw 2d), ε_d = normal(draw 2d+1)이다. Key가 행 단위이므로 한 쌍의 네 점수가 같은 draw를 쓴다.
- **통계.** z = mean(D)/SE. 문턱은 A-Q 3.26, INFO_64와 INFO_4는 Φ⁻¹(1 − 0.01/5) = 2.8782, S는 Φ⁻¹(1 − 0.0005) = 3.2905다.
- **대칭성 fixture.** 쌍 안에서 label을 서로 바꾸면 D가 정확히 부호만 바뀌어야 한다.

---

## 10. 평가 stream과 후보 원장

### 10.1 Stream

- **Stream id**는 (stage, window, rung, pipeline 또는 source, method, seed_id)다. 학습 모델 stream은 Main, Shuffled, MC이고, Random stream은 source 단위다.
- 블록 j는 trial [T_b·(j−1), T_b·j)를 덮는다.
- **Batch마다 처리 순서.** `charge_work`(attempts, nfe) → 조건(요청 target, MC는 donor target) → key → sample → decode → NumPy H12 → hit → 학습 일치(SHA-256 앞 16 byte를 stream digest 정렬 배열에서 `searchsorted`로 조회) → 레코드 append.

### 10.2 레코드 (36 byte, little-endian)

| Offset | 크기 | 내용 |
|---:|---:|---|
| 0 | 31 | payload(길이 뒤는 0) |
| 31 | 1 | 길이 |
| 32 | 1 | flags: bit0 valid, bit1 hit, bit2 학습 일치, bit3 strict_valid(Gaussian), bit4–7은 0 |
| 33 | 2 | margin(float16). Discrete와 Random은 NaN |
| 35 | 1 | 0 |

C 원장은 최대 약 8,850만 행, 3.2 GB다.

### 10.3 파일과 commit

- `block-<j>.bin`(레코드)과 `block-<j>.commit.json`(committed_rows, batch, generator 식별)을 둔다. 8 batch 또는 30초마다 bin을 fsync한 뒤 commit JSON을 원자적으로 갱신한다.
- **재개.** commit의 행 수로 bin을 잘라내고 이어서 생성한다. 행 수는 항상 batch 경계다.
- **블록 완료.** 검증(§10.4) → 재생성 감사(§10.5) → `block-<j>.trials.npy`(trial 요약) → `block-<j>.json` 봉인. 봉인 내용은 bin·trial 요약 sha256, 행 수, batch, NFE, 경과 시간, work, 검증 결과다.

### 10.4 독립 검증기

생성 경로와 다른 구현으로 다시 계산한다.

- 행마다 `hashlib.md5`로 재해시해 window 값을 구하고 target과 비교해 hit를 재계산한다. valid와 SHA-256·학습 digest 조회도 다시 계산한다. flag가 하나라도 다르면 무결성 실패다.
- **Trial 요약 재계산.** success@1/@10/@100 bit, hits(≤ 100), 첫 hit attempt(없으면 255), trial 내 중복 수, 학습 일치 수. 생성 시 기록과 같아야 한다.
- 성공 후보가 학습 메시지와 일치하면 무결성 실패다(train/test group 분리 계약).
- **중복**은 같은 trial에서 앞선 attempt와 (길이, payload)가 같은 후보다. 중복도 attempt를 소비한다.

### 10.5 재생성 감사

- **선택.** `key_words(("regen-audit", stream_id, j), idx)`의 하위 word mod 100 = 0인 후보(약 1%).
- 선택된 후보를 idx 순서로 64개씩 묶어 같은 key와 조건으로 다시 생성한다. Payload와 길이가 원장과 bitwise 같아야 한다. Random은 `prior_candidates`로 재생성한다. 비용은 블록 생성의 약 1%다.

### 10.6 Trial schedule

- 해당 stage의 모든 학습 checkpoint를 봉인한 **뒤**, `rng("trials", stage, window, rung).choice(test_groups, T_max)`를 한 번 뽑아 checkpoint seal hash와 함께 봉인한다.
- T_max는 C가 3 × 블록, R이 16,384, P가 4,096(또는 2,048), S가 4,096이다.
- MC donor는 trial 번호에 대한 derangement다.

---

## 11. 통계·판정 엔진

### 11.1 구간

`d_t = mean_s(M[s,t] − C[s,t])`, `est = mean_t d_t`, `var = Σ(d_t − est)²/(T−1)`, `se = √(var/T)`, `[L, U] = est ∓ z·se`. V5 `interval_from_moments`를 그대로 쓴다.

### 11.2 Stage C 순차 판정

- `z_j = Φ⁻¹(1 − 0.0025·share_j)`, share = (0.1, 0.4, 0.5)이므로 z = 3.4808, 3.0902, 3.0233이다.
- Look j에서 활성 파이프라인마다 trial 0..T_j−1로 Random·Shuffled 구간을 계산한다.
- **interim 분류**: `POSITIVE`(L_R > 0 ∧ L_S > 0), `REJECTED_BOUNDED`(U_R < δ ∧ U_S < δ), 그 외 미결.
- **full 분류**: V5 순서(`POSITIVE`, `REJECTED_BOUNDED`, `REJECTED_NO_CONDITION_GAIN`, `REJECTED_NO_RANDOM_ADVANTAGE`, `UNDECIDED`).
- **중단 규칙.**
  - j < 3이고 모든 활성 파이프라인의 interim 분류가 미결이 아니면 중단하고, 그 분류를 최종으로 한다.
  - j = 3이면 full 분류를 최종으로 하고, `UNDECIDED`는 `NOT_ESTABLISHED_UNRESOLVED`로 한다.
  - 예산 때문에 look j+1을 완료할 수 없으면 look j의 full 분류를 최종으로 하고, `UNDECIDED`는 `NOT_ESTABLISHED_BY_BUDGET`으로 한다.
- **활성 파이프라인**은 Q에서 무결성 실패로 빠진 것을 제외한 집합이다. 가족 크기는 5로 고정한다(보수적).
- Look 결과는 `C/looks/look-<j>.json`에 봉인한다. 같은 입력에서 설계 검산 스크립트의 `classify`, `headline`과 같은 결과를 내야 한다(테스트).

### 11.3 Stage R

m을 R에 들어간 파이프라인 수라 하면 `z_R = Φ⁻¹(1 − 0.025/m)`이다. T = 16,384, 3 seeds. L_R > 0 ∧ L_S > 0이면 `SUPPORTED`, 아니면 `NOT_ESTABLISHED_NOT_REPLICATED`다.

### 11.4 Stage P

Seed 0, T = 4,096에서 `d_t = M_t − C_t`(C는 Random 또는 MC)다. One-sided z_P = Φ⁻¹(1 − 0.002) = 2.8782를 쓴다. L_Random > 0 ∧ L_MC > 0이면 GEN_4, CLP z > 2.8782이면 INFO_4다.

### 11.5 대비

- 6쌍 × {Δ_R, Δ_S}를 공통 trial(0..T_final−1)의 `d_t^i − d_t^j`로 계산한다. z = Φ⁻¹(1 − 0.05/24) = 2.8653.
- **분류.** 구간이 0을 제외하면 `DIFFERENT`, 구간이 (−δ, δ) 안에 있으면 `EQUIVALENT_WITHIN_DELTA`, 그 외는 `UNDETERMINED`다.

### 11.6 C4

- 파이프라인마다 V5 `compute_advantage`를 적용한다. ρ = MD5 prior 처리량 / 모델 처리량이며, 모델에 유리하도록 A-prof-1의 burst 처리량을 쓴다.
- C3 ≠ `SUPPORTED`이면 `NO_ADVANTAGE`다. 손익분기 ρ·2⁻¹² ≥ 1이면 `ARITHMETICALLY_IMPOSSIBLE` 표시를 붙인다.

### 11.7 종합 판정

파이프라인 C3는 `SUPPORTED`, `REJECTED_*`, `NOT_ESTABLISHED_*`, `UNTESTABLE` 중 하나다.

| 조건 | 종합 판정 |
|---|---|
| `SUPPORTED`가 하나 이상 | `FINAL_SUPPORTED` |
| 5개 모두 `REJECTED_*` | `FINAL_REJECTED` |
| `REJECTED_*`가 하나 이상 | `FINAL_REJECTED_WITH_EXCEPTIONS` |
| 그 외 | `FINAL_NOT_ESTABLISHED` |

### 11.8 Production calibration (A-impl)

- 생산 판정 엔진에 trial 단위 가상 결과(seed별 Bernoulli)를 넣는다.
- 시나리오: 5개 모두 무효과, P-DISC만 +δ, R-G-BGV prior 이득, 무효과 + look 2 뒤 예산 종료. 각 2,000회.
- **통과 조건.**
  - 무효과에서 어느 파이프라인이든 `POSITIVE`인 비율의 one-sided 95% CP 상한 ≤ 0.025
  - 무효과에서 5개 모두 `REJECTED_*`인 비율의 CP 하한 ≥ 0.95
  - +δ 파이프라인이 `POSITIVE`인 비율의 CP 하한 ≥ 0.95

### 11.9 Planted-lift fixture (A-impl)

- W3 validation group 하나를 target으로 한다. Random 원장 경로 전체를 쓰며, 가상 파이프라인 1개 × 3 seeds × 24,576 trials다.
- **+0.** Main이 Random과 같은 prior namespace를 쓰게 결합해 Δ_R = 0으로 만든다. Shuffled는 독립이다. 기대 판정은 `REJECTED_BOUNDED`.
- **+δ.** Main의 trial 중 비율 δ/(1−p0)에서 attempt 0을 사전 계산한 역상으로 바꾼다. 기대 판정은 `POSITIVE`.
- 난수가 결정적이므로 결과가 고정된다. 기대와 다르면 A-impl 실패다.

---

## 12. 단계 실행

### 12.1 A-impl

| 게이트 | 내용과 통과 조건 |
|---|---|
| G1 해시 | 두 source × 100,000 메시지, rung {4,5,6,7,8,10,12,16,32,64}에서 벡터 = 참조. r=64 = hashlib. W1–W4 추출 = `window_value`. r=4의 W1이 bytes 0–3에만 의존. RFC 벡터 |
| G2 codec | token·BGV·CGGE에서 모든 기호(P 94, R 256)와 길이 4–31의 round-trip 100%(prototype decoder). 깨끗한 이미지에서 strict decoder 100%. 동점 fixture. 글꼴 SHA-256 |
| G3 모델 | 6개 모델의 파라미터 수(§6) |
| G4 sampler | D1-S(P·R): 4,096 후보 scalar = 벡터화 bitwise, batch 1/64/1,024 bitwise. G3(3 파이프라인 × S_G 셋): batch 64/256/1,024/2,048 bitwise, scalar 256개 허용오차(§8.3). D1-T-L 8 후보 bitwise |
| G5 학습 | 계열별(D1-S P, D1-S R, G3 BGV, G3 CGGE, D1-T-L) 4 update 중단·재개 bitwise. 보완 이어 학습(4→8) = 처음부터 8 update bitwise |
| G6 stream | 결정성. Train group rejection(W1, W3, W4 × 두 source). 같은 source 파이프라인 간 동일 메시지. Main/Shuffled 난수 공유. Permutation 기록. Derangement에 고정점 없음 |
| G7 CLP | 5개 파이프라인 대칭성 |
| G8 원장 | 쓰기·잘라내기·재개 동일성, flag 변조 검출, payload 변조를 재생성 감사가 검출, invalid·중복의 attempt 소비 |
| G9 calibration | §11.8 |
| G10 planted-lift | §11.9 |

출력은 `A-impl.json`(소스 manifest 포함)이고, 실패하면 `A-impl-failure.json`이다. 예상 소요 시간은 30–45분이다.

### 12.2 A-prof-1 (A-Q 전)

- **학습.** 파이프라인별(D1-T-L 포함)로 무작위 초기화 모델과 synthetic stream을 써서 10 update 워밍업 후 50 update를 잰다. 데이터 생성, 인코딩, `charge_work`를 포함한 중앙값 초/update다.
- **생성 burst.** B ∈ {256, 1,024, 2,048}마다(Gaussian은 S_G ∈ {25, 50, 100}마다도) 1 batch 워밍업 후 60초 동안 잰다. B*는 처리량이 최대인 값이고, 1% 이내 동률이면 작은 B다. 최대 메모리도 기록한다.
- **기타.** MD5 prior 처리량(hashlib 단일 core, 65,536개), 검증기 처리량(1,000,000행 fixture), 레코드 저장량.
- 출력은 `A-prof-1.json`이다.

### 12.3 A-Q 학습

5 파이프라인 × seed {0, 1, 2}, synthetic, 40,000 update. `pipeline_order`, seed 순서로 진행한다.

### 12.4 A-dev (S_G 선택)

- Gaussian 3개 파이프라인에서 seed 0 checkpoint로 dev 256개 조건 × {정상, 반전} × S_G ∈ {25, 50, 100}을 생성해 joint를 잰다. dev 앞 64개 조건(정렬 순서) × 100 후보로 중복 비율을 잰다.
- S_G는 joint ≥ 231/256(두 변형 모두)이고 중복(S_G) ≤ 중복(100) + 0.005인 가장 작은 값이다. 없으면 100이다.
- 출력은 `A-dev.json`이다.

### 12.5 A-Q 평가

- Acceptance 512 × {정상, 반전} × 후보 1개. Key는 `("A-Q", pipeline, variant, seed)`다.
- 기준은 계획 §5.3과 같다. CLP는 acceptance group에서 4,096쌍.
- 결과는 run 폴더의 `qualification.json`에 봉인하고 `A.json`에 집계한다.

### 12.6 보완

실패한 파이프라인은 §7.7로 이어 학습한다. Gaussian이면 A-dev를 다시 적용하고 재평가한다. A_repair cap을 쓰며, 결과는 `A.json`에 round 2로 추가한다. 그래도 실패하면 `UNTESTABLE`이다.

### 12.7 A-prof-2 (지속 측정)

- Q의 각 파이프라인에서 seed 0 A-Q checkpoint(보완했으면 80k), 봉인된 B*, 선택된 S_G를 쓴다.
- 무작위 12-bit target(`rng("A-prof-2", pipeline)`)과 r=64 W1 해시로 실제 생성 경로 전체(생성 → decode → 해시 → 학습 조회 → 원장 → 검증)를 600초 워밍업 후 900초 동안 잰다.
- 처리량 = min(누적, 마지막 300초)이다. 임시 원장은 요약한 뒤 지운다.
- 출력은 `A-prof-2.json`이다.

### 12.8 예산 계획 (`budget-plan.json`)

```text
train_C     = Σ_{p∈Q} 6 × updates_p × sec_per_update_p                  (A-prof-1)
gen_block   = Σ_{p∈Q} 6 × T_b × 100 / sustained_cps_p                    (A-prof-2)
            + Σ_{source} 3 × T_b × 100 / random_cps
verify_block= rows_block / verify_rows_per_s                             (A-prof-1)
regen_block = 0.01 × gen_block                                            (재생성 감사)
clp_C       = Σ_{p∈Q} 3 × 65,536 × 32 / forward_rows_per_s_p
look2(T_b)  = train_C + 2 × (gen_block + regen_block + verify_block) + clp_C
```

- 1.5 × look2(8,192) ≤ cap_C이면 블록 8,192로 진행한다.
- 아니고 1.5 × look2(6,144) ≤ cap_C이면 블록 6,144로 진행한다.
- 둘 다 아니면 `HALT`다(§13.3).
- A 실측 + C 최악(3 블록) + P가 필수 경로 cap을 넘으면 P trials를 2,048로 줄이고 S를 생략한다.

### 12.9 노출 감사와 동결

- 노출 감사(§14) 결과로 `exposure-audit.json`과 `window.json`을 만든다.
- `protocol.frozen.json`에는 registration, 소스 manifest, 환경, 봉인 산출물 hash(A-impl, A-prof-1, A-prof-2, A-dev, A, budget-plan, exposure-audit, window, cap-override)를 담는다. 여기에 Q, 파이프라인별 updates·S_G·B*, 블록 크기, P trials, S 실행 여부, 유효 cap도 기록한다.
- `protocol.seal.json`을 만든다. 동결 뒤에는 A 산출물 변경을 거부한다. `verify_frozen`은 V5와 같이 registration, 소스, 환경, 산출물 hash를 모두 비교한다.

### 12.10 Stage C

1. Q × {Main, Shuffled} × seeds {0, 1, 2}를 학습한다(파이프라인별 updates).
2. 모든 C checkpoint를 봉인한 뒤 trial schedule(T_max = 3 × 블록)을 봉인한다.
3. 블록 j = 1, 2, 3마다:
   - 모든 stream(학습 모델 Q × 2 × 3, Random 2 source × 3)의 블록 j를 생성·검증·재생성 감사한다. Random과 검증은 CPU 프로세스로 병행할 수 있다.
   - Look j를 자동 계산해 봉인한다(§11.2). CLI는 `{"look": j, "action": "continue" | "stop"}`만 출력한다.
   - 중단이 아니고 j < 3이면, 직전 블록의 실측 시간 × 1.2가 남은 C cap을 넘는지 확인한다. 넘으면 다음 블록을 시작하지 않고 look j를 최종으로 한다(`BY_BUDGET` 규칙).
4. 최종 look이 정해진 뒤 파이프라인별 Main seed 3개의 CLP_64를 계산한다(65,536쌍, W3 test group, 같은 source는 같은 쌍).
5. `POSITIVE` 파이프라인에는 artifact 감사(계획 §6.5)를 한다.
6. `C.json`을 봉인한다. 최종 판정, look별 구간, seed별 값, 품질 지표, CLP_64, 감사 결과를 담는다.

실행 중 `BudgetExceeded`가 나면 마지막으로 완료된 look을 최종으로 한다. 첫 look도 없으면 Q 전체가 `NOT_ESTABLISHED_BY_BUDGET`이다. 한 stream이 무결성 실패로 끝나면 영향받는 파이프라인만 `NOT_ESTABLISHED_INTEGRITY`로 빠진다. Random stream이 실패하면 그 source의 파이프라인이 모두 빠진다.

### 12.11 Stage R

감사를 통과한 `POSITIVE` 파이프라인만 들어간다.

- W4 split을 만들고, 파이프라인마다 Main·Shuffled × 3 seeds를 학습한다.
- 모든 R checkpoint를 봉인한 뒤 trial schedule 16,384개를 봉인한다.
- Stream은 학습 모델 stream과 Random(source별, R namespace)이다. Look은 1회이며 `R.json`에 봉인한다.

### 12.12 Stage P

- W1 r=4 split을 만들고, Q의 파이프라인마다 Main seed 0을 학습한다(파이프라인별 updates).
- 모든 P checkpoint를 봉인한 뒤 trial schedule 4,096개(fallback 2,048)를 봉인한다.
- Stream은 Main·MC(파이프라인별)와 Random(source별 seed 0)이다. GEN_4와 CLP_4(16,384쌍)를 계산해 `P.json`에 봉인한다.

### 12.13 Stage S (선택)

- P-DISC D1-T-L을 W3 C split의 train group에서 160,000 update 학습한다(stream `("S", "P", 0)`).
- CLP_64를 65,536쌍으로 계산하고, GEN(Main·MC·Random, `("trials","S",W3,64)` 4,096 trials)을 기술통계로 계산한다.
- `SCALE_SIGNAL`은 CLP z > 3.2905이다. `S.json`에 봉인한다.

### 12.14 보고서

`decision.json`의 구조는 다음과 같다.

```json
{
  "status": "TERMINAL | INCOMPLETE",
  "headline": "FINAL_SUPPORTED | FINAL_REJECTED | FINAL_REJECTED_WITH_EXCEPTIONS | FINAL_NOT_ESTABLISHED | NOT_FINAL",
  "pipelines": {
    "<pipeline>": {
      "C1": "PASS | UNTESTABLE", "updates": 40000, "sampling_steps": 25,
      "C2": {"GEN_4": true, "INFO_4": true, "success_at_100": {"Main": 0.0, "MC": 0.0, "Random": 0.0}},
      "C3": "REJECTED_BOUNDED", "C3_look": 2,
      "intervals": {"Random": {"estimate": 0.0, "lower": 0.0, "upper": 0.0, "se": 0.0}, "Shuffled": {}},
      "seed_estimates": {}, "CLP_64": {"z": 0.0, "INFO_64": false}, "C4": {},
      "quality": {"duplicates_per_trial": 0.0, "training_match_rate": 0.0, "strict_valid_rate": null}
    }
  },
  "contrasts": [], "stage_s": null, "budget": {}, "failures": [],
  "blinding": {"procedure": "automatic looks; CLI shows continue/stop only"}
}
```

`FINAL_REPORT_KO.md`는 계획 §15.3의 구성을 따르고, 계획 §15.2의 사전 작성 문장에 수치를 채워 넣는다. 미완료 실행은 `INCOMPLETE / NOT_FINAL`로 표시하며 효과 없음으로 해석하지 않는다.

---

## 13. 예산과 자원

### 13.1 Budget

- V5 `Budget`을 복사하고 stage 키를 A, A_repair, C, R, P, S로 바꾼다.
- Heartbeat는 1초 이상 간격으로 기록하고, 저장량은 30초마다 다시 잰다. RSS, MLX active memory, 저장량, 최소 여유 디스크를 검사한다.
- Cap은 registration 값을 쓰되, `cap-override.json`이 있으면 그 값을 쓴다.
- 필수 경로 합계는 A + A_repair + C + P이고, 한도는 114시간(보완을 쓰면 126시간)이다.

### 13.2 실패 처리

- `BudgetExceeded`가 나면 `failure.json`(stage, reason `NOT_ESTABLISHED_BY_BUDGET`)을 쓰고 보고서를 만든 뒤 exit 2로 끝난다. C의 look 규칙은 §12.10을 따른다.
- 같은 stream이나 run이 두 번째로 중단되면 무결성 실패다(재시도 1회). 검증 불일치, 재생성 불일치, 성공 후보의 학습 일치도 무결성 실패다.
- 운영 중에는 잠자기를 막는다(`caffeinate -i`). GPU 작업은 한 번에 하나만 돌린다.

### 13.3 사람 cap 결정 (`HALT`)

- 예산 계획이 `HALT`이면 `halt.json`(예측값, 필요 cap)을 쓰고 exit 3으로 끝난다.
- `approve-caps --stage C --hours N --reason TEXT`는 `halt.json`이 있고 C·R·P·S run 폴더가 하나도 없을 때만 허용한다. 즉 MD5 조건 데이터를 만들기 전에만 가능하다.
- 승인하면 `cap-override.json`(stage, hours, reason, 시각)을 봉인하고 동결에 포함한다.

---

## 14. 노출 inventory v6

### 14.1 Schema (`v6-exposure-1`)

```json
{
  "schema": "v6-exposure-1",
  "scope_complete": true,
  "scope_boundary": "…",
  "scopes": {"code": ["../src", "../scripts", "../tests"],
             "archive": ["../local_experiment_archive/analyses", "../local_experiment_archive/retired_2026-09-21", "…"],
             "studies": ["../local_experiment_archive/runs/v5-study", "../local_experiment_archive/runs/v5-study-certified"]},
  "reviewed_files": [
    {"path": "…", "sha256": "…", "condition_use": "prefix | full | raw | toy | none | window",
     "window_roles": [{"window": "W2", "rung": 64, "role": "evaluation | training | validation | fixture | hash-test"}],
     "representatives": "full/raw일 때 대표 payload hex JSON 경로"}
  ],
  "exposed_groups": {"W1": []}
}
```

### 14.2 인증 규칙

- 범위 안의 모든 `.py`, `.json`, `.jsonl`, `.sqlite`, `.db`, `.csv`, `.md`, `.npy`, `.npz`, `.bin` 파일이 sha256과 함께 검토되어야 한다. V5의 확장자 목록에 V6 원장 형식을 더한 것이다. 가중치 파일(`.safetensors`)은 target 정보를 담지 않으므로 제외한다. 빠지거나 남는 파일이 있으면 인증하지 않는다.
- Window W에 role이 evaluation, training, validation인 기록이 하나라도 있으면(rung 64) W 전체를 노출로 본다. fixture와 hash-test role은 노출이 아니다.
- prefix, full, raw의 의미는 V5와 같다. 대표 메시지를 W1–W4로 투영해 해당 group을 제외한다.
- **인증 조건.** W3와 W4의 미노출 group이 각각 1,024개 이상이어야 한다(예상 4,096개). W2는 V5 Stage C 때문에 전체 노출로 기록되며 V6는 쓰지 않는다.

### 14.3 초안 생성 (`inventory --draft`)

1. V5 inventory에서 path와 sha256이 그대로인 항목은 분류를 복사한다.
2. V5 study root의 파일은 경로 규칙으로 자동 분류한다.
   - `*/C/*`: W2 evaluation과 training
   - `*/A-Q/*`, `*/A-dev/*`, `profile-ledger.sqlite`: none(synthetic)
   - `A-impl.json`: W2 fixture(planted, validation group)와 W1–W3 hash-test
3. 규칙에 맞지 않는 파일은 `unreviewed`로 남긴다. 사람이 분류하기 전에는 인증하지 않는다.
4. 2026-09-29 기준으로 V5 inventory 이후 생긴 파일은 9개다(F1·V6 설계 산출물, V5 분석, 스크립트). V5 inventory를 지금 재검증하면 `UNREVIEWED_FILES`가 나온다.

---

## 15. 디렉터리 구조

```text
<root>/
  v6-study.json                       # 표지(마커)
  A-impl.json  A-prof-1.json  A-dev.json  A.json  A-prof-2.json  budget-plan.json
  [halt.json  cap-override.json]  exposure-audit.json  window.json
  protocol.frozen.json  protocol.seal.json  budget.json
  A-Q/runs/<pipeline>/Main-<s>/u40000/        # contract, checkpoint, segments/, work.json, complete.json, qualification.json
  A-Q/runs/<pipeline>/Main-<s>/u80000/        # 보완 이어 학습
  A-dev/<pipeline>/u<updates>/                # S_G 평가
  C/groups-W3-r64.json  C/trials.json
  C/runs/<pipeline>/<Main|Shuffled>-<s>/u<updates>/
  C/streams/<source>-seed<s>/digests-*.npy    # 학습 digest(공유)
  C/eval/<pipeline|source>/<method>-<s>/block-<j>.{bin,commit.json,trials.npy,json}
  C/looks/look-<j>.json  C/clp/<pipeline>-seed<s>.json  C.json
  R/…  P/…  S/…   (C와 같은 구성)
  decision.json  FINAL_REPORT_KO.md  [failure.json]
```

---

## 16. CLI

`PYTHONPATH=src .venv/bin/python -m dhi_v6.study <command>` 또는 `hash-inverse-v6 <command>`.

| 명령 | 동작 |
|---|---|
| `plan [--output PATH]` | `registration()` 출력(선택적으로 봉인 저장) |
| `inventory --draft --v5 examples/v5-exposure-inventory.json --output PATH` | §14.3 초안 생성 |
| `audit --root R --inventory PATH` | 노출 감사. 동결 후 거부 |
| `run --root R --stage {A,C,R,P,S,all} [--phase {impl,prof1,train,dev,evaluate,repair,prof2,budget,freeze,all}]` | 단계 실행. `all`은 A → C → (R) → P → (S) → report |
| `approve-caps --root R --stage C --hours N --reason TEXT` | §13.3 |
| `status --root R` | 단계, 블록, 완료 행 수, 최근 처리량, stage별 경과·남은 cap, ETA만 출력. 성공률·구간·판정은 출력하지 않음 |
| `report --root R` | `decision.json`, `FINAL_REPORT_KO.md` 생성 |
| `check --root R [--quick]` | A-impl 게이트. `--quick`은 개발용이며 PASS를 만들지 않음 |

- 종료 코드: 0 정상, 1 오류, 2 예산 종결, 3 `HALT`(cap 승인 필요).
- 새 study는 빈 디렉터리에서만 시작한다. 같은 디렉터리에서 다시 실행하면 봉인된 단계는 검증 후 건너뛰고, 중단된 run·stream은 같은 identity로 한 번 재개한다.
- `failure.json`이 있으면 `run`을 거부한다.

---

## 17. 테스트 (`tests/test_study_v6.py`)

| 테스트 | 검사 내용 |
|---|---|
| `test_independence_and_registration` | AST 의존 규칙. `registration()` = JSON. JSON sha256 = `.sha256` 파일 |
| `test_hash_windows_sources` | 두 source의 벡터 = 참조, W1–W4 추출, r=4 의존성, W2 금지 |
| `test_token_codec_p_r` | 두 source token round-trip, ID 상수, 잘못된 입력 거부 |
| `test_image_encoders_match_v31` | NumPy BGV·CGGE 인코더 = v3.1 torch 인코더(torch가 없으면 skip) |
| `test_prototype_and_strict_decoders` | Round-trip, 동점 규칙, alphabet 제한, strict = v3.1 decoder, 글꼴 checksum |
| `test_parameter_counts` | 6개 모델 파라미터 수 |
| `test_d1s_parity_with_v5` | 같은 가중치·key에서 V6 D1-S(P) = V5 D1-S(Metal subprocess) |
| `test_g3u_forward_parity_with_v31` | 같은 가중치에서 V6 G3-U = v3.1 `ImageUNet` |
| `test_samplers_reference_and_invariance` | 축소 규모 scalar 참조 비교, batch 불변성 |
| `test_training_resume_and_continuation` | 중단·재개 bitwise, 4→8 이어 학습 = 처음부터 8 update |
| `test_shared_streams_and_controls` | 같은 source 동일 메시지, Main/Shuffled 공유, permutation, derangement |
| `test_clp_antisymmetry` | 5개 파이프라인 D 부호 반전 |
| `test_ledger_commit_resume_tamper_regeneration` | 잘라내기·재개, flag·payload 변조 검출, 중복·invalid 계수 |
| `test_sequential_engine_rules` | 분류 순서, 공동 중단, 예산 종료 시 look 규칙, z 값 |
| `test_design_script_parity` | `scripts/validate_research_plan_v6.py`의 `classify`·`headline`과 같은 결과 |
| `test_replication_p_contrasts_c4_headline` | z_R(m), GEN_4, 대비 분류, C4 불가능 표시, 종합 판정 |
| `test_calibration_quick` | 축소 반복 calibration |
| `test_budget_plan_decisions` | 진행, fallback, `HALT`, P 축소·S 생략 |
| `test_audit_v6_rules` | W2 evaluation → 노출, W3 fixture → 비노출, 누락 파일 → 미인증 |
| `test_stage_order_and_blinding` | C가 P·S보다 먼저, R은 감사된 `POSITIVE`에서만, `status`에 결과 필드 없음 |
| `test_partial_report_is_not_rejection` | 미완료 보고서가 기각으로 표시되지 않음 |

**Metal 테스트 규칙.** MLX가 필요한 테스트는 `@pytest.mark.metal`을 붙이고 V5처럼 subprocess로 실행한다. Metal을 쓸 수 없으면 skip하지만, 환경변수 `DHI_V6_REQUIRE_METAL=1`이면 skip 대신 실패한다. 구현 완료는 이 변수를 켠 실행에서 skip이 0개일 때만 인정한다. 샌드박스에서 Metal 테스트가 조용히 skip되어 통과처럼 보이는 것을 막기 위한 규칙이다.

---

## 18. 구현 순서와 완료 기준

| 순서 | 작업 | 완료 기준 |
|---|---|---|
| 1 | `dhi_v6` 골격, `protocol.py`, `data.py`(P·R, W4, 공유 stream) | 해시·codec·stream·split 테스트 통과, V5 등가 테스트 통과 |
| 2 | `codecs.py`, `models.py`(D1-S 일반화, G3-U, D1-T-L), sampler와 scalar 참조 | 파라미터 수, v3.1 등가, 불변성 게이트(축소 규모) 통과 |
| 3 | `runtime.py` 학습(segment, digest 저장소, 재개, 이어 학습) | G5·G6 게이트 통과 |
| 4 | `runtime.py` 평가 stream, 원장, 검증기, 재생성 감사 | G8 게이트 통과 |
| 5 | `statistics.py` 판정 엔진, calibration, 설계 스크립트 일치 | G9 게이트, 판정 테스트 통과 |
| 6 | `study.py` 단계 실행기, 예산 계획, 노출 감사 v6, CLI, 보고서, 사용 문서 `V6_CLI.md`(구현 결정 절 포함) | 단계 순서·블라인드·보고서 테스트 통과 |
| 7 | 정식 A-impl 실행(`run --stage A --phase impl`) | `A-impl.json` PASS. 이후 계획 §14 순서대로 실행 |

작업 1–6은 MD5 조건 데이터를 만들지 않는다. 작업 7 이후의 순서, cap, fallback은 계획을 따른다.

로컬 Codex로 구현할 때의 phase 분할(위 작업 2를 codec과 모델로 나눈 7개 phase), 금지 사항, 실행 명령, 완료 기준, 요청 문구는 [CODEX_HANDOFF_V6.md](CODEX_HANDOFF_V6.md)에 있다.
