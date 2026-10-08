# V7 실험 계획 — P-G-CGGE 단일 파이프라인: 추론 이미지 추적과 무작위 메시지 기반 Test

**Protocol(제안):** `dhi-v7-20261008` · **Master seed(제안):** `2026100807` · **작성일:** 2026-10-08 KST
**상태:** `PLAN_ONLY` — 계획과 설계 검산만 했다. 구현, 노출 감사, 학습, MD5 조건 데이터 생성은 하지 않았다.

이 문서의 수치는 세 종류다.

- **V6 archive에서 읽은 값.** `local_experiment_archive/runs/v6-study-r2`의 최종 보고서, 동결 protocol, 블록 기록, 그리고 2026-10-08에 만든 [P-G-CGGE 추출 보고서](Presentation/P_G_CGGE_Train_Valid_Test/REPORT_KO.md). 읽기만 했다.
- **설계 검산값.** 정규 근사와 2,000회 모의(§10.2). MD5 조건 데이터와 모델은 쓰지 않았다.
- **추정값.** 시간과 용량(§14). V6 실측 처리량에 저장 부담을 더해 계산했다.

**요약.** V7은 V6에서 정의한 P-G-CGGE 파이프라인 하나만 쓴다. 목표는 세 가지다.

1. **추론 이미지를 저장한다.** Train과 Valid 과정에서 diffusion 모델이 추론한 이미지(학습 중 x̂₀ 예측, DDIM 생성 결과, 생성 궤적)를 원본 메시지·해시와 짝지어 저장한다. V6 추출 보고서에서 "미저장"이던 칸을 모두 채운다(§6).
2. **Test를 무작위 메시지에서 시작한다.** Source prior에서 메시지를 뽑아 그 해시값을 목표 해시로 쓴다. 따라서 모든 Test 시행에 "원본 메시지 → 인코딩 이미지 → 원본 해시 → 모델 추론 이미지 → 디코딩 메시지 → 디코딩 해시"의 6열이 생긴다(§7).
3. **주 판정은 V6 규칙을 그대로 쓴다.** 새 window W4에서 Main이 Random과 Shuffled보다 Success@100이 높은지 사전 규칙으로 판정한다. V6 판정은 바꾸지 않는다(§3, §10).

관련 문서: [V6 계획](RESEARCH_PLAN_V6.md) · [V6 구현 명세](V6_IMPLEMENTATION_SPEC.md) · [V6 등록값](examples/v6-protocol.json) · [P-G-CGGE 추출 보고서](Presentation/P_G_CGGE_Train_Valid_Test/REPORT_KO.md) · V6 최종 보고서 `local_experiment_archive/runs/v6-study-r2/FINAL_REPORT_KO.md`(로컬 전용)

---

## 인계 안내

### 현재 상태 (2026-10-08)

| 항목 | 상태 |
|---|---|
| V7 계획 문서 | 이 문서. 미커밋 |
| V7 구현 명세, `examples/v7-protocol.json`, handoff 문서 | 작성 전(§16) |
| `src/dhi_v7/` 구현 | 시작 전 |
| W4·W5 노출 감사 | 시작 전(§5.2) |
| Stage A–P 실행 | 시작 전. MD5 조건 데이터 없음 |
| `AGENTS.md` | 아직 V6 구현 규칙을 가리킨다. V7 구현을 시작할 때 갱신해야 한다 |

### 착수 전에 사람이 정할 항목 (권장 기본값)

| 번호 | 항목 | 권장 | 대안과 그 결과 |
|---:|---|---|---|
| 1 | Test 메시지의 범위 | **해시가 test group에 속하는 메시지만** 뽑는다(§7.2) | 제한하지 않으면 목표의 약 69%가 학습에서 본 group이 된다. 이 경우 "미학습 target" 질문이 아니게 되고 V6와 비교할 수 없다 |
| 2 | 주 window | **W4** (`>> 84`, 한 번도 쓰지 않음) | W3를 쓰면 V6 C와 같은 window라 직접 비교는 쉽지만, V6에서 test pool 전체를 평가했으므로 노출 window가 된다 |
| 3 | 시행 수와 δ | **16,384 시행 × K=100 × 3 seeds, δ = 0.5%p** (V6와 같은 δ) | δ = 0.25%p로 낮추면 같은 배제 확률(99%)에 약 52,000 시행이 필요하다. Test 생성이 약 3.2 h에서 약 10 h로 늘어난다 |
| 4 | 이미지 정밀도 | **float32** (총 약 6.4 GB) | float16이면 약 3.5 GB다. 대신 저장 이미지를 다시 decode한 결과가 원장과 다를 수 있어 G3 게이트(§13)를 완화해야 한다 |
| 5 | Stage P(r=4 양성 대조) | **포함** (약 1.2 h) | 빼면 "이 기계가 해시 구조를 이용할 수 있는가"에 대한 V7 자체의 증거와, 조건이 실제로 작동하는 추론 이미지 예시가 없어진다 |
| 6 | A-Q 범위 | **seed 0 한 개** (회귀 검사) | V6는 3 seeds 모두 512/512였다. 같은 구현을 재사용하므로 seed 1개로 추적 경로의 회귀만 확인한다 |
| 7 | Protocol ID와 master seed | 위의 제안값 | 등록 전에 바꿀 수 있다. 바꾼 뒤에는 동결한다 |

권장값과 다르게 정하면 구현 전에 이 문서와 `examples/v7-protocol.json`(작성 예정)을 먼저 고친다.

### 반드시 지킬 제약

- V6 판정(`FINAL_REJECTED`, P-G-CGGE `REJECTED_BOUNDED`)은 바꾸지 않는다. V7 결과를 V6 자료와 합치지 않는다.
- `src/dhi_v6/`, `src/dhi_v5/`, `src/diffusion_hash_inv/`, `local_experiment_archive/runs/` 아래 산출물은 수정하지 않는다. V7은 `dhi_v6`을 읽기 전용으로 import한다.
- 결과를 본 뒤 규칙, seed, 학습량, sampler, 시행 수, δ, 추적 범위를 바꾸지 않는다.
- W4·W5의 MD5 조건 데이터는 동결(§8.5) 뒤에만 만든다. Stage M을 Stage P보다 먼저 실행한다.
- 실패, 차단, 예산 초과를 효과가 없다는 증거로 쓰지 않는다. `NOT_ESTABLISHED_*`로 기록한다.
- Python은 `.venv/bin/python`만 쓴다. 새 패키지를 설치하지 않는다(이미지 PNG 출력은 기존 보고서처럼 `zlib`·`struct`로 한다).

---

## 0. 최종 결론 카드 (실험 종료 시 채운다)

| 질문 | 판정 값 | 근거 |
|---|---|---|
| **Q1 추적 완결성.** Train·Valid·Test의 6열 추적이 결손 없이 저장·재현되는가 | `TRACE_COMPLETE` / `TRACE_INCOMPLETE` | §6, §7.4, §13 |
| **Q2 기계 확인.** V7 구현이 synthetic 조건을 V6처럼 쓰는가 | `PASS` / `FAIL` | Stage A-Q |
| **Q3 구조 이용.** r=4에서 해시 구조를 이용하는가 | `GEN_4`, `INFO_4` 성립 여부와 효과 크기 | Stage P |
| **Q4 주 판정.** 무작위 메시지로 만든 미학습 W4 target에서 Main이 Random과 Shuffled를 모두 능가하는가 | `SUPPORTED` / `REJECTED_*` / `NOT_ESTABLISHED_*` + Δ_R, Δ_S 구간 | Stage M, R |
| **Q5 보조 관찰.** 원본 메시지 근접성, 12-bit 일치 bit 수, seen/unseen target, 암기, 조건 민감도 | 추정값과 구간 (판정 없음) | §10.4 |
| **종합** | §15의 세 가지 중 하나 | 위 다섯 줄 |

**사전 예측**(판정 규칙에 영향 없음).

- Q1 `TRACE_COMPLETE`. Q2 `PASS`(V6 A-Q 512/512).
- Q3 `GEN_4`·`INFO_4` 성립. V6 P의 P-G-CGGE는 Success@100 15.1%, Main−Random +13.0%p였다.
- Q4 `REJECTED_BOUNDED`. 무효과라면 상한은 약 +0.22%p다.
- Q5 모든 지표가 우연 수준이다. 길이 일치 1/28, 위치별 byte 일치 1/94, 12-bit 일치 6.0 bits, 원본 메시지 복원 0건.

---

## 1. 배경: V6 결과와 V7을 하는 이유

### 1.1 V6의 P-G-CGGE 결과 (`v6-study-r2`)

| 항목 | 값 |
|---|---|
| 정의 | Printable source, CGGE 2×32×64, G3-U 513,326 params, 40,000 updates × 256, S_G = 25, 생성 batch 256 |
| A-Q (synthetic) | 3 seeds 모두 정상 512/512, 반전 512/512, CLP z 114.8–116.9 |
| C (W3, r=64) | Main Success@100 1,224 / 49,152 = 2.490%. Δ_R −0.004%p [−0.313, +0.305], Δ_S +0.059%p [−0.246, +0.364]. `REJECTED_BOUNDED`(look 2). CLP_64 z = 0.12 |
| P (W1, r=4) | Main Success@100 620 / 4,096 = 15.14%. Main−Random +12.96%p, Main−MC +13.72%p. CLP_4 z = 114.3 |
| 후보 품질(C) | Prototype valid 100%, strict valid 33.3%, 학습 메시지 일치 814개(0.0166%), trial 내 중복 0 |
| 처리량 | 학습 0.0694 s/update(run당 약 47분). 생성 지속 863 후보/s. 블록 819,200 후보에 925–980 s |

### 1.2 V6 추출 보고서에서 드러난 결손

2026-10-08 추출 보고서는 6열 추적을 만들려 했지만 다음 칸을 채우지 못했다. 원인은 V6 설계가 이 자료를 만들지 않았기 때문이다.

| 열 | Train | Valid | Test |
|---|---|---|---|
| 원본 메시지 | 학습 stream 재구성(체크섬 일치) | 재구성(**입력 체크섬 미저장**) | **해당 없음**: 목표 12 bits를 group에서 직접 뽑았다 |
| 원본 인코딩 이미지 | 재구성 | 재구성 | **해당 없음** |
| 원본 해시 | 계산 | 계산 | **해당 없음** |
| 모델 추론 이미지 | **미저장** | **미저장** | **미저장**(원장에는 decode된 payload만 있다) |
| 디코딩 메시지 | **미저장** | **미저장** | 원장에서 추출 |
| 디코딩 해시 | 계산할 메시지 없음 | 계산할 메시지 없음 | 원장 payload로 계산 |

V7은 이 표의 굵은 칸을 설계 단계에서 없앤다. Test 그림도 V6에서는 "decode된 메시지를 다시 인코딩한 이미지"였지만, V7에서는 모델이 실제로 만든 이미지를 저장한다.

### 1.3 이론적 기대

V6 §1.5의 random-function 논증은 그대로 적용된다. MD5의 12-bit window가 미학습 입력에서 이상적 random function처럼 행동하면, MD5를 호출하지 않는 sampler의 후보당 성공확률은 2⁻¹²이고 Success@100은 p₀ = 1 − (1 − 2⁻¹²)¹⁰⁰ = 2.412%다. Test target을 무작위 메시지에서 만들어도 target 분포는 test group 위에서 거의 균일하므로 이 기대는 바뀌지 않는다(§7.2).

**원본 메시지 복원은 원리적으로 불가능하다.** 조건은 12 bits뿐이다. 반면 Printable 메시지 하나의 prior 엔트로피는 평균 약 119.5 bits다(log₂28 + 평균 길이 17.5 × log₂94). 따라서 이상적 역함수라도 원본 메시지를 약 2¹⁰⁷·⁵개의 동등한 후보 중에서 고를 수 없다. V7은 원본과의 근접성을 재지만, **성공 기준은 V6와 같이 12-bit window 일치다.**

---

## 2. V6 대비 변경점

| 항목 | V6 | V7 | 이유 |
|---|---|---|---|
| 파이프라인 | 5개 | **P-G-CGGE 하나** | 사용자 결정. 정의는 V6와 bit 단위로 같다(§4) |
| 주 window | W3 | **W4** | W3는 V6 C가 평가했다. W4는 미사용(§5) |
| Test target | test group에서 iid 복원추출 | **무작위 메시지의 해시값**(test group으로 제한) | 모든 시행에 원본 메시지를 둔다(§7) |
| Train 추론 이미지 | 저장 안 함 | 학습 batch의 x_t·x̂₀, checkpoint batch 전체, 고정 probe의 재구성·생성·궤적 | 목표 1(§6.3) |
| Valid 추론 이미지 | 저장 안 함(loss와 CLP만) | Validation loss의 x̂₀, probe의 재구성·생성·궤적, 입력 체크섬 | 목표 1(§6.4) |
| Train·Valid 생성 평가 | 없음 | 최종 checkpoint에서 TV-final(seen 1,024 / unseen 1,024 target × K=100) | seen/unseen target 비교와 암기 확인(§6.5) |
| Test 추론 이미지 | 저장 안 함 | 앞 256 시행 전수, 모든 hit, 궤적 표본, 같은 key·다른 조건 쌍 | 목표 1(§7.4) |
| 순차 설계 | 8,192 블록 × 최대 3 look | **고정 16,384 시행, 분석 1회** | 파이프라인 하나라 비용이 작다(§10) |
| 동시 구간 | 20개 꼬리 | **4개 꼬리**(대조 2 × 양측) | 파이프라인 하나 |
| 재현 | W4 | **W5**(`>> 20`, 신규, 조건부) | W4를 주 window로 쓴다(§11) |
| 적격성 | A-Q 3 seeds, S_G 선택, 보완 | A-Q seed 0 회귀 검사. S_G = 25와 batch 256은 V6 동결값 고정 | 정의를 바꾸지 않는다 |
| Stage S, C5 대비 | 있음 | 없음 | 파이프라인 하나 |
| 보조 분석 | CLP, 후보 품질 | 위에 더해 원본 근접성, 12-bit 일치 bit 수, 128-bit Hamming, 조건 민감도, 재구성 정확도 | 6열 추적이 생겨 가능해졌다(§10.4) |

이어받는 요소: source와 `H12_r^W` 정의, group ownership, 새 쌍 학습, batch 내 Shuffled, Success@100 원장 계약(invalid·duplicate도 기회 소비), 36-byte 원장 레코드, 독립 verifier, 1% 재생성 감사, CLP, artifact 감사, 무결성·재개 계약, V6 §6.3 판정 순서.

---

## 3. 결정 질문과 원칙

> **결정 질문.** V6에서 정의한 P-G-CGGE 파이프라인을 W4 train group으로 새로 학습했을 때, 무작위 Printable 메시지의 W4 12-bit 값을 목표로 하면 해시 조건 모델(Main)의 Success@100이 source-prior Random과 Shuffled-condition 모델보다 높은가? 높다면 W5에서 재현되는가? 높지 않다면 이득의 상한은 얼마인가? 그리고 이 과정의 모든 추론 이미지를 원본 메시지·해시와 함께 추적할 수 있는가?

1. **V6와의 관계.** V6 계획은 "V6 이후 같은 질문의 revision은 없다"고 정했다. V7은 사용자 요청에 따른 새 연구다. V6를 다시 판정하거나 V6 결론을 뒤집는 근거로 쓰지 않는다. V7의 주된 기여는 추적성(Q1)이다. 주 판정(Q4)은 새 window와 새 target 설계에서 사전 등록 규칙으로 다시 잰 값이다.
2. **입증 책임은 가설 쪽에 있다.** `SUPPORTED`는 사전 규칙의 양성 판정, artifact 감사, W5 재현을 모두 통과해야 나온다.
3. **설계값은 MD5 조건 데이터를 만들기 전에 동결한다.** Stage A는 synthetic 과제와 fixture만 쓴다.
4. **추적은 측정을 바꾸지 않는다.** 이미지 저장을 켜도 학습 parameter, 학습 stream, 후보, 원장이 bit 단위로 같아야 한다(§13 G2). 추적은 관찰일 뿐 개입이 아니다.
5. **사례 선택은 사전에 정한다.** 보고서에 싣는 예시는 §17의 규칙으로 고른다. 결과를 보고 고른 예시(첫 hit 등)는 그렇다고 표시한다.
6. **항상 수치를 남긴다.** 어떤 종결에서도 Δ_R, Δ_S의 추정값과 구간을 보고한다.

---

## 4. 고정 파이프라인: P-G-CGGE (V6 동결값)

V7은 아래 값을 하나도 바꾸지 않는다. 구현은 `dhi_v6.models`·`dhi_v6.codecs`의 함수를 그대로 쓰고, 난수 identity만 V7로 바꾼다(§16).

| 구성 | 값 (V6 출처) |
|---|---|
| Source | Printable, byte 33–126(94 상태), 길이 4–31 균등, 길이가 정해지면 byte iid 균등 (V6 §4.1) |
| 인코딩 | CGGE 2×32×64. 슬롯 s는 8×8 셀 (8·(s//8), 8·(s%8)). 채널 0은 glyph, 채널 1은 활성 mask. x = 2v − 1. Glyph 표 SHA-256 `6ef6d0bf…ed50a` (명세 §5.2) |
| 모델 | G3-U: U-Net width 32, 좌표 채널, 공간 condition-output, length head `Linear(12,28)`. 513,326 params (명세 §6.3) |
| 조건 | 12 bits(MSB first) + L/31. Length head는 12 bits만 받는다 |
| 확산 | 1,000 step, 선형 β 1e-4→0.02, x₀-prediction. Payload 영역(채널 0)만 확산하고 나머지는 고정값 (명세 §5.3) |
| 학습 | 새 쌍 batch 256 × 40,000 updates(1,024만 쌍). Adam 1e-3, warmup 1,000, grad-norm clip 1.0. Loss = 길이 CE + 활성 payload 픽셀의 x₀-MSE 균일 평균. 최종 checkpoint만 사용 (명세 §7.3) |
| Shuffled | Batch 안에서 condition을 permutation으로 재배정. L/31은 실제 길이 (명세 §7.4) |
| Sampler | 길이 1회 categorical → DDIM(eta 0) **S_G = 25**, 매 step x̂₀를 [−1, 1]로 clip. NFE 26. 생성 batch **256** (V6 동결값) |
| Decoder | Source alphabet 94개 glyph prototype 중 MSE 최근접, 동점이면 작은 code. 후보 margin = 슬롯 MSE 최댓값. 진단용 strict decoder(MSE ≤ 0.1) (명세 §5.4–5.5) |
| 원장 | V6 36-byte 레코드(payload 31 B, 길이, flags, margin f16) (명세 §10.2) |

---

## 5. 과제, window, split

### 5.1 과제

| 과제 | 해시 | 용도 | 비고 |
|---|---|---|---|
| Synthetic nibbles | 첫 3자가 조건의 대문자 hex 3자리 | Stage A-Q | V6와 같다. MD5를 쓰지 않는다 |
| **W4, r=64** | `H12_64^{W4}`, `(d4 << 4) \| (d5 >> 4)` | **Stage M (주 판정)** | 한 번도 학습·평가에 쓰지 않았다 |
| W1, r=4 | `H12_4^{W1}`, bytes 0–3만의 함수 | Stage P (양성 대조) | V6 P와 같은 함수, V7 split |
| W5, r=64 | `(digest >> 20) & 0xFFF` = `(d12 << 4) \| (d13 >> 4)` | Stage R (조건부 재현) | 신규 정의. 코드에 사용 흔적 없음(2026-10-08 grep) |

**쓰지 않는 window.** W2(V5 C가 평가), W3(V6 C가 평가), W1 r=64(V5 감사에서 1,885 group 이상 노출).

### 5.2 노출 감사

1. V6 inventory에 `v6-study`, `v6-study-r2` 실행 root와 `Presentation/P_G_CGGE_Train_Valid_Test` 산출물을 추가해 V7 inventory를 만든다.
2. **W4.** V6에서 W4는 재현 window로 등록만 되었고, Stage R은 실행되지 않았다(`decision.json`의 `stage_r` = null). V6 노출 감사의 W4 제외 group은 0개였다. W4 값을 계산한 곳은 두 군데다. A-impl 해시 검사(무작위 메시지, `hash-test`)와 stream 검사(V6 split의 train group 메시지 256개, 모델 입력·평가 없음, `fixture`)다. 이를 기록하고 미노출로 인증한다.
3. **W5.** 어떤 코드와 산출물도 쓰지 않았음을 코드 감사로 확인한다.
4. W1 r=4는 양성 대조이므로 노출 판단 대상이 아니다. V6 P 결과가 알려져 있다는 사실은 기록한다.
5. 동결 전에는 W4·W5 조건 데이터를 만들지 않는다.

### 5.3 Split

- 각 과제(W4 r=64, W1 r=4, W5 r=64)의 4,096 값을 test 1,024 / validation 256 / train 2,816 group으로 한 번씩 나눈다. 알고리즘은 V6 명세 §4.4와 같고 identity만 V7이다.
- 학습 쌍은 source prior에서 뽑은 x 중 H(x)가 train group인 것만 쓴다(수락률 68.75%).
- 같은 seed의 Main과 Shuffled는 메시지 stream, 초기 가중치, corruption 난수를 공유한다.

---

## 6. 추론 이미지 저장 설계 (Train · Valid)

### 6.1 6열 추적 계약

모든 추적 레코드는 아래 6열을 갖는다. **빈 칸이 생기면 `TRACE_INCOMPLETE`다**(§13).

| 열 | Train | Valid | Test (§7) |
|---|---|---|---|
| ① 원본 메시지 | 학습 batch의 행, 또는 고정 Train probe | Validation group 메시지(V-loss, V-probe, TV-final) | 무작위 test 메시지(시행표) |
| ② 원본 인코딩 이미지 | CGGE encoder(①) | 같음 | 같음 |
| ③ 원본 해시 | MD5 128-bit, W4 12-bit(= 조건) | 같음 | 같음(= 목표 해시) |
| ④ 모델 추론 이미지 | (가) 학습 step의 x̂₀, (나) DDIM 생성 최종 이미지와 궤적 | (가) V-loss의 x̂₀, (나) DDIM 생성 | DDIM 생성 최종 이미지와 궤적(Main, Shuffled) |
| ⑤ 디코딩 메시지 | Prototype decoder(④). (가)는 참 길이, (나)는 sampling한 길이 | 같음 | 같음(원장 payload와 bit 단위로 일치) |
| ⑥ 디코딩 해시 | MD5 128-bit, W4 12-bit, ③과의 일치 여부, 일치 bit 수 | 같음 | 같음 + ①·③ 대비 지표(§10.4) |

Random 후보는 모델을 거치지 않으므로 ④가 "해당 없음(모델 없음)"이다. 이는 설계상 정의된 값이며 결손이 아니다. 보고서에서는 Random 후보의 인코딩 이미지를 "추론 아님"으로 표시해 함께 싣는다.

### 6.2 "추론 이미지"의 두 종류

| 종류 | 정의 | 해석 |
|---|---|---|
| **(가) 재구성 추론** | 원본 x₀에 잡음을 넣은 x_t를 모델에 넣어 얻은 x̂₀ = model(x_t, t, 조건) | x_t에 원본 정보(√ᾱ_t·x₀)가 남아 있으므로, ⑤가 ①과 같아지는 것은 **복원이지 해시 역상이 아니다.** 작은 t에서 ⑥ = ③은 당연하다 |
| **(나) 생성 추론** | 조건(12 bits)만 주고 순수 잡음에서 DDIM 25 step으로 만든 이미지 | Test와 같은 역상 시도다. ⑥ = ③이면 hit다 |

보고서는 두 종류를 다른 표에 싣고, (가)의 일치율을 성공률로 표기하지 않는다.

### 6.3 Train 저장 항목 (학습 run마다)

추적 시점은 `U_trace = {0, 500, 1,000, 2,000, 4,000, 8,000, …, 40,000}`(14개)이다. 시점 u는 u번 update를 마친 가중치다. 4,000의 배수가 아닌 시점은 checkpoint를 쓰지 않는 추적 전용 시점이다.

| 이름 | 내용 | 저장 시점 | 이미지 수 |
|---|---|---|---:|
| **T-step** | 실제 학습 batch의 0–7번 행. 메시지, 조건, t, x_t, x̂₀(모델 출력, clip 전), loss_row, ⑤·⑥ | 100 update마다(400회) | 6,400 |
| **T-ckpt** | Checkpoint 직전 update batch의 256행 전체 x̂₀와 ⑤·⑥ | 4,000 update마다(10회) | 2,560 |
| **T-probe 재구성** | 고정 Train probe 64개(seed별 update 0 batch의 0–63번 행)를 t ∈ {50, 250, 500, 750, 999}와 고정 ε로 재구성한 x̂₀ | U_trace | 8,960 (V-probe 포함) |
| **T-probe 생성** | 같은 probe의 조건으로 후보 4개씩 DDIM 생성. Key는 시점과 무관하게 고정한다. 같은 잡음이 학습 진행에 따라 어떻게 달라지는지 보기 위해서다 | U_trace | 7,168 (V-probe 포함) |
| **probe 궤적** | probe 0–7번, 후보 0번의 25 step x_t 상태 26장과 x̂₀ 25장 | u ∈ {0, 4,000, 40,000} | 2,448 (V-probe 포함) |

- x̂₀는 학습 graph 안의 출력과 같은 값이어야 한다. 구현은 같은 batch로 순전파를 한 번 더 하거나 aux 출력으로 꺼낸다. 어느 쪽이든 G2 게이트(§13)로 학습 결과가 변하지 않음을 확인한다.
- Train probe는 실제로 학습에 쓴 메시지다. 새 쌍 학습이므로 각 메시지는 한 번만 본다. Train과 Valid의 실질적 차이는 **target group을 학습 조건으로 본 적이 있는가**다.

### 6.4 Valid 저장 항목 (학습 run마다)

| 이름 | 내용 | 저장 시점 | 이미지 수 |
|---|---|---|---:|
| **V-loss** | V6 validation 진단(validation group 256쌍 = 512 메시지, `(t, ε)` draw 8개)의 정합 조건 x̂₀. Draw 0은 매 시점, draw 1–7은 최종 시점만 | U_trace | 10,752 |
| **V-probe 재구성·생성·궤적** | Validation group 고정 probe 64개(모든 seed 공통). T-probe와 같은 절차 | U_trace | (T-probe 행에 포함) |
| **입력 체크섬** | V-loss와 V-probe 입력(payload‖길이‖조건)의 SHA-256 | 매 시점 | — |

- V6와 같은 validation loss와 CLP 차이를 함께 기록한다. 4,000의 배수 시점에서는 V6와 같은 값이 나와야 한다(G1).
- 입력 체크섬으로 V6 추출 보고서의 "Valid 원 입력 체크섬 미저장" 문제를 없앤다.

### 6.5 TV-final — 최종 checkpoint의 Train·Valid 생성 평가

최종 checkpoint에서, Test 시행표를 만들기 전에 실행한다.

| 집합 | 메시지 | 조건 | 의미 |
|---|---|---|---|
| **TV-Train** | 마지막 4 update(39,996–39,999)의 batch 1,024행. 실제로 학습한 메시지 | 각 메시지의 W4 값(train group) | **seen target** |
| **TV-Valid** | Validation group 메시지 1,024개(모든 seed 공통 namespace) | 각 메시지의 W4 값(validation group) | **unseen target**, Test와 같은 조건 |

- 메시지마다 K=100 후보를 생성하고 V6 원장 형식으로 기록한다(run당 204,800 후보).
- 앞 32개 메시지의 후보 이미지 전부(3,200 × 2 집합)를 저장한다. 모든 hit 이미지를 저장한다.
- 지표: Success@1/@10/@100, 원본 메시지 재현(TV-Train은 학습 메시지 그 자체), 학습 메시지 일치, §10.4 보조 지표.
- **판정에 쓰지 않는다.** Seen target에서 hit가 많다면 암기나 group 단위 과적합을 뜻한다. 학습 일치 flag와 함께 보고한다.

### 6.6 저장 형식과 정밀도

- **이미지.** 채널 0만 float32로 저장한다(8,192 B/장). 채널 1과 비-payload 영역은 길이 L로 정확히 복원된다(`structure(L)`). 이 복원이 sampler 출력과 bit 단위로 같은지 G3로 확인한다.
- **파일.** Run·종류·시점별 chunk `.npy`(이미지)와 `.npz`(메타데이터: 메시지 bytes, 길이, 조건, t, 키 identity, ⑤·⑥, margin, 슬롯별 MSE 32개)를 쓴다. 모든 chunk의 SHA-256을 `trace-manifest.json`에 봉인한다. 기록 순서는 V6 segment와 같이 chunk(원자적, fsync) → manifest다.
- **⑤의 기준.** ⑤는 실행 중 float32 텐서로 decode한 값이다. 저장된 이미지를 다시 decode해도 같은 값이 나와야 한다(G3, 100%).
- **재개.** 추적 chunk도 checkpoint pointer와 같은 규칙으로 잘라내고 다시 쓴다. 재개한 run과 끊김 없는 run의 추적 manifest가 같아야 한다.

### 6.7 용량 (float32)

| 항목 | 이미지 수 | 용량 |
|---|---:|---:|
| 학습 run 1개의 Train·Valid 추적(§6.3–6.5) | 44,688 | 0.37 GB |
| Stage M 학습 6 runs | 268,128 | 2.2 GB |
| A-Q, P 학습 각 1 run | 89,376 | 0.73 GB |
| Stage M Test 추적(§7.4) | 252,384 | 2.07 GB |
| Stage P Test 추적(E-img 51,200, E-cond 25,600, E-traj 6,528 frames, E-hit) | 약 84,000 | 0.69 GB |
| 원장(M 14,745,600행, TV-final 1,228,800행, P 1,228,800행) | — | 0.62 GB |
| **합계** | | **약 6.4 GB** |

---

## 7. Test 설계: 무작위 메시지에서 목표 해시를 만든다

### 7.1 시행표 생성

모든 학습 checkpoint를 봉인한 **뒤에** 한 번만 만든다.

1. `generator = rng("test-messages", stage, window, rung)`(V7 identity).
2. Source prior에서 메시지 chunk를 뽑는다(길이 U{4..31}, byte U{33..126}). V6 `fresh_batch`와 같은 결정적 chunk rejection이다.
3. 각 메시지의 `H12^{W}`를 계산하고, test group에 속하는 것만 순서대로 남긴다. 같은 (길이, payload)가 이미 있으면 버린다.
4. T개가 모이면 멈춘다. 시행 t의 레코드는 (원본 메시지 x_t, 길이 L_t, MD5 128-bit digest, 목표 y_t = H12^W(x_t), group)이다.
5. 시행표 SHA-256을 checkpoint seal hash와 함께 봉인한다. 재추첨하지 않는다.

| Stage | Window | T | 예상 prior draw 수(수락률 25%) |
|---|---|---:|---:|
| M | W4, r=64 | 16,384 | 약 65,536 |
| R | W5, r=64 | 16,384 | 약 65,536 |
| P | W1, r=4 | 4,096 | 약 16,384 |

### 7.2 왜 test group으로 제한하는가

- 제한하지 않고 무작위 메시지를 뽑으면 목표의 68.75%가 train group, 6.25%가 validation group에 떨어진다. 그러면 Test가 "학습에서 조건으로 본 적 없는 target"을 묻지 못한다. Seen target 질문은 TV-final이 따로 다룬다.
- 제한하면 목표 y_t의 분포는 test group 위에서 거의 균일하다. 각 group의 prior 역상 수가 거의 같기 때문이다. 따라서 V6의 "test group에서 iid 추출"과 같은 target 분포를 유지하면서, 시행마다 실제 원본 메시지 하나가 생긴다.
- 원본 메시지는 학습 메시지와 겹칠 수 없다. 학습 메시지의 해시는 train group에만 있기 때문이다. 그래도 verifier가 학습 digest 저장소로 다시 확인한다.

### 7.3 비교군 (모든 방법이 같은 시행표를 앞에서부터 쓴다)

| 방법 | 정의 | Stage |
|---|---|---|
| Main | 실제 조건으로 학습한 모델에 y_t를 넣어 생성 | M, R, P |
| Shuffled | Batch 내 permutation으로 학습한 모델에 y_t를 넣어 생성 | M, R |
| Random | Source prior에서 직접 sampling. seed별 stream | M, R, P |
| MC | Main checkpoint에 다른 시행의 목표(고정 derangement)를 넣어 생성 | P |

- 후보 identity는 `(protocol, stage, pipeline, window, rung, method, seed, trial, attempt)`이고, key는 V6 `key_words` 규칙에 V7 identity를 쓴다.
- Success@100은 시행당 K=100 후보 중 `valid ∧ H12^W(후보) = y_t`가 하나 이상인지다. Invalid와 duplicate도 기회를 소비하고, 첫 성공 뒤에도 100개를 끝까지 만든다.
- 생성은 8,192 시행 블록 2개로 나눈다. 블록은 재개와 봉인 단위일 뿐 중간 분석(look)이 아니다.

### 7.4 Test 저장 항목

| 이름 | 범위 | 내용 | 이미지 수(M) |
|---|---|---|---:|
| **시행표** | 16,384 시행 전부 | ①·③과 group. ②는 ①에서 결정적으로 만든다 | — |
| **E-img** | 시행 0–255 × 후보 100 × 학습 stream 6개 | 최종 생성 이미지, ⑤, ⑥, margin, 슬롯별 MSE | 153,600 |
| **E-traj** | 시행 0–15 × 후보 0–3 × 학습 stream 6개 | 25 step 궤적(x_t 26장 + x̂₀ 25장) | 19,584 |
| **E-hit** | 모든 시행·stream의 hit 후보 | 최종 생성 이미지 | 약 2,400 |
| **E-cond** | 시행 0–255 × 후보 100 × Main 3 seeds | **Main과 같은 key**로 다른 시행의 목표(derangement)를 넣은 생성. 같은 잡음에서 조건만 바꾼 쌍이다 | 76,800 |
| **원장** | 모든 후보 | V6 36-byte 레코드 | — |

- 시행표가 무작위 순서이므로 시행 0–255는 무작위 표본이다. 결과를 보고 고른 것이 아니다.
- E-img 밖의 후보도 RNG identity로 이미지를 bit 단위로 다시 만들 수 있다(V6 batch 불변성). V7은 `trace regen` 명령을 제공하고, E-img 일부를 다시 만들어 저장본과 비교하는 것을 게이트로 둔다(G5).
- E-cond는 진단이다. 조건만 바꿨을 때 길이, 슬롯, 픽셀이 얼마나 바뀌는지 잰다(§10.4 S7).

### 7.5 해석상의 주의

- ⑤ = ①(원본 메시지 복원)은 성공 기준이 아니다. §1.3에 따라 이상적 모델도 거의 0이다.
- ⑥의 128-bit digest가 ③과 같으려면 사실상 ⑤ = ①이어야 한다. 다른 메시지로 128-bit가 같다면 MD5 충돌이며, 기대값은 0이다.
- 성공은 12-bit window 일치이고, 이는 원본과 무관한 다른 역상으로도 이루어진다. 보고서의 Test hit 예시는 "원본과 다르지만 같은 12 bits를 갖는 메시지"로 설명한다.

---

## 8. Stage A — 구현 게이트와 기계 확인 (MD5 조건 데이터 0)

### 8.1 A-impl

§13의 G1–G7을 모두 테스트로 통과해야 이후 단계를 시작한다. Metal 테스트는 `DHI_V7_REQUIRE_METAL=1`로 실행하고, skip을 통과로 보고하지 않는다.

### 8.2 A-prof

- **학습.** 추적 off/on 각각 update 속도를 데이터 생성과 기록 포함으로 잰다. 추적 부담이 5%를 넘으면 원인을 보고한다(설계 변경은 동결 전에만 가능).
- **생성.** Batch 256, S_G 25에서 burst 60 s를 잰다. 실제 Test 경로(생성 → decode → 해시 → 학습 집합 조회 → 원장 → E-img 저장 → 검증)를 10분 warm-up 뒤 15분 이상 연속 실행한다. Synthetic 조건과 폐기용 fixture window만 쓴다. 누적과 마지막 5분 중 느린 값을 예산에 쓴다.

### 8.3 A-Q (seed 0, 추적 on)

- 과제와 기준은 V6 §5.3과 같다: acceptance 512 조건, 정상·반전 joint 각각 ≥ 461/512, 반전 후보의 원래 조건 오성공 ≤ 25/512, CLP 4,096쌍 one-sided z > 3.26. S_G = 25.
- §6의 Train·Valid 추적을 모두 켠다. Synthetic 과제에서는 조건이 실제로 학습 가능하므로, 첫 3자의 glyph가 조건에 따라 나타나는 추론 이미지를 얻는다. 보고서의 "조건이 작동할 때" 예시로 쓴다.
- **실패하면 멈춘다.** V6에서 같은 정의가 512/512였으므로, 실패는 V7 구현의 회귀로 본다. 보완 학습(80k)은 하지 않는다. 원인을 고친 뒤 A-impl부터 다시 한다.

### 8.4 예산 판단

A-prof 실측으로 Stage M 예측 시간을 계산한다. 예측 × 1.5 ≤ M cap이면 진행한다. 넘으면 **MD5 조건 데이터를 만들기 전에** 멈추고 사람에게 cap 증액을 묻는다. 시행 수, K, seed, updates, δ는 줄이지 않는다.

### 8.5 동결

`protocol.frozen.json`(모든 설정과 추적 범위), `window.json`(W4·W5·W1 r=4 split과 감사 hash), `budget-plan.json`, code·환경 manifest, `trace-policy.json`(§6·§7.4의 저장 범위)을 봉인한다.

---

## 9. Stage M — W4 주 실험

| 순서 | 작업 | 산출물 |
|---:|---|---|
| M1 | Main·Shuffled × seeds {0, 1, 2} = 6 runs 학습. W4 train group, 추적 on | run별 checkpoint, segment, Train·Valid 추적 |
| M2 | Run별 TV-final(§6.5) | TV-final 원장과 추적 |
| M3 | 6개 checkpoint 봉인 | `checkpoints.seal.json` |
| M4 | 시행표 16,384개 생성·봉인(§7.1) | `trials.json`, SHA-256 |
| M5 | CLP_64: Main seed별 test group held-out 65,536쌍 | `clp.json` |
| M6 | 생성: 블록 1–2 × (Main 3, Shuffled 3, Random 3) streams. E-img, E-traj, E-hit 저장. 블록마다 검증과 1% 재생성 감사 | 원장, manifest, 추적 |
| M7 | E-cond(시행 0–255) | 추적 |
| M8 | 분석 1회(§10.1), 판정. `POSITIVE`이면 감사(§10.3) 뒤 Stage R | `M.json` |

- M6이 끝나기 전에는 hit 집계와 구간을 화면에 보이지 않는다. 진행률, 자원, 무결성 오류만 표시한다.
- M1–M3에서 Test 시행표와 test group 조건 데이터는 만들지 않는다. TV-final은 train·validation group만 쓴다.

---

## 10. 통계와 판정

### 10.1 주 판정 (Stage M)

시행 t, seed s, 대조군 c ∈ {Random, Shuffled}에 대해 `d_t = mean_s (M_{s,t} − C_{s,t})`로 둔다. T = 16,384, `Δ̂ = mean_t d_t`, `SE = sd(d)/√T`다.

- **오류 배분.** α = 0.05를 대조 2개 × 양측의 4개 꼬리로 나눈다(꼬리당 0.0125). `z = Φ⁻¹(1 − 0.0125) = 2.2414`, `[L, U] = Δ̂ ∓ z·SE`. 두 대조의 구간이 동시에 참값을 포함할 확률은 95% 이상이다.
- **최소 관심 효과** δ = 0.005(0.5%p). V6와 같다.
- **분석은 한 번이다.** 예산 때문에 블록 2를 끝내지 못하면, 모든 stream이 완료한 블록 1(T = 8,192)로 같은 z를 써서 한 번 분석한다. 분석 시점이 자료가 아니라 예산으로 정해지므로 구간은 유효하다.

| 순서 | 조건 | 분류 |
|---:|---|---|
| 1 | `L_R > 0` 그리고 `L_S > 0` | `POSITIVE` |
| 2 | `U_R < δ` 그리고 `U_S < δ` | `REJECTED_BOUNDED` |
| 3 | `U_S < δ` | `REJECTED_NO_CONDITION_GAIN` |
| 4 | `U_R < δ` | `REJECTED_NO_RANDOM_ADVANTAGE` |
| 5 | 그 외 | T = 16,384이면 `NOT_ESTABLISHED_UNRESOLVED`, 예산 축소 분석이면 `NOT_ESTABLISHED_BY_BUDGET` |

### 10.2 설계 검산

p₀ = 2.4121%, seed별 Bernoulli 독립 가정에서 `sd(d) = √(2p₀(1−p₀)/3) = 0.12527`이다.

| T | SE | 반폭(z = 2.2414) | 무효과에서 대조 하나가 δ를 배제할 확률 | +δ에서 대조 하나가 L > 0일 확률 |
|---:|---:|---:|---:|---:|
| 8,192 | 0.138%p | ±0.310%p | 91.5% | 91.5% |
| **16,384** | **0.098%p** | **±0.219%p** | **99.79%** | **99.79%** |
| 24,576 | 0.080%p | ±0.179%p | 99.997% | 99.997% |

판정 규칙 전체를 시나리오별 2,000회 모의했다(seed 3개, 대조군 stream 독립, T = 16,384).

| 가정(seed별 Success@100) | POSITIVE | REJECTED_BOUNDED | NO_CONDITION_GAIN | NO_RANDOM_ADVANTAGE | 미결 |
|---|---:|---:|---:|---:|---:|
| 모두 p₀ | 0.10% | 99.55% | 0.30% | 0.05% | 0% |
| Main·Shuffled −0.1%p(중복 손실) | 0.05% | 99.80% | 0% | 0.15% | 0% |
| Main +δ | 99.15% | 0.10% | 0.45% | 0.30% | 0% |
| Main +δ/2 | 43.9% | 40.0% | 7.55% | 8.55% | 0% |
| Main = Shuffled = p₀ + δ(prior 모델링 이득) | 1.15% | 1.05% | 97.8% | 0% | 0% |
| 모두 p₀, 블록 1만 완료(T = 8,192) | 0.10% | 85.3% | 6.0% | 5.6% | 3.05% |

구현 뒤에는 생산 판정 코드로 같은 시나리오를 다시 실행한다(G6).

### 10.3 POSITIVE 후 artifact 감사

V6 §6.5와 같다. 모든 성공 payload를 독립 verifier로 재해시한다. 성공 후보 중 학습 메시지와 같은 것은 0이어야 한다. 성공이 특정 target에 몰렸는지 보고한다(상위 1% target의 성공 비중). CLP_64 방향을 보고하고, CLP는 무신호인데 hit만 양성이면 `hit-only`로 표시한다. V7에서는 E-hit 이미지의 decode가 원장 payload와 같은지도 확인한다. 감사 실패는 POSITIVE를 무효화하고, 고칠 수 없으면 `NOT_ESTABLISHED_INTEGRITY`다.

### 10.4 보조 분석 (판정에 쓰지 않음)

모든 지표를 Main, Shuffled, Random별로 계산하고, Main−Random과 Main−Shuffled의 시행 단위 paired 차이에 99% 구간(Bonferroni 없이, 기술통계로 표시)을 붙인다. Stage M의 Test, TV-Train, TV-Valid, Stage P에서 같은 표를 만든다.

| 번호 | 지표 | 무효과 기대값 | 근거 |
|---|---|---|---|
| S1 | **12-bit 일치 bit 수** = 12 − popcount(H12(후보) XOR y). 후보 평균 | 6.0 | 모든 후보를 쓰므로 hit보다 민감한 연속 지표다 |
| S2 | **원본 근접성**: 길이 일치율, 위치별 byte 일치율(위치 < min(L, L_orig)), 앞 4 byte 일치, 원본 메시지 복원 수 | 1/28 = 3.571%, 1/94 = 1.064%, —, 0 | 원본 길이와 byte가 조건과 독립이면 **모델 분포와 무관하게** 이 값이 정확한 기대값이다 |
| S3 | **128-bit digest**: 후보 MD5와 원본 MD5의 Hamming 거리, 128-bit 일치 수 | 64, 0 | 일치는 원본 복원과 같다(§7.5) |
| S4 | **CLP_64**: Main seed별·합산 z. 합산 one-sided z > 2.326이면 "이상 신호" | z ≈ 0 | V6 §10 |
| S5 | **Seen/unseen**: TV-Train, TV-Valid, Test의 Success@100과 S1 | 모두 p₀, 6.0 | Seen target 이득은 암기나 group 과적합을 뜻한다 |
| S6 | **암기**: 학습 메시지 일치율, TV-Train에서 원본 학습 메시지 재현 수 | 짧은 길이에서 우연 일치만 | V6 C는 0.0166% |
| S7 | **조건 민감도(E-cond)**: 같은 key에서 조건만 바꿨을 때 길이 변화율, 바뀐 슬롯 비율, 픽셀 L2 거리 | r=64에서 hit와 무관 | Stage P와 대비한다 |
| S8 | **재구성 정확도**: T-probe·V-probe의 t별 문자 정확도와 원본 복원율의 학습 곡선 | Train ≈ Valid | 재구성은 해시와 거의 무관하다 |
| S9 | **후보 품질**: strict-decoder valid, margin 분포, trial 내 중복, 위치별 byte entropy | V6 C와 비슷 | V6 §11.1 |

---

## 11. Stage R — W5 재현 (Stage M이 POSITIVE이고 감사를 통과한 경우만)

- **설계.** W5 r=64, 새 split. Main·Shuffled × seeds {0, 1, 2}(R namespace)를 같은 규약과 추적으로 학습한다. 시행표 16,384개(§7.1), K=100, Random을 생성한다.
- **판정.** `L_R > 0` 그리고 `L_S > 0`을 각각 one-sided z = 1.96으로 본다. 통과하면 Q4 = `SUPPORTED`, 실패하면 `NOT_ESTABLISHED_NOT_REPLICATED`다. +δ 효과에서 대조 하나의 검정력은 99.9%다.
- **비용.** 약 8.2 h(학습 4.8 h + Test 3.4 h).

---

## 12. Stage P — r=4 양성 대조

| 항목 | 값 |
|---|---|
| 해시 | `H12_4^{W1}`(payload bytes 0–3만의 함수), V7 split |
| Run | Main seed 0, 40,000 updates, 추적 on(§6) |
| Test | 무작위 메시지 시행표 4,096개(§7.1). Main, MC, Random × K=100. CLP_4 16,384쌍 |
| 추적 | E-img(시행 0–255, Main·MC), E-cond(시행 0–255), E-traj, E-hit |
| GEN_4 | Main−Random **그리고** Main−MC의 one-sided z > 2.326(α = 0.01, intersection-union) |
| INFO_4 | CLP_4 one-sided z > 2.326 |
| 검정력 | 90% 검정력으로 검출 가능한 차이 약 1.22%p. V6의 효과는 약 13%p |
| 비용 | 약 1.2 h |

- Stage P 결과는 Q4 판정을 바꾸지 않는다. Stage M 뒤에 실행한다.
- **시각적 대조가 목적이다.** r=4에서는 조건이 앞 4 byte를 실제로 제약하므로, 추론 이미지와 E-cond 쌍에서 조건에 따라 앞 4개 glyph가 체계적으로 바뀔 것으로 예상한다. 같은 기계가 r=64에서는 그런 구조를 보이지 않는다는 비교가 V7 보고서의 핵심 그림이다.

---

## 13. 무결성·재현 게이트

| 번호 | 게이트 | 통과 기준 |
|---|---|---|
| **G1 V6 등가** | 같은 가중치·key·identity를 주입했을 때 V7 경로가 `dhi_v6`과 같은 결과를 내는가 | 순전파, 8 update 뒤 parameter, segment `data_sha256`, DDIM 최종 이미지, decode, 원장 레코드가 bit 단위로 같다. 4,000 배수 시점의 validation loss와 CLP가 같다 |
| **G2 추적 비간섭** | 추적 on/off가 측정을 바꾸지 않는가 | 학습 parameter, Adam 상태, segment, 후보, 원장이 bit 단위로 같다(축소판: 학습 500 update, 생성 4,096 후보) |
| **G3 추적 충실도** | 저장 이미지가 원장·⑤와 맞는가 | 저장 이미지를 decode한 결과가 ⑤와 100% 같다. 채널 1·비-payload 복원이 sampler 출력과 같다. 궤적의 마지막 x 상태가 최종 이미지와 같다 |
| **G4 시행표** | 시행표가 계약을 지키는가 | 결정적이고, 모든 원본의 H12가 test group에 있고 y_t와 같다. 중복이 없다. 학습 digest와 겹치지 않는다. Verifier가 독립 구현(`hashlib`)으로 재계산한다 |
| **G5 재생성** | 저장하지 않은 후보도 다시 만들 수 있는가 | V6 1% 재생성 감사(payload bit 일치)에 더해, E-img에서 결정적으로 고른 1%를 다시 생성해 이미지가 bit 단위로 같다 |
| **G6 판정기 calibration** | 생산 판정 코드가 §10.2를 재현하는가 | 시나리오별 2,000회 이상. 무효과 `POSITIVE` 비율의 one-sided 95% CP 상한 ≤ 0.025, 무효과 `REJECTED_*` 하한 ≥ 0.95, Main +δ `POSITIVE` 하한 ≥ 0.95 |
| **G7 planted-lift** | 실제 원장 경로에서 효과를 잡는가 | W4 **validation** group의 사전 계산 역상을 Random 후보에 섞는다. +0은 `REJECTED_*`, +δ는 `POSITIVE`. Test group은 쓰지 않는다 |
| **G8 원장·검증** | V6 계약 | V6 명세 §10.2–10.5: 36-byte 레코드, 독립 재해시, trial 요약 재계산, 학습 일치 조회, 성공 후보의 학습 메시지 일치 0 |

`TRACE_COMPLETE`의 조건은 다음 세 가지다. (1) G2–G5 통과. (2) `trace-policy.json`에 등록된 모든 항목이 존재하고 manifest hash가 맞음. (3) §17 보고서의 6열 표에서 Train·Valid·Test 어느 칸에도 "미저장"이 없음. Random의 ④ "해당 없음(모델 없음)"은 결손으로 보지 않는다.

---

## 14. 실행 순서, 시간, 용량, cap

**순서.** A-impl → A-prof → A-Q(seed 0) → 노출 감사 → 예산 판단 → **동결** → M1 학습 → M2 TV-final → M3 봉인 → M4 시행표 → M5 CLP → M6 Test 생성 → M7 E-cond → M8 판정 → (감사 → R) → P → 보고서.

**예상 시간** (M3 Max, V6 실측 기준: 학습 0.0694 s/update, 생성 863 후보/s):

| 단계 | 내용 | 예상 | Cap |
|---|---|---:|---:|
| A | A-impl 테스트 0.5 h, A-prof 0.7 h, A-Q 학습 0.8 h와 평가 0.1 h | 약 2.1 h | 4 h |
| M | 학습 6 runs 4.8 h, TV-final 1,228,800 후보 0.4 h, CLP 0.05 h, Test 9,830,400 후보 3.2 h, 검증·재생성·E-cond 0.25 h | 약 8.7 h | 13 h |
| P | 학습 0.8 h, Test 819,200 후보 0.27 h, CLP·E-cond 0.05 h | 약 1.2 h | 3 h |
| R (조건부) | 학습 6 runs 4.8 h, Test 3.4 h | 약 8.2 h | 13 h |
| **필수 경로 (A, M, P)** | | **약 12 h** | **20 h** |

- Random 후보 생성과 검증은 CPU에서 병행할 수 있다(prior + MD5 약 140만/s). GPU 작업은 한 번에 하나만 돌린다.
- **용량.** 약 6.4 GB(§6.7). 저장 cap 32 GiB, RSS·GPU cap 각 64 GiB, 최소 디스크 여유 20 GiB.
- **실행 중 cap 초과.** M은 §10.1의 예산 축소 분석을 따른다. P가 미완이면 "부분 측정"으로 보고한다. 고립된 실행 오류는 같은 난수 identity로 한 번 재실행할 수 있다.

**위험과 대응**

| 위험 | 대응 |
|---|---|
| x̂₀ 포착이 학습 graph를 바꿔 결과가 달라짐 | G2. 실패하면 별도 순전파 방식으로 바꾸고, 그 순전파가 graph 안의 출력과 같은지 기록한다 |
| 추적 I/O가 학습·생성을 늦춤 | A-prof에서 on/off 비교. 저장은 별도 thread에서 하되 commit 순서는 지킨다 |
| 궤적 포착용 sampler가 V6 sampler와 달라짐 | G1·G3. 궤적 sampler의 최종 이미지 = `dhi_v6.models.sample_images` 출력 |
| float32 이미지 용량 | 범위를 §6·§7.4로 고정. 6.4 GB |
| 시행표 rejection의 편향 | Target 분포를 group별 빈도로 보고하고 균일성 χ² 검정을 기술통계로 싣는다 |

---

## 15. 최종 결론 규칙

### 15.1 Q4 판정 값

`SUPPORTED`, `REJECTED_BOUNDED`, `REJECTED_NO_CONDITION_GAIN`, `REJECTED_NO_RANDOM_ADVANTAGE`, `NOT_ESTABLISHED_UNRESOLVED`, `NOT_ESTABLISHED_NOT_REPLICATED`, `NOT_ESTABLISHED_INTEGRITY`, `NOT_ESTABLISHED_BY_BUDGET`, `UNTESTABLE`(A-Q 실패로 M을 시작하지 못함).

### 15.2 종합 판정과 사전 작성 문장

| 종합 | 조건 | 결론 문장 |
|---|---|---|
| **`V7_SUPPORTED`** | Q4 = `SUPPORTED` | "P-G-CGGE 파이프라인에서 무작위 Printable 메시지의 MD5 12-bit window W4 값을 목표로 했을 때, 해시 조건 모델은 Random과 Shuffled 대비 Success@100 이득을 보였고({구간}), 이 이득은 W5에서 재현되었다. 이 결과는 W3에서 이득을 배제한 V6 판정과 충돌하며, 두 자료는 합치지 않고 나란히 보고한다. Full MD5 역상, 원본 메시지 복원, 계산 우위는 주장하지 않는다. 추적은 {Q1}이다." |
| **`V7_REJECTED`** | Q4 = `REJECTED_*` | "P-G-CGGE 파이프라인에서 무작위 Printable 메시지의 MD5 12-bit window W4 값을 목표로 한 Test(16,384 시행 × K=100 × 3 seeds)에서, 해시 조건 생성의 Success@100 이득은 0.5%p 미만으로 배제되었다(U_R {…}, U_S {…}). 생성 후보는 원본 메시지와 우연 이상으로 닮지 않았다(길이 일치 {…} vs 1/28, byte 일치 {…} vs 1/94, 12-bit 일치 {…} vs 6.0 bits). 같은 기계는 synthetic 조건을 {A-Q}로 사용했고, r=4에서는 {GEN_4·INFO_4}였다. Train·Valid·Test의 추론 이미지 추적은 {Q1}이다. 이 결과는 V6 판정과 일치한다." |
| **`V7_NOT_ESTABLISHED`** | 그 외 | "사전 규칙으로 판정하지 못했다. 사유는 {사유 코드}이고, 측정된 경우 이득의 상한은 {구간}이다. 추적은 {Q1}이다." |

Q1이 `TRACE_INCOMPLETE`여도 Q4 판정은 유효하다. 다만 V7의 첫째 목표를 달성하지 못한 것이므로 결손 항목과 원인을 보고서 첫 장에 쓴다.

---

## 16. 구현 범위와 산출물

### 16.1 패키지

새 package `src/dhi_v7/`를 만든다. `dhi_v6`은 읽기 전용으로 import한다.

| 모듈 | 내용 |
|---|---|
| `__init__.py` | `PROTOCOL = "dhi-v7-20261008"`, `MASTER_SEED` |
| `protocol.py` | 등록값(`examples/v7-protocol.json` + SHA-256), 수정된 JSON 거부, 동결 |
| `data.py` | V7 identity·rng·key_words, split, 학습 stream, **시행표 생성**, W5 추출. 해시·digest·조건 bit은 `dhi_v6.data`를 쓴다. V6 알고리즘을 V7 identity로 쓰되, V6 identity를 주입하면 V6와 같은 출력을 내야 한다(G1) |
| `trace.py` | 추적 저장소: chunk writer/reader, manifest, 채널 복원, PNG 출력(`zlib`·`struct`), `trace regen` |
| `train.py` | 추적 학습 루프: `dhi_v6.models`의 G3-U, corruption, loss를 쓰고 x̂₀를 포착한다. T-step, T-ckpt, probe, V-loss, TV-final |
| `sample.py` | 궤적을 포착하는 DDIM sampler(최종 결과는 `dhi_v6.models.sample_images`와 bit 단위로 같음) |
| `evaluate.py` | Test stream, V6 원장 레코드, 독립 verifier, 재생성 감사, E-img·E-traj·E-hit·E-cond |
| `statistics.py` | §10.1 판정, R, P 검정, §10.4 보조 분석, G6 calibration, G7 fixture |
| `study.py` | CLI `plan \| audit \| run --stage {A,M,R,P} \| report \| trace {export,regen}` |
| `report.py` | `FINAL_REPORT_KO.md`, `TRACE_REPORT_KO.html` |
| `tests/test_study_v7.py` | G1–G8 |

실행 형태: `PYTHONPATH=src .venv/bin/python -m dhi_v7.study ...`

### 16.2 다음에 작성할 문서

1. `V7_IMPLEMENTATION_SPEC.md`: namespace 표, 추적 chunk schema, 시행표 알고리즘의 정확한 chunk 크기, CLI, 테스트 목록.
2. `examples/v7-protocol.json`과 SHA-256 파일.
3. `CODEX_HANDOFF_V7.md`와 `AGENTS.md` 갱신(현재 V6 규칙을 가리킴).

### 16.3 실행 산출물 (커밋하지 않음)

`protocol.frozen.json`, `window.json`, `budget-plan.json`, `trace-policy.json`, 노출 감사 기록, A-impl 결과, A-prof, A-Q, run별 checkpoint·segment·추적·manifest, TV-final, `checkpoints.seal.json`, `trials.json`, 원장과 verifier 결과, `M.json`·`P.json`(·`R.json`), `decision.json`, `FINAL_REPORT_KO.md`, `TRACE_REPORT_KO.html`.

---

## 17. 보고서 구성

### 17.1 `FINAL_REPORT_KO.md`

1. 최종 결론 카드(§0)와 종합 판정
2. V6 대비 변경점과 V6 결과 요약(§1)
3. Q2: A-Q 결과
4. Q4: 대조별 추정값·구간·seed별 값, 판정, 감사, R 결과(해당 시), CLP_64
5. Q3: r=4의 GEN_4·INFO_4와 효과 크기
6. Q5: §10.4 표(Test, TV-Train, TV-Valid, P)
7. Q1: 추적 범위, 게이트 결과, manifest 요약
8. 적용 범위: P-G-CGGE 하나, Printable, W4(재현 시 W5), K=100, 등록 학습량. 주장하지 않는 것은 V6 §15.4와 같다

### 17.2 `TRACE_REPORT_KO.html` (6열 추적 보고서)

2026-10-08 추출 보고서와 같은 6열 형식을 쓰되, "미저장" 칸이 없어야 한다. 사례는 다음 규칙으로 고정한다.

| 구간 | 사전 지정 사례 | 결과 기반 사례(표시함) |
|---|---|---|
| Train (가) 재구성 | Main seed별 T-step u = 39,900의 0–2번 행. t와 ᾱ_t를 함께 표시 | — |
| Train (나) 생성 | Main seed별 T-probe 0–2번, 후보 0번(최종 시점) + 궤적 띠 | TV-Train의 첫 hit |
| Valid (가) 재구성 | Main seed별 V-loss 0–2번 행 draw 0(최종 시점) | — |
| Valid (나) 생성 | V-probe 0–2번, 후보 0번 + 궤적 띠 | TV-Valid의 첫 hit |
| Test | 시행 0–2의 Main·Shuffled seed 0 후보 0번, Random 후보 0번(인코딩 이미지, 추론 아님) + E-traj 1개 + E-cond 쌍 1개 | Main seed별 첫 hit 시행의 hit 후보와 같은 시행의 후보 0번 |
| P (r=4) | 위 Test와 같은 규칙 | 같음 |
| A-Q (synthetic) | 조건이 작동하는 예: 정상·반전 조건 각 2개 | — |

각 사례에는 ①–⑥과 함께 다음을 싣는다: 메시지의 JSON 표기와 hex, 길이, MD5 128-bit, W4 12-bit 값과 일치 bit 수, margin과 strict 여부, 근거 파일 경로와 SHA-256. 이미지는 내용 채널과 mask를 보간 없이 확대한다(검정 = −1, 흰색 = +1). 학습 진행 그림(같은 잡음의 T-probe 생성 14개 시점)과 t별 재구성 그림을 함께 싣는다.
