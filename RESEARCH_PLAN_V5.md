# v5 실험 계획 — 연구의 최종 결론을 내리는 실험

**Protocol:** `dhi-v5-20260928` · **Master seed:** `2026092806` · **작성일:** 2026-09-28 KST
**상태:** `PLANNED_NOT_IMPLEMENTED` · 다른 작업자에게 넘길 때는 [인계 안내](#인계-안내-codex-등-다른-작업자용)부터 읽는다.

이 문서의 수치는 세 종류로 나뉜다.

- **기존 archive에서 읽은 값.**
- **계획 작성 중 새로 측정한 값.** §1.2의 D0 진단과 §1.3의 처리량 측정이다. Synthetic 모델만 사용했으며 MD5 호출은 0회다. v4 산출물은 읽기만 했다.
- **설계 검산값.** [검산 스크립트](scripts/validate_research_plan_v5.py)의 가상 계산 결과다.

새 MD5 test 접근, v4 산출물 변경, v4-claude 구현은 하지 않았다.

**요약.** v5는 이 연구의 마지막 버전이다. 결과가 무엇이든 **연구 가설에 대한 최종 판정을 내리고 연구를 종료**하도록 설계했다. 이전 버전은 모두 결론 없이 끝났다(형식 실패, 적격성 차단, 노출 감사 미완, 자원 한도). 이 계획은 그 경로를 하나씩 막는다.

1. **v4 차단의 원인을 확인했다(D0).** 같은 v4 모델의 epoch-100 checkpoint는 조건 정확도 460–469/512를 냈다. 등록 규칙이 고른 epoch 11은 125–176/512였다. 기계는 작동했다. 부족했던 것은 checkpoint 선택 규칙이다. 그래서 v5는 매 update마다 새로 해시한 쌍으로 학습하고, 최종 checkpoint를 고정 사용한다.
2. **Sampler를 벡터화한다.** 현재 sampler는 후보마다 난수를 Python loop로 뽑는다. 그래서 batch를 키워도 약 450 후보/s에 머문다. `mx.vmap` 버전은 기존 출력과 **bitwise로 같고** 약 17,500 후보/s(37배)다. 이 여유로 주 시험의 검정력을 4배로 높였다. 최소 관심 효과는 0.5%p에서 **0.25%p**로, trials는 16,384에서 **65,536**으로 바뀐다.
3. **양성 결과도 최종 결론이 되게 한다.** 양성이면 다른 digest window에서 내부 재현(Stage R)을 해야 "지지"로 판정한다. 계산 우위(C4)는 분석식과 실측 처리량으로 함께 판정한다.
4. **"더 큰 모델이면?"이라는 반론에 답한다.** 3.5배 큰 모델에 4배 많은 데이터를 쓰는 규모 탐침(Stage S)을 추가했다.

관련 문서: [원본 계획](RESEARCH_PLAN.md) · [v4 계획](RESEARCH_PLAN_V4.md) · [v4-claude 계획](RESEARCH_PLAN_V4_CLAUDE.md) · [연구 가능성 검토](RESEARCH_FEASIBILITY_REVIEW_KO.md) · [설계 계산 JSON](local_experiment_archive/analyses/v5-design-20260928/design_calculation.json)(로컬 전용) · [D0·처리량 측정](local_experiment_archive/analyses/v5-d0-20260928/)(로컬 전용)

---

## 인계 안내 (Codex 등 다른 작업자용)

이 절은 이 계획을 저장소만 받은 작업자에게 넘기기 위한 요약이다. 명세의 기준은 이 문서의 §3–§16이다. v4-claude 계획과 충돌하면 이 문서가 우선한다.

### 현재 상태 (2026-09-28)

| 항목 | 상태 |
|---|---|
| 계획 문서, 설계 검산 스크립트 | 완료, 커밋됨 |
| D0 진단, 처리량 측정 | 완료. 결과는 로컬 archive에만 있고, 수치는 §1.2·§1.3에 전부 인용됨 |
| §15의 구현 | **시작 전** |
| 노출 감사(§12.1) | **시작 전.** `examples/v4-exposure-inventory.json`은 빈 템플릿 |
| Stage A–S 실행 | **시작 전.** MD5 조건 데이터는 아직 하나도 만들지 않음 |

### 저장소에 없는 자료

`local_experiment_archive/`는 `.gitignore` 대상이다. 이 문서에서 그 아래를 가리키는 링크는 모두 작성자 컴퓨터에만 있다. 필요한 값은 다음 위치에 인용되어 있다.

| 로컬 전용 자료 | 문서 안의 위치 |
|---|---|
| v4 V1 실행 기록, D0 재평가 | §1.1, §1.2 |
| 처리량, 벡터화 sampler 시제품 결과, Transformer proxy 시간 | §1.3 |
| 설계 계산 JSON | §6.3, §8.3, §11, §13 |
| MD5 PoC·toy 결과 | §1.1, [연구 가능성 검토](RESEARCH_FEASIBILITY_REVIEW_KO.md) |

[검산 스크립트](scripts/validate_research_plan_v5.py)는 archive가 없으면 기록된 측정값을 대신 쓴다. 출력의 `measurement_sources`에 어느 쪽을 썼는지 남긴다. 두 경우의 출력은 같다.

### 착수 전에 사람이 정할 항목

1. **주 window(§12.1).** 노출 감사를 수행해 W1을 쓸지, 감사 없이 W2를 쓸지 정한다. 이 결정을 `window.json`에 기록하기 전에는 MD5 조건 데이터를 만들지 않는다.
2. **모델 코드.** §15는 기존 `mlx_models`의 D1-S를 재사용한다고 되어 있다. 모델을 MLX로 새로 구현하기로 하면, §5.1의 "기존 sampler와 bitwise 동일" 게이트를 무엇으로 대체할지 먼저 정한다.
3. **자원.** 실행 기계(Apple Silicon, MLX/Metal)와 §13의 hard cap(필수 경로 50시간)을 확인한다.
4. **커밋 범위.** [NEW_EXPERIMENT_PLAN.md](NEW_EXPERIMENT_PLAN.md)의 규칙을 따른다. 코드, 테스트, 고정 설정, 계획만 커밋한다. 후보 원장, 배열, checkpoint, 결과 보고서는 커밋하지 않는다.

### 반드시 지킬 제약

- v3.1·v4 실행기와 `local_experiment_archive/runs/` 아래 산출물은 수정하지 않는다. 기존 `mlx_models.sample`도 바꾸지 않는다(§15).
- 결과를 본 뒤 규칙, seed, 학습량, sampler, trial을 바꾸지 않는다. 사전 등록된 분기만 허용한다(§3 원칙 2).
- Stage C를 B·S보다 먼저 실행한다(§13).
- 실패, 차단, 자원 초과를 MD5 효과의 증거로 해석하지 않는다. §14.1의 `NOT_ESTABLISHED_*`로 기록한다.

### 작업 순서와 완료 기준

| 순서 | 작업 | 완료 기준 |
|---|---|---|
| 1 | §15의 구현 항목 1–7 | §5.1의 A-impl 게이트와 §12.2의 무결성 검사를 테스트로 통과 |
| 2 | 판정기 calibration | §12.3의 통과 조건과 planted-lift fixture 통과 |
| 3 | A-prof, A-dev, 봉인 | `protocol.frozen.json`, `window.json`, fallback 기록(§13) |
| 4 | A-Q (필요 시 §5.4 보완) | Q와 A\* 결정 |
| 5 | C → (확장) → (R) → B → S | 각 단계 봉인과 §16의 산출물 |
| 6 | 최종 보고 | §14.2 구성의 `FINAL_REPORT_KO.md`, `decision.json` |

---

## 0. 최종 결론 카드 (실험 종료 시 이 표를 채운다)

| 질문 | 판정 값 | 근거 단계 |
|---|---|---|
| **C1 기계.** 조건부 생성 기계가 학습 가능한 조건을 실제로 사용하는가 | `PASS` / `FAIL` | Stage A |
| **C2 구조 이용 한계.** MD5 압축함수를 몇 step까지 줄여야 모델이 해시 구조를 이용하는가. 규모를 키우면 그 한계가 움직이는가 | `r*_gen`, `r*_info`, `SCALE_SHIFT` 여부, `PIVOT_*` | Stage B, S |
| **C3 연구 가설.** 미학습 MD5-12 target에서 해시 조건 모델이 Random과 Shuffled를 모두 능가하는가 | `SUPPORTED` / `REJECTED_*` / `NOT_ESTABLISHED_*` + 효과 구간 | Stage C, R |
| **C4 계산 우위.** 같은 계산 비용에서 MD5 직접 시도보다 빨리 역상을 찾는가 | `NO_ADVANTAGE` / `PER_QUERY_ADVANTAGE` | §11 분석식 + A-prof 실측 |
| **종합.** 연구 종료 판정 | §14.1의 세 가지 중 하나 | 위 네 줄 |

**사전 예측**(판정 규칙에 영향 없음): C1 `PASS`, C2 `r*_gen` ∈ {4, 5, 6}이고 `SCALE_SHIFT` 없음, C3 `REJECTED_BOUNDED`(상한 약 0.12%p), C4 `NO_ADVANTAGE`. 근거는 §1.4의 이론적 사전 기대와 §8.2의 모델 없는 난이도 profile이다. 이 예측이 맞을 가능성이 높다는 점이 이 실험을 무의미하게 만들지는 않는다. 이 실험의 목적은 그 예측을 **기계 결함·검정력 부족·규모 부족이라는 반론이 닿지 않는 형태로** 확정하는 것이다.

---

## 1. 지금까지의 증거

### 1.1 버전별 결산

| 버전 | 결과 | 결론이 나지 않은 이유 |
|---|---|---|
| 2026-09-21 MD5 PoC (q12) | Printable Main 0/305, Random 9/305, Shuffled 0/305. Random Bytes도 같은 양상 | Gaussian Main의 valid가 1/247,500이었다. 형식 실패 때문에 해시 신호를 분리할 수 없었다 |
| Toy MD5-8 `ABCD^4` | Diffusion 3/18, Source-prior 7/18 | 표본이 작다. 우위의 증거는 없다 |
| v3 / v3.1 (P2 계열) | P-DISC 123/125, R-G-BGV 128/128. Gaussian Printable 두 pipeline과 R-DISC는 미달 | 다섯 pipeline 적격성을 완료하지 못했다. 위치 특혜 loss(p2struct)는 synthetic 전용이다 |
| v4 | V0 PASS, V1 `BLOCKED_QUALIFICATION`, 과학적 판정 `NOT_EVALUATED` | Checkpoint 선택 규칙 때문에 적격성 미달(§1.2). 노출 감사 inventory도 미작성 상태 |
| v4-claude | 계획만 있고 구현하지 않음 | 이 문서가 대체한다. 설계 요소는 대부분 이어받는다(§2) |

### 1.2 D0 진단 — v4 V1 실패의 원인 (신규 실측)

v4 V1의 세 seed checkpoint를 V1 acceptance 512 조건에 대해 등록 sampler와 batch 4로 다시 평가했다. v4 study 폴더는 읽기만 했다. 결과는 [`d0_summary.json`](local_experiment_archive/analyses/v5-d0-20260928/d0_summary.json)(로컬 전용)에 있다.

| Seed | Epoch 11 (등록 규칙이 선택): 정상 / 반전 joint | Epoch 100 (최종): 정상 / 반전 joint | 기준 |
|---|---:|---:|---:|
| 0 | 176 / 167 | **469 / 460** | ≥ 461 |
| 1 | 160 / 151 | **469 / 469** | ≥ 461 |
| 2 | 125 / 134 | **462 / 466** | ≥ 461 |

- Epoch 11 수치는 저장된 V1 결과와 정확히 같다. 재평가 경로가 등록 경로와 동일하다는 확인이다.
- 모든 평가에서 valid 512/512, 반전 후보의 원래 조건 오성공 0, MD5 호출 0이었다.
- **해석:** Validation loss는 suffix 암기 때문에 epoch 10 이후 증가했다. 반면 조건 정확도는 학습 내내 올라갔다. "최소 validation loss" 규칙이 조건 학습이 가장 덜 된 시점을 골랐다. v4-claude §1.2의 추정이 맞았다.
- **남는 위험:** Epoch 100에서도 seed 0 반전이 460으로 기준에 1 부족하다. 균일 loss의 D1-S가 v4 데이터량(update 15,700 × batch 64 ≈ 1.0M 쌍)에서는 기준선에 겨우 닿는 수준이다. v5는 run당 10.24M 개의 새 쌍(10배)을 쓰고, 더 큰 D1-T를 함께 적격성 검사하며, 1회 사전 등록 보완(§5.4)을 둔다.

### 1.3 처리량 측정 (신규 실측, MLX/Metal, float32)

| 항목 | 값 | 출처 |
|---|---:|---|
| 현재 D1-S sampler, batch 64 / 256 / 1024 / 4096 | 414 / 476 / 406 / 398 후보/s | [`throughput.json`](local_experiment_archive/analyses/v5-d0-20260928/throughput.json) |
| `vmap` 벡터화 D1-S sampler, batch 1024 | **17,565 후보/s** | [`vectorized_sampler_check.json`](local_experiment_archive/analyses/v5-d0-20260928/vectorized_sampler_check.json) |
| 벡터화 출력 = 기존 `mlx_models.sample` 출력 | 256/256 bitwise 동일 | 같은 파일 |
| Batch 구성 불변성(후보 하나만 따로 생성해도 같은 출력) | 동일 | 같은 파일 |
| MD5 + 앞 12 bits, Python 단일 core | 2,104,757 /s | `throughput.json` |
| Transformer proxy d=192, 4 layers (1,823,714 params) | update 0.0199 s (batch 256), 약 1,418 후보/s | [`transformer_timing.txt`](local_experiment_archive/analyses/v5-d0-20260928/transformer_timing.txt) |
| Transformer proxy d=256, 8 layers (6,372,962 params) | update 0.0592 s, 약 455 후보/s | 같은 파일 |

현재 sampler가 batch 크기와 무관하게 느린 이유는 `mlx_models.sample`이 매 step마다 `mx.stack([mx.random.categorical(row, key=k) for ...])` 형태로 **후보별 kernel을 따로 호출**하기 때문이다. `mx.vmap`으로 바꾸면 난수 identity와 출력이 그대로 보존된다. Transformer 수치는 일반 MLX Transformer로 잰 proxy다. 실제 모델은 A-prof에서 다시 측정해 봉인한다.

### 1.4 이론적 사전 기대

학습에 쓰지 않은 새 입력에서 MD5의 12-bit window가 이상적 random function처럼 행동한다고 하자. 그러면 **어떤 sampler든** MD5를 호출하지 않고 만든 새 후보 하나의 성공확률은 2⁻¹²다. K=100이면 `p0 = 1−(1−2⁻¹²)¹⁰⁰ = 2.412%`다. 학습 메시지를 그대로 내놓는 후보는 disjoint한 train group의 해시를 가지므로 항상 실패한다. 따라서 Main의 기대 이득은 0 이하다.

이 논증은 sampler의 구조(Gaussian BGV/CGGE, discrete), source(Printable, Random Bytes), q(8, 12, 16)에 의존하지 않는다. 그래서 §14.3의 일반화 논리의 기반이 된다. 반대로 이득이 재현된다면, full 64-step MD5에 대한 신경망 distinguisher라는 강한 결과가 된다. 선행 신경망 distinguisher 성과는 round를 줄인 암호에 한정된다([Gohr 2019](https://eprint.iacr.org/2019/037), [Goncharov 2019](https://arxiv.org/abs/1901.02438)).

---

## 2. v4-claude 대비 변경점

| 항목 | v4-claude | v5 | 근거 |
|---|---|---|---|
| 결론 구조 | `CONTINUE` / `CONCLUDE_*` | C1–C4 판정 카드 + 종합 판정. **모든 경로가 연구 종료** | 최종 결론 요구 |
| D0 | 선택 사항, 미실행 | **실행 완료.** 선택 규칙이 원인임을 확인 | §1.2 |
| Sampler | 기존 per-row 난수 | `vmap` 벡터화, 기존과 bitwise 동일성을 구현 게이트로 둠 | §1.3 |
| 최소 관심 효과 δ | 0.5%p | **0.25%p** (p0 대비 상대 약 10%) | 처리량 여유 |
| Stage C trials / test groups | 16,384 / 512 | **65,536 / 1,024** | 같은 이유. 새 target 일반화 범위도 넓어짐 |
| 양성 경로 | CONTINUE 후 별도 확증 연구 | **Stage R**: 새 digest window에서 즉시 재현 | 양성도 최종 판정 |
| 규모 반론 | D1-S와 D1-T 비교(18 ladder runs) | Ladder는 주 architecture 하나로. 대신 **Stage S**(3.5배 params, 4배 data)로 규모 효과를 직접 측정 | 비용 대비 정보 |
| 계산 우위 | 다루지 않음(후속 연구) | **C4**: 분석식 + 실측 처리량으로 판정 | 연구 질문의 4번째 수준 |
| 노출 감사 | 미완이면 대체 window | 같음. **Window 결정을 MD5 조건 데이터 생성 전에 기록** | v4 inventory가 빈 템플릿 |
| 적격성 실패 | 즉시 `CONCLUDE_UNTESTABLE` | 1회 사전 등록 보완(4배 updates) 후 판정 | D0: 학습량이 늘수록 정확도 상승 |

이어받는 요소는 다음과 같다: source와 `H12_r` 정의, 새 쌍 학습, batch 내 Shuffled, MC 대조, CLP, Bonferroni 동시 구간과 1회 확장, artifact 감사, v4 원장·verifier·checkpoint·resume 계약. v4와 v4-claude의 산출물과 상태는 변경하지 않는다. v5 결과를 v4나 v3.1의 완료로 보고하지 않는다.

---

## 3. 결정 질문과 원칙

> **최종 결정 질문.** 조건부 생성 기계가 작동함을 확인한 상태에서, 미학습 MD5 12-bit target에 대해 해시 조건을 학습한 모델이 source-prior Random과 Shuffled-condition 모델보다 Success@100에서 높은가? 높다면 그 결과가 새 window에서 재현되는가? 높지 않다면 이득의 상한은 얼마인가?

1. **입증 책임은 가설 쪽에 있다.** `SUPPORTED`는 사전 규칙의 양성 판정, artifact 감사, 새 window 재현을 모두 통과해야만 나온다. 그 외의 모든 결과는 `REJECTED_*` 또는 `NOT_ESTABLISHED_*`다. 어느 쪽이든 연구는 종료한다.
2. **이 버전 이후의 revision은 없다.** 결과를 보고 규칙, seed, 학습량, sampler, 후보를 바꾸거나 재추첨하지 않는다. 사전 등록된 분기(확장 1회, 보완 1회, fallback 1회, 재현 1회)만 허용한다.
3. **설계값은 MD5 조건 데이터를 만들기 전에 봉인한다.** Stage A는 synthetic만 사용한다. 봉인 뒤에 Stage C를 먼저 실행하고, Stage B·S 결과는 C의 판정을 바꾸지 않는다.
4. **항상 수치를 남긴다.** 어떤 종결에서도 Main−Random, Main−Shuffled의 추정값과 동시 구간을 보고한다. `NOT_ESTABLISHED`도 상한과 함께 보고한다.

---

## 4. 공통 설정

### 4.1 Source와 해시 family

- **Source:** Printable ASCII 33–126. 길이는 4–31 균등이고, 주어진 길이에서 bytes는 iid uniform이다(원본 계획 §7.1과 같음). 모든 메시지는 MD5 한 block(≤55 bytes)에 들어간다.
- **`H12_r^W(x)`:** MD5 padding 후 압축함수의 처음 r step을 실행하고, IV를 word별로 더한다(feed-forward). 표준 little-endian 직렬화한 128-bit digest에서 window W의 12 bits를 취한다. `r=64`는 실제 MD5다. 참조 구현은 v4-claude 검산 스크립트의 `md5_steps`다. RFC 벡터 5개와 무작위 Printable 메시지 2,000개에서 `hashlib.md5`와 전부 일치했다.
- **Window:** 모든 window는 digest를 big-endian 정수로 읽은 값에서 추출한다.

| 이름 | 정의 | 용도 |
|---|---|---|
| W1 | `int.from_bytes(md5(x).digest(), 'big') >> 116` (앞 12 bits, 원본 계획의 정식 절단) | 노출 감사를 통과하면 주 window |
| W2 | `int.from_bytes(md5(x).digest(), 'big') & 0xFFF` (마지막 12 bits) | 감사 미완 시 주 window. W1이 주 window이면 재현 window |
| W3 | `(int.from_bytes(md5(x).digest(), 'big') >> 52) & 0xFFF` (중간 12 bits) | W2가 주 window일 때 재현 window |

- Stage B·S의 `r<64` 사다리는 W1 위치를 쓴다. `H12_r (r<64)`는 이 프로젝트에서 한 번도 조건으로 쓰지 않았다.
- `r=4`에서 W1은 payload 0–3번째 byte만의 함수다. 이 rung은 **MD5형 연산 구조를 가진 양성 대조**로 쓴다.

### 4.2 Group ownership과 새 쌍 학습

- 12-bit 값 4,096개를 과제별로 **test 1,024 / validation 256 / train 2,816 groups**로 나눈다. 학습 쌍은 prior에서 뽑은 x 중 `H(x)`가 train group인 것만 쓴다(수락률 68.75%). 따라서 학습 메시지는 test target의 역상이 될 수 없다.
- **새 쌍 학습:** batch 256 × 40,000 updates로 run당 새 쌍 10,240,000개(MD5 호출 약 14.9M)를 쓴다. Stage S는 160,000 updates다. Data stream은 `(protocol, task, window, rung, seed, update)` namespace의 결정적 난수로 만든다. 따라서 중단 후 재개해도 같은 batch가 나온다. 같은 seed의 Main과 Shuffled는 stream, 초기 가중치, corruption 난수를 공유한다.
- **메시지 생성과 해시는 NumPy로 벡터화한다.** 순수 Python 생성은 약 87k/s라 run당 수 분이 걸린다. `H12_r`의 NumPy 구현은 참조 구현과 전수 일치해야 한다(§12).
- **Shuffled:** 각 batch 안에서 condition을 무작위 permutation으로 재배정한다. Length head와 payload에 같은 donor condition을 쓴다. 평가 때는 실제 target을 넣는다.

### 4.3 모델, 손실, sampler

| 항목 | D1-S | D1-T | D1-T-L (Stage S 전용) |
|---|---|---|---|
| Payload denoiser | v4 D1 그대로(width 128, embedding 16, condition-output 잔차) | Pre-LN Transformer 4 layers, d=192, 4 heads, FFN 768 | 8 layers, d=256, 8 heads, FFN 1024 |
| Parameters | 508,668 | 약 1.82M (proxy 1,823,714) | 약 6.37M (proxy 6,372,962) |
| 공통 | 시간과 13-d condition(12 bits + L/31)을 투영해 모든 위치에 더한다. Length head `Linear(12,28)`은 생성 시 1회 sampling한다 | | |
| Optimizer | Adam lr 1e-3 | Adam, lr은 A-dev에서 {1e-3, 3e-4, 1e-4} 중 선택. Warmup 1,000, grad-norm clip 1.0 | D1-T에서 선택한 lr × (192/256), 나머지는 같음 |

- **손실:** `length CE + mean(가려진 payload 위치의 CE)`. 위치 특혜 loss는 쓰지 않는다.
- **Sampler:** 32 intervals, temperature 1, remask 없음, 후보당 NFE 33. v4 sampler와 수학적으로 같고 난수 사용만 `vmap`으로 벡터화한다. Hidden length, argmax 치환, MD5 기반 reranking은 없다.
- **Checkpoint:** 최종 update의 가중치 하나만 쓴다. 4,000 updates마다 validation loss와 validation CLP를 기록하지만 진단용이며 선택에 쓰지 않는다.

### 4.4 방법과 후보 원장

| 방법 | 정의 | 쓰는 곳 |
|---|---|---|
| Main | 실제 condition으로 학습하고, 요청 target을 condition으로 생성 | A, B, C, R, S |
| Shuffled | §4.2의 batch 내 permutation으로 학습하고, 요청 target으로 생성 | C, R |
| Random | 원래 source prior에서 정확히 직접 sampling | B, C, R, S |
| MC | Main checkpoint에 다른 trial의 target(고정 derangement)을 넣어 생성. 성공은 요청 target 기준 | B, S(주 대조), C(진단) |

- 성공은 `valid(x) ∧ H(x) = 요청 target`이다. **Success@100**은 trial당 K=100 후보 중 성공이 하나 이상인지로 정한다.
- Invalid와 duplicate도 기회를 소비한다. 첫 성공 뒤에도 100개를 끝까지 생성한다.
- 원장 키 `(protocol, task, window, rung, method, seed, trial, attempt)`, 독립 재해시 verifier, 난수 identity 기록은 v4 원장 계약을 따른다.

---

## 5. Stage A — 기계 적격성 (MD5 호출 0회)

### 5.1 A-impl (구현 게이트)

아래 항목이 모두 통과해야 이후 단계를 시작한다.

- 벡터화 sampler가 기존 `mlx_models.sample`과 **bitwise 동일**: 고정 fixture 4,096 후보, D1-S 가중치.
- Batch 구성 불변성: 같은 후보를 batch 1, 64, 1,024에서 생성해도 출력이 같다.
- `H12_r` NumPy 구현이 참조 구현과 일치한다: r ∈ 사다리 전체와 64, 무작위 메시지 100,000개, window 세 개. r=64가 hashlib과 같고, r=4의 W1이 bytes 0–3에만 의존한다.
- 새 쌍 stream이 결정적이고, 재개했을 때 같은 batch를 낸다.

### 5.2 A-prof와 A-dev

- **A-prof:** D1-S, D1-T, D1-T-L의 update 시간과 batch {256, 1024, 4096}의 생성 처리량·메모리를 측정한다. MD5 기준 처리량도 같은 기계에서 잰다(§11). 이 결과로 batch와 §13의 fallback 적용 여부를 봉인한다.
- **A-dev:** Synthetic **dev split**(acceptance와 disjoint한 조건)에서 D1-T의 lr을 고른다. 최대 6 runs × 10,000 updates다. 그다음 전체 설정과 window 결정(§12.1)을 `protocol.frozen.json`으로 봉인한다.

### 5.3 A-Q (적격성)

v4 V1과 같은 synthetic_nibbles 과제로, 새 쌍 학습과 최종 checkpoint를 쓴다. 설정은 D1-S, D1-T × seeds 0, 1, 2 = 6 runs다. D1-T-L은 seed 0 한 run만 실행하며 Stage S 사용 여부만 정한다.

| 기준 (seed별, 모두 충족) | 값 |
|---|---|
| Acceptance 512 조건, 정상 joint | ≥ 461/512 |
| 반전 joint | ≥ 461/512 |
| Valid (정상·반전 각각) | 512/512 |
| 반전 후보의 원래 조건 오성공 | ≤ 25/512 |
| CLP 양성 대조 (§10) | one-sided z > 3.26 |

- **Q** = 세 seed 모두 통과한 architecture 집합이다. 주 architecture **A\*** = D1-T ∈ Q이면 D1-T, 아니면 D1-S다.
- 생성 기준은 통과했는데 CLP만 실패하면 CLP 구현 결함으로 본다. 이 경우 CLP를 모든 판단에서 제외하고 보고한다.

### 5.4 사전 등록 보완 (1회)

Q가 비면, 두 architecture를 **160,000 updates**(4배)로 같은 seed에서 한 번만 다시 학습해 같은 기준으로 판정한다. 근거는 D0에서 학습량이 늘수록 조건 정확도가 올랐다는 점이다. 그래도 Q가 비면 C1 = `FAIL`이고, 종합 판정은 `NOT_ESTABLISHED_UNTESTABLE`이다(§14). D1-T-L이 seed 0에서 실패하면 Stage S는 D1-T를 4배 updates로 학습하는 것으로 대체한다.

---

## 6. Stage C — MD5-12 결정 시험 (주 판정)

### 6.1 설계

| 항목 | 고정값 |
|---|---|
| 해시 | `H12_64^{W_primary}` (§12.1) |
| Test pool | 인증된 미노출 group 1,024개. Validation 256, train 2,816 |
| Architecture | A\* |
| Learned runs | {Main, Shuffled} × seeds {0,1,2} = 6 runs. 확장 시 seeds {3,4,5} 6 runs 추가 |
| Trials | 6 checkpoint 봉인 **후** pool에서 iid 복원추출 65,536개. 모든 seed와 방법이 같은 목록을 쓰며 재추첨하지 않음 |
| 후보 | Main·Shuffled·Random 각 65,536 × 100 per seed. MC는 16,384 trials × 100 per seed(진단) |
| 주 지표 | Success@100, trial 단위 paired |
| 보조 | CLP_64 (65,536 pairs per seed), @1/@10, valid·duplicate 비율, valid일 때 hit, 학습 메시지 일치, NFE, 시간 |

### 6.2 통계

대조군 c ∈ {Random, Shuffled}, trial t, seed s에 대해 `d_t^c = mean_s (M_{s,t} − C_{s,t})`로 둔다. `Δ̂_c = mean_t d_t^c`이고 `SE_c = sd(d^c)/√T`다.

- **동시 구간:** α=0.05를 두 look × 두 대조군 × 양측으로 Bonferroni 분할한다. 각 꼬리 0.00625, `z = 2.4977`이다. `[L_c, U_c] = Δ̂_c ∓ z·SE_c`.
- **최소 관심 효과:** δ = **0.0025 (0.25%p)**. p0 = 2.412%의 약 10% 상대 증가이고, 후보당 성공확률로는 약 1.1배다.
- **기대 정밀도(효과 없음일 때):** 첫 look의 반폭은 ±0.122%p, 확장 후는 ±0.086%p다.
- **추론 범위:** 봉인된 test pool, checkpoint, seed에 조건부이고, target 추출과 생성 난수에 대한 추론이다. Seed별 추정값을 모두 보고한다.

### 6.3 판정

| 순서 | 조건 | Stage C 결과 |
|---|---|---|
| 1 | `L_Random > 0` **그리고** `L_Shuffled > 0` | `POSITIVE` → §6.4 감사 → Stage R |
| 2 | `U_Shuffled < δ` 그리고 `U_Random < δ` | `REJECTED_BOUNDED` — 두 비교 모두 0.25%p 이상 이득을 배제 |
| 3 | `U_Shuffled < δ` (Random 비교는 미확정 또는 양) | `REJECTED_NO_CONDITION_GAIN` — 해시 조건의 기여를 배제 |
| 4 | `U_Random < δ` (Shuffled 비교는 미확정 또는 양) | `REJECTED_NO_RANDOM_ADVANTAGE` — prior sampling보다 유용하지 않음 |
| 5 | 그 외 | 첫 look이면 **1회 확장**(seeds 3–5 추가, 6 seeds 합산으로 재판정). 두 번째 look이면 `NOT_ESTABLISHED_UNRESOLVED` + 구간 보고 |

**설계 검산**(가상 결과, 시나리오당 1,000회, 확장 포함; 양성 행은 Stage R까지 포함):

| 가정 (seed별 Success@100) | 최종 `SUPPORTED` | 주 결과 |
|---|---:|---|
| 모두 p0 (효과 없음) | ≈ 0 (첫 look POSITIVE 0.1% × 재현 통과율 0.45%) | `REJECTED_BOUNDED` 99.3% |
| Main = p0 + 0.125%p (δ의 절반) | — | POSITIVE 35.6%, REJECTED 64.4%. δ 아래 효과이므로 양쪽이 섞임 |
| Main = p0 + 0.25%p | **98.6%** | 재현 실패 0.4%, REJECTED 1.0% |
| Main = p0 + 0.5%p | 100% | |
| Main = p0 + 0.5%p, 매 look 한 seed는 p0 | (재현 전 POSITIVE 100%) | |
| Main = Shuffled = p0 + 0.25%p (prior 모델링 이득) | — | `REJECTED_NO_CONDITION_GAIN` 98.5% |
| Main = p0 − 0.3%p (학습 메시지 재생산) | 0% | `REJECTED_BOUNDED` 100% |

### 6.4 POSITIVE 후 artifact 감사 (필수)

1. 모든 성공 payload를 독립 verifier로 재해시한다.
2. 성공 후보 중 학습 메시지와 같은 것은 0이어야 한다(group disjoint 계약). 위반은 무결성 실패다.
3. 성공이 특정 target에 몰렸는지 확인한다. 상위 1% target의 성공 비중을 보고한다.
4. MC와 CLP_64의 방향을 보고한다. 둘 다 무신호인데 hit만 양성이면 `hit-only`로 표시한다. 이 표시는 Stage R의 필요성을 바꾸지 않는다.

감사 실패는 POSITIVE를 무효화한다. 원인을 고친 뒤 해당 stream을 같은 난수 identity로 재생성한다(1회). 고칠 수 없으면 `NOT_ESTABLISHED_INTEGRITY`다.

---

## 7. Stage R — 양성 재현 (Stage C가 POSITIVE이고 감사를 통과한 경우만)

- **Window:** 주 window가 W1이면 W2, 주 window가 W2이면 W3. 새 group split(1,024 / 256 / 2,816)을 쓰고, 해당 window의 노출 인증은 §12.1과 같이 한다.
- **설계:** Stage C와 같은 architecture, lr, updates를 쓴다. Main·Shuffled × seeds {0,1,2}(새 namespace), trials 65,536, K=100, Random.
- **판정:** `L_Random > 0` 그리고 `L_Shuffled > 0`. 각 one-sided 97.5% 구간(z = 1.96)으로 판정하며 intersection-union이라 추가 보정이 필요 없다. 효과 없음에서 가상 통과율은 0.45%(상한 2.5%)다.
- 통과 → C3 = `SUPPORTED`. 실패 → C3 = `NOT_ESTABLISHED_NOT_REPLICATED`. 확장 look은 없다.

---

## 8. Stage B — Step-reduced MD5 사다리 (C2; C3 불변)

### 8.1 목적

두 해석을 구분한다. "기계가 해시 구조를 전혀 이용하지 못한다"인가, 아니면 "구조가 있으면 이용하지만 MD5의 mixing이 그것을 지운다"인가. 결과는 최종 보고서의 핵심 근거이며, 별도 새 연구(pivot)를 검토할 가치가 있는지 표시한다.

### 8.2 Rung과 모델 없는 난이도

Rung은 `R_B = {4, 5, 6, 7, 8, 10, 12, 16, 32}`이다. 모델 없는 profile은 v4-claude 검산값을 그대로 쓴다. 단일 byte 변경 시 출력 bit 반전율은 r=4에서 0.036, r=8에서 0.336, **r=12에서 0.486**, r≥16에서 약 0.50이다. 단일 byte와의 최대 MI는 r=4에서 1.876 bits, r=5에서 0.021 bits이고, r≥6은 plug-in bias 수준이다. 완전 mixing rung은 `r_mix = 12`로 봉인한다.

### 8.3 실행과 검정

| 항목 | 값 |
|---|---|
| Runs | A\* × 각 rung × seed 0 = 9 runs. Main만 학습 |
| Split | Rung별 무작위 test 1,024 / validation 256 / train 2,816 groups |
| 평가 | Trials 16,384 × K=100로 Main, MC, Random. CLP 32,768 pairs |
| GEN(r) | Main−Random **그리고** Main−MC의 paired Success@100 one-sided z > 3.06 (셀당 α = 0.01/9, intersection-union) |
| INFO(r) | CLP one-sided z > 3.06 |
| 검정력 | 90% 검정력으로 검출 가능한 Success@100 차이 ≈ 0.74%p |
| 지평선 | `r*_gen` = GEN이 성립한 가장 큰 rung. `r*_info`도 같은 방식. 비단조 패턴은 그대로 보고 |
| 재현 | `r*_gen`과 바로 위 rung에 seeds 1, 2 추가(최대 4 runs). **확정 `r*_gen`** = 세 seed 중 2개 이상에서 GEN이 성립한 가장 큰 rung |

- **r=4 양성 대조:** r=4에서 GEN이 없으면 C2에 "MD5형 연산 구조를 가장 강한 rung에서도 이용하지 못함"을 기록한다. C3 판정은 바뀌지 않는다.
- **`PIVOT_SUPPORTED`:** 확정 `r*_gen ≥ 16`(`r_mix`보다 깊은 rung)인 경우다. "step-reduced MD5에서 신경망 역상 후보 생성" 연구를 **별도 새 연구로** 제안할 근거가 된다. 그 외는 `PIVOT_NOT_SUPPORTED`다.

---

## 9. Stage S — 규모 탐침 (C2; C3 불변)

**질문:** "더 큰 모델과 더 많은 데이터면 되지 않았을까?"라는 반론에 측정으로 답한다.

| 항목 | 값 |
|---|---|
| 모델 | D1-T-L (약 6.37M params, D1-T의 3.5배). A-Q seed 0에서 실패했다면 D1-T를 대신 사용 |
| 학습량 | 160,000 updates × 256 = 새 쌍 40.96M개(4배). 계산량은 D1-T run의 약 12배 |
| Rung | `r_edge` = Stage B seed 0에서 INFO가 처음 실패한 rung, 그리고 r=64(주 window). r=32까지 모두 INFO가 성립했다면 r=64만 |
| 측정 | CLP 65,536 pairs. `r_edge`에서는 GEN도 측정(4,096 trials × 100, Main·MC·Random) |
| 판정 | `SCALE_SHIFT` = `r_edge`에서 INFO(z > 3.06)가 성립하는 경우. 규모를 키우면 지평선이 움직인다는 뜻이다 |

해석은 다음과 같다.

- `SCALE_SHIFT`가 없으면: 약 12배 계산으로도 지평선이 한 rung도 움직이지 않았다. r=64까지는 적어도 그 rung들을 모두 넘어야 한다는 사실과 함께 C2에 기록한다.
- `SCALE_SHIFT`가 있으면: 움직인 rung 수를 보고한다. 이것도 C3 판정을 바꾸지는 않는다. r=64까지의 외삽은 주장하지 않는다.
- **r=64 CLP가 양성이면**(z > 3.26): full MD5에 대한 신경망 distinguisher 후보라는 이례적 신호다. 최종 보고서에 "독립 재현이 필요한 이상 신호"로 기록한다. C3 판정은 hit로만 내린다.

---

## 10. CLP — Conditional Likelihood Probe

v4-claude §9를 그대로 따른다.

- **점수:** `s(x, y)` = −(length CE + 가려진 payload CE)를 고정 corruption draw 8개로 평균한 값이다. Test group의 held-out 쌍 `(x_i, y_i)`, `(x_j, y_j)`에 대해 `D = s(x_i,y_i) + s(x_j,y_j) − s(x_i,y_j) − s(x_j,y_i)`로 정의한다.
- **귀무:** x와 y가 독립이면 D는 0에 대해 대칭이다. Condition에만 의존하는 성분은 상쇄된다. 검산에서 가상 null 2,000회의 α=0.01 기각률은 1.15%였다.
- **쓰는 곳:** 양성 대조(A-Q), 지평선(B, S), 진단(C)이다. **C3 판정에는 쓰지 않는다.** CLP 양성은 "조건이 held-out 우도를 설명한다"는 뜻일 뿐 생성 성공을 보장하지 않는다.
- **비용:** Forward pass만 쓰므로 싸다. D1-T 기준 65,536 pairs × 4 점수 × 8 draws ≈ 2.1M 행으로 1분 내외(proxy 46,758 행/s).

---

## 11. C4 — 계산 우위 판정

### 11.1 분석식 (실험 전에 이미 확정되는 부분)

| 양 | 값 |
|---|---:|
| 학습 run 하나가 쓰는 MD5 호출 (10.24M 쌍 / 수락률 0.6875) | 14,894,545 |
| 4,096 target 전부의 역상 lookup table을 무작위 탐색으로 만드는 기대 비용 (coupon collector, `4096·H_4096`) | 36,434 |
| Test 1,024 target만의 lookup 기대 비용 | 30,758 |
| 학습 1 run의 MD5 비용 ÷ 전체 lookup 비용 | **약 409배** |
| Success@100에서 가능한 최대 lift (`1/p0`) | 41.5배 |

q=12에서 모델 학습은 **완전한 역상 table을 약 400번 만들 수 있는 MD5 호출**을 소비한다. 따라서 학습비를 포함한 계산 우위(amortized)는 q=12에서 C3 결과와 무관하게 성립하지 않는다. 이 문장은 실험 결과를 기다리지 않는 산술이며, 최종 보고서에 그대로 쓴다.

### 11.2 질의당 계산 우위 (학습비를 제외한 가장 관대한 비교)

- A-prof에서 같은 기계로 두 처리량을 잰다. `thr_MD5`는 prior sampling + MD5, CPU 단일 core다. `thr_M`은 A\* 생성, GPU다.
- 비용 비율은 `ρ = thr_MD5 / thr_M`이다. 현재 proxy 기준으로 D1-T는 약 1,484, D1-S는 약 120이다.
- 질의당 우위가 성립하려면 Main의 후보당 성공확률이 `ρ · 2⁻¹²` 이상이어야 한다. D1-T 기준 약 **36%**, D1-S 기준 약 **2.9%**다. Random의 후보당 성공확률은 0.024%다.

**판정 규칙:**

- C3 ≠ `SUPPORTED`이면 C4 = `NO_ADVANTAGE`다.
- C3 = `SUPPORTED`이면 Stage C Main의 후보당 성공률 one-sided 97.5% 하한을 계산한다. 그 하한이 `ρ · p̂_Random,candidate`를 넘을 때만 `PER_QUERY_ADVANTAGE`, 아니면 `NO_ADVANTAGE`다.
- 어느 경우든 §11.1의 amortized 불성립을 함께 기록한다.
- MD5를 CPU 단일 core로 재는 것은 모델에 유리한 비교다. 다중 core나 SIMD MD5는 `ρ`를 더 키운다.

---

## 12. 노출·무결성·calibration

### 12.1 Window 결정과 노출 인증

1. `protocol.frozen.json` 봉인 시점, 즉 **MD5 조건 데이터를 하나도 만들기 전**에 주 window를 결정해 `window.json`에 기록한다.
2. **W1:** v4 노출 감사 도구(`study_v4 audit`)와 inventory를 실제 증거로 작성해 완료한다. 미노출 group이 1,024개 이상 인증되면 W1이 주 window다. 현재 `examples/v4-exposure-inventory.json`은 빈 템플릿(검토 범위 전부 `false`)이다. 확인된 노출 하한은 1,885 groups이고, 따라서 최대 여유는 2,211개다.
3. 감사가 완료되지 않았거나 1,024개에 못 미치면 **W2**가 주 window다. 이 결정은 결과와 무관하므로 편향이 생기지 않는다.
4. **W2/W3 인증은 기계적으로 한다.**
   - 코드 감사: 현재 코드의 모든 조건 경로는 digest 앞쪽 q bits(`digest_prefix_hex`, `>> 116`)나 toy hash에서 나온다.
   - 과거 archive 중 full digest나 raw digest를 조건으로 쓴 run이 있으면, 그 run의 validation/test 대표 메시지를 해당 window로 투영해 제외한다.
   - 조사 범위와 제외 수를 감사 기록에 남긴다.
5. Stage B·S의 `H12_r (r<64)`는 새 함수이므로 노출 대상이 아니다.

### 12.2 무결성 검사

v4 V0를 재사용하고 다음을 추가한다.

- §5.1의 A-impl 항목 전부
- Train group rejection의 정확성(학습 메시지의 해시가 모두 train group)과 학습 메시지 hash set 기록
- Shuffled permutation 기록, MC derangement에 자기 자신이 없는지
- CLP 대칭성 fixture
- 원장 → trial → counts 전수 fixture와 독립 재해시

### 12.3 Production calibration

- 실제 판정 코드(§6.3, §7)에 가상 joint outcome을 넣어 §6.3의 시나리오를 시나리오당 20,000회 실행한다.
- **통과 조건:**
  - 효과 없음에서 POSITIVE 비율의 one-sided 95% CP upper ≤ 0.01
  - +0.25%p에서 POSITIVE의 lower ≥ 0.95
  - 효과 없음에서 `REJECTED_BOUNDED`의 lower ≥ 0.95
- **Planted-lift fixture:** Random sampler에 **validation group**의 사전 계산 역상 table을 일정 비율로 섞는다. 실제 원장 경로에서 +0.25%p가 POSITIVE로, +0이 REJECTED로 판정되는지 확인한다. Test pool은 사용하지 않는다.

---

## 13. 실행 순서, 자원, fallback

**순서:** A-impl → A-prof → A-dev → **봉인**(protocol + window) → A-Q(→ 보완) → **C 첫 look** → (C 확장) → (R) → B → S → 최종 보고.

주 판정(C)을 먼저 확보한다. 보조인 B·S가 자원을 먼저 쓰지 않게 한다.

**예상 실행량과 시간** (D1-T proxy 기준, [설계 계산](local_experiment_archive/analyses/v5-design-20260928/design_calculation.json), 로컬 전용이며 검산 스크립트로 재생성 가능):

| 단계 | Learned runs | Learned 후보 | 예상 학습 | 예상 생성 | Hard cap |
|---|---:|---:|---:|---:|---:|
| A (impl, prof, dev, Q) | ≤ 13 | 약 7,000 | 약 1.5 h | 무시 가능 | 8 h |
| C 첫 look | 6 | 44,236,800 | 1.3 h | 8.7 h | 16 h |
| C 확장 (해당 시) | 6 | 44,236,800 | 1.3 h | 8.7 h | 16 h |
| R (POSITIVE 시) | 6 | 44,236,800 | 1.3 h | 8.7 h | 16 h |
| B (+ 재현) | 9 + 4 | 42,598,400 | 2.9 h | 8.3 h | 16 h |
| S | 2 | 819,200 | 5.3 h | 0.5 h | 10 h |
| **필수 경로 합계 (A, C, B, S)** | | | | **약 30 h** | **50 h** |

- Random 후보(C는 seed당 6.55M)는 NumPy 벡터화 생성과 hashlib으로 처리하며, 단계 시간에 포함된다.
- 저장량, RSS, GPU, 최소 디스크 여유 한도는 v4와 같다(64 / 64 / 64 / 10 GiB).
- 후보 원장이 커지므로 C 원장은 payload hex, 성공 여부, 난수 identity만 기록한다. 원장은 run별 SQLite로 나누고 저장량을 A-prof에서 추정해 봉인한다.

**사전 등록 fallback** — A-prof 직후, 결과를 보기 전에 한 번만 적용한다. 예상 시간 × 1.5가 cap을 넘으면 아래 순서로 적용한다.

1. B trials 16,384 → 8,192
2. S updates 160,000 → 80,000
3. B 재현에서 `r*_gen` 바로 위 rung 생략
4. C의 MC 16,384 → 4,096 trials
5. B rung 32 생략
6. C architecture를 D1-S로 변경(D1-S ∈ Q인 경우)

**C와 R의 Main/Shuffled/Random trials, K, seeds, updates는 절대 줄이지 않는다.**

**실행 중 cap 초과 시:**

- C 첫 look이 완료되지 않았다면 C3 = `NOT_ESTABLISHED_BY_BUDGET`이다.
- B·S가 미완이면 C2를 "부분 측정"으로 보고하고, 최종 결론은 그대로 낸다.
- 고립된 실행 오류는 같은 난수 identity로 1회 재실행할 수 있다(v4 resume 계약).

---

## 14. 최종 결론 규칙

### 14.1 종합 판정 (연구 종료 판정)

| 종합 판정 | 조건 | 최종 보고서에 쓰는 결론 문장 (사전 작성) |
|---|---|---|
| **`FINAL_SUPPORTED`** | C1 PASS, C3 `SUPPORTED` | "고정 Printable source, MD5 12-bit window, D1 계열에서 해시 조건 모델은 Random과 Shuffled 대비 Success@100 이득을 보였고(구간 제시), 이 이득은 두 번째 digest window에서 재현되었다. 계산 우위는 {C4}. Full MD5 역상, 보안 붕괴, 학습비 포함 계산 우위는 주장하지 않는다." Stage I 가설 지지로 종료. 후속은 **별도의 새 연구**(Stage II 계산 효율)로만 가능하다 |
| **`FINAL_REJECTED`** | C1 PASS, C3 `REJECTED_*` | "해시 조건 diffusion 생성의 Success@100 이득은 Random 대비 {U_R} 이하, Shuffled 대비 {U_S} 이하이며, 사전 정한 최소 관심 효과 0.25%p를 {두 비교 모두 / 조건 기여 / prior 대비} 배제한다. 같은 기계는 synthetic 조건과 step-reduced MD5 r ≤ {r*_gen}에서는 조건을 이용했다. 12배 규모에서 지평선 이동은 {SCALE_SHIFT}. 계산 우위는 없다." Stage I 가설 기각으로 연구 종료 |
| **`FINAL_NOT_ESTABLISHED`** | C3 `NOT_ESTABLISHED_*` (UNRESOLVED, NOT_REPLICATED, INTEGRITY, BY_BUDGET) 또는 C1 FAIL (UNTESTABLE) | "사전 규칙으로 이득을 입증하지 못했다. 이득의 상한은 {구간}이다(측정된 경우). 사유는 {사유 코드}다." 추가 증액 없이 연구 종료. UNTESTABLE이면 "현재 기계 계열로는 질문을 검정할 수 없었고, MD5 효과는 측정되지 않았다"를 명시 |

어느 판정이든 **v5 이후 같은 질문의 revision은 없다.** `PIVOT_SUPPORTED`나 CLP_64 이상 신호는 원래 연구의 지속이 아니다. 새 연구를 제안할 근거일 뿐이다.

### 14.2 최종 보고서 필수 내용

1. 최종 결론 카드(§0)와 종합 판정
2. 버전별 결산(§1.1), D0 진단, 처리량 측정
3. C1: A-Q 결과, Q와 A\*, 보완 적용 여부
4. C2: rung별 GEN/INFO, 확정 `r*_gen`, `r*_info`, S 결과, `SCALE_SHIFT`, `PIVOT_*`
5. C3: 모든 대조의 추정값·구간·seed별 값, 판정 경로(확장 여부), artifact 감사, R 결과(해당 시), CLP_64
6. C4: §11.1 산술, 측정된 `ρ`, 판정
7. 적용 범위와 일반화(§14.3)

### 14.3 적용 범위와 일반화 논리

- **실측 범위:** Printable source, MD5 12-bit window(W1 또는 W2; 재현 시 W2/W3 추가), D1 계열(D1-S, D1-T, D1-T-L), 균일 masked-diffusion loss, 등록 sampler, 명시한 학습량과 K=100.
- **이론적 외삽(측정 아님):** §1.4의 random-function 논증은 sampler 구조, source, q에 의존하지 않는다. 따라서 Gaussian BGV/CGGE pipeline, Random Bytes source, q=8/16에서도 같은 결론을 기대한다. 최종 보고서에는 이것을 **이론적 기대**로 구분해 쓴다. v3.1의 다섯 pipeline 적격성은 완료되지 않았고, 이 연구에서 완료로 보고하지 않는다.
- **주장하지 않는 것:** full MD5 역상, 임의 target MD5 역상, SHA-256, 보안 붕괴, 최신 암호분석 공격과의 비교, 모든 diffusion의 불가능성.

---

## 15. 구현 범위

**재사용:** `study_v4_data`(seed derivation, 노출 감사, split), `study_v4_runtime`(Budget, checkpoint·resume, 원장·verifier, 평가 루프), `study_v4_statistics`(CP primitive), `mlx_models.SequenceDenoiser`/`MaskedDiffusion`(D1-S, 균일 loss), TokenCodec.

**신규:**

1. `mlx_models.sample_vectorized`: `vmap` 난수의 discrete sampler. 기존 `sample`과 bitwise parity 테스트를 둔다. 시제품은 [`vectorized_sampler_check.py`](local_experiment_archive/analyses/v5-d0-20260928/vectorized_sampler_check.py)(로컬 전용)에 있다. 저장소만 받은 경우 명세는 §4.3과 §5.1이다.
2. `hashing`: NumPy 벡터화 `H12_r^W`와 참조 일치 테스트.
3. 새 쌍 data stream: NumPy 벡터화 prior sampling, train group rejection, 결정적 재개, 학습 메시지 hash set.
4. `mlx_models`: D1-T, D1-T-L (같은 호출 계약).
5. Batch 내 Shuffled, MC derangement 생성기.
6. CLP scorer와 대칭성 fixture.
7. 판정기: Stage C(§6.3), R(§7), ladder·S 검정, C4 규칙, 종합 판정(§14.1). Production calibration과 planted-lift fixture를 포함한다.
8. `study_v5` CLI: `plan | audit | run --stage {A,C,R,B,S} | report`. 단계 봉인, resume, cap, fallback, window 결정을 기록한다. 수정된 protocol JSON은 거부한다.

v3.1, v4 실행기와 산출물은 수정하지 않는다. 벡터화 sampler는 새 함수로 추가한다. 기존 `sample`은 바꾸지 않는다.

## 16. 산출물과 재현

1. `protocol.frozen.json`, `window.json`, code·환경 manifest, fallback 적용 기록, 노출 감사 기록
2. A-impl 테스트 결과, A-prof, A-dev, A-Q 결과와 Q, A\*
3. Stage C (및 확장, R): 학습 기록, 최종 checkpoint 봉인, trial 목록, 모든 원장, verifier 감사, 효과·구간·판정, artifact 감사
4. Stage B·S: 셀별 GEN/INFO, 지평선, 재현, `SCALE_SHIFT`, `PIVOT_*`
5. `decision.json`: 실행 상태, C1–C4, 종합 판정, 사유 코드
6. 한국어 최종 보고서 `FINAL_REPORT_KO.md`(§14.2 구성)

**설계 검산 재현** (가상 계산과 저장된 측정값만 사용한다. 실제 실험 명령이 아니다):

```sh
.venv/bin/python scripts/validate_research_plan_v5.py
```

출력은 `local_experiment_archive/analyses/v5-design-20260928/design_calculation.json`이다. Archive가 없는 저장소에서도 실행되며, 이때는 기록된 측정값을 쓴다. 내용은 D0 판정, Stage C·R의 가상 작동 특성, ladder 검정력, C4 산술, 실행량·시간 산술이다.

**D0와 처리량 측정 재현** (로컬 전용: v4 V1 checkpoint와 스크립트가 archive에 있어야 한다. synthetic 모델만 사용, MD5 test 접근 없음):

```sh
.venv/bin/python local_experiment_archive/analyses/v5-d0-20260928/d0.py
```

```sh
.venv/bin/python local_experiment_archive/analyses/v5-d0-20260928/throughput.py
```

```sh
.venv/bin/python local_experiment_archive/analyses/v5-d0-20260928/vectorized_sampler_check.py
```
