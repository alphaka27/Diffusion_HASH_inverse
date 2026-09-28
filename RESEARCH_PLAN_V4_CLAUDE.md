# v4-claude 실험 계획 — 연구 지속(CONTINUE)과 종료(CONCLUDE)를 결정하는 실험

**Protocol:** `dhi-v4-claude-20260928` · **Master seed:** `2026092805` · **작성일:** 2026-09-28 KST
**상태:** `PLANNED_NOT_IMPLEMENTED`. 이 문서의 수치 중 실측값은 기존 archive에서 읽은 것이다. 설계 검산값은 [검산 스크립트](scripts/validate_research_plan_v4_claude.py)의 가상 계산 결과다. 새 학습·새 MD5 test 접근·v4 산출물 변경은 하지 않았다.

**요약.** 이 실험이 끝나면 결론은 **반드시 `CONTINUE` 또는 `CONCLUDE` 중 하나**로 나오도록 설계했다. 실험은 세 부분으로 구성된다.

- **기계 적격성(Stage A):** 조건이 쉽게 학습되는 synthetic 과제를 먼저 통과해야 한다.
- **실제 MD5-12 결정 시험(Stage C):** Main을 Random 및 Shuffled와 비교한다. 결정은 이 결과로 내린다.
- **Step-reduced MD5 사다리(Stage B):** MD5 압축함수의 step 수를 4부터 32까지 늘리면서 모델이 해시 구조를 어디까지 이용하는지 측정한다. 결정을 바꾸지 않고 종료 보고서와 후속 방향을 뒷받침한다.

학습은 고정 10,000개 대신 **매 update마다 새로 해시한 쌍**으로 한다. 이렇게 하면 v4 V1을 막은 과적합과 checkpoint 선택 문제가 구조적으로 사라진다. 입증 책임은 지속 쪽에 둔다. 사전에 정한 효과 크기를 통계적으로 입증해야만 CONTINUE이며, 그 외 모든 종결 상태는 사유를 붙인 CONCLUDE다.

관련 문서: [원본 계획](RESEARCH_PLAN.md) · [v4 계획](RESEARCH_PLAN_V4.md) · [v4 CLI](V4_CLI.md) · [연구 가능성 검토](RESEARCH_FEASIBILITY_REVIEW_KO.md) · [최신 P2-struct 분석](V3_1_P2STRUCT_ANALYSIS_KO.md) · [설계 계산 JSON](local_experiment_archive/analyses/v4-claude-design-20260928/design_calculation.json)

---

## 1. 현재까지의 증거와 이 계획의 출발점

### 1.1 확인된 사실

| 증거 | 출처 | 의미 |
|---|---|---|
| 실제 MD5-12 PoC: Printable Main 0/305, Random 9/305, Shuffled 0/305 | 2026-09-21 PoC 비교 JSON, [검토](RESEARCH_FEASIBILITY_REVIEW_KO.md) | 양의 증거 없음. 당시 형식 실패(valid 1/247,500)가 커서 효과 부재의 증거도 아님 |
| Toy MD5-8 `ABCD^4`: Diffusion 3/18, Source-prior 7/18 | 같은 검토의 재검증 | 우위 증거 없음. 표본이 작음 |
| p2struct: P-DISC 123/125 (분모 128), R-G-BGV 128/128 | [P2-struct 분석](V3_1_P2STRUCT_ANALYSIS_KO.md) | 위치 특혜 loss를 쓴 synthetic에서는 조건부 생성이 작동함 |
| **v4 V1 (uniform payload loss D1, synthetic): 세 seed 모두 `BLOCKED_QUALIFICATION`** | `local_experiment_archive/runs/v4-study/V1` | 아래 1.2 참조. MD5 과제는 시작하지 않았음 |
| v4 V1의 valid 1024/1024, 반전 후보의 원래 조건 오성공 0 (세 seed 모두) | 같은 위치의 `evaluation_summary.json` | 형식 생성과 조건 방향성은 정상. 부족한 것은 조건 정확도 |

### 1.2 v4 V1 실패의 구조 (저장 기록에서 확인)

| Seed | 정상 joint / 512 | 반전 joint / 512 | 기준 | 선택된 checkpoint | Validation loss: epoch 10 → 100 |
|---|---:|---:|---:|---|---|
| 0 | 176 | 167 | ≥461 | epoch 10 | 6.802 → 7.200, 단조 증가 |
| 1 | 160 | 151 | ≥461 | epoch 10 | 최소값 6.829 (epoch 10) |
| 2 | 125 | 134 | ≥461 | epoch 10 | 최소값 6.819 (epoch 10) |

세 seed 모두 validation loss가 첫 측정인 epoch 10에서 최소였다. 이후 training loss는 6.63→6.07로 계속 감소했고 validation loss는 증가했다. 고정 10,000개 메시지에 대한 suffix 암기로 해석할 수 있다. p2struct에서도 prefix CE는 감소하고 suffix CE는 증가하는 같은 패턴이 관측됐다. 등록 규칙인 "최소 validation objective checkpoint"가 조건 학습이 가장 덜 된 시점을 고른 것으로 **추정**한다. Epoch 100 checkpoint가 저장되어 있으므로 이 추정은 §5의 D0에서 싸게 확인할 수 있다. v4 계획은 V1 실패 후 규칙 변경을 금지하므로, 이 문제를 고치려면 어차피 새 revision이 필요하다.

### 1.3 실험 없이도 말할 수 있는 이론적 사전 기대

학습에 쓰지 않은 새 입력에서 MD5-12가 이상적 random function처럼 행동한다고 하자. 그러면 **어떤 sampler든** 새로운 후보 하나의 성공확률은 2⁻¹²이다. 학습 메시지를 그대로 내놓는 후보는 test group과 disjoint한 train group의 해시를 가지므로 항상 실패한다. 따라서 Main의 Random 대비 기대 이득은 0 이하다. 거꾸로 full 64-step MD5에서 이득이 재현된다면, 그것은 신경망 기반의 MD5 distinguisher라는 강한 결과가 된다.

신경망 distinguisher의 선행 성과는 round를 줄인 암호에 한정되어 있다. 예로 [Gohr, CRYPTO 2019](https://eprint.iacr.org/2019/037)는 Speck32/64의 축소 round를, [Goncharov 2019](https://arxiv.org/abs/1901.02438)는 축소 round MD5를 다뤘다. 이 사전 기대가 설계에 주는 함의는 두 가지다.

- (a) Stage C는 효과를 "찾는" 실험보다 **효과 상한을 정밀하게 확정하는** 실험으로 설계한다.
- (b) 음성 결과가 "기계가 고장 났기 때문"이라는 반론을 막도록 기계 적격성(Stage A)과 난이도 사다리(Stage B)를 붙인다.

---

## 2. v4를 그대로 재실행하지 않는 이유

| v4의 설계 | 문제 | v4-claude의 대응 |
|---|---|---|
| 고정 10,000 메시지 × 100 epochs | 과적합 → validation 최소 = epoch 10 → V1 차단 | 해시 라벨은 계산 비용이 거의 없다. 매 batch를 새 쌍으로 만든다(run당 10.24M 쌍). 과적합이 구조적으로 없으므로 **최종 checkpoint를 고정 사용**하고 선택 절차를 없앤다 |
| 단일 모델(0.51M param MLP) | 음성 결과가 "모델 용량 부족"으로 반박될 수 있음 | D1-S(v4 D1 그대로)와 D1-T(Transformer, 약 1.8M)를 함께 적격성 검사하고 강한 쪽을 Stage C에 사용 |
| r=64 단일 이진 시험 | 기계 결함과 MD5 mixing이라는 두 원인을 구분하지 못함 | Step-reduced MD5 사다리로 "해시 구조를 이용할 수 있는 한계 step"을 측정 |
| `BLOCKED_*` / `INCONCLUSIVE`가 연구 결정을 남기지 않음 | 의사결정 실험의 목적과 어긋남 | 모든 종결 상태를 CONTINUE/CONCLUDE에 사상한다. 불확정일 때만 사전 등록한 1회 확장을 허용 |
| 새 test pool 1,024 groups와 전체 노출 감사가 선행 조건 | 감사가 끝나지 않아 차단될 위험(`BLOCKED_EXPOSURE`) | Test pool을 512 groups로 줄인다(검정력은 trial 수가 결정함). 인증 실패 시 사전 등록한 대체 window를 사용 |
| GO 판정에 여섯 비교 모두 L>1%p 요구 | "유용한 이득" 문턱이 목적에는 맞지만, seed 이질성 때문에 NO_GO_REPRODUCIBILITY가 쉽게 나옴 | 결정의 질문은 "해시 조건의 이득이 존재하는가"다. Seed 평균 paired 효과의 동시 구간으로 판정하고 δ=0.5%p로 더 좁게 상한을 잡는다 |

v4의 검증된 부품은 그대로 재사용한다. RFC MD5 검증, codec, SQLite 후보 원장, atomic checkpoint·resume, seed derivation 규칙, 노출 감사 도구, CP/McNemar primitive가 해당된다(§12). v4의 산출물과 상태(`BLOCKED_QUALIFICATION`)는 변경하지 않는다. v4-claude의 결과를 v4나 v3.1의 완료로 보고하지 않는다.

---

## 3. 결정 질문과 원칙

> **결정 질문.** 조건부 생성 기계가 작동함을 확인한 상태에서, 실제 MD5-12의 새 target에 대해 해시 조건을 학습한 모델이 source-prior Random과 Shuffled-condition 모델보다 Success@100에서 **0.5%p 이상** 높은가? 만약 아니라면 그 상한은 얼마인가?

1. **입증 책임은 지속 쪽에 있다.** CONTINUE는 사전 규칙의 양성 판정에서만 나온다. 음성·불확정·실행 불가는 모두 사유를 붙인 CONCLUDE다. 불확정은 단 1회의 사전 등록 확장 후에도 남으면 CONCLUDE다.
2. **모든 설계값은 test 접근 전에 봉인한다.** Stage B 결과가 Stage C 설계를 바꾸지 않는다. Stage C를 먼저 실행한다(§11).
3. **기계 실패와 해시 음성을 구분하되, 둘 다 결정으로 이어진다.** 기계 실패는 "MD5 효과 없음"이 아닌 `CONCLUDE_UNTESTABLE`로 기록한다.
4. **판정 문턱을 결과를 보고 바꾸지 않는다.** Seed 교체, 추가 학습, sampler 변경, 후보 재추첨을 하지 않는다.

---

## 4. 공통 설정

### 4.1 Source와 해시 family

- **Source:** Printable ASCII 33–126, 길이 4–31 균등, 주어진 길이에서 bytes iid uniform. v4와 같다. 모든 메시지는 MD5 한 block(≤55 bytes)에 들어간다.
- **`H12_r(x)`:** MD5 padding 후 압축함수의 **처음 r step만** 실행한다. 그다음 IV를 word별로 더하고(feed-forward) 표준 little-endian 직렬화한 digest의 **앞 12 bits**를 취한다. `r=64`는 실제 MD5-12와 같다. 참조 구현은 RFC 벡터 5개와 무작위 Printable 메시지 2,000개에서 `hashlib.md5`와 전부 일치했다.
- `r<4`에서는 digest의 첫 word가 상수라 출력이 변하지 않는다. `r≥4`에서 첫 word는 step `r−4`의 결과다. 따라서 `r=4`의 y는 **payload 0–3번째 byte만의 함수**다. 무작위 5,000건에서 4번째 이후 byte 변경으로 y가 바뀐 경우는 0건이었다.

### 4.2 Group ownership과 새 쌍 학습

- 12-bit 값 4,096개를 과제별로 **test 512 / validation 256 / train 3,328 groups**로 나눈다. 학습 쌍은 prior에서 뽑은 x 중 `H(x)`가 train group인 것만 사용한다(수락률 81.25%). 학습 메시지는 test target의 역상이 될 수 없다.
- **새 쌍 학습:** batch 256 × 40,000 updates, run당 새 쌍 10,240,000개다. Data stream은 `(protocol, task, seed, update)` namespace의 결정적 난수로 만든다. 따라서 중단 후 재개해도 같은 batch가 나온다. 같은 seed의 Main과 Shuffled는 같은 stream·초기 가중치·corruption 난수를 공유한다.
- **Shuffled:** 각 batch 안에서 condition을 무작위 permutation으로 재배정한다. Length head와 payload에 같은 donor condition을 쓰며, 우연히 자기 condition을 받은 수를 기록한다. 평가 때는 실제 target을 넣는다.

### 4.3 모델, 손실, sampler

| 항목 | D1-S | D1-T |
|---|---|---|
| Payload denoiser | v4 D1 그대로: 1 hidden layer width 128, embedding 16, condition-output 잔차 | Pre-LN Transformer 4 layers, d=192, 4 heads, FFN 768, 학습 위치 embedding. 시간과 13-d condition(12 bits + L/31)은 투영해 모든 위치에 더하고 condition-output 잔차를 유지 |
| Parameters | 508,668 (코드에서 계수) | 약 1.8M (구현 후 계수해 봉인) |
| Length head | `Linear(12,28)`, 생성 시 1회 sampling | 동일 |
| Optimizer | Adam lr 1e-3 (v4 값) | Adam lr ∈ {1e-3, 3e-4, 1e-4} 중 A-dev에서 synthetic dev split으로만 선택. Warmup 1,000, grad-norm clip 1.0 |

- **손실:** v4와 같은 `length CE + mean(가려진 payload 위치의 CE)`다. 위치 특혜 loss는 쓰지 않는다(`prefix_balanced_loss=false`).
- **Sampler:** v4와 같다. 32 intervals, temperature 1, remask 없음, 후보당 NFE 33. Hidden length·argmax 치환·MD5 기반 reranking은 없다.
- **Checkpoint:** 최종 update의 가중치 하나만 쓴다. 4,000 updates마다 validation loss와 validation CLP(§9)를 기록하지만 **진단용**이며 선택에 쓰지 않는다.

### 4.4 방법과 후보 원장

| 방법 | 정의 | 쓰는 곳 |
|---|---|---|
| Main | 실제 condition으로 학습하고, 요청 target을 condition으로 생성 | A, B, C |
| Shuffled | §4.2의 batch 내 permutation으로 학습하고, 요청 target으로 생성 | C |
| Random | 원래 source prior에서 정확히 직접 sampling | B, C |
| MC (mismatched condition) | Main checkpoint에 **다른 trial의 target**(고정 derangement)을 condition으로 넣어 생성. 성공은 요청 target 기준 | B(주 대조), C(진단) |

- 성공은 `valid(x) ∧ H(x) = 요청 target`이다. **Success@100**은 trial당 K=100 후보 중 성공이 하나 이상인지로 정한다.
- Invalid·duplicate도 기회를 소비한다. 첫 성공 뒤에도 100개를 끝까지 생성한다.
- 원장 키, 독립 재해시 verifier, 난수 identity 기록은 v4 원장 계약(`(protocol, task, method, seed, trial, attempt)`)을 따른다. 후보 난수는 method·seed·trial·attempt별로 독립이다.

---

## 5. Stage 0 — D0 진단 (선택, 결정에 사용하지 않음)

v4 V1의 epoch-100 checkpoint(`update-00015700-epoch-0100.safetensors`, 세 seed)를 V1 acceptance 512 조건에 대해 등록 sampler로 다시 평가한다. 정상/반전 K=1이다.

- **목적:** §1.2의 추정(선택 규칙 때문에 V1 미달)을 확인한다. Epoch 100에서도 ≥461/512에 못 미치면 균일 loss 자체의 학습 효율 문제가 섞여 있다는 뜻이 된다. 그 경우 새 쌍 학습에서도 A-S0 실패 위험이 커진다.
- **범위:** v4 study 폴더에는 쓰지 않고 `local_experiment_archive/analyses/v4-claude-d0/`에만 기록한다. v4 판정을 PASS로 바꾸지 않는다. MD5 호출은 0회이고 비용은 수 분이다.

---

## 6. Stage A — 기계 적격성

**A-prof (자원 측정).** Synthetic 과제에서만 D1-S/D1-T의 update 시간과 생성 batch {64, 256, 1024}의 처리량·메모리를 측정한다. 방법은 v4 V2 profiler와 같다(warm-up 후 반복 측정, 전체 경로 포함). 이 측정으로 batch와 §11의 fallback 적용 여부를 봉인한다.

**A-dev (개발).** Synthetic **dev split**(acceptance와 disjoint한 별도 조건들)에서 D1-T의 lr을 선택하고 학습이 안정적인지 확인한다. 최대 6 runs다. MD5 계열 과제는 사용하지 않는다. 이후 전체 설정을 `protocol.frozen.json`으로 봉인한다.

**A-S0 (적격성).** v4 V1과 같은 synthetic_nibbles 과제다. y의 세 nibble을 앞 3 bytes로 표현하고 suffix는 prior에서 뽑는다. 새 쌍 학습을 사용하며, 설정은 D1-S/D1-T × seeds 0, 1, 2로 6 runs다.

| 기준 (seed별, 모두 충족) | 값 |
|---|---|
| Acceptance 512 조건, 정상 joint | ≥ 461/512 (v4 V1과 동일) |
| 반전 joint | ≥ 461/512 |
| Valid (정상·반전 각각) | 512/512 |
| 반전 후보의 원래 조건 오성공 | ≤ 25/512 |
| CLP 양성 대조 (§9) | one-sided z > 3.26 |

- **적격 architecture 집합 Q** = 세 seed 모두 통과한 architecture다.
- Q가 비면 → **`CONCLUDE_UNTESTABLE`**로 종료한다(Stage B·C 미실행).
- Stage C architecture = D1-T ∈ Q이면 D1-T, 아니면 D1-S. Stage B는 Q의 모든 architecture로 실행한다.
- 생성 기준은 통과했지만 CLP만 실패하면 CLP 구현 결함이다. 이 경우 CLP를 모든 판단에서 제외하고 보고서에 명시한다.

---

## 7. Stage C — 실제 MD5-12 결정 시험 (주 판정)

### 7.1 설계

| 항목 | 고정값 |
|---|---|
| 해시 | `H12_64` = MD5 전체 digest의 앞 12 bits |
| Test pool | 노출 감사로 인증한 미노출 group 512개(§10). Validation 256은 노출 group이어도 됨. 나머지는 train |
| Architecture | §6 규칙으로 정한 하나 |
| Learned runs | {Main, Shuffled} × seeds {0,1,2} = 6 runs. 확장 시 seeds {3,4,5} 6 runs 추가 |
| Trials | 6 checkpoint 봉인 **후** pool에서 iid 복원추출 16,384개. 모든 seed와 방법이 같은 목록을 쓰며 재추첨하지 않음 |
| 후보 | Main·Shuffled·Random 각 16,384 × 100 per seed. MC는 4,096 trials × 100 per seed(진단) |
| 주 지표 | Success@100, trial 단위 paired |
| 보조 | CLP_64 (32,768 pairs per seed), @1/@10, valid·duplicate 비율, valid일 때 hit, 학습 메시지 일치, NFE, 시간 |

### 7.2 통계

대조군 c ∈ {Random, Shuffled}, trial t, seed s에 대해 `d_t^c = mean_s (M_{s,t} − C_{s,t})`로 둔다. Trials는 iid이므로 `Δ̂_c = mean_t d_t^c`이고 `SE_c = sd(d^c)/√T`다.

- **동시 구간:** 두 look × 두 대조군 × 양측으로 α=0.05를 Bonferroni 분할한다. 각 꼬리 0.00625, `z = 2.4977`이다. `[L_c, U_c] = Δ̂_c ∓ z·SE_c`.
- **최소 관심 효과:** δ = **0.005 (0.5%p)**. 계획용 Random 성공률 `p0 = 1−(1−2⁻¹²)¹⁰⁰ = 2.412%`의 약 21% 상대 증가다. v4의 δ=1%p보다 좁은 상한이다.
- **추론 범위:** 봉인된 test pool·checkpoint·seed에 조건부이며, target 추출과 생성 난수에 대한 추론이다. Seed별 추정값은 모두 보고한다.

### 7.3 판정

| 순서 | 조건 | Stage C 결과 |
|---|---|---|
| 1 | `L_Random > 0` **그리고** `L_Shuffled > 0` | **`CONTINUE`** (단, §7.4 artifact 감사 통과가 필요) |
| 2 | `U_Shuffled < δ` 그리고 `U_Random < δ` | `CONCLUDE_BOUNDED` — 두 비교 모두 0.5%p 이상 이득을 배제 |
| 3 | `U_Shuffled < δ` (Random 비교는 미확정 또는 양) | `CONCLUDE_NO_CONDITION_GAIN` — 해시 조건의 기여를 배제. Random 대비 이득이 있어도 조건과 무관 |
| 4 | `U_Random < δ` (Shuffled 비교는 미확정 또는 양) | `CONCLUDE_NO_RANDOM_ADVANTAGE` — prior sampling보다 유용하지 않음 |
| 5 | 그 외 | 첫 look이면 **1회 확장**(seeds 3–5를 추가하고 6 seeds 합산으로 다시 판정). 두 번째 look이면 `CONCLUDE_UNRESOLVED` + 구간 보고 |

확장은 결과를 보고 결정하는 증액이 아니다. 사전 등록한 단일 추가 look이며, α는 이미 두 look으로 나눠 두었다.

**설계 검산**(가상 결과, 시나리오당 1,000회, 확장 포함):

| 가정 (seed별 Success@100) | CONTINUE | CONCLUDE (주 사유) |
|---|---:|---|
| 모두 p0 (효과 없음) | 0.1% | 99.9% (BOUNDED 99.1%) |
| Main = p0 + 0.25%p | 33.7% | 66.3% — δ 아래의 효과이므로 양쪽이 섞임 |
| Main = p0 + 0.5%p | 99.0% | 1.0% |
| Main = p0 + 1%p 또는 2p0 | 100% | 0% |
| Main = p0 + 1%p, 매 look 한 seed는 p0 | 100% | 0% |
| Main = Shuffled = p0 + 0.5%p (prior 모델링 이득) | 0.7% | 99.3% (NO_CONDITION_GAIN 98.2%) |
| Main = p0 − 0.3%p (학습 메시지 재생산) | 0% | 100% (BOUNDED) |

### 7.4 CONTINUE 전 artifact 감사 (필수)

1. 모든 성공 payload를 독립 verifier로 재해시한다.
2. 성공 후보 중 학습 메시지와 같은 것은 0이어야 한다(group disjoint 계약). 위반이 있으면 무결성 실패다.
3. 성공이 특정 target에 몰렸는지 확인한다. 상위 1% target의 성공 비중을 보고한다.
4. MC와 CLP_64의 방향을 보고한다. 둘 다 무신호인데 hit만 양성이면 `CONTINUE (hit-only)`로 표시하고, 후속 연구의 첫 과제를 새 holdout 재현으로 지정한다.

감사 실패는 CONTINUE를 무효화한다. 원인을 고친 뒤 해당 stream을 같은 난수 identity로 재생성한다.

---

## 8. Stage B — Step-reduced MD5 사다리 (보조; 결정 불변)

### 8.1 목적

"기계가 해시 구조를 전혀 이용하지 못하는가", 아니면 "구조가 있으면 이용하지만 MD5의 mixing이 그것을 지우는가"를 구분한다. 결과는 종료 보고서의 핵심 근거가 되고, CONCLUDE일 때 후속 연구 전환(pivot)을 고려할 가치가 있는지 표시한다.

### 8.2 Rung과 모델 없는 난이도 측정

Rung `R_B = {4, 5, 6, 7, 8, 10, 12, 16, 32}`이다. r=64는 Stage C에서만 다룬다. 아래 값은 모델 없이 계산한 profile이다. 무작위 byte 1개를 다른 Printable 값으로 바꿨을 때 출력 12 bits의 변화를 보고, 단일 byte와 y 상위 4 bits 사이의 상호정보량을 추정했다.

| r | 출력 bit 반전율 | y 불변 확률 | 단일 byte와의 최대 MI (bits) |
|---:|---:|---:|---:|
| 4 | 0.036 | 0.813 | 1.876 (byte 3) |
| 5 | 0.111 | 0.553 | 0.021 |
| 6 | 0.179 | 0.409 | ≈ bias |
| 7 | 0.277 | 0.256 | ≈ bias |
| 8 | 0.336 | 0.154 | ≈ bias |
| 10 | 0.445 | 0.034 | ≈ bias |
| **12** | **0.486** | **0.004** | ≈ bias |
| 16 / 32 / 64 | 0.499 / 0.500 / 0.501 | ≤ 0.0004 | ≈ bias |

"≈ bias"는 plug-in 추정 bias 상한(0.012 bits) 이하라는 뜻이다. 이 표를 근거로 **완전 mixing rung `r_mix = 12`**를 지금 봉인한다. 정의는 "반전율이 0.48 이상인 첫 rung"이다. 단순 통계로 보면 단일 byte 구조는 r=5에서 이미 거의 사라지고(0.021 bits) r≥6에서는 bias 수준이다. r=12부터는 출력이 완전히 뒤섞인다. 따라서 모델 성능이 급격히 떨어지는 구간은 4–12에 있을 가능성이 높아, rung을 그 구간에 촘촘히 두었다.

### 8.3 실행과 검정

| 항목 | 값 |
|---|---|
| Runs | Q의 각 architecture × 각 rung × seed 0 (최대 18 runs). Main만 학습 |
| Split | Rung별 무작위 test 512 / validation 256 / train 3,328 groups. `H12_r (r<64)`는 이 프로젝트에서 한 번도 조건으로 쓰지 않았으므로 노출 감사가 필요 없음 |
| 평가 | Trials 4,096 × K=100. Main, MC, Random. Random은 rung마다 두 architecture가 공유. CLP 32,768 pairs |
| GEN(r) | Main−Random **그리고** Main−MC의 paired Success@100 one-sided z > 3.26 (셀당 α = 0.01/18, intersection-union) |
| INFO(r) | CLP one-sided z > 3.26 |
| 지평선 | `r*_gen` = GEN이 성립한 가장 큰 rung. `r*_info`도 같은 방식. 비단조 패턴은 그대로 보고 |
| 재현 | 각 architecture의 `r*_gen`과 바로 위 rung에 seeds 1, 2를 추가(최대 8 runs). **확정 지평선** = 세 seed 중 2개 이상에서 GEN이 성립한 가장 큰 재현 rung. 성립하지 않으면 바로 아래 rung |

Ladder에서 셀당 검정력 90%로 검출 가능한 Success@100 차이는 약 1.5%p다(p0 부근, `z=3.26`). 지평선 부근의 작은 효과를 놓칠 수 있다. 그래서 더 민감한 CLP로 `r*_info`를 따로 측정하고, 둘의 차이("정보는 있으나 생성으로 이용하지 못함")를 진단으로 보고한다.

### 8.4 해석 표시 (결정에는 영향 없음)

- **`PIVOT_SUPPORTED`:** 어느 architecture든 확정 `r*_gen ≥ 16`, 즉 `r_mix`보다 깊은 rung에서 생성 우위가 확인된 경우. 단순 통계가 보지 못하는 구조를 모델이 이용한다는 뜻이다. "Step-reduced MD5에서 신경망 역상 후보 생성의 지평선"을 별도 새 연구로 제안할 근거가 된다.
- **`PIVOT_NOT_SUPPORTED`:** 그 외. `r*_gen`의 값(예: 4, 6, 없음)과 D1-S와 D1-T의 차이를 보고한다.
- **`r=4`에서도 GEN이 없으면:** MD5에서 구조가 가장 강한 rung(y가 4 bytes만의 함수)조차 이용하지 못한다는 뜻이다. 종료 보고서에서 "현재 기계는 MD5형 연산 구조를 역으로 이용하지 못한다"의 근거로 쓴다.

---

## 9. CLP — Conditional Likelihood Probe

Hit 수 계산은 성공이 드물어서 정보가 적다. CLP는 학습된 모델의 우도가 **정답 condition에서 더 높은가**를 쌍 대비로 직접 측정한다. 생성 없이 forward pass만 쓰므로 싸고 민감하다.

- **점수:** `s(x, y)` = −(length CE + 가려진 payload CE)를 고정 corruption draw 8개로 평균한 값. 두 condition에 같은 draw를 쓴다. Test group에 속한 held-out 메시지 쌍 `(x_i, y_i)`, `(x_j, y_j)`에 대해 `D = s(x_i,y_i) + s(x_j,y_j) − s(x_i,y_j) − s(x_j,y_i)`로 정의한다.
- **정확한 귀무:** x와 y가 독립이면(r=64에서 random-oracle 이상화) `(x_i,y_i,x_j,y_j)`와 `(x_i,y_j,x_j,y_i)`의 분포가 같다. 따라서 D는 0에 대해 대칭이다. Condition에만 의존하는 점수 성분(`g(y)`)은 상쇄된다. 서로 독립인 쌍 32,768개로 one-sided 검정을 한다(정규 근사, sign-flip과 동치). 이 통계는 ELBO가 정확한 하한이 아니어도 유효하다.
- **검산:** 가상 null 2,000회에서 α=0.01 기각률 1.15%로, condition-only 성분이 있어도 보정이 맞았다.
- **한계:** CLP 양성은 "조건이 held-out 데이터의 우도를 설명한다"는 뜻이지 생성 성공을 보장하지 않는다. 반대로 prior 밖의 좁은 역상 집합에 질량을 모으는 모델은 CLP로 잘 보이지 않을 수 있다. 그래서 결정은 hit(Stage C)로만 내리고, CLP는 지평선 측정·양성 대조·진단에 쓴다.

---

## 10. 노출·무결성·calibration

**노출.** Stage C test pool 512 groups는 v4 노출 감사 도구(`study_v4 audit`와 schema)로 인증한다. 제외 대상은 v4 inventory 범위의 기존 노출(하한 1,885 groups)에 다음을 더한 것이다.

- v4-study V0/V1 fixture target
- v4-claude D0·A 단계에서 계산한 MD5 target (D0·A는 MD5 호출 0회가 목표이며 실제 호출 수를 기록)

인증된 미노출 group이 512개 미만이거나 미해결 항목이 남으면, **한 번에 한해 사전 등록한 대체 window**로 Stage C 전체를 수행한다. 대체 window는 `int.from_bytes(md5(x).digest(), 'big') & 0xFFF`, 즉 digest의 마지막 12 bits다. 이 window가 과거 어떤 학습·선택·평가의 조건으로도 쓰이지 않았음을 감사 기록에 명시한다. Stage B의 `H12_r (r<64)`는 새로운 함수이므로 노출 대상이 아니다.

**무결성 검사**(v4 V0 재사용 + 추가):

- `H12_r` 참조 구현: r=64가 hashlib과 일치, r=4가 bytes 0–3에만 의존
- 새 쌍 stream의 결정성과 재개 일치
- Train group rejection이 정확한지(학습 메시지의 해시가 모두 train group)
- Shuffled permutation 기록, MC derangement에 자기 자신이 없는지
- CLP 대칭성 fixture
- 원장→trial→counts 전수 fixture

**Production calibration** (v4 §8.3 방식):

- 실제 판정 코드에 가상 joint outcome을 넣어 §7.3의 일곱 시나리오를 시나리오당 20,000회 실행한다.
- 통과 조건: 효과 없음에서 CONTINUE 비율의 one-sided 95% CP upper ≤ 0.01, +0.5%p에서 CONTINUE의 lower ≥ 0.95, 효과 없음에서 CONCLUDE_BOUNDED의 lower ≥ 0.95.
- **Planted-lift fixture:** Random sampler에 사전 계산한 역상 table을 일정 비율로 섞어, 실제 원장 경로에서 +0.5%p가 CONTINUE로 판정되는지 확인한다. 이 fixture는 test pool을 사용하지 않는다.

---

## 11. 실행 순서, 자원, fallback

**순서:** D0 → A-prof → A-dev → 봉인 → A-S0 → **C (첫 look)** → (C 확장, 해당 시) → B → 최종 보고. 주 판정을 먼저 확보하고, 보조인 B가 자원을 먼저 쓰지 않게 한다.

**고정 실행량** (검산 스크립트 산술):

| 항목 | 수량 |
|---|---:|
| Learned runs, 첫 통과 (A-S0 6 + B 18 + C 6) | 30 |
| Learned runs, 최대 (+ B 재현 8, C 확장 6) | 44 |
| Updates, 첫 통과 (run당 40,000) | 1,200,000 |
| B learned 후보, seed 0 (Main + MC) | 14,745,600 |
| C learned 후보, 첫 look (Main, Shuffled 각 16,384 + MC 4,096, × 3 seeds × 100) | 11,059,200 |
| C Random 후보, 첫 look | 4,915,200 |
| Learned sampling NFE, 첫 통과 | 851,761,152 |

**참고 실측값** (v4 V1, D1-S, MLX/Metal):

- 학습: batch 64 기준 15,700 updates에 196–206 s(≈0.013 s/update).
- 생성: batch 4 기준 3,072 후보에 21.3 s(≈144 후보/s). 작은 batch의 launch overhead가 지배적인 값이다.

본 계획은 A-prof의 실측으로 시간을 다시 추정한다. 이 수치로 D1-T 비용을 외삽하지 않는다.

| Hard cap (active time) | 값 |
|---|---:|
| D0 + A 전체 | 8시간 |
| C 첫 look | 20시간 |
| C 확장 (해당 시) | 20시간 |
| B 전체 | 24시간 |
| v4-claude 전체 | 72시간 |
| 저장량 / RSS / GPU / 최소 디스크 여유 | 64 / 64 / 64 / 10 GiB (v4와 동일) |

**사전 등록 fallback.** A-prof 직후, 결과를 보기 전에 한 번만 적용한다. 예상 시간×1.5가 cap을 넘으면 아래 순서로 적용한다.

1. B trials 4,096 → 2,048
2. B의 D1-S 재현 생략
3. C의 MC 4,096 → 1,024 trials
4. B의 rung 32 생략
5. C architecture를 D1-S로 변경

**C의 Main/Shuffled/Random trials·K·seeds·updates는 절대 줄이지 않는다.**

**Cap 초과 시:** 실행 중 hard cap에 걸리면 해당 단계를 멈춘다. C 첫 look이 완료되지 않았다면 결론은 **`CONCLUDE_BY_BUDGET`**이다. 이는 과학적 음성이 아니라, 이 연구에 추가로 투자할 근거를 만들지 못했다는 뜻이다(원칙 1). 고립된 실행 오류는 같은 난수 identity로 1회 재실행할 수 있다(v4 resume 계약).

---

## 12. 최종 결정 규칙과 허용되는 주장

| 최종 결정 | 조건 | 다음 행동 | 허용되는 주장 |
|---|---|---|---|
| **`CONTINUE`** | Stage C §7.3 ①과 §7.4 감사 통과 | 원래 연구를 확장한다. 새 holdout 재현, 두 번째 source(Random Bytes), 계산비용 비교(학습·lookup 전처리 포함)를 담은 확증 계획을 작성한다. Stage B 지평선으로 architecture를 선택한다 | 고정 Printable·MD5-12·설정에서 해시 조건 모델이 두 대조군보다 Success@100이 높았다(구간 제시). Full MD5 역상, 보안 붕괴, 계산 우위는 주장할 수 없다 |
| **`CONCLUDE_BOUNDED`** | ② | 연구를 종료하고 종료 보고서를 작성한다 | 이 설정에서 해시 조건 이득은 Random·Shuffled 대비 모두 0.5%p 미만이다(상한 제시) |
| **`CONCLUDE_NO_CONDITION_GAIN`** | ③ | 종료 | 해시 조건의 기여는 0.5%p 미만이다. Random 대비 차이가 있다면 조건과 무관한 prior 효과다 |
| **`CONCLUDE_NO_RANDOM_ADVANTAGE`** | ④ | 종료 | Prior sampling 대비 유용한 이득이 없다 |
| **`CONCLUDE_UNRESOLVED`** | 확장 후에도 ⑤ | 종료. 구간을 보고하고 추가 증액은 없음 | 결론 불충분. 효과가 δ 근처일 가능성만 남는다 |
| **`CONCLUDE_UNTESTABLE`** | A-S0에서 Q가 빔 | 종료. 기계 결함 분석을 보고한다 | MD5 효과는 검정되지 않았다. 현재 기계로는 이 질문을 시험할 수 없다 |
| **`CONCLUDE_BY_BUDGET`** | C 첫 look 전에 cap 소진 | 종료 | 자원 안에서 지속 근거를 만들지 못했다. MD5 음성이 아니다 |

모든 CONCLUDE의 종료 보고서에는 다음을 포함한다.

- (a) 조건부 생성 기계의 성과: P-DISC/R-G-BGV synthetic, A-S0 결과
- (b) 과거 MD5 PoC와 toy 결과
- (c) Stage B 지평선 `r*_gen`/`r*_info`, `PIVOT_*` 표시, D1-S와 D1-T의 비교
- (d) Stage C의 상한과 모든 대조 결과
- (e) 적용 범위: Printable, MD5-12, 설정한 모델·데이터·예산

`PIVOT_SUPPORTED`는 원래 연구의 지속이 아니다. 별도 새 연구를 제안할 근거일 뿐이다.

---

## 13. 구현 범위

**재사용:** `study_v4_data`(seed derivation, 노출 감사, split), `study_v4_runtime`(Budget, checkpoint·resume, 원장·verifier, 평가 루프), `study_v4_statistics`(CP primitive), `mlx_models.SequenceDenoiser`/`MaskedDiffusion`(D1-S, 균일 loss), TokenCodec, v4 CLI 구조.

**신규:**

1. `hashing`: 벡터화한 `H12_r` (NumPy)와 참조 테스트. 검산 스크립트의 구현이 참조가 된다.
2. 새 쌍 data stream: train group rejection, 결정적 재개, 학습 메시지 hash set 기록.
3. `mlx_models`: D1-T (같은 호출 계약).
4. Batch 내 Shuffled, MC derangement 생성기.
5. CLP scorer와 대칭성 fixture.
6. Stage C 분류기(§7.3)와 ladder 검정. Production calibration, planted-lift fixture 포함.
7. `study_v4c` CLI: `plan | d0 | audit | run --stage {A,C,B} | report`. 단계 봉인, resume, cap, fallback 기록.

v3.1/v4 실행기와 산출물은 수정하지 않는다. 새 CLI는 수정된 protocol JSON을 거부한다.

## 14. 산출물

1. `protocol.frozen.json`, code·환경 manifest, fallback 적용 기록, 노출 감사(또는 대체 window 사용 기록)
2. D0 진단, A-prof/A-dev/A-S0 결과와 Q
3. Stage C: 6 (또는 12) runs의 학습 기록·최종 checkpoint 봉인, trial 목록, 모든 원장, verifier 감사, 효과·구간·판정, artifact 감사
4. Stage B: 셀별 GEN/INFO, 지평선, 재현, `PIVOT_*`
5. `decision.json`: 실행 상태와 최종 결정을 분리하고 사유 코드를 기록
6. 한국어 최종 보고서(`report.ko.md`): 결정, 근거, 적용 범위, 다음 행동

**설계 검산 재현** (가상 계산만 수행. 실제 실험 명령이 아님):

```sh
.venv/bin/python scripts/validate_research_plan_v4_claude.py
```

출력: `local_experiment_archive/analyses/v4-claude-design-20260928/design_calculation.json`. 내용은 `H12_r` 참조 일치, rung별 난이도 profile, CLP null 보정, Stage C 판정의 가상 작동 특성, 실행량 산술이다.
