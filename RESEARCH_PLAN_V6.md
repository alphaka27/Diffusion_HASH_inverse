# v6 실험 계획 — 5개 파이프라인 비교와 연구의 최종 결론

**Protocol:** `dhi-v6-20260929` · **Master seed:** `2026092907` · **작성일:** 2026-09-29 KST
**상태:** `PLAN_ONLY` — 계획과 설계 검산만 완료했다. 구현, 노출 감사 갱신, 적격성 검사, MD5 조건 데이터 생성은 하지 않았다.

이 문서의 수치는 세 종류로 나뉜다.

- **기존 archive에서 읽은 값.** V5 실행 기록(`v5-study-certified`)과 v3.1 p2struct 분석이다. 읽기만 했다.
- **계획 작성 중 새로 측정한 값.** §1.4의 처리량 proxy다. 임의 가중치만 썼고 연구 데이터 접근과 MD5 호출은 0회다.
- **설계 검산값.** [검산 스크립트](scripts/validate_research_plan_v6.py)의 가상 계산 결과다.

**요약.** V6의 목표는 두 가지다. 첫째, 원본 계획의 5개 파이프라인(P-G-BGV, P-G-CGGE, P-DISC, R-G-BGV, R-DISC)을 실제 MD5-12에서 비교한다. 이때 target, 후보 예산, 학습 데이터량, 판정 규칙을 모두 같게 둔다. 둘째, 어떤 결과가 나와도 연구 가설에 대한 최종 판정을 내리고 연구를 종료한다.

1. **V5가 판정 없이 끝난 원인을 설계로 막는다.** V5는 기계 적격성(C1)을 통과했지만 Stage C 첫 look이 16시간 cap에 걸렸다. 실제 생성 속도가 계획에 쓴 proxy의 약 55%였기 때문이다(§1.2). V6는 end-to-end 실측으로 예산을 봉인한다. 또 3-look 순차 설계를 써서, 예산이 중간에 끝나도 마지막으로 완료된 look의 유효한 판정이 남게 한다.
2. **Printable Gaussian의 적격성 실패 원인을 해결한다.** v3.1 p2struct에서 P-G-BGV는 요청 prefix를 256/256 맞혔지만 suffix 바이트의 38.6%가 Printable 범위를 벗어나 탈락했다(§1.3). V6는 이미지 decoder를 "source alphabet 안의 최근접 prototype"으로 통일한다. D1이 출력 logits를 payload 기호로만 제한하는 것과 같은 조건이다(§4.4).
3. **비교를 공정하게 만든다.** 5개 모두 같은 규약을 쓴다. 새 쌍 학습 40,000 updates × 256(1,024만 쌍), 최종 checkpoint, 균일 payload loss, 같은 W3 test pool과 trial 목록이다. 파라미터 수도 0.51M–1.25M 범위로 맞췄다.
4. **효과가 없을 때의 해석력을 보강한다.** 각 파이프라인에서 step-reduced MD5(r=4) 양성 대조를 따로 잰다(Stage P). "MD5-64에서 효과가 없다"는 결론이 기계 결함 때문이 아님을 보이기 위해서다.

관련 문서: [V6 구현 세부 명세](V6_IMPLEMENTATION_SPEC.md) · [원본 계획](RESEARCH_PLAN.md) · [V5 계획](RESEARCH_PLAN_V5.md) · [V5 구현 안내](V5_CLI.md) · [F1 후속 계획](FOLLOWUP_CONFIRMATORY_F1_PLAN_KO.md) · [p2struct 분석](V3_1_P2STRUCT_ANALYSIS_KO.md) · [연구 가능성 검토](RESEARCH_FEASIBILITY_REVIEW_KO.md) · [설계 계산 JSON](local_experiment_archive/analyses/v6-design-20260929/design_calculation.json)(로컬 전용) · [처리량 측정](local_experiment_archive/analyses/v6-design-20260929/throughput.json)(로컬 전용)

---

## 인계 안내

명세의 기준은 이 문서의 §3–§17이다. V5·F1 계획과 충돌하면 이 문서가 우선한다. 단, V5의 공식 판정은 바꾸지 않는다.

### 현재 상태 (2026-09-29)

| 항목 | 상태 |
|---|---|
| 계획 문서, 설계 검산 스크립트 | 완료, 미커밋 |
| 처리량 proxy 측정 | 완료. 결과는 로컬 archive에 있고 수치는 §1.4에 인용 |
| 구현 세부 명세 | 완료: [V6_IMPLEMENTATION_SPEC.md](V6_IMPLEMENTATION_SPEC.md), 등록값 [examples/v6-protocol.json](examples/v6-protocol.json) |
| `src/dhi_v6/` 구현 | 시작 전(§16) |
| 노출 감사 갱신(V5 실행 포함, W3·W4 인증) | 시작 전(§13.1) |
| Stage A–S 실행 | 시작 전. MD5 조건 데이터 없음 |
| F1 후속 계획 | 계획만 있음. V6와의 관계는 아래 결정 항목 2 |

### 저장소에 없는 자료

`local_experiment_archive/`는 `.gitignore` 대상이다. 필요한 값은 다음 위치에 인용했다.

| 로컬 전용 자료 | 문서 안의 위치 |
|---|---|
| V5 Stage C 부분 결과, 학습·생성 시간 | §1.2 |
| V5 A-Q 결과, v3.1 p2struct 결과 | §1.3 |
| V6 처리량 proxy | §1.4 |
| 설계 계산 JSON | §6.4, §11, §12, §14 |

[검산 스크립트](scripts/validate_research_plan_v6.py)는 archive가 없으면 기록된 값을 대신 쓰고, 출력의 `measurement_sources`에 어느 쪽을 썼는지 남긴다.

### 착수 전에 사람이 정할 항목 (권장 기본값)

1. **자원 cap.** 권장값은 A 24시간(+보완 12시간), C 80시간, P 10시간, R 36시간, S 10시간이다. 필수 경로(A, C, P)의 합은 114시간이다. M3 Max 1대 기준 예상 실행 시간은 기대 약 73시간, 최악 약 83시간이다(Gaussian 25 step 기준, §14). Gaussian이 50 step으로 정해지면 기대 약 97시간이다.
2. **F1 처리.** 권장: 실행하지 않고 V6로 대체한다. F1은 W2의 고정 D1-T checkpoint만 다룬다. V6의 P-DISC arm은 새 window와 새 학습으로 같은 질문을 더 넓게 다룬다. F1을 실행하더라도 V6와 자료를 합치지 않는다.
3. **최소 관심 효과 δ = 0.5%p.** V5의 0.25%p를 5개 파이프라인에 적용하면 생성 시간만 약 97시간이 더 든다(§6.6).
4. **Prototype decoder 도입(§4.4).** 도입하지 않으면 Printable Gaussian 두 파이프라인은 과거처럼 적격성에서 탈락할 가능성이 크고, 5개 비교가 3개 비교로 줄어든다.
5. **Stage S 포함 여부.** 선택 항목이며 약 6.5시간이 든다(§9).
6. **커밋 범위.** [NEW_EXPERIMENT_PLAN.md](NEW_EXPERIMENT_PLAN.md) 규칙대로 코드, 테스트, 고정 설정, 계획만 커밋한다.

위 항목이 권장값과 다르게 정해지면 구현 전에 [examples/v6-protocol.json](examples/v6-protocol.json)과 그 SHA-256 파일을 먼저 갱신한다.

### 반드시 지킬 제약

- V3.1·V4·V5 실행기, `src/dhi_v5/`, `local_experiment_archive/runs/` 아래 산출물은 수정하지 않는다.
- V5 공식 판정(`FINAL_NOT_ESTABLISHED` / `NOT_ESTABLISHED_BY_BUDGET`)은 바꾸지 않는다. V6 결과를 V5나 F1의 완료로 보고하지 않는다.
- 결과를 본 뒤 규칙, seed, 학습량, sampler, trial, δ를 바꾸지 않는다. 사전 등록된 분기만 허용한다(§3).
- W3·W4·r=4의 MD5 조건 데이터는 봉인(§5.6) 뒤에만 만든다. Stage C를 P·S보다 먼저 실행한다.
- 실패, 차단, 자원 초과를 효과가 없다는 증거로 쓰지 않는다. §15의 `NOT_ESTABLISHED_*`로 기록한다.

### 작업 순서와 완료 기준

| 순서 | 작업 | 완료 기준 |
|---|---|---|
| 1 | §16 구현 | §5.1 A-impl 게이트와 §13.2 무결성 검사를 테스트로 통과 |
| 2 | 판정기 calibration | §13.3 통과 조건 |
| 3 | A-prof-1 → A-Q → A-dev → (보완) → A-prof-2 → 예산 계획 → 감사·봉인 | `protocol.frozen.json`, `window.json`, `budget-plan.json` |
| 4 | C 학습 → trial 봉인 → look 1–3 | look별 판정 기록 |
| 5 | (감사 → R) → P → (S) | 단계별 봉인 |
| 6 | 최종 보고 | §15.3 구성의 `FINAL_REPORT_KO.md`, `decision.json`, 비교표 |

---

## 0. 최종 결론 카드 (실험 종료 시 이 표를 채운다)

| 질문 | 판정 값 (파이프라인별 5칸) | 근거 단계 |
|---|---|---|
| **C1 기계.** 조건부 생성이 학습 가능한 조건을 실제로 쓰는가 | `PASS` / `UNTESTABLE` | Stage A |
| **C2 구조 이용.** 해시 구조가 약할 때(r=4) 그것을 이용하는가 | `GEN_4`, `INFO_4` 성립 여부와 효과 크기 | Stage P |
| **C3 연구 가설.** 미학습 MD5-12 target에서 Main이 Random과 Shuffled를 모두 능가하는가 | `SUPPORTED` / `REJECTED_*` / `NOT_ESTABLISHED_*` + Δ_R, Δ_S 구간 | Stage C, R |
| **C4 계산 우위.** 같은 비용에서 MD5 직접 시도보다 빨리 역상을 찾는가 | `NO_ADVANTAGE` / `PER_QUERY_ADVANTAGE` | §12 |
| **C5 파이프라인 비교.** 표현·모델 계열·source에 따라 효과나 능력이 다른가 | 사전 지정 대비 6쌍의 구간, 능력 지표 | §11 |
| **종합.** 연구 종료 판정 | §15.2의 네 가지 중 하나 | 위 다섯 줄 |

**사전 예측**(판정 규칙에 영향 없음)은 다음과 같다.

- C1: 5개 모두 `PASS`.
- C3: 5개 모두 `REJECTED_BOUNDED`. 상한은 파이프라인별로 약 0.25–0.30%p.
- C2: 대부분 `GEN_4`가 성립한다. 불확실성이 크다.
- CLP_64: 무신호. C4: 모두 `NO_ADVANTAGE`. 종합: `FINAL_REJECTED`.

근거는 §1.2의 V5 부분 결과와 §1.5의 이론적 기대다. 이 예측이 맞을 가능성이 높다는 점이 실험을 무의미하게 만들지는 않는다. V6의 목적은 그 예측을 **5개 표현·모델 계열 모두에서, 기계 결함·검정력 부족·예산 부족이라는 반론이 닿지 않는 형태로** 확정하는 것이다.

---

## 1. 지금까지의 증거

### 1.1 버전별 결산

| 버전 | 결과 | 결론이 나지 않은 이유 |
|---|---|---|
| 2026-09-21 MD5 PoC (q12) | Printable Main 0/305, Random 9/305, Shuffled 0/305 | Gaussian Main valid 1/247,500. 형식 실패 |
| Toy MD5-8 | Diffusion 3/18, Source-prior 7/18 | 표본이 작다 |
| v3 / v3.1 (p2struct) | P-DISC 123/125, R-G-BGV 128/128, R-DISC 121/126, P-G-CGGE 39/47, P-G-BGV 5/11 | 5개 적격성 미완(§1.3) |
| v4 | `BLOCKED_QUALIFICATION` | Checkpoint 선택 규칙 |
| v5 | C1 `PASS`(D1-S, D1-T 모두 512/512), C3 `NOT_ESTABLISHED_BY_BUDGET` | Stage C 첫 look 미완(§1.2) |
| F1 | 계획만 있음 | — |

### 1.2 V5 Stage C 부분 결과 (W2, 기술통계, 확증 아님)

V5는 W2 test pool에서 65,536 trials × K=100으로 D1-T stream을 생성했다. 완료된 것은 Main 3 seed, Shuffled 2 seed, Random 2 seed, MC 2 seed다. Shuffled seed 2는 62%에서 중단되었고 Random·MC seed 2는 시작하지 않았다.

| Stream | Success@100 | 후보당 hit |
|---|---|---|
| Main seed 0 / 1 / 2 | 2.347% / 2.481% / 2.438% | 0.000237 / 0.000251 / 0.000246 |
| Shuffled seed 0 / 1 | 2.353% / 2.350% | 0.000238 / 0.000237 |
| Random seed 0 / 1 | 2.443% / 2.437% | 0.000248 / 0.000247 |
| 이론값 | p0 = 2.412% | 2⁻¹² = 0.000244 |

- Seed 0–1의 짝 비교는 Main−Random = −0.026%p, Main−Shuffled = +0.063%p다. 각 기술 SE는 약 0.06%p로, 둘 다 0과 구별되지 않는다.
- 이 값은 판정 규칙을 적용한 결과가 아니며 V5 판정을 바꾸지 않는다. V6 설계자는 이 값을 본 상태에서 계획을 세웠다. V6는 이 값을 판정 자료로 쓰지 않고, 한 번도 조건으로 쓰지 않은 W3에서 새로 학습·평가한다.
- **예산 실패의 원인.** 학습 모델 stream의 실제 속도는 748–874 후보/s였다. A-prof의 짧은 측정값 1,369 후보/s의 55–64%다. Stage C의 57,601초 중 평가 stream이 약 45,700초, 학습 6 run이 약 11,900초를 썼다. 학습은 run당 약 1,983초로 update 시간 proxy의 1.6배다. 계획은 proxy 1,418 후보/s로 생성 8.7시간을 예상했었다.

### 1.3 파이프라인별 적격성 이력

v3.1 p2struct는 source별 10,000개 메시지를 100 epoch 반복 학습했다(seed 100, 최종 checkpoint). 분모는 128이다.

| Pipeline | p2struct 정상 / 반전 joint | valid /256 | 실패 원인 | V5 |
|---|---:|---:|---|---|
| P-G-BGV (G3) | 5 / 11 | 16 | prefix는 256/256 정답. suffix 바이트 38.6%가 0x21–0x7E 밖 | — |
| P-G-CGGE (G3) | 39 / 47 | 86 | suffix glyph의 prototype 거리가 0.1 초과. valid 86개는 prefix가 모두 정답 | — |
| P-DISC (D1) | 123 / 125 | 256 | 통과 | D1-S·D1-T 3 seed 모두 512/512, CLP z 106–121 |
| R-G-BGV (G3) | 128 / 128 | 256 | 통과 | — |
| R-DISC (D1) | 121 / 126 | 256 | 정상 조건 1개 부족(확률적 오선택) | — |

- Printable Gaussian의 실패는 조건 사용의 실패가 아니라 **suffix 형식의 실패**다. 길이가 길수록 valid가 떨어졌다. P-G-BGV는 21–31자에서 0/103이었다.
- 같은 BGV라도 Random Bytes source는 모든 바이트가 유효하므로 통과했다.
- v3.1 학습은 적은 메시지를 반복했고, G3는 synthetic 과제의 정답 위치를 이용하는 prefix 가중 loss를 썼다. V5는 D1에서 새 쌍 학습, 최종 checkpoint, 균일 loss로 512/512를 얻었다. V6는 이 V5 규약을 5개 모두에 적용한다.

### 1.4 처리량 측정 (신규, MLX 0.32.2, Apple M3 Max 128 GB, float32, 임의 가중치)

| 모델 | Params | 후보/s (NFE) | 학습 update (batch 256) |
|---|---:|---|---:|
| G3 U-Net w32, BGV 2×32×128 | 910,638 | 587 (26) / 299 (51) / 151 (101) | 0.113 s |
| G3 U-Net w32, CGGE 2×32×64 | 513,326 | 1,168 (26) / 596 (51) / 301 (101) | 0.057 s |
| D1-S (Printable) | 508,668 | 20,344 (33) | 0.0018 s (V5 A-prof) |
| D1-T | 1,866,316 | 1,234 (33) (V5 A-prof 1,369) | 0.030 s (V5 A-prof) |
| D1-T-L | 6,415,308 | 380 (33) (V5 A-prof) | 0.103 s (V5 A-prof) |

- Gaussian 후보/s는 최적 batch(256–2,048)의 순전파 속도를 NFE로 나눈 값이다. 순전파 속도는 BGV 15,267행/s, CGGE 30,371행/s였다. Width 64 U-Net은 BGV 100 step에서 약 60 후보/s라 채택하지 않았다.
- **V6 비용의 대부분은 Gaussian 파이프라인이다.** BGV는 25 step에서도 D1-S보다 약 35배 느리다.
- 예산 계산에는 V5 실측 보정을 적용한다. 생성 stream 효율은 0.546(V5 최저 실측 748.04 / A-prof 1,369.36)이고, 학습 update당 부대비용은 19 ms다.
- R-DISC(259 상태 출력)는 측정하지 않았다. 예산에는 P-DISC의 절반 속도를 가정했고 A-prof에서 실측한다.

### 1.5 이론적 사전 기대

V5 §1.4의 random-function 논증을 따른다. 학습에 쓰지 않은 입력에서 MD5의 12-bit window가 이상적 random function처럼 행동하면, MD5를 호출하지 않는 어떤 sampler든 새 후보 하나의 성공확률은 2⁻¹²다. 이 논증은 sampler 구조(Gaussian, discrete), 표현(BGV, CGGE, token), source에 의존하지 않는다. 따라서 **5개 파이프라인의 MD5-64 효과는 모두 0으로 예측된다.**

파이프라인이 실제로 달라질 수 있는 곳은 다음 네 가지다. synthetic 조건 정확도, r=4 구조 이용 능력, 후보 다양성(duplicate가 많으면 Main−Random이 음수가 된다), 비용이다. V6는 이 네 가지를 비교표에 함께 싣는다(§11).

---

## 2. V5 대비 변경점

| 항목 | V5 | V6 | 근거 |
|---|---|---|---|
| 비교 대상 | D1 계열 하나(주 모델 D1-T) | 5개 파이프라인, 같은 규약 | 목표 1 |
| Discrete 모델 | D1-T (1.87M) | D1-S (P 0.51M, R 1.25M) | Gaussian과 규모를 맞추고 비용을 줄인다. V5에서 D1-S도 512/512 |
| Gaussian 모델 | 없음 | G3-U: G3 + 균일 payload loss + 새 쌍 학습 + 최종 checkpoint | synthetic 전용 prefix loss 제거 |
| 이미지 decoder | strict(bit threshold, 거리 0.1 거부) | source alphabet 최근접 prototype | §4.4 |
| 주 window | W2 | W3, 재현은 W4 | W2는 V5가 노출 |
| δ | 0.25%p | 0.5%p | 5개 파이프라인 비용(§6.6) |
| Trials | 65,536 × 3 seeds + 확장 look 1회 | 8,192-trial 블록 × 최대 3 look, 전체 공동 중단 | 예산이 끝나도 유효한 판정이 남음 |
| 동시 구간 | 2 looks × 2 대조군 × 양측 | 3 looks × 5 파이프라인 × 2 대조군 × 양측 | 다중성 |
| MC | C 진단 | C에서 제외, P·S에서 사용 | 비용 |
| 구조 탐색 | Stage B 사다리 9 rung | Stage P: r=4 양성 대조, 5개 모두 | 해석력 대비 비용 |
| 규모 탐침 | Stage S 필수 | Stage S 선택(CLP 중심) | 비용 |
| 예산 | A-prof 짧은 측정 | end-to-end 15분 이상 실측 + MD5 데이터 전 go/no-go | V5 실패 원인 |
| 원장 | 후보별 SQLite, 학습 hash 1,000만 행 SQLite | chunk binary + SHA-256 manifest, 학습 hash는 메모리 정렬 배열 | V5 학습 부대비용 19 ms/update |

이어받는 요소는 다음과 같다: source와 `H12_r` 정의, group ownership, 새 쌍 학습, batch 내 Shuffled, Success@100 원장 계약(invalid·duplicate도 기회 소비), CLP, artifact 감사, 무결성·재개 계약, 판정 순서.

---

## 3. 결정 질문과 원칙

> **최종 결정 질문.** 조건부 생성이 작동함을 확인한 5개 파이프라인 각각에서, 미학습 MD5 12-bit target(W3)에 대해 해시 조건을 학습한 모델이 source-prior Random과 Shuffled-condition 모델보다 Success@100이 높은가? 높다면 새 window(W4)에서 재현되는가? 높지 않다면 이득의 상한은 얼마인가? 표현·모델 계열·source에 따라 효과나 능력이 다른가?

1. **입증 책임은 가설 쪽에 있다.** `SUPPORTED`는 사전 규칙의 양성 판정, artifact 감사, W4 재현을 모두 통과해야만 나온다.
2. **V6 이후 같은 질문의 revision은 없다.** 허용하는 분기는 사전 등록된 것뿐이다: look 1–3의 공동 중단, 적격성 보완 1회, 예산 fallback 1회, 재현 1회. V5도 같은 원칙을 두었지만 예산 때문에 판정 없이 끝났다. V6는 사용자 요청에 따른 새 연구이며 V5 판정은 그대로 둔다.
3. **설계값은 MD5 조건 데이터를 만들기 전에 봉인한다.** Stage A는 synthetic만 쓴다. 봉인 뒤에 Stage C를 먼저 실행하며, P·S 결과는 C의 판정을 바꾸지 않는다.
4. **공정성.** 5개 파이프라인은 source별 prior, split, trial 목록, K, 학습 쌍 수, 판정 규칙이 같다. 다른 것은 표현, 모델, sampler뿐이다.
5. **항상 수치를 남긴다.** 어떤 종결에서도 파이프라인별 Δ_R, Δ_S의 추정값과 동시 구간을 보고한다.

---

## 4. 공통 설정

### 4.1 Source, 해시, window

- **Source.** Printable(P)은 ASCII 33–126, Random Bytes(R)는 0–255다. 두 source 모두 길이는 4–31 균등이고, 주어진 길이에서 bytes는 iid uniform이다. 모든 메시지는 MD5 한 block에 들어간다.
- **`H12_r^W(x)`.** V5 §4.1과 같다. MD5 압축함수의 처음 r step과 feed-forward를 계산하고 window W의 12 bits를 취한다. r=64가 실제 MD5다.
- **Window.** Digest를 big-endian 정수로 읽은 값에서 추출한다.

| 이름 | 정의 | V6 용도 |
|---|---|---|
| W1 | `>> 116` | Stage P의 r=4 위치. `H12_4`는 새 함수라 노출 대상이 아니다 |
| W2 | `& 0xFFF` | **사용 금지.** V5 Stage C가 test pool 전체를 평가했다 |
| W3 | `(>> 52) & 0xFFF` | 주 window(Stage C) |
| W4 | `(>> 84) & 0xFFF` | 재현 window(Stage R). 이 프로젝트에서 한 번도 쓰지 않았다 |

### 4.2 Split과 새 쌍 학습

- 각 과제(W3 r=64, W4 r=64, W1 r=4)의 4,096개 값을 test 1,024 / validation 256 / train 2,816 group으로 한 번씩 나눈다. **두 source와 5개 파이프라인은 같은 split을 쓴다.**
- **학습 쌍.** source prior에서 뽑은 x 중 H(x)가 train group인 것만 쓴다(수락률 68.75%). Run당 batch 256 × 40,000 updates = 1,024만 쌍이다. Stream namespace는 `(protocol, stage, source, window, rung, seed, update)`다.
  - 같은 source의 파이프라인은 같은 seed에서 **같은 메시지 stream**을 쓴다. 예를 들어 P-G-BGV, P-G-CGGE, P-DISC의 seed s는 같은 메시지를 같은 순서로 본다. 파이프라인 간 차이가 데이터 추출의 우연에서 오지 않게 하기 위해서다.
  - 같은 파이프라인·seed의 Main과 Shuffled는 메시지 stream, 초기 가중치, corruption 난수를 공유한다.
- **Shuffled.** 각 batch 안에서 condition을 무작위 permutation으로 재배정한다. Length head와 payload condition에 같은 donor를 쓰고, 내부 길이 조건(L/31)은 메시지의 실제 길이를 쓴다. 평가 때는 실제 target을 넣는다.

### 4.3 5개 파이프라인의 모델

| Pipeline | 표현 | 모델 | Params | 생성 | NFE |
|---|---|---|---:|---|---:|
| P-G-BGV | BGV 2×32×128 | G3-U | 910,638 | 길이 1회 + DDIM S_G step | S_G+1 |
| P-G-CGGE | CGGE 2×32×64 | G3-U | 513,326 | 같음 | S_G+1 |
| P-DISC | Token 32, vocab 97 | D1-S | 508,668 | 길이 1회 + 32 intervals | 33 |
| R-G-BGV | BGV 2×32×128 | G3-U | 910,638 | 길이 1회 + DDIM S_G step | S_G+1 |
| R-DISC | Token 32, vocab 259 | D1-S (출력 256 상태) | 1,252,572 | 길이 1회 + 32 intervals | 33 |

- **조건.** 12 bits(MSB first)는 length head `Linear(12,28)`에 들어간다. Denoiser는 12 bits와 L/31을 합친 13-d 조건과 시간 t를 받는다.
- **D1-S.** V5와 같다. Embedding 16, hidden 128, condition-output 잔차다. Adam 1e-3을 쓴다.
- **Token ID.** V5 규칙(PAD = 상태 수, EOS = +1, MASK = +2)을 두 source에 적용한다. P는 PAD 94, EOS 95, MASK 96이고, R은 PAD 256, EOS 257, MASK 258이다.
- **G3-U.** v3.1 G3 구조를 그대로 쓴다: U-Net width 32, 좌표 채널, 공간 condition-output, length head. 확산은 1,000 step, 선형 β 1e-4→0.02, x0-prediction이다. 매 step x̂0를 [−1,1]로 clip하고 DDIM(eta 0)으로 생성한다. Header, mask, padding은 뽑은 길이로 고정하고 payload glyph만 확산한다. 최적화는 Adam 1e-3, warmup 1,000, grad-norm clip 1.0이다.
- **Loss.** length CE + payload 오차의 평균이다. D1-S는 가려진 payload 위치의 CE, G3-U는 활성 payload glyph 픽셀 전체의 x0-MSE를 **균일하게** 평균한다. Prefix 가중 loss는 쓰지 않는다.
- **학습.** 5개 모두 새 쌍 40,000 × 256을 쓰고 최종 checkpoint만 쓴다. 4,000 updates마다 validation objective와 CLP를 진단으로만 기록한다.
- **Gaussian step 수 S_G ∈ {25, 50, 100}.** A-dev에서 파이프라인별로 선택한다(§5.4).
- **Sampler 공통.** Temperature 1, remask 없음. Argmax 치환, reranking, MD5 사용은 없다. 후보마다 명시적 RNG key를 두고 `vmap`으로 벡터화한다.

### 4.4 인코딩과 decoder 계약

Encoder는 원본 계획 §7.6–§7.8과 같다. Decoder는 V6에서 다음과 같이 등록한다.

| 표현 | V6 decoder | 이전 decoder(진단으로만 기록) |
|---|---|---|
| Token | EOS 앞 payload 토큰을 byte로 바꾼다. D1은 payload 상태만 sampling한다 | 같음 |
| BGV | 활성 슬롯마다 8개 4×4 block 평균을 [0,1]로 바꾸고, source alphabet의 byte bit 패턴(P 94개, R 256개) 중 제곱거리 최근접을 고른다. 동점이면 작은 byte | bit별 0.5 threshold. alphabet 밖이면 invalid |
| CGGE | 활성 슬롯마다 8×8 glyph와 94개 prototype의 MSE 최근접을 고른다. 동점이면 작은 code | MSE가 0.1을 넘으면 invalid |
| 길이 | G3 구조. 뽑은 길이로 header/mask 고정 | header 복호 |

이렇게 바꾸는 근거는 네 가지다.

1. D1은 출력 logits를 payload 상태로 제한해 source 밖 기호를 원천적으로 만들지 않는다. BGV·CGGE에 같은 제한을 두어야 모델 계열 간 비교가 공정하다.
2. CGGE는 원래도 최근접 prototype 분류였다. BGV의 bit threshold는 256개 byte prototype에 대한 최근접 분류와 같다. 따라서 R source에서는 V6 decoder와 이전 decoder가 동일하다. V6는 prototype 집합을 source alphabet으로 제한하고 거부 문턱을 없앨 뿐이다.
3. Decoder는 공개 codec 상수만 쓰고 target이나 길이 정보를 쓰지 않는다. Main과 Shuffled에 똑같이 적용되므로 Main−Shuffled 비교는 decoder 변경의 영향을 받지 않는다. Random은 prior에서 직접 뽑으므로 decoder를 거치지 않는다.
4. 결과를 보고 바꾼 것이 아니다. V6 MD5 자료를 만들기 전에 등록하며, 과거 결과를 재판정하지 않는다.

그 결과 5개 파이프라인의 valid는 구조적으로 100%가 된다. 대신 후보 품질을 다음 진단으로 보고한다: 이전 strict decoder 기준 valid 비율(Gaussian), 최근접 prototype 거리 분포, trial 내 duplicate, 학습 메시지 일치, 위치별 byte entropy.

### 4.5 방법과 후보 원장

| 방법 | 정의 | 쓰는 곳 |
|---|---|---|
| Main | 실제 condition으로 학습하고 요청 target으로 생성 | A, C, R, P, S |
| Shuffled | §4.2의 batch 내 permutation으로 학습하고 요청 target으로 생성 | C, R |
| Random | source prior에서 직접 sampling. 같은 source의 파이프라인이 seed별 stream을 공유 | C, R, P, S |
| MC | Main checkpoint에 다른 trial의 target(고정 derangement)을 넣어 생성 | P, S |

- 성공은 `valid(x) ∧ H(x) = 요청 target`이다. **Success@100**은 trial당 K=100 후보 중 성공이 하나 이상인지로 정한다. Invalid와 duplicate도 기회를 소비하며, 첫 성공 뒤에도 100개를 끝까지 생성한다.
- 후보 identity는 `(protocol, stage, pipeline, window, rung, method, seed, trial, attempt)`다.
- **원장.** Stream·블록 단위 binary chunk에 payload bytes, 길이, hit, duplicate, 학습 일치를 기록하고 trial 요약을 따로 둔다. Chunk마다 SHA-256을 manifest에 봉인한다. 학습 메시지 집합은 메모리의 정렬된 digest 배열로 조회한다.
- **검증.** 독립 verifier가 모든 후보를 재해시하고 trial 집계를 다시 계산한다. 결정적으로 고른 1%의 후보를 64개씩 묶어 RNG identity로 다시 생성하고, payload가 bitwise 일치하는지 확인한다.

---

## 5. Stage A — 기계 적격성 (MD5 조건 데이터 0)

### 5.1 A-impl (구현 게이트)

아래 항목이 모두 통과해야 이후 단계를 시작한다.

- **해시.** NumPy `H12_r^W`가 참조 구현과 일치한다. r은 V5 사다리 전체와 4, 64이고, window는 W1–W4, 두 source의 무작위 메시지 100,000개에서 검사한다. r=64는 hashlib과 같고, r=4의 W1은 bytes 0–3에만 의존한다.
- **Codec.** 5개 표현에서 모든 기호와 길이 4–31의 round-trip이 100%다. Prototype 집합의 쌍별 최소 거리는 0보다 크다. 동점 규칙은 fixture로 확인한다.
- **Sampler.** D1-S는 scalar 참조 sampler와 4,096 후보를 비교한다. G3-U는 S_G 셋 각각에서 비교한다.
  - D1-S: bitwise 동일하고, batch 1 / 64 / 1,024에서 출력이 같다(V5와 같음).
  - G3-U: batch 64 / 256 / 1,024 / 2,048 사이에서 bitwise 동일해야 한다. 실측에서 순전파는 batch 64 이상에서 bitwise 불변이었고 batch 1만 최대 1.8×10⁻⁷ 달랐다. 그래서 batch 1 scalar 참조는 후보 256개에서 decode 255개 이상 일치, 이미지 최대 절대차 1e-3 이하로 비교한다. 생성 batch는 64의 배수로 제한한다(세부 명세 §8.3).
- **이전 구현과의 등가.** V6 D1-S(P)는 같은 가중치와 key에서 `dhi_v5` D1-S와 같은 출력을 낸다. G3-U 순전파는 같은 가중치에서 v3.1 `mlx_models.ImageUNet`과 같다.
- **데이터 stream.** 결정적이고 재개해도 같은 batch를 낸다. Train group rejection이 정확하고, 같은 source 파이프라인이 같은 메시지를 받는다.
- **대조군.** Main/Shuffled가 난수를 공유하고, Shuffled permutation을 기록하며, MC derangement에 자기 자신이 없다.
- **통계·원장.** CLP 대칭성 fixture(5개 모두), 원장 → trial → counts 전수 fixture, 독립 재해시, 판정기 calibration과 planted-lift fixture(§13.3)를 통과한다.

### 5.2 A-prof

- **학습.** 5개 파이프라인과 D1-T-L의 update 시간을 데이터 생성과 원장 기록을 포함해 잰다.
- **생성.** batch {256, 1,024, 2,048}의 처리량과 메모리를 잰다. Gaussian은 S_G 셋 모두 잰다.
- **End-to-end stream.** 파이프라인마다 실제 평가 경로(생성 → decode → 해시 → 학습 집합 조회 → 원장 → 검증)를 10분 warm-up 뒤 15분 이상 연속 실행한다. Synthetic 조건과 폐기용 fixture window만 쓴다. 누적 처리량과 마지막 5분 처리량 중 느린 값을 예산에 쓴다.
- **구현 순서.** 위 측정은 두 번에 나눈다. A-prof-1은 A-Q 전에 학습 속도, burst 생성, batch 선택을 잰다. A-prof-2는 A-dev 뒤에 선택된 S_G로 end-to-end stream을 잰다(세부 명세 §12.2, §12.7). 지속 측정을 선택된 설정에만 하므로 A 시간이 줄어든다.
- **MD5 기준 처리량.** 같은 기계에서 prior sampling + MD5 속도를 잰다(§12).

### 5.3 A-Q (적격성)

- **과제는 synthetic_nibbles다.** P는 첫 3자가 조건의 대문자 hex 3자리다(V5와 같음). R은 첫 3바이트가 조건의 nibble 값 0x00–0x0F다. 나머지 길이와 payload는 source prior를 따른다.
- **Split.** Acceptance 512개 조건과 그 bitwise complement는 train/dev에서 제외한다. Dev split 256개 조건은 acceptance와 disjoint하다.
- **학습.** 5개 파이프라인 × seeds {0, 1, 2} = 15 runs를 새 쌍 40,000 × 256으로 학습하고 최종 checkpoint를 쓴다.

| 기준(seed별, 모두 충족; V5와 같음) | 값 |
|---|---|
| Acceptance 512 조건, 정상 joint | ≥ 461/512 |
| 반전 joint | ≥ 461/512 |
| Valid(정상·반전 각각) | 512/512 (구조적으로 보장되며 수치 오류 검사로 쓴다) |
| 반전 후보의 원래 조건 오성공 | ≤ 25/512 |
| CLP 양성 대조(§10), 4,096 pairs | one-sided z > 3.26 |

- **Q**는 세 seed가 모두 통과한 파이프라인의 집합이다.
- 생성 기준은 통과했는데 CLP만 실패하면 CLP 구현 결함으로 본다. 그 파이프라인의 CLP를 이후 판단에서 제외하고 보고한다(V5 규칙).

### 5.4 A-dev — Gaussian step 수 선택 (synthetic dev만)

- Gaussian 3개 파이프라인마다 seed 0의 A-Q checkpoint로 dev 256개 조건 × 정상/반전을 S_G ∈ {25, 50, 100}에서 생성한다. 추가로 dev 64개 조건 × 100 후보로 duplicate 비율을 잰다.
- **S_G는 다음 두 조건을 모두 만족하는 가장 작은 값이다.** (a) 정상·반전 joint가 각각 231/256 이상. (b) duplicate 비율이 S_G = 100일 때보다 0.5%p 넘게 높지 않음. 만족하는 값이 없으면 100이다.
- A-Q의 acceptance 평가는 선택된 S_G로 한다. Dev와 acceptance는 disjoint이므로 선택 과정이 적격성 판정에 섞이지 않는다.

### 5.5 사전 등록 보완 (1회)

- A-Q를 통과하지 못한 파이프라인은 같은 seed의 run을 update 80,000까지 이어서 학습하고 같은 기준으로 한 번 더 판정한다. Stream이 결정적이고 lr이 warmup 뒤 일정하므로, 처음부터 80,000 update를 학습한 것과 같다.
- 통과하면 그 파이프라인은 C와 P에서도 80,000 updates를 쓴다. 데이터량 차이는 비교표에 표시한다. 그래도 실패하면 C1 = `UNTESTABLE`이고 C와 P에서 제외한다.
- 보완한 Gaussian 파이프라인은 80k seed-0 checkpoint로 §5.4 규칙을 다시 적용해 S_G를 고른 뒤 재평가한다. 이어 학습의 봉인 규칙은 세부 명세 §7.7에 있다.
- 추가 비용은 A에서 BGV 약 4.4시간, CGGE 약 2.6시간, Discrete 약 0.7시간이다. C 학습에서는 BGV 약 8.8시간, CGGE 약 5.1시간이 추가된다.

### 5.6 예산 봉인, fallback, 동결

1. A-prof와 S_G로 **Stage C를 look 2까지 완료하는 예측 시간**을 계산한다. 예측 × 1.5 ≤ C cap이면 기본 설계로 진행한다.
2. 넘으면 한 번만 fallback을 적용한다. 블록을 8,192에서 6,144로 줄인다(look 6,144 / 12,288 / 18,432).
3. 그래도 넘으면 **MD5 조건 데이터를 하나도 만들기 전에 멈추고** 사람에게 cap 증액 여부를 묻는다. 이 시점에는 MD5 결과가 없으므로 결정이 결과에 영향을 받을 수 없다. 결정 내용은 봉인 기록에 남긴다.
4. 필수 경로 합계가 cap을 넘으면 P의 trials를 4,096에서 2,048로 줄이고 S를 생략한다.
5. **동결.** `protocol.frozen.json`(모든 설정, S_G, 블록 크기, Q), `window.json`(W3/W4와 감사 hash), `budget-plan.json`, code·환경 manifest를 봉인한다.

**C와 R의 K, seeds, updates, δ는 fallback으로 줄이지 않는다.**

---

## 6. Stage C — 5개 파이프라인 MD5-12 시험 (주 판정)

### 6.1 설계

| 항목 | 고정값 |
|---|---|
| 해시 | `H12_64^{W3}` |
| Test pool | 인증된 W3 test group 1,024개 |
| 파이프라인 | Q(최대 5개) |
| Learned runs | Q × {Main, Shuffled} × seeds {0, 1, 2} = 최대 30 runs |
| Trial 목록 | 30개 checkpoint를 봉인한 **뒤** pool에서 iid 복원추출 24,576개를 한 번 뽑는다. 모든 파이프라인·방법·seed가 같은 목록을 앞에서부터 쓴다. 재추첨하지 않는다 |
| 블록과 look | 8,192-trial 블록 3개. Look j는 앞 8,192·j trials로 분석한다 |
| 블록당 후보 | 파이프라인별 Main·Shuffled × 3 seeds × 8,192 × 100 = 4,915,200. Random은 source별 3 seeds × 8,192 × 100 |
| 주 지표 | Success@100, trial 단위 paired |
| 보조 | CLP_64(Main seed별 65,536 pairs), @1/@10, duplicate, 학습 일치, strict-decoder valid, prototype 거리, 위치별 entropy, 시간 |

### 6.2 통계

파이프라인 i, 대조군 c ∈ {Random, Shuffled}, trial t, seed s에 대해 `d_t = mean_s (M_{i,s,t} − C_{s,t})`로 둔다. Look j의 trial 수는 `T_j = 8,192·j`이고, `Δ̂ = mean_{t≤T_j} d_t`, `SE = sd(d)/√T_j`다.

- **오류 배분.** α = 0.05를 5 파이프라인 × 2 대조군 × 양측의 20개 꼬리로 나눈다(꼬리당 0.0025). 이를 다시 look 1 / 2 / 3에 0.1 / 0.4 / 0.5 비율로 나눈다(Bonferroni). 그 결과 `z = 3.4808 / 3.0902 / 3.0233`이다. `[L, U] = Δ̂ ∓ z·SE`.
- **보장.** 모든 look·파이프라인·대조의 구간이 동시에 참값을 포함할 확률은 95% 이상이다. 따라서 공동 중단 규칙이 어느 look에서 멈추든 판정이 유효하다.
- **최소 관심 효과** δ = 0.005(0.5%p)다. 후보당 성공확률로는 약 1.21배다.
- **효과 없음일 때의 정밀도.** 반폭은 look별로 ±0.482 / ±0.302 / ±0.242%p다. 대조 하나가 δ를 배제할 확률은 55.2% / 97.8% / 99.94%다.
- **추론 범위.** 봉인된 test pool, checkpoint, seed에 조건부이고, target 추출과 생성 난수에 대한 추론이다. Seed별 추정값을 모두 보고한다.

### 6.3 판정

각 look에서 파이프라인별로 V5 §6.3의 순서를 적용한다.

| 순서 | 조건 | 분류 |
|---|---|---|
| 1 | `L_R > 0` **그리고** `L_S > 0` | `POSITIVE` |
| 2 | `U_R < δ` 그리고 `U_S < δ` | `REJECTED_BOUNDED` |
| 3 | `U_S < δ` | `REJECTED_NO_CONDITION_GAIN` |
| 4 | `U_R < δ` | `REJECTED_NO_RANDOM_ADVANTAGE` |
| 5 | 그 외 | 미결 |

**공동 중단 규칙**은 다음과 같다.

- Look 1이나 2에서 Q의 **모든** 파이프라인이 1 또는 2에 해당하면 Stage C를 멈추고, 그 look의 분류를 최종 판정으로 한다.
- 아니면 모든 파이프라인이 다음 블록을 생성한다. 이미 결정된 파이프라인도 계속한다. 비교에 쓰는 trial 수를 파이프라인 사이에 같게 유지하기 위해서다.
- Look 3에서는 1–5의 전체 순서로 분류하고, 5는 `NOT_ESTABLISHED_UNRESOLVED`로 한다.
- Cap 때문에 다음 look을 완료할 수 없으면 마지막으로 완료된 look에서 1–5의 전체 순서로 분류한다. 5는 `NOT_ESTABLISHED_BY_BUDGET`으로 하고 구간을 함께 보고한다. 첫 look도 완료하지 못하면 Q 전체가 `NOT_ESTABLISHED_BY_BUDGET`이다.
- `POSITIVE`인 파이프라인은 §6.5 감사를 거쳐 Stage R로 간다.

### 6.4 설계 검산

가상 결과로 시나리오마다 2,000회 모의했다. 양성 파이프라인은 W4 재현까지 포함했다.

| 가정(seed별 Success@100) | 주요 결과 | 중단 look 1 / 2 / 3 | 평균 trials |
|---|---|---|---:|
| 5개 모두 p0 | `FINAL_REJECTED` 100%. 파이프라인별 `REJECTED_BOUNDED` 99.75–99.95%. 거짓 `POSITIVE` 0/2,000 | 1.3 / 81.7 / 17.0% | 17,670 |
| 현실적 무효과(Gaussian Main·Shuffled −0.1%p, Discrete −0.01%p; duplicate 손실) | `FINAL_REJECTED` 99.95% | 2.7 / 85.6 / 11.8% | 17,134 |
| P-DISC만 Main +δ | P-DISC `SUPPORTED` 99.55%, 나머지 `REJECTED` | 1.4 / 79.3 / 19.4% | 17,859 |
| P-G-BGV만 Main +δ | P-G-BGV `SUPPORTED` 99.55% | 0.8 / 79.7 / 19.6% | 17,924 |
| P-DISC만 Main +δ/2 | `SUPPORTED` 19.6%, `NOT_REPLICATED` 14.9%, `REJECTED_*` 65.3%. δ 아래 효과이므로 섞인다 | 0.05 / 23.6 / 76.4% | 22,634 |
| R-G-BGV Main = Shuffled = p0 + δ (prior 모델링 이득) | R-G-BGV `REJECTED_NO_CONDITION_GAIN` 99.6% | 0 / 0.15 / 99.85% | 24,564 |
| P-G-CGGE의 seed 하나만 +3δ(평균 +δ) | `SUPPORTED` 99.25% | 0.6 / 80.6 / 18.9% | 17,883 |
| 5개 모두 Main +δ | 파이프라인별 `SUPPORTED` 97.3–98.2%, `FINAL_SUPPORTED` 100% | 0.9 / 72.0 / 27.1% | 18,530 |
| 무효과, look 2 뒤 예산 종료 | `FINAL_REJECTED` 97.95%. 파이프라인별 `NOT_ESTABLISHED_BY_BUDGET` 0.3–0.55% | — | 16,249 |
| 무효과, look 1 뒤 예산 종료 | `FINAL_REJECTED` 21.9%, `FINAL_REJECTED_WITH_EXCEPTIONS` 77.7% | — | 8,192 |
| Fallback 블록 6,144, 무효과 | `FINAL_REJECTED` 99.3% | 0.3 / 46.2 / 53.5% | 15,557 |
| Fallback 블록 6,144, P-DISC +δ | P-DISC `SUPPORTED` 96.45% | 0 / 41.3 / 58.8% | 15,898 |

### 6.5 POSITIVE 후 artifact 감사 (필수)

V5 §6.4와 같다.

1. 모든 성공 payload를 독립 verifier로 재해시한다.
2. 성공 후보 중 학습 메시지와 같은 것은 0이어야 한다.
3. 성공이 특정 target에 몰렸는지 확인하고, 상위 1% target의 성공 비중을 보고한다.
4. CLP_64의 방향을 보고한다. CLP는 무신호인데 hit만 양성이면 `hit-only`로 표시한다.

감사 실패는 POSITIVE를 무효화한다. 원인을 고친 뒤 해당 stream을 같은 난수 identity로 한 번 재생성한다. 고칠 수 없으면 `NOT_ESTABLISHED_INTEGRITY`다.

### 6.6 δ와 trial 수를 정한 이유

- V5의 δ 0.25%p를 같은 z로 5개 파이프라인에 적용하면, 효과 없음에서 대조별 배제 확률 99%를 얻기 위해 파이프라인당 약 72,000 trials × 3 seeds가 필요하다. Gaussian 25 step 기준 생성만 약 97시간이다.
- δ 0.5%p는 후보당 1.21배 lift다. 가장 빠른 P-DISC도 질의당 계산 우위를 얻으려면 후보당 약 104배 lift가 필요하다(§12). 따라서 δ는 실용적 의미가 생기는 문턱보다 훨씬 작은 효과까지 검출 대상으로 삼는다.
- 효과가 없으면 실제 상한은 δ보다 좁게 나온다. 반폭은 look 2에서 ±0.30%p, look 3에서 ±0.24%p다.

---

## 7. Stage R — 양성 재현 (POSITIVE이고 감사를 통과한 파이프라인만)

- **설계.** Window는 W4이고 새 split을 쓴다. 파이프라인 설정(S_G, updates)은 C와 같다. Main·Shuffled × seeds {0, 1, 2}(새 namespace), trials 16,384 × K=100, Random을 생성한다.
- **판정.** `L_R > 0` 그리고 `L_S > 0`이다. 각각 one-sided이고 `z = Φ⁻¹(1 − 0.025/m)`이다. m은 R에 들어간 파이프라인 수다. Look은 1회이고 확장은 없다.
- 통과하면 C3 = `SUPPORTED`, 실패하면 `NOT_ESTABLISHED_NOT_REPLICATED`다.
- 비용은 최악의 경우 BGV 파이프라인 1개에 약 17.3시간이다(cap 36시간).

---

## 8. Stage P — 해시 구조 양성 대조 (r=4, C2)

**목적.** 두 해석을 구분한다. MD5-64에서 효과가 없을 때, "기계가 해시 구조를 전혀 이용하지 못해서"인가, "구조가 있으면 이용하지만 MD5의 mixing이 그것을 지워서"인가? 결과는 파이프라인별 구조 이용 능력의 비교 지표가 된다.

| 항목 | 값 |
|---|---|
| 해시 | `H12_4^{W1}`. Payload bytes 0–3만의 함수이며 새 split을 쓴다 |
| Runs | Q × Main seed 0 = 최대 5 runs. 보완한 파이프라인은 80,000 updates |
| 평가 | Trials 4,096 × K=100으로 Main, MC, Random(source별). CLP_4 16,384 pairs |
| GEN_4 | Main−Random **그리고** Main−MC의 one-sided z > 2.878 (α = 0.01을 5개 파이프라인에 Bonferroni, intersection-union) |
| INFO_4 | CLP_4 one-sided z > 2.878 |
| 검정력 | 90% 검정력으로 검출 가능한 Success@100 차이 약 1.41%p |
| 비용 | 약 6.1시간 |

- Test group의 y는 학습에서 본 적이 없다. 따라서 성공하려면 bytes 0–3과 출력 bit 사이의 연산 구조를 일반화해야 하며, 암기로는 통과할 수 없다.
- GEN_4가 성립하고 C3가 `REJECTED`이면 "구조가 있으면 이용하지만 full MD5에서는 이득이 없다"로 기록한다. 가장 강한 형태의 음성 결론이다.
- GEN_4가 성립하지 않으면 그 파이프라인의 음성을 "r=4 구조도 이용하지 못한 기계의 음성"으로 표시한다. 어느 경우든 C3 판정은 바뀌지 않는다.

---

## 9. Stage S — 규모 탐침 (선택, C3 불변)

**질문.** "더 큰 모델이면?"이라는 반론에 정보 측정으로 답한다.

| 항목 | 값 |
|---|---|
| 모델 | P-DISC의 D1-T-L(6.42M params, D1-S의 약 13배) |
| 학습 | W3 train group, 160,000 updates × 256(새 쌍 4,096만 개). 약 5.4시간 |
| 측정 | CLP_64 65,536 pairs. GEN(Main·MC·Random 4,096 trials × 100)은 기술통계 |
| 판정 | CLP_64 one-sided z > 3.29이면 `SCALE_SIGNAL`. "독립 재현이 필요한 이상 신호"로 기록한다 |

결과는 V5 D1-T의 W2 부분 결과(§1.2)와 함께 보고한다. C3 판정은 hit로만 내린다.

---

## 10. CLP — Conditional Likelihood Probe

V5 §10의 정의를 5개 파이프라인으로 넓힌다. Test group의 held-out 쌍 `(x_i, y_i)`, `(x_j, y_j)`에 대해 `D = s(x_i,y_i) + s(x_j,y_j) − s(x_i,y_j) − s(x_j,y_i)`로 정의한다.

- **점수.**
  - D1-S: `s = −(length CE + 가려진 payload CE)`를 고정 corruption draw 8개로 평균한다(V5와 같음).
  - G3-U: `s = −(length CE + payload x0-MSE)`를 고정 `(t, ε)` draw 8개로 평균한다. t는 U{0,…,999}, ε는 N(0, I)에서 뽑으며, 학습 loss의 t 분포와 같다. 한 쌍의 네 점수는 같은 draw를 쓴다.
- **귀무.** x와 y가 독립이면 D는 0에 대해 대칭이다. 대칭성 fixture로 검사한다.
- **쓰는 곳.** A-Q(z > 3.26), P(INFO_4), C 진단(INFO_64: 3 seeds 합산, one-sided z > 2.878이면 "이상 신호"), S다. **C3 판정에는 쓰지 않는다.**
- **비용.** 순전파만 쓰므로 작다. BGV에서도 seed당 약 2분이다.

---

## 11. 파이프라인 비교 (C5, 목표 1)

### 11.1 비교표 (파이프라인별)

| 영역 | 지표 |
|---|---|
| 규모·비용 | Params, S_G, NFE, 실측 후보/s, 학습 시간, ρ(§12) |
| 기계 능력(A) | A-Q 정상/반전 joint(3 seeds), CLP_syn z, 보완 여부 |
| 구조 이용(P) | r=4 Success@100(Main / MC / Random), GEN_4, CLP_4 z |
| MD5-64 효과(C) | Δ_R, Δ_S와 동시 구간, 판정, 중단 look, seed별 값 |
| 정보 신호 | CLP_64 z |
| 후보 품질 | Trial 내 duplicate, 학습 일치율, 위치별 entropy, strict-decoder valid, prototype 거리 |

### 11.2 사전 지정 대비

| 질문 | 대비 |
|---|---|
| 표현(Gaussian, Printable) | P-G-BGV vs P-G-CGGE |
| 모델 계열(Printable) | P-G-BGV vs P-DISC, P-G-CGGE vs P-DISC |
| 모델 계열(Random Bytes) | R-G-BGV vs R-DISC |
| Source | P-G-BGV vs R-G-BGV, P-DISC vs R-DISC |

- **추정.** 공통 trial에서 `d_t^i − d_t^j`를 Δ_S 대비와 Δ_R 대비로 각각 계산한다. 12개 구간을 Bonferroni로 묶어 `z = 2.8653`(양측 0.05/24)을 쓴다.
- **반폭.**
  - T = 16,384: Δ_S ±0.397%p, 같은 source의 Δ_R ±0.280%p(Random 공유로 상쇄), 다른 source의 Δ_R ±0.397%p.
  - T = 24,576: 각각 ±0.324 / ±0.229 / ±0.324%p.
- **해석.** 구간이 0을 제외하면 "차이 확인", 구간이 ±δ 안에 있으면 "δ 이상의 차이 배제", 그 외는 "미확정"이다. 대비 결과는 C3 판정을 바꾸지 않는다.
- 공동 중단 규칙 때문에 모든 파이프라인의 trial 수는 같다. 다만 중단 시점이 주 판정 자료에 의존하므로 대비 구간은 보조 분석으로 취급한다.
- 능력 지표(A-Q, r=4, CLP)는 기술통계와 구간으로 비교하며, 가설 검정으로 쓰지 않는다.

### 11.3 예상되는 비교 결과의 형태

- 5개 모두 C3가 `REJECTED`이고 대비가 "δ 이상 차이 배제"나 "미확정"이면 다음 결론을 쓴다: "표현, 모델 계열, source와 무관하게 해시 조건 이득이 없다."
- 파이프라인 간 차이는 주로 능력 지표(A, P), 후보 품질, 비용에서 나타날 것으로 예상한다. **"구조 이용 능력은 파이프라인마다 다르지만 MD5-64 효과는 모두 없다"**가 이 연구가 얻을 수 있는 가장 강한 음성 결론이다.

---

## 12. C4 — 계산 우위 판정

### 12.1 분석식 (실험 전에 확정)

학습 1 run이 쓰는 MD5 호출은 약 1,490만 회다(1,024만 쌍 / 수락률 0.6875). 4,096개 target 전부의 역상 lookup table을 무작위 탐색으로 만드는 기대 비용은 36,434회다. **학습 1 run은 완전한 역상 table을 약 409번 만들 수 있는 MD5 호출을 소비한다.** 5개 파이프라인의 학습량이 같으므로 q=12에서 학습비를 포함한 계산 우위는 5개 모두 성립하지 않는다.

### 12.2 질의당 계산 우위 (학습비를 제외한 가장 관대한 비교)

`ρ = thr_MD5 / thr_model`로 둔다. 모델에 유리하도록 모델의 순수 생성 속도(proxy)와 CPU 단일 core MD5 속도(2,104,757/s)를 쓴다. 질의당 우위가 성립하려면 Main의 후보당 성공확률이 `ρ · 2⁻¹²` 이상이어야 한다.

| Pipeline | Proxy 후보/s (S_G = 25) | ρ | 손익분기 후보당 성공확률 |
|---|---:|---:|---:|
| P-G-BGV, R-G-BGV | 587 | 3,584 | 0.875 (S_G = 100이면 3.40 → **불가능**) |
| P-G-CGGE | 1,168 | 1,802 | 0.440 (S_G = 100이면 1.71 → **불가능**) |
| P-DISC | 20,344 | 104 | 0.0253 |
| R-DISC | 10,172 (가정) | 207 | 0.0505 |

Random의 후보당 성공확률은 0.0244%다. 판정 규칙은 V5 §11.2와 같다.

- C3 ≠ `SUPPORTED`이면 `NO_ADVANTAGE`다.
- C3 = `SUPPORTED`이면 Main 후보당 성공률의 one-sided 97.5% 하한이 `ρ · p̂_Random`을 넘을 때만 `PER_QUERY_ADVANTAGE`다. ρ는 A-prof 실측값을 쓴다.
- 손익분기 값이 1 이상이면 "산술적으로 불가능"으로 기록한다.

---

## 13. 노출·무결성·calibration

### 13.1 노출 감사 갱신과 window 인증

1. V5 inventory(보존된 로컬 코드와 과거 실행 5,267개 파일)에 V5 실행 root(`v5-study`, `v5-study-certified`)와 F1 설계 산출물을 추가해 v6 inventory를 만든다.
2. **W2.** V5 Stage C가 test pool 1,024 group 전체를 평가했으므로 노출로 기록하고 쓰지 않는다.
3. **W3.** V5가 W3 값을 계산한 곳은 두 군데뿐이다. A-impl의 해시 일치 검사(무작위 메시지)와 data-stream 검사(train 메시지 256개)이며, 둘 다 모델 입력이나 평가가 없다. 이를 기록하고 미노출로 인증한다. 과거 full/raw digest를 조건으로 쓴 사례는 V5 감사에서 0건이었다.
4. **W4.** 어떤 코드도 쓰지 않았음을 코드 감사로 확인한다.
5. 인증은 두 source 모두에 적용한다. Stage P의 `H12_4`(r<64)는 새 함수이므로 노출 대상이 아니다.
6. 봉인 전까지 W3·W4 조건 데이터를 만들지 않는다.

### 13.2 무결성 검사

§5.1 전부에 더해 V5 §12.2(train group rejection, 학습 메시지 hash 집합, Shuffled permutation, MC derangement, CLP 대칭성, 원장 전수 fixture)를 적용한다. V6에서 추가하는 항목은 다음과 같다.

- Prototype decoder fixture
- Gaussian 재개 결정성
- 같은 source 파이프라인의 공통 메시지 stream 확인
- 무작위 1% 후보의 재생성 감사

### 13.3 Production calibration

- 실제 판정 코드(공동 중단을 포함한 §6.3, §7)에 가상 trial 결과를 넣어 §6.4의 시나리오를 각 2,000회 이상 실행한다.
- **통과 조건.**
  - 무효과에서 어느 파이프라인이든 `POSITIVE`가 나오는 비율의 one-sided 95% CP 상한 ≤ 0.025
  - 무효과에서 5개 모두 `REJECTED`인 비율의 하한 ≥ 0.95
  - 한 파이프라인 +δ에서 그 파이프라인이 `POSITIVE`인 비율의 하한 ≥ 0.95
- 설계 검산 스크립트와 생산 판정 코드가 같은 입력에서 같은 판정을 내는지 fixture로 확인한다. 두 곳의 판정 로직이 서로 달라지는 것을 막기 위해서다.
- **Planted-lift fixture.** W3 **validation** group의 사전 계산 역상을 Random 후보에 섞는다. 실제 원장 경로에서 +0과 +δ가 각각 `REJECTED`와 `POSITIVE`로 판정되는지 확인한다. Test pool은 쓰지 않는다.

---

## 14. 실행 순서, 자원, fallback

**순서.** A-impl → A-prof-1 → A-Q 학습 → A-dev(S_G) → A-Q 평가 → (보완) → A-prof-2 → 예산 계획 → 노출 감사·**봉인** → C 학습 30 runs → trial 봉인 → **C 블록·look** → (감사 → R) → P → (S) → 최종 보고.

주 판정(C)을 먼저 확보한다. 보조 단계(P, S)가 자원을 먼저 쓰지 않게 한다.

**예상 시간** (S_G = 25, 설계 검산값):

| 단계 | Runs | 학습 | 생성 | 합계 (무효과 기대 / 최악) | Cap |
|---|---:|---:|---:|---:|---:|
| A (impl, prof, Q, dev) | 15 | 12.8 h | 무시 가능 | 17.3 h | 24 h (+보완 12 h) |
| C | 30 | 25.7 h | 블록당 11.0 h | 49.7 h / 59.0 h | 80 h |
| R (POSITIVE 시) | 파이프라인당 6 | — | — | 최악 17.3 h | 36 h |
| P | 5 | 4.3 h | 1.8 h | 6.1 h | 10 h |
| S (선택) | 1 | 5.4 h | 1.1 h | 6.5 h | 10 h |
| **필수 경로 (A, C, P)** | | | | **73.2 h / 82.5 h** | **114 h** |

- **Run당 학습 시간.** BGV 1.47 h, CGGE 0.85 h, P-DISC 0.23 h, R-DISC 0.25 h다.
- **C 블록 1개(8,192 trials).** P-G-BGV 4.26 h, R-G-BGV 4.26 h, P-G-CGGE 2.14 h, P-DISC 0.12 h, R-DISC 0.25 h다.
- **S_G별 필수 경로(기대 / 최악).** 25 step은 73 / 83 h, 50 step은 97 / 115 h, 100 step은 145 / 180 h다. 50 step 이상이면 §5.6의 fallback이나 사람 결정이 필요할 가능성이 크다.
- **기타 한도.** 저장량, RSS, GPU는 각각 64 GiB다. 최소 디스크 여유는 20 GiB다. C 원장은 최대 약 8,850만 행(약 3.5 GB)이다.
- **동시 실행.** GPU 작업은 한 번에 하나만 돌린다. Random 생성과 검증은 CPU에서 병행할 수 있다.
- **실행 중 cap 초과.** C는 §6.3 규칙을 따른다. P·S가 미완이면 "부분 측정"으로 보고하고 최종 결론은 그대로 낸다. 고립된 실행 오류는 같은 난수 identity로 한 번 재실행할 수 있다.
- **운영 화면.** 진행률, 자원, 무결성 오류만 표시한다. Look 분석은 자동이며 화면에는 "계속" 또는 "중단"만 나온다.

**위험과 대응**

| 위험 | 대응 |
|---|---|
| Gaussian이 25 step에서 적격성 미달 → 비용 2–4배 | §5.4 선택 규칙, §5.6 fallback, MD5 데이터 전 사람 결정 |
| G3-U 균일 loss에서 prefix 정확도 저하 | §5.5 보완 1회(80,000 updates) |
| Batch에 따라 conv 부동소수 결과가 다름 | 봉인된 batch 구성으로만 생성·재개, decode 일치율 게이트 |
| R-DISC 처리량 미측정 | A-prof 실측으로 대체 |
| 공동 중단 규칙의 추가 비용 | 파이프라인별 중단보다 무효과 기대 약 7시간 더 든다. 비교 정밀도를 위한 의도된 비용 |

---

## 15. 최종 결론 규칙

### 15.1 파이프라인별 C3 판정 값

`SUPPORTED`, `REJECTED_BOUNDED`, `REJECTED_NO_CONDITION_GAIN`, `REJECTED_NO_RANDOM_ADVANTAGE`, `NOT_ESTABLISHED_UNRESOLVED`, `NOT_ESTABLISHED_NOT_REPLICATED`, `NOT_ESTABLISHED_INTEGRITY`, `NOT_ESTABLISHED_BY_BUDGET`, `UNTESTABLE`(C1 실패).

### 15.2 종합 판정 (연구 종료 판정)

| 종합 판정 | 조건 | 최종 보고서의 결론 문장 (사전 작성) |
|---|---|---|
| **`FINAL_SUPPORTED`** | `SUPPORTED`인 파이프라인이 하나 이상 | "고정 source, MD5 12-bit window W3, 등록된 학습량에서 {파이프라인}의 해시 조건 모델은 Random과 Shuffled 대비 Success@100 이득을 보였고({구간}), 이 이득은 W4에서 재현되었다. 다른 파이프라인의 판정은 {…}다. 계산 우위는 {C4}다. Full MD5 역상, 보안 붕괴, 학습비 포함 계산 우위는 주장하지 않는다." 가설 지지로 종료한다. 후속은 **별도의 새 연구**(Stage II 계산 효율)로만 가능하다 |
| **`FINAL_REJECTED`** | 5개 모두 `REJECTED_*` | "Gaussian BGV, Gaussian CGGE, Discrete token 표현과 Printable, Random Bytes source로 구성한 5개 파이프라인 모두에서, 해시 조건 diffusion 생성의 Success@100 이득은 사전 최소 관심 효과 0.5%p 미만으로 배제되었다(파이프라인별 상한 {U_R}, {U_S}). 같은 기계들은 synthetic 조건을 {A-Q} 정확도로 사용했고, step-reduced MD5 r=4에서는 {GEN_4 파이프라인}이 구조를 이용했다. 파이프라인 간 효과 차이는 {대비 요약}이다. 계산 우위는 없다." 가설 기각으로 종료한다 |
| **`FINAL_REJECTED_WITH_EXCEPTIONS`** | `SUPPORTED` 없음. `REJECTED_*`가 하나 이상이고 나머지는 `UNTESTABLE` 또는 `NOT_ESTABLISHED_*` | "{n}개 파이프라인에서 0.5%p 이상의 이득을 배제했다. {파이프라인}은 {사유}로 판정하지 못했다(구간 {…}). 이득이 지지된 파이프라인은 없다." 적용 범위를 제한한 가설 기각으로 종료한다 |
| **`FINAL_NOT_ESTABLISHED`** | `SUPPORTED`도 `REJECTED_*`도 없음 | "사전 규칙으로 판정하지 못했다. 사유는 {사유 코드}이고, 측정된 경우 이득의 상한은 {구간}이다." 추가 증액 없이 종료한다 |

어느 판정이든 **V6 이후 같은 질문의 revision은 없다.** CLP 이상 신호, `SCALE_SIGNAL`, r=4에서 관측된 구조 이용은 원래 연구를 이어가는 근거가 아니다. 새 연구를 제안할 근거일 뿐이다.

### 15.3 최종 보고서 필수 내용

1. 최종 결론 카드(§0)와 종합 판정
2. 버전별 결산, V5 부분 결과, 처리량 측정(§1)
3. C1: 파이프라인별 A-Q 결과, S_G 선택, 보완 여부, Q
4. C3: 파이프라인별 모든 대조의 추정값·동시 구간·seed별 값, 중단 look, artifact 감사, R 결과(해당 시), CLP_64
5. C2: r=4의 GEN_4·INFO_4와 효과 크기
6. C5: §11.1 비교표와 §11.2 대비 구간
7. C4: §12.1 산술, 실측 ρ, 판정
8. S 결과(해당 시)
9. 적용 범위와 일반화(§15.4)

### 15.4 적용 범위와 일반화

- **실측 범위.** Printable과 Random Bytes source, MD5 12-bit window W3(재현 시 W4 추가), 5개 파이프라인(G3-U BGV·CGGE, D1-S), 등록된 decoder·sampler·학습량(1,024만 쌍), K=100, r=4 양성 대조.
- **이론적 외삽(측정 아님).** §1.5의 random-function 논증은 sampler, 표현, source, q에 의존하지 않는다. 따라서 q=8/16과 다른 구조에서도 같은 결론을 기대한다. 원본 계획의 15개 설정 중 V6가 측정하는 것은 q=12의 5개다.
- **주장하지 않는 것.** Full MD5 역상, 임의 target MD5 역상, SHA-256, 보안 붕괴, 최신 암호분석 공격과의 비교, 모든 diffusion의 불가능성.

---

## 16. 구현 범위

새 package `src/dhi_v6/`를 만든다. `src/dhi_v5/`와 `src/diffusion_hash_inv/`는 수정하지 않고, 등가 테스트에서 읽기 전용으로만 import한다. 모듈별 알고리즘, 난수 namespace, 원장 형식, 게이트 기준, CLI, 테스트 목록은 [V6_IMPLEMENTATION_SPEC.md](V6_IMPLEMENTATION_SPEC.md)에 있고, 등록값은 [examples/v6-protocol.json](examples/v6-protocol.json)이다.

1. **`protocol.py`**: 고정 설정, 등록 정보, 수정된 protocol JSON 거부.
2. **`data.py`**: 두 source, synthetic 과제(P·R), split, 공통 메시지 stream, NumPy `H12_r^W`(W1–W4), token codec(P·R).
3. **`codecs.py`**: BGV·CGGE encoder, V6 prototype decoder, 진단용 strict decoder.
4. **`models.py`**: D1-S(P·R), G3-U(BGV·CGGE), D1-T-L. 명시적 key의 `vmap` sampler, scalar 참조 sampler, CLP scorer.
5. **`runtime.py`**: 예산·cap, checkpoint·재개, chunk 원장, 독립 verifier, 학습 집합 조회, 재생성 감사.
6. **`statistics.py`**: 공동 중단 Stage C 판정기, R, P 검정, 대비, C4, 종합 판정. Production calibration과 planted-lift fixture 포함.
7. **`study.py`**: CLI `plan | audit | run --stage {A,C,R,P,S} | report`. 단계 봉인, 재개, cap, fallback, window 결정을 기록한다.
8. **`tests/test_study_v6.py`**: §5.1과 §13.2의 모든 게이트.

실행 형태는 V5와 같다: `PYTHONPATH=src .venv/bin/python -m dhi_v6.study ...`.

---

## 17. 산출물과 재현

1. `protocol.frozen.json`, `window.json`, `budget-plan.json`(예산 계획과 fallback 결정), `budget.json`(실행 시간 계상), code·환경 manifest, v6 노출 감사 기록
2. A-impl 테스트 결과, A-prof, A-dev(S_G), A-Q 결과와 Q, 보완 기록
3. Stage C(및 R): 학습 기록, checkpoint 봉인, trial 목록, look별 분석, 모든 원장과 manifest, verifier 결과, 판정, artifact 감사
4. Stage P·S: 셀별 GEN·INFO, CLP 결과
5. `decision.json`: 실행 상태, 파이프라인별 C1–C4, C5 비교, 종합 판정, 사유 코드
6. 한국어 최종 보고서 `FINAL_REPORT_KO.md`(§15.3 구성)

**설계 검산 재현** (가상 계산과 기록된 측정값만 쓴다. 실제 실험 명령이 아니며 약 2분 걸린다):

```sh
.venv/bin/python scripts/validate_research_plan_v6.py
```

출력은 `local_experiment_archive/analyses/v6-design-20260929/design_calculation.json`이다. Archive가 없는 저장소에서도 실행되며, 이때는 기록된 측정값을 쓴다.

**처리량 proxy 재현** (로컬 전용. Metal 접근 필요. 임의 가중치만 쓰며 연구 데이터와 MD5를 사용하지 않는다):

```sh
.venv/bin/python local_experiment_archive/analyses/v6-design-20260929/throughput.py
```
