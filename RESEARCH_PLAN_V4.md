# v4 실험 계획 — 실제 MD5 조건부 생성 연구의 지속 여부 결정

**Protocol:** `dhi-v4-decision-20260928` · **작성일:** 2026-09-28 KST  
**상태:** `IMPLEMENTED_NOT_QUALIFIED` — v4 전용 실행기와 [단일 CLI·실행 명세](V4_CLI.md)를 구현했다. 축소 MLX/실제 MD5 E0 검증과 생산 통계 calibration은 본실험 결과가 아니다. 정식 V1 적격성·노출 감사·해당 study의 실측 자원 봉인·새 MD5 holdout 평가는 아직 수행하지 않았다.

**권고하는 실험은 Printable source의 P-DISC 한 경로에서, 실제 MD5-12 조건을 사용한 모델을 Random 및 Shuffled와 비교하는 고정 규모 실험이다. 세 seed 모두에서 Success@100이 두 대조군보다 최소 1%p 높다는 근거를 얻으면 연구를 확대한다. 유용한 이득을 배제하면 현재 접근의 확대를 중단한다. 실행 실패와 통계적 불확정은 별도로 보고한다.**

이 문서는 [원본 연구 계획](RESEARCH_PLAN.md)의 해시 조건부 후보 우위라는 질문을 유지하면서, [v3.1](RESEARCH_PLAN_V3_1.md)의 다섯 pipeline 전체 완주와 구분되는 **새 의사결정 연구**를 제안한다. 과거 결과·protocol·gate를 변경하지 않으며, v4 완료를 v3.1 완료로 보고하지 않는다. 여기서 “중단”은 등록한 모델·자료·예산에 대한 추가 투자를 중단한다는 뜻이다. 모든 diffusion의 불가능성을 뜻하지 않는다.

## 1. 기존 증거와 v4가 답해야 할 질문

| 문서에서 확인한 사실 | v4 설계에 주는 의미 |
|---|---|
| 최신 synthetic P-DISC 정상/반전 joint는 123/128, 125/128, valid는 256/256 | 후보를 생성할 수 있는 D1을 우선 검증한다. 이 점수를 MD5 성공률로 사용하지 않는다. |
| 같은 실행에서 R-G-BGV는 통과했으나 Printable Gaussian 두 개와 R-DISC는 미달 | 다섯 경로의 형식 개선을 계속하는 대신 한 경로의 실제 MD5 효능을 먼저 확인한다. |
| 이전 실제 MD5-12 결과는 Printable Main 0/305, Random 9/305, Shuffled 0/305 | 양의 MD5 증거가 없다. 당시 형식 실패가 커서 최신 모델의 효과 부재를 입증한 결과도 아니다. |
| p2struct의 prefix3/suffix 손실은 합성 과제의 정답 위치를 이용한다 | 실제 MD5에는 위치 특혜가 없는 payload 손실을 사용하고, 그 설정의 양성 대조 검사를 새로 수행한다. |
| 과거 q≥12 validation/test만으로 Printable 노출 prefix 하한이 1,885개 | 2,048개의 새 test pool을 전제하지 않는다. 노출 감사 뒤 1,024개를 확보한다. |
| 정식 E0·M0–M3·전체 통계 calibration은 아직 미구현 | 문서·개발 검사 통과를 본실험 실행 준비 완료로 표시하지 않는다. |

근거: [연구 가능성 검토](RESEARCH_FEASIBILITY_REVIEW_KO.md), [최신 P2-struct 분석](V3_1_P2STRUCT_ANALYSIS_KO.md), [p2struct 변경 범위](V3_1_P2STRUCT_MODIFICATION_KO.md), [구현 현황](V3_1_IMPLEMENTATION.md). 위 수치는 해당 문서의 관측값이며 이번 계획에서 다시 학습하거나 전체 과거 ledger를 재감사한 결과가 아니다.

v4의 질문은 다음과 같다.

> 고정한 Printable 분포·MD5-12·D1·학습량에서, 새로운 평가 target에 대해 실제 hash condition을 학습한 모델이 정확한 source-prior sampler와 shuffled-condition 모델보다 **반복 가능한 유용한 후보 이득**을 보이는가?

BGV/CGGE 중 어느 표현이 우수한지, Gaussian과 Discrete 중 어느 쪽이 일반적으로 우수한지는 v4의 질문에서 제외한다. P-DISC 선택은 공개된 과거 synthetic 성능을 근거로 한 사전 선택이며, 새 MD5 test 결과로 선택한 것이 아니다. 성공하면 표현 비교를 후속 연구로 열고, 실패하면 현재 자원 범위에서 광범위한 모델 확장을 중단한다.

## 2. 의사결정 기준을 먼저 고정한다

기본 투자 기준은 **절대 성공확률 차이 δ = 0.01, 즉 1%p**다. 계획용 Random 성공률2.412%에서는 약41.5%의 상대 증가에 해당하므로 단순히 양수인 효과보다 높은 문턱이다. 이는 사업적 수익이나 암호학적 가치를 입증하는 기준이 아니라, 다음 연구 단계에 투자할 최소 후보 효능으로 제안한 값이다. 사용자가 정한 기존 기준은 아니며 v4가 제안하는 사전 기준이다. 변경은 실제 test 접근 전에만 가능하며 표본·검정력도 다시 계산해야 한다.

seed `s ∈ {0,1,2}`, 대조군 `c ∈ {Random, Shuffled}`에 대해

`Δ[s,c] = P(Main이 K100 내 성공) − P(c가 K100 내 성공)`

로 정의한다. 여섯 차이 각각의 동시 신뢰구간을 `[L[s,c], U[s,c]]`로 계산한다. 견고한 효과의 요약은 `θ = min Δ[s,c]`다. 평균이 큰 seed가 다른 seed의 실패를 가리지 못하게 한다.

| 최종 상태 | 사전에 고정할 판정 | 연구에 대한 행동 |
|---|---|---|
| `GO` | 모든 무결성·완료 조건 충족, **여섯 L 모두 > 0.01** | 새 데이터에서의 재현·두 번째 source·계산비용 비교를 위한 후속 연구를 설계한다. |
| `NO_GO_SMALL` | 유효한 전체 실험에서 **여섯 U 모두 < 0.01** | 시험한 모든 비교에서 최소 유용 이득이 배제됨. 현재 MD5 후보 우위 접근의 확대를 중단하고 음성 결과를 정리한다. |
| `NO_GO_REPRODUCIBILITY` | 모든 비교를 완료했고 **적어도 하나의 U < 0.01**, 단 위 행은 아님 | “세 seed × 두 대조군에서 모두 유용함”이라는 요구를 충족하지 못함. 현재 접근을 확대하지 않는다. 일부 비교의 효과나 seed 간 이질성까지 부정하지 않는다. |
| `INCONCLUSIVE` | 위 판정 어디에도 해당하지 않음; 경계의 등호도 포함 | 통계적으로 결론이 부족함. v4를 종료하고 자동 증액하지 않는다. 추가 연구는 별도 근거·새 계획이 있을 때만 검토한다. |
| `BLOCKED_QUALIFICATION` | 같은 설정의 양성 대조 검사 미달 | 조건부 생성 구현/학습의 검증 실패. MD5 가설에 대한 음성 결과로 쓰지 않는다. 현재 설정의 본실험 진입을 중단한다. |
| `BLOCKED_EXPOSURE` / `BLOCKED_RESOURCE` | 새 holdout 또는 실행 예산 확보 실패 | 현 설계의 실행 가능성 문제로 기록한다. 효과 없음으로 치환하지 않는다. |
| `INVALID_OR_INCOMPLETE` | 누출·원장 오류·미완료·수치 실패 등 | 유효한 부분 결과만 보존하고 과학적 GO/NO-GO를 내리지 않는다. |

`NO_GO_REPRODUCIBILITY`는 seed 간 차이가 통계적으로 입증됐다는 뜻이 아니다. 여섯 비교 전체의 기준을 충족할 수 없다는 제한된 판정이다. `p ≥ .05`만으로 `NO_GO_SMALL`을 선언하지 않는다. 반대로 작은 양의 효과가 유의하더라도 1%p 기준을 충족하지 못하면 GO가 아니다.

## 3. 최소 실험 행렬

| 항목 | v4 고정값 |
|---|---|
| Source | Printable ASCII 33–126, 94 symbols; 길이 4–31 균등, 주어진 길이에서 bytes iid uniform |
| Hash | 전체 MD5를 계산한 뒤 표준 digest의 첫 12 bits; `int.from_bytes(md5(x).digest(), 'big') >> 116` |
| 표현·모델 | Token `[32]`, 길이를 먼저 생성하는 D1, width128/embedding16, 학습 가능한 condition-output 잔차 유지 |
| Learned methods | 실제 조건 Main, epoch별 조건을 섞는 Shuffled |
| Random | 원래 source prior를 정확히 직접 sampling; 모델·사후 보정 없음 |
| 학습 seed | 0, 1, 2; fresh initialization, 과거 checkpoint·optimizer 재사용 금지 |
| 학습량 | run당 unique messages 10,000, batch64, 100 epochs, 15,700 updates |
| 본실험 learned runs | 2 methods × 3 seeds = 6 |
| Random streams | seed별 1개, 총 3개 |
| 평가 pool | 감사된 새 MD5-12 targets 1,024개 |
| 평가 trials | pool에서 독립 균등 복원추출 **16,384개**, 모든 method/seed에 같은 trial 목록 |
| 후보 예산 | trial마다 정확히 K=100 attempts; @1/@10은 같은 stream의 탐색적 prefix |
| 주 지표 | source-valid 후보 중 target의 MD5-12를 맞힌 후보가 하나 이상인 trial의 비율, Success@100 |
| Backend | 최신 개발에서 사용한 MLX/Metal GPU, float32, GPU 동시 run 1개 |

MD5의 입력은 payload bytes뿐이다. EOS/PAD·표현 헤더·hidden source length는 hash나 외부 condition에 포함하지 않는다. MD5 test vectors와 직렬화는 [RFC 1321](https://www.rfc-editor.org/rfc/rfc1321.html)로 확인한다. 12-bit 절단은 MD5 round 수를 줄이는 작업이 아니다.

원본과 달라지는 것은 범위, backend, test pool 크기, trial 수, 통계 판정 및 적격성 계약이다. v4용 namespace와 protocol identity를 사용한다. **v3.1 JSON에서 pipeline만 삭제하거나 gate를 PASS로 바꾸는 방식으로 실행하지 않는다.**

## 4. 실제 MD5용 objective와 선택 규칙

D1의 기존 기능을 재사용하되 `prefix_balanced_loss=false`로 고정한다. 공개 condition은 12 bits, 내부 payload condition은 여기에 **모델이 생성한** `L/31`을 더한 것이다. Length head는 `Linear(12,28)`이고 temperature1로 L을 한 번 뽑는다.

학습 손실은 메시지별

`length CE + mean(가려진 payload 위치의 CE)`

다. 첫 3바이트를 특별 취급하지 않고 모든 payload 위치를 동일 규칙으로 다룬다. 가려진 payload가 없으면 payload 항만 0이다. 학습 때는 해당 training message의 길이를 사용한다. 생성 때는 hidden representative의 길이를 절대로 사용하지 않는다. Payload vocabulary만 sample하고 EOS/PAD는 생성한 길이에 따라 배치한다. 기존 strict decoder는 유지한다.

Sampling은 32 intervals, temperature1, 확정한 토큰의 remask 없음이다. Length head 1회와 payload denoiser32회를 합쳐 **후보당 NFE33**을 계수한다. Argmax 변경, MD5 기반 reranking, 후보를 고른 뒤 길이/문자를 고치는 절차는 없다.

Adam lr=.001, betas=.9/.999, eps=1e-8, weight decay0, float32를 고정한다. Epoch마다 전체 training permutation을 한 번 사용하며 마지막 16개 batch도 유지한다. Validation은 매10 epochs, 512 messages × 고정 corruption draws4개다. **최소 validation objective**의 checkpoint를 선택하며 동률은 이른 epoch다. Validation/test MD5 hit로 checkpoint를 선택하지 않는다.

Main/Shuffled는 seed별 같은 초기 weights·training 순서·corruption 난수를 사용한다. Shuffled는 epoch마다 train records 전체의 condition donor를 균등 무작위 permutation한다. Length와 payload 양쪽에 동일한 donor condition을 적용한다. 우연히 같은 condition이 남는 수를 기록하고, 이를 없애려고 재추첨하지 않는다. Validation과 inference에서는 두 모델 모두 요청한 실제 target을 받는다.

학습용 paired randomness와 달리, 평가 후보의 난수는 method·seed·trial·attempt별로 독립 배정한다. 길이와 payload 난수도 namespace를 분리한다. 성공 여부가 이후 난수를 바꾸지 않도록 한다.

## 5. 노출 감사와 데이터 구성

### 5.1 노출 계약

현재 1,885개는 기존 노출의 **하한**이다. 파일이 없다는 이유로 미노출로 판정하지 않는다. 로컬/외부 실행, 이전 learned validation/test, q8에서 원문/full digest를 열람한 범위, q≥12의 12-bit 투영, 개발·리허설에서 사용한 MD5 targets를 inventory에 기록한다. 교차 source에서 Printable 원문이나 해당 과제의 결과를 재사용했으면 그 연결도 포함한다.

`E`는 이전 MD5 평가·모델 선택·개발에 사용한 것으로 감사된 Printable 12-bit target 집합이다. 과거 단순 training draw나 무관한 hash 계산도 목록에는 기록하되, 해당 target이 평가/선택에 이용됐는지와 새 모델의 정보 경계를 구분한다. v4의 “새 target”은 이 사전 정의한 평가 노출 집합에 없는 target이며, 프로젝트에서 그 숫자를 한 번도 계산하지 않았다는 뜻은 아니다. 감사 불확실성이 남아 exclusion을 결정할 수 없으면 차단한다.

`A = {0,…,4095} − E`에서 1,024개 이상 확보해야 한다. 부족하면 `BLOCKED_EXPOSURE`다. q16으로 몰래 전환하거나 seed만 바꿔 재사용하지 않는다. 높은 q도 노출·표본·예산을 다시 설계해야 하는 새 연구다.

### 5.2 사전에 결정할 생성 절차

1. E0 리허설은 이미 E에 있는 MD5 groups에서만 수행하고 최종 E를 봉인한다. 리허설에서 추가 노출이 발생했다면 반드시 E에 합친다.
2. `A`를 한 번 섞어 첫 1,024개를 test pool로 지정한다. 나머지 전체 3,072개를 별도 난수로 섞어 train1,536 / validation512 / reserve1,024 groups로 배정한다. E는 test에서 배제되지만 train/validation에 들어갈 수 있다. Reserve는 이번 연구에서 사용하지 않는다.
3. 원래 source prior에서 메시지를 순서대로 draw하고 raw duplicate를 제거한다. Train owner의 unique messages10,000개, validation 각 group의 첫 대표512개, test 각 group의 첫 대표1,024개를 수집한다. 전체 draw cap은 1,000,000회다. Quota를 채우지 못하면 구성 실패로 중단한다.
4. Draw·duplicate·owner별 surplus·reject 수, 길이/문자 분포, raw와 12-bit group의 split 간 overlap=0, 파일 hash를 기록한다. Retained training data는 group ownership에 조건화되어 있으므로 원래 prior의 무조건 iid 표본이라고 기술하지 않는다.
5. Test 대표 원문은 별도 evaluator 영역에 두고 생성 경로에는 12 bits만 전달한다. Training/validation 코드가 이 파일을 읽지 못하는지 fixture로 검사한다. Baseline 후보는 train owner에 제한하지 않고 원래 source prior 전체에서 뽑는다.
6. 여섯 본실험 checkpoint를 모두 봉인한 뒤 test pool에서 16,384 targets를 iid replacement로 한 번 추출한다. 중복 target이나 낮은 coverage를 이유로 재추첨하지 않는다. 같은 target의 다른 trial에는 독립적인 후보 stream을 사용한다.

Study master는 `2026092804`, model labels는 0/1/2다. Python/NumPy/MLX 버전과 lockfile을 봉인한다. 데이터·ownership·shuffle·validation·trial·length·payload의 seed namespace를 분리하고, UTF-8 compact JSON 배열의 SHA-256 첫8bytes big-endian으로 seed를 유도한다. 정확한 배열 필드와 라이브러리별 seed/key 변환은 실행 명세에 기록하고 테스트 후 봉인한다. 공통 training 난수의 method 필드만 null로 하며, inference에는 실제 method를 넣는다.

Pool 1,024개와 trials16,384개는 다른 수다. 복원추출 반복으로 **같은 고정 pool에서의 평균 성공확률**을 정밀하게 추정한다. 새로운 16,384 digest를 시험했다고 쓰지 않으며, 전체 4,096 targets나 새로운 데이터 분할·임의 학습 seed로 자동 일반화하지 않는다.

## 6. 실행 단계와 기술 실패의 해석

| 단계 | 실행 내용 | 통과 조건 / 다음 단계 |
|---|---|---|
| V0 구현·무결성 | v4 dispatch, 실제 MD5 train/evaluate/report, trial ledger, 통계 경로, 정보 경계, resume 구현·검사 | Required checks PASS; 기존 `development_only` 실행기로 대체 불가 |
| V1 양성 대조 | MD5용과 **같은 D1 구조·균일 payload loss·sampler**로 synthetic_nibbles를 fresh seeds0/1/2 학습 | 아래 engineering 기준 모두 충족 |
| V2 E0·calibration·자원 | 작은 실제 MD5 전체 경로, 실제 통계 구현 검산, 비용 실측, 노출 감사 확정 | 모든 test 접근 전 준비 사항 봉인 |
| V3 데이터·학습 | Ownership·자료 봉인, Main/Shuffled × 3 seeds 전체 학습·checkpoint 선택 | 여섯 checkpoint 봉인; 학습/후보 성과에 따른 seed 교체 없음 |
| V4 단회 평가 | 공통 trial 목록에서 9 streams × 16,384 × 100 attempts | 모든 ledger 완성 및 verifier 검사 |
| V5 판정·보고 | 여섯 효과와 동시 CI, 비용, 진단, 결측 사유, GO/NO-GO 결정 | 유효한 음성 결과도 완료로 인정 |

V1은 v3.1 P3를 통과했다고 인정하는 단계가 아니다. v4의 단일 경로 정보 사용을 확인하는 별도 engineering screen이다. Synthetic는 기존과 같이 y의 세 nibbles를 앞3 bytes로 표현하고 suffix를 prior에서 뽑는다. Complement pair를 함께 소유시켜 train3,072 / validation512 / acceptance512 conditions로 분리하고 train10,000 messages를 사용한다. 이 synthetic acceptance는 새 split에서 개발과 분리할 뿐 프로젝트 전체에서 처음 본 condition 숫자라는 주장은 하지 않는다.

V1의 seed별 학습은100 epochs·15,700 updates, validation 선택은 §4와 동일하다. Acceptance512개에 normal/flipped 각각 K1을 생성한다. **각 seed에서 normal joint≥461/512, flipped joint≥461/512, valid 각각512/512, flipped 후보의 원래 조건 오성공≤25/512**를 요구한다. 서로 반전한 condition은 같은 초기 length/payload RNG로 비교한다. Joint는 strict valid와 요청 synthetic prefix 만족을 함께 뜻한다.

90% 수준의 조건 사용 기준은 위치 균형 손실을 제거한 단일 경로의 작동 확인용으로 새로 등록한 값이다. 기존 v3.1의 475/512 적격성 문턱을 통과했다고 바꾸지 않으며, MD5 성능의 대리 기준으로 사용하지 않는다. V1 실패 시 새로운 loss·seed·profile을 현장에서 추가하지 않는다. 이 경우 v4는 조건 사용 검증 실패로 종료하고 MD5 효과를 기각하지 않는다. 과거 p2struct 가중 loss의 성공은 이 V1을 대신할 수 없다.

E0는 Main/Shuffled 각 seed99로 train256·2 epochs·8 updates/run, validation32를 사용한다. Train/validation/test groups는 서로 겹치지 않는64/32/32개이며 모두 기존 E에서 선택한다. 자료 draw cap은1,000,000회이고, 부족하면 실패로 기록한다. Test pool32에서 replacement trials32개, K100; learned6,400 + Random3,200 =9,600 rows다. E0의 hash hit 수는 통과 조건이 아니다. Main용 함수가 실제 MD5 판정·partial 보고·복구를 수행하는지 확인하는 검사다. E0 weights는 본실험에 넘기지 않는다.

무결성 검사는 적어도 RFC hash vectors, 12-bit leading zero, raw-byte roundtrip, hidden length 차단, Main/Shuffled loss 일관성, target 반복 trial 분리, invalid/duplicate/성공 후 attempt 계수, resume 중복 commit 방지, verifier 재해시를 포함한다. 학습 중단/재개와 후보 batch 중단/재개의 연속 실행 대비 일치는 실제 사용할 MLX backend·batch에서 검사한다.

## 7. 성공 지표와 후보 원장

Trial의 성공은 후보100개 중 `valid(x) && H12(x)==requested_target`가 하나 이상 존재하는 경우다. 원래 대표 메시지의 복원이나 길이 일치는 요구하지 않는다.

모든 attempt를 기록한다. Invalid도 기회를 소비하며 duplicate를 추가 후보로 대체하지 않는다. 첫 성공 이후에도100회까지 생성한다. Payload bytes가 존재하면 실제 MD5 호출을 계수하고, decode 실패로 bytes가 없으면 call0과 사유를 남긴다. 생성기에는 검증 결과를 반환하지 않는다.

기존 SQLite 원장·atomic checkpoint·seal을 재사용한다. 최소 키는 `(protocol_id, method, seed, trial_id, attempt)`이며 target만으로 unique key를 만들지 않는다. 후보에는 source/domain 판정, payload 또는 invalid 사유, 재계산 digest prefix, 성공, checkpoint/config identity, RNG identity를 남긴다. 모든 payload는 별도 verifier로 다시 해시한다. Duplicate/학습 메시지와 일치/첫 성공 위치는 진단으로 보고하되 주 분석의 행을 삭제하지 않는다.

필수 지표는 method/seed별 Success@100, `n11/n10/n01/n00`, absolute Δ와 동시 CI, valid 비율, duplicate 비율, valid일 때 hash hit, 길이 분포, 학습 메시지 일치, NFE, 실제 MD5 calls, 전체 시간, 저장량이다. @1/@10과 상대 lift는 탐색적이며 baseline 성공0이면 ratio는 null이다. Valid만 남긴 성능을 primary 성능으로 대체하지 않는다.

## 8. 통계 분석과 표본 수의 근거

### 8.1 추론 대상과 동시 신뢰구간

추론은 **봉인한 데이터·test pool·학습된 여섯 checkpoint·지정 세 seeds에 조건부인**, 무작위 target trial 및 독립 generation randomness에 대한 것이다. Trial마다 같은 target을 비교하는 paired 분석을 한다. Seeds를 합쳐 `3N`개의 독립 관측처럼 처리하지 않는다.

한 비교의 discordant counts를 `n10 = Main만 성공`, `n01 = 대조군만 성공`으로 둔다. `Δ = p10 − p01`, 추정값은 `(n10−n01)/N`이다. iid replacement trial과 독립 stream 계약 아래 각 discordance count의 주변분포는 Binomial(N,p)다. 비교 간 독립성은 요구하지 않는다.

각 비교의 p10/p01에 Clopper–Pearson 양측 구간을 구하고, 각 tail의 오류를

`ε = 0.05 / (6 comparisons × 2 probabilities × 2 tails) = 0.05/24`

로 둔다. `CP(x,N)=[l(x),u(x)]`라면

`[L,U] = [l(n10)−u(n01), u(n10)−l(n01)]`.

따라서 union bound로 **여섯 차이를 동시에 포함할 확률이 최소95%인 보수적 구간**을 얻는다. 이 구간으로 §2를 한 번만 판정한다. 추가 Holm이나 CI95 bootstrap을 다른 GO 문턱으로 겹치지 않는다. 별도로 계산한 unadjusted p값은 진단용일 뿐 결정 규칙이 아니다.

CP는 binomial CDF를 역산한다. x=0의 lower=0, upper=`1−ε^(1/N)`이며 x=N은 대칭이다. Binomial-CDF 정의와 검산 예시는 [NIST의 proportion confidence limits](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/propconf.htm)를 사용한다. CP를 discordance에 적용해 차이의 동시 구간을 만드는 방식은 위에 명시한 v4 설계다. 수치 역산의 정확도와 경계값을 실행 코드에서 검사한다.

모든 비교를 마칠 때까지 효과별 결과를 이용한 중단·증액·checkpoint 변경을 하지 않는다. 동일 target의 서로 다른 trials는 독립 난수로 생성하며 target별 후보 cache를 재사용하지 않는다. 이 계약이 깨지면 위 이항 추론을 그대로 사용할 수 없다.

### 8.2 N=16,384를 선택한 근거

균등 독립 hash를 가정한 계획용 Random Success@100은

`p0 = 1 − (1−1/4096)^100 = 0.02412136`, 약2.4121%다.

Stream당 기대 성공 수는 약395.20개다. 이 값은 실제 MD5 baseline의 관측값이나 정확한 성공확률을 대신하지 않는다. 가용 pool1,024개에서의 실제 source별 난이도·중복·모델 편향은 별도로 보고한다.

[설계 검산 스크립트](scripts/validate_research_plan_v4.py)는 실제 해시·모델·archive를 읽지 않고, 각 시나리오20,000회의 가상 실험에서 **여섯 비교를 함께 판정**한다. Main을 두 비교가 공유하고, 난이도 시나리오에서는 같은 target 난이도를 모든 method/seed가 공유한다. 기본 시나리오는 target 확률에 조건부 독립 Bernoulli다.

| 가정한 Success@100 | 설계상 의미 | 2026-09-28 계산 |
|---|---|---:|
| Main=Random=Shuffled=p0 | 효과가 없을 때 여섯 차이 모두 1%p 미만으로 배제 | `NO_GO_SMALL` 약85%; 그 외 대부분 `NO_GO_REPRODUCIBILITY` |
| Main=2p0, 두 controls=p0, 세 seed 동일 | 약2.412%p의 실제 이득이 있을 때 진행 | `GO` 약99% |
| Main=p0+0.01, 두 controls=p0 | 최소 기준 바로 경계 | 거의 전부 `INCONCLUSIVE` |
| 두 seeds는2p0, 한 seed는p0+0.005 | 일부 seed가 최소 기준 미달 | 주로 `INCONCLUSIVE`; 충분한 배제 근거가 생기면 `NO_GO_REPRODUCIBILITY` |
| Main=2p0, Random=p0, Shuffled=1.5p0 | Shuffled 대비 차이가1.206%p로 기준에 가까움 | 거의 전부 `INCONCLUSIVE` |

추가로 한 비교만 null, 공통 난이도 .2/1.8, 두 strata에서 효과 방향이 뒤집히는 평균-null을 계산한다. 자세한 수치는 [계산 JSON](local_experiment_archive/analyses/v4-design-20260928/design_calculation.json)에 남긴다. 이는 **검정 설계에 대한 가상 결과**이며 MD5에서 99% 확률로 성공한다는 예측이 아니다.

설계 검산은 NIST의 x8/n30 CP 예제, 성공0/전체 성공, 잘못된 입력 거부, paired 차이 대칭, 판정 분기 및 row/NFE 산술을 포함한다. Reference twofold의 GO 확률과 all-null의 NO_GO_SMALL 확률에 대한 one-sided95% CP 하한이 각각.80 이상인지도 확인한다.

### 8.3 실행 코드 calibration은 별도 필수 조건

위 스크립트가 통과해도 production ledger→집계→판정 경로의 calibration을 완료한 것은 아니다. V2에서 실제 보고서가 사용하는 통계 함수에 가상 joint outcome counts를 넣어 같은 여덟 시나리오를20,000회씩 고정 seed로 실행하고, 위 계산과 대조해야 한다. 큰 모의 실험은 multinomial counts로 계산하고 수백억 개의 가상 후보를 SQLite에 쓰지 않는다. Ledger→trial→counts의 연결은 별도의 작은 완전 원장 fixture로 전수 검사한다. 수학 primitive는 재사용하고 다른 판정 로직을 복제하지 않는다.

실행 gate는 simultaneous coverage의 one-sided95% CP lower≥.94, 최소 기준 미달/null 시나리오에서 잘못된 GO의 one-sided95% CP upper≤.06, reference twofold의 GO lower≥.80, all-null의 NO_GO_SMALL lower≥.80이다. Boundary에서는 잘못된 GO와 잘못된 NO-GO의 합도 upper≤.06이어야 한다. 여섯 결과의 결측, 반복 target, seed별 shared target, success0, invalid와 중복 ledger fixtures도 검사한다. 수학적 최소95% 보장을 calibration .94로 완화하는 것이 아니라 Monte Carlo 오차와 구현 오류를 검사하는 기준이다.

Calibration이 실패하면 test에 접근하지 않고 원인을 수정한다. 계획값을 변경해야 하면 새 revision을 만든다. 결과가 좋아지는 seed로 calibration을 반복하지 않는다. 기존 `study_statistics.analyze_family()`의 unique-target·seed별 Holm 계약은 v4와 달라 그대로 호출할 수 없다.

## 9. 자원 한도와 중단 정책

v4는 모델 수를 줄이는 대신 **음성 결과의 정밀도**에 예산을 쓴다. 본실험 row 수가 작아졌다고 주장하지 않는다.

| 고정 실행량 | 수량 |
|---|---:|
| V1 양성 대조 learned runs / updates | 3 / 47,100 |
| V1 acceptance candidates / NFE | 3,072 / 101,376 |
| E0 learned runs / updates | 2 / 16 |
| E0 learned+Random rows / learned NFE | 9,600 / 211,200 |
| 본실험 learned runs / updates | 6 / 94,200 |
| 본실험 learned candidates | 9,830,400 |
| 본실험 Random candidates | 4,915,200 |
| 본실험 전체 candidate rows | **14,745,600** |
| 본실험 sampling NFE | **324,403,200** |

위 수량에 validation corruption forward, V0 fixtures, profiler, verifier 재검산과 crash replay는 포함하지 않았다. 이 비용도 예산에는 모두 포함한다. 검증용 rehash는 후보 기회를 늘리지 않지만 hash-call/시간 비용에 별도로 계수한다.

V2에서는 warm-up20 updates 후100 updates 구간3개와 생성 batch1/4/16/64를 warm-up1회·측정3회씩 측정한다. 데이터 준비, 동기화, decode, 실제 MD5, SQLite commit, telemetry를 포함한 전체 주기 중 보수적인 값을 사용한다. 유효하고 메모리 상한을 만족하는 batch 중 처리량이 가장 큰 것을 택하고1% 이내 동률은 작은 batch를 선택한다. 성능 점수로 batch를 선택하지 않는다. 본실험 전에 batch를 봉인하고 바꾸지 않는다.

`예상 본실험 시간 = 6회 학습/validation/checkpoint + 9,830,400×learned 후보 비용 + 4,915,200×Random 후보 비용 + 전체 재검산/보고 비용`.

예상 시간×1.5를 soft budget, 큰 SQLite/index/WAL fixture의 row당 비용으로 구한 저장량×2를 저장 여유로 요구한다. 두 보수적 추정이 아래 hard cap 안에 들어와야 착수한다. 기존 synthetic의 작은 resource 파일을 본실험 전체 비용 실측으로 사용하지 않는다.

| Hard cap | 고정값 |
|---|---:|
| V0+V1+V2 전체 active time | 24시간 |
| V3+V4+V5 전체 active time | 72시간 |
| v4 전체 active time | 96시간 |
| learned run 하나 | 24시간 및 해당 stage 잔여 예산 이내 |
| Study 저장량 / process RSS / GPU allocated | 각각64GiB / 64GiB / 64GiB |
| 디스크 최소 여유 | 10GiB |

이 시간은 실측 완료 예상이 아니라 **투자 상한**이다. RSS와 GPU memory는 중복될 수 있으므로 합산하지 않는다. 구현에 투입할 사람의 작업시간은 이 active-time 한도에 포함하지 않으며 별도로 기록한다. 실행 불가를 감추기 위해 품질·trial 수·epochs를 줄이거나 cap을 사후 늘리지 않는다.

Soft 초과는 경고 후 hard cap 안에서 진행한다. Hard 초과는 resource stop이다. Exact resume는 일시 중단에 대해 run당1회까지 허용하며 replay한 실제 시간·NFE를 합산한다. Hard cap 소진이나 품질 미달은 새 seed 재시도의 사유가 아니다. Numerical failure나 incomplete outcomes를0으로 넣지 않는다. 한 run의 고립된 오류 뒤 독립 run은 전역 무결성·자원 상태가 정상인 경우에만 계속하고, 최종 판정은 incomplete로 남긴다.

## 10. 비용과 최종 해석

동일 K에서 후보 효능이 좋아져도 신경망을33회 호출하는 비용을 회수했다는 뜻은 아니다. V5에는 데이터 생성·학습·checkpoint·generation·verification·저장/보고 비용을 나누고, inference-only 및 training 포함 성공 trial당 비용을 함께 기록한다. Source-prior Random의 같은 수의 trial에 대한 실측값도 보고한다. 성공0인 비용 비율은 null로 남긴다.

이 비교는 throughput 진단이지, 최적화한 전통 탐색이나 전처리 lookup에 대한 공정한 계산량 우위 검정이 아니다. q12의 작은 출력 공간에는 전처리 table이 강한 경쟁법이므로 **GO 뒤 비용 연구**에서는 학습 전처리와 lookup 전처리를 함께 계수해야 한다. v4 평가 전에 table을 만들어 test 대표를 생성기에 제공하지 않는다.

GO가 나오면 허용하는 결론은 다음 범위다.

> 고정한 Printable source, MD5-12, D1 및 학습량에서, 지정한 세 학습 seed 각각이 두 사전 대조군보다 Success@100에서 최소1%p 높은 효과를 보였고, 동시 신뢰구간이 그 기준을 지지했다. 결과는 봉인한 test pool 및 checkpoints에 조건부다.

허용하지 않는 결론은 full MD5/SHA-256 역상 가능성, 암호학적 보안 붕괴, 새로운 seed/모든 targets의 우위, BGV/CGGE의 우월성, 학습비를 포함한 계산량 우위다.

NO-GO 결과는 **현 설정의 효과 크기 상한**과 함께 제시한다. 연구 자원을 조건부 생성/표현의 engineering 연구로 돌릴 수는 있지만, 그 선택을 MD5 역상 성공으로 표현하지 않는다. INCONCLUSIVE도 연구를 무한히 연장하는 허가가 아니다. 한 번의 제한된 v4로 의사결정 자료를 만들고 현 revision은 닫는다.

## 11. 구현 경계와 필수 산출물

재사용할 것은 D1 모델·균일 payload loss 분기, codec·hash verifier, candidate ledger, checkpoint/recovery, telemetry, binomial/McNemar 등 검증 가능한 통계 primitive다. 새 모델 프레임워크, 전체 pipeline 탐색기, 외부 실험 관리 서비스는 필요하지 않다.

필요한 구현은 **v4 전용 고정 설정, 실제 MD5 데이터/학습/평가 연결, trial 단위 분석, 노출 감사, 생산 경로 calibration과 보고**다. 실행 명세에는 본 문서의 값과 seed derivation·파일 schemas·batch/resource 실측값을 담고 code/dependency hash를 봉인한다. 실행 명세가 완성되기 전에는 v4 CLI가 준비됐다고 안내하지 않는다.

필수 산출물은 아래 여섯 묶음이다. 하나의 study directory 아래 기존 artifact 구조를 활용한다.

1. `protocol.frozen.json`, 코드/환경 manifest, exposure inventory와 audit, data ownership/checksums.
2. V0 검사, V1 세 seed 결과, E0 전체 경로 결과, production calibration 결과, 최종 resource/batch 봉인.
3. 여섯 model의 training history, 최소 validation checkpoint, complete/failure 및 resume 기록.
4. Checkpoint 봉인 이후 생성한 공통 trial schedule와 아홉 candidate ledgers.
5. 독립 verifier 감사, 여섯 비교의 effect/동시 CI, 비용·진단·결측 표.
6. `decision.json`과 한국어 최종 보고서: execution status와 scientific decision을 분리하고 근거·적용 범위·다음 행동을 명시.

계획 작성 시점의 산출물은 **이 계획과 설계 계산 스크립트/출력**이었다. 이후 구현한 고정 설정은 [v4-protocol.json](examples/v4-protocol.json), 실행·감사·재개·검증 명령은 [V4_CLI.md](V4_CLI.md)에 정리했다. 개발 검증용 checkpoint·원장·calibration을 정식 study의 V0–V2 PASS로 재사용하지 않는다. 새 본실험 holdout·학습된 여섯 본실험 모델·과학적 판정은 아직 없다.

검산 재현 명령:

```sh
.venv/bin/python scripts/validate_research_plan_v4.py
```

이 명령은 가상 결과·신뢰구간·실행량만 계산한다. 실제 실험 실행 명령이 아니다.
