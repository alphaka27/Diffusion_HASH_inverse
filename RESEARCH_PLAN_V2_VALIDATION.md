# 연구 계획 v2 검증 보고서

**검증일:** 2026-09-24 KST · **대상:** `dhi-poc-v2-20260923`  
**판정:** 개발·측정 파이프라인 PoC에는 조건부 적합. **현재 코드로 확증적 hash 본실험을 즉시 시작하는 것은 부적합(BLOCKED).**

이는 연구 계획 작성 실패나 모델 가설의 기각을 뜻하지 않는다. 수치·구성·일부 재사용 컴포넌트는 검증됐지만 실제 학습 positive control, v2 실행 경로, 최종 holdout/analysis/resource seal은 남아 있다는 뜻이다.

대상 문서는 [RESEARCH_PLAN_V2.md](/Users/choisoonwook/Experiments_local/DHI_AI_gen/RESEARCH_PLAN_V2.md), 고정값은 [poc-v2-protocol.json](/Users/choisoonwook/Experiments_local/DHI_AI_gen/examples/poc-v2-protocol.json)이다. 검증 당시 plan SHA-256은 `f2a2779a722ad52820a35480affb08d07b594b6c7513cfd61d33ea5d3509c0ba`, protocol JSON SHA-256은 `4896d530655d140e4059b9d94f7d8ed07a2e70fcfa6cb3c26694c54915d8355f`다. Source 및 validator hashes와 시작 timestamp는 [원시 검증 기록](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/2026-09-23-plan-v2-validation/validation_results.json)에 있다.

원본 RESEARCH_PLAN.md의 기존 사용자 수정은 보존했다. 제품 코드(src/)는 변경하지 않았다. 과거 candidate bytes나 성능률을 분석 대상으로 열지 않았으며 노출 감사에서는 target-list metadata만 추출했다. 실제 모델 학습, primary hash holdout 평가, scientific gate 인증은 수행하지 않았다.

## 1. 수행 범위와 판정표

| 점검 | 실제 수행 내용 | 판정과 한계 |
|---|---|---|
| Matrix·예산 산술 | 5 pipelines, 3 seeds, 2 learned methods, 100 candidates | PASS: 30 learned runs, 6,144,000 candidates |
| 과거 노출 metadata | 알려진 retired PoC의 MD5 q=12/16 test·validation 12개 metrics metadata | 중요 보완 발견; 전체 접근 감사는 미완료 |
| Source별 dataset 구성 | 실제 prior draws와 MD5로 quota/ownership dry-run | PASS: 두 source에서 각각 10,000/512/2,048 확보; primary dataset 아님 |
| Codec | 5개 applicable 조합의 clean round-trip 1,214건 | PASS: 합성 generation 능력은 검증하지 않음 |
| Synthetic 조건 분할 | complement-pair allocation과 교집합 검사 | PASS: 반전 조건도 held-out 유지 |
| Component 통계 | exact McNemar 225개 작은 정수 조합을 독립 binomial 합과 비교 | PASS: 모델 가정의 참임을 증명한 것은 아님 |
| Holm·missing 규칙 | 알려진 수치 fixture, max-p에 missing=1 포함 | PASS: 새로운 family orchestration은 구현 필요 |
| Power | 5개 시나리오 × 20,000회 simulation | Reference 약 80.9%; 약한 seed·강한 control에 민감 |
| 기존 구현 정합성 | Shuffled inference 및 old positive-control input 비교 | FAIL: v2 실행에 직접 재사용 불가 |
| Sampler 공학적 확인 | 새 model dimensions/schedules로 untrained synthetic CPU sampling | PASS: 학습/condition use/생성 유효도 근거 아님 |
| 관련 기존 tests | Codec·domain·prefix·budget·paired 통계 등 | 43 passed, 학습 runner tests 2개 제외 |
| 전체 실행 준비 | Learned G1-B, narrow boundary, final data seal, resources | BLOCKED |

## 2. 공통 holdout 제안을 source별 holdout으로 수정한 근거

권장 수정안은 두 source에 동일 ownership을 사용하는 방안을 포함했다. 실제 archive를 확인하니 `retired_2026-09-21/artifacts/poc_md5_truncated`에 q=12뿐 아니라 q=16의 모델 test·validation target 이력이 남아 있었다. q=16 target도 첫 12 bits를 제외하는 보수적 규칙으로 환산했다.

| 배제 범위 | 이미 평가된 unique 12-bit prefixes | 잔여 prefixes |
|---|---:|---:|
| Printable의 기존 test/validation | 1,885 | 2,211 |
| Random Bytes의 기존 test/validation | 1,860 | 2,236 |
| 두 source 합집합 | 2,915 | 1,181 |

따라서 project-wide 공통 미노출 2,048개 확보는 이 배제 규칙과 양립하지 않는다. Source별로는 현재 알려진 목록에서 가능하다. v2는 **source별 ownership, same-source target 공유**로 바꾸었다. 이는 단순 seed 변경으로 노출을 지우는 방식이 아니다.

이 선택은 다른 source에서 관찰한 같은 prefix까지 미노출이라고 주장하지 않는다는 제한을 수반한다. 연구자는 source-specific novelty와 새 독립 weights/data라는 범위가 적절한지 최종 기록에 명시해야 한다. 또한 q<12 자료의 full digest/원문 활용, 다른 저장 위치, 사람의 이전 접근까지 이번 metadata scan으로 완전히 감사한 것은 아니다. 추가 배제 후 pool이 2,048개 미만이면 계획을 개정해야 한다.

## 3. Dataset 구성 가능성

최종 primary seed를 사용하지 않고 별도 engineering seed로 source-specific ownership과 실제 prior sampling을 수행했다. Source messages는 결과 파일에 기록하지 않았다.

| Source | 총 draws | Unique train messages | Validation targets | Test targets | Raw split overlap |
|---|---:|---:|---:|---:|---:|
| Printable | 35,181 | 10,000 | 512 | 2,048 | 0 |
| Random Bytes | 31,438 | 10,000 | 512 | 2,048 | 0 |

각 source의 draw cap 1,000,000 이내에서 구성됐다. Test와 알려진 same-source exposure 목록의 교집합도 0이었다. 이 결과는 지정 quota가 구조적으로 실현 가능한지 보여준다. Final ownership/config/hash와 연결된 G0 report, primary corpus seal을 대신하지 않는다.

Final test는 과거 노출에서 제외한 available pool에 조건부로 선택된다. 동일한 4,096개 digest 전체로의 일반화나 두 source의 동일 target 비교로 해석하지 않는다. Train source prior도 digest ownership 및 raw uniqueness로 조건화되므로 원래 prior와 동일하다고 보고하면 안 된다.

## 4. Codec와 positive-control 설계 검증

| Pipeline | Clean round-trip 검사 |
|---|---:|
| P-G-BGV | 178/178 |
| P-G-CGGE | 178/178 |
| P-DISC | 178/178 |
| R-G-BGV | 340/340 |
| R-DISC | 340/340 |
| 합계 | **1,214/1,214** |

모든 symbol의 길이-4 반복문과 모든 허용 길이의 extrema/mixed pattern을 검사했다. 잘못된 구조·shape·비유한 값·token grammar에 대한 관련 기존 tests도 별도로 통과했다. CGGE 고정 table checksum은 계획 값과 일치한다.

권장안의 단순 condition split에는 조건을 반전한 값이 training split으로 들어갈 여지가 있었다. v2는 `(y,y XOR 4095)` pair를 같은 split에 배정하여 수정했다. Train/validation/test condition 수는 3,072/512/512이며 교집합이 없고 complement closure가 확인됐다.

512 trials에서 지정 Wilson 기준은 정상/반전된 실제 조건에 대해 각각 최소 **475건**의 joint success, 반전 생성물이 원래 조건을 만족하는 것은 최대 **15건**에 해당한다. 이 산술 기준과 corpus 구성만 검증했다. 학습한 모델이 이 기준을 통과했는지는 **NOT RUN**이며 G1-B는 미통과 상태다.

## 5. 통계적 일관성과 검정력 한계

Exact McNemar 구현은 n10,n01=0…14의 225개 조합에서 독립 binomial 합과 일치했다. Discordance 0에서 p=1, Holm fixture의 조정값, missing component가 max-p를 1로 만드는 규칙도 확인했다.

여섯 component 중 true-null p-value가 하나라도 있으면 `Pr(max(p)≤a)≤a`이므로 max-p 결합은 타당하다. 이후 다섯 composite에 대한 Holm은 이 다섯 반복 우위 주장에 대한 family다. 그러나 **component p-value의 모형 타당성을 보정이 만들어 주지는 않는다.**

v2는 독립 target pairs 및 discordant-direction exchangeability라는 working model을 명시했다. Heterogeneous target 효과가 상쇄되는 임의의 평균-null 전체에 대해 exact 보장을 주장하지 않는다. 이 가정을 선택한 finite benchmark에 적용하기 어렵다면 G3 확증 지위를 부여할 수 없다. 이번 수치 simulation은 가정이 실제 모델에서 성립한다는 증명이 아니다.

Power 검증은 reference p0=0.02412136, N=2048, K=100으로 수행했다. 아래 통과 확률은 한 pipeline의 여섯 component p-value가 모두 .01 미만인 Holm 충분조건이다. 전체 Holm이나 전체 다섯 pipeline의 공동 성공을 모의한 값이 아니다.

| 시나리오 | 통과 확률 | Monte Carlo 표준오차 |
|---|---:|---:|
| Main=2p0, 양쪽 control=p0, 세 seed 동일 | 80.940% | 0.278 percentage points |
| 한 seed만 Main=1.5p0, 나머지 2p0 | 23.695% | 0.301 pp |
| Main=2p0, Random=p0, Shuffled=1.5p0 | 3.025% | 0.121 pp |
| 같은 비율로 두 난이도 strata .2/1.8 | 82.980% | 0.266 pp |
| Main=3p0, 양쪽 control=p0 | 20,000회 모두 통과 | 실제 확률이 1이라는 뜻은 아님 |

Reference는 이전 100,000회 계산의 약 81.3%와 Monte Carlo 변동 범위에서 부합한다. 다만 “2,048개면 항상 80% 검정력”이라는 표현은 틀리다. 특히 stronger shuffled 시나리오에서 Main은 Shuffled의 약 1.33배일 뿐이므로 원래의 두 배 대안보다 작은 효과를 검출하는 문제가 된다.

판정은 **큰 효과가 모든 seed에서 안정적인 경우의 PoC에는 조건부 적합, 작은 개선이나 seed 변동을 배제하는 연구로는 부족**이다. 이 범위를 받아들이고 음성 결과를 제한해서 보고하면 설계를 사용할 수 있다. 더 넓은 음성 결론이 목표라면 test 전에 목표 효과/N/K/family를 다시 설계해야 한다.

## 6. 모델·sampler와 실제 비용

Gaussian T=1,000의 terminal alpha-bar를 직접 계산하여 약 `4.0358297654e-05`를 확인했다. 기존 100-step 설정보다 pure-noise 근사에 적합하지만 실제 학습 모델의 generation 품질을 입증하지는 않는다.

Untrained model에 synthetic zero condition을 넣고 CPU 2 threads, batch 4, warm-up 1회 이후 3회 median을 측정했다. 생성물을 MD5 target과 비교하거나 학습하지 않았다.

| 구조 | Parameters | Sampling steps | 후보당 시간 |
|---|---:|---:|---:|
| BGV ImageUNet | 114,978 | 100 | 약 0.1351초 |
| CGGE ImageUNet | 114,978 | 100 | 약 0.0672초 |
| Printable SequenceDenoiser | 465,168 | 32 | 약 0.00203초 |
| Random-Bytes SequenceDenoiser | 1,136,496 | 32 | 약 0.00534초 |

이 정책을 그대로 전체 primary 후보 수에 곱하면 sampling 추론만 약 **117.7시간, 4.9일**이다. 학습·validation·decode·verification·ledger IO·positive controls·중단은 제외했다. 최적화된 batch나 accelerator의 성능을 예측한 것이 아니다. 이 실행 환경에서는 CUDA와 MPS가 available=False였으며, 물리 장비 전체에 accelerator가 없다는 판단으로 확대하지 않는다.

Discrete를 32 steps로 고정하여 primary NFE 합은 447,283,200이다. 모든 모델을 100 steps로 가정했던 제안서의 614,400,000과 다르다. Learned candidates는 6,144,000개, source별 공유 random을 포함한 ledger는 7,372,800 rows다. 전체 raw image pool을 Python list로 누적하는 기존 경로는 이 규모에 부적합하다.

따라서 **30 runs라서 가벼운 실험이라는 판단은 부적절**하다. Actual training profile과 최종 batch/device에서 benchmark한 뒤 전체 resource cap을 채워야 한다. 이 자료는 계산량 우위의 증거가 아니다.

## 7. 현재 구현에서 실행을 막는 확인된 차이

| 항목 | 확인 결과 | v2에 필요한 작업 |
|---|---|---|
| Shuffled inference | Legacy `_conditions`가 생성 시에도 실제 target을 permutation함 | Training donor pairing과 inference condition 분리 |
| 기존 positive control | Hash 모델 입력 2 channels, reversible control 4 channels | 동일 12-bit 경로·동일 architecture의 실제 학습 control |
| Condition boundary | Legacy 함수가 full DigestRecord를 받음 | Evaluator metadata가 없는 narrow model-facing input과 mutation audit |
| Candidate prefixes | Legacy runner는 config.k별 독립 실행 가능 | 한 번의 100-attempt stream에서 1/10/100 평가 |
| Final statistics | 기존 family aggregation은 seed별 correction 중심 | 6-component max-p 후 5-setting Holm과 missing 상태 연결 |
| 메모리·재개 | Legacy는 전체 raw representation을 attempts에 누적 | Streaming ledger·제한 raw sample·정확한 resume 검증 |

이 차이를 지적한 것은 기존 archived 실험을 v2 실험으로 판정했다는 뜻이 아니다. Legacy runner 자체도 engineering/legacy scope로 표시되어 있다. 해당 코드를 v2 확증 실행기로 그대로 사용하지 말아야 한다는 의미다.

## 8. 재현 방법과 실제 테스트 결과

프로젝트 루트에서 다음 명령으로 plan 검증을 재실행할 수 있다.

```sh
.venv/bin/python scripts/validate_research_plan_v2.py
```

환경은 Python 3.12.4, NumPy 2.5.3, Torch 2.14.0, arm64/Darwin이었다. Script는 engineering 데이터 구성, codec round-trip, metadata audit, 통계 산술·simulation, untrained sampler timing을 수행한다. 결과는 ignored local archive에 저장한다. 실제 학습·primary hash 평가를 실행하지 않는다.

관련 기존 tests의 실행 명령과 결과는 다음과 같다.

```sh
.venv/bin/python -m pytest -q \
  tests/test_bgv_byte.py tests/test_bgv_edge_cases.py \
  tests/test_bgv_mask.py tests/test_bgv_roundtrip.py tests/test_cgge.py \
  tests/test_dataset_evaluation.py tests/test_candidate_budget.py \
  tests/test_gaussian_discrete_protocol.py \
  -k 'not discrete_runner_engineering_smoke and not completed_training_is_checked_and_not_repeated'
```

**43 passed, 2 deselected in 1.01s.** 두 제외 항목은 실제 학습 runner를 호출하므로 이번 문서/설계 검증 범위에서 제외했다. 이 결과는 기존 컴포넌트 검증이며 v2의 전체 실행 경로가 구현됐다는 뜻이 아니다.

## 9. 최종 적합성 판정과 닫아야 할 항목

| 판단 대상 | 결론 |
|---|---|
| 연구 질문·범위·측정 기준을 가진 PoC 계획인가 | 예. v1보다 명확하며 대부분의 주요 과학 상수를 정의했다. |
| 알려진 archive 범위에서 2,048개 same-source target 구성이 가능한가 | 예. Source별 분할로 dry-run 확인했다. |
| 안정적인 두 배 효과를 검출하는 기준 시나리오가 있는가 | 예. 약 80.9%; 가정과 제한을 함께 명시해야 한다. |
| 작은 효과·모든 seed 변동에 충분한 설계인가 | 아니오. 검정력 민감도가 크다. |
| 현재 구현으로 바로 확증 hash PoC를 실행해도 되는가 | 아니오. G0/G1-B/G2와 최종 분석·자원 기록이 닫히지 않았다. |

다음 완료 기준은 확인 가능하게 정의되어 있다.

1. 최종 노출 inventory를 봉인하고 source별 available pool≥2,048 및 test overlap 0을 확인한다.
2. v2 전용 narrow generation path와 train-only shuffle, prefix accounting, streaming/resume를 통합하고 mutation/negative fixtures를 통과한다.
3. 다섯 pipeline×세 seed의 학습된 synthetic control 15개가 지정 문턱을 통과한다.
4. 실제 실험에 적용할 통계 모형·가정·claim 범위 및 민감도 한계를 analysis seal에 명시한다. 방어할 수 없으면 confirmatory claim을 중단한다.
5. 최종 장비·batch에서 학습/평가/저장량을 산정하고 per-run/study 시간·storage 상한을 채운다.

**진행해도 되는 다음 단계는 v2 구현 정합화와 학습 positive control이다. Primary hash 실험의 성능을 주장할 준비는 아직 되지 않았다.**

## 10. 판정 근거 요약 및 GPU 확인 후 보완 — 2026-09-24

| 판정 요소 | 관측 근거 | 허용되는 결론 | 아직 입증되지 않은 것 |
|---|---|---|---|
| 계획의 구체성 | q=12, K=100, N=2,048, 모든 seeds, 두 controls, max-p/Holm, 실패 처리 고정 | 개발할 수 있는 명시적 프로토콜 | 해당 설계로 모델이 우위를 보일지 |
| 데이터 구성 | Engineering seed로 두 source 모두 지정 quota 확보, 검사한 split overlap 0 | 알려진 노출 목록 아래에서 구성 가능 | 전체 과거 접근 감사, 최종 primary holdout 봉인 |
| 표현·기존 컴포넌트 | Codec 1,214건과 당시 관련 tests 43개 통과 | 확인한 codec와 계산 경로의 정합성 | 학습 모델의 생성 유효도·조건 활용 |
| 검정력 | 두 배 reference 80.940%; 약한 seed 23.695%; 강한 shuffled 3.025% | 안정적인 큰 효과를 찾는 제한된 PoC에 적합 | 작은 효과·seed 변동의 부재, 일반적 불가능성 |
| 실행기 | Inference shuffle 및 positive-control 입력 구조 불일치 재현 | 기존 runner의 v2 확증 실행 재사용 불가 | 완전한 v2 generation·통계·resume 통합 |
| GPU 실행 가능성 | M3 Max/MPS에서 실제 학습·추론, 다섯 pipeline checkpoint/RNG 복원 검사 통과 | 현재 장비의 GPU를 사용할 수 있음 | 전체 v2의 GPU 자원 예산과 실제 학습 positive-control 통과 |

따라서 **개발 PoC는 진행 가능하나 확증적 본실험은 준비 미완료**라는 판정은 유지한다. GPU 동작 확인은 공학적 실행 가능성을 보강하지만 독립 평가·통계 가정·측정 gate의 미완료를 해소하지 않는다.

GPU 작업에서는 별도 설정으로 작은 학습·추론 검증을 수행했다. 전체 tests는 108 passed, CUDA 장비가 없어 5 skipped였다. 이는 앞 절의 설계 검증 이후 수행한 추가 검사이며 v2 primary hash 평가가 아니다. 당시 원시 검증 기록과 source hashes는 그 시점의 기록으로 보존한다.

최초 프로세스의 MPS unavailable은 GPU 부재를 뜻하지 않았다. 호스트 프로세스에서 MPS available=True 및 실제 GPU 연산이 확인됐다. **§6의 약 117.7시간은 기존 CPU 정책의 추정치로만 유효하며 GPU 전체 실행 시간으로 해석하면 안 된다.** GPU 실행 방법·검사 기록과 제한은 [GPU 실행 문서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/GPU_EXECUTION.md)에 정리했다.

## 11. 최소 경로 구현 및 합성 pilot 후속 — 2026-09-24

12-bit-only 입력, train-only shuffle, epoch 학습과 validation-only checkpoint 선택, SQLite 후보 기록·재개를 별도 개발 경로에 구현했다. 최종 전체 tests는 120 passed, CUDA 부재로 5 skipped였다. MPS Discrete의 미세한 비결정적 차이를 발견해 결정적 연산을 강제하고 Gaussian/Discrete CPU/MPS 중간 학습 복구를 검사했다.

P-G-BGV·seed 0·Main에 대해 train 2,048개·20 epochs, validation/test 128개씩의 사전 고정 합성 pilot을 수행했다. 정상·반전 joint success는 각각 0/128, 전체 valid decode는 0/652였다. 주된 최초 거부 사유는 length_out_of_range 574건이었다. 정규 G1-B 학습량·표본 수를 사용하지 않은 개발 결과이므로 전체 모델 가설이나 다른 pipeline으로 일반화하지 않는다.

따라서 실행 경로의 공학적 증거는 보강됐지만 본실험 진입 판정은 여전히 미충족이다. 관측상 다음 병목은 header/mask를 포함한 구조 생성이다. 상세 결과와 모든 실행 기록은 [pilot 보고서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/PILOT_V2_REPORT.md)에 있다. 이전 절과 원시 검증 기록은 각 검증 시점의 이력으로 보존했다.
