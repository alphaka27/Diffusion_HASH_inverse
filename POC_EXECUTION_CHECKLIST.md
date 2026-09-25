# 개발 PoC와 본실험 실행 준비 사항

기준일: 2026-09-24. 기준 문서는 [연구 계획 v2](/Users/choisoonwook/Experiments_local/DHI_AI_gen/RESEARCH_PLAN_V2.md), 현재 증거는 [검증 보고서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/RESEARCH_PLAN_V2_VALIDATION.md)와 [GPU 실행 문서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/GPU_EXECUTION.md)다. 이 체크리스트는 실행 준비를 정리하며 v2의 과학적 고정값을 변경하지 않는다.

여기서 **개발 PoC**는 측정·학습·생성 경로가 가설을 시험할 수 있는지 검증하는 단계다. 독립 engineering 자료와 합성 과제를 사용한다. **본실험**은 봉인한 source별 q=12 평가 집합에서 모델과 두 대조군의 우위를 검정하는 v2 확증 실험이다. Full MD5 역산 연구를 뜻하지 않는다.

v2 pilot 구현과 실행 결과는 사용자 요청으로 삭제했다. 현재 재사용 가능한 기반은 기존 모델·codec·GPU 지원이며, v2 최소 실행 경로와 pilot은 다시 구현·검증해야 한다. 본실험 진입 조건은 충족되지 않았다.

## 1. 이미 확보한 기반

| 항목 | 확인한 증거 | 재사용 범위 |
|---|---|---|
| GPU 실행 | M3 Max/MPS에서 Gaussian·Discrete optimizer update 및 sampling 성공 | 기존 모델 primitives와 GPU 연결 |
| 다섯 pipeline | CPU/MPS 학습·완료 checkpoint 복원·RNG 복원 후 sample 일치 | 작은 engineering test 범위 |
| 관련 소프트웨어 | 전체 108 tests 통과, CUDA 부재로 5개 제외 | 현재 소스의 검사된 동작 |
| Codec | Clean round-trip 1,214건 통과 | 해당 correctness corpus |
| Dataset 구성 | 두 source에서 train 10,000 / validation 512 / test 2,048 확보 | Engineering seed와 알려진 노출 목록 아래의 구성 가능성 |
| 통계 산술 | Exact McNemar 작은 정수 fixture, Holm·missing 처리, power simulation | 계산 구현과 특정 모형의 simulation |

기존 GPU smoke는 legacy runner를 사용한다. 그 결과만으로 v2의 12-bit 입력 경로, epoch 학습·validation 선택, 합성 positive control의 완성을 주장할 수 없다. 현재 남은 과제는 아래 상태표를 기준으로 판단한다.

## 2. 개발 PoC에서 완료할 작업

| ID | 필요한 작업 | 완료를 확인할 증거 | 현재 상태 |
|---|---|---|---|
| D1 | v2 설정을 실제 실행 경로에 연결 | 다섯 pipeline 모두 condition dimension=12, 지정 architecture·loss·schedule·precision 사용; 설정 불일치 시 실행 차단 | 재구현·검증 필요 |
| D2 | 모델 입력에서 평가 원문과 metadata 분리 | 원문·길이·digest suffix·ID·padding metadata를 바꾸어도 고정 target/RNG의 생성물이 불변인 mutation 검사 | 재구현·검증 필요 |
| D3 | Main/Shuffled 학습·추론 분리 | 학습 epoch마다 train 안에서 donor permutation; validation/inference에는 실제 y; Main과 초기화·순서·예산 일치 | 재구현·검증 필요 |
| D4 | v2 학습·checkpoint 선택 구현 | Train 10,000개, batch 64, 100 epochs, 마지막 batch 유지; 10 epochs마다 고정 validation noise로 loss 비교, 최소 loss/이른 tie 선택 | 재구현·검증 필요 |
| D5 | 후보 예산과 독립 평가 연결 | 한 target당 정확히 100 attempts; @1/@10/@100은 동일 stream의 prefix; invalid·중복·성공 후 시도도 계수; 검증은 payload bytes에 적용 | 재구현·검증 필요 |
| D6 | Streaming 저장과 중단·재개 구현 | 중간 학습 및 후보 기록 경계에서 강제 중단해도 완료 rows와 난수 순서 보존; 누락·중복 0; 실패·미완료 상태 구분 | 기존 checkpoint 기반만 보존; v2 통합 검증 필요 |
| D7 | v2 합성 positive control 학습 | 아래 문턱을 다섯 pipeline×세 seeds가 모두 충족하는 held-out generation 기록 | Pilot 결과 삭제; 정규 15개 미실행 |
| D8 | 전체 평가·통계 경로의 작은 통합 검사 | 성공·실패·invalid·missing이 포함된 고정 fixture에서 6개 component→max-p→5개 Holm, CI와 상태 출력 일치 | 일부 계산만 확인 |
| D9 | 실제 GPU 자원 측정 | v2 dimensions로 학습·validation·sampling·decode·hash·IO 시간과 메모리/디스크 측정; 실행 상한 기록 | 기존 GPU smoke만 보존; 전체 자원 예산은 미확정 |

D4에서 Gaussian은 width 32, epsilon prediction, T=1,000 및 100-step sampling, Discrete는 width 128/embedding 16 및 32-step sampling을 사용한다. Optimizer·고정 seed·validation corruption 등 세부값은 v2와 protocol JSON을 따른다. Legacy의 step 기반 학습과 259차원 canonical condition을 v2 구현으로 오인하지 않는다.

D6은 완료 checkpoint 로드뿐 아니라 **실행 중간의 optimizer·epoch permutation·noise RNG·후보 위치 및 저장 transaction 복구**를 다시 검증해야 한다. Raw tensor는 v2 계획에 따라 run별 첫 16개 target의 첫 candidate만 보존하도록 구현한다.

### 합성 positive control의 정확한 통과 기준

12-bit 조건의 세 nibble을 첫 세 payload 위치에 대응시키는 합성 과제를 사용한다. Hash 모델과 같은 condition 경로·architecture·training profile·sampler·codec을 사용한다. 원문 image나 전체 record를 조건으로 주는 legacy G1은 이 검사를 대체하지 않는다.

- Train/validation/test conditions는 3,072/512/512개다. `(y, y XOR 4095)`를 같은 split에 넣어 반전 조건도 held-out을 유지한다.
- 다섯 pipeline의 seeds 0/1/2, **총 15개 control 모델**을 학습한다.
- 각 seed에서 정상 조건의 valid AND constraint success에 대한 양측 95% Wilson 하한 ≥ 0.90.
- 반전 조건으로 생성했을 때 반전 조건 자체의 동일 성공률 하한 ≥ 0.90.
- 반전 생성물이 원래 조건을 만족하는 비율의 Wilson 상한 ≤ 0.05.
- N=512에서는 앞의 두 항목 각각 최소 475건, 마지막 항목 최대 15건에 해당한다. K=1이며 후보를 고르거나 재생성하지 않는다.

Control 실패 시 해당 pipeline의 hash 본실험은 차단한다. Engineering 자료에서 원인을 수정하고 v2.x 변경을 기록한 뒤 영향받은 control을 다시 검증한다. 실패한 seed를 제외하지 않는다.

개발 PoC에서는 model-training smoke, q=8의 별도 engineering fixture, synthetic control을 사용할 수 있다. Primary q=12 test success로 architecture·N·K·seed를 조정하지 않는다. 구조·sampler·학습값 변경이 필요하면 변경 이력과 control 재검증 범위를 먼저 남긴다.

## 3. 본실험 진입 전에 필요한 조건

| ID | 진입 조건 | 필요한 기록·산출물 | 진행하지 못하는 경우 |
|---|---|---|---|
| E1 | 전체 노출 감사와 source별 test pool 확정 | 과거 평가·validation 및 추가 경로의 inventory, 배제 prefix 목록·근거·hash | 어떤 source든 가용 prefix가 2,048개 미만 |
| E2 | 과학적 설정 및 분석 계획 고정 | Protocol/config/code hashes, 목표 모집단, 통계 가정, family, missing 처리, 민감도와 주장 범위 | McNemar working model을 방어하지 못하거나 작은 효과 배제가 연구 목적일 때 |
| E3 | Primary dataset 생성·봉인 | Source별 train 10,000 unique messages, validation 512/test 2,048 unique digests, ownership·dataset hashes, overlap 0 보고서 | 중복·노출·구성 실패 또는 미봉인 |
| E4 | 측정 경로 gate 완료 | G0 정보 경계·데이터 검사, G1-A codec 검사, G1-B 실제 학습 controls, G2 engineering 통합 검사 | 해당 pipeline의 필수 gate 미충족 |
| E5 | GPU 운영 예산 고정 | MPS/backend·versions·최종 batch·시간/저장 상한·재개 정책·telemetry 기록 | 처리량·자원 상한·복구 경로 미확정 |
| E6 | Test 평가 전 모든 적용 checkpoint 봉인 | 세 seeds의 Main/Shuffled 학습 완료, validation-only 선택 결과·checkpoint hashes·test 접근 기록 | Test를 보고 checkpoint나 설정을 다시 선택 |

E1/E2는 비용이 큰 control 학습에 앞서 확인하는 것이 효율적이다. 현재 알려진 가용 pool은 Printable 2,211개, Random Bytes 2,236개로, 요구량 대비 여유는 각각 **163개·188개**다. 추가 감사에 따라 부족해질 수 있다. 새 seed를 고르는 것만으로 이미 노출된 target이 미노출로 바뀌지는 않는다.

Source별 미노출 정의를 사용하며 다른 source에서 본 같은 prefix까지 제외한 project-wide novelty는 주장하지 않는다. q<12 자료의 원문·full digest·모델 출력 사용 여부도 감사 범위다. Engineering dry-run을 primary dataset으로 승격하지 않는다.

E2에서는 독립 target pairs와 discordant-direction exchangeability 등 v2 §11의 충분한 working model을 명시한다. 단위검사나 max-p/Holm 보정이 이 가정을 입증하지 않는다. 방어가 어렵다면 test 공개 전에 분석을 개정하고 power를 재검토하거나, 기술적 결과만 보고하는 연구로 범위를 바꾼다.

G2는 본실험 전 engineering fixture에서 경로를 검증하고, 실제 평가가 끝난 뒤 실제 ledger에도 다시 적용한다. G3(통계 결과)와 G4(모든 seed 완료)는 본실험 수행 후 판단할 항목이며 착수 전에 PASS를 요구하지 않는다.

## 4. 본실험 실행 규모와 완료 기준

| 구분 | v2 고정 규모 |
|---|---:|
| Pipeline | 5개: P-G-BGV, P-G-CGGE, P-DISC, R-G-BGV, R-DISC |
| Primary hash model 학습 | 5 × 3 seeds × 2 methods = **30 runs** |
| 합성 control 학습 | 5 × 3 seeds = **15 runs**; 본실험 30개와 별도 |
| 평가 표적 | source별 고유 12-bit digest **2,048개**; same-source pipeline 공유 |
| 후보 예산 | target·method·seed당 **100 attempts** |
| Learned candidates | **6,144,000개** |
| Source·seed별 공유 Random candidates | **1,228,800개** |
| Learned + Random ledger | **7,372,800 rows**; control·validation 제외 |
| Learned sampling NFE | **447,283,200** candidate-level model evaluations |

최초 control과 모든 본실험을 수행하면 최소 45개 학습 runs가 필요하다. 개발용 재시도·benchmark·validation 비용은 추가다. 현행 batch/epoch를 그대로 적용하면 run당 optimizer updates는 15,700회다. NFE는 GPU 호출 수 또는 hash 호출 수와 같지 않다.

GPU benchmark는 primary target 성능을 열지 않고 고정된 engineering 입력으로 수행한다. 학습 batch=64와 float32는 계획 고정값이며, 실제 처리량을 확인하기 위한 inference batch와 저장 정책은 평가 전에 고정한다. 학습 batch·precision을 바꾸어야 한다면 단순 운영 변경으로 처리하지 말고 protocol 변경 여부와 재검증 범위를 기록한다. 전체 소요 시간은 pipeline별 측정값에 workload를 곱하고 validation·control·decode·hash·IO 및 운영 여유를 별도로 더해 산정한다. 기존 약 118시간 CPU 추산은 GPU 예산이 아니다.

실제 평가에서는 모든 적용 seed를 결과와 무관하게 수행한다. Pipeline별 두 controls×세 seeds의 6개 component p-value를 max-p로 결합하고, 고정한 다섯 pipeline family에 Holm을 적용한다. Component가 빠지면 보정에는 p=1을 넣되 outcome을 0으로 채우지 않는다.

실험 **완료**와 모델 **성공**은 다르다. 모든 계획된 실행과 무결성·자원 기록·분석이 끝났지만 유의한 우위가 없을 수도 있다. 성공 주장은 측정 gates 완료, 가정의 명시, 여섯 효과 모두 양수, composite Holm adjusted p<0.05 및 세 seeds 완전 실행을 요구한다. 비유의 결과로 작은 효과나 일반적인 학습 불가능성을 단정하지 않는다.

Pipeline 일부가 blocked여도 나머지가 자신의 선행 조건을 충족하면 실행할 수 있다. 다만 family 5를 유지하고 제외 이유를 남기며, 다섯 pipeline 전체 연구가 완결됐다고 표현하지 않는다.

## 5. 권장 실행 순서

1. **노출 감사·분석 범위 확인:** 2,048개 same-source test pool과 확증적 주장 가능성을 먼저 확인하고 과학적 설정을 고정한다.
2. **v2 경로 구현:** 12-bit 정보 경계, train-only shuffle, epoch 학습/validation 선택, 100-prefix 생성·streaming·복구를 연결한다.
3. **작은 통합 검사:** Primary test와 분리한 자료로 G0/G1-A/G2 검사와 통계 fixture를 통과한다. 설정 변경 시 변경 이력을 갱신한다.
4. **학습 control 실행:** 최종 예정 구조·학습값으로 15개 synthetic controls를 수행하고 pipeline별 적격성을 판정한다.
5. **GPU 예산·실행 manifest 확정:** 최종 처리량·메모리·저장량·시간 상한·재개 정책을 기록한다. 초기 소규모 profile은 4단계 전에도 수행할 수 있다.
6. **Primary data와 학습 결과 봉인:** Final corpus의 G0를 확인하고 적격 pipeline의 Main/Shuffled를 모든 seeds에서 학습·validation 선택한다.
7. **한 번의 본 평가와 보고:** 고정 checkpoint로 100-attempt streams를 생성하고 실제 G2, G3/G4, CA0 및 모든 실패·미완료를 보고한다.

다시 개발을 시작한다면 **D1–D6의 최소 실행 경로 구현과 E1–E2의 데이터·추론 조건 확인**이 필요하다. 현재 체크리스트는 삭제한 pilot 결과를 본실험 준비의 증거로 사용하지 않는다.
