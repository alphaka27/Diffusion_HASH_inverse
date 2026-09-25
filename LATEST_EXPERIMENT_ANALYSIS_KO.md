# 최신 실험 결과 및 실패 원인 분석

기준: 2026-09-25 로컬 저장 산출물. 최신 정식 study는 `v3-cli-p0-mps-final-20260925`이며, 디렉터리 이름과 달리 P0 이후 P1·P2·P3 실행 기록이 포함되어 있다. 분석 과정에서 학습·생성을 재실행하거나 원본 결과·실행 코드를 수정하지 않았다.

**결론: P0–P2의 기술 검증은 통과했지만, P3는 시간 예산 초과로 미완료다. 먼저 평가를 마친 Printable Gaussian 모델 6개는 유효 후보를 하나도 생성하지 못했다. 따라서 실행 중단과 생성 성능 미달이라는 두 문제가 동시에 존재한다. MD5 본실험의 실패로 해석할 수는 없다.**

## 1. 최신 실행 상태

| 단계 | 실제 상태 | 소요 시간 | 해석 |
|---|---|---:|---|
| P0 | COMPLETE / PASS | 1.13초 | 47 checks, codec round-trip 1,214개 통과 |
| P1 | COMPLETE / PASS | 86.36초 | 5 pipelines × Main/Shuffled 10 runs; Main의 학습·생성 중단/복구 검사 통과 |
| P2 | COMPLETE / PASS | 490.28초 | 5 models, 각 10 epochs·1,570 updates; 자원 profile 및 선택 batch 복구 검사 통과 |
| P3 | INCOMPLETE / exit 5 | 6,205.98초, 약 103.43분 | 15 models 중 6개 평가 완료, 1개 학습 중단, 8개 미착수 |
| MD5 본실험·통계 calibration | NOT_RUN | — | 본실험 CLI도 아직 미구현 |

P3는 **2026-09-25 17:20:26–19:03:52 KST**에 실행됐다. 중단 메시지는 `Per-run active wall-clock cap reached`다. Stage 전체 예산 13,680초에 도달한 것이 아니라 P-DISC seed 0의 개별 실행 상한에 도달했다.

근거: [실행 보고서](local_experiment_archive/runs/v3-cli-p0-mps-final-20260925/report.md), [P3 state](local_experiment_archive/runs/v3-cli-p0-mps-final-20260925/pilot/P3/state.json), [resources](local_experiment_archive/runs/v3-cli-p0-mps-final-20260925/resources.json).

## 2. 성능 결과

### P3 정규 평가

P=Printable, R=Random Bytes. 정규 문턱은 각 seed의 512 cases에서 정상 조건 성공 ≥475, 반전 조건 성공 ≥475, 원래 조건 오성공 ≤15다. 정상/반전 각각 K=1이며 성공은 유효한 메시지와 지정된 첫 세 nibble 조건을 함께 만족해야 한다.

| Pipeline | 완료 seed | 유효 후보 | 정상 / 반전 성공 | 판정 |
|---|---|---:|---|---|
| P-G-BGV | 0, 1, 2 | 0 / 3,072 | 각 seed 모두 0/512 / 0/512 | 완료한 세 seed 모두 부적격 |
| P-G-CGGE | 0, 1, 2 | 0 / 3,072 | 각 seed 모두 0/512 / 0/512 | 완료한 세 seed 모두 부적격 |
| P-DISC | 없음 | 미측정 | seed 0 학습 중단 | 평가 불가 |
| R-G-BGV | 없음 | 미측정 | 미착수 | 평가 불가 |
| R-DISC | 없음 | 미측정 | 미착수 | 평가 불가 |

합계 6,144개는 후보 원장의 기술적 집계이며 서로 독립인 통계 표본 수라는 뜻이 아니다. 미실행 모델의 결과는 0으로 채우지 않는다. P3 전체의 적격성 판정도 아직 완료되지 않았다.

### P2에서 이미 나타난 경고

각 pipeline은 128 conditions × 정상/반전 = 256 candidates를 평가했다.

| Pipeline | 유효 후보 | 정상 성공 /128 | 반전 성공 /128 |
|---|---:|---:|---:|
| P-G-BGV | 0/256 | 0 | 0 |
| P-G-CGGE | 0/256 | 0 | 0 |
| P-DISC | 53/256, 20.70% | 0 | 0 |
| R-G-BGV | 0/256 | 0 | 0 |
| R-DISC | 42/256, 16.41% | 0 | 1 |

다섯 pipeline 모두 `LEARNING_SIGNAL_ABSENT` 경고를 기록했다. P2 PASS는 실행·자원·복구 검증 통과이며 학습 능력 인증이 아니다. P2 성공률 0만으로 자동 중단하지 않는 것은 현행 계획의 명시적 정책이다.

근거: [P2 summary](local_experiment_archive/runs/v3-cli-p0-mps-final-20260925/pilot/P2/summary.json), P3 각 `runs/<pipeline>/<seed>/main/{complete,metrics,training}.json` 및 `candidates.sqlite`.

## 3. 실행 중단의 원인: 측정 범위와 제한 범위 불일치

**확인된 직접 원인:** P-DISC seed 0의 누적 실행 시간이 **242.4408초**가 되어, P2에서 봉인한 **242.4262초** 상한을 넘었다. 학습은 **10,421/15,700 updates, epoch 67**에서 중단됐다. 마지막 저장 checkpoint는 update 10,400이다. OOM이나 non-finite 오류로 중단됐다는 기록은 없다.

**코드에서 확인한 원인:** 시간 예측에 사용하는 `training_update_seconds`는 `budget.check()` 뒤부터 재므로 그 검사의 비용을 제외한다. 반면 실행 제한은 검사·상태 저장·기타 루프 비용을 포함하는 경과 시간에 적용한다.

- `Budget.check()`는 매 update마다 상태 JSON을 저장하고, study 전체와 현재 run의 파일을 재귀 순회해 저장량을 합산한다. 상태 저장에는 `fsync`도 포함된다.
- `profile()`은 학습 연산 측정값의 세 구간 평균을 사용한다.
- `seal_resources()`는 그 값에 validation·checkpoint·생성 비용을 더하고 1.5배를 곱하지만, 매 update의 budget 검사 및 telemetry 등 일부 루프 비용은 별도로 포함하지 않는다.

P-DISC 예산의 계산은 `(학습 150.535 + validation 0.859 + checkpoint 4.899 + 생성 5.324) × 1.5 = 242.426초`다.

실제 중단 시점의 학습 연산 합계는 **91.567초**, run 전체는 **242.441초**였다. 약 150.874초의 차이에는 검사·상태 저장·checkpoint·validation 등 부대 비용이 포함된다. 마지막 checkpoint까지 별도 기록된 validation은 0.482초, checkpoint 저장은 2.492초다.

분석 시점의 study는 1,025개 파일·약 464 MB였다. 읽기 전용 재귀 순회·파일 크기 합산을 두 차례 각 10회 측정한 중앙값은 **14.16ms와 14.69ms**였다. P3의 평균 학습 연산은 update당 **8.79ms**였으므로 파일 검사만으로 연산보다 큰 비용이 생길 수 있다. 이는 중단 당시의 직접 profiler 기록은 아니지만, 코드상 누락과 실제 시간 차이를 뒷받침한다. 개별 부대 비용의 정확한 기여율은 당시 기록만으로 분리할 수 없다.

Gaussian은 연산 시간이 상대적으로 길어 이 비용을 기존 여유분으로 감당했지만, 빠른 Discrete에서는 예산 과소평가가 드러난 것으로 해석된다. 같은 상한과 누적 시간을 유지한 `--resume`만으로는 해결되지 않는다.

근거 코드: [Budget](src/diffusion_hash_inv/study_pilot.py), 해당 파일의 242–280행, 학습 루프 410–444행, profile 821–831행, 자원 예산 841–850행.

## 4. 생성 성능 미달의 원인: 길이·마스크 구조에서 탈락

후보 원장을 직접 읽어 집계한 최초 거부 사유는 다음과 같다. Decoder는 먼저 발견한 오류에서 반환하므로 아래 비율은 모든 잠재 오류의 비율은 아니다.

| Pipeline | 최초 거부 사유 | 건수 | 해당 pipeline 후보 중 비율 |
|---|---|---:|---:|
| P-G-BGV | 길이 범위 위반 | 2,490 | 81.05% |
| P-G-BGV | 길이와 mask 불일치 | 551 | 17.94% |
| P-G-BGV | 길이 slot mask 무효 | 29 | 0.94% |
| P-G-BGV | padding 불일치 | 2 | 0.07% |
| P-G-CGGE | 비연속/범위 밖 mask | 2,937 | 95.61% |
| P-G-CGGE | glyph 거리 문턱 초과 | 135 | 4.39% |

**확인된 병목은 생성된 표현을 유효한 메시지로 복호화하는 단계다.** 완료한 6개 모델 모두 verifier 호출이 0건이므로, 목표 조건을 맞히는 능력은 이 평가에서 분리해 측정하지 못했다. P0의 clean codec 검증 통과와 생성 결과의 형식 실패는 양립한다.

손실은 감소했다. P-G-BGV의 선택 checkpoint validation loss는 seed별 0.00857 / 0.00936 / 0.00836, P-G-CGGE는 0.02232 / 0.02589 / 0.02392다. 그러나 모든 seed의 유효 생성률은 0%였다.

**추가 검증이 필요한 설명:** 현재 Gaussian 학습·checkpoint 선택은 전체 픽셀의 noise-prediction MSE를 사용한다. 길이 필드의 범위, 길이와 mask의 일치, mask의 연속성, 첫 세 문자 조건의 성공을 직접 선택 기준으로 삼지는 않는다. 따라서 noise loss 개선과 자유 생성의 구조적 정합성 사이에 간극이 생긴다는 해석이 결과와 일치한다. 구조·조건 주입·sampling 중 어느 변경이 이를 해결할지는 아직 실험으로 확인되지 않았다. MPS만의 결함, 조건 신호의 완전한 부재, diffusion 전체의 불가능성으로 단정할 근거도 없다.

P-DISC는 P3 validation loss가 epoch 20의 2.29745에서 epoch 60의 2.38142로 악화됐다. 단순히 epochs를 늘리면 성능이 해결된다고 볼 근거는 부족하다. 다만 P3 생성 평가 자체가 없으므로 최종 성능 실패로 판정하지 않는다.

근거 코드: [Gaussian loss](src/diffusion_hash_inv/models.py), [BGV decoder](src/diffusion_hash_inv/encoding/bgv.py), [CGGE decoder](src/diffusion_hash_inv/encoding/cgge.py), [P-DISC telemetry](local_experiment_archive/runs/v3-cli-p0-mps-final-20260925/pilot/P3/runs/P-DISC/0/main/telemetry.jsonl).

## 5. 다음 조치의 우선순위

1. **운영 비용부터 바로잡는다.** 모델 구조를 바꾸기 전에 전체 update 주기의 시간으로 자원 예측을 검증한다. 매 update의 전체 디렉터리 순회 빈도와 상태 저장 범위를 검토하되, 복구 보장과 자원 제한은 유지한다. 단순 상한 증액은 누락된 비용을 해결하지 않는다.
2. **별도 개발 revision에서 생성 구조를 진단한다.** Synthetic train/validation만 사용해 길이·mask·padding·glyph와 조건 prefix를 나누어 측정한다. Gaussian은 유효 생성이 선행 병목이며, Discrete도 P2에서 EOS 개수·EOS 뒤 PAD 규칙·payload 오류가 많았다. Decoder 문턱 완화나 생성 후 수선으로 기존 성능을 소급해 높이지 않는다.
3. **새 평가 범위를 명시한다.** P3 test 결과가 이미 관측됐으므로 이를 보고 수정한 실행은 기존 독립 검증의 단순 재개로 취급하지 않는다. 변경 이유·사용한 cases·새 검증 범위를 기록하고 영향을 받는 P0–P2부터 재검증한다. 기존 실패 산출물은 보존한다.
4. **MD5 본실험 진입은 보류한다.** 합성 과제 적격성뿐 아니라 노출 감사·통계 calibration·본실험 실행기 구현도 아직 필요하다.

이 문서는 위 조치를 제안하며 실제 코드 수정이나 새 실험을 실행하지 않았다.

## 6. 보고 및 증거의 한계

- `PILOT_V3_CLI.md`와 `RESEARCH_PLAN_V3.md`의 정식 P1–P3 미실행 표기는 이날 오전 상태다. 오후의 실제 실행 산출물이 더 최신이다.
- 자동 `report.md`에서 P3 판정이 `NOT_RUN`인 것은 P3 최종 gate가 없기 때문이다. 실행 상태는 `INCOMPLETE`이며 완료한 6개 run이 존재한다. 보고 코드가 P3 gate 완료 전에는 개별 결과를 출력하지 않아 부분 결과를 가린다.
- 과거 retired study의 G5 `INCONCLUSIVE` 및 G6 `NOT_RUN`은 다른 protocol의 결과다. 이를 이번 v3 성능과 합산하거나 v3의 적격성 증거로 사용하지 않았다.
- 분석 시 P0–P2 seal, 완료한 P3 6개 run seal, resources·데이터·protocol 체크섬을 검증했다. 현재 소스의 파일별 해시도 실행 manifest와 모두 일치했다. P3 후보 수와 거부 사유는 읽기 전용 SQLite 조회로 재집계했다.
- 분석 범위는 이 workspace에 저장된 산출물이다. 새로운 학습·표본 생성·통계적 우위 검정은 수행하지 않았다.
