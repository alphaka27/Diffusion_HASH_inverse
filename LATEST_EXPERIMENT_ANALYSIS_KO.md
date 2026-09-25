# 최신 실험 결과 및 실패 원인 분석

기준: 2026-09-25 로컬 저장 산출물. 최신 정식 study는 `v3-cli-p0-mps-final-20260925`이며, 디렉터리 이름과 달리 P0 이후 P1·P2·P3 실행 기록이 포함되어 있다. 분석 과정에서 학습·생성을 재실행하거나 원본 결과·실행 코드를 수정하지 않았다.

**추가 상세 분석:** §7–§10에는 저장 checkpoint로 수행한 CPU validation 복원·logit 진단을 추가했다. 새 학습, 정식 sampler 평가, P3 test 후보 생성은 수행하지 않았다. 특히 Discrete에서는 조건 학습 신호가 확인되어, joint success 0을 조건 학습의 완전한 부재로 해석하면 안 된다.

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

## 7. 추가 진단: Gaussian은 낮은 잡음 복원과 높은 잡음 복원이 다르다

기존 결과만으로는 형식 실패의 발생 위치를 충분히 좁힐 수 없어 다음 제한된 진단을 수행했다.

- P3 P-G-BGV·P-G-CGGE seed 0의 저장된 BEST checkpoint를 사용했다. 두 모델 모두 epoch 100이다.
- 저장된 Printable validation의 첫 16개 메시지에 고정 Gaussian noise(seed 20260925)를 더했다.
- t=0/100/300/500/700/999 각각에서 한 번의 denoiser forward로 clean image 추정치를 구했다. Decode에는 [-1,1] clipping을 적용했다.
- CPU float32로 검사했다. 이는 원래 MPS 실행과 동일한 backend 재현 검사가 아니며, validation 16개에 국한된 탐색 진단이다. 모델 파라미터는 갱신하지 않았다.

| Noise index t | BGV 복원 valid /16 | BGV exact /16 | CGGE 복원 valid /16 | CGGE exact /16 |
|---:|---:|---:|---:|---:|
| 0 | 16 | 16 | 16 | 16 |
| 100 | 16 | 16 | 16 | 16 |
| 300 | 15 | 13 | 2 | 1 |
| 500 | 5 | 0 | 0 | 0 |
| 700 | 0 | 0 | 0 | 0 |
| 999 | 0 | 0 | 0 | 0 |

낮은 잡음에서는 이미 입력에 원문 정보가 많이 남아 있다. 따라서 16/16 exact는 조건만으로 메시지를 생성할 수 있다는 증거가 아니다. 그럼에도 codec과 모델 경로가 모든 입력에서 완전히 고장 난 상태가 아니라는 점, 잡음이 커질 때 구조 복원이 급격히 나빠진다는 점은 확인된다.

### 7.1 Noise MSE가 작아도 구조 복원이 나쁜 이유

현재 epsilon parameterization의 수식은 다음과 같다.

`x0_hat = (x_t - sqrt(1-alpha_bar_t) * epsilon_hat) / sqrt(alpha_bar_t)`

알려진 clean image에서 만든 x_t에 대해서는 다음 오차 관계가 정확히 성립한다.

`MSE(x0_hat, x0) = ((1-alpha_bar_t) / alpha_bar_t) * MSE(epsilon_hat, epsilon)`

t=999에서 alpha_bar는 약 0.0000403583이다. Epsilon 오차의 진폭은 clean 추정치에서 약 157.4배, 제곱오차는 약 24,777배로 확대된다. 실제 진단에서도 아래 관계가 수치적으로 일치했다.

| 모델 | t=999 noise MSE | clipping 전 clean MSE | clean 추정치가 [-1,1] 밖인 좌표 |
|---|---:|---:|---:|
| P-G-BGV | 0.001060 | 26.263 | 84.81% |
| P-G-CGGE | 0.000857 | 21.238 | 83.92% |

참고로 같은 t에서 항상 clean=0을 암묵적으로 예측하는 `epsilon_hat=x_t/sqrt(1-alpha_bar)` 기준의 noise MSE는 약 0.00004036이다. 이 기준은 유효한 이미지를 만들지 못하지만, 높은 잡음에서 작은 noise loss 자체가 유효 생성의 증거가 아니라는 점을 보여준다.

이는 현재 공식 생성이 실패한 현상과 일치하는 진단이다. 다만 한 단계의 x0 추정과 100-step 반복 sampling은 다르다. 이 결과만으로 모든 실패를 특정 timestep이나 clipping 부재 하나에 귀속할 수는 없다. 높은 잡음에서는 원문의 임의 suffix를 정확히 식별할 수도 없으므로, exact 복원 실패 자체를 곧바로 생성 불가능으로 해석하지 않는다.

### 7.2 구조적 제약과 평균 loss의 간극

BGV는 32 slots 중 하나에 length byte를 넣고, 모든 slot의 mask·padding과 그 길이를 맞춰야 한다. CGGE는 mask가 처음부터 연속한 valid 영역이어야 하고 각 glyph도 허용 거리 안이어야 한다. 평균 pixel 오차와 이 불연속적인 record 유효성은 다른 척도다.

BGV 전체 tensor에서 length header의 glyph 영역은 1/64=1.5625%, 세 prefix glyph 영역은 3/64=4.6875%다. CGGE의 세 prefix glyph도 전체 tensor의 4.6875%다. 이 숫자는 좌표 비중이지 실제 loss 또는 gradient 기여도의 측정값이 아니다. 하지만 평균 loss 하나로 작은 필드의 정확성을 판단하기 어렵다는 구조적 이유를 설명한다.

또한 ImageUNet은 명시적 좌표 없이 공유 convolution과 공간 전체에 동일하게 더한 condition embedding을 사용한다. 위치별 역할과 전역적인 mask 연속성 학습이 어렵다는 가설은 타당한 검사 대상이다. 경계 효과와 반복 sampling이 정보를 전달할 수 있으므로 위치 학습이 불가능하다고 단정하지 않는다. 좌표 채널·더 넓은 문맥·objective 변경 중 무엇이 필요한지는 아직 대조 실험으로 확인하지 않았다.

## 8. 추가 진단: Discrete는 조건을 배우지만 그것만으로 생성 성공이 되지 않는다

저장된 validation 첫 64개 조건에 대해 모든 token이 MASK인 입력을 넣고 logits를 조사했다. 정답 prefix 자체는 입력에 들어가지 않으며 모델 입력은 MASK sequence, time=1, 12-bit condition뿐이다. P2 P-DISC/R-DISC의 BEST는 epoch 10, 중단된 P3 P-DISC의 BEST는 epoch 20이다.

| Checkpoint | 첫 세 token의 argmax 정답 수 /192 | 정답 token 평균 확률 | 조건 반전 후 원래 정답 token 평균 확률 |
|---|---:|---:|---:|
| P2 P-DISC | 169/192, 88.02% | 23.93% | 0.8413% |
| P2 R-DISC | 178/192, 92.71% | 25.91% | 0.6935% |
| P3 P-DISC BEST | 192/192, 100% | 82.63% | 0.0006634% |

P3 P-DISC에서는 검사한 64개 조건 모두 첫 세 token의 argmax가 맞았다. 조건을 반전하면 원래 정답 확률도 크게 감소했다. 이는 검사 범위에서 condition 경로가 작동하고 prefix 관계를 학습했다는 직접 증거다. Validation은 checkpoint 선택에 사용됐으므로 새로운 독립 test 결과로 표현하지 않는다.

### 8.1 정답이 argmax인 것과 정답을 sampling하는 것은 다르다

P2의 argmax 정확도는 높지만 정답 token에 배정한 평균 확률은 약 24–26%에 머문다. 실제 sampler는 argmax가 아니라 temperature 1의 categorical sampling을 사용한다. 따라서 정답이 가장 큰 logit이어도 자주 다른 token을 선택할 수 있다. 세 token의 관계와 반복 reveal 과정을 고려해야 하므로 평균 확률 세 개를 곱해 실제 joint 성공률로 보고하지 않는다.

이 차이는 P2에서 prefix 관계가 학습됐음에도 성공률이 낮을 수 있는 설명이다. P3 BEST에서는 정답 확률이 크게 높아져 같은 문제가 완화된 신호가 있지만, 공식 P3 생성 평가는 실행되지 않았다.

### 8.2 EOS/PAD는 메시지 전체의 제약이다

동일한 all-MASK logits를 모든 위치에서 argmax로 채웠을 때 P3 P-DISC의 64개 sequence는 **모두 EOS가 0개**였다. 정답 prefix 100%와 전체 형식 valid 0%가 동시에 관측됐다. 위치별 EOS 확률을 합한 기대 개수는 평균 약 0.792이고, 위치별 최대 EOS 확률의 평균은 약 0.071이었다. 이는 EOS에 확률을 전혀 주지 않는 상황과 다르며, EOS 위치에 확률이 분산된 상태다.

실제 P2 categorical sampling에서도 문법 오류가 대다수였다.

| P2 결과 /256 | P-DISC | R-DISC |
|---|---:|---:|
| EOS 개수 오류 | 136 | 133 |
| EOS 앞의 잘못된 payload token | 30 | 44 |
| EOS 뒤 non-PAD | 37 | 37 |
| 유효 sequence | 53 | 42 |

현재 학습은 가려진 위치의 CE 평균이며 EOS/PAD도 일반 token처럼 취급한다. Forward 하나에서 위치별 marginal logits를 만들고, reveal한 token은 나중에 수정하지 않는다. 정확히 한 EOS, 그 앞의 payload, 그 뒤의 PAD라는 전역 제약이 구조적으로 보장되지 않는다. 초기 오류가 고정되거나 여러 위치가 동시에 EOS를 고르는 경로가 오류에 기여할 가능성이 있다. 전체 reverse trajectory를 계측하지 않았으므로 특정 단계별 기여율은 아직 모른다.

**Argmax로 바꾸면 해결된다는 결론도 성립하지 않는다.** 위 argmax 진단은 원인 분리용이며 정식 32-step stochastic sampler의 결과가 아니다. 오히려 정답 prefix와 EOS 문법을 동시에 해결해야 함을 보여준다.

## 9. 실패가 늦게 드러난 이유와 배제할 수 없는 것

P0는 결정적 fixture, P1은 짧은 통합/복구 검사, P2는 finite 학습·진단·자원 측정 완료를 검사한다. 따라서 P2의 모든 normal joint가 0이어도 PASS가 가능하다. `LEARNING_SIGNAL_ABSENT`는 코드상 `normal_joint == 0`이라는 이름일 뿐, gradient나 조건 반응을 직접 검사한 판정이 아니다. 위 Discrete 진단은 이 명칭을 실제 조건 학습 부재로 읽으면 잘못된 결론에 이를 수 있음을 보여준다.

P3는 한 run의 예산 오류가 stage 예외로 전파되어 뒤의 run들도 수행되지 않았다. 보고서도 P3 gate 완료 전에는 개별 결과를 표시하지 않는다. 이 두 동작이 성능 실패 자체를 만들지는 않았지만, 이후 모델의 평가 기회를 없애고 이미 완료한 실패 결과를 가렸다.

다음 항목도 구분해야 한다.

- **현재 epsilon 학습/sampling parameterization 불일치:** 이번 실행은 epsilon 경로끼리 일치한다. 기존 epsilon oracle 검사 및 Pilot 단일 후보와 기본 sampler의 일치 검사를 재실행해 2 passed를 확인했다. 작은 fixture 통과가 전체 sampler의 완전한 무결성을 증명하지는 않는다.
- **x0로 한 줄 변경할 때의 위험:** Pilot validation/sampler의 epsilon 가정 때문에 향후 변경 시 수정이 필요하다는 뜻이지, 이번 epsilon 실험의 확인된 bug라는 뜻은 아니다.
- **데이터 손상/잘못된 합성 라벨:** 앞선 21,024개 train/validation 검사에서 위반이 없었고, 이번에도 데이터·코드·checkpoint 및 완료 산출물의 체크섬이 일치했다. 모든 종류의 데이터 설계 문제를 배제한 것은 아니다.
- **수치 폭주/OOM:** 공식 중단 원인은 시간 상한이다. 완료 run에서 NaN/Inf 오류 기록은 없다. 높은 잡음의 부정확한 finite x0 추정과 nonfinite crash는 구분한다.
- **단순 학습 부족:** Gaussian 6개는 각각 100 epochs를 마쳤고 valid 0이다. Discrete의 validation loss는 epoch 20 이후 악화됐지만 prefix 조건 지식은 존재한다. 더 오래 학습하는 것만으로 해결된다는 근거는 없다.
- **MD5 역상 학습 불가능:** 이번 대상은 synthetic_nibbles다. MD5 본실험은 수행하지 않았으며 그 결론을 낼 수 없다.

## 10. 상세 진단의 결론과 재현

확정된 운영 원인은 **예산 측정에서 누락된 부대 비용**이다. 확인된 생성 병목은 **Gaussian의 length/mask/glyph 정합성**과 **Discrete의 EOS/PAD 및 stochastic joint generation**이다. 추가 진단은 Gaussian의 고잡음 복원 취약성과 Discrete의 실제 조건 학습을 구분해 보여준다. 따라서 하나의 원인으로 모든 pipeline을 설명하거나 조건 학습 실패로 묶어서는 안 된다.

우선순위는 전체 실행 시간 측정 수정, 형식·조건·noise 수준별 지표 분리, 그리고 원인별 작은 대조 실험이다. 특정 구조나 objective로 바꾸면 해결된다는 인과적 증거는 아직 없다.

재현 파일: [진단 코드](local_experiment_archive/analyses/pilot-failure-detail/diagnose.py), [상세 JSON](local_experiment_archive/analyses/pilot-failure-detail/diagnostics.json). JSON에는 데이터·checkpoint·진단 코드 SHA-256, sample 수, device, noise seed, 지표가 기록돼 있다. 코드에는 clean/epsilon MSE의 수식 관계를 확인하는 assert가 포함된다.

재현 명령: 프로젝트 루트에서 `.venv/bin/python local_experiment_archive/analyses/pilot-failure-detail/diagnose.py`. 동일 진단 JSON을 다시 쓰며 봉인된 study와 모델 weights는 변경하지 않는다.
