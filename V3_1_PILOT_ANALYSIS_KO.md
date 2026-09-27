# 최신 v3.1 Pilot 분석 및 다음 실험 진행 판단

분석일: 2026-09-27 KST. 대상: `local_experiment_archive/runs/v31-mlx-20260926-232405`.

**판정: 다음 실험은 P2A가 타당하다. 다만 P2A/B 실행기가 미구현이므로 지금 즉시 실행할 수는 없다. 구현·검증 후 작은 과적합 검사로 진행하고, P3 및 MD5 본실험은 보류한다.** 이번 결과는 개발용 P0/P1 기술 검사 통과이며, 정식 `TECHNICAL_READY`, `POC_QUALIFIED`, `MAIN_READY`를 충족하지 않는다.

## 1. 실제 실행 범위와 증거 검증

실행은 MLX 0.32.2 / Metal GPU / float32 / seed0이며, 수정되지 않은 v3.1 명세를 사용했다. P0/P1 모두 `--development`로 실행했다. P1은 축소 테스트가 아니라 명세에 등록된 P1 전체 규모다. 다만 P1 자체가 모델당 256개 학습 자료·2 epochs·8 updates의 짧은 기술 검사다.

| 항목 | 결과 |
|---|---|
| P0 | COMPLETE / PASS, active 4.58초, 검사 75개, codec 왕복 1,214개 |
| P1 | COMPLETE / PASS, active 139.76초, 13개 pipeline/profile × Main/Shuffled = 26 runs |
| 학습 | 26개 모두 8 updates 완료; validation loss는 모두 유한하고 epoch1→2 감소 |
| 중단·복구 | 13개 Main 조합 모두 update5 및 attempt7 commit 전 복구 PASS 기록 |
| 평가 | learned 4,160개 + source별 공유 Random 320개 = 4,480개 |
| 평가 NFE | learned 329,600회; 복구 재실행·P0·학습 연산은 이 수치에 포함하지 않음 |
| 후속 단계 | P2/P3 미실행, E0·본실험·통계 calibration 미실행 |

이번 분석에서 P0/P1 gate의 봉인 파일 **1,040개**를 SHA-256으로 재검증했으며 불일치는 없었다. 현재 source 파일 hash, synthetic data hash, frozen protocol과 현재 명세의 내용도 일치했다. 원 평가 SQLite 28개를 읽기 전용으로 열어 integrity check, 각 run의 16 trials × 10 attempts, normal variant, 성공 재계산 및 집계와 원본 metrics 일치를 확인했다. **4,480개 모두 검증을 통과했다.** 복구 PASS는 보존된 검사 결과를 확인한 것으로, 이번 분석에서 GPU 복구 실험을 재실행한 것은 아니다.

근거: [실행 보고](local_experiment_archive/runs/v31-mlx-20260926-232405/report.md), [P1 상세 집계](local_experiment_archive/runs/v31-mlx-20260926-232405/pilot/P1/summary.json), [구현 상태](local_experiment_archive/runs/v31-mlx-20260926-232405/implementation_readiness.json).

## 2. 생성 품질: 형식 유효성과 조건 학습을 구분해야 한다

아래 표는 Main/Shuffled와 해당 source/pipeline을 합산한 기술 진단이다. 서로 다른 모델의 후보를 독립 표본으로 합쳐 통계적 우위를 검정한 결과가 아니다. Joint는 strict valid이면서 요청한 synthetic prefix를 맞힌 후보 수다.

| Profile | 후보 수 | Strict valid | 정상 joint |
|---|---:|---:|---:|
| G0 | 960 | 0 / 960 (0%) | 0 |
| G1 | 960 | 0 / 960 (0%) | 0 |
| G2 | 960 | 0 / 960 (0%) | 0 |
| D0 | 640 | 2 / 640 (0.3125%) | 0 |
| D1 | 640 | 640 / 640 (100%) | 0 |
| Random | 320 | 320 / 320 (100%) | 0 |

**Gaussian은 현재 형식 생성 단계에서 막혀 있다.** BGV는 길이 범위·길이 slot·mask 오류가 나타났고, BGV/G2는 모든 후보가 `length_out_of_range`였다. CGGE는 G0/G1/G2 모두 `mask_inconsistent`였다. Clean codec 왕복은 통과했으므로 이것을 codec 자체의 왕복 실패로 해석할 수는 없다. 생성된 표본이 strict 형식을 만족하지 못한 결과다. 8 updates만으로 학습 부족, 표현의 난도, sampler와 학습의 문제 중 어느 것이 주원인인지 확정할 수 없다.

**D1은 형식 문제를 해결했지만 조건 학습은 입증하지 못했다.** D0 Main은 두 source 모두 0/160 valid이며, Shuffled에서만 각각 1/160 valid였다. D1은 Main/Shuffled·두 source 모두 160/160 valid다. 길이를 먼저 뽑고 EOS/PAD를 고정하는 설계의 의도와 일치하지만, 640개 모두 조건 일치에는 실패했다. 형식 유효성 100%를 synthetic 과제 적격성으로 바꾸어 해석하면 안 된다.

모든 learned run의 Success@1/10도 0이다. 그러나 P1은 품질 문턱이 없는 기술 검사이므로 이는 P1 PASS와 모순되지 않으며, 학습 불가능성의 증거도 아니다. Validation loss 감소만으로 조건 사용을 입증할 수도 없다. 특히 G2는 x0 objective, G0/G1은 epsilon objective, D1은 길이 CE를 포함하므로 profile 사이의 loss 절댓값으로 우열을 정하면 안 된다.

**보고서의 ‘반전 joint 0’은 미측정으로 읽어야 한다.** P1 명세와 4,480개 ledger rows 모두 normal variant만 포함한다. 반전 0건은 반전 실험 실패가 아니라 빈 집계의 0이다. 후속 보고에서는 `NOT_MEASURED`로 구분할 필요가 있다.

또한 이 평가는 `synthetic_nibbles`이며 모든 평가 row의 MD5 호출 수는 0이다. P0의 MD5 fixture 통과와 별개로, 이번 P1에서 MD5 역상 성능을 측정한 것은 아니다.

## 3. 다음 단계별 판단

| 다음 작업 | 판단 | 이유 및 조건 |
|---|---|---|
| P2A/B 실행기·진단 구현 | 진행 | 개발 P0/P1과 복구 경로가 통과했고, 남은 핵심 질문은 실제 조건부 생성의 학습 가능성 |
| P2A 작은 과적합 실험 | 구현·검증 후 진행 | 현재 CLI가 v3.1 P2 실행을 명시적으로 차단함 |
| P2B 100-epoch 개발 실험 | 후보별 P2A 통과 후 | 형식 유효성만으로 건너뛸 수 없음 |
| P3 최종 적격성 | 보류 | P2 선택, E0, 자원 봉인, exposure 감사, calibration 등이 없음 |
| MD5 본실험 | 보류 | production 경로 미구현 및 다섯 pipeline의 P3 적격성 미확보 |

다음 실험은 [계획 §6](RESEARCH_PLAN_V3_1.md#6-p0p3-및-e0-실행-절차)에 고정된 **P2A**다. Source별 training split에서 정한 8 complement pairs / 16개 사례, seed99, batch16, 2,000 updates, 최종 checkpoint를 사용한다. 정상·반전 각각 64회 중 strict joint **61회 이상**, 반전의 원래 조건 오성공 **1회 이하**가 통과 기준이다. P1 checkpoint를 이어 학습하지 않는다.

Pipeline별 G0→G1→G2, D0→D1의 등록 순서를 유지하고, P2A→P2B→자원 조건을 모두 통과한 첫 profile을 선택한다. 이번 P1 결과만 보고 D1/G2를 미리 선택하거나 G0/D0를 제외하지 않는다. P2B는 fresh seed100으로 100 epochs를 완료하고, 마지막 개발 평가에서 정상·반전 joint와 valid가 각각 122/128 이상, 원래 조건 오성공이 3/128 이하여야 한다. 모든 등록 후보가 실패하면 `BLOCKED_DEVELOPMENT`로 남기고 revision을 새로 정한다.

구현에는 형식 오류, prefix 위치별 정확도, valid일 때의 조건 정확도, normal/flipped joint, 원래 조건 오성공을 분리한 진단이 필요하다. 계획 §11에 따라 실제 MD5 실행 경로·감사·통계 구현도 선행 작업으로 진행하고, E0와 자원·감사·calibration을 확정한 뒤 P3에 진입한다.

## 4. 자원 및 실행상의 제한

P1 active time 139.76초는 1,800초 hard cap의 약 7.8%이며 이번 실행에서 시간 상한은 장애가 아니었다. 분석 시 보존 디렉터리 크기는 약 346.84 MiB다. 다만 짧은 P1 측정으로 P2/P3/본실험의 시간·메모리·저장량을 보장할 수 없다. 선택 profile의 전체 주기 측정과 실제 MD5·SQLite 비용을 포함한 E0 추정이 필요하다.

현재 구현 문서의 ‘원본 규모 P1은 실행하지 않았다’는 설명은 문서 작성 시점의 기록이며, 이번 최신 실행으로 개발 P1 원본 규모 완료 증거가 추가됐다. 정식 PoC 완료라는 뜻은 아니다.

P2 구현으로 source hash가 바뀌면 기존 봉인 workdir에 이어 쓰지 않는다. 기존 결과를 보존하고 새 workdir에서 변경된 코드의 P0/P1을 검증한 뒤 진행해야 한다. 이번 분석은 원 실행 결과와 코드를 변경하거나 다음 실험을 시작하지 않았다.
