# v3.1 구현 현황

2026-09-26 · **첫 구현 묶음 완료 / 전체 IMPLEMENTATION_READY는 아직 아님**

[실험 계획](RESEARCH_PLAN_V3_1.md)의 모델 및 실행 기반을 구현했다. 현재 실행 가능한 범위는 **개발용 P0/P1과 보고서**이며, 정식 PoC 적격성이나 본실험 완료를 뜻하지 않는다. 기존 v3 자료·checkpoint는 변경하지 않았다.

| 영역 | 구현 상태 |
|---|---|
| G0/G1/G2 | 기존 epsilon 모델, 고정 좌표 입력, x0 예측 및 sampling 정합성 구현 |
| D0/D1 | 기존 Token 모델, 길이 head + 길이 조건부 payload diffusion 구현 |
| Profile 식별 | 13개 pipeline/profile 조합, 초기화·checkpoint·ledger 식별 및 NFE 기록 |
| 개발 P0 | Codec·모델·sampler·입력 경계 등 75개 검사; 실제 MD5 본실험 경계 검사는 남음 |
| 개발 P1 | 26개 learned runs와 2개 shared Random streams; update5 및 attempt7 중단/복구 검사 |
| 예산 기반 | v3.1 soft 경고/hard 중단 분리, 누적 PoC 시간, 쓰기 예약 및 제한된 디렉터리 검사 |
| 시간 측정 | Budget check·입력 준비·연산·telemetry를 포함한 update 전체 시간 기록; validation/checkpoint 별도 |
| 부분 보고 | Stage gate가 없어도 완료 run의 seal을 검증하고 COMPLETE/INCOMPLETE/NOT_RUN 표시 |
| P2A/P2B | 미구현: 작은 과적합 검사, 개발 probe·세부 진단, 순차 profile 선택 |
| 자원 확정·E0 | 미구현: 전체 비용 추정/봉인, 노출 감사, production MD5 리허설 |
| P3·M0–M3·calibration | 미구현; 정식 실행 차단 |
| Run 단위 오류 후 독립 실행 계속 | 미구현; 현재 실패 시 부분 결과를 보존하고 stage 중단 |

현재 CLI는 수정되지 않은 v3.0/v3.1 JSON만 받는다. v3.1의 정식 실행과 P2/P3 실행 요청은 출력 디렉터리를 만들기 전에 차단한다. Protocol JSON의 `*_at_authoring` 및 `implementation_status`는 계획 작성 당시 기록이며, 현재 구현 상태는 코드의 `readiness()`와 실행 디렉터리의 `implementation_readiness.json`에 기록한다.

## 검증 결과

- 전체 pytest: **124 passed, 5 skipped**. CPU와 native MPS 모두 새 13개 설정의 학습·생성·exact recovery 검사를 포함한다. 학습/평가 규모와 sampling steps를 줄인 내부 fixture이며 정식 P1 결과가 아니다.
- 수정되지 않은 v3.1 명세의 개발 P0: **CPU/MPS 각각 PASS**, 검사75개·codec 왕복1,214개·모델 설정13개. 학습은 수행하지 않았다.
- G2는 clean sample을 validation target으로 사용하며, 별도 oracle 검사로 epsilon target과 혼동하지 않는 것을 확인했다.
- D1은 모델이 뽑은 길이만 생성에 사용하고, 길이/payload RNG·생성 길이를 기록한다. EOS/PAD 고정 context와 payload vocabulary 제한을 사용하며 길이 CE는 payload mask가 없어도 계산한다.
- 서로 다른 batch 크기의 Gaussian 비교는 `atol=1e-3, rtol=1e-4`와 동일 decoder 결과를 요구한다. 100-step epsilon 복원에서 Float32 오차가 증폭되기 때문이다. **단일 샘플 기준 구현과 fixed-batch 복구 비교는 완전 일치를 유지한다.** 이 허용오차는 성능 gate를 변경하지 않는다.
- 검증 중 처음 설정한 batch 오차 허용치가 너무 작아 CPU P0가 한 차례 실패했다. 해당 결과는 보존했다. 장치 잠금과 충돌한 병렬 테스트 실행 후, 최종 전체 검사는 단독 실행하여 통과했다.

증거: [CPU P0 보고서](local_experiment_archive/runs/v31-implementation-cpu-p0-20260926-r2/report.md), [MPS P0 보고서](local_experiment_archive/runs/v31-implementation-mps-p0-20260926/report.md), [전체 테스트 XML](local_experiment_archive/analyses/v31-implementation-tests-20260926.xml). 실행 증거는 gitignored local archive에 보관한다.

## 현재 사용 가능한 명령

저장소 루트에서 쓰기 없는 계획 조회:

```bash
.venv/bin/python -m diffusion_hash_inv.study_cli pilot \
  --protocol examples/poc-v3.1-protocol.json \
  --workdir local_experiment_archive/runs/v31-dev \
  --stage P2 --dry-run
```

새 개발 디렉터리에서 P0 실행 후, 같은 환경/코드로 P1 실행:

```bash
.venv/bin/python -m diffusion_hash_inv.study_cli pilot \
  --protocol examples/poc-v3.1-protocol.json \
  --workdir local_experiment_archive/runs/v31-dev \
  --stage P0 --device mps --development

.venv/bin/python -m diffusion_hash_inv.study_cli pilot \
  --protocol examples/poc-v3.1-protocol.json \
  --workdir local_experiment_archive/runs/v31-dev \
  --stage P1 --device mps --development
```

위 P1 명령은 명세의 전체 개발 P1 규모를 실행한다. 이번 구현 검증에서는 축소한 통합 테스트만 실행했다. CPU 검사는 별도 디렉터리에서 `--device cpu --development`를 사용한다. Code/environment hash가 달라진 이전 run은 새 코드로 이어 쓰지 않는다. 비정상 종료로 미계수 시간이 불명확하거나 hard cap을 소진한 run은 `--resume`로 우회할 수 없다.

다음 구현 순서는 P2A/P2B의 선택·진단과 전체 비용 측정, exposure/MD5 실행기·E0·통계 calibration, P3 및 본실험 진입 gate다. 정식 실행 차단은 이 계약들이 실제 구현·검증된 뒤 해제한다.
