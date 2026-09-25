# v3 Pilot CLI 사용법

기존 `diffusion_hash_inv` 패키지에 `study_cli`와 Pilot 실행기를 추가했다. 현재 지원 명령은 **`pilot`과 Pilot용 `report`**다. `audit`, 통계 `validate`, 본실험 `prepare/train/evaluate`는 이번 구현 범위에 포함되지 않는다. Pilot CLI가 존재한다는 사실과 실제 P0–P3 완료·학습 적격성은 구분한다.

## 진입점

| 진입점 | 용도 |
|---|---|
| `.venv/bin/python -m diffusion_hash_inv.study_cli` | Python module 실행; 권장 |
| `.venv/bin/hash-inverse-study` | 같은 `main()`을 호출하는 console script |
| `pilot --stage P0/P1/P2/P3` | 선행 gate와 환경을 확인하고 지정 단계 실행 |
| `report` | 봉인된 산출물 무결성 검사와 Pilot Markdown 보고서 생성 |

현재 작업 환경의 `.venv`에는 console script도 등록했다. 다른 환경에서 패키지를 설치할 때는 `pyproject.toml`의 entry point로 함께 설치된다. v3 JSON을 기존 `experiment_cli --config`에 전달하지 않는다.

## 실행 명령

GPU가 보이는 macOS 호스트 터미널에서 실행한다. Sandbox 안에서 `MPS is unavailable`이 나오는 경우 호스트 터미널에서 다시 확인한다. GPU를 사용할 수 없으면 CPU로 자동 전환하지 않는다.

```bash
DHI_ROOT="/Users/choisoonwook/Experiments_local/DHI_AI_gen"
DHI_PY="$DHI_ROOT/.venv/bin/python"
DHI_SPEC="$DHI_ROOT/examples/poc-v3-protocol.json"
DHI_RUN="$DHI_ROOT/local_experiment_archive/runs/v3-pilot-$(date +%Y%m%d-%H%M%S)"

# 파일 작성·GPU 접근·학습 없이 설정과 실행 규모 확인
"$DHI_PY" -m diffusion_hash_inv.study_cli pilot \
  --stage P1 --protocol "$DHI_SPEC" --workdir "$DHI_RUN" --dry-run

# 먼저 P0 실행. 뒤 단계도 동일한 DHI_RUN을 사용한다.
"$DHI_PY" -m diffusion_hash_inv.study_cli pilot \
  --stage P0 --protocol "$DHI_SPEC" --workdir "$DHI_RUN" --device mps
```

각 단계의 종료 코드와 `report.md`를 확인한 뒤 다음 명령을 하나씩 실행한다. 새 터미널에서 이어갈 때는 `DHI_RUN`을 **기존 실행 디렉터리의 정확한 경로**로 다시 지정한다. 날짜 명령으로 새 경로를 만들지 않는다.

```bash
"$DHI_PY" -m diffusion_hash_inv.study_cli pilot \
  --stage P1 --protocol "$DHI_SPEC" --workdir "$DHI_RUN" --device mps

"$DHI_PY" -m diffusion_hash_inv.study_cli pilot \
  --stage P2 --protocol "$DHI_SPEC" --workdir "$DHI_RUN" --device mps

"$DHI_PY" -m diffusion_hash_inv.study_cli pilot \
  --stage P3 --protocol "$DHI_SPEC" --workdir "$DHI_RUN" --device mps

"$DHI_PY" -m diffusion_hash_inv.study_cli report \
  --protocol "$DHI_SPEC" --workdir "$DHI_RUN"
```

`--dry-run`은 GPU 가용성이나 선행 gate의 실제 통과를 보증하지 않는다. P3는 P2가 선택한 batch, 재개 검사, 시간·저장 공간 예산이 적합해야 시작한다. P3의 성능 기준을 통과해도 본실험의 노출 감사·데이터 봉인·통계 calibration과 본실험 실행기 구현은 별도로 필요하다.

## 설정 가능한 CLI 파라미터

| 파라미터 | 명령 | 필수/기본값 | 의미·제약 |
|---|---|---|---|
| `--protocol PATH` | pilot, report | 필수 | v3 protocol JSON. 현재는 `dhi-v3-20260924`, revision `3.0`의 고정 명세만 수용 |
| `--workdir PATH` | pilot, report | 필수 | 모든 단계에서 공유하는 study root. 기존 다른 파일을 덮어쓰지 않음 |
| `--stage P0/P1/P2/P3` | pilot | 필수 | 실행할 한 단계. 선행 단계 건너뛰기 불가 |
| `--device mps/cpu` | pilot | `mps` | 명시적 device. CPU는 `--development` 필수. 현재 CUDA/auto 옵션은 제공하지 않음 |
| `--development` | pilot | false | 개발 전용 study. 정식 GPU Pilot 자격을 부여하지 않음. 규모를 줄이는 옵션은 아님 |
| `--threads N` | pilot | `1` | 양의 정수인 Torch CPU thread 수. GPU 개수가 아님. study 전체에서 고정 |
| `--resume` | pilot | false | 중단된 동일 단계의 동일 설정 재개. 단계당 최대 1회, 완료 단계 재실행 불가 |
| `--dry-run` | pilot | false | protocol 유효성과 단계별 실행량만 출력, study 파일 생성하지 않음 |
| `-h`, `--help` | 모두 | — | 해당 명령의 도움말 |

P0에서 정한 `device`, `development`, `threads`와 코드·환경·protocol은 이후 단계 및 재개에서 동일해야 한다. 코드 수정·환경 업그레이드 후에는 이전 실행을 같은 조건의 재개로 취급하지 않는다.

```bash
# 동일 study의 중단된 P1을 최대 한 번 재개
"$DHI_PY" -m diffusion_hash_inv.study_cli pilot \
  --stage P1 --protocol "$DHI_SPEC" --workdir "$DHI_RUN" --device mps --resume

# CPU 개발 점검은 별도 root 사용; 정식 MPS study와 섞지 않는다.
"$DHI_PY" -m diffusion_hash_inv.study_cli pilot \
  --stage P0 --protocol "$DHI_SPEC" \
  --workdir "$DHI_ROOT/local_experiment_archive/runs/v3-cpu-development-$(date +%Y%m%d-%H%M%S)" \
  --device cpu --development
```

## Protocol에 기록되어 있으나 임의 변경할 수 없는 값

CLI는 설정 파일의 의미를 일부 무시한 채 실행하지 않도록 canonical JSON SHA-256으로 지원 명세를 확인한다. 공백과 key 순서는 바꿀 수 있지만 **값을 수정한 JSON은 차단한다**. 다음 값에 대한 자유 override 옵션은 없다. 변경하려면 scientific/operational revision과 실행기 지원 범위를 함께 개정해야 한다.

| 설정 | v3 고정값 |
|---|---|
| Pipeline | P-G-BGV, P-G-CGGE, P-DISC, R-G-BGV, R-DISC |
| 입력 | 정확히 12-bit condition, payload 길이 4–31 |
| 학습 | Adam, lr .001, batch64, float32, epoch별 비복원 순열 |
| Gaussian | width32, T1000, sampling100 |
| Discrete | width128, embedding16, sampling32, temperature1 |
| P1 | 10 learned runs, train256/validation32, 2 epochs, trial16×K10, inference batch4 |
| P2 | 5 learned runs, train10000/validation512, 10 epochs, 128 cases×정상/반전, batch profile1/4/16/64 |
| P3 | 15 learned runs, train10000/validation512, 100 epochs, test512×정상/반전, seeds0/1/2 |
| P3 수용 기준 | 각 seed: 정상 joint≥475, 반전 joint≥475, 원래 조건 오성공≤15 |
| Active wall-clock cap | P0 10분, P1 30분, P2 4시간, P3 48시간; 재개 시 누적 |
| 저장·메모리 상한 | study64 GiB, run8 GiB, RSS64 GiB, MPS64 GiB, 디스크 여유10 GiB 이상 |

P1/P2에서 유효 생성률·정답률이 낮다는 이유만으로 후보를 재생성하거나 seed를 바꾸지 않는다. P3는 성능 미달 seed가 있어도 지정된 세 seed를 모두 평가한다.

## 저장과 실패 처리

- Study: `protocol.frozen.json`, `manifest.json`, `gates.json`, `report.md`. 통계 결과가 없는 `analysis_validation.json`은 명시적으로 `NOT_RUN`이다.
- Stage: `pilot/P*/state.json`, `summary.json`; P0는 `checks.json` 추가.
- Run: `configuration.json`, `training.json`, `metrics.json`, `telemetry.jsonl`, `evaluation.json`, `candidates.sqlite`, `checkpoints/`, 제한된 `raw/`.
- P2: 각 run의 `profile.json`, 선택 batch의 복구 검사, study의 `resources.json`.
- 후보는 batch 단위 SQLite transaction으로 저장한다. Invalid·중복·성공 뒤 후보도 예산을 소비한다. Synthetic verifier와 MD5 성공을 혼동하지 않는다.
- 학습 checkpoint는 100 updates/epoch 경계와 복구 fixture에서 저장한다. 재개는 마지막 검증된 checkpoint 이후 연산을 replay한다. 무결성 실패를 재학습으로 덮어쓰지 않는다.
- Main/Shuffled는 초기 weights와 학습 순서를 공유한다. Shuffled donor는 학습 행 안에서만 바꾸며 validation/inference에는 실제 조건을 사용한다.

| 종료 코드 | 의미 |
|---:|---|
| 0 | 요청 단계/보고 완료. 과학적 우위나 본실험 준비 완료를 의미하지 않음 |
| 2 | 설정·선행 gate 오류 또는 P3 전체 적격성 미충족 |
| 3 | Runtime/수치 오류 |
| 4 | 체크섬·재개·복구 결과 불일치 |
| 5 | 시간·메모리·저장 공간·실측 자원 예산 초과 |
| 130 | 사용자 중단 |

`report`는 실패·미실행 단계도 표시할 수 있으며 완료 단계의 변경/손실은 무결성 오류로 처리한다.

## 구현 검증과 미실행 범위

2026-09-25 확인 결과:

| 검사 | 결과 |
|---|---|
| 전체 pytest | **116 passed, 5 skipped**; backend가 없는 환경의 기존 GPU 검사는 skip |
| 새 Pilot 회귀·통합 검사 | 8개 통과; 다섯 모델, 학습·생성·재개, stage gate, schema 차단, ledger 변경 탐지, P3 평가/성능 판정 분리, P2 profile/resource 차단 |
| 실제 MPS의 정식 P0 | **PASS**; 47 checks, codec round-trip 1,214개. [보고서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/v3-cli-p0-mps-final-20260925/report.md) |
| 실제 MPS의 축소 P1 | **DEVELOPMENT ONLY / PASS**; 다섯 pipeline×Main/Shuffled, 각 8 updates, 각 20 candidates, Main별 중단/재개 일치. [보고서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/v3-cli-small-mps-check-20260925-final/report.md) |
| 정식 P1/P2/P3 전체 규모 | **NOT_RUN** |
| 본실험 실행·통계 calibration | **NOT_IMPLEMENTED / NOT_RUN** |

축소 MPS 검사는 테스트 harness에서만 작게 만든 protocol을 사용했다. 정식 CLI의 설정 제한을 우회하는 사용자 옵션은 제공하지 않으며 이 결과로 정식 P1 gate를 통과시키지 않는다. 해당 개발 보고서는 테스트가 직접 생성한 파일이며, 정식 protocol을 인자로 전달해 그 개발 root를 `report`/`resume`하는 것은 protocol mismatch로 차단된다.

이 환경에서 MPS의 기본 embedding backward는 동일 학습을 반복해도 optimizer state와 일부 weights의 최하위 비트가 달라졌다. Pilot 실행기는 MPS에서 같은 embedding table을 one-hot 행렬곱으로 계산해 gradient scatter 누적을 피한다. 파라미터 수와 lookup 출력은 유지하고, Adam은 `foreach=False, fused=False`로 고정했다. CPU의 lookup/gradient 동등성 검사와 두 Discrete 모델의 실제 MPS exact recovery를 확인했다. 다른 OS·Torch·device에서도 동일하다는 보장은 하지 않으며 각 study의 P1 검사를 요구한다.

축소 integration 테스트는 정식 protocol의 규모·성능 판정을 대체하지 않는다. 이번 변경에서는 정식 P1/P2/P3 전체 학습이나 본실험을 실행하지 않았다.
