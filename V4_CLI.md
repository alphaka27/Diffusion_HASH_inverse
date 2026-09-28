# v4 실행 명세와 단일 CLI

`RESEARCH_PLAN_V4.md`의 Printable MD5-12 의사결정 연구를 실행한다. CLI 구현과 축소 검증은 실험 적격성 통과 또는 과학적 결과를 뜻하지 않는다. 새로운 study의 V0–V2가 통과해야 V3 본학습에 진입한다. 기존 v3/v3.1 실행기와 결과는 그대로 둔다.

진입점은 `diffusion_hash_inv.study_v4:main` 하나다. 아래 문서는 설치 상태에 의존하지 않는 모듈 명령을 사용한다. 패키지를 다시 설치하면 동일 함수의 console script `hash-inverse-v4`도 등록된다.

## 설치와 실행량 확인

저장소 루트에서 실행한다. 본실행은 Apple Silicon의 MLX/Metal GPU가 필요하다. CPU fallback이나 규모 축소 옵션은 제공하지 않는다.

```sh
uv sync --extra mlx
.venv/bin/python -m diffusion_hash_inv.study_v4 --help
.venv/bin/python -m diffusion_hash_inv.study_v4 plan
```

`plan`은 GPU에 접근하거나 파일을 쓰지 않는다. 고정 본실험은 6 learned runs × 15,700 updates, 후보 14,745,600행, sampling NFE 324,403,200이다. 설정은 [v4-protocol.json](examples/v4-protocol.json)에 기록되어 있다. `--protocol`을 생략하면 같은 내장 명세를 사용하며, 수정한 JSON은 거부한다.

## 한 번에 실행

```sh
DHI_V4_RUN="local_experiment_archive/runs/v4-$(date +%Y%m%d-%H%M%S)"

# 먼저 아래 '노출 감사 입력'에 따라 이 파일의 실제 증거와 검토 내용을 작성한다.
cp examples/v4-exposure-inventory.json /tmp/v4-exposure-inventory.json

.venv/bin/python -m diffusion_hash_inv.study_v4 audit \
  --workdir "$DHI_V4_RUN" --inventory /tmp/v4-exposure-inventory.json

.venv/bin/python -m diffusion_hash_inv.study_v4 run \
  --workdir "$DHI_V4_RUN" --stage all
```

템플릿 그대로의 감사는 `BLOCKED_EXPOSURE`다. `run --stage all`은 V0→V5 순서로 진행하고 첫 미달·차단·실패에서 멈춘다. 이미 완료된 단계는 봉인을 검증하고 건너뛴다. V1이 미달이면 `BLOCKED_QUALIFICATION`으로 종료한다. 결과를 보고 loss, seed, epochs, trial 수를 바꾸는 옵션은 없다.

## 단계별 명령

같은 `DHI_V4_RUN`을 계속 사용한다.

| 단계 | 명령의 마지막 인자 | 내용 |
|---|---|---|
| V0 | `--stage V0` | RFC MD5/leading-zero/codec, 실제 MLX 학습·생성 복구, 정보 경계, 원장→trial→counts 검사 |
| V1 | `--stage V1` | 균일 payload loss D1, synthetic seeds 0/1/2, normal/flipped acceptance |
| V2 | `--stage V2` | 노출 감사 확인, 8×20,000 calibration, 실제 MD5 E0, full-loop 자원 측정 및 batch 봉인 |
| V3 | `--stage V3` | 새 ownership·자료, Main/Shuffled × seeds 0/1/2 학습과 최소 validation checkpoint 봉인 |
| V4 | `--stage V4` | 봉인 후 trial schedule 생성, 9 streams × 16,384 trials × K100 |
| V5 | `--stage V5` | 독립 재해시, 여섯 동시 CI, 과학적 판정 및 한국어 보고서 |

```sh
.venv/bin/python -m diffusion_hash_inv.study_v4 run --workdir "$DHI_V4_RUN" --stage V0
.venv/bin/python -m diffusion_hash_inv.study_v4 run --workdir "$DHI_V4_RUN" --stage V1
.venv/bin/python -m diffusion_hash_inv.study_v4 run --workdir "$DHI_V4_RUN" --stage V2
.venv/bin/python -m diffusion_hash_inv.study_v4 run --workdir "$DHI_V4_RUN" --stage V3
.venv/bin/python -m diffusion_hash_inv.study_v4 run --workdir "$DHI_V4_RUN" --stage V4
.venv/bin/python -m diffusion_hash_inv.study_v4 run --workdir "$DHI_V4_RUN" --stage V5
```

계획만 확인하려면 `run`에 `--dry-run`을 붙인다. 단계 선행 조건이나 GPU 가용성까지 확인했다는 뜻은 아니다. V0/V1은 감사 입력 전에 실행할 수 있고, V2부터는 PASS 감사가 필요하다.

## 노출 감사 입력

`examples/v4-exposure-inventory.json`의 schema 1을 사용한다.

- `reviewer`: 실제 검토자/검토 주체 이름.
- `reviewed_scopes`: local/external runs, q8 원문/full digest, q≥12, 개발·리허설, 교차 source 재사용의 여섯 범위를 모두 검토한 경우에만 `true`.
- `unresolved`: 누락·불확실성 목록. 하나라도 남으면 차단한다.
- `entries`: run별 `id`, `scope`, `purpose`, `source`, `prefixes12`, `evidence`, `notes`.
- `prefixes12`: 0..4095 정수 배열. q≥12 target의 앞 12 bits를 투영한다. q8 원문/full digest를 열람한 경우 그 원문/full digest에서 12 bits를 복원한다. 정확히 결정할 수 없는 기록은 `unresolved`에 둔다.
- `purpose`: `evaluation`, `selection`, `development`, `training_only`, `unrelated_hash` 중 하나. 마지막 둘은 기록하되 자동 평가 노출에 포함하지 않는다.
- `source`: `printable`, `cross_source_linked`, `unrelated_source` 중 하나. Printable와 연결된 교차 source 기록도 제외 집합에 포함한다.
- `evidence`: 증거 파일의 **절대 경로**와 SHA-256 배열. 외부 실행의 원본 파일이 없다면 검토 근거를 담은 로컬 감사 기록을 남기고 그 파일을 연결한다. 원본 누락 자체가 미노출의 근거가 되지 않는다.

```sh
shasum -a 256 /absolute/path/to/evidence.json
.venv/bin/python -m diffusion_hash_inv.study_v4 audit \
  --workdir "$DHI_V4_RUN" --inventory /absolute/path/to/reviewed-inventory.json
```

검토자의 완전성 확인을 코드가 대신 추정하지 않는다. 문서에 이미 확인된 과거 노출 하한 1,885개보다 작은 inventory는 거부한다. 증거 파일 hash와 입력 schema를 검사하고, 새 test pool 1,024개 및 E0용 기존 노출 groups 128개를 확보할 수 있는지 확인한다. 감사 이전 integrity fixture의 MD5 targets와 등록된 E0 통합 검사에서 사용하는 validation/test targets도 `builtin_fixture_exclusions`로 보수적으로 제외한다. 감사 결과의 정확한 목록을 보존한다. 실제 연구 E0는 이 최종 E 안에서만 진행한다.

V2가 시작된 뒤 inventory는 수정할 수 없다. 새로운 노출을 뒤늦게 발견하면 현재 연구의 무결성 문제로 처리해야 한다.

## 재개와 오류

```sh
.venv/bin/python -m diffusion_hash_inv.study_v4 run \
  --workdir "$DHI_V4_RUN" --stage all --resume

.venv/bin/python -m diffusion_hash_inv.study_v4 report --workdir "$DHI_V4_RUN"
```

`--resume`은 중단된 run에만 적용하며 같은 learned run의 학습·평가를 합쳐 최대 1회다. 완료된 stage를 다시 실행하거나 적격성/수치/자원 실패를 재시도하는 수단이 아니다. 다른 run의 독립적인 중단에는 별도 1회 한도가 적용된다. 학습은 마지막 atomic checkpoint에서, 생성은 마지막 SQLite batch commit 다음부터 재개한다. 실패한 batch를 재생성할 때도 모든 난수 identity가 같다. 동일 workdir writer lock과 시스템의 v4 GPU lock으로 중복 실행을 막는다.

코드·Python·NumPy·MLX·PyTorch·lockfile·protocol 또는 봉인 파일이 달라지면 재개를 거부한다. 실행 중 코드를 수정하지 않는다. 임의 파일 복사나 state/complete JSON 수정으로 게이트를 우회하지 않는다.

Graceful interruption의 시간은 누적 보존한다. 강제 종료처럼 실제 종료 시점을 모르면 마지막 heartbeat 이후의 경과 시간에 downtime까지 포함하여 보수적으로 청구한다. `nfe_reserved`는 시작한 sampling batch의 상한이며, 완료한 sampling NFE와 MD5/재검산 호출은 별도 필드로 남긴다. 불확실한 중단 비용을 0으로 처리하지 않는다. Hard cap을 소진하면 재개할 수 없다.

종료 코드: `0` 성공/계획/보고서 작성, `2` 자격·노출·자원 차단, `3` 무결성·미완료·실행 오류, `130` 사용자 중단. 보고서의 `execution_status`와 `scientific_decision`을 함께 확인한다. 부분 원장의 미실행 trial은 실패 0으로 채우지 않는다. `report`는 기존 자료를 검증·집계하며 학습이나 새 후보를 생성하지 않는다.

## 고정 난수 및 파일 계약

Seed는 다음 순서의 UTF-8 compact JSON 배열에 SHA-256을 적용한 첫 8 bytes의 unsigned big-endian 정수다. 생략된 필드는 JSON `null`이다.

```text
[protocol_id, 2026092804, task, namespace, method, model_seed, epoch, trial, attempt]
```

`task`는 `V1`, `E0`, `MAIN`으로 구분한다. Integrity/profiling fixture는 `FIXTURE`, `PROFILE`, `MEASURE`를 사용하며 과학적 결과에 포함하지 않는다.

| namespace | 규칙 |
|---|---|
| ownership / remaining-ownership / data | task별 ownership, 나머지 그룹 배정, 원래 prior draw를 분리 |
| initialization / train-order / train-corruption | method=null; 같은 model seed의 Main/Shuffled가 공유 |
| shuffle | epoch마다 전체 train donor permutation; 동일 condition 잔류도 기록 |
| validation | model/method/epoch=null; group 및 draw 0..3에 고정 |
| trials | 여섯 checkpoint 봉인 이후 한 번 생성; 복원추출 결과를 그대로 보존 |
| length / payload | method·model seed·trial ID·attempt로 구분; 동일 target 반복에도 독립 |

Python `random.Random`에는 전체 정수를 전달한다. MLX에는 `mx.random.key(seed)`로 64-bit 키를 전달한다. Calibration은 계획서의 고정 NumPy seed `2026092804 + scenario_index`를 사용한다. V1 normal/flipped만 의도적으로 length/payload key를 공유한다. 평가 attempt는 1..100, trial ID는 0부터 시작한다. 예제 seed: `MAIN/train-corruption`, method=null, model_seed=0, epoch=1, trial=0, attempt=null → `9254768291057528196`.

| 파일 | 내용 |
|---|---|
| `protocol.frozen.json`, `manifest.json` | 전체 고정값, 실제 코드/환경/hash |
| `exposure_inventory.json`, `exposure_audit.json` | 검토 입력, 증거 hash, 제외 집합, 미해결 사유 |
| `V0..V5/state.json`, `complete.json` | 단계 상태·중단 횟수·artifact 봉인 |
| `V1/data`, `V2/E0/data`, `MAIN/data` | ownership, train/validation, targets, draw·중복·분포 감사, checksum |
| `*/data/evaluator/test_representatives.json` | 생성기에 전달하지 않는 test 대표 원문 |
| `*/runs/<seed>/<method>` | configuration, update telemetry, history, LATEST/BEST atomic checkpoints |
| `*/checkpoints.seal.json`, `*/trials.json` | checkpoint identity 및 공통 trial schedule |
| `*/evaluations/<seed>/<method>/candidates.sqlite` | 모든 attempt의 원장; 이후 success에도 조기 종료 없음 |
| `V2/calibration.json`, `V2/recovery.json` | 생산 통계 검산 및 선택 batch의 실제 MD5 복구 검사 |
| `resource.seal.json`, `budget.json`, `resumes.json` | 실측 batch·시간·저장량, 실제 누적 자원 및 재개 한도 |
| `decision.json`, `report.ko.md` | 실행 상태, 과학적 판정, CI, 비용·진단·결측 |

SQLite는 기존 WAL/FULL 테이블을 재사용한다. 물리 키 `(run_id, unit_id, variant, attempt)`에서 `run_id=protocol/task/method/seed`, `unit_id=trial_id`다. MD5 본실험 variant는 normal 하나다. 같은 target이 여러 trial에 나타나도 키가 충돌하지 않는다. 행 JSON은 target, payload 또는 invalid 사유, source/decode 판정, 실제 MD5-12, success, NFE, seed와 checkpoint/config hash를 가진다. 독립 verifier는 모든 payload를 재해시하고 행 순서·attempt·난수 identity를 확인한다. 중복은 stream 전체 및 trial 내부를 각각 보고한다.

원문 파일의 분리는 애플리케이션 경계다. 다른 OS 사용자/프로세스에 대한 암호학적 접근 통제 기능은 아니다. V0는 실제 trainer가 evaluator/targets 파일을 읽으면 실패하는 fixture를 실행한다.

## 실측 및 판정

V2 profiler는 20 warmup 후 100-update window 3개와 batch 1/4/16/64 각각 warmup 1회·측정 3회를 사용한다. 생성·decode·MD5·SQLite commit·telemetry의 전체 경로를 측정한다. 최대 처리량의 1% 이내이면 작은 batch를 고른다. 100,000행 SQLite/WAL/index fixture로 저장량을 추정한다. E0 hit 성능은 선택/통과 조건이 아니다.

예상 main 시간의 1.5배가 72시간 이내이고, 보수적 저장 추정의 2배가 64GiB 및 실제 disk 여유 안에 들어와야 한다. 실행 중 preparation 24시간, main 72시간, 전체 96시간, run 24시간, RSS/GPU 각각 64GiB, disk 최소 10GiB를 확인한다. 비용은 실패·복구·재검산에도 누적된다. Soft estimate 초과는 경고, hard cap은 차단이다.

정식 판정은 모든 게이트와 9개 stream이 완료된 후에만 가능하다. 여섯 paired differences에 대한 CP 기반 동시 CI로 GO / NO_GO_SMALL / NO_GO_REPRODUCIBILITY / INCONCLUSIVE를 결정한다. 결측·무결성 오류는 `NOT_EVALUATED`로 분리한다. 성공 0인 relative lift/성공당 비용은 `null`이다. 기술적 검증이나 E0 결과를 본실험 GO로 표시하지 않는다.

## 구현 검증 재현

```sh
.venv/bin/python -m pytest tests/test_study_v4.py -q
.venv/bin/python scripts/validate_research_plan_v4.py
```

첫 명령은 작은 실제 MLX 학습·재개, 실제 MD5 E0 9,600행, 320-update profiler, 노출/분할/판정/변조 거부를 검사한다. 두 번째는 원래의 가상 설계 검산이며 생산 경로와 CP·판정 primitive를 공유한다. 테스트 fixture와 자원 측정은 적격성 통과나 본실험 결과가 아니다. 전체 규모 V1 및 새 holdout 본실험은 별도 study에서 위 CLI로 진행한다.

## 이번 구현에서 확인한 결과

- 전체 회귀 검사: **161 passed, 5 skipped**. 이후 v4 입력·노출 하한·run 자원 경계 보강에 대해 전용 검사 **7 passed**를 다시 확인했다. Skip 5개는 기존 CUDA 미지원 검사다.
- 생산 통계 calibration: 8 scenarios × 20,000 repetitions, 모든 coverage/false-GO/boundary/power gate PASS. All-null NO_GO_SMALL 비율 0.8498, reference-twofold GO 비율 0.99015이다. 이 수치는 가상 통계 검증 결과다.
- 실제 MLX의 동일 가중치/optimizer 재개, 후보 batch 중단 후 일치, 실제 MD5 E0 9,600행·NFE 211,200, 네 batch와 320-update resource profiler를 검증했다. 이 E0는 테스트 fixture이며 정식 study의 E0 gate를 대신하지 않는다.
- 실제 CLI V0 실행도 **PASS**: RFC MD5 vectors 7개, codec 왕복 28개, MLX 학습/optimizer·후보 재개의 정확한 일치, 숨은 test 정보 차단, 원장 fixture를 확인했다. 실행 폴더는 `local_experiment_archive/runs/v4-implementation-v0-20260928`이다. `report`가 미완료 연구를 `NOT_EVALUATED`로 기록하는 것도 확인했다.
- 새 study의 V1 적격성, 프로젝트 전체 노출 inventory 승인, 본실험 resource seal, 새 holdout 학습/평가는 미실행이다.

검증 기록: `local_experiment_archive/analyses/v4-implementation-tests.xml`, `v4-final-focused-tests.xml`, `v4-implementation-calibration.json`. 이 로컬 산출물은 Git 제외 영역에 있다.
