# V6 독립 MLX 구현

`src/dhi_v6/`는 `dhi_v5`와 `diffusion_hash_inv`를 import하지 않는 독립 구현이다. 두 패키지는 등가 테스트에서만 읽기 전용으로 쓴다. Diffusion backend는 **MLX/Metal float32**로 고정한다. 기준 문서는 다음 셋이다.

- 과학적 설정: `RESEARCH_PLAN_V6.md`
- 구현 세부: `V6_IMPLEMENTATION_SPEC.md`
- 등록값: `examples/v6-protocol.json`(`registration()`과 byte 단위로 같아야 한다)

JSON을 고쳐서 trials, updates, seed, 블록 크기를 줄일 수 없다. Cap을 바꾸는 방법은 `HALT` 뒤의 `approve-caps`뿐이다.

## 설치

Apple Silicon과 MLX가 필요하다. 의존성은 `uv.lock`에 고정되어 있다.

```sh
uv sync --extra mlx --group dev
```

현재 `.venv`(Python 3.12.4, numpy 2.5.3, mlx 0.32.2)로 바로 실행할 수 있다. Console script `hash-inverse-v6`는 `PYTHONPATH=src .venv/bin/python -m dhi_v6.study`와 같다. Metal 접근이 막힌 샌드박스에서는 GPU를 쓸 수 있는 터미널에서 실행한다. CPU나 torch 모델로 자동 대체하지 않는다.

## 실행 전 검사

```sh
DHI_V6_REQUIRE_METAL=1 .venv/bin/python -m pytest tests/test_study_v6.py -q -rs
DHI_V6_REQUIRE_METAL=1 .venv/bin/python -m dhi_v6.study check --root "$TMPDIR/dhi-v6-quick" --quick
.venv/bin/python -m dhi_v6.study plan
```

Metal 테스트는 `DHI_V6_REQUIRE_METAL=1`에서 skip 대신 실패한다. skip이 0개여야 통과다. `check --quick`은 경로 검사다. `quick_paths_passed`만 의미가 있고, A-impl PASS를 만들지 않는다. 정식 A-impl은 `run --stage A --phase impl`이 만든다.

## 실행 순서

정식 실행은 **빈 새 디렉터리**에서 시작한다. 아래 `ROOT`는 예시다.

```sh
ROOT=local_experiment_archive/runs/v6-study
hash-inverse-v6 inventory --draft --v5 examples/v5-exposure-inventory.json --output "$ROOT-inventory.json"
# 사람이 초안의 unreviewed 항목과 범위를 검토하고 scope_complete를 true로 바꾼다.
hash-inverse-v6 audit --root "$ROOT" --inventory "$ROOT-inventory.json"
caffeinate -i hash-inverse-v6 run --root "$ROOT" --stage A
```

`run --stage A`는 아래 phase를 순서대로 실행한다. `--phase`로 하나씩 실행할 수도 있다.

| Phase | 내용 | 산출물 |
|---|---|---|
| `impl` | G1–G10 정식 게이트(30–45분). 실패하면 멈춘다 | `A-impl.json` 또는 `A-impl-failure.json` |
| `prof1` | 학습 속도, burst 생성(B·S_G별 60초), B* 선택, MD5 prior·검증기 처리량 | `A-prof-1.json` |
| `train` | 5 파이프라인 × seed 0–2 × 40,000 update, synthetic | `A-Q/runs/…/u40000/` |
| `dev` | Gaussian 3개 파이프라인의 S_G 선택 | `A-dev.json` |
| `evaluate` | Acceptance 512 × 정상/반전, CLP 4,096쌍. Q 결정 | `A.json` |
| `repair` | 실패한 파이프라인만 80,000 update로 이어 학습한 뒤 재평가한다(A_repair cap). 대상이 없으면 건너뛴다 | `A-dev-repair.json`, `A.json` round 2 |
| `prof2` | Q의 seed-0 checkpoint로 전체 평가 경로를 600초 워밍업 후 900초 측정 | `A-prof-2.json` |
| `budget` | 예산 계획. `HALT`이면 exit 3 | `budget-plan.json` (`halt.json`) |
| `freeze` | 동결. `exposure-audit.json`이 인증되어 있어야 한다 | `protocol.frozen.json`, `protocol.seal.json`, `window.json` |

Q가 비면 `prof2` 이후를 건너뛰고 연구가 끝난다(`report`). 동결 뒤에는 다음 순서로 실행한다.

```sh
caffeinate -i hash-inverse-v6 run --root "$ROOT" --stage C
caffeinate -i hash-inverse-v6 run --root "$ROOT" --stage R   # 감사를 통과한 POSITIVE가 없으면 빈 R.json만 봉인
caffeinate -i hash-inverse-v6 run --root "$ROOT" --stage P
caffeinate -i hash-inverse-v6 run --root "$ROOT" --stage S   # 예산 계획이 S를 생략하면 skipped로 봉인
hash-inverse-v6 report --root "$ROOT"
```

`run --stage all`은 A → C → (R) → P → S → report를 한 번에 실행한다.

### `HALT`와 cap 승인

예산 계획이 C를 look 2까지 끝낼 수 없다고 예측하면 `halt.json`을 쓰고 exit 3으로 멈춘다. MD5 조건 데이터(C·R·P·S 폴더)가 하나도 없을 때만 C cap을 올릴 수 있다.

```sh
hash-inverse-v6 approve-caps --root "$ROOT" --stage C --hours 120 --reason "사유"
hash-inverse-v6 run --root "$ROOT" --stage A --phase budget
hash-inverse-v6 run --root "$ROOT" --stage A --phase freeze
```

증액만 승인할 수 있다(등록값 80시간 초과). 승인하면 필수 경로 상한(114시간, 보완 시 126시간)도 같은 시간만큼 늘어난다. 예를 들어 C를 120시간으로 승인하면 필수 경로 상한은 154시간(보완 시 166시간)이 된다. 승인 기록 `cap-override.json`은 봉인되어 동결에 포함된다. `halt.json`은 기록으로 남는다.

## 명령

| 명령 | 동작 |
|---|---|
| `plan [--output PATH]` | `registration()` 출력. `--output`이면 봉인 저장 |
| `inventory --draft --v5 PATH --output PATH` | V6 노출 inventory 초안(명세 §14.3). 미검토 항목은 `unreviewed`로 남는다 |
| `audit --root R --inventory PATH` | 노출 감사 → `exposure-audit.json`. 동결 뒤에는 거부 |
| `run --root R --stage {A,C,R,P,S,all} [--phase …]` | 단계 실행. `--phase`는 A에만 쓴다 |
| `approve-caps --root R --stage C --hours N --reason TEXT` | `HALT` 뒤, MD5 데이터 전에만 허용. C 증액만 가능하며 필수 경로 상한도 같이 늘어난다 |
| `status --root R` | 진행 정보만 출력. 성공률·구간·판정은 출력하지 않는다 |
| `report --root R` | `decision.json`, `FINAL_REPORT_KO.md`. 미완료면 `INCOMPLETE / NOT_FINAL` |
| `check --root R [--quick]` | A-impl 게이트. `--quick`은 개발용이며 PASS를 만들지 않는다 |

종료 코드는 0 정상, 1 오류, 2 예산 종결(A 단계 cap 소진, `failure.json`), 3 `HALT`다. C·R·P·S가 cap 때문에 부분 결과로 봉인되면 exit 0이고, 출력의 `budget_stop`(단계 실행) 또는 `budget_stops`(`--stage all`)로 알린다. V6 표지(`v6-study.json`)가 없는 비어 있지 않은 디렉터리는 거부한다. `failure.json`이 있으면 `run`을 거부한다. 한 root에는 한 프로세스만 쓸 수 있다(`.lock`).

## 산출물

명세 §15의 구조를 따른다. 명세에 없는 파일은 다음과 같다.

| 파일 | 내용 |
|---|---|
| `A-dev/<pipeline>/u<updates>/selection.json` | 파이프라인별 S_G 평가(봉인). `A-dev.json`과 `A-dev-repair.json`은 이것을 모은다 |
| `A-Q/runs/<p>/Main-<s>/u<updates>/qualification.json` | seed별 A-Q 평가(봉인) |
| `<stage>/integrity.json` | 무결성 실패로 빠진 파이프라인과 사유 |
| `C/looks/block-<j>-start.json` | 블록 j 시작 시점의 C 누적 시간. 다음 블록 사전 검사에 쓴다 |
| `<stage>/clp/<label>-seed<s>.json` | CLP 쌍별 차이와 probe(봉인) |
| `<stage>/groups-*.json`, `<stage>/trials.json` | 단계 split과 trial 목록(봉인) |
| `implementation-check.json`, `quick-check.json` | `check` 명령 결과. 연구 인증과 무관하다 |

원장·checkpoint·보고서는 `local_experiment_archive/` 아래에만 두고 커밋하지 않는다.

## 운영 주의

- **잠자기 방지.** 긴 단계는 `caffeinate -i`로 실행한다. GPU 작업은 한 번에 하나만 돌린다.
- **재개.** 같은 명령을 다시 실행하면 봉인된 단계는 검증한 뒤 건너뛴다. 중단된 run·stream은 같은 identity로 한 번만 재개한다. 같은 run이나 stream이 두 번째로 중단되면 무결성 실패다.
- **블라인드.** 실행 중 CLI는 `{"look": j, "action": "continue" | "stop"}`만 출력한다. `status`도 진행 정보만 보여 준다. 연구가 끝나기 전에는 `C/looks/`, `*/eval/*/block-*.json`, `C.json`을 열지 않는다. 이 파일들에 결과가 들어 있다.
- **봉인.** `.sha256`이 있는 JSON을 고치면 읽기가 거부된다. 동결 뒤에 소스·환경·registration·A 산출물이 바뀌면 C·R·P·S 실행이 거부된다. 코드 수정이 필요하면 새 root에서 A-impl부터 다시 한다.
- **저장량.** 원장은 후보당 36 B다. Q가 5개일 때 C 블록 하나는 36 stream × 819,200행 ≈ 1.06 GB다. 학습 digest는 source-seed마다 40,000 update 기준 164 MB다. 저장량 cap은 64 GiB이고 최소 여유 디스크는 20 GiB다.
- **처리량 참고(fixture 측정, batch 256).** D1-S stream은 검증과 재생성 감사를 포함해 약 10,000 후보/초, Random은 약 128,000 후보/초였다. D1-T-L은 약 260 후보/초로 느리다. S의 GEN(Main·MC 819,200 후보)은 이 속도면 1시간에 가깝다. 정식 값은 A-prof-1이 잰다.
- **A-Q run 무결성 실패.** A-Q run이 재시도까지 실패하면 exit 1로 멈춘다. 등록 규칙상 재시도는 1회뿐이므로 원인을 기록하고 연구를 종료한다.

## 구현 결정

- Source별 token ID는 `data.TOKENS`로 관리한다. `fresh_batch`는 `(stage, source, seed_id)` 또는 CLP namespace에서 source를 읽으며, 파이프라인과 Main/Shuffled는 메시지 namespace에 넣지 않는다.
- 이미지 codec은 NumPy NCHW float32를 입출력하고, prototype 거리는 float64로 계산한다. 거리 행렬은 256개 후보씩 처리한다.
- Gaussian 재생성 감사의 마지막 묶음이 64개보다 작으면 마지막 선택 key를 반복해 batch 64를 채우고 실제 선택 후보만 비교한다.
- **Calibration의 두 용도를 구분한다.** 200회 축소 검사는 판정 경로·CP 계산·공동 중단의 회귀 검사다. 관측한 CP 한계와 기준 충족 여부는 그대로 출력하지만 `scope=regression-only`, `production=false`, `passed=false`이며 연구 통과 근거가 될 수 없다.
- **정식 calibration**은 등록된 블록 크기 8,192, 시나리오별 최소 2,000회로 수행한다. 무효과 양성률의 one-sided 95% CP 상한 ≤ 0.025, 전체 기각률 CP 하한 ≥ 0.95, +δ 검출률 CP 하한 ≥ 0.95를 모두 만족해야 `passed=true`다. 정식 게이트 실패 시 난수·반복 수·문턱을 사후 조정해 통과시키지 않는다.
- 이 구분은 2026-09-29 사용자의 “연구 최종 결론의 근거로 사용하기에 논리적으로 적절한 기준” 요청에 따른다. 과학적 문턱과 등록 JSON은 변경하지 않았다. Calibration은 판정 절차 검증이며, 실제 효과에 대한 최종 결론은 별도의 봉인된 C 자료·무결성 감사·필요한 R 재현에만 근거한다.
- **Quick 검사.** `check --quick`의 `quick_paths_passed`는 G9를 뺀 모든 게이트 통과와 G9가 네 시나리오를 `regression-only`로 계산했는지만 본다. 200회 calibration의 CP 한계는 우연히 기준을 넘을 수 있기 때문이다.
- **A-prof-1.** 학습 시간은 synthetic stream으로 10 update 워밍업 뒤 50 update의 중앙값이다(데이터 생성·인코딩·`charge_work` 포함). `forward_rows_per_second`는 CLP 점수 batch와 같은 512행 순전파를 5초 이상 잰 값이다. MD5 prior 처리량은 P source prior 표본 추출과 hashlib 단일 core MD5를 65,536개로 잰다(C4의 ρ에 쓴다). 검증기 처리량은 W1 r=64, 무작위 12-bit target의 Random 원장 1,024,000행으로 잰다.
- **A-prof-2.** `runtime.evaluate_block`을 약 30초 분량(512 trial의 배수) 블록으로 반복한다. 따라서 생성 → decode → 해시 → 학습 조회 → 원장 → 검증 → 1% 재생성 감사를 모두 포함한다. 학습 조회는 무작위 digest 1,024만 개 fixture를 쓴다. 지속 처리량은 min(워밍업 뒤 누적, 마지막 300초)이다. Random은 Q에 있는 source마다 10초 워밍업 뒤 120초 재고, `random_cps`는 source 중 최솟값이다. 예산 공식이 검증과 재생성을 따로 더하므로 추정은 검증 시간을 두 번 센다(보수적).
- **예산 계상.** Stage 세션마다 `Budget` 하나를 쓰고, 세션이 끝날 때 `Budget.flush()`로 경과 시간을 저장한다. A_repair 예산은 보완 대상이 있을 때만 연다. 그래야 필수 경로 한도 126시간이 실제로 보완한 경우에만 적용된다.
- **C 블록 사전 검사.** 블록 시간은 `block-<j>-start.json`과 look 봉인 사이의 C 누적 시간이다. 이미 시작한 블록을 재개할 때는 사전 검사를 하지 않는다.
- **무결성 실패.** 학습이나 stream에서 ValueError·RuntimeError·FloatingPointError가 나면 해당 파이프라인을 `<stage>/integrity.json`에 기록하고 뺀다. Random stream이 실패하면 그 source의 파이프라인이 모두 빠진다. 그 밖의 예외는 exit 1이다.
- **예산 소진(2026-09-29 사용자 결정).** 명세 §13.2는 cap 소진 시 `failure.json`을 쓰고 종료하도록 적었지만, 적용 범위를 다음과 같이 정했다.
  - A(보완 포함)에서 cap이 소진되면 `failure.json`을 쓰고 보고서를 만든 뒤 exit 2로 끝난다.
  - C는 §12.10 look 규칙을 따른다(마지막 완료 look이 최종, 미결은 `NOT_ESTABLISHED_BY_BUDGET`). CLP_64가 끝나지 않으면 `clp_partial`로 표시한다.
  - R은 끝나지 않은 진입 파이프라인을 `NOT_ESTABLISHED_BY_BUDGET`으로, P·S는 `partial`로 봉인하고 다음 단계와 보고서로 진행한다.
  - 이유: 단계마다 cap이 따로 있다. 선택 단계인 R의 소진이 필수 경로 단계인 P를 막으면 계획의 구조와 어긋난다. 또 어느 쪽이든 C3와 종합 판정은 같다.
- **세션 시작 시 소진.** C·R·P·S는 세션 안에서 학습·생성·CLP를 수행해 봉인하고, 판정은 세션 밖에서 봉인된 자료만으로 조립한다. 따라서 다음 두 경우가 같은 규칙으로 처리된다. 봉인된 look·블록·CLP는 버려지지 않는다.
  - 실행 도중 cap이 소진된 경우
  - 봉인 직전에 프로세스가 죽은 뒤, cap이 이미 소진된 상태로 다시 실행한 경우
- **Cap 승인과 필수 경로(2026-09-29 사용자 결정).** 필수 경로 상한 114시간은 A 24 + C 80 + P 10의 합이다. 그래서 C 증액을 승인하면 필수 경로 상한(보완 시 126시간 포함)도 같은 만큼 늘린다. 그렇지 않으면 승인한 C 시간을 쓸 수 없고 P가 시간을 잃는다. 명세 §13.3은 이 경우를 정하지 않았다. 승인은 MD5 데이터 전에만 가능하므로 결과가 결정에 영향을 줄 수 없다. 감액 승인은 받지 않는다.
- **CLP.** CLP_64는 Main seed별 65,536쌍을 따로 봉인하고, 세 seed를 모은 z > 2.8782이면 `INFO_64`다. A-Q에서 생성은 통과했는데 CLP만 실패한 파이프라인(계획 §5.3)은 CLP_64와 CLP_4를 계산하지 않고 `disabled`로 표시한다.
- **Artifact 감사(계획 §6.5).** C `POSITIVE` 파이프라인의 모든 성공 후보를 hashlib으로 다시 해시(W3)한다. 추가로 alphabet, 학습 digest 조회, 봉인 산출물 hash를 확인하고, 상위 1% target 비중, CLP 방향, hit-only를 기록한다. 감사에 실패하면 `NOT_ESTABLISHED_INTEGRITY`다. 원인을 고친 뒤의 재생성은 자동화하지 않았다.
- **P와 S.** P는 Q 전체의 seed 0을 쓴다(C에서 무결성 때문에 빠진 파이프라인 포함). S는 C의 W3 split을 쓰고, 생성 batch는 A-prof-1의 D1-T-L B*다.
- **보고서.** `decision.json`은 명세 §12.14 구조에 `batch`, `audit`, `replication`, `stage_c`, `Q`를 더한다. `failures`는 목록이다. `FINAL_REPORT_KO.md`는 계획 §15.3 구성을 따르고, 결론 문장은 §15.2 사전 문장에 값을 채운다.
- **Status.** 산출물 존재 여부, stream 블록 수, 완료 행 수, 최근 처리량(최근 봉인 블록 3개), C look 봉인 수, look 3까지의 ETA 상한, stage별 사용·남은 시간, halt, failure만 출력한다. `halt`는 현재 `budget-plan.json`의 결정을 따른다.
- **`check`(정식).** `implementation-check.json`만 쓰고 `A-impl.json`은 만들지 않는다.
