# V5 독립 MLX 구현

`src/dhi_v5/`는 기존 `diffusion_hash_inv`의 모델·codec·데이터·통계·실행기를 import하지 않는 신규 구현이다. Diffusion backend는 **MLX/Metal float32**로 고정한다. 기존 v3.1/v4 코드와 산출물은 변경하지 않는다.

## 고정 명세와 재작성에 따른 변경

기준은 `RESEARCH_PLAN_V5.md`다. 사용자 지시에 따라 §15의 재사용 항목도 새로 작성했다. §5.1의 기존 sampler와 bitwise 비교는 **새 scalar 참조 sampler와 4,096개 후보 bitwise 비교 + batch 1/64/1,024 불변성**으로 대체한다. 이 변경은 `registration().sampler_gate`에 고정되어 봉인된다. 과거 가중치와 checkpoint는 사용하지 않는다.

- D1-S: embedding 16, hidden 128, condition-output residual, 508,668 parameters. D1-T: Pre-LN 4×192, 4 heads, FFN 768. D1-T-L: 8×256, 8 heads, FFN 1,024. Transformer의 실제 parameter 수와 처리량은 A-prof에서 기록한다.
- 12-bit condition, `Linear(12,28)` length head, 내부 condition의 `L/31`, 균일 masked payload CE. 생성은 length 1회 + 32 reverse intervals, NFE 33, temperature 1, remasking 없음. PAD/EOS는 생성 길이에 따라 배치한다.
- Printable ASCII 33–126, 길이 4–31 균등. Token ID는 payload 0–93, PAD 94, EOS 95, MASK 96이다. Synthetic prefix는 3자리 대문자 ASCII hex다. Acceptance와 그 bitwise complement는 train/dev에서 제외된다.
- NumPy 벡터화 step-reduced MD5, 독립 scalar RFC 참조, hashlib verifier, W1/W2/W3. 매 update의 데이터·corruption을 명시적 namespace로 생성한다. 같은 seed의 Main/Shuffled는 초기 가중치·메시지·corruption을 공유한다.
- MLX embedding gradient의 Metal scatter 누적 순서 차이를 막기 위해 embedding을 one-hot 행렬곱으로 계산한다. Scalar 참조와 vectorized sampler는 명시적 후보/step 키를 공유한다.
- 최종 checkpoint만 평가한다. 4,000 updates마다 validation objective와 256 CLP pairs를 진단으로 기록한다. A-dev는 세 lr×두 seed의 최종 validation objective 평균으로 선택하며 동률은 등록된 lr 순서다.
- A-Q CLP는 4,096 pairs로 고정한다. 조건 생성 기준은 통과했으나 CLP가 실패하면 계획대로 이후 CLP 판정을 제외한다.

이 세부값은 결과를 보기 전에 정한 신규 구현의 등록값이다. 고정 설정은 `examples/v5-protocol.json`에 있다. JSON을 편집하여 trials, updates, seed, backend를 줄일 수 없다.

## 실행

Apple Silicon과 MLX가 필요하다. 프로젝트 의존성을 설치할 때는 `uv sync --extra mlx --group dev`를 사용한다. 현재 저장소의 환경으로는 다음 명령을 사용할 수 있다.

전체 순서를 한 명령으로 진행하려면 완성된 노출 감사 inventory를 지정한다. `examples/v5-exposure-inventory.json`은 2026-09-28 현재 보존된 로컬 코드와 과거 실행 5,267개 파일을 해시로 기록했다. 감사 결과 주 window는 W2, 재현 window는 W3이며 둘의 제외 group은 0개다. W1의 알려진 노출 1,885개는 기록했지만 W1 조사는 미완료로 유지했다. 명령은 감사를 시작 전에 확인하고, A → C → (양성일 때 R) → B → S → 최종 보고를 순서대로 실행한다. C1 적격성 실패나 자원 한도에 도달하면 사전 판정 규칙에 따라 종료한다. 같은 명령을 다시 실행하면 봉인된 완료 단계를 건너뛰고 중단된 작업을 이어간다.

기존 `v5-study`의 A-impl은 이후 CLI 수정 전 소스 해시로 봉인되어 있다. 새 정식 실행에는 감사 결과를 저장해 둔 별도 `v5-study-certified` 경로를 사용한다.

```sh
PYTHONPATH=src .venv/bin/python -m dhi_v5.study run \
  --root local_experiment_archive/runs/v5-study-certified \
  --stage all --inventory /Users/choisoonwook/Experiments_local/DHI_AI_gen/examples/v5-exposure-inventory.json
```

단계별로 진행할 때는 다음 명령을 사용한다.

```sh
PYTHONPATH=src .venv/bin/python -m dhi_v5.study plan
PYTHONPATH=src .venv/bin/python -m dhi_v5.study check \
  --root local_experiment_archive/runs/v5-development-check --quick
.venv/bin/python -m pytest tests/test_study_v5.py -q
```

설치된 console script는 `hash-inverse-v5`다. `--quick`은 개발 검사이며 **A-impl PASS를 만들지 않는다**. Metal 접근이 차단된 실행 환경에서는 GPU 접근 가능한 터미널에서 실행한다. Torch/CPU 모델로 자동 대체하지 않는다.

정식 실행은 비어 있는 새 디렉터리를 지정한다. 같은 디렉터리에서 명령을 다시 실행하면 봉인된 완료 작업은 검증 후 사용하고, 중단된 stream은 같은 identity로 한 번만 재개한다.

```sh
PYTHONPATH=src .venv/bin/python -m dhi_v5.study run \
  --root local_experiment_archive/runs/v5-study-certified --stage A --phase impl
PYTHONPATH=src .venv/bin/python -m dhi_v5.study run \
  --root local_experiment_archive/runs/v5-study-certified --stage A --phase prof
PYTHONPATH=src .venv/bin/python -m dhi_v5.study run \
  --root local_experiment_archive/runs/v5-study-certified --stage A --phase dev
```

A-impl은 100,000 메시지 해시 검산, 4,096 후보 sampler 검산, 실제 optimizer 재개, codec·원장·CLP 검사, 시나리오당 20,000회 calibration을 실행한다. Planted fixture는 별도 validation namespace에서 prior Random 후보에 사전 계산 역상을 섞어 **실제 65,536×100 원장**으로 +0/+0.25%p 판정을 확인한다. 임시 원장은 검사 후 삭제하고 요약을 남긴다. 이 fixture의 hash 호출은 본실험 조건 데이터로 사용하지 않는다.

## 노출 감사와 봉인

`examples/v5-exposure-inventory.json`은 현재 보존된 로컬 범위의 감사 기록이다. 삭제되었거나 외부에 있는 과거 실험은 이 범위에서 복원할 수 없다. 감사 파일은 다음 필드를 사용한다.

- `scopes.code`, `scopes.archive`: inventory 파일 기준 상대 경로 또는 절대 경로. 실제 조사 범위를 모두 열거한다.
- `scope_complete`: 조사 범위가 완전할 때만 true.
- `reviewed_files`: 범위 내 `.py`, `.json`, `.jsonl`, `.sqlite`, `.db`, `.csv`, `.md` 파일 전체. 각 행에 `path`, `sha256`, `condition_use`를 기록한다. 분류는 `prefix`, `full`, `raw`, `toy`, `none`이다.
- `exposed_groups`: 필요한 경우 `{"W1": [0, 1, ...]}` 형태의 실제 노출 group 목록.
- `full`/`raw` 행은 `representatives`에 validation/test 대표 payload hex 배열 JSON 경로를 지정한다. 그 파일도 조사 범위 안에 포함되어야 한다. 해당 payload를 세 window로 투영해 제외한다.
- W1 조사도 완료되었을 때만 `w1_exposure_complete: true`. W1 미완료 또는 미노출 group 부족 시 W2를 선택한다. W2 및 재현 window의 코드/자료 인증까지 미완료이면 봉인을 거부한다.

```sh
PYTHONPATH=src .venv/bin/python -m dhi_v5.study audit \
  --root local_experiment_archive/runs/v5-study-certified --inventory /Users/choisoonwook/Experiments_local/DHI_AI_gen/examples/v5-exposure-inventory.json
PYTHONPATH=src .venv/bin/python -m dhi_v5.study run \
  --root local_experiment_archive/runs/v5-study-certified --stage A --phase freeze
PYTHONPATH=src .venv/bin/python -m dhi_v5.study run \
  --root local_experiment_archive/runs/v5-study-certified --stage A --phase qualify
```

봉인은 `protocol.frozen.json`, `protocol.seal.json`, `window.json`, 코드·환경 hash, A-prof/A-dev, audit, fallback을 연결한다. 수정된 protocol, 환경, 코드 또는 봉인 자료는 거부한다. Audit을 미리 완료해 두었다면 `run --stage A`로 모든 A 단계를 순서대로 실행할 수 있다.

## 주 시험과 보조 단계

```sh
PYTHONPATH=src .venv/bin/python -m dhi_v5.study run \
  --root local_experiment_archive/runs/v5-study-certified --stage C
```

C는 여섯 checkpoint를 봉인한 뒤 공통 trials를 뽑는다. 첫 look이 `EXTEND`이면 동일 trials에 seeds 3–5를 자동 추가하여 한 번만 재판정한다. `POSITIVE`이면 독립 verifier와 학습 메시지 중복 감사를 통과한 뒤 R을 실행한다.

```sh
# C가 POSITIVE인 경우에만 실행한다.
PYTHONPATH=src .venv/bin/python -m dhi_v5.study run \
  --root local_experiment_archive/runs/v5-study-certified --stage R

PYTHONPATH=src .venv/bin/python -m dhi_v5.study run \
  --root local_experiment_archive/runs/v5-study-certified --stage B
PYTHONPATH=src .venv/bin/python -m dhi_v5.study run \
  --root local_experiment_archive/runs/v5-study-certified --stage S
PYTHONPATH=src .venv/bin/python -m dhi_v5.study report \
  --root local_experiment_archive/runs/v5-study-certified
```

B/S는 C 및 필요한 R이 완료되기 전에 실행할 수 없다. C의 Main/Shuffled/Random trials·K·seed·updates는 fallback으로 줄이지 않는다. B·S 양성은 C3를 변경하지 않는다.

## 무결성·자원·결과

모든 후보는 `(protocol, stage/task, window, rung, method, seed, trial, attempt)`와 명시적 RNG identity로 구분된다. SQLite 원장은 invalid·duplicate·첫 성공 이후 후보까지 K개를 전부 기록한다. 별도 순회가 payload를 재해시하고 trial·counts를 다시 계산한다. 학습 SHA-256 집합과의 일치, @1/@10/@100, valid·duplicate·길이·NFE·시간·성공 집중도를 기록한다.

Checkpoint는 model과 Adam moments/step을 같은 safetensors에 저장한 후 atomic pointer로 전환한다. 재개 시 마지막 checkpoint 이후의 학습 원장을 되돌리고 같은 update stream을 재생한다. `work.json`은 재생 비용을 합산하며, 연산 중 죽은 batch는 전량을 보수적으로 계상한다. 미완료 결과를 실패 0으로 채우지 않는다.

단계별 cap과 필수 경로 50시간, 저장량/RSS/GPU 각각 64 GiB, 최소 여유 디스크 10 GiB를 검사한다. A-prof에서 학습·생성·prior MD5·원장 처리량과 저장량을 측정하여 계획의 fallback 순서대로 한 번만 고정한다. Cap 종료는 연구 효과의 음성 증거가 아니며 `NOT_ESTABLISHED_BY_BUDGET` 또는 C2 부분 측정으로 보고한다.

`decision.json`과 한국어 `FINAL_REPORT_KO.md`를 생성한다. 미완료 실행은 `INCOMPLETE / NOT_FINAL`이며 완료를 가장하지 않는다. 이전 look만 완료한 경우 그 구간을 보존한다. 원장·checkpoint·보고서는 `local_experiment_archive/`에만 저장하고 커밋하지 않는다.

MLX API 참고: [난수](https://ml-explore.github.io/mlx/build/html/python/random.html), [자동 미분과 vmap](https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html), [checkpoint 직렬화](https://ml-explore.github.io/mlx/build/html/usage/saving_and_loading.html).
