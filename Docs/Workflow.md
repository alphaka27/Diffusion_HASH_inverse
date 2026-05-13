# Diffusion Hash Inverse Workflow

이 문서는 레포의 전체 실행 흐름을 기준으로, 해시 중간 로그 생성부터 RGB 이미지 데이터셋 생성, 분석, DDPM 학습, 산출물 확인까지의 워크플로우를 정리한다.

## 1. Environment

프로젝트 루트에서 설치한다.

```bash
pip install -r requirements.txt
pip install -e .
```

DDPM 학습까지 실행하려면 PyTorch/Pillow 의존성을 포함한다.

```bash
pip install -e ".[train]"
```

MLX 예제를 실행하려면 다음 의존성을 사용한다.

```bash
pip install -e ".[mlx]"
```

현재 `pyproject.toml`의 `diffhash` 콘솔 스크립트는 `diffusion_hash_inv.cli:main`을 가리키지만, 해당 `cli.py`는 아직 없다. 지금은 `python -m diffusion_hash_inv...` 형태나 Python API로 실행하는 경로를 기준으로 사용한다.

## 2. End-to-End Data Flow

```text
RuntimeConfig
  -> MainEP
  -> NBitsGenerator
  -> hashing.MD5.digest
  -> validate(..., hashlib.md5)
  -> FileIO writes binary input and JSON trace logs
  -> RGBImgMaker converts latest JSON traces to PNG image data
  -> Analyze / BetaScheduleAnalyzer can stream JSON traces
  -> DDPM trainers read data/images plus output/json
  -> output/<training-run> receives configs, checkpoints, samples, traces
```

주요 산출물은 다음 경로에 생긴다.

| Stage | Output |
| --- | --- |
| Message/bit generation | `data/binary/*.bin` |
| Hash trace logging | `output/json/<run-start>/<HASH>_<LENGTH>_<run-start>_<index>.json` |
| RGB image conversion | `data/images/<json-stem>/message.png` and step/round/loop PNG files |
| Analysis | in-memory summaries from `output/json` |
| DDPM training | `output/<model-run>/{train_config.json,beta_schedule.json,checkpoints,samples,process_traces}` |

## 3. Generate Hash Logs

권장 실행 방식은 Python API로 `RuntimeConfig`를 명시하는 것이다. 아래 예시는 128-bit 입력을 순차 값으로 만들고 MD5 trace JSON을 생성한다.

```python
from diffusion_hash_inv.config import (
    Byte2RGBConfig,
    HashConfig,
    MainConfig,
    MessageConfig,
    OutputConfig,
)
from diffusion_hash_inv.main import MainEP, RuntimeConfig

length = 128

runtime_config = RuntimeConfig(
    main=MainConfig(
        verbose_flag=False,
        clean_flag=False,
        debug_flag=False,
        make_image_flag=False,
    ),
    message=MessageConfig(
        message_flag=False,
        length=length,
        random_flag=False,
        seed_flag=True,
    ),
    hash=HashConfig(hash_alg="md5", length=length),
    output=OutputConfig(),
    rgb=Byte2RGBConfig(),
)

MainEP(runtime_config).run(iteration=256, mode="sequential")
```

동작 흐름:

1. `MainEP._loop_preprocess()`가 `Metadata`, `BaseLogs`, `StepLogs`, 해시 알고리즘 인스턴스를 준비한다.
2. 각 iteration에서 `NBitsGenerator`가 입력 바이트를 만든다.
3. `MD5.digest()`가 step별 중간 상태를 `StepLogs`에 기록한다.
4. `validate()`가 `hashlib.md5` 결과와 비교한다.
5. `FileIO.file_writer()`가 binary input과 JSON trace를 저장한다.

주의할 점:

- `MessageConfig.length`와 `HashConfig.length`는 양수이면서 8의 배수여야 한다.
- `random_flag=False`이면 iteration index가 입력값으로 사용된다.
- `random_flag=True`이면 iteration마다 난수 입력을 생성한다.
- `MainConfig.clean_flag=True`는 `data/`와 `output/` 아래 생성물을 지운 뒤 실행한다.
- 현재 `src/diffusion_hash_inv/hash_main.py`의 CLI 인자는 파싱되지만, 마지막에 하드코딩된 `main()`을 호출하므로 인자값이 실제 실행에 반영되지 않는다.

## 4. Generate RGB Image Dataset

로그 생성과 이미지 생성을 한 번에 수행하려면 `make_image_flag=True`로 실행한다.

```python
runtime_config = RuntimeConfig(
    main=MainConfig(
        verbose_flag=False,
        clean_flag=False,
        debug_flag=False,
        make_image_flag=True,
    ),
    message=MessageConfig(message_flag=False, length=128, random_flag=False),
    hash=HashConfig(hash_alg="md5", length=128),
    output=OutputConfig(),
    rgb=Byte2RGBConfig(seed_flag=False, input_seed=42),
)

MainEP(runtime_config).run(iteration=256, mode="sequential")
```

이미 만들어진 최신 JSON 로그만 이미지로 변환하려면 같은 `hash_alg`/`length` 설정으로 `rgb_image_maker()`를 호출한다.

```python
entrypoint = MainEP(runtime_config)
entrypoint.rgb_image_maker()
```

이미지 변환 흐름:

1. `RGBImgMaker.main()`이 `output/json`에서 해당 hash/length의 최신 run JSON을 찾는다.
2. `Logs.log_parser()`와 `Logs.steplogs_parser()`가 `Message`와 `Logs` leaf 값을 파싱한다.
3. `Byte2RGB.rgb_encoder()`가 byte/hex 값을 RGB tuple로 인코딩한다.
4. `RGBImgMaker.image_formatter()`가 `ImgConfig.img_size` 기준 PNG를 만든다.
5. `data/images/<json-stem>/message.png`와 step/round/loop PNG가 저장된다.

DDPM 학습의 기본 입력은 `data/images/<run-id>/message.png`이다. loop 이미지 학습을 켜는 경우에는 `NNth Loop.png` 패턴의 파일도 사용한다.

## 5. Analyze JSON Traces

JSON 로그 개수 확인:

```bash
python -m diffusion_hash_inv.analyze.analyze -f output/json -q
```

Step log에서 beta schedule 통계를 스트리밍 방식으로 계산할 수 있다.

```python
from diffusion_hash_inv.analyze.analyze import Analyze

summary = Analyze("output/json").summarize_beta_schedules(step_name="4th Step")
print(summary.count, summary.length)
print(summary.mean)
```

`BetaScheduleAnalyzer`는 step log leaf 값을 byte sequence로 읽고 누적합 schedule을 만든다. 큰 JSON 세트도 전체를 메모리에 올리지 않고 처리하도록 설계되어 있다.

## 6. Train DDPM Models

모든 학습 명령은 `data/images`와 `output/json`을 입력으로 사용한다. 빠른 확인에는 `--max-images`, 작은 `--image-size`, 작은 `--train-steps`, 작은 `--timesteps`를 먼저 쓴다.

### Unconditional DDPM

라벨 없이 `message.png` 이미지만 사용한다.

```bash
python -m diffusion_hash_inv.models.unconditional_ddpm \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/unconditional_ddpm_smoke \
  --max-images 256 \
  --image-size 32 \
  --batch-size 8 \
  --train-steps 2 \
  --timesteps 4 \
  --base-channels 8 \
  --time-dim 16 \
  --device cpu
```

### Conditional DDPM

`message.png`를 입력 이미지로 쓰고, matching JSON의 final hash를 condition label로 사용한다.

```bash
python -m diffusion_hash_inv.models.conditional_diffusion \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/conditional_diffusion_smoke \
  --label-source final-hash \
  --max-images 256 \
  --image-size 32 \
  --batch-size 8 \
  --train-steps 2 \
  --timesteps 4 \
  --base-channels 8 \
  --time-dim 16 \
  --beta-schedule linear \
  --save-train-batches-every 1 \
  --device cpu
```

주요 옵션:

- `--fit-mode reshape`: 이미지를 flatten한 뒤 같은 면적의 정사각형으로 재배열한다.
- `--fit-mode height-flatten`: `ImgConfig.img_size` 단위 블록을 정사각형 배열로 재배치한다.
- `--beta-schedule linear`: 일반 DDPM linear beta schedule.
- `--beta-schedule file`: 외부 JSON/TXT/CSV/NPY/NPZ beta 값을 사용한다.
- `--beta-schedule hash-approach1` 또는 `hash-approach2`: hash trace 기반 schedule을 사용한다.
- `--save-process-traces`: forward noising과 reverse denoising PNG trace를 저장한다.
- `--use-loop-images`: `NNth Loop.png` 이미지와 loop temporal metadata를 사용한다.

### Guided Conditional DDPM

Classifier-free guidance:

```bash
python -m diffusion_hash_inv.models.guided_conditional_diffusion \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/guided_conditional_diffusion_cfg_smoke \
  --label-source final-hash \
  --guidance-mode classifier-free \
  --guidance-scale 2.0 \
  --condition-dropout 0.1 \
  --max-images 256 \
  --image-size 32 \
  --batch-size 8 \
  --train-steps 2 \
  --timesteps 4 \
  --base-channels 8 \
  --time-dim 16 \
  --device cpu
```

Classifier guidance:

```bash
python -m diffusion_hash_inv.models.guided_conditional_diffusion \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/guided_conditional_diffusion_classifier_smoke \
  --label-source final-hash \
  --guidance-mode classifier \
  --guidance-scale 1.0 \
  --classifier-base-channels 32 \
  --max-images 256 \
  --image-size 32 \
  --batch-size 8 \
  --train-steps 2 \
  --timesteps 4 \
  --base-channels 8 \
  --time-dim 16 \
  --device cpu
```

### Structured Loop-Conditioned DDPM

`Logs/4th Step/1st Round`의 `Loop Start`, `1st Loop` ... `64th Loop`, `Loop End` 상태를 A/B/C/D word tensor로 변환해 condition으로 넣는다.

```bash
python -m diffusion_hash_inv.models.loop_conditioned_diffusion \
  --data-root data/images \
  --json-root output/json \
  --output-dir output/loop_conditioned_diffusion_smoke \
  --condition-step "4th Step" \
  --condition-round "1st Round" \
  --loop-count 64 \
  --max-images 256 \
  --image-size 32 \
  --batch-size 8 \
  --train-steps 2 \
  --timesteps 4 \
  --base-channels 8 \
  --time-dim 16 \
  --device cpu
```

## 7. Inspect Training Outputs

학습이 끝나면 run별 `output-dir`에서 다음을 확인한다.

| Path | Meaning |
| --- | --- |
| `train_config.json` | 실행 설정 snapshot |
| `condition_to_idx.json` | conditional/guided label mapping |
| `beta_schedule.json` | 실제 사용한 beta schedule |
| `checkpoints/step_*.pt` | 모델/optimizer checkpoint |
| `samples/final.png` | 최종 generated sample grid |
| `samples/final.source.png` | source image grid |
| `samples/final.with_source.png` | source/generated 비교 grid |
| `train_batches/step_*.png` | 실제 training batch grid |
| `process_traces/forward` | forward noising trace |
| `process_traces/reverse` | reverse denoising trace |

## 8. Verification

기본 테스트:

```bash
python -m pytest
```

학습 파이프라인만 빠르게 확인하려면 smoke 설정으로 먼저 실행한 뒤, 생성된 `samples/final.png`, `train_config.json`, `beta_schedule.json`이 기대한 `output-dir`에 있는지 확인한다.

## 9. Recommended Full Run Order

1. 환경 설치: `pip install -e ".[train]"`
2. 작은 iteration으로 MD5 JSON 로그 생성
3. `make_image_flag=True` 또는 `rgb_image_maker()`로 PNG 데이터셋 생성
4. `python -m diffusion_hash_inv.analyze.analyze -f output/json -q`로 JSON 인덱스 확인
5. `unconditional_ddpm` smoke run 실행
6. `conditional_diffusion` smoke run 실행
7. 필요한 guidance 또는 loop-conditioned trainer 실행
8. `samples/`, `checkpoints/`, `process_traces/` 확인
9. smoke가 통과하면 `--max-images` 제한을 제거하고 `--epochs` 또는 큰 `--train-steps`로 본 학습 실행
