# Diffusion Hash 실험 검증 보고서

## 1. Repository 분석

현재 데이터 흐름은 다음과 같다.

```text
message: bytes (4..31 bytes)
  -> BGV / CGGE / Direct Bits encoder
  -> float32 tensor in {0,1}
  -> x * 2 - 1: float32 tensor in {-1,+1}
  -> q(x_t | x_0), condition, normalized timestep
  -> ImageUNet or BitDenoiser
  -> epsilon or x_0 prediction
  -> DDIM-style reverse sampling
  -> clamp [-1,1]
  -> (x + 1) / 2
  -> deterministic decoder and threshold quantization
  -> candidate bytes
  -> hashlib.new(algorithm, candidate)
  -> q-bit target prefix comparison
```

| 경계 | 구현 및 검증 결과 |
| --- | --- |
| Dataset | `SourceSpec`과 local `random.Random(seed)`로 unique message를 생성하고 정렬한다. 기본 길이는 4–31 byte다. Digest group 전체를 하나의 split에 배치하므로 message 및 q-bit digest condition이 split을 넘지 않는다. |
| Hash | Python `hashlib`로 full digest를 계산한다. q-bit condition은 digest 첫 byte부터 big-endian/MSB-first로 자른다. candidate 판정 시 다시 full digest를 계산한 뒤 동일 q-bit prefix를 비교한다. |
| BGV | `[2,32,128]`, `float32`, `{0,1}`. 첫 slot은 1-byte length, 이후 payload. 각 byte는 MSB-first 8 bit를 2×4 logical glyph와 4×4 block으로 확장한다. 두 번째 channel은 contiguous validity mask이며 나머지는 zero padding이다. |
| CGGE | `[2,32,64]`, `float32`, `{0,1}`. Printable ASCII만 허용하고 고정 8×8 glyph table과 contiguous mask를 사용한다. 별도 length byte 없이 mask로 길이를 복원하며 마지막 reserve cell은 invalid여야 한다. |
| Direct Bits | `[32,8]`, `float32`, `{0,1}`. 첫 row는 1-byte length, payload 및 zero padding을 모두 MSB-first로 저장한다. |
| Normalization | 모든 representation에 대해 `x -> 2x-1 -> (x+1)/2`가 exact하다. Decoder threshold는 unit range 0.5이며 normalized range에서는 0과 같다. |
| Condition | 기존 hash condition은 Direct Bits에서 digest bit vector, image에서 ASCII caption byte/127이다. 새 reversible control은 각 representation의 exact encoded tensor를 flatten하여 보존하고 image에서는 공간 channel로 reshape한다. |
| Model | Hash path는 기존 global-conditioned `ImageUNet`/dense `BitDenoiser`를 유지한다. Reversible image path는 같은 U-Net의 input에 encoded condition channel을 연결한다. Reversible Direct Bits path는 noisy bit, aligned condition bit, timestep을 받는 shared pointwise denoiser다. |
| Training | Adam, seeded batch/timestep/noise generator, MSE. 기존 epsilon target과 새 explicit x0/sample target을 모두 지원한다. |
| Scheduler | 선형 beta. 기존 100-step `beta_end=0.02`의 terminal `alpha_bar=0.363563`과 G1용 50-step `beta_end=0.4`의 terminal `alpha_bar=8.185905e-06`을 분리했다. |
| Sampling | Seeded Gaussian에서 시작하는 deterministic DDIM-style update다. epsilon 및 x0 prediction을 동일 `predicted_clean` 경계로 통합했고 마지막 tensor는 `[-1,1]`로 clamp한다. |
| Checkpoint | 각 stage가 config와 `model_state`를 `checkpoint.pt`로 저장한다. 기존 runner에는 별도 resume CLI는 없으며 저장 checkpoint는 PyTorch로 직접 load할 수 있다. |
| Evaluation | 기존 hash evaluator는 invalid decode도 K를 소비하며 exact source와 hash prefix match를 구분한다. G1 runner는 MSE, encoded bit/pixel accuracy, decoded byte accuracy, valid decode, exact message recovery를 추가 저장한다. |

## 2. 변경 파일

| 파일 | 변경 내용 | 이유 |
| --- | --- | --- |
| `src/diffusion_hash_inv/models.py` | explicit noise addition/x0 conversion, epsilon·x0 parameterization, configurable schedule, spatial reversible condition, aligned bit denoiser | train/sampling 수식 경계와 reversible condition 경로를 분리·검증하기 위해 |
| `src/diffusion_hash_inv/runner.py` | `beta_end`, `prediction_type`, representation-native reversible condition 및 model wiring | 기존 hash path를 제거하지 않고 G1 option을 제공하기 위해 |
| `src/diffusion_hash_inv/positive_control.py` | G1-A→N=4/16/64 G1-B→G1-C 순차 runner, fail-stop, metrics/checkpoint/diagnostic artifact | 기존 runner에는 single/small overfit gate가 없었기 때문에 |
| `examples/g1-{bits,bgv,cgge}.json` | seed 0 고정 G1 config | 같은 실험을 재실행하기 위해 |
| `tests/test_diffusion_pipeline.py` | deterministic codec/normalization, condition alignment, hash reference, forward/sampling, epsilon oracle regression | 수정된 수식과 경계를 자동 회귀 검증하기 위해 |
| `pyproject.toml` | `hash-inverse-g1` entry point | G1 runner 실행을 위해 |

## 3. Gate 결과

| Gate | 목적 | 결과 | 핵심 metric | 판정 |
| --- | --- | --- | --- | --- |
| G0 | Encoding integrity 및 전체 regression | 48/48 tests, independent oracle 2/2 | exact round-trip 및 normalization exact | PASS |
| G1-A | Single-sample overfit | Bits/BGV/CGGE 모두 3/3 sampling seed exact | exact=1.0, byte=1.0, valid=1.0 | PASS |
| G1-B | N=4→16→64 train memorization | 세 representation, 모든 N에서 exact | Exact Recovery Rate=1.0 | PASS |
| G1-C | 64 train / 16 validation / 16 unseen test | 세 representation 모두 train/validation/test exact | 각 split exact=1.0, 3 sampling seeds | PASS |
| G2 | Conditional dependence | 실행하지 않음 | 없음 | NOT RUN |
| G3 | Information ladder | 실행하지 않음 | 없음 | NOT RUN |
| G4 | Toy hash inversion | 실행하지 않음 | 없음 | NOT RUN |
| G5 | Candidate budget/random baseline | 실행하지 않음 | 없음 | NOT RUN |
| G6 | Full experiment | 실행하지 않음 | 없음 | NOT RUN |

G1-C의 세부 결과는 다음과 같다. 각 test metric은 16 unseen message × 3 sampling seed = 48회 복원 결과다.

| Representation | Train exact | Validation exact | Test exact | Test bit accuracy | Test byte accuracy | Test MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct Bits | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.00021157 |
| BGV | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.00033519 |
| CGGE | 1.0 | 1.0 | 1.0 | 0.99996948 | 1.0 | 0.00063714 |

## 4. 실제 실행 명령

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q --junitxml=output/g0-seed0/pytest.xml
.venv/bin/python -m pytest -q tests/test_encoding_independent.py
.venv/bin/python -m diffusion_hash_inv.positive_control --config examples/g1-bits.json --output output/g1-bits-seed0
.venv/bin/python -m diffusion_hash_inv.positive_control --config examples/g1-bgv.json --output output/g1-bgv-seed0
.venv/bin/python -m diffusion_hash_inv.positive_control --config examples/g1-cgge.json --output output/g1-cgge-seed0
.venv/bin/python -m compileall -q src tests
```

실험 공통 설정은 model seed 0, sampling seed 0/1/2, 1,500 optimizer steps, batch 16, learning rate 1e-3, 50 diffusion/sampling steps, x0 prediction, `beta_end=0.4`, CPU다. Image width는 8이며 Direct Bits pointwise hidden width는 32다. 각 stage의 정확한 command/config/dataset size/checkpoint는 해당 `metrics.json`에 함께 저장했다.

## 5. G1 상세 분석

기존 `output/md5-bgv-q8-s0-reversible-mps/checkpoint.pt`를 CPU에서 같은 config/seed로 다시 sampling했다. 25 target 모두 invalid decode였고 exact recovery는 0/25였다. 실패 사유는 `length_out_of_range` 21건, `mask_inconsistent` 4건으로 기존 보고서와 일치했다.

기존 실험은 이름과 달리 G1-A가 아니었다. 12,000-message dataset의 train split을 1,000 step 학습한 뒤 held-out target 25개를 평가했고, manifest의 `g1_round_trip`은 model positive control이 아니라 codec round-trip만 뜻했다.

격리 결과는 다음과 같다.

1. 기존 100-step schedule은 terminal `alpha_bar=0.363563`이었다. 학습의 마지막 상태에는 원본 신호가 크게 남지만 sampling은 순수 Gaussian에서 시작하므로 terminal train/sample distribution이 맞지 않았다.
2. Direct Bits 단일 sample에서 기존 epsilon/기본 schedule은 2,000 step 후 bit accuracy 0.457, exact 0이었다. terminal SNR만 낮춘 epsilon run도 5,000 step과 3 sampling seed에서 bit accuracy 0.395–0.477, exact 0이었다.
3. 낮은 terminal SNR과 x0 prediction을 함께 사용하자 단일 Direct Bits가 exact 복원됐다. epsilon 수식 자체는 oracle test에서 loss≈0 및 exact sampling을 통과하므로 구현 식의 불일치가 아니라 이 설정의 optimization/initial-distribution 문제로 격리됐다.
4. Image condition을 global channel bias로 압축하던 기존 경로는 arbitrary spatial byte/glyph 위치를 보존하기 어렵다. G1에서만 exact encoded condition을 공간 channel로 제공해 같은 U-Net의 train/sampling path를 통과시켰다.
5. Dense Direct Bits model은 N=64 train exact 1.0이지만 initial unseen exact 0으로 memorization/generalization이 분리됐다. 최종 control은 위치 공유 pointwise denoiser로 noisy value, condition, timestep을 모두 사용하며 unseen exact 1.0을 달성했다.

G1-C Diagnostic A–F에서도 no-noise, one-step, t=0/1/12/25/49 sweep, sampler bypass, decoder bypass를 분리했다. 모든 representation에서 모든 timestep의 bit accuracy가 1.0(BGV/Bits) 또는 0.999969 이상(CGGE)이었고 decoded exact recovery는 1.0이었다.

수정 전·후 수치는 같은 실험군의 paired comparison이 아니다. 수정 전 run은 12,000-message held-out control, 수정 후 G1은 1/4/16/64 ladder와 별도 16-message unseen control이다. 따라서 여기서 주장하는 것은 failure boundary 제거와 proper positive-control PASS뿐이다.

## 6. Hash 결과

G1 이전 결과는 strict gate 규칙상 hash inversion evidence로 재사용하지 않았다.

| K | Diffusion Exact@K | Diffusion HashMatch@K | Random HashMatch@K | Unique Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| 10 | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| 100 | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| 1000 | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

## 7. 발견된 문제

| Severity | 문제 | 상태 |
| --- | --- | --- |
| Critical | model positive control이 실제 G1-A/B/C 없이 codec round-trip gate로 표시됨 | 별도 fail-stop G1 runner로 수정 |
| Critical | 100-step schedule terminal signal과 pure-noise sampling start의 분포 불일치 | schedule을 config화하고 G1에서 near-zero terminal signal 사용 |
| Major | reversible image condition이 global bias로 압축되어 위치 정보를 직접 보존하지 못함 | G1 전용 spatial condition path 추가 |
| Major | dense Direct Bits가 작은 train set을 암기하지만 reversible identity mapping을 unseen data에 일반화하지 못함 | aligned pointwise denoiser로 격리 |
| Minor | 기존 main evaluator에는 bit/byte accuracy와 timestep diagnostics가 없음 | G1 artifacts에는 추가; G2 이후 main evaluator 통합은 아직 미수행 |
| Minor | candidate diversity metric은 아직 구현되지 않음 | G5 전 구현 필요 |

## 8. 연구적으로 가능한 결론

현재 코드에서 BGV, CGGE, Direct Bits의 codec/normalization은 exact하며, complete reversible condition과 명시된 x0/schedule 설정을 사용할 때 diffusion training/sampling/denormalization/decoder pipeline은 single sample, N=4/16/64 memorization, 16-message unseen reconstruction을 deterministic byte criterion으로 통과한다.

## 9. 아직 주장할 수 없는 내용

- 모델이 hash condition을 실제로 사용한다는 결론
- partial-information 또는 cryptographic digest에서 유효 candidate 확률을 높인다는 결론
- random/source-prior/exhaustive baseline보다 우수하다는 결론
- MD5/SHA-256 preimage capability 또는 full-scale generalization
- model seed 간 통계적 재현성: 이번 실행은 model seed 0 하나이며 sampling seed만 3개다.

따라서 핵심 질문인 “condition을 이용해 정답 후보 확률을 random baseline보다 높였는가?”에는 아직 **답할 수 없다**. 현재 판정은 G1 PASS, hash evidence NOT RUN이다.

## 10. 다음 권장 실험

다음 한 단계는 동일 G1 configuration에서 correct / deranged shuffled / zero condition을 같은 train/evaluation budget과 seed로 실행하는 G2다. Correct condition이 두 negative control보다 명확히 높지 않으면 G3로 진행하지 않아야 한다.

## 산출물

- `output/gate_summary.json`
- `output/g0-seed0/pytest.xml`, `output/g0-seed0/metrics.json`
- `output/g1-bits-seed0/`
- `output/g1-bgv-seed0/`
- `output/g1-cgge-seed0/`

