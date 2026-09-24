# GPU 학습·추론 실행

2026-09-24 확인: Apple M3 Max 40-core GPU, PyTorch 2.14.0의 **MPS**에서 Gaussian·Discrete 학습과 추론이 동작한다. Python/Torch 재설치나 추가 패키지는 필요하지 않았다.

기존 experiment 및 G1 CLI에 `--device`를 추가했다. 설정에 device가 없으면 `auto`이며 CUDA → MPS → CPU 순서로 선택한다. 기존 JSON에 `"device": "cpu"`가 있으면 그대로 CPU를 사용하므로 GPU 실행에는 `--device mps`를 지정한다. 명시적으로 요청한 GPU가 보이지 않으면 오류를 내며 CPU로 바꾸지 않는다.

## 작은 GPU 실행 확인

프로젝트 루트에서 새 출력 폴더를 사용한다.

```sh
.venv/bin/python -m diffusion_hash_inv.experiment_cli \
  --config examples/gpu-smoke.json \
  --device mps \
  --output local_experiment_archive/runs/my-gpu-smoke
```

이 설정은 학습 2 steps, 표적 2개, 표적당 후보 2개만 실행한다. Gaussian schedule은 1,000, sampling은 100 steps다. 실제 모델 학습·생성·CPU decode·checkpoint·ledger 경로를 확인하는 engineering fixture이며 성능 평가용 학습량이 아니다. 모든 후보가 invalid여도 GPU 동작 실패를 뜻하지 않는다.

`run_manifest.json`의 `execution_device: "mps:0"`를 확인한다. 자동 선택 결과는 `configuration_frozen.json`에도 기록되므로 같은 run을 다른 backend에서 재개하면 설정 불일치로 차단된다. CPU 비교는 새 출력 폴더와 `--device cpu`를 사용한다.

기존 실험 설정도 같은 CLI로 실행할 수 있다. Legacy G1 실행에는 다음 옵션을 사용한다.

```sh
.venv/bin/python -m diffusion_hash_inv.positive_control \
  --config examples/g1-bgv.json --device mps \
  --output local_experiment_archive/runs/my-g1-mps
```

Legacy G1은 원문 표현을 조건으로 받는 기존 과제다. **v2의 12-bit same-path positive control을 대신하지 않는다.** `poc-v2-protocol.json`은 이 CLI의 ExperimentConfig가 아니다.

## 실제 검증과 한계

- GPU/CPU backend tests: **12 passed**, CUDA tests **5 skipped**. 다섯 pipeline의 실제 optimizer update, finite weights/loss, checkpoint 복원, RNG 복원 후 동일 sampling을 확인했다.
- 전체 tests: **108 passed, 5 skipped in 9.26s**. GPU가 없는 환경에서는 MPS tests도 skip되므로 통과 수와 skip 이유를 함께 확인해야 한다.
- CLI smoke: `mps:0`, 학습 2 steps, 4 attempts, checkpoint/manifest 생성 완료. Training loss 1.03615, 학습 구간 약 1.61초는 작은 smoke의 수치이며 전체 학습 시간 예측치가 아니다.
- 별도 primitives 확인: condition dimension 12로 BGV Gaussian 1,000/100 schedule 및 Random Bytes Discrete 32-step sampling도 GPU에서 finite output을 생성했다.

검사 재실행:

```sh
.venv/bin/python -m pytest -q tests/test_devices.py
```

[Backend 검사 기록](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/2026-09-24-gpu-validation/backend-tests.xml) · [GPU CLI 실행 기록](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/2026-09-24-gpu-validation/bgv-smoke/run_manifest.json)

최초 제한된 실행 프로세스에서는 `is_built=True`, `is_available=False`였으나 호스트 실행에서는 `True`였고 실제 MPS tensor 연산도 성공했다. 같은 오류가 나면 GPU 접근이 가능한 호스트 터미널에서 위 명령을 실행한다. Codex의 실행 제한 아래에서는 해당 GPU 명령에 호스트 실행 권한이 필요할 수 있다. GPU가 없다고 단정하거나 PyTorch를 재설치할 근거는 아니다. [PyTorch MPS 공식 문서](https://docs.pytorch.org/docs/stable/notes/mps.html)

Model·loss·reverse sampler는 GPU에서 실행하며, 생성 tensor를 CPU로 옮긴 뒤 기존 strict decoder와 hash verifier를 실행한다. Accelerator 작업 완료 시점을 기준으로 시간을 측정한다. 모델 구조·학습 objective·과학적 판정 기준은 이번 GPU 연결 작업에서 바꾸지 않았다.

같은 backend·환경에서 RNG 복원을 검사했으며 CPU와 MPS 사이의 bitwise 동일성이나 모든 PyTorch 버전의 동일성을 보장하지 않는다. CUDA 분기는 장치 선택 로직만 검사했고 실제 CUDA 학습은 검증하지 못했다. MPS 전체 학습·추론 비용 및 최적 batch는 별도 profile이 필요하며 이전 약 118시간 CPU 추산을 GPU 예상 시간으로 사용하면 안 된다.
