# v3.1 구현 현황

2026-09-27 · **모델·개발 P0/P1/P2 구현 완료 / 전체 IMPLEMENTATION_READY는 아직 아님**

[실험 계획](RESEARCH_PLAN_V3_1.md)의 모델 및 실행 기반을 구현했다. 현재 실행 가능한 범위는 **개발용 P0/P1/P2와 보고서**이며, 정식 PoC 적격성이나 본실험 완료를 뜻하지 않는다. 기존 v3 자료·checkpoint는 변경하지 않았다.

| 영역 | 구현 상태 |
|---|---|
| G0/G1/G2 | 기존 epsilon 모델, 고정 좌표 입력, x0 예측 및 sampling 정합성 구현 |
| D0/D1 | 기존 Token 모델, 길이 head + 길이 조건부 payload diffusion 구현 |
| MLX backend | G0/G1/G2/D0/D1의 native MLX 모델·학습·생성·저장/복구 및 개발 P0/P1/P2 CLI 구현 |
| Profile 식별 | 13개 pipeline/profile 조합, 초기화·checkpoint·ledger 식별 및 NFE 기록 |
| 개발 P0 | Codec·모델·sampler·입력 경계 등 75개 검사; 실제 MD5 본실험 경계 검사는 남음 |
| 개발 P1 | 26개 learned runs와 2개 shared Random streams; update5 및 attempt7 중단/복구 검사 |
| 예산 기반 | v3.1 soft 경고/hard 중단 분리, 누적 PoC 시간, 쓰기 예약 및 제한된 디렉터리 검사 |
| 시간 측정 | Budget check·입력 준비·연산·telemetry를 포함한 update 전체 시간 기록; validation/checkpoint 별도 |
| 부분 보고 | Stage gate가 없어도 완료 run의 seal을 검증하고 COMPLETE/INCOMPLETE/NOT_RUN 표시 |
| P2A/P2B | 구현: 작은 과적합 검사, epoch별 개발 probe·세부 진단, 순차 profile 선택 및 잠정 자원 측정 |
| 자원 확정·E0 | 미구현: 전체 비용 추정/봉인, 노출 감사, production MD5 리허설 |
| P3·M0–M3·calibration | 미구현; 정식 실행 차단 |
| Run 단위 오류 후 독립 실행 계속 | 미구현; 현재 실패 시 부분 결과를 보존하고 stage 중단 |

현재 CLI는 수정되지 않은 v3.0/v3.1 JSON만 받는다. v3.1의 정식 실행과 P3 실행 요청은 출력 디렉터리를 만들기 전에 차단한다. Protocol JSON의 `*_at_authoring` 및 `implementation_status`는 계획 작성 당시 기록이며, 현재 구현 상태는 코드의 `readiness()`와 실행 디렉터리의 `implementation_readiness.json`에 기록한다.

## 기존 PyTorch 구현 검증 결과

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

다음 구현 순서는 exposure/MD5 실행기·E0·최종 자원 확정·통계 calibration, P3 및 본실험 진입 gate다. 정식 실행 차단은 이 계약들이 실제 구현·검증된 뒤 해제한다.

## MLX 구현 및 실행

[mlx_models.py](src/diffusion_hash_inv/mlx_models.py)는 이미지 U-Net, sequence denoiser, Gaussian DDIM 및 masked categorical diffusion을 MLX로 구현한다. G1/G2의 고정 좌표, G2의 clean x0 target·clipping·epsilon 재구성, D1의 길이 head·payload 조건·EOS/PAD 고정 context를 포함한다. 학습은 MLX autograd와 Adam을 사용한다. 기존 학습 설정을 유지하도록 Adam의 `bias_correction=True`를 명시했다([MLX Adam 문서](https://ml-explore.github.io/mlx/build/html/python/optimizers.html)).

[mlx_backend.py](src/diffusion_hash_inv/mlx_backend.py)는 기존 Pilot의 데이터·예산·로그·복구 흐름에 연결한다. PyTorch는 기존 CPU codec과 decoder 경계에서 사용한다. 모델 forward/backward, loss, optimizer, sampling은 MLX 연산이다. 외부 이미지 형태는 NCHW, convolution 내부는 NHWC다. 모델의 parameter 수와 초기화 분포를 기존 구현에 맞췄다.

체크포인트는 MLX 가중치와 optimizer 상태를 `.safetensors`에 저장한다. Train/validation/trajectory마다 64-bit seed에서 explicit MLX key를 만들며, 재개 시 저장된 epoch·offset·update와 실행 식별자로 동일 key를 복원한다. 실행 manifest와 candidate ledger에 backend를 기록한다. **서로 다른 backend의 PRNG 및 부동소수점 연산 결과는 동일하지 않으며, 기존 PyTorch checkpoint의 MLX 재개는 지원하지 않는다.** Backend·코드·환경이 바뀌면 새 workdir를 사용한다.

Apple silicon macOS의 기존 가상환경에 설치:

```bash
uv pip install --python .venv/bin/python -e '.[mlx]'
```

새 환경을 lockfile로 설치할 때는 `uv sync --locked --extra mlx`를 사용한다. MLX는 선택 의존성이며, 이번 구현의 lockfile은 `mlx==0.32.2`, `mlx-metal==0.32.2`를 기록한다.

개발 P0를 먼저 실행하고 같은 디렉터리에서 P1을 실행한다:

```bash
.venv/bin/hash-inverse-study pilot \
  --protocol examples/poc-v3.1-protocol.json \
  --workdir local_experiment_archive/runs/v31-mlx-dev \
  --stage P0 --backend mlx --device gpu --development

.venv/bin/hash-inverse-study pilot \
  --protocol examples/poc-v3.1-protocol.json \
  --workdir local_experiment_archive/runs/v31-mlx-dev \
  --stage P1 --backend mlx --device gpu --development

.venv/bin/hash-inverse-study report \
  --protocol examples/poc-v3.1-protocol.json \
  --workdir local_experiment_archive/runs/v31-mlx-dev
```

`--backend mlx`의 기본 device는 Metal `gpu`다. `mps`도 별칭으로 받으며 CPU 검증은 별도 workdir에서 `--device cpu`로 실행한다. GPU를 사용할 수 없으면 실패하며 CPU로 자동 전환하지 않는다. 기본 backend는 기존 `torch`다. MLX 역시 v3.1 개발 P0/P1/P2를 지원하며, P3와 정식 PoC·본실험 진입은 여전히 차단된다. 기존에 봉인한 실험 명세 JSON은 변경하지 않았다.

### MLX 검증 결과

- 전체 pytest: **130 passed, 5 skipped**. 기존 PyTorch 회귀 검사와 MLX CPU/Metal 검사를 포함한다.
- 수정되지 않은 v3.1 명세로 MLX Metal GPU의 개발 P0 **PASS**: 검사75개, codec 왕복1,214개, 모델 조합13개. Gaussian 100-step·discrete 32-step sampling을 실행했다. Gaussian batch 간 최대 절대 차이는 약 `2.02e-4`이며 기존 허용오차와 동일 decoder 결과 요건을 통과했다. 동일 key 반복은 완전 일치했다.
- 13개 pipeline/profile 조합에 동일한 가중치·입력·corruption을 주입해 forward·loss·gradient를 PyTorch와 비교했다. Adam은 동일 gradient를 입력하여 bias correction을 포함한 1회 업데이트를 비교했다. 프레임워크마다 계산한 미세한 gradient 차이를 Adam 비교에 섞으면 0 근처에서 오차가 증폭되므로 두 검사를 분리했다.
- CPU/Metal 각각 축소한 P0/P1 통합 검사에서 26개 learned runs·2개 Random streams, native safetensors 저장, update5·attempt7 중단 후 exact recovery를 확인했다. 이때 PyTorch 모델 forward와 optimizer 호출을 금지해 MLX 모델 연산의 독립성을 검사했다.
- D1의 길이/payload RNG 분리, 모든 생성 결과의 EOS/PAD 유효성, 빈 payload mask의 length CE, 잘못된 입력·nonfinite 출력 거부, backend가 다른 실행의 재개 거부를 검사했다.
- 원본 규모 P1은 실행하지 않았다. 개발 검사 통과는 생성 품질이나 정식 PoC 적격성을 의미하지 않는다.

증거: [MLX 테스트 코드](tests/test_mlx.py), [최종 전체 테스트 XML](local_experiment_archive/analyses/v31-mlx-tests-20260926-final.xml), [MLX GPU P0 보고서](local_experiment_archive/runs/v31-mlx-p0-20260926/report.md). MLX 설치 전 환경에서는 MLX 전용 검사를 skip한다.


## P2A/B 구현 및 실행 (2026-09-27)

검증: 전체 pytest **137 passed, 5 skipped**. [테스트 XML](local_experiment_archive/analyses/v31-p2-tests-20260927.xml)과 [P2 검사 코드](tests/test_study_p2.py)를 보존했다. PyTorch CPU·MLX Metal에서 축소한 실제 P2A/B 학습·생성, 학습 및 epoch probe 중단/재개, 선택 batch 복구, 진단과 봉인 변경 탐지를 검사했다. MLX P2 학습 경로는 PyTorch 모델/optimizer 호출을 금지한 상태에서도 통과했다. 이 검사의 축소 규모와 완화한 내부 통과 기준은 정식 성능 증거가 아니다.

`pilot --stage P2 --development`가 pipeline별 등록 순서 G0→G1→G2 / D0→D1을 관리한다. 각 후보는 P2A→P2B→잠정 자원 조건을 통과해야 선택되며, 선택 뒤 남은 후보는 `NOT_NEEDED`로 남긴다. 모든 후보가 실패하면 `BLOCKED_DEVELOPMENT`와 exit2를 반환한다. 낮은 품질로 실패한 후보를 같은 seed로 다시 돌리지 않는다.

- P2A: source별 training corpus에 양쪽이 있는 첫8 complement pairs에서 최초 record16개를 고정한다. Seed99·batch16·2,000 updates의 최종 weights로 정상/반전 각64회 K1을 평가한다. 각61회 이상 joint, 반전 원래 조건 오성공1회 이하를 요구한다.
- P2B: fresh seed100·100 epochs. Epoch10/30/100마다 최소 고정 validation loss checkpoint를 평가한다. 초기 joint0은 경고로 남기고 학습을 계속한다. 최종 정상/반전 joint·valid 각각122/128 이상, 원래 조건 오성공3/128 이하가 기준이다.
- 진단: variant별 strict validity·prefix 위치별 정답 수·valid 조건부 정확도, Gaussian noise/x0/영역별 MSE·clipping, Discrete mask별 logits·정답 확률·EOS 분포를 보존한다. Argmax 진단은 성공 집계에 포함하지 않는다.
- 자원: warm-up20 이후 세100-update 전체 주기 평균 중 최댓값, batch1/4/16/64 전체 생성·검증·SQLite 주기, 최대 길이 decoder/ledger stress를 측정한다. 1% 처리량 동률은 작은 batch를 선택한다. Profile/batch 선택과 자원값은 개발용 잠정 결과이며 `resources.json`의 `final_sealed=false`를 유지한다. 실제 MD5 본실험 비용은 E0에서 확정해야 한다.
- 복구: 중단된 학습과 미완료 probe를 이어 수행한다. 완료 probe/run의 봉인을 검증하고, 선택 batch에서 생성 중단·복구를 추가 검사한다. 수치/개별 run 시간 오류는 결과를 보존하고 global budget과 backend가 정상일 때 독립 후보를 계속한다. Hard global cap·무결성 오류는 중단한다.
- 보고: `pilot/P2/<pipeline>/<profile>/A|B/`, B의 `probes/epoch-*`, `profile_selection.json`, `profile.frozen.json`, `resources.json`에 저장한다. 부분 보고에서도 완료한 probe의 봉인을 검증한다. P1의 반전 결과는 `NOT_MEASURED`로 표시한다.

코드 hash가 변경되었으므로 기존 `v31-mlx-20260926-232405`에는 이어 쓰지 않는다. **새 workdir에서 변경된 코드의 P0/P1을 먼저 실행한 뒤 같은 workdir의 P2로 진행한다.** 아래 명령은 축소 검사가 아니라 등록된 원본 규모를 실행한다. P2는 최대24시간의 기존 hard cap을 유지한다.

```bash
for stage in P0 P1 P2; do
  .venv/bin/python -m diffusion_hash_inv.study_cli pilot \
    --protocol examples/poc-v3.1-protocol.json \
    --workdir local_experiment_archive/runs/v31-mlx-p2-20260927 \
    --stage "$stage" --backend mlx --device gpu --development || break
done
```

쓰기 없는 계획 조회는 같은 명령에서 `--stage P2 --dry-run`을 사용한다. 정상적으로 보존된 중단만 동일 명령에 `--resume`를 추가해 한 차례 재개할 수 있다. Code/environment 변경, 미계수 시간이 있는 강제 종료, hard cap 소진은 재개를 차단한다. 이 구현 작업에서는 원본 규모 P2 학습을 실행하지 않았다.

## P2A 10,000-update 개발 개정 (2026-09-27)

최신 P2A 실패에 대한 첫 수정 묶음이다. [추가 명세](examples/poc-v3.1-p2a10k-protocol.json)의 protocol ID는 `dhi-v3.1-p2a10k-20260927`이며, 실행기 schema revision은 `3.1`을 유지한다. 원본 v3.1 명세와 이미 봉인된 실행 결과는 변경하지 않는다. 추가 명세 역시 정확한 SHA-256 whitelist로 검증하므로 임의 기준 완화·설정 변경은 CLI에서 거부한다.

- P2A를 fresh weights에서 **10,000 updates** 학습한다. 정상·반전 joint 각각 61/64 이상, 반전 원래 조건 오성공 1 이하라는 기준은 그대로다. 학습률·모델·손실 가중치·sampler·P2B 설정도 그대로다.
- 2,000·5,000 updates에서 중간 생성 평가와 진단을 남기며, 이를 선택·조기 종료·P2B 진입에 사용하지 않는다. 최종 10,000-update checkpoint만 P2A gate에 사용한다. 중간 진단으로 추가되는 후보는 실행당 256개이고, 최종 후보 128개와 분리 보관한다. 후보당 NFE는 기존 profile과 같다.
- 비교에서 자료와 초기화까지 바뀌는 것을 피하려고 `seeds.protocol_namespace`를 원본 protocol ID로 고정했다. 기존 namespace·seed labels·case selection을 공유하되, 새 protocol ID/hash와 workdir로 실행 정체성을 구분한다. 학습 자료를 공유하는 개발 비교이므로 새로운 미관측 조건에 대한 검증으로 주장하지 않는다.
- 모든 P2A 최종 평가에 실제 선택된 학습 사례의 `diagnostics.json`을 남긴다. Gaussian은 timestep별 noise/x0/mask/padding/실제 payload 영역/BGV header 오차를 기록한다. Discrete는 mask 비율별 정답 확률과 prefix 위치별 정답 확률을 기록한다.
- D1의 `telemetry.jsonl` update 행에 같은 forward와 corruption에서 계산한 `length_ce`, `payload_ce`를 추가했다. 기존 합산 loss와 gradient는 유지한다. 진단에는 조건별 길이 분포·관측 길이 확률·argmax 길이 및 실제 생성 길이/prefix 성공/학습 메시지 일치 여부를 남긴다. 학습 메시지와 길이가 다르다는 것 자체를 실패로 세지 않는다.
- 기존 P2B 진단은 validation 자료를 계속 사용한다. P2A 진단은 training 자료를 사용하며 split과 자료 hash를 기록한다. Argmax·teacher-forced 값은 candidate 성공에 포함하지 않는다.
- 중간 평가 전 checkpoint를 저장하고, 중단된 probe를 학습 재개 전에 완료한다. 완료 probe는 봉인을 검증해 재사용한다. 원본 및 개정 명세의 P2A 모두 최종 품질 미달에 대한 warning을 기록한다.

저장 위치는 `pilot/P2/<pipeline>/<profile>/A/probes/update-00002000/`, `update-00005000/`이다. 최종 평가·진단은 기존 `A/` 루트에 저장된다. 중간 결과는 보고서에 `diagnostic only`로 표시되며, 단계가 중단돼도 완료된 probe를 확인할 수 있다. 기존 최신 run은 코드 hash가 달라졌으므로 새 코드로 재개하지 않는다.

계획 조회(학습 실행 없음):

```bash
.venv/bin/python -m diffusion_hash_inv.study_cli pilot \
  --protocol examples/poc-v3.1-p2a10k-protocol.json \
  --workdir local_experiment_archive/runs/v31-p2a10k-plan \
  --stage P2 --backend mlx --device gpu --development --dry-run
```

원본 규모 개발 실행은 새 workdir에서 P0부터 수행한다. P2A를 통과한 후보는 기존 규칙대로 P2B까지 자동 진행하므로, 아래 명령은 P2A만 실행하는 명령이 아니다.

```bash
DHI_RUN="local_experiment_archive/runs/v31-p2a10k-$(date +%Y%m%d-%H%M%S)"
for stage in P0 P1 P2; do
  .venv/bin/python -m diffusion_hash_inv.study_cli pilot \
    --protocol examples/poc-v3.1-p2a10k-protocol.json \
    --workdir "$DHI_RUN" --stage "$stage" \
    --backend mlx --device gpu --development || break
done
```

이번 수정에는 조건부 후속 후보인 length-head 전용 학습률, 추가 조건 주입층, Gaussian 영역별 가중 손실, argmax sampler를 적용하지 않았다. 먼저 동일 모델·sampler에서 추가 학습과 진단으로 병목을 확인한다. 기존 formal gate와 MD5 본실험 미구현 상태는 유지된다.

검증 결과: 전체 pytest **138 passed, 5 skipped**. [테스트 XML](local_experiment_archive/analyses/v31-p2a10k-tests-20260927.xml)에 보존했다. 추가 검사에서 PyTorch CPU·MLX Metal의 중간 probe 중단/재개, 진단 없는 학습 대비 최종 weights/optimizer 일치, 최종 checkpoint 판정, D1 loss 구성요소의 backend 간 수치 일치, 새 명세의 seed·자료 유지와 변조 거부를 확인했다. 원본 v3.1 명세 정합성 검사 및 새 명세 CLI dry-run도 통과했다. 이 검증에서는 원본 규모 10,000-update 실험을 실행하지 않았다.

## P2 실패 대응 개정 (2026-09-27)

[실패 분석](/Users/choisoonwook/Experiments_local/DHI_AI_gen/V3_1_P2A10K_ANALYSIS_KO.md)에서 확인한 checkpoint 선택 문제와 잔여 조건·형식 오류에 대응한다. 새 명세는 [poc-v3.1-p2fix-protocol.json](/Users/choisoonwook/Experiments_local/DHI_AI_gen/examples/poc-v3.1-p2fix-protocol.json), protocol ID는 `dhi-v3.1-p2fix-20260927`이다. 원본 및 P2A-10k 명세는 그대로 보존하며 새 명세도 정확한 SHA-256으로 검증한다.

- **P2B checkpoint:** epoch10/30 probe는 해당 epoch를 마친 최신 가중치를 사용하고, gate는 사전 고정한 최종 epoch100 가중치를 사용한다. Validation loss 최소 checkpoint도 진단용으로 보관한다. `best_epoch`와 실제 `selected_epoch`를 구분해 기록하고 최종 probe/checkpoint hash가 다르면 중단한다. 중간 성공률로 checkpoint를 고르거나 조기 종료하지 않는다. P2A의 final10,000-update 규칙은 유지한다.
- **D1 조건 전달:** 13차원 내부 조건에서 모든 위치의 logits로 가는 학습 가능한 선형 잔차를 추가한다. 기존 denoiser와 길이 head를 유지하며, 조건을 입력 첫 층에서만 받던 병목을 줄인다. 숨은 원문·길이·검증 결과를 추가하지 않는다. 정답 prefix를 코드로 삽입하지 않는다.
- **Gaussian 조건 전달:** 기존 time/condition embedding에서 모든 출력 픽셀로 가는 학습 가능한 선형 잔차를 추가한다. G2의 좌표 입력, x0 예측, clipping, DDIM sampling은 유지한다.
- **Gaussian 영역별 학습:** 전체 이미지 평균 MSE를 mask·prefix3 glyph·suffix glyph·padding glyph·BGV length header의 영역별 평균 MSE 합으로 바꾼다. 각 영역 가중치는 1이고 빈 영역은 0이다. 배경·무작위 suffix의 픽셀 수가 prefix/header 오류를 희석하지 않도록 한다. Training과 validation이 같은 loss 함수를 사용하며 PyTorch·MLX 모두 구현했다. Strict decoder와 출력 threshold는 바꾸지 않았다.
- **등록 후보:** Gaussian은 G2, Discrete는 D1만 사용해 5개 pipeline마다 한 후보를 검증한다. G0/G1/D0 구현과 이전 protocol 동작은 남겨 둔다. D0의 EOS/PAD 오류는 새 실행에서 기존 D1의 학습된 길이 기반 생성으로 대응한다. D0 자체의 unconstrained sampler가 교정됐다는 뜻은 아니다.

P2A/B의 정상·반전 joint 기준, wrong-original 상한, 학습량, seed namespace, source data, temperature1, strict codec은 유지한다. 구조가 달라졌으므로 동일 seed가 이전과 동일한 초기 가중치를 뜻하지는 않는다. 신규 파라미터 수는 P0에서 검증하고 자원 비용은 새로 측정한다.

이 개정은 **합성 과제 개발용**이다. Prefix3 가중 loss는 합성 과제의 알려진 구조를 사용하므로 MD5 objective에 그대로 적용하거나 formal 적격성 증거로 사용할 수 없다. 기존 P3·E0·MD5 본실험 차단은 유지된다. 모델 변경의 성능 개선 및 품질 기준 통과는 새 원본 규모 P2 실행으로 확인해야 한다.

새 workdir 실행:

```bash
DHI_RUN="local_experiment_archive/runs/v31-p2fix-$(date +%Y%m%d-%H%M%S)"
for stage in P0 P1 P2; do
  .venv/bin/python -m diffusion_hash_inv.study_cli pilot \
    --protocol examples/poc-v3.1-p2fix-protocol.json \
    --workdir "$DHI_RUN" --stage "$stage" \
    --backend mlx --device gpu --development || break
done
```

기존 run은 코드 hash가 달라져 재개할 수 없다. 쓰기 없는 계획 확인은 새 명세에 `--stage P2 --dry-run`을 사용한다.

검증: 전체 pytest **143 passed, 5 skipped** ([XML](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2fix-tests-20260927.xml)). 구버전 v3의 봉인된 `models.py`·`discrete.py`는 변경하지 않았고, 새 Torch 기능은 v3.1 전용 모듈에 구현했다. 추가 검사에서 영역별 오차의 면적 독립성·빈 영역, 새 조건 출력 층의 CPU/MLX CPU/Metal forward·loss·gradient·Adam 일치, BEST가 epoch10에 남아 있어도 최종 epoch를 평가하는 중단/재개, probe/checkpoint 일치, 변조된 명세 거부를 확인했다. 원본 v3.1 명세 정합성 및 새 명세 dry-run도 통과했다.

등록된 새 명세 그대로 MLX Metal의 개발 **P0/P1 PASS**를 확인했다. Workdir은 `local_experiment_archive/runs/v31-p2fix-validation-20260927-145713`이며 [실행 보고서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/v31-p2fix-validation-20260927-145713/report.md)에 보존했다. P0는 1,214개 codec 왕복과 새 5개 모델 조합을, P1은 10개 learned run·2개 Random stream 및 학습/생성 복구를 검사했다. 봉인과 최종 source hash도 대조했다. 이 작업에서는 원본 규모 P2를 실행하지 않았으므로 생성 품질 개선이나 P2 통과를 주장하지 않는다.

## P2 구조·prefix 학습 개정 (2026-09-28)

[최신 실행 분석](/Users/choisoonwook/Experiments_local/DHI_AI_gen/V3_1_P2FIX_ANALYSIS_KO.md)에 따른 [권장 수정안](/Users/choisoonwook/Experiments_local/DHI_AI_gen/V3_1_P2STRUCT_MODIFICATION_KO.md)을 구현했다. 새 [p2struct 명세](/Users/choisoonwook/Experiments_local/DHI_AI_gen/examples/poc-v3.1-p2struct-protocol.json)는 Gaussian G3와 Discrete D1만 등록하며, 기존 데이터·seed namespace·학습량·최종 checkpoint 규칙·품질 기준을 유지한다.

- D1은 masked prefix3와 suffix CE를 각각 평균하여 length CE와 더한다. P2B ledger에는 prefix가 확정된 step·선택/정답 확률·argmax를 기록한다. 진단은 후보나 RNG 소비를 바꾸지 않는다.
- G3는 공개 조건으로 길이를 한 번 생성하고, 그 길이의 header·연속 mask·padding을 고정한 채 payload glyph만 확산한다. 길이 head와 prefix/suffix glyph loss를 학습하며 평가 원문의 길이나 정답 prefix를 주입하지 않는다. Strict decoder는 그대로이며 전체 출력의 NaN/Inf를 검사한다.
- PyTorch/MLX 학습·생성·진단·복구 경로와 NFE 계수를 함께 반영했다. G3는 후보당 length head 1회 + denoiser 100회로 NFE 101이다. 기존 G0–G2/D0 동작과 이전 명세·실험 결과는 보존한다.

검증은 전체 **154 passed, 5 skipped** ([XML](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2struct-tests-20260927.xml)), 최종 NFE 명세 보존 검사, 연구 명세 정합성 및 P2 dry-run까지 완료했다. Skip은 CUDA 미지원 5개다. 새 길이·loss·진단 무간섭 검사와 Torch/MLX 수치 비교, 축소 P2 및 중단/복구 검사를 포함한다.

최종 등록 명세로 MLX Metal 개발 **P0/P1 PASS**를 확인했다 ([보고서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/v31-p2struct-validation-20260927/report.md)). P1의 10개 learned run·2개 Random stream과 다섯 모델의 학습/생성 복구가 통과했다. Source 49개와 봉인 파일 408개도 대조했다 ([무결성 결과](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2struct-validation-audit-20260928.json)). 원본 규모 P2는 아직 실행하지 않았다. P1 joint는 모두 0이므로 이 결과를 생성 품질 개선으로 해석하지 않는다. 합성 과제 개발용 개정이며 P3·E0·MD5 본실험 차단은 유지된다.
