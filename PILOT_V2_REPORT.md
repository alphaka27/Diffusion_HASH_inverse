# v2 최소 실행 경로 및 합성 pilot 결과

실행일: 2026-09-24. **구현·실행 완료 / 축소 pilot 기준 FAIL / 정규 G1-B 미인증 / primary hash 평가 미실행.**

## 구현한 범위

[pilot_v2.py](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/pilot_v2.py)는 기존 모델·diffusion·codec을 재사용한다. Legacy runner와 별도로 다음을 연결했다.

- 12-bit public integer만 condition으로 변환하는 경계. Generator에는 원문·길이·digest suffix·record ID를 전달하지 않는다.
- 다섯 pipeline의 v2 model dimensions와 sampler. Main/Shuffled는 초기화·학습 순서를 맞추고, donor permutation은 학습 epoch에만 적용한다. Validation/inference에는 실제 condition을 사용한다.
- Epoch별 전체 train permutation, 마지막 작은 batch 유지, 고정된 validation corruption 4회, validation loss 최소/이른 tie checkpoint 선택.
- Model·Adam·batch cursor·noise RNG·학습 이력·best checkpoint를 저장하는 중간 학습 재개. 데이터 내용과 설정을 checkpoint identity에 포함한다.
- 시도별 독립 seed와 SQLite transaction으로 후보·invalid·중복·선택한 raw tensor를 보존한다. 완료 rows는 건너뛰며 새 결과로 바꾸지 않는다. @1/@10/@100은 같은 stream의 prefix다.
- Code/config/data/selected checkpoint/완료 artifact hashes를 연결한다. 완료된 run은 무결성을 확인하고 학습·생성을 반복하지 않는다.

Primary hash dataset 생성·노출 감사, MD5 본 평가 orchestrator, 5-pipeline max-p/Holm 전체 분석은 이번 범위에 포함하지 않았다. 합성 verifier는 payload domain과 첫 세 symbol 제약을 검사하며 모델로 결과를 되먹임하지 않는다.

## 사전 고정한 pilot

[실행 설정](/Users/choisoonwook/Experiments_local/DHI_AI_gen/examples/pilot-v2-synthetic.json)은 개발용이며 v2 고정 프로토콜을 수정하지 않는다.

| 항목 | 실행값 |
|---|---|
| Pipeline / seed / method | P-G-BGV / 0 / Main |
| 장치 | Apple M3 Max, MPS, float32 |
| Architecture | ImageUNet width 32, input channels 2, condition 12; 114,978 parameters |
| Gaussian | epsilon, T=1,000, beta .0001→.02, sampling 100 steps |
| 합성 과제 | 12-bit 조건의 세 nibble을 첫 세 payload symbol에 대응 |
| 데이터 | Train 2,048 unique messages, validation 128 conditions, test 128 conditions |
| 분할 | 전체 조건은 complement pair 단위로 train/validation/test 분리; 축소 validation/test도 complement closure 유지 |
| 학습 | Batch 64, Adam .001, 20 epochs = 640 updates |
| Validation | Epoch 10/20; 대표 원문당 고정 corruption 4회 |
| 평가 | 정상 K=1 128개 + 반전 K=1 128개 + 사전 선택 첫 4개 정상 표적의 stream을 100개까지 확장 |
| 총 후보 | 128+128+4×99 = **652 attempts** |
| Raw 보존 | 결과와 무관한 첫 16개 정상 표적의 첫 candidate |

Train 수·epochs·validation/test 수를 정규 v2보다 줄였다. 첫 4개 표적은 결과를 보기 전에 정했으며 @100은 이 네 표적의 engineering diagnostic이다. 전체 128개에 대한 @100 결과로 해석하지 않는다. Shuffled 모델의 실제 학습 및 다른 pipeline/seed는 이번 pilot에서 실행하지 않았다.

## 최종 관측 결과

| 지표 | 결과 |
|---|---:|
| Epoch 1 → 20 평균 training loss | 0.556771 → 0.050671 |
| Epoch 10 → 20 validation loss | 0.092504 → 0.058065 |
| 선택된 checkpoint | Epoch 20 |
| 정상 조건 valid AND constraint success | **0/128**, Wilson 95% [0, 0.02914] |
| 반전 조건 자체에 대한 동일 성공 | **0/128**, Wilson 95% [0, 0.02914] |
| 반전 생성물이 원래 조건을 만족 | 0/128; 유효 후보가 없어 조건 활용의 증거가 아님 |
| 전체 유효 decode | **0/652** |
| 첫 4개 표적의 Success@1/@10/@100 | 모두 **0/4** |
| Decoder length_out_of_range | 574건 |
| Decoder mask_inconsistent | 69건 |
| Decoder length_slot_invalid | 9건 |

Loss 감소는 관측됐지만 유효한 메시지 생성은 확인하지 못했다. 실패 이유는 decoder가 최초로 거부한 사유다. 뒤 단계의 payload/조건 제약이 올바르다는 뜻은 아니다. 고정 보존한 raw 16개는 모두 finite였으며, 후속 읽기 전용 점검에서 header가 39·255·192 등 허용 범위 밖으로 복원되는 사례를 확인했다. 후보를 수선하거나 재채점하지 않았다.

**판정:** 축소 학습 budget의 P-G-BGV·seed 0은 pilot 성공 기준을 충족하지 못했다. 이를 정규 10,000개·100 epochs control의 실패나 다른 pipeline/seed의 실패로 확대하지 않는다. 현재 관측상 다음 개발 진단 대상은 길이 header와 validity mask를 포함한 구조 생성이다. Mask/header 학습과 reverse sampling을 구분해 진단해야 하며, loss만 보고 본실험으로 진행할 근거는 없다.

측정된 학습 구간은 약 **14.3초**, 후보 generation+decode 합은 약 **94.8초**였다. 데이터 구성과 전체 IO를 포함한 end-to-end 시간이나 정규 본실험 예산은 아니다. 최종 artifact는 약 3.36 MB였다.

## 재현성 수정과 검증

첫 실행과 추가 복구 검사에서 Discrete/MPS 가중치에 최대 약 1.19×10^-7 차이를 관측했다. 이 차이를 허용 오차로 덮지 않고 `torch.use_deterministic_algorithms(True)`를 적용하고 실행 기록에 고정했다. PyTorch는 이 설정에서 알려진 비결정적 연산에 결정적 구현을 사용하거나 지원되지 않으면 오류를 낸다. 설정만으로 모든 재현성이 보장되는 것은 아니므로 실제 중단·재개 검사도 수행했다. [PyTorch 공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)

첫 pilot은 그대로 보존하고, 최종 코드로 동일한 데이터·학습량·평가 기준을 새 run에서 재실행했다. BGV의 선택된 가중치 tensor는 두 실행에서 동일했고 결과도 같았다. 효과를 높이기 위한 hyperparameter 탐색이나 불리한 결과 삭제는 하지 않았다.

- 최종 전체 tests: **120 passed, 5 skipped in 6.06s**. CUDA 장비가 없어 5개 제외.
- Gaussian/Discrete 각각 CPU/MPS에서 batch 2 저장 후 강제 중단·재개: 중단 없는 실행과 history 및 최종 선택 weights의 정확한 일치 확인.
- 학습 데이터 변경 시 resume 거부, inference-time shuffle 부재, 고정 validation noise, 마지막 작은 batch 사용 확인.
- 후보 저장 중 중단: 이미 기록된 rows 불변, 재개 시 누락·중복 없음, invalid/중복의 예산 소비 및 100-prefix 계수 확인.
- 실제 완료 run의 652 rows·raw 16개·artifact hashes 확인. 재호출 시 학습과 생성을 호출하지 않음을 검사.

이 검증은 검사한 환경·fixture의 증거다. 다른 PyTorch/장치/batch 또는 모든 GPU 커널의 bitwise 재현성을 보증하지 않는다.

## 재실행과 결과 파일

프로젝트 루트에서 다음 명령을 사용한다. 아래 완료된 run은 검증 후 재사용한다. 설정이나 코드를 변경하면 새 출력 디렉터리가 필요하다.

```sh
.venv/bin/python -m diffusion_hash_inv.pilot_v2 \
  --protocol examples/poc-v2-protocol.json \
  --config examples/pilot-v2-synthetic.json \
  --output local_experiment_archive/runs/2026-09-24-v2-synthetic-pg-bgv-seed0-deterministic
```

- [최종 report.json](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/2026-09-24-v2-synthetic-pg-bgv-seed0-deterministic/report.json)
- [고정한 run 설정](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/2026-09-24-v2-synthetic-pg-bgv-seed0-deterministic/run.json)
- [후보 ledger](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/2026-09-24-v2-synthetic-pg-bgv-seed0-deterministic/candidates.jsonl)
- [최종 회귀 검사](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/2026-09-24-v2-pilot-validation/tests-deterministic.xml)
- [완료 실행 재호출·무결성 검사](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/2026-09-24-v2-pilot-validation/completed-replay.json)
- [보존한 첫 pilot](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/2026-09-24-v2-synthetic-pg-bgv-seed0/report.json)

정규 G1-B 15개 실행, 본실험의 전체 접근 감사·데이터·분석·자원 봉인은 남아 있다. 현재 결과로 primary hash 본실험을 시작하지 않는다.
