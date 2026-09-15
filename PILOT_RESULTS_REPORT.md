# MD5/BGV Pilot 결과 보고서

## 요약

이번 Pilot은 **유효한 hash-inversion 성능 판정을 제공하지 못했다**. 데이터 분할과 결정적 BGV codec은 정상으로 확인됐지만, 학습된 diffusion 모델은 hash 조건과 reversible 조건 모두에서 유효한 BGV record를 한 건도 생성하지 못했다. 따라서 hash 조건 run의 0% 성공률은 hash signal 부재가 아니라 현재 학습·sampling 설정의 생성 실패를 반영한다.

## 목적과 설정

Printable ASCII 메시지(길이 4--31 byte)에서 MD5의 앞 8 bit를 조건으로 BGV image diffusion 모델이 held-out target의 preimage를 생성하는지 점검했다.

| 항목 | 값 |
| --- | --- |
| Dataset | 12,000 messages; data/split/model seed 0 |
| Split | train/validation/test = 9,580 / 1,234 / 1,186 records |
| 평가 단위 | digest-group representative |
| Unique digest target | train/validation/test = 204 / 27 / 25 |
| Model | BGV, 138,402 parameters, 1,000 training steps |
| Diffusion / sampling | 100 / 100 steps |
| Candidate budget | K=1; target 25개당 candidate 25개 |

`q=8`에서는 동일 digest를 가진 message를 같은 split에 둬야 하므로 held-out record 1,186개가 25개 unique target으로 집계됐다. 이는 설계상 정상이나, 결과의 검정력은 낮다.

## 실행 결과

| Run | Device / condition | PreimageSuccess@1 | Valid decode | 주요 invalid 사유 | 학습 시간 |
| --- | --- | ---: | ---: | --- | ---: |
| Diffusion | CPU / hash | 0/25 | 0/25 | length out of range 21, mask inconsistent 4 | 703.9 s |
| Diffusion | MPS / hash | 0/25 | 0/25 | length out of range 18, mask inconsistent 7 | 168.1 s |
| Source-prior random baseline | MPS / hash | 0/25 | 25/25 | 없음 | - |
| Diffusion positive control | MPS / reversible record | exact recovery 0/25 | 0/25 | length out of range 19, mask inconsistent 6 | 144.0 s |

두 hash-model run은 동일 seed의 device 비교이며, 서로 독립적인 model-seed 재현 실험은 아니다. random baseline은 25개의 유효 candidate를 모두 rehash했지만 target prefix 일치는 관측되지 않았다.

성공이 0회인 25 target의 95% binomial CI 상한은 13.3%이며, rule-of-three 상한은 약 12.0%다. 따라서 0회 관측은 작은 차이의 부재를 입증하지 않는다.

## Gate 판정

| Gate | 상태 | 근거 |
| --- | --- | --- |
| G0: split independence | PASS | message와 digest-group의 모든 split pairwise overlap이 0 |
| G1: representation / pipeline | **FAIL** | deterministic BGV round-trip은 통과했지만, actual-model reversible positive control exact recovery가 0%로 요구치 99%에 미달 |
| G2: candidate-budget fairness | PASS | 모든 run에서 actual candidates = 25 = target count x K |
| G3: paired superiority | 미평가 | strict G1 failure 및 3-seed matched matrix 부재 |
| G4: seed reproducibility | 미평가 | model seed 0만 실행 |

연구 계획의 정의에 따르면 G1 실패가 있으므로 이번 결과는 **L0 invalid**이며, 경쟁 baseline보다 낫거나 못하다는 과학적 결론을 낼 수 없다.

## 해석

BGV encoder/decoder의 결정적 round-trip은 통과했고, random baseline은 유효한 message를 생성했다. 반면 diffusion output은 모두 BGV decoder의 length 또는 mask validation을 통과하지 못했다. Reversible record를 직접 제공한 positive control에서도 같은 실패가 재현됐으므로, 병목은 hash의 난이도가 아니라 현재 diffusion 모델의 학습·sampling 경로에 있다.

CPU와 MPS 모두 같은 정성적 실패를 보였으므로 이 현상을 MPS backend만의 문제로 해석할 근거도 없다. 다만 두 run은 같은 model seed이며, device 비교는 seed-level 재현성을 대체하지 않는다.

## 다음 단계

1. Test target을 다시 사용하지 말고 validation split에서 reversible positive control exact recovery가 99% 이상이 되는 최소 설정을 찾는다.
2. 그 설정, decoder threshold, sampling steps, K, baseline family를 test 전에 freeze한다.
3. 그 뒤에만 hash condition, random baseline, positive/negative control을 seed 0--2의 matched matrix로 재실행하고 G0--G4를 계산한다.

## 원본 산출물

- `output/md5-bgv-q8-s0/` — CPU hash-model run
- `output/md5-bgv-q8-s0-mps/` — MPS hash-model run
- `output/md5-bgv-q8-s0-random/` — source-prior random baseline
- `output/md5-bgv-q8-s0-reversible-mps/` — MPS reversible positive control

