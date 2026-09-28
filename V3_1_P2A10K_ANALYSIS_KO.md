최신 실행 결과 및 실패 원인 분석 — 2026-09-27

대상: `v31-p2a10k-20260927-094002`, protocol `dhi-v3.1-p2a10k-20260927`, MLX 0.32.2 / Metal GPU / float32 / development 실행.

**실행은 정상 완료됐지만 최종 판정은 `BLOCKED_DEVELOPMENT`다. P2A는 13개 후보 중 5개가 통과했고, 이어 수행한 P2B는 5개 모두 탈락했다. 가장 중요한 추가 발견은 Discrete의 checkpoint 선택 기준과 실제 성공률의 불일치다. 전체 validation loss는 10~20 epoch 가중치를 선택했지만, 동일 난수로 비교한 100 epoch 가중치의 생성 성공률은 훨씬 높았다. 다만 최종 가중치도 등록 기준에는 미달한다.**

이 문서는 2,000-update 실행을 다룬 [이전 P2A 분석](/Users/choisoonwook/Experiments_local/DHI_AI_gen/V3_1_P2A_ANALYSIS_KO.md)의 후속이다. 이전의 “모든 P2A 후보 실패”와 “D1 길이 분포가 주된 단서”를 이번 P2B 실패에 그대로 적용하면 안 된다.

| 단계 | 실행 상태 | 결과 | Active time |
|---|---|---|---:|
| P0 | COMPLETE | PASS | 3.07초 |
| P1 | COMPLETE | PASS | 138.05초 |
| P2 | COMPLETE | BLOCKED_DEVELOPMENT, exit 2 | 3,430.88초 |
| P3·E0·MD5 본실험 | 미실행 | 적격성 미확보 | — |

Active time 합계는 3,572초, 약 59분 32초다. 13개 P2A는 각각 10,000 updates, 5개 P2B는 각각 100 epochs / 15,700 updates를 마쳤다. 총 208,500개 update 로그가 연속이며 loss가 모두 유한하다. 개별 `FAILED_RUNTIME`이나 `failure.json`, 시간 상한 종료 기록은 없다. Exit 2는 품질 미달을 알리는 의도된 종료다. P0/P1 PASS는 실행·복구 검사를 통과했다는 뜻이며 생성 품질이나 formal PoC 적격성을 뜻하지 않는다.

**결과 검증.** 봉인된 파일 2,344개와 현재 source 파일 49개의 SHA-256, frozen protocol, synthetic data hash가 일치했다. SQLite 82개를 읽기 전용으로 열어 integrity check와 후보 13,312개의 평가 키·조건·checkpoint/config identity·성공 판정을 대조했다. P1 4,480개, P2A 최종 1,664개, P2A 중간 진단 3,328개, P2B probe 3,840개다. P2B probe 세 번이 같은 checkpoint를 쓰는 경우도 있으므로 독립 후보 3,840개로 통계적 표본 수를 늘려 해석하지 않는다.

Train/validation 조건 집합은 분리돼 있으며, 저장된 21,024개 학습·검증 메시지의 domain과 prefix 라벨도 확인했다. 앞선 실행과 data/case selection이 같고, 이번 2,000-update probe의 13개 후보별 정상·반전 결과, invalid 사유, 중복 수가 이전 실행의 최종 집계와 일치한다. 모든 생성 tensor의 재디코딩이나 전체 학습 재현까지 수행한 검증은 아니다.

**P2A: 추가 학습은 실제 효과가 있었다.** P2A는 source별 학습 메시지 16개를 암기할 수 있는지 검사한다. 정상·반전 각각 64개에서 joint 61개 이상, 반전의 원래 조건 오성공 1개 이하가 기준이다. Joint는 strict valid와 요청 prefix 정답을 동시에 만족한 후보 수다. 반전 조건도 같은 16개 학습 조건 안에 있다.

| Pipeline / profile | 2,000 joint 정상/반전 | 5,000 joint 정상/반전 | 10,000 joint 정상/반전 | 최종 valid 정상/반전 | P2A |
|---|---:|---:|---:|---:|---|
| P-G-BGV / G0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | FAIL |
| P-G-BGV / G1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | FAIL |
| P-G-BGV / G2 | 0 / 0 | 0 / 2 | 12 / 13 | 27 / 30 | FAIL |
| P-G-CGGE / G0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | FAIL |
| P-G-CGGE / G1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | FAIL |
| P-G-CGGE / G2 | 1 / 0 | 18 / 22 | 64 / 64 | 64 / 64 | PASS |
| P-DISC / D0 | 50 / 50 | 59 / 58 | 64 / 64 | 64 / 64 | PASS |
| P-DISC / D1 | 46 / 59 | 64 / 64 | 64 / 64 | 64 / 64 | PASS |
| R-G-BGV / G0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | FAIL |
| R-G-BGV / G1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | FAIL |
| R-G-BGV / G2 | 0 / 0 | 5 / 4 | 15 / 9 | 44 / 47 | FAIL |
| R-DISC / D0 | 57 / 50 | 64 / 64 | 64 / 64 | 64 / 64 | PASS |
| R-DISC / D1 | 58 / 54 | 63 / 61 | 64 / 64 | 64 / 64 | PASS |

각 수의 분모는 64다. P는 printable, R은 random bytes다. 중간 probe는 진단용이고 gate는 최종 10,000 updates만 사용한다. 모든 P2A의 반전 원래 조건 오성공은 0이다. 통과한 5개 조합은 128개 후보 모두 해당 조건의 학습 메시지 전체를 재현했다. 이는 작은 학습집합 암기 성공이며 미학습 조건 일반화의 증거는 아니다.

D1의 목표 학습 길이 평균 확률은 P 82.54% → 97.32% → 99.77%, R 85.67% → 97.92% → 99.83%로 증가했다. 최종 길이 CE는 P 0.00228, R 0.00170이다. 기존 sampler를 유지한 채 P2A 길이 불확실성과 생성 오류가 크게 줄었다. 이번 범위에서는 “추가 학습이 무의미하다”거나 “Discrete가 조건을 학습하지 못한다”는 결론이 맞지 않는다.

**P2B: 공식 평가에서는 형식과 조건 대응이 모두 충분하지 않았다.** P2B는 P2A 가중치를 이어 학습하지 않는다. Fresh seed100으로 source별 10,000개 메시지를 학습하고, 학습에서 제외한 validation 조건을 평가한다. 정상·반전 각각 128개에서 joint와 valid 모두 122개 이상, 반전의 원래 조건 오성공 3개 이하가 기준이다.

| Pipeline / profile | 선택 epoch | 정상 valid /128 | 정상 joint /128 | 반전 valid /128 | 반전 joint /128 |
|---|---:|---:|---:|---:|---:|
| P-G-CGGE / G2 | 100 | 0 | 0 | 0 | 0 |
| P-DISC / D0 | 20 | 34 | 12 (9.38%) | 38 | 15 (11.72%) |
| P-DISC / D1 | 10 | 128 | 35 (27.34%) | 128 | 39 (30.47%) |
| R-DISC / D0 | 20 | 39 | 17 (13.28%) | 44 | 15 (11.72%) |
| R-DISC / D1 | 10 | 128 | 17 (13.28%) | 128 | 22 (17.19%) |

반전의 원래 조건 오성공은 모두 0이다. 하지만 요청한 반전 조건 자체를 맞히지 못한 경우가 많으므로 이 값만으로 충분한 조건 반응을 주장할 수 없다. 나머지 8개 profile은 P2A 실패로 P2B를 실행하지 않았다. 어떤 pipeline도 최종 profile을 선택하지 못했고, 자원 검사는 `NOT_MEASURED`, final seal은 false다.

**확인된 주요 원인: 전체 validation loss가 prefix 성공률이 낮은 checkpoint를 선택한다.**

[선택 코드](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/study_pilot.py:600)는 전체 validation loss가 낮아졌을 때만 BEST를 갱신한다. [P2B probe](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/study_p2.py:217)는 매번 그 BEST를 평가한다. D0는 epoch20, D1은 epoch10 이후 BEST가 갱신되지 않았다. 따라서 epoch30·100의 같은 성공 수는 같은 가중치·조건·난수를 다시 평가한 결과다. 이를 “그 이후 모델 학습이 완전히 멈췄다”로 해석하면 안 된다.

영향을 확인하기 위해 별도 분석 디렉터리에서 **가중치만 BEST와 최종 epoch100으로 교체**했다. 원래 validation 조건 128개, normal/flipped, 후보별 payload/length RNG, temperature1 sampler, batch4, MLX Metal GPU를 유지했다. BEST 재실행 1,024개는 payload·valid·실패 사유·성공·원래 조건 일치 등 비교 필드가 원본과 모두 일치했다. 학습과 원본 실험 수정은 없었다.

| 모델 | 공식 BEST joint 정상/반전 | 최종 epoch100 joint 정상/반전 | 최종 valid 정상/반전 | 최종 가중치도 품질 미달인가? |
|---|---:|---:|---:|---|
| P-DISC D0 | 12 / 15 | 63 / 62 | 69 / 65 | 예 |
| P-DISC D1 | 35 / 39 | 114 / 110 | 128 / 128 | 예 |
| R-DISC D0 | 17 / 15 | 58 / 62 | 68 / 66 | 예 |
| R-DISC D1 | 17 / 22 | 114 / 107 | 128 / 128 | 예 |

각 수의 분모는 128이다. 최종 D1 정상 성공률은 두 source 모두 89.06%, 반전은 P 85.94%, R 83.59%다. BEST보다 크게 좋아졌지만 122/128 = 95.3125%에 못 미친다. 이 비교는 관측된 조건·난수에 대한 checkpoint 교체 효과를 직접 보여주며, 여러 seed에서의 일반적 효과나 새 revision의 통과를 입증하지는 않는다. 새 후보 2,048개는 사후 진단용이며 기존 공식 gate의 결과를 대체하지 않는다.

**왜 잘못된 방향으로 선택되는가: prefix 개선보다 무작위 suffix의 loss 증가가 더 크다.**

[합성 메시지 생성](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/study_pilot.py:167)에서 첫 3개 token만 조건으로 정해지고, 나머지 payload와 길이는 난수로 생성된다. 반면 [학습·선택 loss](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/mlx_models.py:169)는 가려진 전체 token CE의 평균이며 D1은 여기에 길이 CE를 더한다. 성공 판정은 유효 메시지의 첫 3개 token을 요구한다. 따라서 선택 loss와 gate가 중시하는 항목이 다르다.

기존 validation512 전체와 등록된 고정 noise/mask를 그대로 써서 loss를 분해했다. 아래 prefix/suffix는 같은 per-record masked-position 분모를 사용한 전체 loss 기여도이며, 각 영역만의 token 평균 CE가 아니다. 구성요소 합은 저장된 validation loss와 1e-5 이내로 일치했다.

| D1 모델·checkpoint | Prefix 기여 | Suffix 기여 | Length CE | 전체 validation loss |
|---|---:|---:|---:|---:|
| P BEST epoch10 | 0.07216 | 3.33697 | 3.34935 | 6.75848 |
| P 최종 epoch100 | 0.00778 | 3.75497 | 3.34501 | 7.10776 |
| R BEST epoch10 | 0.12728 | 4.22758 | 3.35016 | 7.70501 |
| R 최종 epoch100 | 0.01073 | 5.50169 | 3.35206 | 8.86448 |

P는 prefix 기여가 약 0.0644 감소했지만 suffix가 약 0.4180 증가했다. R도 prefix는 약 0.1166 감소했지만 suffix가 약 1.2741 증가했다. Length CE의 변화는 작다. 이 때문에 실제 prefix sampling은 개선됐는데 전체 loss는 악화돼 초기 가중치가 선택됐다. D0에서도 같은 방향을 확인했다. Train loss는 감소하고 validation suffix loss는 증가하므로 무작위 suffix의 학습집합 과적합과 부합하는 결과다.

여기서 P2B length CE가 약 3.35라는 이유로 길이 head 고장이라고 판단하면 안 된다. 생성 코드상 길이는 조건과 독립인 4~31의 균등분포이고 균등 예측의 CE는 ln(28) ≈ 3.332다. P2A에서는 조건마다 고정한 메시지 한 개의 길이를 암기했지만, P2B에서는 임의의 validation 메시지와 같은 길이를 맞힐 필요가 없다. D1은 이미 전체 256개 후보에서 형식을 만족한다. 현재 증거에서 길이 head 강화보다 checkpoint 선택과 prefix 학습·sampling이 우선이다.

추가 CPU 진단도 같은 해석을 지지했다. MLX F32 가중치를 기존 PyTorch 동형 모델로 읽어, 공식 all-MASK 진단의 prefix 확률과 argmax 정확도를 재현한 뒤 validation512를 검사했다. D1 BEST의 위치별 평균 정답 확률은 P 67.8/68.5/70.9%, R 53.0/51.1/53.6%였고 최종 가중치는 P 97.9/97.6/98.4%, R 97.0/96.5/96.3%였다. D1의 이 진단에는 실제 길이와 EOS/PAD 문맥을 주므로 실생성 성공률로 세지 않는다. 세 위치 확률의 평균을 곱해 joint 성공률로 대체하지도 않는다.

**남은 Discrete 실패: D0의 전역 문법, D1의 잔여 prefix sampling 오류.**

D0의 공식 BEST에서는 P 256개 중 184개, R 256개 중 173개가 형식 오류다. P는 EOS 개수 106·payload token 29·EOS 뒤 non-PAD 49개, R은 각각 94·32·47개다. 최종 epoch100으로 바꿔도 형식 오류가 두 source 각각 122/256개 남는다. EOS/PAD를 개별 위치의 일반 token으로 생성하는 경로에는 전역 문법 문제가 지속된다. D1의 길이 기반 생성은 이 문제를 구조적으로 피하며, 이번 BEST·최종 모두 형식 오류가 0이다.

D1 최종에서는 P 32/256개, R 35/256개가 prefix를 틀렸다. P는 30개가 세 위치 중 한 위치만 틀렸고 2개가 두 위치를 틀렸다. R은 34개가 한 위치, 1개가 두 위치를 틀렸다. 따라서 checkpoint만 최종으로 고르는 것으로는 충분하지 않다. 순차 reveal 중 어느 시점·문맥이 오류를 키우는지는 이번 진단에서 추적하지 않았다. 조건 입력 경로가 완전히 끊겼다는 가설은 높은 정답 확률과 생성 결과에 맞지 않는다.

**Gaussian 실패는 Discrete의 선택 문제와 구분해야 한다.** G0는 epsilon 예측, G1은 좌표를 추가한 epsilon 예측, G2는 좌표를 포함한 x0 예측이다.

| 계열 | 이번 실행에서 확인된 병목 |
|---|---|
| BGV/CGGE G0·G1 | 6개 조합 모두 P2A valid 0/128. 작은 집합조차 형식 생성 실패 |
| P-G-BGV G2 | valid 57/128, joint 25/128. Source domain 오류 61, 길이 오류 5, mask 오류 5 |
| R-G-BGV G2 | valid 91/128, joint 24/128. 길이 오류 13, mask 오류 24; valid 중에도 다수 prefix 오류 |
| P-G-CGGE G2 | P2A 128/128 성공, P2B 0/256 valid. P2B mask 오류 242, glyph 거리 오류 14 |

실패 사유는 decoder가 처음 발견한 오류다. 후보에 다른 오류가 없었다는 뜻은 아니다.

G0/G1의 P2A 저장 진단에서는 낮은 잡음 t=0의 x0 MSE가 약 0.000055~0.000082인 반면, t=999에서는 33.93~44.69다. t=999 noise MSE는 약 0.00137~0.00180으로 작아 보이지만, epsilon에서 x0를 계산할 때 작은 alpha-bar로 나누어 오차가 크게 증폭된다. 실제 [복원 코드](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/mlx_models.py:108)와 일치하는 현상이다. 고잡음에서 이미지 범위를 크게 벗어난 finite 복원과 NaN/Inf crash는 다르다. 좌표 추가만으로 이 병목이 해결되지는 않았다.

G2의 x0 objective는 일부 계열에서 개선을 보였고 CGGE는 P2A를 통과했다. 그러나 BGV에서는 작은 집합에서도 byte/domain·길이·mask·prefix 제약이 충분히 맞지 않았다. 작은 평균 픽셀 MSE가 전체 메시지의 strict decode를 보장하지 않는다는 결과다.

P-G-CGGE G2의 P2B는 epoch100이 BEST이므로 Discrete처럼 초기 checkpoint 선택으로 설명할 수 없다. Validation loss는 epoch10 0.25401에서 epoch100 0.21829로 줄었지만 세 probe 모두 valid/joint 0이다. P2B t=999 진단의 mask MSE는 0.6048, active glyph MSE는 0.7357이다. P2A의 해당 값은 0.000719, 0.0753이었다. 서로 다른 학습집합·seed·평가 split의 teacher-forced 진단이라는 한계가 있지만, 작은 집합 암기 성공이 큰 집합·미학습 조건의 형식 생성으로 이어지지 않았음은 직접 확인된다. 손실, 모델 용량, sampler, 표현 중 어느 변경이 이를 해결하는지는 별도 대조가 필요하다.

**다음 revision의 우선순위.**

1. **Checkpoint 선택부터 정렬한다.** 고정된 개발 validation에서 형식 유효성과 조건 성공을 반영하는 선택 규칙을 사전 등록하고, 전체 denoising loss는 보조 지표로 기록한다. 또는 최종 checkpoint 고정 규칙을 별도 revision으로 비교한다. 이번 결과를 보고 기존 BEST pointer나 통과 판정을 바꾸지는 않는다.
2. **D1을 우선 진단한다.** 문법 문제가 없고 최종 가중치의 정상 성공률이 89.06%까지 올라왔다. Prefix/suffix/length loss를 분리해 유지하고, 실제 reverse trajectory의 prefix 정답 확률·오류 reveal 시점을 기록한다. 그 증거로 prefix 가중치나 조건 주입 개선을 비교한다. 이번 데이터만으로 특정 변경의 통과를 보장할 수는 없다.
3. **Gaussian은 형식 병목을 따로 해결한다.** G0/G1의 고잡음 복원, BGV G2의 byte/domain·header·prefix 오류, CGGE G2의 P2B mask/glyph 실패를 구분한다. 전체 MSE 감소만으로 성공을 예측하지 말고 작은 대조에서 strict decoder 결과를 함께 측정한다.
4. **P3·MD5 확대는 보류한다.** 최종 가중치 사후 평가도 기준에 못 미쳤고 모든 pipeline 선택이 실패했다. 새 규칙은 새 protocol/workdir에서 검증해야 한다. 모든 공식 P1/P2 후보의 MD5 호출 수는 0이며, 이번 결과는 합성 prefix 과제의 개발 검사다.

분석 중 실험 코드·명세·checkpoint·원본 gate는 변경하지 않았다. 새 문서와 별도 분석 스크립트/JSON만 추가했다. GPU 비교는 샌드박스에서 Metal 접근이 불가능해 승인된 확장 권한으로 수행했다. 전체 학습을 재실행하거나 새 hyperparameter를 탐색하지 않았다.

원본: [실행 보고서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/v31-p2a10k-20260927-094002/report.md), [후보 선택 기록](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/v31-p2a10k-20260927-094002/profile_selection.json), [등록 명세](/Users/choisoonwook/Experiments_local/DHI_AI_gen/examples/poc-v3.1-p2a10k-protocol.json).

재현 자료: [원본 검증 스크립트](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2a10k-20260927-094002/analyze.py), [검증·집계 JSON](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2a10k-20260927-094002/audit.json), [CPU 가중치 진단](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2a10k-20260927-094002/checkpoint_diagnostic.py), [CPU 진단 JSON](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2a10k-20260927-094002/checkpoint_diagnostic.json), [동일 난수 BEST/최종 비교](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2a10k-20260927-094002/paired_generation.py), [후보·loss 분해 JSON](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2a10k-20260927-094002/paired_generation.json).

프로젝트 루트에서 `.venv/bin/python local_experiment_archive/analyses/v31-p2a10k-20260927-094002/analyze.py`로 원본 검증을 재실행할 수 있다. 같은 디렉터리의 `checkpoint_diagnostic.py`는 CPU, `paired_generation.py`는 Metal 접근이 필요하다. 세 스크립트 모두 assert 검증을 포함하며 출력은 해당 분석 디렉터리에만 저장한다.
