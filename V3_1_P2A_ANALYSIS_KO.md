# 최신 v3.1 Pilot 분석: P2A 완료 및 개발 차단

분석 대상: `local_experiment_archive/runs/v31-mlx-20260927`. 분석일: 2026-09-27 KST.

**판정: 실행은 정상 완료됐지만, 13개 profile/pipeline 후보 모두 P2A 품질 기준에 미달했다. 현재 상태는 `BLOCKED_DEVELOPMENT`이며 P2B·P3·MD5 본실험으로 진행할 근거는 없다. 가장 구체적인 개선 단서는 D1의 길이 분포와 길이에 따른 다른 학습 메시지 생성이다.**

이 문서는 이전 실행 `v31-mlx-20260926-232405`를 다룬 [기존 분석](V3_1_PILOT_ANALYSIS_KO.md)의 후속이다. 기존 분석의 ‘P2 실행기 미구현’과 ‘다음 단계 P2A’는 당시 상태이며, 최신 실행에서는 P2A 구현·실행이 완료됐다.

| 단계 | 실행 상태 | 판정 | Active time |
|---|---|---|---:|
| P0 | COMPLETE | PASS | 3.69초 |
| P1 | COMPLETE | PASS | 136.27초 |
| P2 | COMPLETE | BLOCKED_DEVELOPMENT | 432.23초 |
| P2B | 13개 모두 NOT_RUN_P2A_FAILED | 미측정 | — |
| P3·E0·본실험 | 미실행 | 적격성 미확보 | — |

P0/P1/P2 active time 합계는 572.19초, 약 9분 32초다. P2 exit code 2는 품질 미달에 따른 종료이며, 보존된 13개 실행에서 runtime failure·시간 초과·중도 학습 종료는 없다. MLX 0.32.2 / Metal GPU / float32 / `--development` 실행이다. P0/P1 통과를 formal `TECHNICAL_READY`, `POC_QUALIFIED`, `MAIN_READY`로 해석하지 않는다.

**결과 신뢰성.** 단계 및 개별 실행의 봉인 파일 1,380개를 SHA-256으로 대조했고 불일치는 없었다. 현재 source hash, frozen protocol, synthetic data hash도 일치했다. 원본 평가 SQLite 41개를 읽기 전용으로 열어 integrity check 및 6,144개 후보의 평가 키·목표 조건·checkpoint/config identity·성공·원래 조건 일치 여부를 검증하고 집계와 대조했다. P2A 사례 선택도 원본 training data에서 재계산해 일치했다. 후보 6,144개는 P1 4,480개와 P2A 1,664개다. 이는 저장된 payload/ledger와 봉인 검증이며, 모든 생성 tensor를 다시 decode하거나 GPU 학습·복구를 재실행한 것은 아니다.

**P2A는 16개 학습 사례의 작은 과적합 검사다.** Source별 8 complement pairs, seed99, batch16, 2,000 updates, 최종 checkpoint로 평가했다. 정상·반전 각각 16조건 × 4회 = 64개이며, 둘 다 joint 61/64 이상(95.3125%), 반전의 원래 조건 오성공 1/64 이하가 필요하다. Joint는 strict valid이면서 요청한 prefix를 맞힌 후보다. 반전 대상도 같은 16개 학습 조건 집합 안에 있으므로 미학습 조건 일반화 검사가 아니다.

| Pipeline / profile | 정상 valid / 64 | 정상 joint / 64 | 반전 valid / 64 | 반전 joint / 64 |
|---|---:|---:|---:|---:|
| P-G-BGV / G0 | 0 | 0 | 0 | 0 |
| P-G-BGV / G1 | 0 | 0 | 0 | 0 |
| P-G-BGV / G2 | 0 | 0 | 0 | 0 |
| P-G-CGGE / G0 | 0 | 0 | 0 | 0 |
| P-G-CGGE / G1 | 0 | 0 | 0 | 0 |
| P-G-CGGE / G2 | 1 | 1 | 0 | 0 |
| P-DISC / D0 | 55 | 50 (78.13%) | 54 | 50 (78.13%) |
| P-DISC / D1 | 64 | 46 (71.88%) | 64 | 59 (92.19%) |
| R-G-BGV / G0 | 0 | 0 | 0 | 0 |
| R-G-BGV / G1 | 0 | 0 | 0 | 0 |
| R-G-BGV / G2 | 6 | 0 | 12 | 0 |
| R-DISC / D0 | 61 | 57 (89.06%) | 55 | 50 (78.13%) |
| R-DISC / D1 | 64 | 58 (90.63%) | 64 | 54 (84.38%) |

P는 printable, R은 random bytes다. 반전의 원래 조건 오성공은 모든 후보에서 0이다. 그러나 valid나 joint 자체가 0인 Gaussian 후보에서 이 0은 조건 반응의 증거가 아니다. 정상 variant의 `wrong_original`은 정상 성공과 같은 사건이므로 실패율로 해석하지 않는다.

**Gaussian의 병목은 여전히 형식 생성이다.** 9개 실행 1,152개 후보 중 strict valid은 19개(1.65%), joint는 1개다. 모델을 합친 숫자는 기술 진단용 집계이며 통계적 우위 검정이 아니다.

| 계열 | 주요 실패 |
|---|---|
| BGV G0/G1 | Printable에서는 128개 중 길이 범위 오류 109/112개, random bytes에서는 81/75개. 나머지는 주로 mask 불일치 |
| P-G-BGV G2 | mask 76, source domain 31, 길이 범위 20, padding 1; valid 0 |
| R-G-BGV G2 | mask 68, 길이 범위 40, padding 2; valid 18개도 joint 0 |
| CGGE G0/G1 | mask 불일치 118/108개, glyph 거리 초과 10/20개 |
| CGGE G2 | glyph 거리 초과 127개, valid·joint 1개 |

G2는 일부 오류 양상을 바꿨지만 필요한 품질에 접근했다고 보기 어렵다. 특히 CGGE는 mask 중심 실패에서 glyph 중심 실패로 옮겨갔다. Decoder의 첫 실패 사유를 집계한 것이므로 각 후보에 다른 오류가 함께 없다고 단정할 수 없다. Clean codec 왕복 1,214개가 통과한 것과 생성 결과가 strict decode를 통과하지 못한 것은 별개의 사실이다.

**Discrete는 학습 신호가 확인되지만, D0와 D1의 잔여 문제가 다르다.**

| 모델 | 후보 수 | 형식 오류 | Valid지만 조건 불일치 | Joint |
|---|---:|---:|---:|---:|
| P-DISC D0 | 128 | 19 | 9 | 100 |
| R-DISC D0 | 128 | 12 | 9 | 107 |
| P-DISC D1 | 128 | 0 | 23 | 105 |
| R-DISC D1 | 128 | 0 | 16 | 112 |

D0는 EOS 개수, payload token, EOS 이후 padding 오류가 남았다. Valid일 때 조건 일치율은 P 91.74%, R 92.24%로, 형식만 고쳐도 자동으로 95.31% 기준을 만족한다고 말할 수 없다. D1은 형식 유효율을 100%로 만들었지만 조건 오류가 남았다. P-DISC D1의 정상 46건·반전 59건을 평균내어 통과시킬 수 없으며, R-DISC D1도 정상 3건·반전 7건이 각각 기준에 부족하다. 이 표만으로 D1이 모든 면에서 D0보다 우월하다고 결론내리지는 않는다.

**D1의 가장 강한 단서: 길이의 최빈값은 맞지만, 확률 분포가 충분히 집중되지 않았다.** 최종 checkpoint의 선형 length head 가중치를 CPU에서 읽어 16개 학습 조건의 분포를 재계산했다. 새로운 후보를 생성하거나 평가 gate를 바꾸지 않았다.

| 길이 진단 | P-DISC D1 | R-DISC D1 |
|---|---:|---:|
| 목표 학습 길이가 argmax인 조건 | 16/16 | 16/16 |
| 목표 학습 길이의 평균 확률 | 82.54% | 85.67% |
| 목표 학습 길이의 최소 확률 | 66.82% | 65.07% |
| Length CE 평균 | 0.1943 | 0.1581 |
| 실패 중 목표 학습 길이와 다른 길이 | 22/23 | 14/16 |
| 실패 중 다른 조건의 학습 메시지와 완전 동일 | 20/23 | 15/16 |

등록 sampler는 argmax가 아니라 temperature 1의 categorical sampling으로 길이를 선택한다. 따라서 16조건 모두 가장 가능성 높은 길이를 맞혀도, 실제 추출에서는 다른 길이가 상당수 나온다. 실제 joint 성공 217건은 전부 목표 학습 메시지와 같은 길이였고, 이 중 215건은 목표 메시지 전체를 그대로 재현했다. 반대로 실패 39건 중 36건은 목표 학습 길이와 달랐으며, 35건은 다른 학습 메시지를 그대로 생성했다.

코드에서 학습은 실제 메시지 길이를 denoiser에 주고, 생성은 length head에서 뽑은 길이를 준다. **이 작은 학습집합에서는 길이 분포의 불확실성과 denoiser의 길이 의존이 함께 오류를 만드는 것으로 추정된다.** 이는 상관관계와 실행 구조에 근거한 진단이며, 길이를 통제한 생성 비교로 인과관계를 확정한 것은 아니다. 길이가 달라도 prefix를 맞힐 수 있으므로 목표 학습 길이의 확률을 joint의 수학적 상한으로 취급해서도 안 된다.

Length argmax 16/16은 진단일 뿐 128개 후보의 성공 결과를 대체하지 않는다. Argmax로 sampler를 바꾸거나 학습 길이를 주입한 출력을 기존 gate의 성공으로 세면 안 된다. 16개 메시지를 암기하는 것은 P2A 목적에 부합하지만 P2B의 미학습 조건 일반화를 보장하지 않는다.

**Loss 감소는 생성 성공과 분리해서 봐야 한다.** 13개 실행 모두 2,000개 update telemetry가 연속으로 존재하고 loss가 유한하다. 각 실행의 처음 100회 대비 마지막 100회 평균 loss는 감소했다. D0는 P 1.3969→0.00649, R 1.7097→0.00850까지 낮아졌지만 여전히 생성 실패가 있다. D1은 P 4.3580→0.2039, R 4.5817→0.1670이며, 최종 length CE 0.1943/0.1581이 잔여 loss와 비슷한 규모다. 마지막 100회 평균과 최종 checkpoint의 값이므로 정확한 loss 분해로 보지는 않는다. P2A의 `curve=[]`, `best_loss=null`은 validation/best 선택 대신 final-update를 사용하는 설정과 맞으며, 학습 로그 누락으로 해석할 근거는 없다. 서로 다른 objective의 Gaussian/Discrete loss 절댓값도 직접 비교하지 않는다.

최신 P1은 이전 실행과 같은 집계다. Gaussian valid 0/2,880, D0 2/640, D1 640/640이며 joint는 전부 0이다. P2A의 Discrete 개선은 8 updates에서 2,000 updates로 늘고 학습·평가 데이터도 달라진 결과이므로, 동일 조건의 성능 향상 실험으로 비교하지 않는다. P1 반전은 `NOT_MEASURED`, P2A 반전은 실제 측정이다. 모든 P1/P2A 후보의 MD5 호출 수는 0이며 이번 결과는 MD5 역상 성능을 측정한 것이 아니다.

**다음 작업의 우선순위는 P2B 확대가 아니라 원인 진단과 새 revision 설계다.**

1. D1에서 length CE와 payload CE를 분리하고, 조건별 길이 확률·생성 길이·prefix 오류를 함께 기록한다. 작은 진단 실험으로 categorical/argmax 길이 선택 및 길이 통제에 대한 payload 반응을 비교해 위 가설을 검증한다. 진단 출력을 등록 성공에 섞지 않는다.
2. Gaussian은 기존 작은 학습집합에서 timestep별 x0·mask·길이 slot·active glyph 오차와 strict decode 실패를 함께 관찰한다. 현재 고급 `diagnostics.json` 생성은 P2B probe에 연결되어 있어 이번 P2A에는 없다. 먼저 이것을 개발 진단에 연결해야 sampler·학습·표현 중 병목을 좁힐 수 있다.
3. 그 증거로 다음 revision의 학습 예산·길이 학습/샘플링·표현 변경 여부를 정한다. 같은 revision에서 seed 재시도, 기준 완화, 사후 repair로 통과시키지 않는다. 기존 봉인 workdir은 보존하고 변경된 코드·명세로 새 workdir을 사용한다.
4. 모든 pipeline의 등록 후보 선택이 실패했으므로 P2B·P3·MD5 본실험은 보류한다. 현 명세의 다음 상태는 `BLOCKED_DEVELOPMENT_NEW_REVISION_REQUIRED`다. D1은 우선 진단 대상이지 이미 선택된 공식 profile이 아니다.

P2A 평가 NFE는 131,840회이며 학습 연산은 포함하지 않는다. `resources.json`은 `NOT_MEASURED`, 최종 resource seal은 false다. 이번 짧은 과적합 검사 시간으로 P2B/P3의 자원 조건을 보장할 수 없다. P2A summary의 `warnings=[]`도 품질 양호를 뜻하지 않는다. 코드가 P2A에서 warning 목록을 비우므로 `passed=false`, `FAIL_QUALITY`, 실제 지표를 기준으로 판단해야 한다.

원본 근거: [실행 보고](local_experiment_archive/runs/v31-mlx-20260927/report.md), [후보별 선택 결과](local_experiment_archive/runs/v31-mlx-20260927/profile_selection.json), [원본 P2 요약](local_experiment_archive/runs/v31-mlx-20260927/pilot/P2/summary.json), [등록 명세](examples/poc-v3.1-protocol.json), [D1 학습·생성 구현](src/diffusion_hash_inv/mlx_models.py), [P2 gate·diagnostics 구현](src/diffusion_hash_inv/study_p2.py).

재검증 및 추가 집계: [audit.json](local_experiment_archive/analyses/v31-mlx-20260927/audit.json), [분석 스크립트](local_experiment_archive/analyses/v31-mlx-20260927/analyze.py). 프로젝트 루트에서 `.venv/bin/python local_experiment_archive/analyses/v31-mlx-20260927/analyze.py`로 재실행할 수 있다. 원본 실행 파일·실험 코드·명세는 수정하지 않았다.
