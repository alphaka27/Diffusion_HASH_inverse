최신 실행 결과 분석 — 2026-09-27

대상: `v31-p2fix-20260927-183935`, protocol `dhi-v3.1-p2fix-20260927`. 2026-09-27 18:39~19:42 KST에 수행한 MLX 0.32.2 / Metal GPU / float32 개발 실행이다.

**실행은 정상 완료됐지만 최종 판정은 `BLOCKED_DEVELOPMENT`다. P2A는 5개 모두 통과했고 P2B는 5개 모두 실패했다. 이번 개정으로 작은 집합 암기와 Discrete의 미학습 조건 성능은 개선됐다. Gaussian도 저장된 최종 샘플에서는 조건 prefix를 정확히 생성했지만, 메시지 전체의 mask·길이 문법이 무너져 성공으로 인정되지 않았다.**

[원본 실행 보고서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/v31-p2fix-20260927-183935/report.md) · [이전 10k 실행 분석](/Users/choisoonwook/Experiments_local/DHI_AI_gen/V3_1_P2A10K_ANALYSIS_KO.md)

**실행 상태와 검증 범위**

| 단계 | 상태 | 판정 | Active time |
|---|---|---|---:|
| P0 | COMPLETE | PASS | 1.43초 |
| P1 | COMPLETE | PASS | 61.71초 |
| P2 | COMPLETE | BLOCKED_DEVELOPMENT, exit 2 | 3,725.37초 |
| P3·E0·MD5 본실험 | 미실행 | 적격성 미확보 | — |

Active time 합계는 **63분 8.52초**다. P2A 5개는 각각 10,000 updates, P2B 5개는 fresh seed100으로 각각 100 epochs / 15,700 updates를 완료했다. 총 128,500개의 update 번호가 연속이고 loss가 모두 유한하다. `failure.json`은 없으며 exit 2는 품질 기준 미달을 알리는 정상적인 차단이다. P0/P1 PASS는 실행 경로와 복구 검사를 통과했다는 의미다.

분석 과정에서 다음을 확인했다.

- 봉인된 파일 1,142개와 실행 당시 source 49개의 SHA-256이 현재 파일과 일치한다. Frozen protocol, 데이터 hash, profile 선택·자원 기록의 봉인도 일치한다.
- 이전 실행과 synthetic data 및 P2A case 선택이 같다. 학습·validation 조건 집합은 분리돼 있고, 저장된 21,024개 메시지의 source domain과 prefix 라벨이 맞다.
- SQLite 42개를 읽기 전용으로 검사했다. 후보 7,680개의 평가 키, 요청 조건, checkpoint/config identity, 성공 집계가 일치한다. P1 1,920개, P2A 최종 640개, P2A 중간 probe 1,280개, P2B probe 3,840개다. P1 복구용 후보 DB는 이 집계에 포함하지 않았다.
- P2에 저장된 raw tensor 480개를 기존 strict decoder로 다시 디코딩해 ledger의 payload·valid·success와 대조했다. Raw 저장은 각 평가의 첫 16개 정상 조건 샘플에 한정되므로, 모든 후보 tensor를 다시 검증했다는 뜻은 아니다.

[검증 집계 JSON](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2fix-20260927-183935/audit.json) · [Raw 및 이전 최종 가중치 비교 JSON](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2fix-20260927-183935/raw_diagnostic.json)

**P2A: 작은 집합 암기 문제는 이번 평가에서 해소됐다**

P2A는 source별 16개 학습 메시지를 사용한다. 정상·반전 각각 64개에서 joint 61개 이상, 반전의 원래 조건 오성공 1개 이하가 기준이다. Joint는 형식 유효성과 요청한 앞 3바이트의 정확성을 동시에 만족한 수다.

| Pipeline / profile | 2,000 updates 정상/반전 | 5,000 updates 정상/반전 | 10,000 updates 정상/반전 |
|---|---:|---:|---:|
| P-G-BGV / G2 | 64 / 64 | 64 / 64 | 64 / 64 |
| P-G-CGGE / G2 | 64 / 64 | 64 / 64 | 64 / 64 |
| P-DISC / D1 | 45 / 58 | 64 / 64 | 64 / 64 |
| R-G-BGV / G2 | 64 / 64 | 64 / 64 | 64 / 64 |
| R-DISC / D1 | 58 / 55 | 63 / 61 | 64 / 64 |

분모는 각각 64다. 최종 640개 후보는 모두 요청 조건의 학습 메시지 전체와 일치한다. Gaussian 세 모델은 2,000 updates부터 이 검사에서 100%를 달성했다. 이전 10k 실행의 P-G-BGV G2는 최종 12/13, R-G-BGV G2는 15/9였으므로 BGV의 작은 집합 학습은 뚜렷하게 개선됐다.

이는 학습 메시지 암기 성공이다. 조건·길이·suffix가 다양한 큰 집합으로 일반화했다는 증거는 P2B에서 따로 확인해야 한다. 새 개정은 출력 조건 경로와 Gaussian loss를 함께 바꿨으므로, 개선을 한 변경만의 효과로 분리할 수는 없다.

**P2B: Discrete는 근접했지만 Gaussian은 형식 통과가 거의 없다**

P2B는 P2A 가중치를 이어 쓰지 않고 source별 10,000개 메시지로 새로 학습한다. 학습에서 제외한 validation 조건에 대해 정상·반전 각각 128개를 평가하며, 두 변형 모두 valid와 joint **122/128 이상**, 반전의 원래 조건 오성공 3개 이하가 필요하다.

| Pipeline / profile | 정상 valid | 정상 joint | 반전 valid | 반전 joint | Joint 기준까지 부족한 수 정상/반전 |
|---|---:|---:|---:|---:|---:|
| P-G-BGV / G2 | 0 | 0 (0%) | 0 | 0 (0%) | 122 / 122 |
| P-G-CGGE / G2 | 1 | 1 (0.78%) | 0 | 0 (0%) | 121 / 122 |
| P-DISC / D1 | 128 | 118 (92.19%) | 128 | 119 (92.97%) | 4 / 3 |
| R-G-BGV / G2 | 0 | 0 (0%) | 0 | 0 (0%) | 122 / 122 |
| R-DISC / D1 | 128 | 120 (93.75%) | 128 | 122 (95.31%) | 2 / 0 |

분모는 각 열 128이다. 반전의 원래 조건 오성공은 모두 0이다. R-DISC는 반전 조건 기준을 충족했지만 정상 조건에서 2개가 부족해 탈락했다. 평균이나 두 변형의 합계로 개별 기준을 대신할 수 없다. Gaussian의 오성공 0은 대부분 아예 invalid였다는 사실과 함께 읽어야 한다.

최종 profile은 하나도 선택되지 않았다. 자원 측정은 품질 통과 뒤에 수행되는 구조이므로 `NOT_MEASURED`이며, 자원 부족이 확인돼 실패한 실행은 아니다. `final_sealed=false`, `main_ready=false`다.

**Checkpoint 문제는 교정됐고, Discrete의 남은 실패는 모두 한 바이트 오류다**

이번 P2B probe는 실제 epoch10·30·100 가중치를 각각 평가했다. 최종 판정에는 등록 규칙대로 epoch100을 사용했다. D1의 `best_epoch=10`은 최소 validation loss 기록이며, 이번 평가에 epoch10이 사용됐다는 뜻이 아니다. 최종 summary와 probe의 checkpoint hash도 일치한다.

| D1 모델 | epoch10 joint 정상/반전 | epoch30 joint 정상/반전 | epoch100 joint 정상/반전 |
|---|---:|---:|---:|
| P-DISC | 42 / 41 | 99 / 106 | 118 / 119 |
| R-DISC | 20 / 28 | 94 / 93 | 120 / 122 |

최종 후보는 두 source 모두 256/256 valid이고 중복 payload가 없다. P-DISC의 실패 19개와 R-DISC의 실패 14개는 **전부 prefix 세 위치 중 정확히 한 위치만 틀렸다.** EOS/PAD나 길이 유효성 문제를 고치는 것이 이번 D1 실패의 직접적인 해결책은 아니다.

저장된 validation64의 all-MASK 진단에서도 최종 prefix 위치별 평균 정답 확률은 P 98.82/98.40/98.94%, R 98.16/98.77/97.12%다. 다만 이 진단은 관측된 길이와 EOS/PAD 문맥을 주고 계산하므로 실제 생성 성능과 동일하지 않다. 확률 평균을 곱해 joint 성공률로 대체할 수도 없다. 실제 reveal 과정에서 어느 시점에 오답이 확정되는지는 현재 저장 결과로 확인되지 않는다.

전체 validation loss는 P 6.7564 → 7.1091, R 7.7016 → 8.8775로 증가했지만 실제 joint는 개선됐다. 따라서 전체 denoising loss만 최소화하는 checkpoint 선택이 이 과제의 성공률과 어긋난다는 이전 관찰은 여전히 성립한다. 이번 개정은 최종 epoch 고정 선택으로 그 선택 문제를 피했다. 이번 분석에서는 loss를 prefix/suffix별로 새로 분해하지 않았으므로, loss 상승의 구성요소별 기여를 이번 실행에서 직접 측정했다고 주장하지 않는다.

이전 실행의 공식 BEST와만 비교하면 checkpoint 교정 효과와 모델 변경 효과가 섞인다. 더 적절한 비교는 이전 분석에서 별도로 생성했던 **동일 epoch100** 결과다.

| D1 모델 | 이전 epoch100 사후 진단 | 이번 epoch100 등록 평가 | 관측된 증가 정상/반전 |
|---|---:|---:|---:|
| P-DISC | 114 / 110 | 118 / 119 | +4 / +9 |
| R-DISC | 114 / 107 | 120 / 122 | +6 / +15 |

조건과 payload RNG identity가 같음을 후보별로 대조했다. P 정상에서는 이전 실패 9개가 성공으로, 이전 성공 5개가 실패로 바뀌었다. P 반전은 개선 11·악화 2, R 정상은 개선 10·악화 4, R 반전은 개선 17·악화 2다. 전체 방향은 좋아졌지만 일부 후보는 악화됐다. 구조 변경으로 초기 가중치도 달라졌고 단일 seed의 개발 validation 결과이므로, 일반적인 우월성이나 특정 층의 단독 인과효과를 입증한 비교는 아니다.

**Gaussian의 핵심 병목은 조건 prefix보다 메시지 전체의 구조다**

| Gaussian 모델 | 최종 256개에서 최초 검출된 형식 오류 |
|---|---|
| P-G-BGV G2 | mask 불일치 225, 길이 범위 오류 31 |
| P-G-CGGE G2 | mask 불일치 241, glyph 거리 초과 14; 나머지 1개 valid |
| R-G-BGV G2 | mask 불일치 240, 길이 범위 오류 16 |

전체 768개 중 **767개가 invalid**다. 이는 decoder가 처음 발견한 오류의 분류이며, mask 뒤에 다른 오류가 없다는 의미는 아니다.

원인을 더 구체적으로 보기 위해 최종 평가에서 저장된 **각 모델의 첫 16개 정상 조건 raw 샘플**을 조사했다. 모두 strict invalid지만, mask를 무시하고 prefix 위치의 glyph만 읽는 사후 진단에서는 세 모델 모두 16/16이 요청 prefix와 일치했다. CGGE prefix glyph의 최대 최근접 거리도 각 샘플에서 0.00048~0.00123이었다.

| 최종 raw 샘플 진단 | P-G-BGV | P-G-CGGE | R-G-BGV |
|---|---:|---:|---:|
| 조사 샘플 수 | 16 | 16 | 16 |
| Strict valid | 0 | 0 | 0 |
| Prefix 세 위치만 읽었을 때 정답 | 16 | 16 | 16 |
| 활성 mask가 앞에서부터 연속하지 않음 | 15 | 14 | 15 |
| BGV header가 요구하는 mask와 불일치 | 16 | 해당 없음 | 16 |

이 진단은 성공 기준을 완화하거나 후보를 수정한 결과가 아니다. 원본 gate는 그대로 실패이며, 48개 prefix 정답을 48개 생성 성공으로 세지 않는다. 첫 16개 정상 샘플만 보존돼 있어 전체 768개나 반전 조건의 prefix 정확도로 일반화할 수 없다.

그럼에도 이 샘플들은 “Gaussian은 조건을 전혀 못 읽는다”는 설명에 반하는 직접적인 증거다. 현재 관찰되는 문제는 이미지의 여러 위치가 하나의 일관된 길이와 활성 구간을 형성하지 못한다는 것이다. 무작위 길이·suffix를 가진 P2B에서 각 위치를 맞히는 loss 감소가 전체 문법 보장을 대신하지 못했다.

Validation16의 고잡음 t=999 진단에서도 mask MSE는 P-G-BGV 0.6163, P-G-CGGE 0.6185, R-G-BGV 0.6051로 크다. 반면 t=0에서는 각각 0.00153, 0.00265, 0.00121이다. 약한 잡음에서의 복원과 순수 잡음에 가까운 상태에서의 전체 구조 생성 사이에 큰 차이가 남아 있다. Gaussian validation loss는 epoch10→100 모두 감소했지만 strict valid는 개선되지 않거나 1개에 그쳤다. 이번 Gaussian loss는 영역별 합으로 바뀌었으므로 이전 개정의 전체 평균 MSE와 절댓값을 직접 비교하면 안 된다.

**다음 실험의 우선순위**

1. **D1의 잔여 prefix 오류를 먼저 추적한다.** 현재 checkpoint 선택과 형식 생성은 작동한다. 실패 33개의 실제 reverse trajectory에서 prefix 정답 확률, reveal 시점, 당시 문맥을 남기는 작은 진단이 다음 판단에 가장 직접적이다. 그 결과로 학습량·prefix 가중치·조건 출력 경로 중 한 변경만 새 명세에서 비교한다. 단순 추가 학습의 통과를 현재 수치만으로 보장하지 않는다.
2. **Gaussian은 길이와 mask의 공동 생성을 우선 검토한다.** Prefix 조건 주입만 더 강화하기보다 하나의 생성된 길이에 header·연속 mask·padding이 일관되게 종속되는 구조를 별도 개정에서 검토할 근거가 있다. 검증 메시지의 정답 길이를 넣는 방식은 아니다. Decoder 기준 완화나 생성 후 강제 보정으로 이번 실패를 통과 처리하지 않는다. Mask를 해결해도 suffix glyph/domain 오류가 남을 수 있다.
3. **Formal 확대는 계속 보류한다.** 이번 등록 P2B를 통과한 모델이 없으며 E0 자원·exposure audit·통계 calibration도 완료되지 않았다. 이전 결과를 보고 수정한 같은 개발 validation의 성적이므로 프로젝트 전체에서 한 번도 보지 않은 조건에 대한 독립 검증으로 해석하지 않는다. P1/P2 ledger의 MD5 호출 수는 모두 0이며 MD5 역상 성능을 측정한 실행이 아니다.

이번 분석에서는 학습·재생성·hyperparameter 탐색을 수행하지 않았다. 실험 코드·명세·checkpoint·gate는 수정하지 않았으며, 이 문서와 별도 분석 JSON 두 개만 추가했다.
