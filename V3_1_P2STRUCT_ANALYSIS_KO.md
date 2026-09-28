최신 P2-struct 실행 실패 원인 분석 — 2026-09-28

대상: `v31-p2struct-20260928-024118`, protocol `dhi-v3.1-p2struct-20260927`. 2026-09-28 02:41~03:50 KST, MLX 0.32.2 / Metal GPU / float32 개발 실행.

**학습은 정상 완료됐다. 전체 `BLOCKED_DEVELOPMENT`의 직접 원인은 P-G-BGV·P-G-CGGE의 Printable suffix 품질 미달과 R-DISC의 정상 조건 성공 수 1개 부족이다. P-DISC와 R-G-BGV는 품질·잠정 자원 검사·선택 배치 복구를 통과해 선택됐다. 이전 실행의 “전 모델 실패”와는 다른 결과다.**

[원본 보고서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/v31-p2struct-20260928-024118/report.md) · [분석 집계](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2struct-20260928-024118/audit.json) · [재현 가능한 분석 스크립트](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2struct-20260928-024118/analyze.py)

**실행 오류와 품질 실패를 구분한 결과**

| 단계 | 실행 상태 | 판정 | Active seconds |
|---|---|---|---:|
| P0 | COMPLETE | PASS | 1.48 |
| P1 | COMPLETE | PASS | 57.36 |
| P2 | COMPLETE | BLOCKED_DEVELOPMENT, exit 2 | 4,111.90 |

합계 active time은 69분 30.74초다. P2A 5개가 각각 10,000 updates, P2B 5개가 각각 100 epochs / 15,700 updates를 끝냈다. 총 128,500 update 번호가 연속이고 모든 학습 loss가 유한하다. `failure.json`은 없으며 재개 횟수도 0이다. Exit 2는 품질 gate 미달에 따른 반환값이다. 현재 선택된 두 모델의 자원 추정은 `PROVISIONAL_PASS`이며, 세 모델의 실패 원인은 자원 검사 이전의 품질 미달이다.

실행 당시 source 49개와 봉인 파일 1,695개의 SHA-256을 대조했다. Frozen protocol, 데이터, 선택·자원 파일의 hash도 일치한다. 이전 p2fix와 synthetic data 및 P2A case 선택이 같으며, 학습·validation 조건은 분리돼 있다. 저장된 학습·validation 메시지 21,024개의 source domain과 prefix 라벨을 확인했다.

주요 평가 ledger 42개, 후보 7,680개의 평가 키·조건·checkpoint/config identity·성공 집계를 재검증했다. P1 1,920개, P2A 최종 640개, P2A 중간 probe 1,280개, P2B probe 3,840개다. 이 집계에는 복구·자원 측정용 추가 후보를 포함하지 않는다. P2의 저장 raw tensor 480개는 기존 strict decoder로 재검증했다. Raw tensor는 각 평가의 첫 16개 정상 조건 후보에 한정되므로 전체 후보 tensor를 검증한 것은 아니다.

**최종 품질 결과와 이전 실행 비교**

P2A는 5개 모두 통과했다. P-G-CGGE만 정상/반전 64/63이고 나머지는 64/64다. 작은 학습 집합의 암기는 가능한 상태다. P2B는 별도 fresh seed100으로 source별 10,000개 메시지를 학습하고, 학습에서 제외한 validation 조건을 평가한다.

아래 각 성공 수의 분모는 128이다. 성공은 형식 유효성과 요청 prefix 정확성을 동시에 만족한 `joint`다. 정상·반전 **각각 122/128 이상**이어야 하며, 반전의 원래 조건 오성공은 3개 이하가 필요하다. 이번 최종 반전 오성공은 모두 0이다.

| 모델 | 이전 p2fix 정상/반전 | 이번 p2struct 정상/반전 | 이번 valid 합계 | 최종 판정 |
|---|---:|---:|---:|---|
| P-G-BGV, G2 → G3 | 0 / 0 | 5 / 11 | 16/256 | 실패 |
| P-G-CGGE, G2 → G3 | 1 / 0 | 39 / 47 | 86/256 | 실패 |
| P-DISC, D1 | 118 / 119 | 123 / 125 | 256/256 | 통과·선택 |
| R-G-BGV, G2 → G3 | 0 / 0 | 128 / 128 | 256/256 | 통과·선택 |
| R-DISC, D1 | 120 / 122 | 121 / 126 | 256/256 | 정상 조건 1개 부족 |

P는 Printable ASCII source, R은 임의 바이트 source다. 전체 판정은 다섯 pipeline 모두의 통과를 요구하므로 두 모델의 통과만으로 다음 단계가 열리지 않는다. Gaussian 세 모델의 최종 valid 합계는 이전 1/768에서 358/768로 증가했다. G3는 구조·학습 목표·초기 가중치가 달라져, 그 증가를 특정 변경 하나의 독립 효과로 분리할 수는 없다.

**P-G-BGV: prefix·구조는 맞지만 suffix의 바이트 범위가 잘못된다**

최종 256개 모두 BGV 구조 디코딩을 통과해 payload가 ledger에 남았다. 256개 전부 요청한 앞 3바이트가 정확하다. 그런데 240개는 뒤쪽 suffix에 허용 범위 `0x21–0x7e` 밖의 바이트가 있어 `source_domain`으로 탈락했다. Prefix에는 범위 위반이 없고, suffix 총 3,868바이트 중 1,493바이트, **38.60%**가 범위를 벗어났다.

이는 이전의 길이·mask 불일치가 남아서 생긴 실패가 아니다. G3가 길이를 한 번 뽑아 header·mask·padding을 고정한 뒤에도, 활성 payload glyph는 연속값으로 생성하고 BGV decoder는 각 bit block을 threshold로 읽는다. 그 경로에서 Printable 범위에 속하는 바이트만 생성된다는 제약은 없다. 실제 출력이 그 허용 범위를 충분히 학습하지 못한 것이 이번 직접 병목이다.

R-G-BGV는 모든 바이트 값 0–255가 source domain에 들어간다. 따라서 같은 계열 표현에서도 구조와 prefix가 맞으면 이번 synthetic gate를 통과할 수 있다. R-G-BGV의 256/256 통과가 suffix의 균등분포나 실제 데이터 분포까지 검증했다는 뜻은 아니다.

**P-G-CGGE: 남은 오류는 glyph prototype과의 거리다**

최종 256개 중 170개가 `glyph_too_distant`, 86개가 valid이며 valid 86개는 모두 prefix도 정확하다. 길이·mask 오류는 기록되지 않았다. Strict decoder는 활성 glyph 하나라도 가장 가까운 Printable 문자 prototype과의 MSE가 0.1을 넘으면 전체 메시지를 거부한다.

저장된 첫 16개 정상 조건 raw tensor를 조사하면 5개는 valid, 11개는 invalid다. 앞 3개 glyph는 16개 모두 요청 prefix와 일치하며, 샘플별 prefix 최대 MSE는 0.000262–0.000964로 임계값보다 작다. 거리 초과 glyph 28개는 모두 suffix에 있다. 이 표본의 suffix는 211개 glyph다. 저장되지 않은 나머지 invalid 후보의 prefix 정확성을 이 표본으로 단정할 수는 없다.

두 Printable Gaussian 모두 길이가 길수록 전체 형식 통과가 더 어려워지는 양상이 관측된다.

| 생성 길이 | P-G-BGV valid | P-G-CGGE valid |
|---|---:|---:|
| 4–10 | 15/60, 25.00% | 51/75, 68.00% |
| 11–20 | 1/93, 1.08% | 31/102, 30.39% |
| 21–31 | 0/103, 0% | 4/79, 5.06% |

이는 길이별 관측 집계다. 오류가 발생할 위치가 많아질수록 메시지 전체 통과가 어려워진다는 해석과 부합하지만, 길이가 오류를 일으킨다는 독립 인과 실험은 아니다. Prefix MSE를 더 줄이거나 길이·mask를 다시 수정하는 것보다 suffix의 byte/glyph 유효성을 개선할 근거가 강하다. 픽셀 평균 MSE 감소가 메시지 전체의 strict validity를 보장하지 않는 상태다.

**R-DISC: 형식은 모두 유효하며, 확률적 prefix 선택에서 실패한다**

R-DISC의 정상 121/128은 94.53%로 기준 95.31%에 한 후보가 부족하다. 반전 126/128은 통과했다. 두 변형 합계 247/256을 평균내어 개별 기준을 대신할 수 없다.

새로 저장된 실제 prefix reveal 기록은 다음을 보여 준다. 각 모델의 256개 후보 × prefix 3위치, 총 768개 토큰 선택을 확인했다.

| 모델 | 실패 후보 수 | 오답 prefix 토큰 수 | 오답 선택 당시 정답이 argmax였던 수 | 전체 reveal 중 argmax 자체 오답 |
|---|---:|---:|---:|---:|
| P-DISC | 8 | 9 | 9/9 | 0/768 |
| R-DISC | 9 | 10 | 9/10 | 1/768 |

P-DISC와 R-DISC 모두 실패 후보 하나는 두 prefix 위치가 틀렸으며, 나머지 실패 후보는 한 위치만 틀렸다. 이전 실행의 “모든 실패가 한 바이트 오류”를 이번 실행에 그대로 적용하면 틀린다.

R-DISC 오답 토큰 9개는 정답이 가장 높은 확률이었지만 temperature 1의 categorical sampling에서 다른 값이 선택됐다. 이는 sampler가 등록 방식대로 동작하면서 남아 있는 오답 확률 질량을 추출한 경우다. 예를 들어 정상 `case:d7c`의 첫 바이트는 정답 확률 98.92%였으나 오답 15가 선택됐다. 정상 `case:045`의 마지막 prefix 바이트는 정답 5의 확률이 64.72%, 선택된 오답 4가 35.25%였다.

나머지 하나는 정상 `case:1cb`, 생성 길이 29, prefix 두 번째 위치의 step29다. 정답 12의 확률은 28.67%, 오답 14는 66.09%로 실제 모델의 순위도 틀렸다. 따라서 이번 R-DISC 실패 전부를 “정답 분류는 완벽하고 난수만 불운했다”로 설명할 수 없다.

현재 sampler는 한 번 reveal한 토큰을 다시 MASK로 돌리지 않는다. 오답 선택이 발생하면 그대로 최종 prefix가 된다. 오답 step은 R-DISC 11–30에 분포하므로 초반 한 단계의 문제도 아니다. 이 기록은 실제 생성 경로의 관측이며 teacher-forced all-MASK 진단보다 직접적인 증거다.

관측 경로에서 argmax가 맞았다는 사실만으로 “argmax sampler로 바꾸면 128/128”이라고 계산할 수는 없다. 토큰 선택을 바꾸면 후속 문맥과 logits도 바뀐다. Temperature나 결정적 선택은 별도의 생성 규칙 변경이며, 새 경로 전체를 다시 평가해야 한다. 원본 gate 점수를 사후 치환하지 않는다.

**이번 수정으로 개선된 부분과 여전히 남은 loss 불일치**

D1은 이전과 source data·평가 조건·payload RNG·length RNG·샘플링 길이가 동일함을 후보별로 확인했다. P-DISC는 정상 +5·반전 +6으로 통과했다. R-DISC는 정상 +1·반전 +4로 개선됐지만 정상에서 한 개가 부족하다. R-DISC 정상에서는 이전 실패 6개가 성공으로 바뀌는 동시에 이전 성공 5개가 실패로 바뀌었다. 단일 seed의 개발 결과이므로 개선의 안정성을 별도 검증해야 한다.

| 모델 | epoch10 joint 정상/반전 | epoch30 | epoch100 |
|---|---:|---:|---:|
| P-G-BGV | 2 / 2 | 4 / 3 | 5 / 11 |
| P-G-CGGE | 3 / 9 | 28 / 26 | 39 / 47 |
| P-DISC | 84 / 89 | 117 / 118 | 123 / 125 |
| R-G-BGV | 128 / 128 | 128 / 128 | 128 / 128 |
| R-DISC | 81 / 82 | 119 / 120 | 121 / 126 |

모든 최종 probe가 epoch100 checkpoint를 사용했고, training summary와 checksum이 일치한다. `best_epoch`가 P-DISC 20, R-DISC 10이라는 기록은 최소 validation loss의 위치를 뜻하며 최종 평가에 그 가중치가 사용됐다는 뜻은 아니다.

전체 validation loss는 epoch10→100에서 P-DISC 7.631→8.034, R-DISC 8.724→10.055로 증가했다. 반면 joint는 개선됐다. 별도 validation64 all-MASK 진단에서는 P의 prefix CE가 0.1041→0.0041로 감소하는 동안 suffix CE는 4.6799→5.0790으로 증가했고, R은 prefix CE 0.1349→0.0064, suffix CE 5.7809→6.9803이었다. 전체 loss와 prefix 성공의 방향이 엇갈리는 구성요소가 확인된다. 이 진단은 관측 길이를 준 일부 validation 메시지의 고정 mask 진단이므로 전체 validation loss의 정확한 분해나 실제 생성 성공률과 동일하지 않다.

**다음 수정의 우선순위**

1. Printable Gaussian은 **suffix가 허용 byte/glyph 집합에 들어가도록 학습·생성 경로를 개선하는 것**이 우선이다. BGV는 Printable byte 조합, CGGE는 실제 문자 prototype에 대한 정합성을 직접 확인할 변경이 필요하다. 구체적인 objective 또는 출력 표현 변경은 새 개정에서 하나씩 비교한다. 현재 결과만으로 MSE 가중치 증가나 추가 학습이 충분하다고 보장할 수 없다.
2. D1은 **정답 확률의 여유와 생성 문맥에 대한 안정성**을 우선 확인한다. 같은 checkpoint에서 사전 고정한 한 가지 sampling 변경을 별도 진단하거나, temperature 1을 유지하는 새 학습 목표 변경을 비교할 수 있다. 두 가지를 한꺼번에 바꾸면 원인 분리가 어려워진다. Argmax 자체가 틀린 `case:1cb`도 별도로 추적해야 한다.
3. 기존 strict decoder·122/128 기준·seed를 바꾸어 이번 실행을 통과 처리하지 않는다. P-DISC·R-G-BGV의 통과는 보존하되, 모든 pipeline의 formal 준비 완료로 확대 해석하지 않는다. E0, P3, exposure audit, 본실험, 통계 calibration은 여전히 별도 과제다.

이 분석은 저장된 결과에 대한 읽기 전용 조사다. 학습·재생성·hyperparameter 탐색을 하지 않았으며 실험 source·protocol·checkpoint·gate를 변경하지 않았다. 주요 평가 ledger의 MD5 호출은 모두 0이다. 합성 prefix 과제의 개발 결과이며 MD5 역상 성능이나 프로젝트 전체에서 미노출인 조건에 대한 독립 검증 결과는 아니다.
