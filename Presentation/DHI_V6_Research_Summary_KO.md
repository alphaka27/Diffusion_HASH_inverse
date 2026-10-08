# Diffusion 기반 MD5-12 역상 탐색: V6 실험 종합 보고

- 작성일: 2026년 10월 4일
- 실험 완료일: 2026년 10월 3일
- 등록 프로토콜: `dhi-v6-20260929`
- 기준 실행: `local_experiment_archive/runs/v6-study-r2/`
- 최종 상태: `TERMINAL`
- 종합 판정: **`FINAL_REJECTED`**
- 발표자료: [PPTX 18장 · Windows 한글 호환본](DHI_V6_Research_Summary_KO_Windows.pptx)

이 보고서는 완료된 V6 실행의 목적, 과정, 결과와 결론을 정리한다. 연구 설계는 [연구 계획](../RESEARCH_PLAN_V6.md)과 [등록값](../examples/v6-protocol.json)을, 실측 결과는 [최종 판정 파일](../local_experiment_archive/runs/v6-study-r2/decision.json)과 단계별 실행 기록을 따른다. 계획서에 적힌 사전 예측은 실험 결과로 사용하지 않았다.

## 핵심 요약

학습하지 않은 MD5 12-bit 목표값을 조건으로 제공했을 때, diffusion 모델이 무작위 생성 및 조건을 섞어 학습한 모델보다 일치 입력을 더 잘 찾는지 검증했다. 이미지 기반 Gaussian diffusion과 token 기반 discrete diffusion을 두 입력 분포에서 비교한 총 5개 파이프라인이 대상이다.

5개 모두 조건 사용 적격성을 통과했고, 약화한 MD5인 r=4에서는 구조를 이용하는 능력을 보였다. 그러나 정규 MD5의 W3 12-bit 과제에서는 모든 파이프라인이 두 번째 순차 분석에서 `REJECTED_BOUNDED`로 판정되었다. **두 대조군에 대한 Success@100 이득의 상한은 모두 +0.5%p 미만이며, 가장 큰 상한도 +0.378%p**였다. 등록된 계산 우위 기준을 통과한 파이프라인도 없었다.

이 결과는 실험한 조건에서 최소 관심 효과 이상의 이득을 배제한다. 효과가 정확히 0이라는 주장이나 모든 diffusion 모델의 불가능성, 전체 128-bit MD5 역상 문제에 대한 증명으로 확대하지 않는다.

## 1. 실험 목적

### 1.1 연구 질문

**학습하지 않은 해시값을 조건으로 주면, diffusion 모델이 그 값에 일치하는 입력을 더 잘 생성할 수 있는가?**

과제는 정규 MD5의 64개 step을 계산한 뒤, 출력 중 지정한 W3의 12 bits가 목표값과 일치하는 입력을 찾는 것이다. 목표값을 처음 만들었던 원문을 복원할 필요는 없다. 허용된 입력 범위 안에서 일치하는 임의의 입력을 찾으면 성공으로 센다.

V6는 다음을 확인하도록 설계했다.

1. 조건부 모델이 학습 가능한 조건을 실제 생성에 사용하는지 확인한다.
2. 약화한 해시 구조를 활용할 능력이 있는지 확인한다.
3. 정규 MD5에서 Main이 Random과 Shuffled보다 높은 성공률을 보이는지 판정한다.
4. 생성과 학습에 드는 비용을 고려해 직접 탐색 대비 계산 우위를 평가한다.
5. 입력 표현, 모델 계열, 입력 분포에 따른 차이를 비교한다.

### 1.2 이전 실험에서 남은 문제

| 이전 단계 | 확인한 한계 | V6의 대응 |
| --- | --- | --- |
| v3.1 | Printable Gaussian 출력의 형식 실패로 적격성 미완료 | 입력 alphabet 안의 최근접 prototype으로 디코딩 |
| v4 | checkpoint 선택 규칙 문제로 적격성 차단 | 사전 지정한 최종 checkpoint 사용 |
| v5 | Stage C 첫 분석을 마치기 전에 예산 소진 | 지속 처리량 실측, 예산 봉인, 순차 분석 도입 |
| 음성 결과의 해석 | 효과가 없을 때 모델의 조건 활용 능력을 따로 확인할 필요 | synthetic 적격성 과제와 r=4 양성 대조 수행 |

V5의 공식 판정은 `NOT_ESTABLISHED_BY_BUDGET`로 유지한다. V6 결과는 새 W3 데이터로 수행한 별도 실험의 판정이다. [근거: 연구 계획 §1](../RESEARCH_PLAN_V6.md)

## 2. 실험 과정

### 2.1 비교한 파이프라인

| 파이프라인 | 입력 분포 | 표현 | 모델 | 파라미터 수 |
| --- | --- | --- | --- | --- |
| P-G-BGV | Printable | BGV 비트 이미지 | G3-U | 910,638 |
| P-G-CGGE | Printable | CGGE 문자 이미지 | G3-U | 513,326 |
| P-DISC | Printable | Token | D1-S | 508,668 |
| R-G-BGV | Random Bytes | BGV 비트 이미지 | G3-U | 910,638 |
| R-DISC | Random Bytes | Token | D1-S | 1,252,572 |

- **Printable(P):** `0x21–0x7E`의 94개 문자 기호를 사용한다.
- **Random Bytes(R):** `0x00–0xFF`의 256개 바이트를 사용한다.
- **BGV:** 바이트의 8-bit 패턴을 블록 이미지로 표현한다.
- **CGGE:** 문자를 glyph 이미지로 표현한다.
- **G3-U:** 이미지 표현을 사용하는 Gaussian diffusion 모델이다.
- **D1-S:** token 표현을 사용하는 discrete diffusion 모델이다.

메시지 길이는 4–31이다. 이미지 모델은 입력 분포에서 허용하는 기호 중 최근접 prototype으로 출력 이미지를 디코딩하고, discrete 모델은 출력 가능한 token을 허용 기호로 제한한다. [근거: 등록값](../examples/v6-protocol.json), [연구 계획 §4](../RESEARCH_PLAN_V6.md)

### 2.2 공통 학습과 생성 조건

| 항목 | 설정 |
| --- | --- |
| 모델 run당 학습 | 40,000 updates × batch 256 = 새로 생성하는 학습 쌍 1,024만 개 |
| 주 실험 seed | 0, 1, 2 |
| 주 실험 학습 run | 5개 파이프라인 × Main/Shuffled × 3개 seed = 30개 |
| checkpoint | 사전 지정 최종 checkpoint |
| 학습 규약 | 균일 payload loss, 같은 입력 분포의 공통 학습 stream |
| 해시 목표값 분할 | train 2,816개, validation 256개, test 1,024개 |
| Gaussian 생성 | 선택된 sampling step 25 |
| Discrete 생성 | 32개 interval, NFE 33 |
| 후보 예산 | trial별, seed별 K=100 |
| 공정 비교 | 모든 파이프라인과 방법이 동일한 trial 목록 사용 |

생성 모델은 목표 조건과 난수를 받아 후보 입력을 만든다. 후보를 디코딩한 뒤 해시값을 확인하고, 100개 안에서 한 번이라도 목표값과 일치하면 해당 trial을 성공으로 기록한다.

### 2.3 실험군과 대조군

| 방법 | 후보 생성 방식 | 비교 목적 |
| --- | --- | --- |
| Main | 올바른 해시 조건으로 학습하고 해당 조건으로 생성 | 검증 대상 |
| Random | 입력 분포에서 무작위 후보 생성 | 모델 없이 탐색하는 기준 성능 |
| Shuffled | 학습 데이터의 조건 대응을 섞어서 학습 | 올바른 해시 조건 대응 학습의 추가 효과 |
| MC | 같은 Main 모델에 다른 target 조건 입력 | 생성 시 입력 조건을 사용하는지 점검 |

Stage C의 주 판정은 Main과 Random, Main과 Shuffled의 비교다. MC는 Stage P와 S의 보조 비교에 사용한다.

### 2.4 평가 지표와 사전 판정 기준

**Success@100**은 후보 100개 중 목표값에 일치하는 후보가 하나 이상 나온 trial의 비율이다. 주 효과는 다음 절대 차이로 정의한다.

- `Δ_R = Success@100(Main) − Success@100(Random)`
- `Δ_S = Success@100(Main) − Success@100(Shuffled)`

효과 단위인 **%p는 퍼센트포인트**다. 예를 들어 2.4%에서 2.9%로 상승하면 +0.5%p다.

주 실험은 test 목표값 1,024개에서 공통 trial 목록을 복원추출한다. checkpoint 봉인 후 최대 24,576개 목록을 정하고, 8,192개씩 평가해 최대 3번의 누적 분석(look)을 수행한다. trial마다 3개 seed의 짝 차이를 평균한 뒤 전체 trial의 평균과 표준오차를 계산한다. 따라서 3개 seed를 서로 독립인 test target 표본으로 단순 합산하지 않는다.

최소 관심 효과는 **δ=+0.5%p**다. 등록된 통계 절차는 5개 파이프라인, 2개 대조군, 양측, 3번의 분석을 보정하는 Bonferroni 동시 구간을 사용한다. 보고한 구간은 일반적인 개별 95% 구간과 구분된다.

| 우선순위 | 조건 | 판정 |
| --- | --- | --- |
| 1 | Δ_R과 Δ_S의 하한이 모두 0 초과 | `POSITIVE`, 감사 후 W4에서 재현 시험 |
| 2 | 위 양성 조건에 해당하지 않고, 두 상한이 모두 +0.5%p 미만 | `REJECTED_BOUNDED` |
| 최종 분석의 추가 분류 | 한 대조군에서만 +0.5%p 이상의 이득을 배제하거나 아직 결정하지 못함 | 해당 제한적 기각 또는 미확정 상태 |

Look 1 또는 2에서 모든 파이프라인이 `POSITIVE`나 `REJECTED_BOUNDED`로 결정되면 함께 중단한다. 이번 실행에서는 모두 두 번째 분류에 해당했다. [근거: 연구 계획 §6](../RESEARCH_PLAN_V6.md)

### 2.5 단계별 실행

| 단계 | 목적 | 기록된 사용 시간 | 수행 결과 |
| --- | --- | --- | --- |
| A | 구현 검사, 조건 사용 적격성, 처리량 측정, 설정과 예산 봉인 | 15.7시간 | 5개 모두 적격, 보완 학습 불필요 |
| C | 정규 MD5 W3 주 시험 | 41.2시간 | Look 2에서 5개 모두 판정 후 공동 중단 |
| R | 양성 결과의 W4 재현 | 미실행 | 양성 파이프라인 없음 |
| P | r=4 W1 양성 대조 | 5.0시간 | 5개 모두 구조 이용 확인 |
| S | 더 큰 모델과 학습량의 규모 탐침 | 5.4시간 | 완료, 추가 신호 없음 |
| 합계 | 실행 원장의 단계별 시간 합계 | **67.3시간** | `TERMINAL`, pending 0건 |

Stage R은 양성이 나온 경우에만 수행하는 조건부 단계다. 이번에는 실행 대상이 없었다. Stage C의 사용 시간은 cap 80시간 중 41.2시간이며, `budget_stop=false`다. 예산이 소진되어 미결 상태로 끝난 경우와 구분된다. [근거: 실행 원장](../local_experiment_archive/runs/v6-study-r2/budget.json), [최종 판정](../local_experiment_archive/runs/v6-study-r2/decision.json)

## 3. 실험 결과

### 3.1 조건 사용 적격성: 5개 모두 통과

정상 조건과 반전 조건을 주는 synthetic 과제에서 5개 파이프라인의 3개 seed가 모두 정상적으로 작동했다. 아래 수치는 각 seed에서 유효한 후보이면서 요청 조건까지 만족한 수다.

| 파이프라인 | 정상 조건 | 반전 조건 | seed별 판정 |
| --- | --- | --- | --- |
| P-G-BGV | 512/512 | 512/512 | 3개 모두 PASS |
| P-G-CGGE | 512/512 | 512/512 | 3개 모두 PASS |
| P-DISC | 512/512 | 512/512 | 3개 모두 PASS |
| R-G-BGV | 512/512 | 512/512 | 3개 모두 PASS |
| R-DISC | 512/512 | 512/512 | 3개 모두 PASS |

총 15개 run 모두 생성 적격성과 CLP 검사를 통과했고, 보완 학습 없이 주 실험에 들어갔다. 이 결과는 모델이 학습 가능한 조건을 사용할 수 있다는 근거다. 정규 MD5에서도 효과가 있다는 결론은 별도의 주 실험에서 판단한다. [근거: A.json](../local_experiment_archive/runs/v6-study-r2/A.json)

### 3.2 정규 MD5 성공률: 대조군과 비슷한 수준

Look 2의 누적 trial 수는 seed별 16,384개다. 아래 값은 3개 seed의 Success@100 평균이다.

| 파이프라인 | Main | Random | Shuffled |
| --- | --- | --- | --- |
| P-G-BGV | 2.309% | 2.494% | 2.401% |
| P-G-CGGE | 2.490% | 2.494% | 2.431% |
| P-DISC | 2.551% | 2.494% | 2.482% |
| R-G-BGV | 2.407% | 2.445% | 2.427% |
| R-DISC | 2.411% | 2.445% | 2.411% |

독립 균등 해시를 가정한 무작위 참고값은 `1 − (1 − 2⁻¹²)¹⁰⁰ ≈ 2.412%`다. 주 실험 성공률은 약 2.3–2.6% 범위였다. 실제 판정에는 이론값 대신 실측 대조군과의 짝 비교를 사용했다. [근거: decision.json의 seed_estimates](../local_experiment_archive/runs/v6-study-r2/decision.json)

### 3.3 주 판정: 모든 이득 상한이 +0.5%p 미만

다음은 효과 추정값과 등록된 동시 95% 구간이다. **표의 수치는 모두 %p 단위**다.

| 파이프라인 | Main − Random: 추정 [구간] | Main − Shuffled: 추정 [구간] | 판정 |
| --- | --- | --- | --- |
| P-G-BGV | −0.185 [−0.487, +0.117] | −0.092 [−0.389, +0.206] | REJECTED_BOUNDED |
| P-G-CGGE | −0.004 [−0.313, +0.305] | +0.059 [−0.246, +0.364] | REJECTED_BOUNDED |
| P-DISC | +0.057 [−0.252, +0.366] | +0.069 [−0.239, +0.378] | REJECTED_BOUNDED |
| R-G-BGV | −0.039 [−0.343, +0.265] | −0.020 [−0.325, +0.285] | REJECTED_BOUNDED |
| R-DISC | −0.035 [−0.338, +0.269] | 0.000 [−0.305, +0.305] | REJECTED_BOUNDED |

모든 구간이 0을 포함하고, 모든 상한이 최소 관심 효과 +0.5%p보다 작았다. 가장 큰 상한은 P-DISC의 Main−Shuffled에서 **+0.378%p**다. 따라서 5개 모두 등록된 최소 관심 효과 이상의 개선을 배제했다. [근거: C.json](../local_experiment_archive/runs/v6-study-r2/C.json), [decision.json](../local_experiment_archive/runs/v6-study-r2/decision.json)

| 분석 시점 | 누적 trial 수/seed | 한계 기각 | 미결 | 조치 |
| --- | --- | --- | --- | --- |
| Look 1 | 8,192 | 2/5 | 3/5 | 계속 |
| Look 2 | 16,384 | 5/5 | 0/5 | 공동 중단 |

Look 1에서 한계 기각된 파이프라인도 다음 블록에 참여했으므로 최종 비교의 trial 수는 모두 같다.

### 3.4 r=4 양성 대조: 약한 해시 구조는 활용

MD5의 step 수를 4로 줄인 W1 과제에서 4,096 trials, K=100, seed 0으로 Main, Random, MC를 비교했다.

| 파이프라인 | Main | Random | MC | Main − Random (%p) |
| --- | --- | --- | --- | --- |
| P-G-BGV | 18.042% | 2.173% | 0.171% | +15.869 |
| P-G-CGGE | 15.137% | 2.173% | 1.416% | +12.964 |
| P-DISC | 18.042% | 2.173% | 0.171% | +15.869 |
| R-G-BGV | 99.976% | 2.246% | 0.684% | +97.729 |
| R-DISC | 100.000% | 2.246% | 0.146% | +97.754 |

5개 모두 Random과 MC 대비 생성 이득 기준인 `GEN_4`를 만족했고, 조건 정보 진단인 `INFO_4`도 만족했다. R-G-BGV의 성공은 4,095/4,096, R-DISC는 4,096/4,096이다. PPTX에서 R-G-BGV가 100.0%로 보이는 것은 소수 첫째 자리 반올림 때문이다.

이 양성 대조는 실험 장치가 약한 해시 구조를 활용할 수 있음을 보여 준다. 정규 MD5의 결과를 기계가 어떤 조건도 사용하지 못해서 생긴 실패로 해석하기는 어렵다. r=4의 성공을 정규 MD5로 일반화하지는 않는다. [근거: P.json](../local_experiment_archive/runs/v6-study-r2/P.json)

### 3.5 조건 정보 진단과 파이프라인 간 비교

CLP(Conditional Likelihood Probe)는 올바른 입력·조건 쌍과 조건을 바꾼 쌍의 모델 점수 차이를 측정하는 보조 진단이다. z가 크면 올바른 대응을 선호하는 신호가 강하다는 뜻이다. Stage C의 최종 판정은 생성 성공률에 근거하며, CLP가 그 판정을 바꾸지 않는다.

| 파이프라인 | r=4 CLP z | 정규 MD5 CLP z |
| --- | --- | --- |
| P-G-BGV | 187.34 | 0.10 |
| P-G-CGGE | 114.31 | 0.12 |
| P-DISC | 181.76 | 0.32 |
| R-G-BGV | 197.35 | 0.25 |
| R-DISC | 183.25 | -0.36 |

r=4에서는 5개 모두 강한 신호를 보였다. 정규 MD5에서는 z가 −0.36에서 +0.32 사이였고, 등록된 신호 기준 z>2.878을 통과한 파이프라인이 없었다.

사전 지정한 6개 파이프라인 쌍에 대해 두 대조군 기준의 효과 차이를 비교한 결과, 총 12건 중 8건은 ±0.5%p 범위의 동등성 조건을 만족했고 4건은 미확정이었다. 통계적으로 차이를 확인한 대비는 0건이다. 미확정 대비가 남아 있으므로 모든 파이프라인이 동등하다고 단정하지 않는다. [근거: decision.json의 CLP_64와 contrasts](../local_experiment_archive/runs/v6-study-r2/decision.json)

### 3.6 계산 비용: 5개 모두 NO_ADVANTAGE

직접 탐색의 기준 처리량은 단일 CPU core에서 입력 분포 샘플링과 MD5 계산을 함께 수행한 **1,436,163개/초**였다. 등록된 비용 비교는 이 처리량을 모델의 burst 생성 처리량으로 나눈 비율 `ρ`를 사용한다.

| 파이프라인 | 후보 생성 비용 비율 ρ | 손익분기 후보당 성공확률 | 등록 판정 |
| --- | --- | --- | --- |
| P-G-BGV | 2,891.3배 | 70.59% | NO_ADVANTAGE |
| P-G-CGGE | 1,545.7배 | 37.74% | NO_ADVANTAGE |
| P-DISC | 78.4배 | 1.91% | NO_ADVANTAGE |
| R-G-BGV | 3,024.7배 | 73.85% | NO_ADVANTAGE |
| R-DISC | 200.6배 | 4.90% | NO_ADVANTAGE |

손익분기 후보당 성공확률은 `ρ × 2⁻¹²`로 계산한다. 이는 **후보 한 개의 성공확률**이며, 후보 100개를 묶어 평가하는 Success@100과 단위가 다르다.

가장 빠른 P-DISC에서도 직접 탐색과 비용을 맞추려면 후보당 성공확률이 무작위 기준보다 약 78배 높아야 한다. 이번 실행은 모든 파이프라인에서 C3의 효과 지지를 얻지 못했고, 등록된 C4 판정도 모두 `NO_ADVANTAGE`였다. 위 표는 학습비를 제외한 생성 비용 비교다. [근거: 최종 보고서 §2·§7](../local_experiment_archive/runs/v6-study-r2/FINAL_REPORT_KO.md), [decision.json의 C4](../local_experiment_archive/runs/v6-study-r2/decision.json)

### 3.7 규모 확대: 추가 신호 없음

선택적 Stage S는 Printable token 파이프라인에서 D1-T-L을 사용했다.

| 항목 | 설정 또는 결과 |
| --- | --- |
| 파라미터 수 | 6,415,308개, 주 실험 P-DISC의 D1-S 대비 약 12.6배 |
| 학습 | 160,000 updates × batch 256 = 4,096만 쌍, 주 실험의 4배 |
| seed | 0 |
| CLP 평가 | 65,536 pairs |
| CLP z | -0.28 |
| 신호 기준 | z>3.2905 |
| 규모 신호 판정 | `SCALE_SIGNAL=false` |
| Success@100 | Main 2.246%, Random 2.319%, MC 2.393% |

생성 성공률은 4,096 trials의 보조 기술통계다. 이번 한 번의 확대 실험에서는 신호를 찾지 못했다. 더 큰 모든 모델이나 임의의 학습량에 대한 결론으로 확대하지 않는다. [근거: S.json](../local_experiment_archive/runs/v6-study-r2/S.json)

## 4. 결론

### 4.1 최종 판정

**V6의 종합 판정은 `FINAL_REJECTED`다.**

등록된 5개 파이프라인은 synthetic 조건과 r=4 해시 구조를 활용할 수 있었다. 그러나 정규 MD5의 미학습 W3 12-bit 목표값에 대한 생성에서는 Random과 Shuffled 각각에 대한 Success@100 이득 +0.5%p 이상을 모두 배제했다. 계산 우위 기준을 통과한 파이프라인도 없었다.

| 판단 항목 | 최종 결과 |
| --- | --- |
| C1 조건 사용 기계 적격성 | 5개 모두 `PASS` |
| C2 약한 구조 이용 | 5개 모두 `GEN_4=true`, `INFO_4=true` |
| C3 정규 MD5 연구 가설 | 5개 모두 `REJECTED_BOUNDED` |
| C4 계산 우위 | 5개 모두 `NO_ADVANTAGE` |
| C5 파이프라인 비교 | 12개 대비 중 동등성 범위 충족 8건, 미확정 4건 |
| 보조 규모 탐침 | `SCALE_SIGNAL=false` |
| 종합 | **`FINAL_REJECTED`** |

등록된 종료 규칙에 따라 V6로 해당 연구 질문을 종료한다. 연구 계획은 같은 질문의 추가 revision을 두지 않는다. [근거: 연구 계획 §15](../RESEARCH_PLAN_V6.md), [최종 판정](../local_experiment_archive/runs/v6-study-r2/decision.json)

### 4.2 결론의 적용 범위

실측 결론은 다음 조건에 한정된다.

- Printable과 Random Bytes 입력 분포, 길이 4–31.
- 정규 64-step MD5의 W3 12-bit 목표값과 등록된 test pool.
- 지정한 5개 모델·표현 조합, decoder, sampler, 주 실험 학습량, K=100.
- 고정된 test pool, checkpoint, seed에 조건부인 통계적 추론.

다음은 이번 결과로 입증하지 않았다.

- 효과가 정확히 0이라는 명제. +0.5%p보다 작은 효과는 남아 있을 수 있다.
- 모든 diffusion 모델과 모든 학습량에서 역상 탐색이 불가능하다는 명제.
- 전체 128-bit MD5 역상, SHA-256, 다른 window·bit 수의 동일한 결과.
- W4 독립 재현. 양성 파이프라인이 없어 재현 단계는 실행하지 않았다.
- 암호의 보안 붕괴 또는 최신 암호분석 공격과의 우열.

## 부록 A. 주 실험 seed별 기술통계

Success@100은 %이며, 두 차이 열은 %p다. 각 seed는 같은 16,384개 trial 목록을 사용한다.

| 파이프라인 | seed | Main Success@100 | Main − Random (%p) | Main − Shuffled (%p) |
| --- | --- | --- | --- | --- |
| P-G-BGV | 0 | 2.374% | −0.092 | +0.012 |
| P-G-BGV | 1 | 2.368% | −0.262 | −0.037 |
| P-G-BGV | 2 | 2.185% | −0.201 | −0.250 |
| P-G-CGGE | 0 | 2.747% | +0.281 | +0.275 |
| P-G-CGGE | 1 | 2.509% | −0.122 | +0.031 |
| P-G-CGGE | 2 | 2.216% | −0.171 | −0.128 |
| P-DISC | 0 | 2.661% | +0.195 | +0.189 |
| P-DISC | 1 | 2.429% | −0.201 | −0.031 |
| P-DISC | 2 | 2.563% | +0.177 | +0.049 |
| R-G-BGV | 0 | 2.496% | −0.098 | −0.128 |
| R-G-BGV | 1 | 2.417% | +0.061 | +0.153 |
| R-G-BGV | 2 | 2.307% | −0.079 | −0.085 |
| R-DISC | 0 | 2.313% | −0.281 | +0.018 |
| R-DISC | 1 | 2.417% | +0.061 | −0.055 |
| R-DISC | 2 | 2.502% | +0.116 | +0.037 |

## 부록 B. 파이프라인 간 효과 대비

아래 값은 각 파이프라인의 **대조군 대비 이득 간 차이**다. 단위는 %p이며, 구간은 등록된 파이프라인 대비 절차의 보조 구간이다.

| 파이프라인 대비 | 기준 대조군 | 추정 [구간] (%p) | 분류 |
| --- | --- | --- | --- |
| P-G-BGV − P-G-CGGE | Random | −0.181 [−0.462, +0.100] | ±0.5%p 내 동등 |
| P-G-BGV − P-G-CGGE | Shuffled | −0.151 [−0.546, +0.245] | 미확정 |
| P-G-BGV − P-DISC | Random | −0.242 [−0.524, +0.039] | 미확정 |
| P-G-BGV − P-DISC | Shuffled | −0.161 [−0.557, +0.236] | 미확정 |
| P-G-CGGE − P-DISC | Random | −0.061 [−0.350, +0.228] | ±0.5%p 내 동등 |
| P-G-CGGE − P-DISC | Shuffled | −0.010 [−0.413, +0.393] | ±0.5%p 내 동등 |
| R-G-BGV − R-DISC | Random | −0.004 [−0.285, +0.276] | ±0.5%p 내 동등 |
| R-G-BGV − R-DISC | Shuffled | −0.020 [−0.420, +0.379] | ±0.5%p 내 동등 |
| P-G-BGV − R-G-BGV | Random | −0.146 [−0.545, +0.252] | 미확정 |
| P-G-BGV − R-G-BGV | Shuffled | −0.071 [−0.467, +0.325] | ±0.5%p 내 동등 |
| P-DISC − R-DISC | Random | +0.092 [−0.312, +0.495] | ±0.5%p 내 동등 |
| P-DISC − R-DISC | Shuffled | +0.069 [−0.332, +0.470] | ±0.5%p 내 동등 |

## 부록 C. 근거 자료

| 자료 | 역할 |
| --- | --- |
| [RESEARCH_PLAN_V6.md](../RESEARCH_PLAN_V6.md) | 연구 질문, 설계, 판정과 종료 규칙 |
| [examples/v6-protocol.json](../examples/v6-protocol.json) | 모델, 학습량, seed, δ 등 등록값 |
| [decision.json](../local_experiment_archive/runs/v6-study-r2/decision.json) | 최종 판정, 효과 구간, seed별 결과, 비용과 보조 분석 |
| [FINAL_REPORT_KO.md](../local_experiment_archive/runs/v6-study-r2/FINAL_REPORT_KO.md) | 실행기가 남긴 최종 연구 보고서 |
| [A.json](../local_experiment_archive/runs/v6-study-r2/A.json) | 조건 사용 적격성과 선택 설정 |
| [C.json](../local_experiment_archive/runs/v6-study-r2/C.json) | 주 시험의 순차 분석, 중단과 최종 결과 |
| [P.json](../local_experiment_archive/runs/v6-study-r2/P.json) | r=4 양성 대조 결과 |
| [S.json](../local_experiment_archive/runs/v6-study-r2/S.json) | 규모 확대 탐침 결과 |
| [budget.json](../local_experiment_archive/runs/v6-study-r2/budget.json) | 단계별 사용 시간 |
| [protocol.frozen.json](../local_experiment_archive/runs/v6-study-r2/protocol.frozen.json) | 실행 전 봉인한 설정과 환경 |
| [window.json](../local_experiment_archive/runs/v6-study-r2/window.json) | 주 시험 W3와 재현 W4의 역할 |

원자료 링크는 이 저장소의 로컬 archive를 참조한다. Markdown과 PPTX만 별도로 전달하면 해당 원자료는 함께 전달되지 않는다.
