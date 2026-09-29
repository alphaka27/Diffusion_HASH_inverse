# F1 독립 후속 확증 평가 계획

작성일: 2026-09-29 KST · Protocol: `dhi-followup-f1-20260929`

**현재 상태: 계획 수립·설계 검산 완료. 평가기 구현, 실행 전 검증, 독립 검토, 사전 등록, 본 평가는 아직 하지 않았다.** 새 평가 seed와 test schedule도 생성하지 않았다. [설정 명세](examples/followup-f1-protocol.json)의 상태는 `PLAN_ONLY_NOT_REGISTERED_NOT_EXECUTED`다.

**목표는 고정된 V5 D1-T 모델에서 Random·Shuffled 대비 Success@100 이득이 +0.25%p 이상인지, 새 평가 자료만으로 확증하거나 배제하는 것이다.** 기존 결과를 이어 붙이지 않고, 새 표본과 한 번의 최종 분석으로 판단한다. 기각·지지 어느 쪽으로도 결과를 유도하지 않으며, 경계에 가까운 실제 효과는 최종적으로 미확정일 수 있다.

## 1. 연구 이력과 확증 범위

V5는 C1 적격성을 통과했으나 C의 16시간 한도로 첫 look을 완성하지 못했다. 기존 공식 결과 `FINAL_NOT_ESTABLISHED / NOT_ESTABLISHED_BY_BUDGET`과 원본 디렉터리를 보존한다. 이 계획은 사용자의 독립 후속 평가 요청에 따라 별도 연구로 작성했으며, V5의 종료 규칙을 소급 변경하지 않는다.

설계자는 V5의 모든 완료·부분 결과를 이미 보았다. Main의 완료 성능, 대조군, CLP, 실행 비용을 설계 배경으로 공개한다. 따라서 가설의 착상 자체를 미관측 상태에서 정했다고 주장하지 않는다. **새 자료 수집 전에 고정한 예측을, 기존 자료와 합산하지 않은 새 난수 실험으로 검증**하는 구조다.

확증 대상은 다음에 조건부인 유한 집합의 평균 효과다.

- V5 최종 D1-T Main 3개와 Shuffled 3개의 **고정 checkpoint**.
- V5의 고정 W2 test group 1,024개와 동일 source·K·sampler.
- 세 checkpoint 쌍을 균등하게 선택하고 target을 균등하게 추출했을 때의 새로운 생성 난수에 대한 기대 성공률.

기존 학습 seed 0·1·2는 여기서 고정 모델을 식별하는 번호다. 새 학습 seed 모집단에 대한 랜덤 효과로 취급하지 않는다. 새로운 학습으로 얻는 모델의 평균 성능, 새 test group, 다른 window, 다른 모델 계열까지 일반화하지 않는다. 이를 주장하려면 **별도 사전 등록된 재학습·새 과제 재현 연구**가 필요하다. F1의 주 질문에는 재학습이 필요하지 않다.

## 2. 독립성이 의미하는 것

| 항목 | F1 처리 |
|---|---|
| 가중치·모델 선택 | 6개 모두 그대로 고정. 좋은 seed만 고르거나 ensemble 가중치를 학습하지 않음 |
| Target 값의 집합 | 기존 W2 test pool 유지. 값 자체는 이미 관측됐으며 “새로운 미노출 target 집합”이라고 부르지 않음 |
| 개별 trial | 새 root seed에서 checkpoint 번호와 target을 독립적으로 새 추출 |
| 생성 난수 | F1 전용 namespace에서 새 생성. 기존 후보·난수 identity 재생 금지 |
| 통계 자료 | F1 결과만 사용. V5 완료 seed·부분 Shuffled·과거 CLP를 합산하지 않음 |
| 학습 메시지 | 원래 test/train 분리를 유지하며, 성공 payload의 학습 집합 일치가 0인지 재확인 |

같은 target이 여러 trial 또는 과거 평가에 등장해도 삭제하지 않는다. 고정된 target별 성공확률 아래에서 새 target 추출과 후보 난수가 독립이면 trial 단위 반복 측정은 독립 난수 실험이 된다. Target 중복을 없애면 등록한 복원추출 분포가 바뀐다. 다만 고정 sampler가 과거 seed·trial 난수를 재생하면 독립성이 깨지므로 identity 감사를 필수로 한다.

단순히 W3로 이름을 바꾸는 것으로 독립성이 생기지는 않는다. W3는 재학습과 새로운 노출 감사를 필요로 하는 별도 과제다. 여기서는 **알려진 W2 모집단에서 고정 모델의 조건부 효과**로 질문을 명시해 감사 범위를 분명히 한다.

## 3. 고정 실험 명세

| 항목 | 고정값 |
|---|---|
| 해시 | MD5 64-step, `int.from_bytes(md5(x).digest(), 'big') & 0xFFF` |
| Source | Printable ASCII 33–126, 길이 4–31 균등; Random의 payload bytes는 길이 조건부 iid uniform |
| 모델 | D1-T, 1,866,316 parameters, 기존 최종 40,000-update checkpoint |
| 학습 추가 | **0 updates**; 학습률·가중치·length head를 수정하지 않음 |
| 생성 | MLX float32, temperature 1, 32 reverse intervals + length 1회, NFE=33, remasking 없음 |
| 배치 | 1,024. Batch 1·64·1,024 및 중단 재개에서 후보가 같아야 함 |
| Trials | **N=393,216**개의 새 paired trials |
| 후보 | 각 trial에서 Main·Shuffled·Random 각각 **K=100** |
| 주 지표 | Success@100: 100회 중 한 번 이상 hit인 binary 값 |
| 비교 | Main−Random, Main−Shuffled 두 개 |
| 최소 관심 효과 | **δ=0.0025 = +0.25%p**; 상대 +0.25%와 구분 |
| 분석 시점 | **1회**. 표본 확장·중간 효능/무익성 판정 없음 |
| 보조 지표 | 같은 원장에서 @1/@10, valid, duplicate, hit, 길이, 학습 일치, 시간·NFE |
| 추가 생성 과제 | MC·CLP·B·S·다른 window 재현은 F1에 포함하지 않음 |

각 trial `t`에서 `s_t ~ Uniform({0,1,2})`, `y_t ~ Uniform(test_pool)`을 서로 독립적으로 추출한다. 선택한 `Main-s_t`와 `Shuffled-s_t`, 그리고 Random이 같은 `y_t`를 평가한다. 각 방법의 후보 난수는 별도 namespace를 쓴다.

N개 trial 각각에서 **한 checkpoint 쌍만 평가**한다. 세 쌍을 매 trial 모두 실행하는 설계가 아니다. 모델당 평균 약 131,072 trials가 배정되지만 실제 수는 무작위다. 균등한 수가 나오도록 재추첨하거나 결과를 본 뒤 seed별 평균으로 재가중하지 않는다. 전체 trial 평균이 균등 checkpoint 혼합의 효과를 추정한다. Seed별 수와 효과는 기술통계로 함께 보고한다.

총 후보 수는 다음과 같다.

| 구분 | 후보 |
|---|---:|
| Main | 39,321,600 |
| Shuffled | 39,321,600 |
| Random | 39,321,600 |
| **합계** | **117,964,800** |

생성 효율을 위해 checkpoint별로 묶어 실행할 수 있다. 이때 원장에는 고유한 `global_trial_id`와 해당 모델 내 local index의 대응을 고정 저장한다. 원래 global 순서로 합친 후 paired 차이를 계산한다. 후보 identity는 `(F1, root_seed, purpose, method, training_seed_id, global_trial_id, attempt)`다. 완료한 checkpoint부터 보고 표본을 줄이지 않는다.

## 4. 사전 등록과 난수 봉인 순서

1. 이 계획·JSON·최종 평가 코드·통계 코드·환경·6개 checkpoint·원래 group 파일의 hash를 연결한 manifest를 작성한다. 명세 JSON에는 실제 checkpoint SHA-256이 이미 기록돼 있다.
2. §7의 구현·통계 검사와 validation 전용 비용 검사를 통과한다. 이 단계는 F1 test schedule을 만들거나 test 성능을 계산하지 않는다.
3. 연구 이력, 추론 범위, 코드 검토자, gate 결과, 예산을 포함해 변경 불가능한 timestamp 기록으로 등록한다. 공개 OSF 등록 또는 embargo 등록 등의 외부 기록을 사용하고 receipt/URL을 보존한다. **로컬 SHA-256만으로는 사전 등록 시점이 독립적으로 증명되지 않는다.** 현재 외부 등록은 수행하지 않았다. [OSF 등록 안내](https://help.osf.io/article/330-welcome-to-registrations)
4. 등록 receipt를 저장한 뒤 지정된 실행 담당자가 `secrets.token_hex(32)`를 **한 번만** 호출해 F1 root seed를 만든다. 호출 시각·값·실행 로그를 저장하고 schedule 생성 전에 seed manifest를 별도 봉인한다. 미리 정한 “마음에 드는 seed”로 대체하지 않는다.
5. Root에서 schedule, Main, Shuffled, Random의 namespace를 분리하고 N개의 `(s_t,y_t)`를 한 번 생성·봉인한다. Test draw의 label별·seed별 개수를 보고 다시 뽑지 않는다.
6. 본 평가를 시작한다. 모든 후보·검증이 완료된 후 §5를 한 번 실행한다.

기존 모듈의 전역 `PROTOCOL`·`MASTER_SEED`를 수정하지 않는다. 별도 F1 평가기에서 F1 protocol과 새 root를 **모든 난수 namespace에 포함**해 기존 결정적 sampler를 호출한다. Counter/shape/draw 이름을 기록하고 batch 분할·재개가 같은 identity를 재현하는지 확인한다.

통계적 iid 보장은 독립적인 target·후보 난수라는 실험 가정에 조건부다. 실제 구현은 검증된 의사난수 스트림을 사용하므로 “새 hash 문자열을 만들었다”는 사실만을 수학적 독립성 증명으로 제시하지 않는다. 과거 namespace를 잘못 재사용하거나 난수 stream이 중복되면 무결성 실패다.

## 5. 주 분석: 유한 표본의 동시 효과 구간

방법 `c`가 Random 또는 Shuffled일 때, trial별 성공을 `M_t,C_t ∈ {0,1}`로 두고 `D_t^c=M_t−C_t`를 계산한다. 추정 대상은 고정 모델·pool에서 새로운 checkpoint 선택·target·후보 난수에 대한 `Δ_c=E[D_t^c]`다.

정규 근사 대신 bounded iid 자료의 **empirical Bernstein 구간**을 쓴다. 두 비교 × 두 방향에 각각 실패확률 `a=0.05/4=0.0125`를 배정한다. `D∈[-1,1]`로 스케일을 변환한 식은 다음과 같다.

```text
d̄ = mean(D_t)
s² = sum((D_t − d̄)²) / (N−1)
b  = sqrt(2 s² ln(160) / N) + 14 ln(160) / (3(N−1))
L  = max(−1, d̄ − b)
U  = min( 1, d̄ + b)
```

Maurer–Pontil Theorem 4의 [0,1] 일방향 bound를 [-1,1]에 변환하고 네 방향에 union bound를 적용했다. iid 조건 아래 두 효과 구간의 **동시 포함확률은 최소 95%**다. 두 대조군이 Main을 공유해 서로 상관되어도 union bound에는 비교 간 독립성 가정이 필요 없다. 표본분산이 0이어도 둘째 항을 남긴다. [원 논문, Theorem 4](https://www.learningtheory.org/colt2009/papers/012.pdf)

이 구간은 보수적이다. 더 좁은 Wald 구간을 사후 계산해 주 판정을 대체하거나, 개별 seed·길이·target에서 가장 유리한 구간을 고르지 않는다. 추가 탐색 결과에는 `EXPLORATORY`를 표시하며 전체 효과의 확증 판정을 변경하지 않는다.

**최종 주 판정은 아래 순서로 정확히 한 번 적용한다.**

| 조건 | F1 판정 | 허용되는 결론 |
|---|---|---|
| `L_R > δ` 그리고 `L_S > δ` | `CONFIRMED_MEANINGFUL_GAIN` | 고정 모델 혼합에서 두 비교 모두 +0.25%p를 초과하는 이득 확인 |
| `U_R < δ` 그리고 `U_S < δ` | `EXCLUDED_BOTH` | 두 비교 모두 +0.25%p 이상 이득 배제 |
| `U_S < δ` | `EXCLUDED_CONDITION_GAIN` | Shuffled 대비 최소 관심 효과 배제; 두 대조 모두에 유용해야 한다는 공동 가설 기각 |
| `U_R < δ` | `EXCLUDED_RANDOM_ADVANTAGE` | Random 대비 최소 관심 효과 배제; 공동 가설 기각 |
| 그 외 | `INCONCLUSIVE` | 고정 표본으로 최소 관심 효과를 확증하거나 배제하지 못함 |

등호는 확증·배제에 넣지 않는다. “우월성 검정이 유의하지 않음”만으로 기각하지 않는다.

별도 보조 flag `POSITIVE_VS_ZERO = (L_R>0 and L_S>0)`를 같은 구간에서 계산한다. 작은 양의 효과는 존재하면서 +0.25%p 이상은 배제될 수 있다. 따라서 `POSITIVE_VS_ZERO=true`와 `EXCLUDED_BOTH`는 모순이 아니며 둘 다 보고한다. 이 조항은 작은 실제 효과를 “효과 0”으로 오해하는 것을 막는다.

F1의 양성도 새 학습·새 window의 재현을 뜻하지 않는다. V5의 `SUPPORTED`나 프로젝트 전체의 성공으로 이름을 바꾸지 않는다. 같은 데이터로 여러 번 threshold를 바꾸어 원하는 판정을 찾지 않는다.

## 6. 표본 수와 모의실험 근거

N=393,216은 세 checkpoint를 평균 약 131,072 trial씩 평가하는 크기다. V5 첫 look의 총 model-seed별 trial 수 196,608의 두 배이며, 기존 자료를 포함해 두 배로 만든 것이 아니다. 소표본·정규 근사 의존을 줄이는 보수적 구간의 비용을 반영했다.

독립 Bernoulli 기준률 `p0=1−(4095/4096)^100≈2.4121%`에서 참고 반폭은 **약 ±0.1163%p**다. 실제 구간은 실제 paired 분산으로 결정하며, 이 예상 반폭을 보장하지 않는다.

[설계 검산 스크립트](scripts/validate_followup_f1.py)는 실제 MD5 평가 없이 (Main,Shuffled,Random)의 8개 joint outcome을 multinomial로 추출한다. 공유 Main에 따른 두 비교의 상관을 유지했다. 각 시나리오 20,000회, 고정된 별도 calibration seed를 사용했다. 아래는 **가상 자료의 operating characteristics**다.

| 가정 | 주요 결과 |
|---|---|
| 두 효과 모두 0 | `EXCLUDED_BOTH` **19,997/20,000 = 99.985%** |
| 두 효과 모두 +0.125%p | `EXCLUDED_BOTH` 41.985%, 단일 비교 배제 31.595%, `INCONCLUSIVE` 26.42% |
| 두 효과 모두 정확히 +0.25%p | `INCONCLUSIVE` **99.915%**; 두 비교 모두 배제 0회 |
| 두 효과 모두 +0.5%p | `CONFIRMED_MEANINGFUL_GAIN` **19,989/20,000 = 99.945%** |
| Main·Shuffled 모두 Random보다 +0.5%p | 조건 기여 배제 **99.965%** |

추가로 한 checkpoint에만 이득이 있는 경우, target별 난이도가 다른 경우, 방법 간 hit가 상호배타적인 경우도 점검했다. 8개 시나리오의 동시구간 누락은 20,000회당 25–30회였다. 이는 구현·시나리오 검산이며 모든 실제 분포에서의 검정력을 증명하는 것은 아니다. 오류율 보장의 근거는 §5의 가정과 정리다.

실제 효과가 경계 δ와 같거나 가까우면 미확정이 많은 것이 올바르다. **확증 가능한 설계는 원하는 이진 결론을 보장하는 설계가 아니다.** 이때 추가 seed나 trial을 붙이지 않는다.

검산 결과: [design.json](local_experiment_archive/analyses/followup-f1-design-20260929/design.json). 재현 명령:

```sh
.venv/bin/python scripts/validate_followup_f1.py
```

이 명령은 설계 계산만 수행한다. 본 평가 명령은 아직 구현하지 않았다.

## 7. 실행 전 통과해야 할 검사

**코드·자료 검사.** 명세에 적힌 6개 checkpoint와 training DB, group 파일, V5 frozen source hash를 검증한다. 기존 checkpoint를 읽기 전용으로 로드하며 모델 선택·학습 없이 실행한다. 현재 검산 스크립트는 6개 checkpoint와 group hash를 확인했다. Training DB 전체 hash, 본 평가 구현의 정확성은 실행 준비 단계에서 재확인한다.

**Sampler·identity 검사.** Validation group과 고정 fixture에서 새 wrapper가 같은 key를 준 기존 sampler와 bitwise 동일한지 확인한다. Checkpoint 6개를 모두 포함하고 batch 1·64·1,024, 서로 다른 작업 순서, 중단·재개를 비교한다. F1 global trial에서 local stream으로의 mapping을 포함해 후보 재사용·누락·중복 commit이 없는지 검사한다. 반환된 payload가 우연히 과거와 같다는 이유로 다시 생성하지 않는다.

**통계·원장 검사.** Invalid·duplicate·첫 성공 이후 시도도 K에 포함한다. 주 interval과 판정은 생산 코드의 함수를 직접 호출하는 fixture로 검산한다. 이번 설계 스크립트와 생산 통계 함수를 복사해 서로 다른 로직으로 방치하지 않는다. `N<2`, 분산 0, 모든 성공/실패, δ 등호, 작은 양성+배제의 공존, 미완료 trial 거부를 검사한다. 8개 시나리오 calibration을 생산 경로로 다시 통과한다. 설계 스크립트의 누락률 ≤6% assertion은 코딩 오류 탐지용 여유치이며 명목 α=5%를 변경하는 기준이 아니다.

**독립 검토.** 최종 구현 담당자와 구분되는 검토자가 estimand, 4개 tail의 오류 배분, 원장→outcome→구간의 연결, 새 난수 namespace를 검토하고 이름·검토 대상 hash·검토 결과를 등록 기록에 남긴다. 현재 독립 검토를 완료했다고 주장하지 않는다.

어느 gate라도 실패하면 test를 시작하지 않는다. 수리 과정과 버전을 남기고 **새 결과를 보기 전에** 동일 명세로 gate를 다시 검사할 수 있다. Sample size·효과 기준·architecture를 성능에 맞춰 변경하지 않는다.

## 8. 비용 검증과 예산

V5의 짧은 sampler profile 1,369 후보/s 대신, 완료된 평가 stream 중 느린 **748.04 후보/s**와 Random **126,700 후보/s**를 참고했다. 원장·검증을 포함한 기존 측정으로 F1을 외삽하면 **약 29.29시간**, 안전계수 1.5 적용 시 **43.93시간**이다. 새 wrapper와 더 큰 원장에서 같은 속도가 유지된다는 보장은 없으므로 실행 허가 근거는 아래의 새 측정이다.

| 구분 | 고정 한도 |
|---|---:|
| 등록 전 검사·preflight 실행 비용 | 4시간 |
| F1 생성·검증·재개·최종 집계 합계 | **48시간** |
| 실행 허용 조건 | 전체 비용 예측 ×1.5 ≤48시간, 즉 예측 ≤32시간 |
| Storage / RSS / GPU | 각각 64 GiB |
| 시작·실행 중 최소 여유 디스크 | 20 GiB; 예상 peak 저장 여유도 별도 충족 |
| 운영상 중단 포함 calendar 기한 | 본 평가 시작 후 7일 |

**Preflight는 원래 validation 256 groups만 쓴다.** 예비 test target을 만들지 않는다. 고정된 seed 1의 Main·Shuffled checkpoint와 최종 평가기 경로에서 10분 warm-up 후 Main·Shuffled 각각 **2,097,152 후보**, Random **1,048,576 후보**를 생성한다. 각 learned stream은 학습 집합 조회·전수 재해시·중복 집계·파일 hash까지 포함한다. 지속 생성 중 15분 구간 처리량, 파일별 검증 비용, 마지막 구간의 속도 저하를 기록한다.

예측은 총 누적 처리량과 마지막 구간을 사용한 예측 중 더 느린 값을 택한다. Final join/중복 집계·DB 크기 증가 비용을 포함해 전체 크기로 외삽하고 ×1.5 안전계수를 적용한다. 저장량은 후보당 최대 byte 측정 ×117,964,800에 작업 중 임시 파일·인덱스·복구 여유를 더한다. 과거 단순 선형 추정은 원장 약 **6.96 GiB**이며 peak 보장값이 아니다.

32시간 예측 기준을 넘으면 **미착수**다. 결과를 보지 않은 상태에서 환경 문제를 해결하고 같은 명세의 preflight를 다시 할 수 있으나, 총 preflight 예산은 4시간이다. 이를 넘으면 `NOT_STARTED_RESOURCE`로 종료한다. 더 작은 모델·적은 trial로 자동 대체하지 않는다.

48시간 한도는 실제 작업과 프로세스가 동작 중인 대기시간, 재시도 비용을 모두 누적한다. 계획된 중단은 checkpointed 운영 중단 시각을 남기고 calendar 기한으로 제한한다. Crash가 발생하면 마지막 heartbeat 이후부터 확인 시각까지를 보수적으로 비용에 포함한다. 남은 quota를 증가시키거나 budget 파일을 초기화하지 않는다.

## 9. 실행 중 감시·복구·자료 품질

운영 화면은 완료 후보 수, 소요 시간, 속도, 메모리, 디스크, 무결성 오류만 표시한다. 성공률·방법별 차이·CI·CLP·중간 판정은 모든 N개 paired trials 완료 전에는 열지 않는다. 검증기는 정확성 확인을 위해 hit를 내부적으로 계산할 수 있으나 집계 효과를 노출하지 않는다.

- 모든 method에서 K=100을 소진한다. 성공 뒤 조기 종료, invalid 재추출, duplicate 대체, 미완료 trial의 실패 0 대입은 금지한다.
- 후보 key, payload, target, global trial, attempt를 기록하고 원본 출력에서 별도 경로로 MD5를 전수 재계산한다. 성공 후보의 학습 SHA-256 집합 일치는 0이어야 한다.
- 기술적 crash는 같은 코드·identity에서 stream당 한 번만 이어간다. Commit된 후보는 그대로 사용하고, commit 전 batch는 동일 난수로 재생한다. 재생 비용도 시간에 포함한다.
- Code·sampler·checkpoint를 바꿔야 하는 오류, RNG 재사용, 수정된 봉인, 성공 payload의 학습 누출은 `NOT_ESTABLISHED_INTEGRITY`로 종료한다. 좋은 stream만 남겨 분석하지 않는다.
- 시간·디스크·메모리·calendar 한도 전에 N×3×K와 최종 검증을 완성하지 못하면 `NOT_ESTABLISHED_RESOURCE`다. 부분 결과는 기술통계로만 공개한다.
- 예기치 않게 중간 효과를 열었다면 시점과 접근자를 기록한다. 조기 판정·표본 변경 없이 끝까지 수행하더라도 눈가림 위반을 공개하며, 사후 변경이 있었으면 `PROTOCOL_DEVIATION`으로 분리한다.

본 평가 시작 뒤 늘어난 자원을 적용하는 예외는 두지 않는다. 결과가 미확정이어도 새로운 seed·window를 같은 연구의 확장으로 붙이지 않는다.

## 10. 구현 범위와 실행 순서

학습 코드를 새로 만들지 않는다. 별도 F1 평가 진입점에서 기존 모델 로더·sampler·codec·verifier를 재사용하고, **새 schedule/namespace, global trial mapping, F1 예산, 주 구간·판정, 보고서**만 구현한다. V5 CLI는 종료된 연구의 재개를 거부하므로 `failure.json` 삭제나 V5 `Budget` 우회로 실행하지 않는다. V5의 `stage_c`는 seed×공통-trial 배열과 다른 판정 규칙을 전제하므로 F1의 주 분석에 그대로 호출하지 않는다.

| 순서 | 작업 | 완료 조건 |
|---|---|---|
| 1 | F1 평가기·통계 연결 구현 | §7 fixture·resume·identity 검사 통과 |
| 2 | Validation 전용 비용 측정 | 예측 ≤32시간, 저장/메모리 gate 통과 |
| 3 | 독립 검토·등록 | 검토 기록과 변경 불가능한 registration receipt 확보 |
| 4 | 새 root·schedule 봉인 | Root 단회 추출, N개의 global trial 확정 |
| 5 | 후보 생성·전수 검증 | 모든 N개 trial의 Main/Shuffled/Random K=100 완료 |
| 6 | 한 번의 주 분석 | 두 구간·주 판정·zero 대비 양성 flag 기록 |
| 7 | 연구 종료·공개 | 양성·배제·미확정·자원 실패 모두 같은 형식으로 보고 |

## 11. 산출물과 허용 결론

실행 디렉터리는 새 `local_experiment_archive/runs/followup-f1/`를 사용한다. 계획·코드·고정 설정만 버전 관리하고 원장·가중치·후보·결과는 archive에 둔다. 최소 산출물은 registration receipt, frozen manifest, root/schedule, input hashes, preflight, 검토 기록, 원장, 별도 verifier 결과, 운영 로그, final comparisons, 한국어 보고서다.

`EXCLUDED_BOTH`의 보고 문장:

> 고정된 V5 D1-T Main·Shuffled 각 3개 checkpoint와 기존 W2 test pool에서, 새로 등록한 독립 난수 평가의 Success@100 이득 상한은 Random 대비 {U_R}%p, Shuffled 대비 {U_S}%p였다. 최소 95% 동시 포함확률의 구간으로 두 비교 모두 사전 기준 +0.25%p 이상의 이득을 배제했다. 이 결론은 새 학습 seed·다른 window·모든 diffusion 모델로 일반화하지 않는다.

양성인 경우 실제 하한·상한과 고정 모델 범위를 함께 쓴다. 미확정이면 동일하게 구간과 사유를 보고한다. 어느 경우에도 기존 V5 공식 판정을 덮어쓰거나, 작은 양의 효과가 없다고 단정하거나, full MD5 역상이 불가능하다는 결론으로 확대하지 않는다.

**현 단계에서 완료된 것은 계획·설정·설계 검산이다. 엄격한 확증 결과는 이 계획을 실제로 등록하고 새 평가를 완료한 뒤에만 얻을 수 있다.**
