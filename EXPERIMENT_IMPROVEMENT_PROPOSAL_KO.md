# 실패 원인에 따른 개선 제안

2026-09-25. 대상은 `v3-cli-p0-mps-final-20260925`다. [직전 결과 분석](LATEST_EXPERIMENT_ANALYSIS_KO.md)을 바탕으로 추가 데이터·저장 출력 검사와 코드 검토를 수행했다. 아래 변경은 제안이며 구현·재학습하지 않았다.

**권고: 전체 실행 비용 측정을 먼저 고친 뒤, P-DISC와 P-G-BGV에서 작은 생성 진단을 수행한다. Gaussian은 위치 정보 추가를 첫 모델 변경으로, Discrete는 EOS/PAD 구조를 첫 진단 대상으로 삼는다. Objective·sampling·모델 크기를 동시에 바꾸지 않는다.**

## 1. 원인별 증거와 확실성

| 문제 | 확인된 사실 | 원인 해석 | 확실성 |
|---|---|---|---|
| P3 중단 | P-DISC가 242.426초 상한에서 중단; 학습 연산 합계 약 91.567초 | 예산 산정에서 매 update의 budget 검사·상태 저장 등 비용 누락 | 코드에서 확인 |
| Gaussian 유효 생성 0% | BGV 3,072개·CGGE 3,072개 모두 invalid | 길이·mask·padding·glyph 구조의 생성 실패 | 원장으로 확인 |
| 낮은 loss와 생성 실패 공존 | noise MSE는 감소하나 strict valid rate는 0 | denoising proxy와 자유 생성 품질의 불일치 | 현상 확인, 구체적 해결책은 미검증 |
| 절대 위치 학습 부족 가능성 | ImageUNet에 명시적 위치 채널 없음; condition embedding을 공간 전체에 broadcast | slot 0/header, 앞쪽 prefix, 연속 mask의 역할 구분이 어려울 가능성 | 검증할 가설 |
| Discrete 문법 오류 | P2에서 유효율 P-DISC 20.70%, R-DISC 16.41%; EOS 개수와 PAD 오류 다수 | EOS/PAD를 일반 token처럼 만들고 한 번 reveal한 token을 다시 수정하지 않는 방식의 제약 가능성 | 오류 확인, 인과 기여는 미검증 |
| 데이터/라벨 오류 | 추가 검사한 21,024개 train/validation records의 조건 위반 0; condition split 교집합 0 | 검사 범위에서 주된 실패 원인이라는 증거 없음 | 해당 불변식만 확인 |

추가 검사에서 source별 train unique messages는 각각 10,000개였으며 실제 관측 train conditions는 Printable 2,949/3,072, Random Bytes 2,945/3,072였다. 미관측 train conditions가 일부 있지만 이것만으로 생성 형식 0%를 설명할 수는 없다.

P2에 저장된 첫 16개 정상 조건 raw output도 조사했다. BGV·CGGE 각각 16/16개에서 mask가 비연속적이었다. BGV의 header에는 0, 32, 108, 224 등 허용 길이 4–31 밖의 값이 나타났다. P-DISC는 EOS 개수가 0/1/2/3개인 출력이 각각 4/7/3/2개였다. 이 16개 진단은 전체 후보율의 추정이나 추가 성공 판정이 아니다.

## 2. 최우선: 시간 예산과 관측을 바로잡기

**최소 수정:** `study_pilot.py`의 학습 profile에서 연속 update 구간의 실제 경과 시간을 재도록 한다. 포함 범위는 batch 준비, budget 검사, 연산·동기화, telemetry, 상태 저장이다. Checkpoint·validation은 별도 측정해 합산하거나 구간 안에 포함하되 중복 합산하지 않는다. 생성도 batch 검사·ledger commit까지 같은 원칙을 적용한다.

현재 profile 구간은 `budget.check()`를 제외하지만 제한은 그 비용을 포함한다. 이 불일치부터 제거하면 기존 1.5배 여유 계수의 의미가 복원된다. 과거 실행의 cap을 사후에 수정하거나 누적 시간을 초기화하지 않는다.

**그다음 비용 절감:** 매 update에 수행하는 전체 디렉터리 재귀 순회를 파일 쓰기·checkpoint 등 필요한 경계 중심으로 줄인다. 가벼운 시간·메모리 검사는 유지한다. 큰 쓰기 전 예상 크기와 여유 공간 검사를 유지하고, 상태 저장 빈도를 바꿀 경우 crash 이후 누적 시간 과소계상 및 checkpoint 재연산 비용을 함께 처리한다. 성능 개선을 위해 복구 내구성을 약화시키지 않는다.

**확인할 것:** 새 예산이 긴 Discrete 실행의 실제 전체 시간을 포괄하는지, checkpoint 이후 중단/재개 결과가 동일한지, 재개해도 사용 시간이 리셋되지 않는지 확인한다. 작은 디렉터리에서만 profile하지 말고 완료 run들이 쌓인 상태의 검사 비용도 포함한다. 단순한 242초→큰 상수 변경은 권하지 않는다.

자동 보고서도 P3 전체 gate가 없어도 각 `complete.json`을 검증해 6개 완료·1개 중단·8개 미착수를 표시하도록 바꾸는 것이 좋다. 현재 P3 `NOT_RUN` 판정 표시는 부분 결과를 숨긴다.

근거: [Budget·profile·report 구현](src/diffusion_hash_inv/study_pilot.py), [봉인된 자원 예산](local_experiment_archive/runs/v3-cli-p0-mps-final-20260925/resources.json).

## 3. Gaussian: 위치 정보부터, objective는 별도 비교

### 첫 변경 후보: 공개 위치 채널 두 개

ImageUNet 입력에 고정 x/y 좌표 채널을 추가한다. Noise가 적용되는 생성 대상은 기존 glyph·mask 두 채널 그대로 유지하고, 좌표는 각 denoiser 호출에서 붙이며 출력은 두 채널로 유지한다. 숨겨진 메시지 길이나 정답 이미지는 입력하지 않는다.

이 변경은 header와 prefix처럼 위치마다 역할이 다른 표현을 직접 다루기 위한 작은 가설 검증이다. 기존 구조·width·epsilon objective·sampling steps는 고정하고, 동일한 개발 데이터·학습 횟수·seed로 baseline과 비교한다. 경계 padding으로 위치를 간접 추론할 수 있으므로 현재 모델이 위치를 전혀 알 수 없다고 주장하지 않는다.

좌표 채널이라는 방법 자체는 [CoordConv 원 논문](https://arxiv.org/abs/1807.03247)에 근거한다. 본 실험에서 mask 또는 조건 학습을 개선한다는 것은 아직 검증되지 않은 적용 가설이다.

### 두 번째 후보: x0 예측 또는 구조별 loss의 단일 변경

위치 변경으로 해결되지 않고 denoising 진단에서 구조 오류가 남으면, 다음 실험에서 epsilon 예측과 clean sample(x0) 예측을 비교할 수 있다. 기존 `GaussianDiffusion`이 `prediction_type="sample"`을 지원하므로 새 diffusion 구현은 필요하지 않다. 다만 **설정 한 줄만 바꾸면 안 된다.**

- Pilot의 `model_and_diffusion()`은 현재 epsilon을 기본으로 생성한다.
- Pilot의 `validation_loss()`는 모델 출력을 언제나 noise와 비교한다.
- Pilot의 `sample()`은 모델 출력을 언제나 epsilon으로 사용한다.

따라서 모델 설정·학습 target·validation target·역과정 계산을 모두 일치시켜야 한다. 기존 `predicted_clean()` 등을 재사용하고, x0 모드의 noise 재구성 및 clipping 정책을 명시한다. Oracle 산술 검사는 경로 정합성만 검증하며 learned control의 성공으로 취급하지 않는다.

x0 회귀도 고잡음에서 평균적인 glyph를 만들 수 있으므로 개선을 보장하지 않는다. 손실 변경과 중간 clipping 변경의 효과도 구분해야 한다. 두 변경을 하나의 묶음으로 비교한다면 단일 원인의 효과라고 해석하지 않는다.

구조별 가중 loss는 region별 진단 후의 대안이다. mask·header에 높은 가중치를 무작정 주기 전에 해당 영역의 clean 복원 오차를 측정한다. 좌표 비중은 gradient 중요도의 실측치가 아니다. 합성 과제의 정답 prefix를 직접 출력에 넣는 방식은 condition 학습 검증을 대신하므로 제안하지 않는다.

근거: [ImageUNet 및 GaussianDiffusion](src/diffusion_hash_inv/models.py), [Pilot의 별도 validation/sampler](src/diffusion_hash_inv/study_pilot.py).

## 4. Discrete: EOS/PAD 오류를 조건 예측과 분리하기

현재 reverse sampler는 각 MASK 위치를 확률적으로 열고 이미 생성한 token은 유지한다. 따라서 초기에 EOS나 PAD를 잘못 선택하면 이후 단계가 그 위치를 고치지 않는다. 하지만 이것만으로 실패의 전부를 설명하지 말고 다음 두 상황을 먼저 비교한다.

1. 알려진 clean sequence를 일부 가린 입력에서 위치별 복원 정확도와 EOS 개수를 측정한다.
2. 모든 위치가 MASK인 실제 생성에서 동일 지표와 strict valid rate를 측정한다.

둘 다 나쁘면 표현·학습·조건 경로를 우선 검사한다. 부분 복원은 좋은데 자유 생성만 나쁘면 reverse 과정에서 구조 오류가 고정되는 가설을 우선 검사한다. 복원 진단에 사용한 true sequence/length는 생성 입력으로 전달하지 않는다.

**문법 오류가 계속 지배하면 다음 revision에서 길이를 먼저 생성하는 별도 모델을 검토한다.** 예를 들어 4–31 길이를 categorical로 생성하고 그 길이의 payload를 생성한 뒤 EOS/PAD를 정해진 위치에 표현한다. 사용 길이는 모델이 생성한 값이며 숨은 정답 길이가 아니다. 조건 prefix는 모델이 학습해야 한다.

이 변경은 현재 EOS/PAD도 자유 생성하는 protocol을 바꾸므로 별도 pipeline으로 보고해야 한다. 생성 형식이 구조적으로 보장되는 효과와 조건 학습 효과를 따로 보고하고, 같은 구조를 사용하는 Random/Shuffled 대조군에도 같은 규칙을 적용한다. 실패 후보를 나중에 잘라 맞추고 기존 점수로 보고하는 방식은 제안하지 않는다. 비용이 더 큰 Transformer 도입은 이 진단보다 뒤다.

근거: [MaskedDiffusion](src/diffusion_hash_inv/discrete.py), [TokenCodec](src/diffusion_hash_inv/encoding/tokens.py).

## 5. 최소 검증 순서와 판단 규칙

아래 순서는 새 개발 revision의 제안이다. 실행 전 조건·규모·종료 기준을 고정하며 기존 v3 gate를 소급 변경하지 않는다.

| 순서 | 실험 | 알아낼 것 | 다음 행동 |
|---|---|---|---|
| A | 전체 주기 시간 측정 및 복구 검사 | 모델 외 비용을 포함한 예산이 실제로 맞는가 | 일치해야 학습 비교 진행 |
| B | P-DISC·P-G-BGV 각각 16개 고정 train records의 작은 과적합 검사 | 가장 쉬운 학습/생성 경로가 동작하는가 | 같은 train conditions에서도 실패하면 큰 데이터·긴 학습보다 경로 진단 우선 |
| C | 저장 checkpoint로 validation의 정상/반전 조건과 고정 RNG 생성; clean 부분 복원과 비교 | 형식·조건·free generation 중 병목 구분 | 관련된 한 변경만 선택 |
| D | P-G-BGV baseline 대 좌표 채널; 필요할 때만 x0 별도 비교 | 위치 정보와 objective 가설 | 유효율·조건 지표가 개선되는 변경만 유지 |
| E | 개선 후보의 validation 평가 및 지정 seeds 확인 | 특정 seed/학습 case에만 맞춘 개선인지 | 동작·품질이 확인된 뒤 정규 P3 설계 |

B의 제안 규모는 source Printable, pipeline별 1 seed, 최대 2,000 updates, 학습에 쓴 16 conditions × 고정 generation seeds 4개다. 진단 목표로 strict valid와 joint 성공 각각 95% 이상을 둔다면 64회 중 최소 61회가 필요하다. 이는 제안한 작은 과적합 진단 기준이며 미관측 조건 일반화나 정식 P3 통과가 아니다. 개발 실패 기록을 보존하고 성공 seed만 고르지 않는다.

C의 비용은 checkpoint 하나당 고정 validation 32 conditions × 정상/반전, K=1로 제한할 수 있다. Gaussian sampling discretization이 의심될 때만 같은 checkpoint·초기 noise의 100-step 대 200-step을 추가 비교한다. DDIM은 sampling 비용과 품질을 조절하는 계열이라는 근거가 있지만, 이 과제에서 step 증가가 해결책이라는 증거는 없다. [DDIM 원 논문](https://arxiv.org/abs/2010.02502)

진단값은 strict valid rate, 최초/독립 구조 오류, 각 prefix 위치 정확도, valid 후보 중 constraint 정확도, 전체 normal/flipped joint, 원래 조건 오성공, 시간/후보를 포함한다. 유효 후보가 없으면 conditional accuracy는 0이 아니라 미정의로 표시한다. Invalid raw tensor의 prefix 검사값은 진단용이며 성공 횟수에 포함하지 않는다. 출력이 조건 교체에 반응한다는 사실만으로 올바른 조건 학습이라고 판정하지 않는다.

우선 새 지표를 기존 checkpoint 선택과 나란히 기록한다. Synthetic validation 성공률을 선택 기준으로 바꾸려면 별도 revision으로 규칙을 사전 고정하고 Main/Shuffled에 동일하게 적용한다. MD5 test 성공으로 선택하지 않는다.

## 6. 실험 해석을 보존하는 범위

기존 P3 test 결과를 이미 분석했으므로 수정한 모델을 같은 test에서 평가하고 독립적인 최초 검증으로 다시 부를 수 없다. 새 revision에는 개발에 사용한 cases·변경 사유·최종 검증 범위를 명시한다. 같은 12-bit 공간은 4,096개 조건뿐이므로 split seed를 바꾼 것만으로 프로젝트 전체에서 새 조건이라고 주장하지 않는다.

기존 strict decoder와 P3의 seed별 정상/반전 각 475/512 문턱은 기존 결과에 그대로 유지한다. 단순 epochs 증가, 대형 모델 도입, 실패 후보 재추첨, decoder 문턱 완화, synthetic 정답 prefix 주입은 우선 개선안에서 제외한다.

**권장 착수 범위는 시간 측정 수정, 부분 결과 보고, 두 모델의 작은 학습·생성 진단까지다.** 이 결과가 나오기 전에는 15개 정규 모델 전체 재학습이나 MD5 본실험에 비용을 쓰지 않는 편이 타당하다.
