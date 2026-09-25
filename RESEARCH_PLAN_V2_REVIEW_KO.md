# 현재 실험 계획 엄밀 검토 보고서

**검토일:** 2026-09-24 KST  
**대상:** `dhi-poc-v2-20260923` — [연구 계획 v2](/Users/choisoonwook/Experiments_local/DHI_AI_gen/RESEARCH_PLAN_V2.md), [protocol JSON](/Users/choisoonwook/Experiments_local/DHI_AI_gen/examples/poc-v2-protocol.json) 및 현재 남아 있는 소스  
**종합 판정:** 개발·측정 경로를 검증하는 PoC에는 조건부 적합. 현재 상태로 확증적 hash 본실험을 시작하는 것은 부적합이며, 통계 해석·노출 감사·v2 실행 경로·학습 대조군·자원 예산을 먼저 닫아야 한다.

## 1. 검토 범위와 결론

v2는 연구 질문, 성공 판정, 대조군, 후보 예산, 실패 처리와 주장 범위를 상당히 구체적으로 정의했다. 특히 원문 복원과 임의의 유효 preimage 발견을 구분하고, invalid·중복 후보를 예산에 포함하며, 세 seed에서 두 대조군을 모두 이기도록 한 점은 타당하다.

그러나 **명세가 구체적인 것, 실행 경로가 완성된 것, 그 실험으로 원하는 과학적 주장을 할 수 있는 것은 서로 다른 상태**다. 현재는 첫 번째가 상당 부분 충족됐고 두 번째와 세 번째에 중요한 조건이 남아 있다. 문서의 `DESIGN_DEFINED_EXECUTION_BLOCKED` 상태는 현재 증거와 부합한다.

| 판단 대상 | 판정 | 핵심 근거 |
|---|---|---|
| 측정 가능한 연구 질문인가 | 적합 | q=12, K=100, 고유 표적 2,048개, 두 대조군과 세 seed 고정 |
| 입력 누출과 후보 선별을 통제하는 명세인가 | 적합 | 12-bit-only 경계, train-only shuffle, 모든 attempts 기록 |
| 미관측 digest 일반화를 검토하는 PoC인가 | 조건부 적합 | train/test digest 분리로 암기 성공을 차단하지만 매우 강한 일반화를 요구 |
| 확증적 평균 성능 우위를 검정할 준비가 됐는가 | 미완료 | 고정 표적 estimand와 McNemar·bootstrap의 확률 모형을 최종 정합화해야 함 |
| 현재 코드로 v2를 실행할 수 있는가 | 부적합 | 259차원 입력, step 학습, inference shuffle, 1,000표적 제한 등 불일치 |
| 현재 GPU를 활용할 기반이 있는가 | 있음 | 보존된 MPS 학습·추론 검증 기록. 전체 v2 검증은 아님 |
| 작은 개선이나 일반적인 학습 불가능성을 판정할 수 있는가 | 부적합 | 큰 효과와 안정적인 세 seed에 맞춘 제한된 검정력 |
| 계산량 우위나 암호학적 공격 성능을 입증할 수 있는가 | 범위 밖 | 후보 수만 맞춘 비교이며 전처리·학습·신경망 추론 비용은 별도 |

이번 검토에서는 계획·소스 정적 분석, 수식 검산, CPU에서 모델 구조 생성 및 parameter count 확인, 원 논문·공식 문서 확인을 수행했다. 모델 학습, sampler 실행, primary holdout 평가는 수행하지 않았다. 삭제된 v2 pilot 구현·결과를 복원하거나 판정 근거로 사용하지 않았다. 기존 계획과 구현도 수정하지 않았다.

증거는 다음과 같이 구분한다.

- **현재 직접 확인:** 현행 문서·소스, 아래의 새 산술 검산과 모델 parameter count.
- **보존된 검증 기록:** 기존 codec 검사, engineering dataset 구성, 검정력 simulation, GPU smoke. 이번 검토에서 재실행한 결과가 아니다.
- **설계상 추론:** 분할·목적함수·통계 모형으로부터 도출한 위험과 해석 범위. 실제 학습 실패나 성공의 관측으로 표현하지 않는다.

## 2. 현재 실험이 실제로 답하는 질문

현재 질문은 “새로운 12-bit MD5 조건에 대해, 학습된 후보 생성기가 100번의 생성 기회 안에서 유효한 preimage를 찾는 비율이 원래 source에서의 무작위 생성 및 shuffled-training model보다 높은가”다.

| 구성 | 고정 내용 |
|---|---|
| Sources | Printable 94문자, Random Bytes 256값; 길이 4–31 균등 |
| Primary pipelines | P-G-BGV, P-G-CGGE, P-DISC, R-G-BGV, R-DISC |
| Hash/condition | MD5 serialized digest의 앞 12 bits만 입력 |
| Ownership | Source별 train 1,536 / validation 512 / test 2,048 digest 값 |
| Training corpus | Source별 unique messages 10,000개 |
| Learned methods | Main, 별도 학습 Shuffled; 각 seeds 0/1/2 |
| Primary outcome | 표적별 100 attempts 내 valid preimage 존재 여부 |
| 추론 | Pipeline당 6개 component의 max-p, 5개 composite에 Holm |
| 허용되는 결론 | 지정 source·표적 집합·학습 절차·세 seed·budget에서의 제한된 우위 |

MD5 계산과 앞 12-bit 선택은 별개의 명세다. `Int_big(MD5(payload)) >> 116`을 사용하고 header/EOS/PAD를 hash하지 않는 정의는 명확하다. 표준 MD5의 serialized bytes를 기준으로 확인해야 하며 내부 word endian을 임의로 바꾸면 다른 과제가 된다. [RFC 1321](https://www.rfc-editor.org/rfc/rfc1321)

### 2.1 가장 중요한 연구적 한계: 단순 생성 성능보다 강한 일반화를 요구한다

Digest ownership 분리는 유출·암기 성공을 막는 좋은 통제다. 동시에 훈련 데이터의 모든 메시지는 train digest에만 속하고, 모든 test condition은 train에서 한 번도 관측하지 않은 조합이 된다. 비트 0/1 자체가 새로운 것은 아니지만, 12-bit 조건 값의 조합은 미관측이다.

따라서 학습 문자열을 그대로 재생하는 모델은 test에서 성공할 수 없다. 학습 분포의 주변분포를 잘 모방하는 것만으로도 충분하지 않다. 모델은 보지 못한 digest 값에 맞는 문자열을 생성할 수 있을 정도의 관계를 일반화해야 한다. 이는 일반적인 자연 이미지의 class conditioning보다 강한 요구다. 두 source의 payload 자체도 iid uniform이므로, 자연어의 의미나 자주 나타나는 문자열 패턴이 주어지는 설정은 아니다.

이 해석은 **학습이 불가능하다는 증명은 아니다.** 실제 MD5가 이상적인 random oracle과 동일하다는 가정도 하지 않는다. 다만 denoising 성능과 codec round-trip이 좋아도 본 가설의 성립 가능성이 높아졌다고 바로 결론 내릴 수 없다는 뜻이다.

**권장:** 연구 목적에 “digest-disjoint 조건 일반화”를 명시하고, 결과를 생성 유효도·조건 반응·실제 hash 우위의 세 층으로 나누어 설명한다. 해시 성능이 없을 때 어느 층까지 검증했는지 남겨야 연구 결과가 해석 가능하다.

### 2.2 q=12의 의미와 실용적 가치

Hash 출력이 균등하다는 설계 근사에서 random의 성공률은 다음과 같다.

\[
p_0(K)=1-(1-2^{-12})^K.
\]

| K | Random success 근사 |
|---:|---:|
| 1 | 0.024414% |
| 10 | 0.243873% |
| 100 | 2.412136% |

N=2,048이면 random 성공 표적은 평균 약 49.40개다. 설계상의 두 배 대안은 Main 4.824272%, 즉 약 98.80개로, 절대 차이는 약 2.412 percentage points다. 이는 관측 결과가 아닌 이론적 설계값이다.

같은 균등·독립 출력 근사에서 한 target의 무작위 탐색 기대 hash 수는 4,096회다. 제한 없이 10,000개를 hash하면 임의 digest가 한 번 이상 나타날 확률은 약 91.30%이고, 모든 4,096개 값을 모으는 coupon-collector 기대치는 약 36,434회다. 이 계산은 전처리 table이 q=12에서 강한 실용 비교 대상임을 보여준다.

현재 benchmark는 의도적으로 test preimage를 모델에서 감추므로 이 table을 primary control에 그대로 추가하면 정보 접근 조건이 달라진다. **현재 설계를 무효화하는 이유는 아니지만, 성공하더라도 효율적인 역산 도구가 입증되는 것은 아니다.** 계산량 비교를 별도 후속 연구로 둔 v2 §14는 타당하다.

## 3. 데이터 분할·노출 감사·일반화 범위

### 3.1 미노출 표적의 여유가 작다

보존된 audit 기록을 기준으로 한 값이며, 이번 검토에서 외부 경로까지 추가 감사한 수치가 아니다.

| Source | 알려진 배제 prefix | 잔여 pool | Test 요구량 | 추가 배제 가능 여유 | 잔여 pool 중 test 비중 |
|---|---:|---:|---:|---:|---:|
| Printable | 1,885 | 2,211 | 2,048 | 163 | 92.63% |
| Random Bytes | 1,860 | 2,236 | 2,048 | 188 | 91.59% |

Project-wide 공통 미노출 pool은 1,181개여서 공통 test 2,048개를 만들 수 없다. Source별 test로 바꾼 것은 알려진 제약에 대한 일관된 해결이지만, 다른 source에서 본 prefix까지 미노출이라는 주장은 허용하지 않는다.

삭제된 파일이나 이전 실험에 대한 연구자의 지식은 파일 삭제만으로 없어지지 않는다. 추가 audit에서 잔여 pool이 2,048개 미만이면 seed 교체로 해결되지 않는다. 따라서 **노출 audit는 비용이 큰 15개 control 학습보다 먼저 끝내야 할 중단 조건**이다.

근거: [v2 노출·ownership 규칙](/Users/choisoonwook/Experiments_local/DHI_AI_gen/RESEARCH_PLAN_V2.md:63), [보존된 audit 검증](/Users/choisoonwook/Experiments_local/DHI_AI_gen/RESEARCH_PLAN_V2_VALIDATION.md:29).

### 3.2 Training 데이터 수와 조건 다양성은 다르다

10,000 messages가 10,000개의 독립 hash condition을 의미하지 않는다. Train ownership 1,536개에 대해 단순 평균은 digest당 약 6.51개 메시지다. 조건부 hash 질량이 균등하고 중복 메시지 제외의 영향이 작다는 근사에서 관측 digest 수 기대치는 약 1,533.72개다. 실제 coverage와 빈도 분포는 dataset 생성 시 기록해야 한다.

세 model seeds가 같은 corpus·ownership을 공유하므로 세 seed 반복은 초기화·학습 난수에 대한 반복이다. 새로운 학습 데이터와 새로운 ownership에서의 재현성까지 입증하지 않는다. 이를 확인하려고 지금 임의로 dataset seeds를 더하는 것은 별도 연구 확장이다.

### 3.3 가중 방식이 목적에 맞는지 명시해야 한다

학습 데이터는 source draw에서 train ownership과 unique message 조건을 통과한 분포다. 반면 validation/test는 digest별 동일 가중치이며 representative는 조건별 첫 draw다. 따라서 학습 loss와 test metric이 평균내는 분포가 완전히 같지는 않다.

현재 질문이 “선택된 digest들에 대한 평균 성공률”이므로 test의 동일 가중치는 타당하다. 다만 원래 source prior에서 임의 메시지를 뽑고 그 hash를 target으로 삼는 경우와 동일한 estimand라고 부르면 안 된다. `p_y≈1/4096` 역시 모든 실제 MD5/source 질량에 대한 증명이 아니다.

## 4. 정보 경계와 재현성

Generator에 12-bit vector만 전달하고 원문·길이·suffix·padding metadata를 분리하는 규칙은 강점이다. Header, EOS, validity mask까지 모델이 생성하게 한 설계도 숨은 길이 oracle을 차단한다.

그러나 인터페이스의 명세와 검증은 별개다. 현재 legacy 함수는 `DigestRecord`를 받으므로 다음 항목을 실제 v2 경로에서 검사해야 한다.

1. 같은 target·checkpoint·난수에서 원문과 길이를 바꾸거나 제거해도 candidate bytes가 같음.
2. Batch 순서·cache key·filename·resume 위치가 hidden metadata에 의존하지 않음.
3. Verifier 결과가 다음 후보의 seed, 재시도 여부, 후보 선택에 들어가지 않음.
4. 중단 전 저장한 rows를 보존하고, 지정 환경·batch policy에서 누락·중복 없이 정확히 이어감.

하위 seed의 문자열 형식은 정의됐지만, namespace마다 어떤 필드를 비워야 하는지와 초기 weight seed의 namespace는 더 명시할 필요가 있다. 예를 들어 ownership/source data에는 pipeline·method·model seed를 넣지 않아야 same-source data 공유가 유지된다. Main/Shuffled 초기화와 train order는 method가 달라도 같아야 하고, generation은 method별로 달라야 한다.

**최소 보완:** namespace별 사용 필드·PRNG·공유 범위 표 한 개와 결정적 fixture를 추가한다. Batch 크기 변경이 난수 소비와 생성 결과를 바꿀 수 있으므로 inference batch와 resume 단위도 봉인한다. GPU 간 또는 버전 간 bitwise 동일성까지 요구할 필요는 없지만, 동일 run의 continuation 조건은 명확해야 한다.

근거: [seed·정보 경계](/Users/choisoonwook/Experiments_local/DHI_AI_gen/RESEARCH_PLAN_V2.md:100), [legacy generation 함수](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/runner.py:285).

## 5. 표현·모델·학습 목표 검토

### 5.1 표현 비교는 알고리즘 계열의 인과 비교가 아니다

현재 소스로 지정 모델을 CPU에 생성하여 parameter count를 다시 확인했다. 학습이나 sampling은 하지 않았다.

| 구조 | Parameters | 상태 표현 | Sampling steps |
|---|---:|---|---:|
| BGV ImageUNet | 114,978 | 2×32×128 pixels | 100 |
| CGGE ImageUNet | 114,978 | 2×32×64 pixels | 100 |
| Printable SequenceDenoiser | 465,168 | 32 tokens | 32 |
| Random-Bytes SequenceDenoiser | 1,136,496 | 32 tokens | 32 |

Discrete는 Gaussian의 약 4.05배 또는 9.88배 parameter를 갖고, architecture·objective·sampler·decode 규칙도 다르다. BGV와 CGGE도 pixel 수와 invalid 판정이 다르다. 따라서 차이를 “discrete가 Gaussian보다 우수하다” 또는 “문자 이미지 표현 자체가 유리하다”로 일반화할 수 없다. v2에서 이 비교들을 탐색적으로 제한한 것은 올바르다.

당장 parameter 수를 억지로 같게 만들 필요는 없다. 현재 연구는 각 pipeline의 두 control 대비 우위가 목적이다. Parameter-matched 또는 compute-matched 비교가 필요하면 별도 요인 실험으로 설계해야 한다.

### 5.2 Gaussian validation loss는 hash 성공의 약한 proxy다

Full-tensor epsilon MSE와 DDIM-style sampling은 서로 연결된 diffusion 구성이다. 다만 일반적인 diffusion 생성 근거가 이 데이터의 preimage 학습을 보장하지 않는다. [DDPM 원 논문](https://arxiv.org/abs/2006.11239), [DDIM 원 논문](https://arxiv.org/abs/2010.02502)

BGV의 평균 길이 17.5를 적용하면 전체 tensor 좌표의 구성은 validity channel 50%, payload glyph 27.34%, padding glyph 21.09%, header glyph 1.56%다. 이는 좌표 비중이지 실제 loss나 gradient 기여율 측정이 아니다. 그래도 평균 denoising loss가 개선되면서 정작 한 byte의 header, mask의 연속성 또는 hash 조건 사용은 충분히 개선되지 않을 수 있음을 보여준다.

Terminal alpha-bar는 새 검산에서도 `4.0358297654×10^-5`였다. Epsilon 오차를 한 시점의 clean 예측으로 변환하는 계수 `sqrt((1-alpha_bar)/alpha_bar)`는 terminal에서 약 157.41이다. 이는 해당 변환식의 민감도이며, 최종 생성 오차가 반드시 157배라는 뜻은 아니다. 계획이 중간 clean prediction을 clip하지 않으므로 큰 예측값·비유한 값·최종 clipping 비율을 engineering 진단으로 확인할 가치가 있다.

Zero terminal SNR 연구는 이런 학습·추론 분포 차이의 근거가 되지만, 현재 모델 실패의 원인이 그 차이라고 입증하는 자료는 아니다. 이를 이유로 즉시 schedule을 변경하는 것도 근거가 부족하다. [관련 원 논문](https://arxiv.org/abs/2305.08891)

**권장 진단:** final-checkpoint의 valid rate, header/length/mask/padding 실패 비율, payload 분포, condition 변경 반응을 별도로 기록한다. Primary hash test를 열어 checkpoint를 다시 고르는 데 사용하지 않는다.

### 5.3 Discrete objective는 명시됐지만 MDLM 재현 실험은 아니다

현재 loss는 sequence별 masked token CE를 masked count로 나눈 surrogate다. MDLM의 weighted likelihood bound와 동일하다고 주장하지 않는 문구는 정확하다. Architecture도 논문의 언어 모델과 다르다. [MDLM 원 논문](https://arxiv.org/html/2406.07524v1)

32개 위치를 확률 t로 masking하고 t를 균등하게 뽑으면 한 sequence에서 mask가 전혀 없는 확률은 `∫₀¹(1-t)^32 dt = 1/33 ≈ 3.03%`다. 이때 loss를 0으로 두는 현재 규칙은 명세와 일치하며 자체적인 bug는 아니다.

Reverse 과정은 reveal한 token을 다시 수정하지 않는다. EOS나 초반 token의 잘못된 선택을 뒤에서 repair하지 않으므로 전체 grammar 유효도가 중요한 병목이 될 수 있다. 이 역시 실제 실패 관측이 아니라 구조적 위험이다. Masked loss, token 정확도, 완성된 sequence의 strict valid rate를 구분해서 기록해야 한다.

### 5.4 100 epochs의 의미

Epoch별 무복원 permutation, 마지막 batch 유지, validation draws 고정 및 최소 loss checkpoint 선택은 재현 가능한 규칙이다. 그러나 100 epochs·Adam .001이 이 과제에 충분하거나 최적이라는 실증 근거는 아직 없다. Search trial을 1개로 고정한 것은 선택 편향을 줄이는 대신 모델 부적합의 위험을 수용하는 선택이다.

따라서 positive control 미통과는 해당 profile로 가설을 시험할 준비가 안 됐음을 뜻한다. Diffusion이나 MD5 조건 학습 일반의 불가능성으로 해석해서는 안 된다.

## 6. 대조군과 후보 예산

### 6.1 Random·Shuffled 구성은 타당하다

Random이 원래 source prior를 사용하고 target의 실제 길이나 ownership으로 후보를 걸러내지 않는 규칙은 공정하다. Shuffled가 학습에서만 pairing을 깨고 validation/inference에는 실제 target을 주는 규칙도 조건 관계 학습의 효과를 확인하는 데 적절하다.

다만 Shuffled는 “조건 정보를 어느 단계에서도 보지 않은 모델”은 아니다. Validation의 실제 condition으로 checkpoint를 선택하므로 그 경로의 supervision은 공유한다. 이는 동일한 선택 절차를 적용한다는 장점이 있으며, 그 대조군을 정확히 **shuffled-training + true-condition validation**으로 설명하면 된다.

Accidental same-digest pairing의 기대 비율은 training digest 빈도 f_y에 대해 `Σ_y(f_y/n)^2`다. Row fixed point 비율 `1/n`과 다르다. 따라서 fixed point만 기록하면 부족하고 계획처럼 same-digest fraction을 남기는 것이 맞다.

### 6.2 Positive control은 필요한 검사이며 충분한 증거는 아니다

첫 세 payload symbol에 12-bit condition을 대응시키고 complement pair를 같은 split에 유지한 설계는 좋은 조건 경로 검사다. 전체 원문을 spatial condition으로 주는 legacy G1보다 본과제와 훨씬 가깝다.

새 Wilson 검산은 기존 문서의 문턱과 일치한다.

| 기준 | 경계값 재검산 |
|---|---|
| 512개 중 정상/반전 조건 joint success 하한 ≥ .90 | 474개면 lower=.899769로 실패, 475개면 .901979로 통과 |
| 반전 생성물이 원래 조건을 만족하는 비율 상한 ≤ .05 | 15개면 upper=.047771로 통과, 16개면 .050156으로 실패 |

즉 관측 성공률 최소치는 475/512=92.7734%다. iid Binomial(512,p) 근사에서 정상 success gate 하나의 통과 확률은 p=.90일 때 1.84%, p=.93일 때 62.18%, p=.95일 때 98.90%다. **실제 성능이 정확히 90%인 모델이 보통 통과하는 문턱이 아니다.** 신뢰하한을 요구한 의도라면 맞는 설계지만 15개 모델 전체에 요구되는 기준의 강도를 인식해야 한다. 실제 condition별 확률 이질성과 gate 간 의존성이 있으므로 이 값들을 전체 gate 통과 확률로 곱하지 않는다.

남는 한계는 세 가지다.

- Positive control은 직접적인 nibble-to-symbol 관계이고 MD5 inverse 관계보다 훨씬 단순하다.
- Synthetic train conditions는 3,072개로 primary 1,536개보다 많고 첫 세 위치의 alphabet도 다르다. Same-path이지 동일 분포·동일 난이도는 아니다.
- Wilson 구간은 운영 문턱으로 명시됐지만, 모든 이질적 고정 condition 또는 15개 모델 전체에 대한 동시 95% 보장을 자동으로 주지 않는다.

**권장:** 기존 strict joint gate를 유지하면서 valid rate, valid 조건하 constraint accuracy, 위치별 nibble accuracy를 원인 진단으로만 추가한다. Gate를 본 뒤 임의로 낮추지 않는다. 같은 synthetic test를 반복적으로 보며 수정했다면 그 test를 독립 held-out 근거로 계속 부르지 말고 개발 이력과 최종 검증 범위를 구분한다.

### 6.3 Attempts 단위의 평가를 유지해야 한다

한 개의 100-attempt stream에서 @1/@10/@100을 계산하고 invalid·중복·성공 후 후보도 모두 세는 규칙은 선택 편향을 막는다. Candidate-level rows를 표적 수로 세지 않는 것도 맞다.

유효도와 hash 조건 활용을 구분하는 보조 설명으로, 한 target에서 후보들이 iid라면 `P(success@K)=1-(1-v·r)^K`로 쓸 수 있다. 여기서 v는 valid 확률, r은 valid 후보가 target hash를 맞출 조건부 확률이다. 관측 valid rate나 중복률은 실패를 설명하는 지표이며 K를 사후에 “유효 후보 수”로 바꾸는 근거가 아니다.

## 7. 통계 추론: 가장 먼저 확정해야 할 설계 사항

### 7.1 Estimand와 검정 모형을 연결해야 한다

계획의 estimand는 고정된 ownership·학습 데이터·checkpoint·seed에서 선택한 표적들의 평균 sampling success 차이다. 이를 엄밀히 쓰면 target i에서 D_i=M_i−B_i, μ_i=E[D_i]에 대해

\[
\Delta_T=\frac1N\sum_{i\in T}\mu_i,
\qquad \widehat\Delta_T=\frac1N\sum_iD_i.
\]

McNemar는 discordant count에 조건을 걸고 Main-only 방향의 수에 대한 binomial tail을 계산한다. Target별 `q_i=P(D_i≠0)`, `θ_i=P(D_i=1 | D_i≠0)`라 두면

\[
\Delta_T=\frac1N\sum_i q_i(2\theta_i-1).
\]

계획처럼 θ_i가 공통 θ이고 target pairs가 독립이면 평균 효과의 부호와 θ−1/2의 부호가 일치하며 조건부 binomial 논리가 연결된다. 문제는 고유 digest, 같은 checkpoint, seed 분리만으로 **공통 discordant-direction 모형이 자동으로 성립하지는 않는다**는 점이다.

이것은 “이질성이 있으면 McNemar가 언제나 무효”라는 판정이 아니다. 예를 들어 iid 표적 모집단에서의 matched-pair 분석은 조건부 난이도가 달라도 별도의 정당화가 가능하다. 현재처럼 표적을 고정하고, 노출 기반 유한 pool에서 비복원 선택하며, train ownership을 나머지에 배정한 설계에 그 설명을 그대로 옮기면 안 된다는 뜻이다.

v2 §11은 이미 이 제한을 인정한다. 따라서 문서 오류를 새로 발견했다기보다 **본실험 전에 결정해야 할 조건을 아직 닫지 않은 상태**다. 유리한 test 결과를 본 뒤 가정을 채택하면 사전 결정의 의미가 사라진다. Exact McNemar의 p-value와 평균 비율 차이의 interval이 서로 다른 문제라는 점도 통계 원문에서 다룬다. [Fay·Lumbard 원 논문](https://pubmed.ncbi.nlm.nih.gov/33263202/)

**필수 결정:** test 공개 전에 ① 고정 benchmark에 대한 명시적 working-model 조건부 주장, ② 확증 주장 없이 기술적 benchmark 보고, ③ 모집단·반복 sampling·분석을 함께 재설계하는 새 revision 중 하나를 정한다. 가정을 방어할 근거가 없으면 우선 ②로 한정하는 것이 현재 증거에 맞다. 단순히 검정 이름을 permutation test로 바꾸는 것도 교환가능성 문제를 자동 해결하지 않는다.

### 7.2 Target bootstrap의 불확실성 대상도 명시해야 한다

같은 target row에서 모든 method·seed·K를 함께 resample하는 것은 pairing 보존 측면에서 옳다. 그러나 고정된 target 목록에 대해 sampling 난수만 다시 실행했을 때의 불확실성과, target 목록 자체를 새로 뽑았을 때의 불확실성은 다르다.

고정 checkpoint 아래 D_i가 독립이면 실제 반복-generation 평균의 분산은 `N^-2 Σ_i Var(D_i)`다. Target row bootstrap은 관측된 표적 간 효과 차이까지 재표집하므로 이것과 자동으로 동일하지 않다. 반대로 새로운 ownership·training set까지 포함하는 불확실성도 재학습 없이 포괄하지 못한다.

따라서 현 percentile interval은 어떤 표적 모집단에 대한 근사인지 명시하거나, 경험적 target-resampling 범위로 기술해야 한다. 이를 exact McNemar와 일치하는 exact CI, simultaneous CI 또는 학습 seed 전체의 uncertainty로 표현하면 안 된다. 이 쟁점은 “bootstrap이 항상 과도하게 좁다”는 주장도 아니다. 고정 표적 효과의 이질성이 추가되어 더 넓어질 수도 있다.

### 7.3 max-p와 Holm 결합은 올바르다

Pipeline 주장 자체가 “두 controls×세 seeds의 여섯 비교 모두 우위”이므로 null은 그중 하나 이상이 null인 합집합이다. Component p-values가 유효하면 true-null component j에 대해

\[
P(\max_jp_j\le a)\le P(p_j\le a)\le a.
\]

따라서 max-p는 이 conjunction claim에 타당하고 component 간 독립을 필요로 하지 않는다. 다섯 composite에 Holm을 적용하는 것도 적절하다. Shared random streams와 shared targets 때문에 pipeline들이 의존한다는 이유만으로 Holm이 무효가 되지는 않는다. [R 공식 다중비교 문서](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)

다만 이 보정은 잘못된 component p-value를 유효하게 만들어 주지 않는다. Missing을 보정 계산에만 p=1로 넣고 실제 outcome을 0으로 채우지 않는 규칙은 유지해야 한다. CI lower>0을 별도의 gate로 더하지 않는 v2 규칙도 현재 legacy 통계 코드와 구분해야 한다.

### 7.4 검정력은 큰 효과·안정적인 seed에 한정된다

아래 값은 보존된 20,000회 simulation 결과다. 이번 검토에서는 simulation 코드와 가정을 확인했으며 같은 계산을 재실행하지 않았다.

| 가정된 시나리오 | 한 pipeline의 통계 충분조건 통과율 | MC 표준오차 |
|---|---:|---:|
| Main=2p₀, 두 control=p₀, 세 seed 동일 | 80.940% | 0.278 pp |
| 한 seed만 Main=1.5p₀ | 23.695% | 0.301 pp |
| Main=2p₀, Shuffled=1.5p₀ | 3.025% | 0.121 pp |
| 두 난이도 strata에서 같은 비율 차이 | 82.980% | 0.266 pp |

이는 여섯 raw p-values가 모두 .01 미만이라는 충분조건의 확률이다. 실제 Holm의 모든 순위를 사용한 전체 연구 power, 다섯 pipeline의 공동 성공 확률, G1/G2 통과 확률이 아니다. Simulation은 두 비교에 같은 Main outcome을 공유하는 의존성을 포함하지만, 지정 난이도 아래 method/seed 결과의 확률 모형은 단순화한다.

특히 Shuffled가 강할 때 낮은 power는 비교 효과 자체가 작아진 데 따른 것이다. Main 2p₀와 Shuffled 1.5p₀의 비율은 약 1.33배다. 이 설계에서 비유의는 작은 개선의 부재를 뜻하지 않는다. 또한 Δ>0 검정을 통과해도 “최소 두 배”를 증명한 것은 아니다.

**권장:** 연구 목적이 큰 효과의 존재 탐색이면 이 제한을 명시하고 유지한다. 작은 효과 배제나 평균 seed 성능이 목적이면 표본·반복 단위·가설을 test 전에 함께 바꾼다. 결과를 본 뒤 max-p를 평균 p-value로 교체하는 식의 변경은 허용하지 않는다.

## 8. 현재 구현과 v2 명세의 차이

아래는 현행 소스에서 직접 확인한 사실이다. Legacy engineering 코드의 본래 목적에 대한 결함 판정이 아니라, 이를 v2 실행기로 사용하면 생기는 불일치다.

| 항목 | 현행 동작 | v2 요구와 영향 |
|---|---|---|
| Canonical condition | 259(+length)차원 강제 | 정확히 12차원 경로와 불일치 |
| Model-facing input | Full DigestRecord 전달 | 숨은 원문·metadata 접근을 구조적으로 차단하지 못함 |
| Shuffled | 공통 `_conditions`가 inference에서도 shuffle | Train-only shuffled control을 구현하지 못함 |
| 학습 | Step마다 replacement batch sampling | Epoch permutation·100 epochs와 다름 |
| Checkpoint | Step 기준 저장·마지막 학습 상태 | 고정 validation draws로 10회 평가 후 최소 loss 선택이 없음 |
| Test 표적 수 | **K=100이면 최대 1,000개로 제한** | v2의 2,048개가 실행 단계에서 잘릴 수 있음 |
| 후보 저장 | 모든 sample의 `tolist()`를 메모리에 누적 | Streaming·run당 raw sample 16개 규칙과 다름 |
| 통계 family | Seed별 Holm 및 CI lower>0 판정 | 6개 max-p→5개 Holm 및 v2 gate와 다름 |
| Positive control | `reversible_record` 과제 사용 | 같은 12-bit 입력의 합성 학습 control을 대체하지 못함 |

직접 근거: [condition 제약](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/runner.py:104), [공통 shuffle 함수](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/runner.py:191), [학습 loop](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/runner.py:267), [generation 저장](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/runner.py:285), [1,000표적 제한](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/runner.py:339), [현재 통계 집계](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/study_statistics.py:9).

특히 test_limit만 2,048로 설정해도 K=100 branch의 `min(limit, 1000)` 때문에 해결되지 않는다. 이 조건은 단순히 문서 값을 바꾸는 것으로 제거할 수 없는 실제 실행 경로의 차이다.

삭제 후 현재 재사용 가능한 것은 model·codec·device·통계 primitive 등이다. 이들을 재사용하는 좁은 v2 경로가 필요하다. 기존 PoC 전체를 복제하거나 여러 실행 프레임워크를 만들 필요는 없다.

## 9. 실행 규모와 GPU 준비도

### 9.1 새로 검산한 최소 workload

| 항목 | 계산값 | 해석 |
|---|---:|---|
| Hash 학습 | 30 runs | 5 pipelines×3 seeds×2 learned methods |
| Synthetic control 학습 | 15 runs | 5 pipelines×3 seeds |
| 최초 학습 합계 | 45 runs | 개발 재시도 제외 |
| Optimizer updates | 706,500 | 45×100×ceil(10,000/64) |
| 학습 sample 처리 횟수 | 45,000,000 | 반복 epochs 포함; unique messages 수가 아님 |
| 고정 validation corruption 평가 | 921,600 sample-evaluations | 45×10 checkpoints×512×4 |
| Primary learned candidates | 6,144,000 | 30×2,048×100 |
| 공유 random candidates | 1,228,800 | 2 sources×3 seeds×2,048×100 |
| Primary ledger | 7,372,800 rows | Control·validation 기록 제외 |
| Primary sampling NFE | 447,283,200 | Gaussian 100 steps, Discrete 32 steps |
| Synthetic control sampling NFE | 1,118,208 | 정상/반전 모두 포함; training·validation 제외 |

NFE는 batch API call 수나 MD5 호출 수와 같지 않다. 단위별 계산은 일치하지만, 이것으로 GPU wall-clock을 결정할 수는 없다.

### 9.2 저장 정책은 실제로 필요하다

모든 primary Gaussian raw tensor를 float32로 보관하면 그 데이터만 **100,663,296,000 bytes ≈ 100.66 GB**다. Discrete tensors, Python 객체, JSON 문자열, checkpoints는 제외한 값이다. `tolist()` 방식은 이보다 커질 수 있다.

계획의 “raw는 run별 16개만, 나머지는 candidate ledger streaming” 규칙은 과도한 설계가 아니라 workload에 필요한 최소 조치다. Ledger는 row당 평균 500–1,000 bytes라는 가정에서만 약 3.69–7.37 GB다. 이는 측정값이 아닌 저장 예산의 예시이며 실제 schema·압축·transaction 방식으로 측정해야 한다.

### 9.3 GPU 동작 확인과 본실험 비용 확정은 다르다

[GPU 실행 문서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/GPU_EXECUTION.md)에 M3 Max/MPS의 실제 optimizer update·sampling·checkpoint/RNG 복원 기록이 남아 있다. 당시 전체 tests는 108 passed, CUDA 부재로 5 skipped였다. 이 결과는 이번 검토에서 새로 실행한 test 결과가 아니며, 삭제된 pilot의 인증도 아니다.

남은 것은 최종 v2 조건에서 pipeline별 training·validation·sampling·decode·MD5·IO 처리량, peak memory, 실제 inference batch, run/study 시간 상한 및 중단·재개 정책이다. 기존 약 117.7시간 CPU 추산은 GPU 예상 시간이 아니다.

**운영 권장:** Primary test와 분리한 engineering 입력으로 작은 profile을 먼저 수행한다. 이 단계에서 새 패키지 설치나 모델 확대부터 시작할 근거는 없다. 학습 batch·precision 변경은 과학적 profile 변경 가능성을 검토하고, 단순 inference 운영 정책과 구분한다.

## 10. 우선순위별 권장 보완

P0는 확증적 본실험 착수를 막는 항목, P1은 결과 해석에 필요한 보완, P2는 후속 연구다. 아래 제안은 이번 보고서의 권고이며 기존 protocol에 자동 적용하지 않았다.

| 우선순위 | 보완 사항 | 완료를 판단할 산출물 |
|---|---|---|
| P0 | 노출 audit 완료·source-specific novelty 확정 | Source별 최종 배제 목록·근거·hash, available pool≥2,048 |
| P0 | 추론 대상·working model·CI 의미 결정 | Test 공개 전 analysis seal; 방어 불가능하면 기술적 보고로 한정 |
| P0 | v2 최소 실행 경로 구성 | 12-bit 입력, epoch 학습, train-only shuffle, validation 선택, 2,048표적·100 attempts 검사 |
| P0 | 학습된 same-path controls 검증 | 지정 15개 모델의 정상/반전 generation 및 Wilson gate 기록 |
| P0 | Streaming·resume·자원 예산 검증 | 강제 중단 후 누락·중복 0, final GPU profile, 시간·저장 상한 |
| P1 | 실패 원인의 분해 | Validity·grammar/decoder 이유·중복률·constraint accuracy 기록 |
| P1 | RNG 공유 범위 명세 | Namespace별 사용 필드·PRNG·initialization·batch/resume 규칙 표 |
| P1 | 음성 결과의 의미 고정 | 작은 효과·새 dataset·알고리즘 일반의 불가능성을 주장하지 않는 결과 문구 |
| P2 | 계산량·전처리 비교 | 새 protocol에서 random search/table·학습 상각·wall time 비교 |
| P2 | 계열·표현·큰 q 일반화 | Compute/parameter matching, 새 dataset 반복, q별 power와 예산 |

필요 작업의 효율적인 순서는 다음과 같다.

1. **자료·추론 가능성 판단:** 노출 pool이 충분하고 연구의 제한된 주장 범위를 받아들일 수 있는지 먼저 확정한다.
2. **최소 v2 경로와 작은 fixture:** 기존 primitives를 재사용해 입력·학습·checkpoint·ledger·통계를 연결하고 1,000표적 제한 같은 숨은 legacy 정책을 차단한다.
3. **작은 GPU profile과 학습 controls:** 최종 profile로 control을 수행하고 실패 시 independent development 범위에서 원인을 구분한다. 수정은 revision으로 기록한다.
4. **실행 조건 봉인:** Resource manifest, primary ownership/data, analysis/code/config를 연결한다.
5. **모든 적용 seed의 학습과 본 평가:** Validation으로 checkpoint를 선택·봉인한 뒤 한 번의 test stream을 생성한다. 실제 ledger integrity, 통계, 실패·미완료 및 비용을 함께 보고한다.

계획의 일부 pipeline이 blocked여도 다른 pipeline이 자기 선행 조건을 충족하면 실행할 수 있다. 다만 family 5를 유지하고 전체 다섯 pipeline 연구가 완료됐다고 쓰면 안 된다.

## 11. 결과별 허용되는 해석

| 관측 상태 | 허용되는 해석 | 피해야 할 해석 |
|---|---|---|
| Codec 또는 same-path control 실패 | 현재 구현/profile로 가설을 시험할 준비가 안 됨 | MD5 조건 학습이 불가능함 |
| Measurement gate 통과, 우위 비유의 | 지정 budget에서 확증 기준을 충족하지 못함 | 효과가 정확히 0임, diffusion 전체가 무용함 |
| 일부 seed만 양수·유의 | 현재 반복 우위 기준 미충족; seed 의존성 보고 | 좋은 seed만 골라 성공 선언 |
| 가정·gate·max-p/Holm·세 seed 모두 충족 | 지정 benchmark와 working model 아래 반복 우위 지지 | 최소 두 배 개선, 새 dataset/큰 q/전체 MD5로 일반화 |
| 후보 수 우위만 확인 | Candidate budget에서의 우위 | 학습·전처리를 포함한 계산량 또는 실용적 공격 우위 |

**최종 권고:** 현재 계획을 폐기할 근거는 없다. 다만 “곧바로 본실험을 실행할 완성된 설계”로 취급해서도 안 된다. 가장 먼저 노출 pool과 통계적 주장 범위를 확정하고, 보존된 primitives 위에서 최소 v2 경로·학습 control·GPU 예산을 검증해야 한다. 지금 보유한 증거로 연구 가설의 성공·실패를 판정할 수는 없다.

## 부록. 이번 검토의 검산·추적 정보

새 검산은 `.venv/bin/python`에서 Python `math`, `statistics.NormalDist`, 현재 PyTorch model constructors를 사용했다. 학습과 generation은 호출하지 않았다. 주요 계산식은 다음과 같으며, boundary assertions로 model counts·Wilson thresholds·primary NFE 일치를 확인했다.

| 항목 | 재현식 |
|---|---|
| Random baseline | `-expm1(K * log1p(-1/4096))` |
| Train digest coverage 근사 | `1536 * (1 - (1 - 1/1536)**10000)` |
| 전체 digest table 기대 draw 수 근사 | `4096 * sum(1/k for k in range(1,4097))` |
| Terminal alpha-bar | `prod(1 - (.0001 + (.02-.0001)*i/999) for i in range(1000))` |
| Wilson bounds | Two-sided 95%, `z = NormalDist().inv_cdf(.975)` |
| Positive gate 통과율 예시 | `P[Binomial(512,p) >= 475]`; log-gamma를 이용한 tail 합 |
| Parameter count | `sum(p.numel() for p in model.parameters())` |

검토 시점의 SHA-256:

```text
RESEARCH_PLAN_V2.md
f2a2779a722ad52820a35480affb08d07b594b6c7513cfd61d33ea5d3509c0ba
examples/poc-v2-protocol.json
4896d530655d140e4059b9d94f7d8ed07a2e70fcfa6cb3c26694c54915d8355f
src/diffusion_hash_inv/models.py
b4d244359c419494be40e472a64bf07140faa9d9700a8d004e54b4f287c90558
src/diffusion_hash_inv/discrete.py
dc10b042b725904be06b65d5a2088eed10c626cade936f33aab1b90b99b8b8cb
src/diffusion_hash_inv/runner.py
9faa79ee7f04fbff3560065ccf73b66740999f5f33afda4a0f01eca45a85e1ae
src/diffusion_hash_inv/study_statistics.py
6e93efe7aae9bb8bae108aa6d43d60a3bee5e13cb7eb033b5cd5b78a7fb0dbee
```

이 보고서는 [기존 검증 보고서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/RESEARCH_PLAN_V2_VALIDATION.md)와 [실행 체크리스트](/Users/choisoonwook/Experiments_local/DHI_AI_gen/POC_EXECUTION_CHECKLIST.md)를 대체하는 실행 인증서가 아니다. 새로 확인한 코드 불일치·산술과 설계 해석을 정리한 독립 검토이며, 삭제된 pilot의 결과를 근거로 삼지 않는다.
