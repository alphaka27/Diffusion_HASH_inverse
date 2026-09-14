# Diffusion Model을 이용한 Hash 역상 탐색 실험 계획

- 문서 상태: BGV 인코더·디코더 구현 완료, 모델·데이터셋·평가 파이프라인 미구현
- 작성일: 2026-09-04
- 최종 수정일: 2026-09-14
- 주 실험: E1·E2 — hash caption 조건부 이미지 diffusion
- 길이 정보 추가 실험: E1-L·E2-L — hash와 원본 byte 길이를 caption으로 제공
- 표현 방식 비교 실험: E3·E4 — digest bit 조건부 Bit Diffusion
- 최우선 해시 함수: MD5
- 비교 해시 함수: SHA-256

## 1. Research Question

### 원 질문

> Diffusion Model을 훈련시켜 Hash Algorithm의 역상을 구할 수 있는가?

### 검증 가능한 형태

> 제한된 메시지 분포 $D$, 해시 알고리즘 $a$, 후보 생성 예산 $K$가 주어졌을 때, 조건부 Diffusion Model은 학습에서 보지 않은 $y=H_a(x)$에 대해 $H_a(\hat{x})=y$인 역상 $\hat{x}$를 같은 예산의 기준선보다 높은 확률로 생성할 수 있는가?

이 연구가 검증하는 것은 모든 입력에 적용되는 일반적인 해시 역함수가 아니라, 정해진 분포와 계산 예산에서 학습한 조건부 생성기의 알고리즘별 역상 탐색 능력이다. **Hash caption으로부터 가역 인코딩 이미지를 생성하는 E1·E2를 주 실험으로 하고, digest bit에서 record bit를 생성하는 E3·E4를 표현 방식 비교 실험으로 사용한다.** 해시 알고리즘은 MD5를 최우선으로 하고 SHA-256을 후속 비교에 사용한다.

추가 질문은 **원본 메시지의 byte 길이를 hash caption에 함께 제공하면 E1·E2의 역상 탐색 성능이 개선되는가?**이다. 이를 E1-L·E2-L로 검증하며, 원본 길이가 알려진 조건에서의 결과로 별도 보고한다.

### 가설

- $H_{0,a}$: 알고리즘 $a$의 held-out target에서 모델의 `PreimageSuccess@K`는 같은 후보 예산을 쓰는 최선의 기준선보다 높지 않다.
- $H_{1,a}$: 알고리즘 $a$의 held-out target에서 모델의 `PreimageSuccess@K`는 기준선보다 높고, 이 차이가 3개 이상의 model seed에서 재현된다.

가설은 E1~E4별로 따로 검정하며, **주 실험 E1·E2 각각의 전체 128-bit MD5 결과에서 $H_{1,\mathrm{MD5}}$가 지지되는지**가 최우선 판정이다. E3·E4의 성공은 이진 표현 경로의 비교 결과로 보고하며 E1·E2의 성공을 대신하지 않는다. SHA-256은 full 256-bit에서 별도로 판정하며, 한 알고리즘의 결과를 다른 알고리즘으로 일반화하지 않는다. 축약한 digest에서만 성공하면 `해당 알고리즘의 축약 해시를 사용한 제한된 toy problem에서의 성공`으로만 기록한다.

E1-L·E2-L은 대응하는 E1·E2와의 성능 차이와, 같은 길이 정보를 받는 기준선 대비 우위를 각각 판정한다. 추가 실험의 성공은 길이를 제공하지 않는 주 실험의 성공을 대신하지 않는다.

## 2. 용어와 판정 단위

- **역상 성공**: 생성 후보 $\hat{x}$가 원본 $x$와 달라도 $H_a(\hat{x})=H_a(x)$이면 성공이다.
- **원본 복원**: $\hat{x}=x$인 더 강한 별도 성공 기준이다.
- **학습 pair 복원**: 암기 가능성을 측정하는 진단 결과이며 일반화 증거가 아니다.
- **held-out 복원**: 학습에 포함되지 않은 메시지와 조건 해시에 대한 결과이며 주 판정 대상이다.
- **후보 예산 $K$**: target 하나당 생성·검증할 후보 수이다. 모델과 모든 기준선에 같은 $K$를 적용한다.

암호학적 역상 성공의 주 지표는 해시 재계산으로만 판정한다. 이미지 유사도, 낮은 bit error rate, 사람 눈에 비슷한 결과는 역상 성공으로 세지 않는다.

### 2.1 후보 생성 예산 $K$의 상세 정의

이 문서에서 후보 생성 예산 $K$는 **학습 데이터셋을 만들기 위해 생성하는 데이터 수가 아니라**, held-out hash target 하나에 대해 모델 또는 기준선이 추론 단계에서 제안할 수 있는 역상 후보의 최대 수이다. target $y_i$에 대해 방법 하나는 서로 다른 noise seed 또는 난수 표집으로 최대 $K$개의 후보 $\hat{x}_{i,1},\ldots,\hat{x}_{i,K}$를 생성한다. E1·E2와 E1-L·E2-L에서는 각 생성 이미지를 threshold·decode한 record가, E3·E4에서는 decode한 bit record가 후보 하나이다.

- $K$의 각 시도는 유효한 message record로 decode되지 않아도 예산을 소진한다. invalid decode를 버린 뒤 추가 표집해 $K$개를 채우지 않는다.
- decode에 성공한 후보에는 원래 알고리즘의 해시 $H_a$를 재계산해 target과 비교한다. 따라서 `ValidDecodeRate`와 실제 hash 검증 횟수를 함께 기록한다.
- $K$는 train/validation/test 데이터셋 크기, optimizer update 수, diffusion sampling step 수, 또는 GPU 시간과 다른 예산이다. 이 값들은 별도로 고정·기록한다.
- $K$가 커지면 우연한 성공 확률도 커지므로, 방법 간 비교는 반드시 같은 target 집합과 같은 $K$에서만 한다. $K=1,10,100의 결과는 각각 별도 `PreimageSuccess@K`로 보고한다.

평가 target이 $N$개이면, method 하나가 model seed 하나에서 사용하는 후보 생성 시도 수는 $N\times K$이다. 예를 들어 unique test digest 10,000개를 $K=10$으로 평가하면 해당 방법은 최대 100,000개 후보를 생성한다. 이 수는 모델의 품질뿐 아니라 탐색 기회를 뜻하므로, 더 큰 $K$에서의 성공률을 더 작은 $K$의 기준선과 비교하지 않는다.

학습 데이터 구축 예산은 별도 항목이다. 본 문서의 `train 100,000` 등 데이터셋 크기는 학습 pair 수를 뜻하며 $K$에 포함하지 않는다. 생성 데이터의 품질 필터링이나 데이터셋 후보 탐색을 추가로 수행할 경우에는 그 비용을 `dataset construction budget`으로 따로 정의하고, 추론 후보 예산 $K$와 합치지 않는다.

## 3. 공통 실험 명세

### 3.1 메시지 분포

| 구분 | 정의 | 주 실험 길이 |
|---|---|---|
| Printable message $m$ | ASCII `0x21`~`0x7e`의 94개 문자에서 균등 표집(공백 제외) | 4~31 bytes에서 균등 표집 |
| Random bytes $b$ | 각 byte를 `0x00`~`0xff`에서 독립·균등 표집 | 4~31 bytes에서 균등 표집 |

`임의 길이`는 무한 길이가 아니라 사전에 정한 범위 안의 가변 길이를 뜻한다. 1~2 byte 입력은 작은 도메인을 전수 조사하는 sanity check에서 별도로 사용한다.

유효한 후보는 BGV length glyph가 4~31로 decode되고 validity mask가 header·payload와 일치하는 byte sequence이다. printable 실험에서 후보가 위의 94개 문자(공백 제외)에 속하는지는 `InDomainPreimageSuccess@K`로도 따로 기록한다. padding은 glyph 값이 아니라 validity mask로만 구분하며, payload의 `0x00`은 유효한 값이다.

### 3.2 해시와 난이도 단계

| 우선순위 | 알고리즘 $a$ | Full digest $n_a$ | 역할 |
|---|---|---:|---|
| 1 | MD5 | 128 bits | 최우선 주 실험 |
| 2 | SHA-256 | 256 bits | 후속 비교 실험 |

- 해시는 length header나 padding이 아닌 원래 payload bytes $x$에 적용한다.
- 각 알고리즘의 full digest $H_{a,n_a}(x)$가 최종 판정 조건이다.
- 난이도 곡선은 앞 $q$ bit만 쓰는 $H_{a,q}(x)$로 측정한다. $Q_a=(\{8,12,16,20,24,32,64,128\}\cap[1,n_a])\cup\{n_a\}$를 사용한다.
- $q<n_a$에서의 성공은 해당 알고리즘의 full-digest 역상 성공이 아니다.
- E1~E4와 E1-L·E2-L은 알고리즘별로 모델을 따로 학습한다. MD5 실험을 완료한 뒤 SHA-256을 반복한다.
- 해시 내부 trace는 모델 입력에 넣지 않는다.

E1·E2의 caption은 `<algorithm>-<q>:<hex digest>` 형식으로 고정하고, 손실 없이 각 문자를 구분하는 고정 character tokenizer를 사용한다. 이진 실험 E3·E4에는 같은 digest를 bit vector로 직접 입력한다. E1~E4에는 메시지 길이를 모델 조건으로 제공하지 않는다. **E1-L·E2-L에만** `<algorithm>-<q>:<hex digest>|len_bytes=<L>` 형식으로 원본 payload의 byte 길이 $L$을 추가한다. 길이의 정의와 비교 조건은 4.1절을 따른다.

MD5의 알려진 collision 약점과 fixed-target preimage 탐색은 서로 다른 문제다. 따라서 MD5 판정도 collision 생성이 아니라 목표 digest의 preimage 재검산으로 수행한다.

### 3.3 가역 이미지 인코딩 $E$

이미지 실험은 Byte Glyph Visualization (BGV)을 canonical encoding으로 사용한다. `[LEN][B00]...[B30]`의 32 slot을 4x8로 고정 배치하고, 각 byte는 MSB-first 2x4 glyph를 4x4 pixel block으로 확장한다. 따라서 glyph channel은 32x128이며, slot 전체가 1인 validity-mask channel을 함께 사용해 payload `0x00`과 padding을 구분한다. 모델 입력·출력은 `[2, 32, 128]`이고, 확산 모델이 `[-1, 1]` convention이면 BGV의 `[0, 1]` 값만 `image * 2 - 1`로 변환한다. 저장이 필요하면 lossless PNG만 사용한다.

학습 전에 printable과 bytes 각각에 대해 `decode(E(x)) == x`가 전 표본에서 성립해야 한다. 생성 이미지는 glyph block과 mask cell 평균을 0.5에서 threshold하고, length 범위와 mask 일관성을 검증해 decode한다. 따라서 raw floating-point pixel equality가 아니라 threshold 후 canonical image equality를 측정한다.

### 3.4 데이터 생성과 분할

- pilot: train 10,000 / validation 1,000 / test 1,000 unique messages, model seed `0`
- 본 실험: train 100,000 / validation 10,000 / test 10,000 unique messages
- seed: dataset seed 1개를 고정하고 model seed `0, 1, 2`로 각 실험을 반복
- 동일 메시지는 한 번만 생성한다.
- full-digest 비교에는 같은 base message와 split을 MD5와 SHA-256에서 재사용한다.
- 축약 해시의 충돌로 같은 condition이 여러 개 생기면 알고리즘·$q$별로 해당 digest의 모든 메시지를 같은 split에 넣는 group split을 적용한다.
- test split은 학습이나 hyperparameter 선택에 사용하지 않는다.
- 데이터 manifest에 hash algorithm, full digest 길이, $q$, 분포, 길이 범위, seed, split, encoder version, 실험 ID, caption 형식, length conditioning 여부를 기록한다.

평가 target은 test message 수가 아니라 unique $(a,q,\text{digest})$ condition이다. 축약 digest 하나에 원본이 여러 개면 exact-source 진단용 대표 원본 하나를 dataset seed로 고정한다.

E1-L·E2-L도 대응하는 E1·E2의 base messages, split, unique digest target, 대표 원본 $x_i$, $K=100$ subset을 그대로 재사용한다. 평가에 제공할 길이는 $L_i=|x_i|_{\mathrm{bytes}}$로 고정한다. 같은 축약 digest에 여러 원본 길이가 있더라도 digest당 이 길이 하나만 평가하며, $(\text{digest},L)$별로 target을 늘리지 않는다. Split도 길이와 무관한 digest group 단위를 유지해 같은 digest가 서로 다른 길이로 train/test에 나뉘는 것을 방지한다.

#### 후보 생성 예산 설정

| 목적 | 후보 예산 $K$ | 평가 target | 방법·model seed 하나의 후보 생성 시도 수 |
|---|---:|---|---:|
| 단일 생성 성능 | 1 | 모든 unique test digest $N_{a,q}$개 | $N_{a,q}$ |
| 실용적 반복 생성 성능 | 10 | 모든 unique test digest $N_{a,q}$개 | $10N_{a,q}$ |
| 다중 표집 효과 | 100 | dataset seed로 사전 고정한 $\min(1{,}000,N_{a,q})$개 subset | $100\min(1{,}000,N_{a,q})$ |

같은 $(a,q)$에서 한 방법·model seed의 총 후보 생성 시도 수는 $11N_{a,q}+100\min(1{,}000,N_{a,q})$이다. 예를 들어 full digest처럼 $N_{a,q}=10{,}000$이면 총 210,000회다. $K=100$ subset은 방법과 model seed 사이에서 동일하게 유지한다. 모든 비교 방법은 동일한 target과 후보 수를 사용한다.

E1-L·E2-L에도 같은 $K$와 위 예산을 각각 적용하며 추가 모델·대조군의 비용을 별도 합산한다. Full MD5에서 두 추가 모델의 $N_{a,q}$가 각각 10,000이면 model seed `0, 1, 2` 평가에 총 1,260,000회가 추가된다. 이는 추가 모델의 평가 시도 수이며 학습·대조군 비용은 별도이다.

$K=1{,}000$은 본 확정 실험에 포함하지 않는다. $K=100$에서 3개 model seed 모두 동일 예산 기준선보다 우세한 결과가 재현될 때에만, 새 held-out target을 사용하는 별도 후속 실험으로 추가한다.

### 3.5 모델 원칙

- 주 실험 E1·E2: `[2, 32, 128]` BGV tensor를 입출력하는 작은 pixel-space conditional diffusion을 사용한다. 정확 복원을 방해하는 lossy VAE/latent compression은 사용하지 않는다.
- 추가 실험 E1-L·E2-L: 대응하는 E1·E2의 모델 구조·크기·optimizer update 수·sampling 설정을 사용하고, 학습·validation·test의 caption에 길이를 추가한다. 각 model seed에서 대응 모델과 같은 초기 가중치로 시작해 별도로 학습하며, 학습된 E1·E2에 추론 시에만 길이 문자열을 붙이지 않는다.
- 비교 실험 E3·E4: payload record의 bit를 `{-1, +1}` 연속값으로 확산한 뒤 0에서 threshold하는 conditional Bit Diffusion을 우선 사용한다.
- 모델 크기, optimizer update 수, sampling step 수는 validation split에서 한 번 정한 뒤 같은 모델 계열(이미지: E1·E2·E1-L·E2-L, 이진: E3·E4)과 test 평가 동안 고정한다.
- test 결과를 보고 architecture나 threshold를 다시 고르지 않는다.

## 4. 네 가지 기본 실험과 길이 caption 추가 실험

| ID | 데이터 | 모델 입력 | 모델 출력 | 역할 |
|---|---|---|---|---|
| E1 | Printable $(E(m), C_{a,q}(m))$ | hash caption 문자열 | encoded image $E(\hat{m})$ | 주 실험: image-caption 경로의 printable 원본/역상 복원 |
| E2 | Random bytes $(E(b), C_{a,q}(b))$ | hash caption 문자열 | encoded image $E(\hat{b})$ | 주 실험: image-caption 경로의 random bytes 원본/역상 복원 |
| E3 | Printable $(m, H_{a,q}(m))$ | digest bit vector | message record bit vector | 비교 실험: E1에 대응하는 이진 표현 경로 |
| E4 | Random bytes $(b, H_{a,q}(b))$ | digest bit vector | byte record bit vector | 비교 실험: E2에 대응하는 이진 표현 경로 |
| E1-L | E1과 같은 Printable messages | hash caption + 원본 byte 길이 | encoded image $E(\hat{m})$ | 추가 실험: E1에서 길이 정보의 효과 |
| E2-L | E2와 같은 Random bytes | hash caption + 원본 byte 길이 | encoded image $E(\hat{b})$ | 추가 실험: E2에서 길이 정보의 효과 |

E1·E2가 Research Question을 검증하는 주 실험이다. E3·E4는 각각 E1·E2와 같은 메시지 분포·target·후보 예산에서 image-caption 경로를 이진 입력·출력 경로로 바꾸는 representation ablation으로 해석한다.

E1·E2만 실패하고 대응하는 E3·E4가 성공할 때에만, image 모델의 caption encoder를 digest bit-vector encoder로 바꾼 한 번의 추가 ablation으로 caption과 image 출력 중 어느 경로가 병목인지 구분한다.

각 실험은 다음 순서로 수행한다.

1. target $x$를 정해진 분포에서 생성하고 $H_{a,q}(x)$를 계산한다.
2. E1·E2는 $(E(x), C_{a,q}(x))$, E3·E4는 $(x, H_{a,q}(x))$ pair로 학습한다. E1-L·E2-L은 길이가 포함된 caption과 같은 이미지 $E(x)$로 학습한다.
3. E1~E4에는 held-out $H_{a,q}(x)$만, E1-L·E2-L에는 해당 해시와 원본 byte 길이만 모델 조건으로 주고 서로 다른 noise seed로 후보 $K$개를 생성한다.
4. E1·E2와 E1-L·E2-L은 이미지를 threshold·decode하고, E3·E4는 bit vector를 decode한다.
5. 각 후보에 같은 알고리즘 $H_a$를 적용해 목표 digest와 비교한다.
6. 학습 pair와 held-out pair 결과를 분리해 저장한다.

### 4.1 E1-L·E2-L — 원본 길이 caption의 효과

검증할 효과는 **원본 길이 정보를 제공했을 때 역상 탐색과 길이 복원이 얼마나 개선되는가**이다. E1-L은 E1과, E2-L은 E2와 비교한다.

- $L=|x|_{\mathrm{bytes}}$는 해시를 계산한 원래 payload의 byte 수이다. Length header와 zero padding은 제외하며, 문자 수나 hex 문자열 길이를 사용하지 않는다.
- Caption은 `<algorithm>-<q>:<hex digest>|len_bytes=<L>`로 고정한다. $L$은 앞자리 0이 없는 십진수로 표기한다. 예를 들어 4-byte payload의 full MD5 caption은 `md5-128:<32자리 hex digest>|len_bytes=4`이다.
- 실제 원본 길이를 학습·validation·test에서 모두 제공한다. 별도의 길이 추정기는 사용하지 않으며, 원본 bytes나 해시 내부 trace는 제공하지 않는다.
- 대응 모델 쌍은 길이 필드에 필요한 문자까지 포함한 같은 고정 character vocabulary, caption 최대 길이, padding·mask 규칙, encoder 구조를 사용한다. E1·E2의 caption에는 길이 값을 넣지 않는다. 모델·학습 설정은 대응 쌍에 공통으로 고정하고, 같은 seed의 데이터 순서와 평가 noise seed 목록을 맞춘다.
- 출력은 length glyph와 validity mask를 포함한 `[2, 32, 128]` BGV 이미지이며, threshold와 decoder도 동일하게 사용한다. 제공된 $L$로 length glyph를 덮어쓰거나 payload를 잘라내거나 mask를 강제하지 않는다.
- 길이가 맞지 않는 후보도 한 번의 생성 시도로 계산한다. 유효하게 decode되고 목표 해시가 같으면 공통 `PreimageSuccess@K`에는 성공으로 세되, 제공 길이까지 일치해야 하는 `LengthMatchedPreimageSuccess@K`에는 실패로 센다. 길이 불일치·invalid decode를 이유로 추가 표집하지 않는다.

길이 정보의 효과는 같은 target·대표 원본·$K$에서 `E1-L − E1`, `E2-L − E2`의 `PreimageSuccess@K` 차이로 보고한다. 길이 일치 지표도 양쪽 모델의 후보에 같은 $L_i$로 계산한다. 다만 hash-only 모델의 검증에 쓰는 $L_i$는 모델 입력으로 전달하지 않는다.

## 5. 대조군

최소한 다음 대조군을 각 실험과 같은 target·후보 예산으로 실행한다.

- **Source-prior random search**: 해당 메시지 분포에서 후보를 무작위 표집하고 해시를 확인한다.
- **Nearest training digest**: Hamming distance가 가장 가까운 train digest의 메시지를 후보로 사용한다.
- **Direct conditional predictor**: 같은 digest condition에서 record bit를 바로 예측하는 비-diffusion 모델이다. 확률적 bit 출력에서 같은 $K$개를 표집한다.
- **Unconditional/zero-condition**: hash condition을 제거한 동일 생성 모델이다.
- **Shuffled-condition negative control**: train의 hash-message 대응을 무작위로 섞는다.
- **Reversible-condition positive control**: hash 대신 원본 record 자체 또는 가역 변환을 조건으로 주어 pipeline이 정확 복원을 학습할 수 있는지 확인한다.
- **Exhaustive search**: 1~2 byte의 작은 도메인에서 정답과 기준선 구현을 검증한다.

E1·E2와 E1-L·E2-L은 모델 학습 전에 `message -> E -> decode -> message` 왕복 검사를 추가한다.

E1-L·E2-L의 기준선도 원본 길이 $L_i$를 제공받는다. 다음 비교로 단순한 길이 범위 축소와 hash condition 학습의 효과를 구분한다.

| 길이 정보 추가 실험의 대조군 | 조건과 동작 |
|---|---|
| Length-aware random search | 해당 문자·byte 분포에서 길이가 정확히 $L_i$인 후보를 $K$번 표집 |
| Length-aware direct predictor | 같은 $(\text{digest},L_i)$ 정보를 받아 record bit 후보를 $K$개 표집 |
| Length-aware nearest training digest | 길이가 $L_i$인 train 메시지 안에서 digest 거리로 검색; 검색·누수 진단으로 사용 |
| Length-only control | 동일 모델에서 hash 정보만 제거하고 실제 길이는 유지; 길이만으로 설명되는 성능 측정 |
| Shuffled-hash control | 같은 길이의 train 메시지 사이에서 hash 대응만 섞고 실제 길이는 유지 |
| Reversible-condition positive control | 같은 caption 처리 경로에 가역 조건과 실제 길이를 제공해 정확 복원 확인 |

원래의 길이 없는 random search·unconditional 결과도 참고로 보고하되, 추가 실험의 경쟁 기준선은 length-aware random search, length-aware direct predictor, length-only control로 한다. Nearest training digest와 shuffled-hash는 진단에, reversible-condition과 exhaustive search는 pipeline 검증에 사용하며 경쟁 기준선에 포함하지 않는다. 작은 도메인의 exhaustive 검증에도 같은 $L_i$의 후보 집합을 사용한다.

## 6. 평가 지표

주 지표는 unique test digest target $N$개에 대한 다음 값이다.

$$
\mathrm{PreimageSuccess@K}
=\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}\left[\exists j\le K:\ \operatorname{Valid}(\hat{x}_{i,j})\ \land\ H_{a,q}(\hat{x}_{i,j})=y_i\right]
$$

함께 기록할 지표는 다음과 같다.

- `ExactSourceRecovery@K`: 후보 중 $\hat{x}=x_i$가 하나 이상 존재하는 target 비율. 축약 digest에서는 사전에 고정한 대표 원본 $x_i$에 대한 진단이다.
- `InDomainPreimageSuccess@K`: 성공한 역상이 해당 실험의 길이·문자 분포 범위에도 속하는 target 비율
- `ExactCanonicalImage@K`: E1·E2와 E1-L·E2-L에서 threshold한 이미지가 $E(x)$와 같은 target 비율
- `ValidDecodeRate`: 생성 결과가 정의한 record로 decode되는 비율
- `BitErrorRate`: 원본 record와 생성 record 사이의 bit error; 진단 전용
- train/test gap: 암기와 일반화 구분
- 메시지 길이별 성공률
- 후보 수, 실제 hash 검증 횟수, sampling step, wall-clock time
- model seed별 결과와 95% binomial confidence interval
- 알고리즘별 결과와 같은 $q$에서 N·길이 분포·$K$·model budget을 맞춘 비교

축약 해시에서는 다른 원본이 같은 digest를 가질 수 있으므로 `PreimageSuccess@K`와 `ExactSourceRecovery@K`를 반드시 분리한다. confidence interval의 $N$도 message 수가 아닌 unique test digest 수이다. 성공이 0건이어도 평가 횟수에 따른 95% 성공률 상한(근사값 $3/N$)을 함께 보고한다.

길이 caption 비교에서는 다음을 추가로 기록한다. Target 단위 성공률 지표의 confidence interval에는 기존과 같은 unique digest target 수 $N$을 사용한다. 후보 단위인 `LengthMatchRate`는 전체 생성 시도 수를 분모로 집계하고 model seed별로 보고한다.

- `LengthMatchRate`: 전체 후보 생성 시도 중 유효하게 decode되고 $|\hat{x}|_{\mathrm{bytes}}=L_i$인 비율. Invalid decode는 불일치로 계산한다.
- `LengthMatchedPreimageSuccess@K`: 후보 $K$개 안에 유효한 $\hat{x}$ 중 $H_{a,q}(\hat{x})=y_i$와 $|\hat{x}|_{\mathrm{bytes}}=L_i$를 동시에 만족하는 것이 있는 target 비율. Printable 문자 범위 충족 여부는 `InDomainPreimageSuccess@K`로 별도 확인한다.
- 대응 E1·E2 대비 `PreimageSuccess@K` 차이, 길이별 차이, model seed별 결과와 target별 성공·실패 대응표
- 길이 정보를 제공한 기준선 대비 우위, caption 처리 및 sampling을 포함한 실제 실행 시간

## 7. 성공 기준과 해석

주장을 하기 위한 조건은 모두 다음과 같다.

1. encoder round trip이 100% 성공한다.
2. positive control이 held-out target에서 사전 설정한 정확 복원 기준(기본 99%)을 넘는다.
3. 실제 hash condition 모델의 held-out `PreimageSuccess@K` 95% confidence interval 하한이 최선의 동일 예산 기준선 상한보다 높다.
4. 향상이 model seed 3개에서 같은 방향으로 재현되고 95% confidence interval과 함께 보고된다.
5. train pair만 잘 복원한 결과는 성공으로 해석하지 않는다.

E1-L·E2-L에도 위 검증·재현성 조건을 적용하되, 기준선 우위는 5절에 명시한 길이 정보를 제공한 경쟁 기준선 중 최선의 결과와 비교한다. 길이 제공의 효과는 대응 E1·E2 대비 향상이 model seed 3개에서 같은 방향으로 재현되는지와 양쪽의 95% confidence interval을 함께 보고한다. 길이만으로 설명되는 성능과 해시 조건의 추가 기여는 length-only control로 구분한다.

| 관측 결과 | 해석 |
|---|---|
| train 성공, held-out 실패 | pair 암기이며 역상 일반화 증거 없음 |
| 작은 $q$에서만 기준선 초과 | 해당 알고리즘의 축약 해시 toy problem에서만 제한적으로 성공 |
| E1·E2 실패, E3·E4 성공 | 주 실험의 성공 증거 없음; 이진 경로의 성공은 이미지/caption 표현 병목을 진단하는 비교 결과 |
| E1-L·E2-L만 길이 정보를 제공한 기준선 초과 | 원본 길이가 알려진 조건에서의 역상 탐색 증거; 길이가 없는 E1·E2의 성공 증거는 아님 |
| E1-L·E2-L의 길이 일치율만 개선 | 길이 복원은 개선되었으나 역상 탐색 개선의 증거는 없음 |
| E1-L·E2-L이 E1·E2보다 높지만 length-only 또는 length-aware random과 비슷함 | 길이 정보가 주는 탐색 범위 축소로 설명될 수 있으며 hash condition 학습의 우위는 확인되지 않음 |
| Printable만 성공 | 제한된 source 분포의 구조를 활용한 결과 |
| 역상 성공, 원본 복원 실패 | 충돌에 의한 다른 유효 역상을 생성했을 가능성 |
| E1 또는 E2가 full MD5에서 기준선 초과 | 해당 주 실험의 분포·모델·예산에서 최우선 RQ에 대한 긍정적 증거; 일반적인 MD5 역함수나 보안성 붕괴를 뜻하지 않음 |
| E1 또는 E2가 full SHA-256에서 기준선 초과 | 해당 주 실험의 분포·모델·예산 안에서 SHA-256 역상 탐색 증거 |
| 특정 full digest에서 0건 | 해당 알고리즘의 실험 예산에서 성공 증거 없음; 불가능성의 증명은 아님 |

## 8. 실행 단계와 중단 기준

각 단계에서는 **E1·E2와 대응 대조군을 먼저, E1-L·E2-L과 길이 정보를 제공한 대조군을 다음으로, E3·E4와 대응 대조군을 그 뒤에 실행한다.** Positive control 통과 여부와 중간 $q$의 확장·중단은 실험별로 판정하며, 추가·비교 실험의 성공이나 실패로 E1·E2의 주 실험 실행 여부를 결정하지 않는다. E1-L·E2-L의 중간 $q$ 확장은 자신의 길이 정보를 제공한 기준선 대비 우위로 판정한다. 대응 E1·E2가 그 $q$에서 중단되어 결과가 없으면 같은 예산의 비교용 실행을 추가한다.

### Phase 0 — Pipeline 검증

- **MD5부터** 1~2 byte 도메인과 $q=8$에서 encoder round trip, split, candidate 검증기, exhaustive/random baseline을 확인한다.
- 256개 pair를 과적합해 train exact recall 99% 이상이 가능한지 확인한다.
- MD5 재계산 결과가 Python `hashlib.md5`와 프로젝트 구현에서 일치하는지 확인한다.
- 길이 caption의 byte 수·문자 보존, 기존 split·target 재사용, 길이 일치 지표와 length-aware random baseline을 검증한다. 길이가 다른 유효 역상은 공통 역상 성공과 길이 일치 성공에서 서로 다르게 판정되는지도 확인한다.

### Phase 1 — Positive/negative control

- MD5 E1~E4의 reversible-condition, shuffled-condition, zero-condition을 1개 seed로 실행한다.
- E1-L·E2-L의 reversible-condition, shuffled-hash, length-only control을 seed `0`으로 실행한다.
- positive control이 99%에 미달하면 본 실험 대신 표현·모델 pipeline을 먼저 수정한다.

### Phase 2 — MD5 난이도 pilot

- $q=8,12,16$, MD5 E1~E4를 model seed `0`으로 실행하고, 기준선 우위가 있는 설정만 seed `1, 2`로 재현한다.
- E1-L·E2-L도 같은 $q$와 target에서 seed `0`으로 실행하고, 길이 정보를 제공한 기준선 우위가 있는 설정을 seed `1, 2`로 재현한다. 길이 제공 효과 비교에 필요한 대응 E1·E2의 동일 seed 결과가 없으면 비교용 실행을 추가한다.
- 가장 쉬운 설정에서도 기준선 우위가 없으면 MD5 중간 $q$ 확장을 중단하고 음성 결과를 기록한다.

### Phase 3 — MD5 bit scaling과 full-digest 판정

- pilot에서 3개 seed의 학습 신호가 확인된 경우에만 본 실험 데이터로 $q=20,24,32,64$ 순서로 확장한다.
- 두 단계 연속 기준선 우위가 사라지면 남은 중간 단계를 생략한다.
- 최우선 판정을 위해 full MD5 $q=128$은 pilot 신호 유무와 관계없이 고정 예산과 model seed `0, 1, 2`로 E1·E2를 먼저 실행하고, E3·E4도 같은 조건으로 비교 평가한다. 각 실험은 Phase 1의 positive control을 통과해야 한다.
- E1-L·E2-L도 positive control 통과 후 pilot 신호와 관계없이 full MD5 $q=128$을 model seed `0, 1, 2`로 실행한다. 중간 $q$는 위 확장·중단 규칙을 적용하며, 모든 평가에서 대응 E1·E2와 같은 target·$K$를 사용한다.

### Phase 4 — SHA-256 비교 실험

- 같은 base message pool, 길이 분포, $K$, 모델·학습 예산으로 E1~E4, E1-L·E2-L과 각각의 control을 반복한다.
- $q=8,12,16$ pilot과 full SHA-256 $q=256$의 model seed `0, 1, 2` 평가는 실행한다.
- pilot에서 신호가 확인된 경우에만 $q=20,24,32,64,128$을 순서대로 추가한다.
- MD5와 SHA-256은 같은 $q$끼리만 비교한다. full 결과끼리는 digest 길이가 다르므로 알고리즘 구조 차이로 해석하지 않는다.
- E1-L·E2-L도 positive control 통과 후 $q=8,12,16$ pilot과 full $q=256$의 3개 seed 평가를 수행하며, 길이 제공 효과와 길이 정보를 제공한 기준선 대비 우위를 별도로 보고한다.

### Phase 5 — 최종 보고

- 알고리즘·실험별 `PreimageSuccess@K` curve, exact recovery, train/test gap, 계산 예산과 confidence interval을 함께 보고한다.
- **MD5 결과를 먼저 제시하고 SHA-256 비교 결과를 뒤에 제시한다. 각 알고리즘 안에서는 E1·E2의 주 실험, E1-L·E2-L의 길이 caption 추가 실험, E3·E4의 표현 방식 비교 순서로 제시한다.**
- 각 알고리즘 안에서 네 기본 실험으로 source 구조(printable vs bytes)와 표현 방식(image-caption vs bits)을 비교하고, E1-L 대 E1·E2-L 대 E2로 원본 길이 제공의 효과를 비교한다.

## 9. 재현성 산출물

실행 단계에서 다음을 저장한다.

- `data/`: 생성 데이터와 split manifest (`.gitignore` 대상)
- `output/`: config snapshot, checkpoint, 후보, metric JSON/CSV, figure (`.gitignore` 대상)
- Git commit, Python/package version, device, dataset/model seed
- 실패한 후보도 포함한 target별 판정 결과와 해당 알고리즘의 실제 digest 재검산 값
- 길이 caption 비교의 target별 대표 원본 ID, 제공 byte 길이, 실제 caption, 생성 후보 길이, 길이 일치 여부와 대응 모델의 성공·실패

현재 저장소에서 재사용 가능한 기반은 `generate_message`, `generate_bytes`, MD5·SHA-256을 지원하는 `trace_hash`와 이들의 회귀 test이다. 대량 digest 계산은 `hashlib.md5`·`hashlib.sha256`을 쓰고, 표본을 `trace_hash`와 교차 검증한다. 이미지 encoder/decoder, diffusion model, dataset builder와 평가기는 아직 구현되어 있지 않다.

## 10. 타당성 위협

- train/test에 같은 메시지나 같은 축약 digest가 섞이면 암기를 일반화로 오판할 수 있다.
- caption tokenizer가 hex 문자를 손실하면 해시 역상보다 conditioning 실패를 측정하게 된다.
- 원본 길이는 추가 정보이므로 E1-L·E2-L의 성공을 해시만 제공한 E1·E2의 성공으로 해석할 수 없다. 대조군에도 같은 길이를 제공해야 한다.
- 길이 정보를 추가하면서 split을 $(\text{digest},L)$ 단위로 다시 나누거나 digest당 여러 길이를 별도 평가하면 누수 또는 target·후보 예산 불일치가 생길 수 있다. 기존 digest group split과 대표 원본을 유지한다.
- 생성 header를 실제 길이로 교정하면 caption conditioning과 후처리의 효과가 섞인다. Decoder를 공유하고 길이 강제 보정을 하지 않는다.
- lossy image codec이나 VAE를 쓰면 exact recovery가 모델 이전 단계에서 불가능할 수 있다.
- Printable과 random bytes는 같은 byte 길이에서도 source entropy가 다르므로 결과를 직접 동일 난이도로 간주할 수 없다.
- 후보 수를 늘리면 우연 성공도 늘어나므로 $K$와 총 hash 검증 횟수를 고정해야 한다.
- 작은 도메인 또는 축약 digest의 결과를 해당 알고리즘의 full digest나 임의 메시지로 일반화할 수 없다.
- MD5의 collision 취약성을 fixed-target preimage 성공의 근거로 오해하면 안 된다.
- 같은 $q$끼리만 알고리즘을 비교하며, 길이가 다른 full digest 결과 차이를 구조 효과로 단정하지 않는다.
- 여러 설정 중 좋은 결과만 선택하지 않도록 주 설정, 중단 기준, seed와 지표를 test 실행 전에 고정해야 한다.

## 11. 참고 근거

- [NIST: Preimage resistance](https://csrc.nist.gov/glossary/term/Preimage_resistance)
- [RFC 1321: The MD5 Message-Digest Algorithm](https://www.rfc-editor.org/info/rfc1321/)
- [RFC 6151: Updated Security Considerations for MD5](https://www.rfc-editor.org/info/rfc6151/)
- [NIST FIPS 180-4: Secure Hash Standard](https://csrc.nist.gov/pubs/fips/180-4/upd1/final)
- [Ho, Jain, Abbeel: Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Chen, Zhang, Hinton: Analog Bits](https://research.google/pubs/analog-bits-generating-discrete-data-using-diffusion-models-with-self-conditioning/)
- [Austin et al.: Structured Denoising Diffusion Models in Discrete State-Spaces](https://proceedings.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html)
