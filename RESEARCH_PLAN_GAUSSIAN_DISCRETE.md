# Gaussian / Discrete Diffusion을 이용한 제한된 분포의 Hash Preimage Candidate 생성 연구 계획

## 1. Document Status

- 문서 상태: **새 실험을 위한 최종 연구 계획이며, 실행 configuration은 preregistration 전이다.** `TBD`는 실행 전에 확정해야 하는 값이지 구현 기본값을 자동으로 승인한다는 뜻이 아니다.
- 작성 기준: 기존 [RESEARCH_PLAN.md](RESEARCH_PLAN.md) 전체의 연구 목표, 인코딩 의미, 검증·통계 원칙을 유지하고 core experiment를 Source × Diffusion Family 중심으로 재구성한다. 원본은 변경하지 않는다.
- 실행 순서: MD5를 먼저 완료하고 SHA-256에 같은 protocol을 반복한다. 본 문서 작성 작업에서는 학습·실험을 실행하지 않는다.
- 현재 저장소에는 BGV/CGGE/Direct Bits codec, Gaussian diffusion 및 image U-Net, direct predictor, dataset/split, verifier/evaluator, 통계 validator 코드가 존재한다. 그러나 이것이 새 계획 전체의 구현 완료나 검증 통과를 의미하지 않는다.
- 현재 BGV/CGGE 설정은 기본 최대 길이 31과 grid capacity의 정확한 일치를 전제로 한다. 임의의 최대 길이에 대한 남는 physical cell 처리, Masked Discrete Diffusion/tokenizer, 새 10개 family, 공통 digest condition, 확장된 validity·manifest·통계 family의 구현과 검증은 별도 작업이다.
- 저장소에는 후속 소규모 gate 실험 기록도 있다. [EXPERIMENT_DECISIONS.md](EXPERIMENT_DECISIONS.md)의 G2–G6 단계 이름과 본 문서의 **G0–G4 scientific validation gates**는 다른 체계다. 기존 결과나 중단 상태를 새 계획의 통과 결과로 옮기지 않는다. 이미 관측한 test는 새 confirmatory test로 재사용하지 않는다.

기존 계획과의 주요 차이는 Direct Bits core diffusion을 categorical Masked Discrete Diffusion으로 대체하고, image 실험의 hash caption을 primary comparison에서 canonical digest bits로 대체한 것이다. BGV와 CGGE의 의미는 유지하면서 image shape을 최대 길이에 대해 일반화한다. Gaussian caption 실험은 필요할 때만 별도 auxiliary replication으로 등록한다.

## 2. Research Question

message의 raw byte sequence를 $x$, 사전 고정한 source distribution을 $D$, hash algorithm을 $a$, 해당 full-digest hash 함수를 $H_a$, full digest의 bit 수를 $n_a$로 정의한다. $y=H_a(x)$는 full digest이며, $q$는 model에 제공하는 digest prefix의 bit 수다. $H_{a,q}(x)$는 $H_a(x)$의 앞 $q$ bit이고 $y_q=H_{a,q}(x)$다. $K$는 target 하나에 허용하는 candidate 생성 attempt 수다.

> 제한된 $D,a,q,K$에서 조건부 Diffusion Model이 held-out hash target에 대해 실제로 검증 가능한 preimage candidate를 생성하고, 동일한 candidate budget을 사용하는 모든 preregistered primary competing baseline보다 높은 PreimageSuccess@K를 보이는가?

일반적인 hash inverse 계산 가능성을 주장하지 않는다. 모든 결론은 tested source distribution, message-length 범위, $q$, $K$, condition type, representation, model configuration에만 적용한다.

### Primary Question — Baseline superiority

각 core model family가 동일 target과 $K$에서 모든 적용 가능한 primary baseline보다 높은 held-out PreimageSuccess@K를 보이며 model seed 0, 1, 2에서 같은 방향으로 재현되는지 검정한다. 모델 간 순위는 이 질문을 대체하지 않는다.

### Secondary Question A — Diffusion Family

동일 source, target, $a,q,K$에서 Gaussian image-space diffusion과 Discrete sequence-space diffusion을 비교한다. 정확한 해석은 **end-to-end Gaussian-image versus Discrete-sequence approach comparison**이다. representation과 model architecture도 동시에 달라지므로 순수 diffusion-noise-family ablation, 순수 architecture 효과 또는 순수 representation 효과라고 해석하지 않는다.

### Secondary Question B — Gaussian Representation

Printable Gaussian 실험에서 BGV와 CGGE를 비교한다. 동일 Gaussian framework, digest information, training policy 안에서의 representation ablation이다. image 크기에 따른 capacity·연산량 차이는 별도 보고한다.

### Secondary Question C — Length Information

각 model에서 hash-only와 known-length의 차이를 분석한다. $	heta$를 학습된 model parameter, $L=|x|$를 payload byte length라 하면 두 문제는 각각 다음과 같다.

$$
p_\theta(x\mid H_{a,q}(x)),\qquad
p_\theta(x\mid H_{a,q}(x),L).
$$

known-length 결과는 추가 정보가 있는 문제의 결과이며 hash-only 성공을 대신하지 않는다.

## 3. Definitions and Candidate Budget

| 용어 | 정의 |
|---|---|
| Candidate attempt | model 또는 baseline이 최종 출력 하나를 제안하는 한 번의 시도. 내부 denoising step은 별도의 candidate가 아니다. |
| Valid decode | representation의 format, source alphabet, 길이 범위 검사를 모두 통과해 하나의 raw byte sequence를 얻은 상태. hash 일치를 뜻하지 않는다. |
| Preimage success | valid candidate의 독립 full-digest rehash에서 해당 $q$-bit target이 일치한 상태. 원본과 다른 message도 가능하다. |
| Exact source recovery | collision group에서 고정한 representative source와 raw bytes가 정확히 같은 상태. preimage success와 별도 진단이다. |
| Canonical image | deterministic encoder가 message에서 생성하는 표준 BGV/CGGE tensor. |
| Full-digest experiment | $q=n_a$이며 전체 digest equality를 요구하는 실험. |
| Truncated-digest experiment | $q<n_a$이며 재계산한 full digest의 prefix만 비교하는 실험. |

예산은 $K\in\{1,10,100\}$이다. 평가 target $i$의 $j$번째 출력에서 얻은 candidate를 $\hat x_{i,j}$라 한다. invalid 출력도 attempt 하나이며 $K$를 1회 소비한다. 중복 candidate도 각각 소비한다. valid candidate가 나올 때까지 resampling하거나, invalid·duplicate를 제외하고 budget을 세거나, 성공 직후 남은 attempt를 생략하지 않는다. 생성 중 hash를 조회해 candidate를 수정·선별하는 search도 primary sampler에는 포함하지 않는다.

가능하면 하나의 고정 sampling stream에서 ordered prefix $C_K=(\hat x_{i,1},\ldots,\hat x_{i,K})$를 사용한다. 집합 표기 $C_1\subset C_{10}\subset C_{100}$은 중복을 제거한 집합이 아니라 **같은 stream의 prefix 포함 관계**를 뜻한다. 따라서 같은 target set에서는 다음 단조성이 성립해야 한다.

$$
\mathrm{PreimageSuccess@1}\le\mathrm{PreimageSuccess@10}
\le\mathrm{PreimageSuccess@100}.
$$

기존 resource policy를 유지하여 $K=1,10$은 전체 unique test target에, $K=100$은 사전 고정한 $\min(1{,}000,N_{a,q})$ target subset에 적용한다. $N_{a,q}$는 해당 source·dataset tier의 $(a,q)$ unique test digest condition 수다. subset ID와 selection seed는 test 전에 고정하며 모든 method/seed가 공유한다. 동일 subset에서 $K=1,10$을 stream prefix로 다시 계산하여 **matched-target K curve**를 제공한다. 전체 target의 $K=10$과 subset의 $K=100$을 같은 곡선의 paired 값처럼 비교하지 않는다.

각 target·method·model seed에서 $K_{\mathrm{actual}}=K_{\mathrm{declared}}$여야 한다. 실패한 sampler가 출력한 malformed record는 invalid attempt로 남긴다. 실행 중단으로 attempt 수를 채우지 못한 run은 incomplete이며 G2를 통과하지 못한다. 재개는 저장된 attempt와 RNG state를 이어 쓰며 실패한 candidate를 유리하게 교체하지 않는다.

## 4. Source Distributions and Maximum Length

### 4.1 공통 길이 분포

$L_{\max}$를 사전 고정할 최대 payload byte length로 정의한다. main source distribution은 다음을 따른다.

$$
4\le L\le L_{\max},\qquad
L\sim U\{4,\ldots,L_{\max}\}.
$$

$U$는 유한 정수 집합 위의 균등분포다. 길이를 먼저 표집하고 해당 길이의 각 character/byte를 독립 균등 표집한다. 이는 서로 다른 길이의 모든 가능한 message를 통틀어 균등하게 뽑는 분포와 다르다.

**새 비교 실험의 실행값 $L_{\max}$는 TBD다. $L_{\max}$는 실험 실행 전에 resource budget과 representation size를 고려하여 preregistration 단계에서 확정한다.** 기존 문서와 codec의 고정값 31은 근거가 있는 legacy reference이며, 새 값을 임의로 정하지 않는다. Phase 0에서는 기존 31을 유지할지 다른 지원 가능한 값으로 바꿀지 명시적으로 확정한다. 아래의 31 대입은 기존 shape과의 호환성을 설명하는 예시다.

두 source는 동일 $L_{\max}$를 사용하는 것을 기본으로 한다. 자원 제약으로 달리 정하면 source 간 차이에 length confound가 있음을 명시하고, 공통 범위로 맞춘 별도 preregistered 비교 없이는 source 자체의 효과를 주장하지 않는다. 같은 source의 Gaussian/Discrete 및 모든 baseline은 반드시 같은 길이 범위를 사용한다.

BGV의 length header가 1 byte이므로 **기존 encoding semantics를 유지하는 공통 실험에서 지원 가능한 범위는 $4\le L_{\max}\le255$**다. image shape 수식은 어떤 양의 정수 최대 길이에도 계산되지만, 255 초과 길이를 기존 1-byte header로 표현할 수는 없다. 255 초과 지원에는 별도 version의 header 설계가 필요하며 본 계획의 core BGV를 조용히 변경하지 않는다.

### 4.2 Printable ASCII

- alphabet: ASCII `0x21–0x7E`, 공백 제외, 정확히 94 characters.
- 각 character는 독립적으로 확률 $1/94$로 생성한다. character length와 byte length가 같다.
- Unicode 변환, locale, 자연어 빈도, dictionary prior를 사용하지 않는다.

### 4.3 Random Bytes

- alphabet: byte `0x00–0xFF`, 정확히 256 values.
- 각 byte는 독립적으로 확률 $1/256$로 생성한다. `0x00`도 정상 payload다.
- 텍스트로 변환하지 않고 raw bytes를 해시한다. hex 문자열은 보관·표시용이다.

source alphabet을 $\mathcal A$, 크기를 $A=|\mathcal A|$라 하면 두 source의 message probability는 다음과 같다.

$$
D(x)=\frac{1}{L_{\max}-3}A^{-|x|}
\quad\text{if }4\le|x|\le L_{\max},\ x\in\mathcal A^{|x|},
$$

그 밖의 message에는 확률 0을 부여한다. 1–2 byte는 codec unit check 및 exhaustive sanity domain에만 사용하며 main training/test source가 아니다.

## 5. Hash Algorithms and Digest Difficulty

| Algorithm $a$ | Full width $n_a$ | 평가할 $q$ | Full-digest setting |
|---|---:|---|---|
| MD5 | 128 | 8, 12, 16, 20, 24, 32, 64, 128 | $q=128$만 해당 |
| SHA-256 | 256 | 8, 12, 16, 20, 24, 32, 64, 128, 256 | $q=256$만 해당 |

digest prefix는 표준 full-digest byte 순서에서 첫 byte의 MSB부터 읽은 정확히 $q$ bit다. 12/20 bit처럼 byte 경계에 맞지 않는 prefix도 남은 bit를 올바르게 절단해야 한다. hex 표시에 쓰는 padding은 condition information이 아니다.

$$
q<n_a\Rightarrow\text{truncated-digest experiment},\qquad
q=n_a\Rightarrow\text{full-digest experiment}.
$$

MD5 전체 protocol을 먼저 수행한 뒤 SHA-256을 replication한다. MD5의 $q=128$과 SHA-256의 $q=256$은 low-q pilot 성공 여부와 관계없이 사전 확보한 fixed budget으로 평가한다. SHA-256의 $q=128$은 truncated experiment다. MD5 collision을 찾는 능력은 본 연구의 held-out target preimage 생성 능력과 동일하지 않다.

## 6. Dataset / Split

### 6.1 규모와 generation manifest

| Tier | Train unique messages | Validation unique messages | Test unique messages |
|---|---:|---:|---:|
| Pilot | 10,000 | 1,000 | 1,000 |
| Main | 100,000 | 10,000 | 10,000 |

위 규모는 각 source·$(a,q)$ dataset의 **message quota**이며 statistical target 수가 아니다. 두 source에서 같은 규칙과 규모를 사용한다. dataset generation, duplicate 제거, split assignment, representative selection, subset selection에 쓰는 seed와 알고리즘을 Phase 0에서 고정한다. 기존 작은 실험의 seed 값을 새 dataset seed로 자동 채택하지 않는다.

### 6.2 Raw-message 및 digest-group independence

split $s$의 raw message 집합을 $X_s$, digest condition 집합을 $Y_s^{a,q}=\{H_{a,q}(x):x\in X_s\}$로 정의한다. 서로 다른 train/validation/test split $s,t$의 모든 pair에서 다음을 만족해야 한다.

$$
X_s\cap X_t=\varnothing,\qquad
Y_s^{a,q}\cap Y_t^{a,q}=\varnothing.
$$

각 $(a,q)$마다 digest group 전체에 하나의 split을 할당한다. 같은 digest condition을 가진 message가 다른 split에 존재하면 해당 run은 invalid다. known-length에서도 key를 $(a,q,\mathrm{digest},L)$로 완화하지 않고 **$(a,q,\mathrm{digest})$ 단위**로 분리한다.

low-q에서는 group 크기 때문에 단순 group 배분만으로 정확한 message quota를 맞추기 어렵다. 실행 전 고정한 deterministic 절차로 충분한 reserve source pool을 만들고 group에 split을 할당한 뒤, 각 split에 귀속된 message 안에서만 quota를 선택한다. surplus는 미사용으로 남기며 다른 split으로 이동하지 않는다. 부족하면 사전 고정한 construction budget 내에서 같은 source law로 pool을 보충한다. budget 내에 quota를 채우지 못하면 dataset construction failure로 기록하며, split 경계를 깨거나 숫자를 조용히 줄이지 않는다. seed를 바꿔 유리한 test를 고르는 행위도 금지한다.

원천 sampling law $D$와 duplicate 제거·digest-group holdout·quota 선택 후 empirical distribution을 구분한다. manifest에는 raw draw 수, duplicate/unused 수, group 수, 길이별 실현 빈도, split별 실제 message 수를 남긴다. primary baseline은 여전히 원래 preregistered $D$에서 표집한다. test split에 맞춰 baseline의 digest/length prior를 유리하거나 불리하게 바꾸지 않는다.

### 6.3 Evaluation unit과 target 공유

평가 unit은 unique $(a,q,\mathrm{digest})$ condition이다. collision group의 representative source는 dataset seed에 의해 deterministic하게 사전 고정하고 ID·bytes·length를 manifest에 저장한다. representative를 여러 개 사용해 표본 수를 늘리지 않는다. known-length에는 이 representative의 길이를 제공한다.

같은 source·tier·$(a,q)$에서 Gaussian BGV, 적용 가능한 CGGE, Discrete, baseline, hash-only/known-length counterpart는 동일 split과 representative target을 사용한다. 모델마다 어려운 target을 제외하지 않는다. $N_{a,q}\le2^q$이고 일부 group은 train/validation 소속이므로 실제 test target 수는 더 작다. $q=8$의 main test message가 10,000개여도 statistical sample size가 10,000인 것은 아니다.

MD5/SHA-256 및 서로 다른 $q$는 base source pool을 재사용할 수 있지만 split manifest는 $(a,q)$별로 독립적이다. cross-algorithm 또는 cross-q target-level pairing을 자동 가정하지 않는다. 서로 다른 setting 사이의 checkpoint transfer는 primary protocol에서 사용하지 않으며 test source/condition이 다른 학습 경로로 유입되지 않도록 provenance를 기록한다.

pilot test를 보고 main configuration을 바꾸지 않는다. 기존 관측 test를 제외한 새 target을 확보하고, 모든 confirmatory configuration·선택 절차를 첫 hash test 결과를 열기 전에 freeze한다. 이후 새 연구로 변경하려면 새 preregistration과 오염되지 않은 holdout을 사용한다.

## 7. Gaussian Diffusion Model

message를 BGV 또는 CGGE의 deterministic, lossless canonical image로 encode하고 **pixel-space Gaussian DDPM 계열**에서 학습한다. lossy latent VAE를 사용하지 않는다. channel 정보가 손실되지 않는 tensor 또는 lossless 저장 형식을 사용하며, 시각화용 PNG를 학습 원본으로 오인하지 않는다.

$E_r$를 representation $r$의 encoder, $z_0=2E_r(x)-1$을 $[-1,1]$ 범위의 clean image, $t\in\{1,\ldots,T\}$를 diffusion timestep, $T$를 총 diffusion step 수, $\beta_t$를 noise variance라 한다. $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{u=1}^{t}\alpha_u$로 정의한다. 기본 forward process는 다음과 같다.

$$
z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad\epsilon\sim\mathcal N(0,I).
$$

$I$는 image 좌표의 identity covariance다. condition $c=(a,q,y_q)$를 받는 reverse model은 현재 $z_t,t,c$에서 clean image 또는 noise를 예측하고, 사전 고정한 reverse transition으로 $z_{t-1}$을 생성한다. DDPM의 기본 transition은 mean $\mu_\theta$와 covariance $\Sigma_t$를 갖는 $p_\theta(z_{t-1}\mid z_t,c)=\mathcal N(\mu_\theta(z_t,t,c),\Sigma_t)$다. noise prediction을 선택할 경우 $\mathbb E\|\epsilon-\epsilon_\theta(z_t,t,c)\|^2$를 사용한다. clean-image prediction 등 기존 구현의 선택지는 validation에서 결정하고 loss와 reverse update를 함께 freeze한다.

noise schedule, $T$, reverse variance, prediction target, sampling steps, U-Net depth/width, condition embedding, optimizer 및 checkpoint selection의 실제 값은 TBD다. train/validation만으로 선택하고 test 전에 고정한다. BGV/CGGE는 같은 conditional U-Net family와 optimizer policy를 우선 사용한다. grid height 때문에 추가 architecture padding이 필요하면 source length와 무관한 고정 처리를 명세하고 canonical grid 바깥의 기술적 padding만 제거한다. target length에 맞춘 crop은 금지한다.

sampling은 사전 고정한 Gaussian initial noise에서 시작한다. decoder 전 정규화 역변환은 $(z+1)/2$다. 수치 범위 처리·threshold·tie rule은 validation에서 정한 그대로 사용하며 non-finite 출력은 invalid다. Gaussian에 제공되는 hash 정보는 §11의 bit representation이다.

## 8. BGV — Byte Glyph Visualization

### 8.1 Canonical encoding

BGV는 Printable와 Random Bytes 모두에 적용한다. byte 하나는 8 bit이며 MSB-first 순서로 $2\times4$ binary glyph에 row-major로 배치한다. bit 하나를 $4\times4$ pixel block으로 확장하므로 byte cell의 높이×너비는 $8\times16$ pixels다.

- channel 0: byte glyph.
- channel 1: cell 전체에 채운 validity mask.
- logical slot 0: payload byte length $L$를 unsigned 1 byte로 encode.
- logical slots $1,\ldots,L_{\max}$: payload.
- total logical slots: $L_{\max}+1$.

8-column row-major layout의 row 수를 $R_{\mathrm{BGV}}$라 하면:

$$
R_{\mathrm{BGV}}=\left\lceil\frac{L_{\max}+1}{8}\right\rceil,
\qquad\mathrm{shape}_{\mathrm{BGV}}=[2,\ 8R_{\mathrm{BGV}},\ 128].
$$

$L_{\max}=31$이면 $R_{\mathrm{BGV}}=4$이므로 기존 `[2,32,128]`과 일치한다. physical grid capacity $8R_{\mathrm{BGV}}$가 logical slot 수보다 크면 남는 cell은 spatial padding이다.

canonical encoder는 header와 실제 payload slots $1,\ldots,L$의 mask를 1로 둔다. 나머지 logical payload slot과 extra physical cell은 두 channel 모두 0이다. payload `0x00`은 glyph가 0이어도 mask가 1이므로 padding과 구분된다. slot 0의 1-byte length 의미를 다른 숫자 encoding으로 바꾸지 않는다.

### 8.2 Decoder validity

기존 block-average threshold decode를 유지한다. 각 $4\times4$ block 평균을 사전 고정한 bit threshold와 비교해 8 bit를 복원한다. mask는 cell 평균과 고정 mask threshold로 판정한다. threshold 경계는 `>=`를 1로 하며 값은 validation에서 freeze한다.

다음 검사를 모두 통과해야 한다.

1. tensor shape과 수치가 정상이며 byte glyph가 8개의 유효 block 값으로 decode된다. NaN/Inf 또는 malformed cell은 invalid다. 8-bit 조합 자체에는 금지된 byte codeword가 없으며 exact canonical pixel equality를 추가 요구하지 않는다.
2. valid header: slot 0의 mask가 valid이고 header glyph가 길이로 decode된다.
3. length range: decode된 길이가 $4,\ldots,L_{\max}$ 안이다.
4. contiguous payload mask: 정확히 slots $1,\ldots,L$이 valid이며 header 길이와 일치한다.
5. padding consistency: 나머지 logical/physical cell의 mask는 0이고 glyph는 같은 block threshold 기준으로 zero padding으로 decode된다. canonical tensor의 padding도 0이어야 한다.
6. source domain: Printable run의 payload는 모두 `0x21–0x7E`, Random Bytes run은 `0x00–0xFF`다.

하나라도 실패하면 invalid candidate이며 $K$를 소비한다. padding consistency의 strict 검사는 새 generalized decoder에서 명시적으로 구현·검증해야 하며 기존 코드가 이미 모두 검사한다고 가정하지 않는다. true target length를 header/mask 검사에 사용하지 않는다.

## 9. CGGE — Character Glyph Grid Encoding

### 9.1 Fixed glyph table 및 layout

CGGE는 **Printable ASCII에만** 적용한다. character $c$의 prototype을 $G(c)\in\{0,1\}^{8\times8}$로 정의한다. 94 characters 각각이 unique glyph를 갖고 모든 $c_i\ne c_j$에 대해 $G(c_i)\ne G(c_j)$여야 한다.

기존 repository-fixed public-domain 8×8 glyph table을 그대로 사용한다. 현재 table version은 `font8x8-basic-v1`이며 [cgge.py](src/diffusion_hash_inv/encoding/cgge.py)에 포함되어 있다. 원본 glyph byte table의 SHA-256 checksum은 `6ef6d0bfa6e29c823ad9ff7d6b4b29b9af5f739fee711268ff5b83ca7aeed50a`다. glyph byte ordering·pixel orientation을 유지한다. table artifact, version, checksum을 보존하고 매 실행 시 검증한다. runtime OS font rasterization이나 installed font 대체는 금지한다.

- channel 0: fixed character glyph.
- channel 1: validity mask.
- logical cells $0,\ldots,L_{\max}-1$: payload, 총 $L_{\max}$개.
- logical cell $L_{\max}$: reserve cell 1개.
- total logical cells: $L_{\max}+1$.

8-column layout의 row 수 $R_{\mathrm{CGGE}}$와 image shape은 다음과 같다.

$$
R_{\mathrm{CGGE}}=\left\lceil\frac{L_{\max}+1}{8}\right\rceil,
\qquad\mathrm{shape}_{\mathrm{CGGE}}=[2,\ 8R_{\mathrm{CGGE}},\ 64].
$$

$L_{\max}=31$이면 기존 `[2,32,64]`다. 실제 glyph는 cells $0,\ldots,L-1$에 배치하고 해당 mask만 1이다. unused payload, reserve, extra physical cell은 canonical image의 두 channel 모두 0이다. reserve는 항상 logical index $L_{\max}$에 있으며 extra physical cell과 구분한다. length 숫자를 glyph channel에 새로 넣지 않고 decoded mask의 contiguous prefix 길이로 길이를 얻는다.

### 9.2 Nearest-glyph decoding

generated cell $\hat G$의 nearest prototype과 기본 MSE distance $d$는 다음과 같다.

$$
\hat c=\operatorname*{arg\,min}_{c\in\mathrm{Printable94}}d(\hat G,G(c)),
\qquad d(\hat G,G(c))=\operatorname{mean}\big((\hat G-G(c))^2\big).
$$

최소 distance $d_{\min}$이 glyph threshold $\tau$ 이하일 때만 valid character다. 동점은 고정 table order로 결정한다. 기본 metric은 MSE이며 대안을 쓰려면 validation에서 선택한 뒤 version과 함께 freeze한다. $\tau$, mask threshold, distance metric, 수치 처리 모두 validation only다.

shape/non-finite 검사, contiguous payload mask, decoded length $4,\ldots,L_{\max}$, reserve 및 extra physical cell의 invalid mask를 검사한다. mask-valid glyph 하나라도 $d_{\min}>\tau$이면 전체 candidate가 invalid다. inactive glyph의 수치는 payload로 읽지 않으며 canonical re-encode에서는 0이 된다. 이는 기존 nearest-glyph/mask semantics를 유지하는 처리다. Random Bytes에는 CGGE를 적용하지 않는다. provided length로 mask를 고치거나 glyph를 선택하지 않는다.

## 10. Discrete Diffusion Model

### 10.1 Primary formulation과 vocabulary

primary discrete formulation은 **Masked Discrete Diffusion**이다. message를 이미지로 바꾸지 않고 categorical sequence 자체로 처리한다. 특정 library의 상세 algorithm 이름이나 구현이 이미 선택된 것으로 기술하지 않는다.

| Source | Payload symbols | Special tokens | Vocabulary size |
|---|---|---|---:|
| Printable | 94 ASCII characters 각각 하나의 state | EOS, PAD, MASK | $94+3=97$ |
| Random Bytes | byte 0–255 각각 하나의 state | EOS, PAD, MASK | $256+3=259$ |

EOS는 payload 종료, PAD는 고정 길이 sequence의 나머지 자리, MASK는 diffusion corruption state다. **Random Byte `0x00`과 PAD는 절대로 동일 state가 아니다.** token ID와 역변환 mapping은 별도 versioned table로 freeze한다. payload 값과 세 special state는 서로 겹치지 않는다.

### 10.2 Sequence format

고정 sequence length를 $S=L_{\max}+1$로 정의한다. clean sequence $u_0=E_{\mathrm{token}}(x)$는 다음과 같다.

$$
u_0=[x_1,x_2,\ldots,x_L,\mathrm{EOS},\mathrm{PAD},\ldots,\mathrm{PAD}].
$$

$x_k$는 $k$번째 payload symbol의 token이다. $L=L_{\max}$이면 마지막 위치가 EOS이며 뒤에 PAD가 없어도 valid다. 길이 정보는 clean training target의 EOS/PAD 구조에 존재하지만 hash-only inference condition에는 제공하지 않는다.

### 10.3 Forward / Reverse Process와 objective

$u_t$를 timestep $t$의 corrupted sequence, $\rho_t$를 아직 clean인 위치를 새로 MASK로 바꿀 확률이라 한다. forward process는 각 위치의 clean token을 schedule에 따라 단계적으로 MASK로 바꾸고 이미 MASK인 위치는 그대로 유지한다. payload뿐 아니라 **EOS와 PAD도 corruption 대상**이다. true EOS/PAD 위치를 드러내는 attention mask나 uncorrupted padding 구조를 hash-only sampler에 전달하지 않는다.

reverse model은 $u_t,t,c$를 입력받아 각 masked position의 clean token distribution을 예측한다. clean prediction alphabet에는 payload, EOS, PAD가 포함되며 MASK는 clean target이 아니다. masked position 집합을 $\mathcal M_t$, 그 위치의 예측 분포를 $p_\theta$라 하면 기본 objective는 categorical cross-entropy다.

$$
\mathcal L_{\mathrm{mask}}=
\mathbb E\left[-\frac{1}{|\mathcal M_t|}
\sum_{k\in\mathcal M_t}\log p_\theta(u_{0,k}\mid u_t,t,c)\right].
$$

masked position이 없는 draw의 처리와 timestep/loss weighting은 training specification에 고정한다. 빈 분모를 만들거나 EOS/PAD를 정답으로 미리 채우지 않는다.

sampling은 기본적으로 $S$개 모두 MASK인 sequence에서 시작해 iterative denoising/unmasking한다. 다른 initial state를 쓰면 test 전에 등록하고 hash-only에서는 source 또는 true length 정보에 의존하지 않아야 한다. unmask schedule, 위치 선택, token sampling/temperature, remasking 허용 여부, step 수, architecture/optimizer는 validation에서 정하고 freeze한다. EOS/PAD도 model이 생성하며, output grammar를 맞추기 위한 사후 repair는 사용하지 않는다.

### 10.4 Candidate validity

최종 sequence는 다음을 모두 만족해야 valid다.

1. 정확히 $S$개의 정상 categorical token이고 EOS가 정확히 하나 존재한다.
2. EOS 이전 payload length가 $4,\ldots,L_{\max}$다.
3. EOS 이전에 PAD가 없다.
4. EOS 이후는 모두 PAD다.
5. MASK가 sequence 어느 위치에도 남지 않는다.
6. EOS 이전 payload token은 해당 source alphabet에 속한다.

unknown token, 잘못된 sequence length, EOS 누락/중복, trailing payload 등은 모두 invalid이며 하나의 candidate budget을 소비한다. byte `0x00`의 위치는 위 PAD 검사와 무관하다. hash-only에서 true length를 강제하지 않으며 known-length에서도 true $L$에 맞춘 EOS 이동, PAD 수정, truncation, token correction을 하지 않는다.

## 11. Hash Conditioning

primary condition은 두 diffusion family 모두 다음이다.

$$
c=(a,q,y_q).
$$

$a$는 algorithm identifier, $q$는 prefix width, $y_q$는 canonical MSB-first digest bit vector다. 모델별 condition embedding mechanism은 다를 수 있지만 제공되는 bit 정보 자체는 같아야 한다. 고정 폭 input의 unused bits는 known constant와 $q$에만 의존하는 mask로 채우며 $q$ 이후의 full-digest bits를 넣지 않는다.

hash-only input에는 raw source, source ID, target index, target length, true mask/EOS/PAD, source의 glyph/record가 들어가지 않는다. source type과 $L_{\max}$는 해당 run의 공개 고정 명세이며 target별 추가 정보가 아니다. 평가 manifest의 representative source와 full digest는 모델 API에서 격리한다.

known-length input은 $(a,q,y_q,L)$이다. Gaussian과 Discrete 및 length-aware baseline 모두 동일한 payload byte length를 받는다. condition bit order, algorithm/q encoding, length normalization 및 embedding version은 test 전에 고정한다.

기존 Gaussian hash caption은 primary에서 제거한다. auxiliary caption replication을 실행하려면 별도 ID·configuration·comparison family를 등록하며, caption에는 같은 $a,q,y_q$ 정보만 포함한다. caption 결과를 canonical-bit Gaussian-versus-Discrete primary comparison에 섞지 않는다.

## 12. Core Experiment Matrix

### Hash-only: 5 core families

| ID | Source | Model | Representation |
|---|---|---|---|
| G-P-BGV | Printable | Gaussian Diffusion | BGV |
| G-P-CG | Printable | Gaussian Diffusion | CGGE |
| G-R-BGV | Random Bytes | Gaussian Diffusion | BGV |
| D-P | Printable | Discrete Diffusion | categorical sequence |
| D-R | Random Bytes | Discrete Diffusion | categorical byte sequence |

### Known-length extension: 5 core families

| ID | Source | Model | Representation |
|---|---|---|---|
| G-P-BGV-L | Printable | Gaussian Diffusion | BGV |
| G-P-CG-L | Printable | Gaussian Diffusion | CGGE |
| G-R-BGV-L | Random Bytes | Gaussian Diffusion | BGV |
| D-P-L | Printable | Discrete Diffusion | categorical sequence |
| D-R-L | Random Bytes | Discrete Diffusion | categorical byte sequence |

총 **10개 core experiment family**다. 각 family를 algorithm, 해당 $q$, dataset tier, model seed에 대해 instantiate하고 $K$별 outcome을 계산한다. K-prefix 재평가는 별도 학습 family가 아니다. Direct Bits/Bit Diffusion은 이 matrix에 없다. non-diffusion direct predictor만 baseline/control로 유지한다.

Printable의 paired approach comparison은 G-P-BGV vs D-P 및 G-P-CG vs D-P, Random Bytes는 G-R-BGV vs D-R이다. known-length도 대응 pair를 사용한다. BGV vs CGGE와 각 hash-only vs `-L` pair는 별도 secondary hypothesis다.

## 13. Hash-only / Known-length Experiments

hash-only는 $p_\theta(x\mid H_{a,q}(x))$, known-length는 $p_\theta(x\mid H_{a,q}(x),L)$을 평가한다. 알고리즘과 $q$는 모두 run condition에 포함된다.

| Family | 금지된 true-length 기반 output 처리 |
|---|---|
| Gaussian | generated mask 수정, BGV header 교체, image/payload를 true $L$로 truncate, glyph/payload 자동 correction |
| Discrete | EOS position을 true $L$에 강제 설정, PAD 위치 수정, output truncate, true $L$ 기반 invalid token correction |

target $i$의 representative source 길이를 $L_i$라 한다. 알려진 $L_i$는 condition일 뿐 decoder의 정답 제약이 아니다. **길이가 representative source와 다르더라도 자체 format·source domain이 valid이고 digest가 일치하면 primary PreimageSuccess@K에 포함한다.** length mismatch만으로 candidate를 invalid로 바꾸지 않는다. LengthMatchRate와 LengthMatchedPreimageSuccess@K를 별도로 보고하여 length condition 준수 여부를 드러낸다.

hash-only의 length 진단은 sampling 종료 후 verifier/metric layer에서만 source length에 접근한다. training loss에서 supervised target의 EOS/PAD를 쓰는 것과 inference에서 true length를 주는 것을 혼동하지 않는다.

## 14. Baselines and Controls

### 14.1 Primary competing baselines

| Baseline | 적용 budget | 입력·역할 |
|---|---|---|
| Source-prior random search | $K=1,10,100$ | hash-only에서는 $D$에서 길이와 payload를 iid 표집 |
| Deterministic direct conditional predictor | $K=1$만 | 같은 digest bits에서 message record 또는 categorical sequence 하나를 직접 예측하는 non-diffusion baseline |
| Length-aware source-prior random search | known-length의 모든 $K$ | 제공받은 $L$에서 payload를 source alphabet으로 iid 표집 |
| Length-aware direct predictor | known-length의 $K=1$ | 같은 digest bits와 $L$을 받는 non-diffusion baseline |

direct predictor의 architecture/output codec은 TBD이며 source·condition type별로 test 전에 고정한다. 같은 source에서 Gaussian과 Discrete는 동일하게 등록된 primary baseline outcome과 비교한다. direct predictor를 여러 개 등록했다면 적용 가능한 모든 predictor를 $K=1$ superiority 조건에 포함한다. deterministic predictor 한 출력을 반복해 $K=10,100$ 경쟁자로 부풀리지 않는다.

baseline도 모델과 같은 source domain, split, target, verifier, validity rule을 사용한다. representation-specific format 차이는 각 output decoder로 검사하며, 최종 byte-domain 기준은 공통이다. baseline 선택·강약 또는 성공한 baseline만 남기는 결정은 test 전에 끝낸다.

### 14.2 실제 source-prior 성공 확률

고정 target $y_i$에 대한 single-draw probability를 $\pi_i$라 한다. hash-only의 정확한 정의는 다음과 같다.

$$
\pi_i=\sum_{x\in\operatorname{supp}(D)}D(x)
\mathbf1[H_{a,q}(x)=y_i].
$$

known-length는 $D(x\mid L_i)$를 사용한다. $L_i$는 target $i$ representative의 길이다. iid random search라면 fixed target의 budget 내 성공 확률은 다음이다.

$$
P_{B,i}(K)=1-(1-\pi_i)^K.
$$

source-prior 성공률을 무조건 $2^{-q}$로 놓지 않는다. 가능한 domain에서는 exhaustive/analytic calculation을 사용하고, 큰 domain에서는 preregistered Monte Carlo draw 수·seed·uncertainty procedure로 추정한다. test-target별 baseline 추정은 freeze 후 평가 분석으로만 수행하며 configuration 선택에 쓰지 않는다. Monte Carlo에서 hit가 0이어도 $\pi_i=0$으로 확정하지 않는다.

통계 검정의 baseline binary outcome은 실제 $K$개 candidate에서 계산한다. analytic/Monte Carlo expectation이 이를 대신하지 않는다. 별도 baseline-validation Monte Carlo 비용은 primary $K$와 분리하여 보고한다. 특히 full digest라도 target source의 재표집 확률이 존재하므로 기존의 총 candidate 수에 $2^{-q}$를 곱한 수치를 이 계획의 기대 성공 건수로 재사용하지 않는다.

### 14.3 Diagnostics / Controls

| Control | 목적과 명세 |
|---|---|
| Nearest training digest | train digest bit Hamming-nearest message를 출력하는 retrieval/누수 진단. tie rule 고정, primary competitor 아님. |
| Unconditional / zero-condition | 같은 model family에서 hash 정보를 제거한 negative control. |
| Shuffled-condition | split 내부의 hash-message 대응을 derangement한 negative control. 동일 digest로 우연히 유지된 대응도 검사. |
| Reversible-condition positive control | raw message를 lossless하게 나타내는 가역 condition에서 actual model이 held-out source를 복원하는지 확인. hash 역상 evidence 아님. |
| Exhaustive short-message sanity search | 1–2 byte domain에서 hash verifier, prefix extraction, source-prior 계산의 correctness 확인. main experiment와 분리. |

known-length에는 length-only condition, same-length shuffled-hash condition, length-aware nearest-training 진단 및 reversible positive control을 추가한다. same-length shuffle은 동일 길이 안에서 hash를 바꾸며 donor mapping과 seed를 보존한다. singleton stratum 등 derangement가 불가능한 경우 올바른 hash를 그대로 두고 shuffled라고 부르지 않는다. control-only coverage를 기록하고 미실행 stratum을 명시하며 primary target을 제거하지 않는다.

negative/positive control은 codec-only oracle shortcut이 아니라 actual model의 training/sampling 경로에서 실행한다. 기본 negative control은 train/validation/test 각 split 내부에서 대응을 별도로 섞어 해당 condition으로 학습·평가한다. correct-trained checkpoint에 대한 inference-only intervention을 추가하면 별도 진단 ID로 보고한다. 원래 비교 target에 대해 main-minus-control의 paired effect와 CI를 보고하되 primary baseline superiority를 대신하지 않는다. shuffled/zero/length-only와 main이 구분되지 않으면 hash condition을 사용한다는 강한 주장을 하지 않는다.

## 15. Metrics

### 15.1 유일한 primary metric

평가하는 target set의 크기를 $N$이라 한다. 전체 test이면 $N=N_{a,q}$, subset이면 그 subset의 unique target 수다. $y_i$는 target $i$의 exact $q$-bit digest prefix이며 $\operatorname{Valid}$는 source·representation validity predicate다.

$$
\mathrm{PreimageSuccess@K}=
\frac{1}{N}\sum_{i=1}^{N}
\mathbf1\left[\exists j\le K:\operatorname{Valid}(\hat x_{i,j})
\land H_{a,q}(\hat x_{i,j})=y_i\right].
$$

모든 attempt를 independent verifier에 전달한다. decoded raw bytes가 존재하면 invalid-domain candidate와 duplicate를 포함해 full digest를 다시 계산하고 validity와 hash match를 따로 기록한다. **bytes로 해석할 수 없는 malformed 출력은 rehash 불가능하므로 `digest=null`, invalid reason, verification count 0을 기록한다.** hash를 계산하기 위해 임의 payload를 만들어 넣지 않는다. 따라서 attempt count와 actual hash verification count는 다를 수 있지만 candidate budget에서 invalid를 제외하지 않는다.

verifier는 Python `hashlib.md5` / `hashlib.sha256` 또는 동등한 독립 reference를 사용한다. generator가 보고한 digest/성공 flag를 신뢰하지 않는다. hash 입력은 decoded payload bytes 자체이며 header, EOS, PAD, MASK, image, hex 문자열은 포함하지 않는다. full-digest run은 full equality, truncated run은 재계산된 full digest의 앞 $q$ bit equality로 판정한다.

### 15.2 Common secondary / diagnostic metrics

| Metric | 정의 |
|---|---|
| ExactSourceRecovery@K | valid candidate 중 representative source $x_i$와 bytes가 정확히 같은 출력이 있는 target 비율 |
| InDomainPreimageSuccess@K | source alphabet·길이 범위 안에서의 preimage 성공률. 본 계획은 domain을 Valid에 포함하므로 primary와 같아야 하며 불일치는 evaluator 오류 신호 |
| ValidDecodeRate | valid attempt 수 / 모든 attempt 수 |
| LengthMatchRate | valid이고 candidate 길이가 $L_i$인 attempt 수 / 모든 attempt 수 |
| LengthMatchedPreimageSuccess@K | valid, digest match, candidate length $=L_i$를 동시에 만족한 candidate가 있는 target 비율 |
| Candidate count | target별 declared/actual attempt 수, invalid·duplicate 수 |
| Actual hash verification count | 실제 reference full-hash 호출 수. train/data construction 및 보조 Monte Carlo 비용과 분리 |
| Run diagnostics | model seed, sampling seed/steps, parameter count, wall-clock, compute/FLOPs 또는 NFE |

train/validation/test 결과, representative length별 결과, seed별 결과를 분리한다. length별 평가도 unique digest target representative 기준임을 표기한다.

### 15.3 Gaussian-specific metrics

- **ExactCanonicalImage@K:** valid candidate를 canonical re-encode한 $E_r(\hat x)$가 $E_r(x_i)$와 같은 target 비율. generated continuous image의 pixel-perfect equality가 아니다. lossless encoder에서는 ExactSourceRecovery와 같은 값을 기대하며 이미지 품질의 독립 evidence로 세지 않는다.
- **CharacterDecodeRate (CGGE):** mask-valid generated payload cell 중 $d_{\min}\le\tau$로 decode된 cell 비율. 전체 candidate가 invalid여도 cell 진단에는 포함하며 malformed shape와 분모 0은 NA로 기록한다.
- **Character Error Rate (CER):** Printable decoded payload와 source의 character-level edit distance / source character 수. insert/delete를 포함하며 valid candidate subset의 conditional diagnostic임을 표시한다.
- **Bit Error Rate (BER):** payload를 MSB-first bit sequence로 바꾸어 source와 비교한다. unequal length의 alignment·insert/delete penalty, 집계 방식은 validation 전에 명세하여 freeze한다. padding 우세가 숨지 않도록 payload/header/mask 진단을 분리한다.

### 15.4 Discrete-specific metrics

- token accuracy: source canonical sequence와 final predicted token의 position별 일치율. EOS/PAD를 포함한 전체 값과 payload-only 값을 함께 보고하여 padding만 맞힌 효과를 분리한다.
- character error / byte error: Printable은 character, Random Bytes는 byte 단위 edit error와 normalization을 기록한다. invalid sequence를 자동 복구해 metric 입력으로 사용하지 않는다.
- EOS validity rate: 전체 attempt 중 EOS가 정확히 하나 존재하는 비율.
- sequence-format validity rate: 전체 attempt 중 §10.4의 sequence format 검사 통과 비율. alphabet/domain validity와의 관계도 기록한다.

secondary metric의 invalid/NA 처리, best-of-K 여부, micro/macro aggregation, error alignment 규칙은 manifest에서 확정한다. primary denominator에서는 invalid나 실패 target을 절대 제거하지 않는다. **secondary metric이 좋아도 PreimageSuccess@K가 baseline보다 통계적으로 우수하지 않으면 hash-conditioned preimage advantage를 주장하지 않는다.**

## 16. G0–G4 Validation Gates

confirmatory evidence는 $G_0\land G_1\land G_2\land G_3\land G_4$를 모두 만족해야 한다. gate는 source, model family/representation, condition type 및 해당 run configuration에 적용한다. 미실행 gate는 PASS로 간주하지 않는다.

### G0 — Data and Evaluation Independence

- train/validation/test raw-message overlap = 0.
- 각 $(a,q)$의 digest-group overlap = 0; known-length에서도 같은 digest-group key 사용.
- hyperparameter selection은 validation only이며 test adaptation 없음.
- threshold, architecture, tokenizer, schedule, sampler, baseline, target subset, statistics, stopping rule을 test 이전 freeze.
- timestamped/immutable configuration snapshot 및 가능하면 외부 preregistration 보존. 코드의 `frozen=true`만으로 시간적 독립성을 입증하지 않음.

split 생성 직후와 evaluation 직전에 overlap count, example IDs, audit code version을 저장한다. 하나라도 위반하면 G0 FAIL이다. test를 본 뒤 configuration이나 판정 기준을 바꾼 run은 confirmatory에서 제외하고 exploratory로 표시한다.

### G1 — Representation and Pipeline Correctness

decoder를 각각 $D_{\mathrm{BGV}},D_{\mathrm{CGGE}},D_{\mathrm{token}}$이라 할 때 preregistered correctness corpus에서 다음이 **100%** 성립해야 한다.

$$
D_{\mathrm{BGV}}(E_{\mathrm{BGV}}(x))=x,\qquad
D_{\mathrm{CGGE}}(E_{\mathrm{CGGE}}(x))=x,\qquad
D_{\mathrm{token}}(E_{\mathrm{token}}(x))=x.
$$

최소·최대 길이, 모든 payload symbol, 반복/혼합 message, `0x00`, `0xFF`, padding, EOS 경계, 남는 physical cell 및 invalid 사례를 포함한다. CGGE는 94 glyph uniqueness와 checksum을 검사한다. single-character glyph unit test나 1–2 byte sanity용 codec 설정은 production의 최소 길이 4와 분리한다. production decoder를 느슨하게 해 sanity check를 통과시키지 않는다.

hash verifier는 Printable/Random Bytes의 1–2 byte exhaustive domain에서 reference ground truth와 완전히 일치해야 한다. 각 candidate의 full digest와 모든 계획된 prefix를 독립 구현으로 cross-check하고, 가능한 target group의 exhaustive membership/성공률과 evaluator 판정을 대조한다. 자체 `trace_hash`도 표본에서 hashlib와 비교한다.

각 source × representation/model family × condition type에 대해 독립적인 actual-model reversible positive control을 실행한다. held-out ExactRecovery point estimate가 $\ge0.99$여야 한다. 95% confidence interval의 lower bound를 $\mathrm{LCB}_{95\%}$라 정의하고 보고한다. 기존 계획의 보조 기준인 $\mathrm{LCB}_{95\%}\ge0.98$도 가능하면 확인하되 추가 필수 gate로 사후 승격하지 않는다. 한 representation의 성공을 다른 representation이나 Discrete model에 전이하지 않는다. control용 held-out corpus는 confirmatory hash test와 별개이며 control을 보고 수정한 pipeline으로 이미 공개된 hash test를 재사용하지 않는다.

positive-control target count와 $K$는 preregister한다. **기존 Gaussian G1 ladder 명세를 우선 보존한다:** train sizes 1/4/16/64, held-out 단계의 validation/test 각각 16 messages, sampling seeds 0/1/2 각각 target당 한 candidate인 $K=1$ 반복 측정이다. 이 세 반복을 best-of-3으로 바꾸지 않는다. 기존 코드의 반복평균 exact recovery와 seed별 값을 함께 남기고 target을 cluster로 한 CI를 사용한다. 새 source/shape/condition에 이 명세를 적용할 수 있는지 freeze 전에 확인하며 변경은 별도 version으로 등록한다. Discrete positive control은 $K=1$을 우선 제안하고 target 수·반복 정책은 TBD로 남긴다. 작은 control target 수가 주는 CI 한계를 보고한다.

round-trip, verifier, 해당 family의 positive control 중 하나라도 실패하면 G1 FAIL이며 해당 main hash experiment를 시작하지 않는다. 기존 control checkpoint의 성공을 새 tokenizer/shape/architecture 검증으로 대체하지 않는다.

### G2 — Candidate-budget and Comparison Fairness

비교 대상은 test targets, $a,q$, source distribution, length range, declared $K$, verifier, evaluation logic이 같아야 한다. invalid도 $K$를 소비하고 target별 $K_{\mathrm{actual}}=K_{\mathrm{declared}}$여야 한다. 하나라도 다르면 G2 FAIL이다.

Gaussian과 Discrete의 동일 $K$는 **candidate-count fairness**다. 동일 FLOPs, training compute 또는 wall-clock을 뜻하지 않는다. computational efficiency claim은 별도의 비용 측정과 사전 정의된 비교로만 한다.

### G3 — Statistical Validation

§17의 exact one-sided McNemar, target-level paired bootstrap 10,000 resamples/95% CI 및 preregistered Holm correction을 적용한다. $p_{\mathrm{Holm}}$을 해당 family에서 보정한 p-value, $\Delta_K$를 model-minus-baseline success-rate difference라 정의한다. 모든 적용 가능한 primary baseline에 대해 다음을 동시에 만족해야 PASS다.

$$
p_{\mathrm{Holm}}<0.05
\quad\land\quad
\mathrm{LCB}_{95\%}(\Delta_K)>0.
$$

정확한 paired 정의는 §17에 따른다. primary confirmatory inference는 사전 지정된 model seed 0에서 수행하며 seed 1/2의 결과도 동일 방식으로 보고한다.

### G4 — Seed-level Reproducibility

model seed $s\in\{0,1,2\}$에서 각 primary baseline에 대한 차이를 $\Delta_{K,s}$라 한다.

| 분류 | 모든 primary baseline에 대한 조건 |
|---|---|
| Reproduced / G4 PASS | 세 seed 모두 $\Delta_{K,s}>0$ |
| Strongly Reproduced | 세 seed 모두 $\mathrm{LCB}_{95\%}(\Delta_{K,s})>0$ |
| Unstable / G4 FAIL | 하나 이상의 seed에서 $\Delta_{K,s}\le0$ |

seed 누락은 재현성 미완료이며 G4 PASS가 아니다. seed 3개를 independent population sample처럼 pool하지 않는다. model seed, sampling seed, dataset seed, baseline seed는 별도 namespace로 기록한다. 세 seed의 baseline sampling stream도 test 전에 정한다.

## 17. Statistical Analysis

### 17.1 Paired binary outcomes와 primary test

동일 target $i$와 $K$에서 model success를 $M_i$, baseline success를 $B_i$로 정의한다. 각각 budget 내 valid preimage가 있으면 1, 없으면 0이다. sample success rate를 $P_M=N^{-1}\sum_iM_i$, $P_B=N^{-1}\sum_iB_i$로 두면:

$$
\Delta_K=P_M-P_B=\frac{1}{N}\sum_{i=1}^N(M_i-B_i).
$$

$n_{10}=\#\{i:M_i=1,B_i=0\}$, $n_{01}=\#\{i:M_i=0,B_i=1\}$라 한다. primary null은 model-only discordance probability가 baseline-only discordance probability 이하라는 것이며 대립가설은 그보다 크다는 것이다. exact one-sided McNemar의 p-value는 $m=n_{10}+n_{01}$에 대해 다음과 같다.

$$
p=\Pr[Z\ge n_{10}],\qquad Z\sim\operatorname{Binomial}(m,1/2).
$$

$m=0$이면 $p=1$이다. asymptotic chi-square test나 독립 binomial CI의 overlap로 primary 검정을 대체하지 않는다.

### 17.2 Effect-size interval과 seed 처리

동일 target의 $(M_i,B_i)$ pair를 하나의 unit으로 복원추출하는 **95% percentile paired bootstrap, 10,000 resamples**를 사용한다. bootstrap RNG seed, quantile convention, subset ID를 test 전에 고정한다. $N$ target을 resample하며 $NK$ attempts나 $3N$ seed-target rows를 독립 표본으로 삼지 않는다.

seed 0이 primary inference이고 seed 1/2는 같은 target-level 통계와 CI를 보고하는 reproducibility 분석이다. seed 중 가장 좋은 결과를 primary로 바꾸지 않는다. secondary comparison도 seed 0의 confirmatory 결과와 seed별 robustness를 분리한다.

### 17.3 Holm families와 secondary hypotheses

기존의 setting별 correction 구조를 유지하되 새 matrix의 hypothesis를 반영한다. 기본 primary family key는 **(algorithm, source, condition type, dataset tier, q, K)**다. 각 family에는 해당 setting의 모든 core model–primary-baseline comparison을 넣고, 같은 setting의 사전 등록된 Gaussian–Discrete 및 BGV–CGGE paired comparison도 별개의 hypothesis로 포함한다. source별 적용 model 목록과 comparison ID 전체를 test 전에 manifest로 확정한다.

Holm step-down correction은 family의 모든 raw p-value에 **공동으로** 적용한다. baseline마다 하나씩 따로 보정해서는 안 된다. 결과를 본 뒤 family에서 불리한 model·baseline·comparison을 빼지 않는다. seed 1/2에는 같은 family 구조를 각각 적용하되 3-seed population significance를 만들지 않는다.

Gaussian–Discrete의 방향을 사전에 정할 근거가 없으므로 exact **two-sided paired McNemar**와 paired bootstrap difference CI를 사용한다. BGV–CGGE도 본 계획에서는 two-sided로 둔다. baseline superiority의 one-sided 가설과 혼합하지 않는다. hash-only–known-length comparison은 두 condition type을 걸치므로 별도 length-effect family (algorithm, source, dataset tier, q, K)에 모든 counterpart pair를 포함하고 two-sided paired comparison을 사용한다. 추가 directional hypothesis는 test 전에 근거와 family를 별도로 등록해야 한다.

setting별 Holm은 서로 다른 $q,K$, algorithm 전체에 걸친 project-wide 오류율 보장을 뜻하지 않는다. 전체 matrix 중 어느 곳에서든 우위가 있다는 통합 confirmatory claim이 필요하면 첫 test 전에 더 넓은 family를 등록한다. 그렇지 않으면 모든 결론은 명시된 family/setting에 한정한다. 실패·미실행 comparison도 manifest에서 삭제하지 않는다.

모든 baseline에 대해 $p_{\mathrm{Holm}}<0.05$와 $\mathrm{LCB}_{95\%}(\Delta_K)>0$를 동시에 만족해야 G3 PASS다. 여기의 bootstrap CI는 marginal 95% interval이며 simultaneous confidence interval이라고 부르지 않는다.

### 17.4 Power, minimum detectable effect, zero success

pilot/main 실행 전에 각 $q$의 실제 $N_{a,q}$와 K=100 subset 크기를 사용하여 power 또는 minimum detectable effect 분석을 수행한다. paired discordance rate, baseline rate, 계획된 Holm family 크기, 목표 power(값 TBD)에 대한 가정을 명시하고 train/validation 또는 독립 simulation만 사용한다. test outcome을 보고 sample size를 늘리지 않는다. $q=8$처럼 target 수의 상한이 작은 설정은 message 수를 늘려도 독립 target 수가 그만큼 늘지 않는다.

모든 결과에 absolute gain $\Delta_K$, additional solved targets $N\Delta_K$, baseline이 양수일 때 relative gain $P_M/P_B$를 보고한다. baseline이 0이면 ratio는 NA이며 무한 개선으로 표현하지 않는다.

어떤 method·seed·K에서 성공 target 수가 0이면 success probability가 0이라고 하지 않는다. target-level binomial sampling interpretation 아래 one-sided 95% upper bound는 다음과 같다.

$$
p_{\mathrm{upper}}=1-0.05^{1/N}\approx\frac{3}{N}.
$$

이는 **해당 K 내 target 성공 확률**의 상한이며 single-candidate 확률이나 모든 가능한 digest에 대한 상한이 아니다. $N$은 unique evaluated target 수이고 $NK$나 3-seed 합계가 아니다. finite held-out digest set, group selection, target별 성공 확률 이질성으로 population 해석이 제한되면 이 bound의 sampling 가정과 제한을 함께 쓰고 observed 0/N을 우선 보고한다. $N=0$은 평가 불가이며 upper bound를 계산하지 않는다.

## 18. Evidence Classification

| Level | 조건 | 허용되는 해석 |
|---|---|---|
| L0 — Invalid | G0/G1/G2 중 하나 이상 FAIL | independence/pipeline/fairness 위반으로 유효 evidence 아님 |
| L1 — No Evidence | G0–G2 PASS, G3 또는 G4 FAIL | 통계적으로 재현 가능한 baseline advantage를 확인하지 못함 |
| L2 — Truncated Evidence | G0–G4 PASS 및 $q<n_a$ | 해당 truncated-digest setting의 evidence |
| L3 — Full-Digest Evidence | G0–G4 PASS 및 $q=n_a$ | 해당 full-digest setting의 evidence |
| L4 — Cross-Algorithm Full-Digest Evidence | MD5와 SHA-256 모두 대응 tested setting에서 L3 | 두 algorithm의 명시된 setting에서 full-digest evidence |

미실행·incomplete는 별도 run status로 남기며 관측된 실패나 PASS로 꾸미지 않는다. L4는 같은 source, 대응 approach/representation, condition type, $K$ 및 비교 가능한 configuration을 명시한 MD5 $q=128$과 SHA-256 $q=256$의 별도 L3 결과가 있어야 한다. 서로 다른 유리한 model/source의 성공을 합쳐 하나의 approach가 L4라고 하지 않는다.

L2를 full hash inversion이라 부르지 않는다. known-length L2/L3/L4는 반드시 known-length preimage candidate generation evidence로 표기하고 length-aware baseline 대비 결과임을 명시한다. L1은 일반적인 inversion 불가능성의 증명이 아니다.

## 19. Execution Phases

### Phase 0 — Specification Freeze

실험 실행 전에 $L_{\max}$와 resource envelope를 확정한다. train/validation development 범위와 search budget을 먼저 등록하고, 선택된 최종 configuration을 **첫 confirmatory hash test 공개 전에 전체 matrix에 대해 seal**한다. 이후 phase의 validation-only checkpoint selection은 이때 정한 절차대로만 수행한다. pilot test를 보고 main/SHA-256 hyperparameter, matrix 또는 stopping rule을 바꾸지 않는다.

다음을 고정하고 hash/timestamp가 있는 immutable snapshot에 저장한다.

- source distribution, explicit uniform length distribution, 공통 $L_{\max}$.
- algorithm별 q 목록, K 목록, dataset tier/quota, K=100 subset, dataset/split/representative/sampling/baseline/bootstrap seeds, model seeds 0/1/2.
- BGV/CGGE encoding version, physical padding rule, glyph table/checksum, discrete tokenizer/token IDs, canonical condition bit format.
- architecture, parameterization, optimizer, training updates/batch size, checkpoint selection, Gaussian noise 및 discrete masking schedule, initial state, sampling steps.
- decoding/validity rules, 모든 threshold, token sampling policy, metric aggregation/error alignment.
- primary baseline 목록, controls, positive-control target count/K/seed policy, statistical family 및 correction scope, power assumptions, CI procedure.
- stopping rule, hardware, training/evaluation/verification/저장 budget, 반드시 실행할 full-digest reserved budget.

### Phase 1 — Data / Pipeline Validation

split audit, BGV/CGGE/tokenizer round-trip, glyph checksum, independent hash verifier, exhaustive short-message sanity check, 각 Gaussian 및 Discrete family의 reversible positive control을 수행한다. G0 또는 G1이 실패한 family는 main hash experiment를 실행하지 않는다. pipeline 수정은 hash test 공개 전에 완료하고 version/configuration을 다시 freeze한 뒤 관련 correctness/control을 재검증한다.

negative controls와 baseline-validation 절차도 구현·검증한다. 새 discrete 및 generalized image pipeline을 기존 gate artifact로 대신하지 않는다. 공통 verifier나 split integrity가 실패하면 영향을 받는 모든 family를 중단한다.

### Phase 2 — MD5 Low-q Pilot

$q\in\{8,12,16\}$, pilot dataset, 새 10개 core family, seeds 0/1/2를 평가한다. 목적은 pipeline stability, model learning, condition dependence, ValidDecodeRate, baseline validation, candidate-budget validator, 통계 pipeline, runtime/resource measurement다.

model selection은 train/validation에서만 수행하고 test는 frozen configuration으로 평가한다. target 수와 power 한계를 함께 보고한다. test에서 pipeline 결함이 발견되면 해당 결과는 invalid로 보존하고 동일 test에 맞춘 수정 결과를 confirmatory로 재명명하지 않는다.

### Phase 3 — MD5 Intermediate Difficulty

$q\in\{20,24,32,64\}$를 main dataset에서 수행한다. primary metric의 난이도 곡선 $\mathrm{PreimageSuccess@K}(q)$를 baseline, 실제 $N_{a,q}$, seed별 CI와 함께 보고한다. q별 group/target 구성이 다르므로 동일 target의 난이도 변화로 해석하지 않는다.

본 계획의 예정 matrix에는 모든 intermediate q가 포함된다. 계산 상한이나 사전 정한 validation-only stopping rule로 일부가 미실행되면 ID와 이유를 공개하고 빠진 값을 성공률 0으로 채우지 않는다. test 성능에 기반한 조기 중단·확장은 금지하며 중간 단계의 성능 중단 규칙은 full-digest budget을 취소할 수 없다.

### Phase 4 — Full MD5

$q=128$을 main dataset과 사전 고정한 candidate/compute budget, model seeds 0/1/2로 평가한다. **pilot의 성공 또는 실패와 무관하게 예정된 10개 family의 full MD5 budget을 실행한다.** 단, G0/G1 실패를 무시하고 유효 실험을 강행하지 않는다. prerequisite 실패나 hardware/resource interruption은 blocked/incomplete로 기록하고 full 평가가 완료되었다고 쓰지 않는다.

success=0이면 0/N과 §17.4 upper bound를 보고한다. nonzero full-digest finding은 raw bytes, independent rehash, split leakage, condition leakage, attempt accounting을 별도로 감사한 뒤 gates와 baseline 비교를 해석한다. 감사 자체가 성공 기준 변경을 허용하지 않는다.

### Phase 5 — SHA-256 Replication

MD5 protocol 완료 후 SHA-256에 같은 source/length law, dataset 규모, representation, family, condition information, baseline, K, seeds, 검증·통계 protocol을 적용한다. split은 SHA-256의 $(a,q)$별로 새로 만들고 감사한다. 낮은 q는 8/12/16, intermediate는 20/24/32/64/128이다.

최종 **$q=256$은 MD5 결과와 SHA-256 pilot 성공 여부에 관계없이 reserved fixed budget으로 반드시 평가**한다. G0/G1 prerequisite와 interruption 처리 원칙은 MD5와 같다. SHA-256 $q=128$ 결과를 full digest로 표기하지 않는다. MD5/SHA-256을 독립적으로 보고한 뒤 대응 setting에서만 L4 여부를 판정한다.

### Reporting closure

모든 예정 ID의 completed/failed/invalid/blocked/not-run 상태를 남긴다. MD5 다음 SHA-256 순서로 source × approach, Printable BGV–CGGE, hash-only–known-length, matched-target K curve와 q curve를 보고한다. 좋은 결과만 선택하거나 baseline·seed를 누락하지 않는다.

## 20. Required Artifacts / Manifest

다음은 새 실행에서 **생성해야 할 산출물 명세**다. 현재 모든 항목을 기존 CLI가 자동 생성한다고 가정하지 않는다. 기존 runner/validator를 새 계획에 맞게 확장하고 Phase 1에서 확인해야 한다.

| Artifact group | 필수 내용 |
|---|---|
| Protocol/configuration | document version, preregistration ID, immutable config/hash/timestamp, code commit, package versions, test access log, configuration selection provenance |
| Dataset manifest | source alphabet, length law/range, dataset tier/quota, raw draws/duplicates/unused counts, dataset/split seeds, raw bytes 또는 복원 가능한 저장, group-to-split map, representative ID/length, unique target counts, K subset IDs |
| Representation | family, encoding/tokenizer version, image shape/sequence length, token IDs, BGV header/bit order, logical vs physical cell mapping, padding rule, CGGE table artifact/version/checksum |
| Condition/model | algorithm/full width/q/full-or-truncated, exact bit format, condition type, architecture/parameter count, embedding, optimizer, training schedule/checkpoint, model seed |
| Sampler/decoder | Gaussian noise/discrete masking schedule, initial state, prediction/loss type, sampling steps/NFE, sampler seed, thresholds/tie rules, validity version |
| Candidate ledger | run/target/attempt ID, original image/sequence 또는 재현 가능한 artifact, decoded bytes, invalid reason, candidate length, provided length의 평가용 별도 field, duplicate flag, independently rehashed full digest/null, q-prefix, validity/hash-match flags |
| Outcome ledger | target-level M/B binary outcomes for each K, source recovery/length match, declared/actual candidate count, actual verifier call count, seed, subset |
| Control/audit | raw/digest overlap report, round-trip, checksum, exhaustive ground truth, family별 positive/negative control 결과와 CI, condition-input audit, matched-budget report |
| Statistical output | comparison ID/family ID/direction, N, successes, n10/n01, rates, delta, ratio/NA, additional solved targets, raw/adjusted p, paired CI, bootstrap seed/resamples, power/MDE, zero-success bound |
| Decision/resource | G0–G4 개별 status와 실패 이유, evidence level, run completion status, stopping reason, actual compute/device/memory/storage/wall-clock |

machine-readable JSON/JSONL/CSV와 사람이 읽는 요약을 함께 남긴다. 각 row는 experiment ID, source, condition type, algorithm, q, K, seed, target count를 식별할 수 있어야 한다. model이 볼 수 있는 input과 evaluator-only target metadata를 명확히 분리한다. candidate 원본이나 재현 정보 없이 성공 flag만 보존하지 않는다.

기존 관련 코드의 재사용 후보는 [dataset.py](src/diffusion_hash_inv/dataset.py), [models.py](src/diffusion_hash_inv/models.py), [evaluation.py](src/diffusion_hash_inv/evaluation.py), [validation.py](src/diffusion_hash_inv/validation.py), [positive_control.py](src/diffusion_hash_inv/positive_control.py)다. 기존 `hash-inverse-experiment`/`hash-inverse-validate` 인터페이스는 legacy용이며 새 10개 family가 지원된다고 주장하거나 본 문서 작성 중 실행하지 않는다.

## 21. Resource / Compute Reporting

K와 compute budget을 분리한다. 각 method·seed·setting에서 다음을 기록한다.

- training dataset size, updates, batch size, checkpoint selection 비용, hyperparameter search 총비용.
- parameter count, hardware/device, precision, peak memory, training/evaluation wall-clock.
- sampling step 수, NFE(number of function evaluations; denoiser forward 호출 수), batch size, target/candidate당 latency, 가능하면 FLOPs와 산정 방법.
- candidate 수, invalid/duplicate 수, 실제 hash verifier 호출 수 및 시간.
- dataset construction, source-prior Monte Carlo, exhaustive sanity, controls, storage 비용.

NFE는 architecture가 다른 모델 간 동일 연산량을 뜻하지 않는다. 동일 K 결과로 효율 우위를 주장하지 않는다. training compute를 가능한 한 맞추되 설정 차이를 숨기지 않는다. compute-matched 비교를 원하면 candidate-count primary와 별도 preregistered 분석으로 둔다.

한 method·seed·$(a,q)$에서 전체 target에 K=10을 생성하고 subset을 100까지 연장할 경우 총 생성 수는 다음과 같다.

$$
10N_{a,q}+90\min(1{,}000,N_{a,q}).
$$

K=1/10/100을 prefix로 평가하므로 저장된 candidate를 다시 생성하지 않는다. baseline/direct predictor/controls 및 독립 반복의 비용은 해당 protocol에 따라 별도 합산한다. 기존의 core family 추가에 따른 단순 33% 예산 증가 추정은 새 matrix에 재사용하지 않는다. 10개 family와 Gaussian/Discrete별 실제 측정치를 이용해 Phase 0에서 총 budget과 full-digest reserved budget을 정한다. GPU-hours, storage 및 wall-clock 상한은 TBD다.

## 22. Final Interpretation Rules

1. 모든 결론에 $D$, length law/range, algorithm, $q$, full/truncated, K, model/representation, condition type, model seeds와 actual N을 명시한다.
2. primary superiority와 G0–G4 없이 hash-conditioned preimage advantage를 주장하지 않는다. train recall, glyph/token accuracy 또는 reconstruction similarity는 대체 evidence가 아니다.
3. truncated 결과는 해당 truncated-digest 문제에만 적용한다. full MD5는 128 bit, full SHA-256은 256 bit만 해당한다.
4. L3/L4도 arbitrary hash의 효율적 inverse, 일반적인 preimage resistance 붕괴, brute-force complexity의 일반적 감소를 뜻하지 않는다.
5. Gaussian–Discrete는 end-to-end approach comparison이다. BGV–CGGE는 동일 Gaussian framework 내 representation ablation이지만 capacity/compute 차이도 공개한다.
6. known-length는 $p(x\mid H_{a,q}(x),L)$ 문제다. length-aware baseline 대비 결과와 length compliance를 보고하며 hash-only 결과와 섞지 않는다.
7. source-prior를 실제 D로 평가한다. 2^-q라는 근거 없는 치환이나 기존 full-digest 기대 건수를 사용하지 않는다.
8. zero-success에는 sample-size-limited upper bound와 가정을 포함한다. 성공하지 못했다는 관측을 확률 0 또는 이론적 불가능성으로 바꾸지 않는다.
9. test 이후 hyperparameter, threshold, baseline, target subset, hypothesis direction, Holm family, stopping rule을 바꾸지 않는다. 변경 연구에는 새 preregistration/holdout이 필요하다.
10. control 실패와 main hash failure를 구분한다. 실행하지 않은 실험의 예상 결과를 관측 결과처럼 쓰지 않는다.

허용되는 결론의 예시는 “tested source distribution, length range, condition, model configuration 및 fixed candidate budget에서 해당 model은 preregistered baseline보다 통계적으로 높고 seed 간 재현되는 valid preimage candidate 생성률을 보였다”이다. 이 문장은 실제 G0–G4를 통과한 setting에만 사용할 수 있으며 현재의 결과 진술이 아니다.

### Document Consistency Review

문서 자체의 명세 검토와 실제 실험 gate 통과를 구분한다. 본 문서에서 확인한 사항은 다음과 같다.

| 검사 | 명세 검토 결과 |
|---|---|
| 기존 핵심 질문 및 일반 inverse 주장 금지 | 유지 |
| source 94/256, uniform length, 임의의 새 최대 길이 미가정 | 반영; 새 실행 L_max는 TBD, legacy 31은 명시된 reference |
| BGV/CGGE semantics 및 shape | 유지/일반화; 남는 physical cell과 BGV 1-byte header 상한 명시 |
| Discrete 97/259 vocabulary, 0x00/PAD 분리, EOS/PAD/MASK validity | 반영 |
| hash-only length leakage, known-length output repair 금지 | condition·sampler·decoder 전 구간에 반영 |
| 공통 digest bits와 caption 분리 | primary/auxiliary 구분 반영 |
| fixed K, invalid 소비, matched-target prefix 평가 | 반영 |
| unique digest N, group split, q별 power, quota 충돌 | 별도 construction/analysis 규칙 반영 |
| G0–G4, exact one-sided McNemar, paired bootstrap, Holm, seeds 0/1/2 | 유지; secondary two-sided 및 Holm family 범위 명시 |
| baseline D 기반 확률과 zero-success upper bound | 반영; 2^-q 무조건 가정 제거 |
| full MD5/SHA-256 fixed-budget 평가 | pilot 성능과 무관한 의무 및 integrity prerequisite 구분 |
| 기존 구현/기존 결과/새 실행 계획의 구분 | 반영; 새 모델·generalized pipeline의 구현 완료 주장 없음 |

### Preregistration에서 반드시 확정할 TBD

| 항목 | 결정 기준 |
|---|---|
| 새 실행 L_max 및 source 공통 범위 | resource·representation 크기, BGV 1-byte header 범위; legacy 31 유지 여부 명시 |
| dataset/selection/RNG seeds와 construction budget | independence, 정확한 message quota, unique target 확보, 이전 관측 holdout 제외 |
| Gaussian/Discrete architecture·optimizer·training configuration | train/validation-only 선택, parameter/compute 보고 |
| noise/masking schedule, sampling steps, initial-state/temperature 정책 | validation-only 선택, 같은 K 및 leakage-free sampling |
| generalized encoder/tokenizer version, token IDs, condition embedding | 기존 semantics와 exact digest information 보존 |
| decoding threshold, 수치 처리, diagnostic error/aggregation 규칙 | validation only, invalid budget 유지, 사후 output repair 없음 |
| direct predictor 최종 구성과 전체 comparison manifest | 같은 source의 shared baseline, K=1만 적용, Holm membership/direction 고정 |
| Discrete positive-control target 수·반복 정책 | K=1 우선, ExactRecovery ≥0.99, representation별 독립 검증 |
| baseline Monte Carlo budget/seed/uncertainty, bootstrap seed, power 목표 | actual D와 unique target N에 기반, test-driven size 변경 금지 |
| stopping rule, 총 compute/storage 한도, full-digest reserved budget | MD5/SHA-256 full evaluation이 pilot 결과에 종속되지 않도록 확보 |

위 TBD는 final study design의 빈 부분을 실행 전에 채우기 위한 것이다. 어떤 값도 test에서 유리한 결과를 얻기 위해 선택하지 않는다.

## 23. Pre-run Checklist

- [ ] 원본 RESEARCH_PLAN.md와 기존 실험 artifact를 보존하고 새 protocol/version을 식별했다.
- [ ] 새 실행 $L_{\max}$를 Phase 0에서 확정했고 두 source의 길이 범위와 uniform length law를 manifest에 기록했다.
- [ ] BGV 1-byte header 범위, generalized BGV/CGGE shape, reserve/extra physical padding을 확인했다.
- [ ] Printable94와 Random Bytes256 정의, discrete 97/259 vocabulary, `0x00 != PAD`를 확인했다.
- [ ] **G0:** raw-message 및 digest-group split overlap이 모두 0이며 이전 관측 test가 새 confirmatory holdout에 섞이지 않았다.
- [ ] pilot/main message quota, generation/selection seeds, representative IDs, actual unique target N, subset IDs를 dataset manifest에 저장했다.
- [ ] **G1:** BGV/CGGE/tokenizer round-trip 100%, glyph uniqueness/checksum, independent verifier/exhaustive sanity 검사가 통과했다.
- [ ] 각 source·representation/model·condition type의 reversible positive control을 등록한 target 수/K로 수행하고 ExactRecovery ≥0.99를 확인했다.
- [ ] Gaussian/Discrete에 같은 canonical digest bits를 제공하고 hash-only input에서 true length·mask·EOS/PAD 누수를 차단했다.
- [ ] known-length는 condition으로만 사용하며 header/mask/EOS/PAD/truncation/payload correction이 없다.
- [ ] architecture, optimizer, noise/masking schedule, sampling steps, decoder, thresholds, baseline, metrics, stopping rule을 test 이전 freeze했다.
- [ ] timestamped/immutable config snapshot과 code/package/encoding/tokenizer version을 보존했다.
- [ ] **Model seeds 0, 1, 2**와 dataset/sampling/baseline/bootstrap seed namespace를 분리해 등록했다.
- [ ] **G2:** K=1/10/100, invalid·duplicate의 K 소비, resampling 금지, target별 actual=declared validator가 준비됐다.
- [ ] K=100 subset에서 K=1/10도 같은 stream prefix로 산출하는 matched-target curve를 준비했다.
- [ ] 모든 attempt의 validity 기록과 decoded bytes의 independent full-digest rehash가 준비됐으며 actual hash count를 별도 기록한다.
- [ ] **G3:** exact one-sided McNemar, paired bootstrap 10,000/95% CI, Holm family, 모든 primary baseline 통과 기준을 등록했다.
- [ ] secondary Gaussian–Discrete/representation/length 비교의 two-sided hypothesis와 correction scope를 고정했다.
- [ ] q별 actual unique N에 대한 power/MDE와 zero-success upper bound의 가정을 명시했다.
- [ ] **G4:** seed별 delta/CI와 3/3 방향성 기준을 등록했으며 seed pooling을 하지 않는다.
- [ ] source-prior baseline이 실제 source/length law를 따르고 무조건 2^-q를 사용하지 않는지 확인했다.
- [ ] parameter count, FLOPs 또는 NFE, wall-clock, memory, verification/Monte Carlo/storage 비용을 기록할 준비가 됐다.
- [ ] **full MD5 q=128 및 full SHA-256 q=256**의 fixed budget을 확보했으며 pilot 성능에 따른 취소 규칙이 없다.
- [ ] test 이후 configuration/판정 기준 변경 금지, interruption/invalid/not-run 보고, evidence L0–L4 분리 규칙을 확인했다.
- [ ] 모든 TBD를 해소하고 새 구현의 G0/G1 prerequisite를 확인한 뒤에만 별도 승인된 실험 실행 작업을 시작한다.
