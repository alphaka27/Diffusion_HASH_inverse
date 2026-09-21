# Diffusion Model을 이용한 Hash 역상 탐색 실험 계획

- 문서 상태: BGV·CGGE·Direct Bits encoder/decoder, deterministic source/digest-group split, hashlib verifier와 fixed-budget evaluator가 구현되었다. run은 split-overlap audit, digest representative별 outcome, actual candidate count를 남기며, 별도 validator가 paired McNemar/paired bootstrap/Holm과 G0--G4 결과를 만든다. CGGE는 repository-fixed public-domain 8x8 glyph table과 SHA-256 checksum을 사용한다. 대규모 pilot·main run, validation-only hyperparameter selection과 최종 figure 생성은 아직 실행하지 않았다.
- 우선순위: MD5를 먼저 완료하고 같은 실험 행렬을 SHA-256에 반복한다.
- 대상: Printable ASCII 94 characters와 Random Byte 0x00~0xFF, 모두 길이 4~31 bytes.

## 1. Research Question

> 제한된 message distribution $D$, hash algorithm $a$, 후보 예산 $K$에서 조건부 diffusion model이 held-out $y=H_a(x)$에 대해 $H_a(\hat{x})=y$인 $\hat{x}$를 같은 예산의 기준선보다 높은 확률로 생성하는가?

이는 일반적인 hash inverse가 아니라 사전에 고정한 source distribution과 예산에서의 candidate generation 능력이다. 모든 candidate는 hashlib로 full digest를 재계산하고 target의 앞 $q$ bit를 비교한다. $q=n_a$에서는 full-digest equality가 필요하다.

Secondary representation question은 Printable source에서 다음 셋 중 무엇이 held-out hash target의 candidate generation에 더 적합한가이다.

- BGV: byte-oriented visual representation. byte bit structure를 image local pattern으로 표현한다.
- CGGE: 사람이 읽을 수 있는 character-oriented visual representation.
- Direct Bits: spatial glyph abstraction 없이 record bit를 직접 쓰는 non-image binary representation.

CGGE와 BGV의 상대 우위는 representation ablation이다. 어느 방식도 source-prior baseline을 초과하지 않으면 representation 간 순위만으로 hash-conditioned inversion evidence라고 해석하지 않는다.

algorithm $a$, 적용 가능한 representation $r$마다 독립적으로 검정한다.

- $H_{0,a,r}$: held-out PreimageSuccess@K가 같은 예산의 모든 preregistered primary competing baseline보다 높지 않다.
- $H_{1,a,r}$: 모든 primary baseline보다 높고 model seed 0, 1, 2에서 같은 방향으로 재현된다.

full MD5 $q=128$은 pilot 성공 여부와 무관하게 fixed-budget으로 수행한다. full SHA-256 $q=256$도 별도 fixed-budget으로 수행한다. 축약 digest에서만 성공하면 해당 truncated-hash toy problem 결과로만 기록한다.

## 2. 용어와 후보 예산

- 역상 성공: valid $\hat{x}$의 rehash가 target의 앞 $q$ bit와 일치한다.
- 원본 복원: $\hat{x}=x$인 더 강한 진단이다.
- held-out 복원: 학습에 없던 message와 condition의 결과이며 주 판정 대상이다.
- canonical image: representation encoder가 message에서 deterministically 만든 표준 tensor다.
- 후보 예산 $K$: target 하나에 생성·검증할 최대 candidate 수다.

Diffusion과 source-prior random search는 $K=1,10,100$을 쓴다. deterministic direct predictor는 후보 하나만 내므로 $K=1$ 비교에만 쓴다. invalid decode, CGGE invalid glyph, length mismatch도 한 번 생성되면 $K$를 소비한다. 유효 candidate를 채우려고 추가 sample하지 않는다. evaluator는 run마다 $K_{\mathrm{actual}}=K_{\mathrm{declared}}$를 자동 검증해 manifest와 결과 표에 남긴다. 이를 만족하지 않는 run은 비교·성공 판정에 쓰지 않는다.

unique digest target이 $N_{a,q}$개일 때 method 하나·model seed 하나는 $K=1,10$에서 $N_{a,q},10N_{a,q}$개, $K=100$에서 사전 고정한 $\min(1{,}000,N_{a,q})$ subset에 대해 $100\min(1{,}000,N_{a,q})$개를 생성한다. method별 $K$는 변하지 않는다. core method는 기존 6개에서 8개로 늘고, 새 E1-CG와 E1-CG-L이 각각 $N\times K$를 쓰므로 core project candidate budget은 약 33% 증가한다. dataset size, optimizer update, sampling step, wall-clock, dataset construction budget은 $K$와 별도로 기록한다.

## 3. 공통 실험 명세

### 3.1 Source distribution, hash, split

| Source | 정의 | 길이 |
|---|---|---|
| Printable | ASCII 0x21~0x7E의 94 characters에서 균등 표집, 공백 제외 | 4~31 bytes |
| Random Bytes | byte마다 0x00~0xFF에서 독립 균등 표집 | 4~31 bytes |

Printable에서는 character length와 byte length가 같다. 1~2 byte는 encoder, verifier, exhaustive-search sanity check에만 쓰며 main distribution은 아니다.

| Algorithm | Full digest | 역할 |
|---|---:|---|
| MD5 | 128 bits | 최우선 |
| SHA-256 | 256 bits | 후속 비교 |

- algorithm $a$의 full digest width를 $n_a$로 둔다. $q<n_a$는 truncated condition이고 $q=n_a$만 full-digest condition이다.
- $Q_a=(\{8,12,16,20,24,32,64,128\}\cap[1,n_a])\cup\{n_a\}$로 난이도 곡선을 만든다.
- BGV와 CGGE는 동일 caption "<algorithm>-<q>:<hex digest>"를 쓴다. Direct Bits는 같은 digest bit vector를 condition으로 쓴다.
- length image experiment caption은 "<algorithm>-<q>:<hex digest>|len_bytes=<L>"다. $L$은 payload byte 수이며 header·padding은 제외한다.
- pilot split은 train/validation/test $10{,}000/1{,}000/1{,}000$, main split은 $100{,}000/10{,}000/10{,}000$ unique message다. dataset seed는 하나로 고정하고 model seed는 0, 1, 2다.
- source message는 raw bytes 기준으로 split 간 완전히 분리한다: $X_{\mathrm{train}}\cap X_{\mathrm{val}}=X_{\mathrm{train}}\cap X_{\mathrm{test}}=X_{\mathrm{val}}\cap X_{\mathrm{test}}=\varnothing$. 같은 source message가 둘 이상의 split에 있으면 그 dataset run 전체는 invalid다.
- truncated digest collision은 algorithm·$q$별 digest group split으로 처리한다. 각 $a,q$에 대해 full 또는 $q$-bit condition $Y_s^{a,q}=\{H_{a,q}(x):x\in X_s\}$의 그룹을 한 split에만 둔다. 따라서 $Y_{\mathrm{train}}^{a,q}\cap Y_{\mathrm{test}}^{a,q}=\varnothing$이며 validation을 포함한 모든 split pair도 서로소다. 같은 digest condition을 공유하는 source가 split을 가로지르면 그 experiment는 invalid다. MD5와 SHA-256은 같은 base corpus를 재사용하되, low-$q$ collision 제약 때문에 split은 $(a,q)$별로 독립적이며 algorithm 간 target-level paired comparison은 하지 않는다.
- 평가는 test message 수가 아닌 unique $(a,q,\mathrm{digest})$ condition 기준이다. collision digest의 representative source는 dataset seed로 사전 고정한다.
- Printable BGV, CGGE, Bits와 baseline은 same message, digest, split, $K$, model seed, 가능한 경우 evaluation noise seed list를 공유한다.

### 3.2 G0 — Data and Evaluation Independence

split manifest 생성 직후와 evaluation 직전에 다음 자동 validation을 실행한다. raw source bytes의 set 교집합과 각 $(a,q)$ digest-to-split map을 검사하고, pairwise message 또는 digest-group overlap의 개수·예시 ID·검사 코드 버전을 artifact로 저장한다. overlap이 하나라도 있으면 run을 중단하고 `G0=FAIL`로 기록한다; overlap을 제거한 뒤의 새 split은 새 dataset run이다.

test target을 보기 전에 validation set만으로 아래 항목을 확정하고 config snapshot에 고정한다: $\tau$, mask threshold, model architecture, tokenizer, noise schedule, sampling steps, optimizer configuration, checkpoint selection rule, $K$, candidate filtering/decoding rule, baseline definitions, stopping rule, statistical test family 및 confidence-interval procedure. test 결과를 본 뒤 이들 중 하나라도 바꾼 run은 exploratory로 표시하며 confirmatory evidence에는 사용하지 않는다.

split audit은 자동화할 수 있지만 pre-test freeze의 시점은 artifact만으로 증명할 수 없다. confirmatory claim에는 immutable external preregistration 또는 timestamped config snapshot을 별도로 보존한다.

### 3.3 가역 이미지 인코딩 $E$

image representation은 BGV와 CGGE다. 둘 다 deterministic, lossless canonical encoder여야 하고 lossless PNG 외 저장 변환은 쓰지 않는다. diffusion output이 $[-1,1]$이면 decode 전에 $[0,1]$로 되돌린다.

#### 3.3.1 BGV — Byte Glyph Visualization

BGV는 Printable과 Random Bytes 모두에 쓰는 byte-oriented canonical representation이다.

- message length: 4~31 bytes
- slots: 32, layout: 4 by 8
- slot 0: length; slots 1~31: payload
- byte glyph: MSB-first 2 by 4 binary bits
- bit block: 4 by 4 pixels; cell: 8 by 16 pixels
- glyph channel: 32 by 128; validity-mask channel 포함
- final tensor: [2, 32, 128]

mask는 length slot과 실제 payload만 valid로 표시해 payload 0x00과 padding을 구분한다. BGV decoder는 glyph와 mask threshold 뒤 length range, header, contiguous payload mask, glyph decode를 검증한다. 하나라도 실패하면 invalid candidate다.

#### 3.3.2 CGGE — Character Glyph Grid Encoding

CGGE는 Printable ASCII 전용이며 Random Byte 전체 domain에는 적용하지 않는다. 한 printable character는 사람이 직접 읽을 수 있는 하나의 fixed 8 by 8 binary bitmap glyph다.

$$
c\longmapsto G(c)\in\{0,1\}^{8\times8}
$$

- character set: ASCII 0x21~0x7E, 94 characters, 공백 제외
- glyph는 repository에 versioned fixed glyph table 또는 deterministic bitmap-font table로 포함한다.
- operating-system font renderer 또는 installed system font를 run time에 rasterize하지 않는다.
- 동일 character는 항상 동일 glyph여야 하며 94 prototype은 모두 unique다.

$$
G(c_i)\ne G(c_j)\quad\text{for all }i\ne j
$$

logical grid는 최대 31 characters와 reserve cell 하나로 구성한다.

~~~
[C00][C01][C02][C03][C04][C05][C06][C07]
[C08][C09][C10][C11][C12][C13][C14][C15]
[C16][C17][C18][C19][C20][C21][C22][C23]
[C24][C25][C26][C27][C28][C29][C30][RES]
~~~

각 cell은 8 by 8 pixels이므로 glyph channel은 32 by 64다. channel 0은 character glyph, channel 1은 validity mask이며 final tensor는 [2, 32, 64]다.

길이 $L$의 Printable message $m=(c_0,\ldots,c_{L-1})$에서 $E_{\mathrm{CGGE}}(m)$은 $G(c_i)$를 cell $i$에 배치한다. base glyph channel에 length를 별도 숫자로 encode하지 않는다. mask의 cell 0~$L-1$은 1, cell $L$~30과 reserve cell 31은 0이다. decoded length는 valid cell 수다.

$$
D_{\mathrm{CGGE}}(E_{\mathrm{CGGE}}(m))=m
$$

CGGE-L의 caption length는 model input일 뿐이다. output을 제공 length로 mask-correct, truncate, payload-correct하지 않는다.

generated cell $\hat G$는 Printable94 prototype 중 nearest glyph로 decode한다.

$$
\hat c=\arg\min_{c\in\mathrm{Printable94}}d(\hat G,G(c)),
\qquad d=\operatorname{mean}((\hat G-G(c))^2)
$$

MSE가 기본 distance다. L1 등 대안은 validation에서 한 번 선택해 test 전에 고정할 수 있다. $d_{\min}\le\tau$일 때만 valid character다. $\tau$, glyph distance, mask threshold는 Phase 0/1 또는 validation split에서 고정하고 test 결과로 조정하지 않는다.

mask가 contiguous prefix가 아니거나 reserve cell이 valid이거나 decoded length가 4~31 밖이면 invalid다. strict default는 mask-valid cell 하나라도 $d_{\min}>\tau$이면 전체 candidate invalid다. 이 invalid candidate도 $K$를 소비한다.

### 3.4 G1 — Representation and Pipeline Correctness

main experiment 전에 deterministic reversible representation은 예외 없이 round-trip success rate 100%를 통과해야 한다. 99.x%는 허용하지 않으며, 하나라도 실패하면 그 representation의 downstream run은 `G1=FAIL`이다.

- BGV: 사전 고정한 Printable 및 Random Bytes test corpus 각각에서 $D_{\mathrm{BGV}}(E_{\mathrm{BGV}}(x))=x$가 모든 $x$에 대해 성립해야 한다.
- CGGE: 94 printable glyph 각각의 single-character test에서 $D_{\mathrm{CGGE}}(E_{\mathrm{CGGE}}(c))=c$여야 한다. 또한 length 4~31, 최소·최대 길이, 반복 문자, 숫자-only, 특수문자-only, lowercase-only, uppercase-only, mixed case, mixed character classes를 포함한 고정 corpus의 모든 message에서 $D_{\mathrm{CGGE}}(E_{\mathrm{CGGE}}(x))=x$여야 한다.
- hash verifier: Printable 및 Random Bytes의 1–2 byte exhaustive sanity domain에서 implemented verifier의 판정이 exhaustive ground truth와 모든 입력에서 같아야 한다. repository의 자체 hash implementation 또는 `trace_hash`는 표본 message에서 Python `hashlib` 결과와도 일치해야 한다. verifier는 언제나 full digest를 재계산하며, full-digest preimage verdict는 full-digest equality를 사용한다; truncated run은 그 재계산 digest의 앞 $q$ bit만 condition으로 비교한다.

각 representation의 reversible positive control은 독립적으로 수행하며 held-out exact recovery point estimate가 $\mathrm{ExactRecovery}_{r,\mathrm{reversible}}\ge0.99$여야 한다. 95% lower confidence bound도 기록하고, 가능하면 $LCB_{95\%}(\mathrm{ExactRecovery}_{r,\mathrm{reversible}})\ge0.98$을 확인한다. BGV control의 성공은 CGGE를, CGGE control의 성공은 BGV를 검증하지 않는다.

manifest에는 algorithm, full digest length, $q$, distribution, length range, seed, split, experiment ID, caption format, length conditioning, encoder version 외에 다음을 저장한다.

~~~
representation
representation_version
image_shape
glyph_table_version
glyph_table_checksum
glyph_distance_metric
glyph_valid_threshold
mask_threshold
~~~

CGGE run에는 fixed glyph table 자체 또는 checksum으로 정확히 복원 가능한 artifact를 보존한다.

- BGV image diffusion input/output: [2, 32, 128].
- CGGE image diffusion input/output: [2, 32, 64].
- 두 image model은 같은 pixel-space conditional U-Net family와 optimizer policy를 우선 사용한다. lossy VAE/latent compression은 둘 다 쓰지 않는다.
- spatial size 차이로 capacity가 달라지면 가능한 한 맞추고 parameter count, training updates, sampling steps, wall-clock을 반드시 기록한다.
- Direct Bits는 digest bit vector에서 record bit vector를 생성하는 conditional Bit Diffusion이다.
- architecture, tokenizer, optimizer updates, sampling setting, decode threshold는 validation에서 한 번 고른 뒤 test 동안 고정한다.

## 4. 기본 실험과 Representation Ablation

| ID | Data | Representation | Condition | Output | Role |
|---|---|---|---|---|---|
| E1-BGV | Printable | BGV | Hash caption | [2,32,128] BGV image | byte-oriented primary image |
| E1-CG | Printable | CGGE | Hash caption | [2,32,64] CGGE image | human-readable character primary image |
| E1-BIT | Printable | Direct Bits | Digest bits | message record bits | representation baseline |
| E2-BGV | Random Bytes | BGV | Hash caption | [2,32,128] BGV image | byte-oriented primary image |
| E2-BIT | Random Bytes | Direct Bits | Digest bits | byte record bits | representation baseline |
| E1-BGV-L | Printable | BGV | Hash plus Length | [2,32,128] BGV image | BGV length effect |
| E1-CG-L | Printable | CGGE | Hash plus Length | [2,32,64] CGGE image | CGGE length effect |
| E2-BGV-L | Random Bytes | BGV | Hash plus Length | [2,32,128] BGV image | BGV length effect |

Primary image experiments는 E1-BGV, E1-CG, E2-BGV다. Direct comparison은 E1-BIT, E2-BIT다. length extension은 E1-BGV-L, E1-CG-L, E2-BGV-L다. Direct Bits length ablation은 추가하지 않는다.

모든 method는 target generation, representation-specific training, held-out $K$ sampling, representation-specific decode, full-digest rehash verification, train/held-out result 분리 저장 순서로 실행한다. Printable의 three-way comparison과 Random Byte의 BGV-versus-Bits comparison은 같은 target과 $K$에서만 해석한다.

E1-BGV-L/E1-CG-L/E2-BGV-L은 각각 hash-only counterpart와 비교한다. common PreimageSuccess@K는 다른 길이의 valid hash preimage도 센다. LengthMatchedPreimageSuccess@K는 provided $L_i$까지 같은 preimage를 요구한다.

## 5. 대조군과 control

Printable의 E1-BGV, E1-CG, E1-BIT는 동일 Printable source-prior random search를 모든 $K$의 preregistered primary competing baseline으로 사용한다. deterministic direct conditional predictor는 $K=1$에서만 추가 primary baseline이다. Random Byte와 length extension에도 같은 규칙을 적용한다. primary baseline set은 test 전에 고정하며 test 결과를 본 뒤 더 약하거나 유리한 baseline을 골라 “best competing baseline”으로 부르지 않는다. model은 적용 가능한 모든 primary baseline과 독립적으로 비교하고, 모두에 대한 G3 조건을 통과해야 competing-baseline superiority를 주장할 수 있다.

| Control | 목적 |
|---|---|
| Source-prior random search | source distribution에서 $K$개를 표집하는 경쟁 baseline |
| Nearest training digest | Hamming-nearest train digest 기반 누수·검색 진단 |
| Direct conditional predictor | non-diffusion digest-to-record 비교 |
| Unconditional/zero-condition | hash condition 제거 효과 |
| Shuffled-condition | hash-message 대응을 섞은 negative control |
| Reversible-condition | representation/model pipeline positive control |
| Exhaustive search | 1~2 byte sanity domain에서 verifier와 baseline 검증 |

length controls에는 actual $L_i$를 제공한다. length-aware random search, length-aware direct predictor, length-only control, same-length shuffled-hash control, reversible positive control을 수행한다. length-aware nearest training digest는 진단이며 경쟁 baseline이 아니다.

BGV와 CGGE positive control은 독립적이다. 각 image representation은 held-out reversible-condition에서 Exact Recovery 99% 이상을 통과해야 자신의 main hash result를 해석한다. CGGE control은 held-out Printable message를 정확한 glyph image로 생성하고 decode해야 한다. 한 representation의 control 성공은 다른 representation pipeline의 검증이 아니다.

zero-condition, shuffled-condition, same-length shuffled-hash, length-only condition은 primary hash-inversion baseline 비교와 별도로 보고한다. 이들은 codec-only shortcut이 아니라 동일 model training·sampling 경로에서 실행한다. shuffled condition은 train과 test 모두에서 derangement된 digest condition을 사용한다. 같은 target에서 $\Delta_{\mathrm{main-negative}}=P_{\mathrm{main}}-P_{\mathrm{negative}}$를 기록하고 가능한 경우 사전 고정한 95% paired CI의 $LCB_{95\%}(\Delta_{\mathrm{main-negative}})>0$도 확인한다. 특히 shuffled condition이 main condition과 동등한 성능이면 model이 hash condition을 실제로 사용한다는 강한 주장은 허용하지 않는다.

## 6. 평가 지표

$$
\mathrm{PreimageSuccess@K}=
\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[\exists j\le K:\operatorname{Valid}(\hat{x}_{i,j})\land H_{a,q}(\hat{x}_{i,j})=y_i]
$$

| Metric | 정의 |
|---|---|
| PreimageSuccess@K | primary metric; hashlib full digest 재계산 후 $q$-bit target을 검사 |
| ExactSourceRecovery@K | representative source $x_i$를 정확히 복원한 target 비율 |
| InDomainPreimageSuccess@K | source character/byte domain과 길이 범위 안에서 성공한 target 비율 |
| ExactCanonicalImage@K | BGV/CGGE candidate를 canonical re-encode한 image가 target $E(x_i)$와 같은 target 비율 |
| ValidDecodeRate | representation-specific validator를 통과한 candidate 비율 |
| CharacterDecodeRate | CGGE mask-valid generated cell 중 $\tau$ 이내 printable glyph로 decode된 비율 |
| CharacterErrorRate | 비교 가능한 source/candidate character position 중 틀린 비율; hash success가 아닌 diagnostic |
| BitErrorRate | source와 generated record의 bit error; diagnostic |
| LengthMatchRate | valid decode이면서 generated length가 $L_i$인 candidate attempt 비율 |
| LengthMatchedPreimageSuccess@K | hash와 provided length를 모두 만족한 candidate가 $K$개 안에 있는 target 비율 |

BGV invalid에는 length out of range, mask inconsistency, glyph failure가 포함된다. CGGE invalid에는 mask inconsistency, $d_{\min}>\tau$, valid cell undecodable, decoded length outside 4~31이 포함된다. Direct Bits에는 record validation을 적용한다.

`PreimageSuccess@K`가 유일한 primary metric이다. ExactSourceRecovery, CER/BER, ValidDecodeRate, CharacterDecodeRate, LengthMatchRate, image/reconstruction similarity는 diagnostic 또는 secondary metric이며, 이들이 좋아도 primary metric이 competing baseline을 통계적으로 초과하지 않으면 hash-conditioned preimage advantage라고 하지 않는다.

각 method에 train/test gap, length별 결과, actual hash verification count, sampling steps, wall-clock, seed별 result, 95% binomial confidence interval을 기록한다. 어느 method의 target-level success count가 $S=0$이면 success probability가 0이라고 쓰지 않고 $p_{\mathrm{upper}}\approx3/N$의 rule-of-three 95% upper bound와 “No successful preimage was observed under the tested candidate budget and sample size.”를 기록한다.

## 7. Validation Gates, 성공 기준과 해석

confirmatory result는 다음 Gate를 순서대로 모두 통과해야만 `validated evidence`다.

$$
G_0\land G_1\land G_2\land G_3\land G_4
$$

$G_0$는 data/evaluation independence, $G_1$은 representation/pipeline correctness, $G_2$는 candidate-budget/comparison fairness, $G_3$은 preregistered competing baselines에 대한 statistical superiority, $G_4$는 seed-level reproducibility다. 앞 Gate가 실패하면 뒤 Gate 결과와 관계없이 confirmatory success claim은 인정하지 않는다. full/truncated 여부는 Gate 통과 후의 claim classification이지 Gate가 아니다.

### 7.1 G0–G2

- **G0 — Data and evaluation independence:** §3.1–3.2의 message-level 및 $(a,q)$ digest-group-level split 검사가 모두 pass하고 test-set adaptation이 없어야 한다. 하나라도 fail하면 dataset run 또는 experiment는 invalid다.
- **G1 — Representation and pipeline correctness:** §3.4의 representation별 deterministic round-trip 100%, verifier exhaustive check, 그리고 representation별 reversible positive control $\ge0.99$를 모두 충족해야 한다. positive-control LCB와 negative-control 결과는 함께 보고하되, positive-control pass를 다른 representation으로 전이하지 않는다.
- **G2 — Candidate-budget and comparison fairness:** 비교하는 model과 baseline은 같은 test target set/subset, hash algorithm, $q$, source distribution, message-length range, declared $K$, evaluation procedure, final full-digest verifier를 사용해야 한다. 가능한 경우 random/generation seed pairing도 유지한다. invalid output도 한 candidate attempt이며, valid candidate가 나올 때까지 재생성하는 것은 금지한다. $K_{\mathrm{actual}}\ne K_{\mathrm{declared}}$이면 `G2=FAIL`이다.

### 7.2 G3 — Paired target-level statistical validation

각 preregistered model–baseline comparison과 target $i$에서 같은 budget $K$의 binary outcome을 정의한다.

$$
M_i=\mathbf{1}[\text{model finds a valid preimage within }K],\qquad
B_i=\mathbf{1}[\text{baseline finds a valid preimage within }K]
$$

$$
\Delta_K=\frac{1}{N}\sum_{i=1}^{N}(M_i-B_i)
=\mathrm{PreimageSuccess@K}_{\mathrm{model}}-\mathrm{PreimageSuccess@K}_{\mathrm{baseline}}.
$$

동일 target에 대한 paired design이므로 primary test는 exact one-sided McNemar test다. $n_{10}=\#\{i:M_i=1,B_i=0\}$와 $n_{01}=\#\{i:M_i=0,B_i=1\}$에 대해 $H_0:P(M=1,B=0)\le P(M=0,B=1)$, $H_1:P(M=1,B=0)>P(M=0,B=1)$을 검정한다. 독립 binomial CI의 겹침 여부는 primary significance criterion으로 쓰지 않는다.

effect-size CI는 target을 paired unit으로 resample하는 95% percentile paired bootstrap ($10{,}000$ resamples)을 사용한다. bootstrap seed와 procedure는 test 전 manifest에 고정한다. primary confirmatory inference는 test 전에 지정한 model seed 0의 target-level outcome으로 수행하며, seed 1/2의 target-level McNemar result와 CI도 동일하게 보고하지만 seed를 population-level independent sample으로 pool하거나 별도 population significance test에 쓰지 않는다.

Holm correction family는 test 전에 $(a,\mathrm{source},\mathrm{condition\ type},q,K)$별로 고정한다. 여기서 condition type은 hash-only와 known-length를 분리하며, family에는 그 setting의 모든 preregistered model–primary-baseline comparison과 preregistered representation-pair comparison이 포함된다. 각 primary baseline과의 comparison은 독립적으로 Holm correction을 받고, test 결과를 보고 family나 baseline을 추가·제거하지 않는다. G3는 모든 primary baseline $b$에 대해 $p_{\mathrm{Holm},b}<0.05$ 및 $LCB_{95\%}(\Delta_{K,b})>0$일 때만 pass다.

모든 result row에 absolute gain $\Delta_K$, baseline success가 양수일 때의 relative gain $RR=p_M/p_B$, additional solved targets $G=N(p_M-p_B)$를 함께 기록한다. 작은 baseline 때문에 $RR$만 단독으로 해석하지 않는다. Printable의 BGV/CGGE/Bits representation comparison도 같은 target의 paired outcome과 Holm correction으로 검정하며, baseline 승리가 다른 representation보다의 우위를 뜻하지는 않는다.

hash-inversion advantage와 representation advantage는 별도 hypothesis다. 전자는 representation $r$마다 $H_0^{\mathrm{inv}}:P_r\le P_{\mathrm{baseline}}$를 검정하고, 후자는 예를 들어 $H_0^{\mathrm{rep}}:P_{\mathrm{CGGE}}\le P_{\mathrm{BGV}}$를 검정한다. 따라서 random/direct-predictor baseline을 이겼다는 사실만으로 CGGE가 BGV보다 우수하다고 주장하지 않는다.

### 7.3 G4 — Seed-level reproducibility

model seed $s\in\{0,1,2\}$와 각 primary baseline comparison마다 $\Delta_{K,s}=P_{\mathrm{model},s}-P_{\mathrm{baseline},s}$를 계산한다. G4의 최소 기준은 모든 comparison에서 세 seed 모두 $\Delta_{K,s}>0$이다.

- **Reproduced:** 3/3 seeds에서 $\Delta_{K,s}>0$.
- **Strongly Reproduced:** 3/3 seeds 각각에서 $LCB_{95\%}(\Delta_{K,s})>0$.
- **Unstable:** 하나 이상의 seed에서 $\Delta_{K,s}\le0$.

이는 robustness check이며 seed 3개를 independent population sample로 간주한 significance test는 수행하지 않는다.

### 7.4 Formal evidence level and claim classification

representation $r$, algorithm $a$, budget $K$의 confirmatory success는 최소한 다음을 모두 만족해야 한다.

$$
\boxed{\begin{aligned}
&\mathrm{Leakage}=0\\
&\land\ \mathrm{RoundTrip}=1.0\\
&\land\ \mathrm{PositiveControl}\ge0.99\\
&\land\ K_M=K_B\\
&\land\ \bigwedge_{b\in\mathcal{B}_{\mathrm{primary}}}
\left[p_{\mathrm{Holm},b}<0.05\ \land\ LCB_{95\%}(\Delta_{K,b})>0\right]\\
&\land\ \Delta_{K,s}>0\quad\forall s\in\{0,1,2\}.
\end{aligned}}
$$

The classification is sequential:

| Level | Required condition | Allowed description |
|---|---|---|
| L0 — Invalid | any of G0, G1, G2 fails | not scientific evidence |
| L1 — No Evidence | G0–G2 pass but G3 or G4 fails | no statistically reproducible baseline advantage |
| L2 — Truncated Evidence | G0–G4 pass and $q<n_a$ | truncated-hash / truncated-digest evidence only |
| L3 — Full-Digest Evidence | G0–G4 pass and $q=n_a$ | full-digest evidence for the tested setting |
| L4 — Cross-Algorithm Full-Digest Evidence | both MD5 and SHA-256 reach L3 | cross-algorithm full-digest evidence for the tested settings |

MD5 is full-digest only at $q=128$ and SHA-256 only at $q=256$. L2 must never be described as full hash inversion. Length-conditioned runs model $p(x\mid H(x),L)$, not hash-only $p(x\mid H(x))$; when they pass, the allowed claim is known-length preimage candidate generation evidence, compared only with length-aware baselines.

With the planned $N=10{,}000$, $K=100$ full-digest random-search expectation is about $2.9\times10^{-33}$ solved MD5 targets and $8.6\times10^{-72}$ SHA-256 targets. Full-digest runs are therefore fixed-budget falsification/sanity checks, not powered opportunities to establish an advantage; a nonzero full-digest finding requires an audit before interpretation.

### 7.5 Required result schema and automatic decision

The machine-readable result table must contain the following columns; `CER or BER` uses the applicable representation diagnostic.

| Group | Required columns |
|---|---|
| Identity and setting | experiment_id, representation, source distribution, hash algorithm, condition width ($q$), full/truncated condition, candidate budget ($K$), seed, number of targets ($N$) |
| Paired outcome and effect | model successes, baseline successes, model PreimageSuccess@K, baseline PreimageSuccess@K, $n_{10}$, $n_{01}$, absolute gain ($\Delta_K$), relative gain ($RR$), additional solved targets ($G$), raw McNemar p-value, Holm-adjusted p-value, 95% CI of $\Delta_K$ |
| Diagnostics and controls | ValidDecodeRate, ExactSourceRecovery, CER or BER, positive-control result, leakage validation result, actual candidate count, zero-success upper confidence bound when applicable |
| Gate and classification | G0 status, G1 status, G2 status, G3 status, G4 status, final evidence level |

The planned validation implementation follows this order:

~~~python
if leakage_detected or test_set_adaptation:
    evidence = "L0_INVALID"
elif not representation_roundtrip_pass or positive_control < 0.99:
    evidence = "L0_INVALID"
elif actual_candidate_budget != declared_candidate_budget or not comparison_is_matched:
    evidence = "L0_INVALID"
elif any(adjusted_p[b] >= 0.05 or delta_ci_lower[b] <= 0
         for b in primary_baselines):
    evidence = "L1_NO_EVIDENCE"
elif any(seed_delta <= 0 for seed_delta in seed_deltas):
    evidence = "L1_NO_EVIDENCE"
elif q < full_digest_bits:
    evidence = "L2_TRUNCATED_EVIDENCE"
else:
    evidence = "L3_FULL_DIGEST_EVIDENCE"
~~~

An aggregate receives L4 only if separate MD5 and SHA-256 results both receive L3. The validator must retain the underlying gate failures rather than only the final label.

### 7.6 Interpretation limits

Even L3/L4 does not mean that MD5/SHA-256 preimage resistance is generally broken, that arbitrary digests have an efficient inversion algorithm, that a mathematical inverse was found, or that brute-force complexity generally decreased. The strongest permitted summary is: “Under the tested source distribution, representation, conditioning scheme, and fixed candidate budget, the trained conditional generative model produced valid preimage candidates at a statistically higher rate than the preregistered competing baselines.”

| 관측 | 해석 |
|---|---|
| train 성공, held-out 실패 | pair 암기 |
| 작은 $q$에서만 baseline 초과 | truncated-hash toy problem의 제한적 결과 |
| CGGE > BGV > baseline | character-level visual distribution이 유리할 가능성. baseline 초과가 있어야 hash-conditioned advantage다. |
| BGV > CGGE | human-readable structure가 반드시 유리하지 않으며 byte bit structure가 더 적합할 가능성 |
| CGGE approximately BGV approximately Bits approximately baseline | representation 변경만으로 advantage를 확인하지 못함 |
| CGGE positive control 성공, hash failure | CGGE pipeline은 정상이나 held-out hash reverse signal은 없음 |
| CGGE positive control 실패 | hash 결과보다 CGGE pipeline을 먼저 수정 |
| image failure, Bits success | image/caption pathway bottleneck 진단이며 image primary success는 아님 |
| length model만 length-aware baseline 초과 | known-length setting 결과이며 hash-only 성공을 대체하지 않음 |
| full MD5/SHA-256 baseline 초과 | 해당 distribution·representation·budget evidence이며 일반 inverse나 보안성 붕괴는 아님 |

## 8. 실행 단계와 중단 기준

### Phase 0 — Pipeline 검증

- MD5부터 1~2 byte sanity domain, $q=8$, hashlib verifier, exhaustive/random baseline, digest group split을 확인한다.
- split manifest의 raw-message 및 $(a,q)$ digest-group pairwise overlap을 자동 검사해 G0 artifact를 만든다.
- BGV Printable/Random Byte round-trip과 decoder를 확인한다.
- CGGE 94 glyph uniqueness, character encode/decode 100%, length 4~31 round-trip 100%, mask decoder, nearest-glyph decoder, validation-only $\tau$ selection을 확인한다.
- length caption byte count, target reuse, length metrics, length-aware baseline을 확인한다.
- 256 pair overfit에서 train exact recall 99% 이상이 가능한지 확인한다.

### Phase 1 — Positive/negative control

- MD5에서 E1-BGV, E1-CG, E1-BIT, E2-BGV, E2-BIT의 actual-model reversible, shuffled, zero-condition controls를 seed 0으로 수행한다.
- E1-BGV-L, E1-CG-L, E2-BGV-L의 actual-model reversible, same-length shuffled-hash, length-only controls를 seed 0으로 수행한다.
- BGV 또는 CGGE own positive control이 99% 미만이면 그 representation main experiment 전에 pipeline을 수정하고 control을 재실행한다. test target을 보지 않고, positive-control/negative-control protocol, primary baseline family, Holm family, CI procedure와 stopping rule을 config snapshot에 freeze한다.

### Phase 2 — MD5 pilot

- $q=8,12,16$에서 Printable E1-BGV/E1-CG/E1-BIT, Random Byte E2-BGV/E2-BIT, length E1-BGV-L/E1-CG-L/E2-BGV-L의 model selection과 expansion decision은 validation split만으로 한다.
- validation에서 고른 pilot configuration과 target subset을 freeze한 뒤, test evaluation은 seed 0, 1, 2 모두에서 같은 preregistered matrix로 수행한다. seed 0 test result로 seed 1/2, baseline, $q$, $K$를 선택하지 않는다. validation stopping rule로 선택되지 않은 setting은 미실행으로 남기며 test-based selection을 거친 run은 exploratory로만 표시한다.

### Phase 3 — MD5 scaling과 full 판정

- preregistered validation stopping rule이 허용한 method만 $q=20,24,32,64$로 확장한다. 두 단계 연속 validation signal 부재 시 남은 middle-$q$를 생략하는 rule은 test 결과 전에 고정한다.
- full MD5 $q=128$은 pilot과 무관하게 fixed budget, seeds 0/1/2로 수행한다.
- Printable full matrix: E1-BGV, E1-CG, E1-BIT, E1-BGV-L, E1-CG-L.
- Random Byte full matrix: E2-BGV, E2-BIT, E2-BGV-L.
- image method는 own positive control 통과가 prerequisite이며, BGV full MD5는 CGGE 추가로 제거하지 않는다.

### Phase 4 — SHA-256 비교

- 같은 base message, length distribution, split, $K$, model budget으로 full matrix와 controls를 반복한다.
- $q=8,12,16$ pilot의 configuration/expansion decision은 validation split에서만 내리고, frozen test matrix는 seeds 0/1/2로 평가한다. full SHA-256 $q=256$은 fixed-budget, seeds 0/1/2로 수행한다. validation stopping rule이 허용할 때만 $q=20,24,32,64,128$을 추가한다.
- Printable은 BGV, CGGE, Direct Bits, BGV plus Length, CGGE plus Length; Random Bytes는 BGV, Direct Bits, BGV plus Length를 실행한다.

### Phase 5 — 최종 보고

MD5를 먼저, SHA-256을 다음에 보고한다. algorithm 내부 순서는 Printable의 BGV, CGGE, Direct Bits, BGV plus Length, CGGE plus Length 다음 Random Bytes의 BGV, Direct Bits, BGV plus Length다. Printable에는 BGV versus CGGE versus Direct Bits versus source-prior baseline PreimageSuccess@K plot을 포함한다.

## 9. 재현성 산출물

- data/: generated data와 split manifest
- output/: config snapshot, checkpoint, candidate, metric JSON/CSV, figure
- Git commit, Python/package version, device, dataset/model seed
- invalid candidate를 포함한 target별 decode result, actual hashlib digest, prefix-comparison result
- raw-message/digest-group overlap report, frozen pre-test configuration, matched-budget report, target-level paired outcomes, McNemar/Holm/paired-bootstrap outputs, gate status와 evidence classification
- length run의 representative source ID, provided length, caption, candidate length, length-match result
- CGGE fixed glyph table 또는 checksum으로 복원 가능한 versioned artifact

현재 재사용 기반은 generate_message, generate_bytes, MD5/SHA-256 trace_hash와 회귀 test다. 대량 digest는 hashlib.md5와 hashlib.sha256으로 계산하고 trace_hash 표본으로 교차 검증한다.

## 10. 타당성 위협

- 같은 message 또는 truncated digest가 train/test에 섞이면 암기를 일반화로 오판한다. digest group split을 유지한다.
- tokenizer가 hex를 손실하면 hash conditioning이 아니라 tokenizer 실패를 측정한다.
- length는 추가 정보다. length-conditioned 성공을 hash-only 성공으로 해석하지 않으며 baseline에도 같은 length를 준다.
- provided length로 BGV header, CGGE mask, payload를 고치면 post-processing과 generation 효과가 섞인다.
- lossy codec/VAE, source entropy 차이, $K$ 또는 hash-verification count 차이, truncated-hash generalization, MD5 collision 오해는 각각 별도 threat다.
- CGGE의 O/0, I/l, S/5, B/8 같은 visually similar glyph는 cryptographic meaning이 없는 artificial similarity다. 성능 차이는 BGV와 Direct Bits로 분리 해석한다.
- system font rendering은 환경별 glyph 차이를 만들므로 repository-fixed glyph table만 쓴다.
- test data에 맞춘 glyph threshold는 leakage다.
- post-hoc baseline/family 선택, unpaired CI 비교, seed 결과의 선택적 보고는 false positive 또는 seed instability를 과대해석하게 한다. preregistered Holm family, paired test, 3-seed direction check로 막는다.
- invalid decode를 제외하거나 valid decode까지 재생성하면 숨은 candidate budget 증가가 된다. attempt-level accounting과 matched target subset으로 막는다.
- CGGE는 Printable-only이며 Random Byte domain에 일반화하지 않는다. human readability는 연구 편의성이지 cryptographic advantage 자체가 아니다.
- 좋은 setting만 선택하지 않도록 ID, seed, threshold, $K$, stopping rule을 test 전에 고정한다.

## 11. 현재 실행 인터페이스

`hash-inverse-experiment --config examples/pilot-md5-bgv.json --output output/md5-bgv-q8-s0`로 한 baseline, control 또는 diffusion run을 실행한다. `method`는 `diffusion`, `predictor`, `random`, `nearest_train`, `reversible`, `shuffled`, `zero`이며, 실제 model negative/positive control은 `condition_mode`의 `zero`, `shuffled_hash`, `reversible_record`를 사용한다. 각 run은 split JSONL/audit, config·representation manifest, checkpoint(학습 method), candidate JSONL, target outcome JSONL, metric JSON/CSV를 남긴다. seed별 model/baseline run은 `hash-inverse-validate`로 묶어 G0--G4와 evidence level을 기록한다.

## 12. 참고 근거

- [NIST: Preimage resistance](https://csrc.nist.gov/glossary/term/Preimage_resistance)
- [RFC 1321: The MD5 Message-Digest Algorithm](https://www.rfc-editor.org/info/rfc1321/)
- [RFC 6151: Updated Security Considerations for MD5](https://www.rfc-editor.org/info/rfc6151/)
- [NIST FIPS 180-4: Secure Hash Standard](https://csrc.nist.gov/pubs/fips/180-4/upd1/final)
- [Ho, Jain, Abbeel: Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Chen, Zhang, Hinton: Analog Bits](https://research.google/pubs/analog-bits-generating-discrete-data-using-diffusion-models-with-self-conditioning/)
- [Austin et al.: Structured Denoising Diffusion Models in Discrete State-Spaces](https://proceedings.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html)
