# Truncated MD5에서 Gaussian / Discrete Diffusion의 Hash-conditioned Candidate Generation 가능성을 검증하는 PoC 연구 계획

## 1. Document Status

- 작성일: 2026-09-20. **PoC 연구 설계 문서이며, 실행 사양 일부는 TBD이고 본 PoC는 미실행이다.** 예상 성능이나 필요한 검증을 관측 결과로 기술하지 않는다.
- 독립 문서: [RESEARCH_PLAN.md](RESEARCH_PLAN.md)의 최종 목표와 검증 원칙을 유지한다. [RESEARCH_PLAN_GAUSSIAN_DISCRETE.md](RESEARCH_PLAN_GAUSSIAN_DISCRETE.md)의 Gaussian/Discrete 구분도 참고하되, 두 기존 문서를 수정하거나 대체하지 않는다.
- 이번 작업은 연구 계획 작성만 수행한다. dataset 생성, model training, candidate generation, 실제 PoC 실행은 별도 실행 작업이다.
- 기존 결과의 상태는 [DIFFUSION_GATE_REPORT.md](local_experiment_archive/retired_2026-09-21/documents/DIFFUSION_GATE_REPORT.md), [EXPERIMENT_G2_TO_G6_REPORT.md](local_experiment_archive/retired_2026-09-21/documents/EXPERIMENT_G2_TO_G6_REPORT.md), 최신 [Implementation Audit](local_experiment_archive/retired_2026-09-21/artifacts/automation/IMPLEMENTATION_AUDIT.md), [Final Experiment Report](local_experiment_archive/retired_2026-09-21/artifacts/reports/FINAL_EXPERIMENT_REPORT.md), [Scientific Gate Status](local_experiment_archive/retired_2026-09-21/artifacts/gates/GATE_SUMMARY.md)를 구분하여 참조한다.
- 기존 codec, categorical tokenizer, masked diffusion primitives, canonical digest conditioning, independent evaluator, split/statistics 코드가 존재한다. 그러나 engineering smoke나 과거 positive control의 성공이 이 PoC의 새 configuration에 대한 G0/G1/G2 통과를 의미하지 않는다. 최신 기록의 과학적 실행은 specification 미확정 상태이며 learned control도 별도 검증이 필요하다.
- 과거 실험의 G2 conditional dependence–G6 full experiment 명칭과 이 문서의 **scientific gates G0–G4**는 다른 체계다. 또한 **evidence P0/P1/P2**와 **execution Phase P0–P5**도 구분한다. 미실행은 NOT_RUN/BLOCKED, 중단은 INCOMPLETE로 남기며 실패나 PASS로 바꾸지 않는다.

`TBD — freeze before test`는 실행 전에 결정·기록할 specification을 뜻한다. 구현 기본값이나 engineering fixture 값을 과학적 설정으로 자동 승인하지 않는다. 이 문서는 완결된 PoC 설계 및 미결 사양 목록이며 즉시 실행 가능한 frozen run config는 아니다.

## 2. PoC Objective

raw message를 byte sequence $x$, 사전 고정한 message distribution을 $D$, hash algorithm을 $a$, full hash 함수를 $H_a$, full digest width를 $n_a$라 한다. $q$는 digest prefix의 bit 수, $H_{a,q}(x)$는 full digest 앞 $q$ bits, $y_q=H_{a,q}(x)$는 condition이다. $K$는 target 하나에 허용된 최종 candidate attempt 수다.

> Diffusion Model이 held-out hash condition을 실제로 활용하여 source-prior random search보다 나은 valid preimage candidate를 생성할 가능성이 있는지 확인한다.

이 PoC는 제한된 $D,a,q,K$에서의 **feasibility signal**을 평가한다. 전체 연구를 한 번에 검증하는 최종 confirmatory 계획이나 일반적인 MD5 inversion 증명이 아니다. valid preimage는 원본 $x$와 달라도 허용하며, 독립 verifier의 rehash로 판정한다.

## 3. Relationship to Full Research Goal

최종 연구 목표는 변경하지 않는다.

> 제한된 message distribution $D$, hash algorithm $a$, digest condition $q$, candidate budget $K$에서 조건부 Diffusion Model이 held-out hash target에 대해 valid preimage candidate를 생성할 수 있으며, 동일 candidate budget의 baseline보다 높은 PreimageSuccess@K를 보이는지 검증한다.

PoC는 이 목표의 초기 단계다. G0–G2는 필수 prerequisite로 유지하고, G3의 paired statistical validation과 G4의 seed reproducibility 철학은 단계적인 feasibility validation으로 사용한다. Full Plan의 L0–L4를 PoC 성공 분류로 재사용하지 않는다. 기존 full-digest 의무 실험은 full research phase에 남으며 이 PoC 실행 범위에 포함하지 않는다.

## 4. Research Questions

### Primary Research Question

> **MD5의 낮은 truncated-digest 난이도에서 Gaussian 또는 Discrete Diffusion Model이 held-out hash condition을 실제로 이용하여 동일 candidate budget의 source-prior random search보다 더 높은 valid preimage generation rate를 보이는가?**

valid preimage generation rate의 primary operational metric은 target 단위 **PreimageSuccess@K**다. candidate-level similarity나 candidate별 match 비율로 대체하지 않는다. source-prior뿐 아니라 shuffled-condition보다도 높은 효과가 있어야 hash-conditioned signal로 해석한다.

### Secondary Research Questions

1. Gaussian image-space diffusion과 Discrete sequence-space diffusion 중 어느 접근에서 hash-conditioned signal이 더 명확한가?
2. Printable ASCII와 Random Bytes에서 결과가 어떻게 달라지는가?
3. $q=8\rightarrow12\rightarrow16$으로 증가할 때 hash-conditioned advantage가 유지되는가?

첫 비교의 정확한 표현은 **end-to-end Gaussian-image approach versus Discrete-sequence approach**다. representation, architecture, optimization도 다르므로 순수한 diffusion noise formulation 차이라고 해석하지 않는다. source 간 비교와 q 곡선도 서로 다른 target 집합의 결과이며 자동으로 target-level paired 비교가 되는 것은 아니다.

## 5. Scope and Excluded Scope

| 구분 | PoC primary execution |
|---|---|
| Source | Printable ASCII94, Random Bytes256 |
| Length | $L_{\max}=31$, $L\sim U\{4,\ldots,31\}$ |
| Algorithm / q | MD5 / 8, 12, 16 |
| Model | Gaussian BGV, Masked Discrete Diffusion |
| Condition | Hash-only canonical digest bits |
| Data | Pilot 10,000 / 1,000 / 1,000 messages |
| Budget | $K=1,10,100$; 같은 100-attempt stream의 prefix |
| 필수 비교 | source-prior random search, learned shuffled-condition control |
| Seeds | seed 0 먼저, 사전 규칙을 충족한 setting만 seed 1/2 |

다음은 primary matrix에서 제외한다: SHA-256, full MD5 $q=128$, $q=32/64$, CGGE, Known-length, Direct Bits core model, extensive representation ranking, Main 100k/10k/10k dataset, direct conditional predictor 필수 baseline, compute-matched cryptanalytic baseline, full L0–L4 cross-algorithm classification. 더 긴 message는 후속 **Length Scaling Experiment**로 남긴다.

$q=20$은 선택적 stretch이며 **기본 비활성**이다. 실행하려면 첫 PoC hash test 전에 별도 ID·budget·통계 범위·진입 규칙을 등록한다. 미등록 상태에서 PoC 결과를 보고 추가하면 후속 protocol이며 primary PoC 결과에 합치지 않는다. zero-condition 등 추가 diagnostic도 같은 구분을 따른다.

## 6. Source Distributions

길이 $L=|x|$는 payload byte 수다. 길이를 먼저 균등 표집한 뒤 각 위치를 독립 균등 표집한다. 서로 다른 길이의 모든 message를 통틀어 균등 표집하는 분포가 아니다.

| Source | Alphabet $\mathcal A$ | 크기 $A=\lvert\mathcal A\rvert$ | Payload law |
|---|---|---:|---|
| Printable | ASCII `0x21–0x7E`, 공백 제외 | 94 | 각 character iid uniform |
| Random Bytes | `0x00–0xFF` | 256 | 각 byte iid uniform |

Printable의 character length와 byte length는 같다. Unicode, 자연어 빈도, dictionary prior를 사용하지 않는다. Random Bytes의 `0x00`은 정상 payload이며 hex 문자열이나 text encoding으로 변환하여 해시하지 않는다.

$$
D(x)=\frac{1}{28}A^{-|x|},\qquad
4\le|x|\le31,\quad x\in\mathcal A^{|x|}.
$$

범위 밖 확률은 0이다. duplicate 제거와 digest-group holdout 후 empirical split 분포는 이 원천 sampling law와 구분하여 보고한다. baseline은 split의 실현 빈도에 맞춰 바꾸지 않고 원래 $D$를 따른다.

## 7. Message Length

$$
\boxed{L_{\max}=31},\qquad 4\le L\le31,\qquad L\sim U\{4,\ldots,31\}.
$$

두 source와 모든 model/baseline에 같은 길이 분포를 적용한다. 31을 선택한 이유는 기존 BGV `[2,32,128]`, CGGE 구조, 1-byte length header와 호환되고 encoder/decoder 검증 자산을 재사용할 수 있기 때문이다. PoC에서 encoding 구조 변경으로 생기는 confound를 피한다. CGGE 호환성을 보존하는 것은 CGGE를 이번에 실행한다는 뜻이 아니다.

1–2 byte는 별도의 codec/verifier exhaustive sanity domain에만 쓴다. 이를 위해 production의 최소 길이 4를 완화하지 않는다. 길이 32 이상은 이번 PoC 결과의 적용 범위가 아니다.

## 8. MD5 and q Definition

PoC algorithm은 $a=\mathrm{MD5}$ 하나이며 $n_{\mathrm{MD5}}=128$이다. SHA-256은 이 PoC에서 실행하지 않는다. 표준 digest byte 순서의 첫 byte에서 MSB-first로 앞 $q$ bits를 추출한다. 12-bit처럼 byte 경계에 맞지 않는 prefix도 정확히 절단하며 표시용 hex padding을 정보로 제공하지 않는다.

| q | 역할 | 해석 제한 |
|---:|---|---|
| 8 | end-to-end pipeline, conditioning, evaluator, baseline, budget sanity | 높은 success 자체는 강한 hash-conditioned evidence가 아님 |
| 12 | 초기 feasibility signal | truncated MD5의 해당 setting에만 적용 |
| 16 | 더 강한 truncated-digest feasibility | full MD5 evidence가 아님 |
| 20 | optional stretch, 별도 사전 등록 | primary matrix에 자동 포함하지 않음 |

$$
q\in\{8,12,16\},\qquad q<128\Rightarrow\text{truncated digest}.
$$

full MD5는 $q=128$에서만 해당한다. collision 탐색, 원본 source 복원, 고정 target의 preimage 생성은 서로 다른 문제다. SHA-256 replication과 full SHA-256 $q=256$은 PoC 가능성이 관찰된 후 full research roadmap에 둔다.

## 9. Gaussian Diffusion / BGV

### 9.1 Representation and validity

Gaussian은 pixel-space conditional Gaussian DDPM 계열의 기존 구현을 우선 재사용한다. message-image representation은 **BGV 하나**이며 lossy VAE를 추가하지 않는다. representation ranking이 아닌 hash-conditioned signal 확인이 목적이고, BGV는 두 source에 모두 적용되어 matrix와 계산량을 줄인다.

| BGV 항목 | 고정 의미 |
|---|---|
| Message length / slots | 4–31 bytes / 32 slots, 4×8 row-major |
| Slot 0 / slots 1–31 | unsigned 1-byte length / payload |
| Byte glyph | MSB-first 2×4 logical bits |
| Bit block / cell | 4×4 pixels / 8×16 pixels |
| Channel 0 / channel 1 | glyph / validity mask |
| Tensor shape | `[2,32,128]` |

header와 실제 payload slot만 mask=1이고 나머지는 zero padding이다. payload `0x00`은 mask로 padding과 구분한다. 기존 block-average threshold decoder와 `>=` tie rule을 유지한다. shape/non-finite, header mask, decoded length, contiguous payload mask, padding consistency를 검사하고 Printable domain도 검증한다. 실패는 invalid attempt다. true source length를 header/mask 검사·수정에 사용하지 않는다.

기존 [BGV codec](src/diffusion_hash_inv/encoding/bgv.py)의 bit/mask threshold 기본값은 각각 0.5다. 우선 재사용하되 선택값과 codec version을 freeze한다. deterministic round-trip 100%와 생성 이미지의 valid decode는 별개다.

### 9.2 Gaussian configuration policy

$E_{\mathrm{BGV}}$를 encoder, $z_0=2E_{\mathrm{BGV}}(x)-1$을 normalized clean image라 한다. Gaussian forward는 기존 noise addition 정의를 유지하고 reverse output은 $(z+1)/2$로 되돌려 decode한다. prediction target과 reverse update는 일치해야 한다.

[기존 G1 설정](examples/g1-bgv.json)은 x0/sample prediction, linear beta schedule, 50 diffusion/sampling steps, beta end 0.4, width 8, Adam learning rate $10^{-3}$, batch 16, 1,500 updates를 사용했다. [기존 검증 보고서](local_experiment_archive/retired_2026-09-21/documents/DIFFUSION_GATE_REPORT.md)의 성공은 **reversible spatial condition을 제공한 control**에서의 결과다. 이를 canonical hash-bit conditioning의 검증된 성능이나 새 두-source 설정의 G1 PASS로 옮기지 않는다. 기존 sampler는 DDIM-style update이며 문서에서 새로운 DDPM ancestral sampler가 구현된 것처럼 쓰지 않는다.

최종 architecture, parameterization, noise schedule, sampling steps와 training budget은 이 기존 근거를 우선 검토하고 §24의 제한된 validation-only 절차로 freeze한다. 검증 근거 없이 새 architecture/sampler를 발명하지 않는다. hash용 최종 선택값은 **TBD — freeze before test**다.

## 10. Discrete Diffusion

### 10.1 Primary formulation and tokens

primary formulation은 **Masked Discrete Diffusion**이다. 이미지 변환 없이 categorical sequence를 모델링한다. sequence length를 $S=L_{\max}+1=32$라 한다.

| Source | Payload states | EOS / PAD / MASK IDs | 총 states |
|---|---|---|---:|
| Printable | byte−0x21, IDs 0–93 | 94 / 95 / 96 | 97 |
| Random Bytes | byte 그대로, IDs 0–255 | 256 / 257 / 258 | 259 |

위 mapping은 기존 [TokenCodec](src/diffusion_hash_inv/encoding/tokens.py)을 재사용한다. `0x00 != PAD`이며 EOS/PAD/MASK도 서로 다른 state다. clean sequence $u_0$는 다음과 같다.

$$
u_0=E_{\mathrm{token}}(x)
=[x_1,\ldots,x_L,\mathrm{EOS},\mathrm{PAD},\ldots,\mathrm{PAD}].
$$

$L=31$이면 EOS가 마지막 위치이고 PAD는 없다. clean prediction alphabet은 payload/EOS/PAD이며 MASK는 corruption 전용으로 clean target에서 제외한다.

### 10.2 Forward process and objective

각 training sequence마다 $t\sim U(0,1)$을 표집한다. 모든 위치 $k\in\{1,\ldots,S\}$에서 독립적으로 확률 $t$로 clean token을 MASK로 바꾼다. cumulative mask probability는 $\rho(t)=t$다.

$$
m_k\mid t\sim\operatorname{Bernoulli}(t),\qquad
u_{t,k}=\begin{cases}\mathrm{MASK}&m_k=1,\\u_{0,k}&m_k=0.\end{cases}
$$

**payload, EOS, PAD에 같은 corruption law를 적용한다.** masked position 집합을 $\mathcal M_t$, 모델 parameter를 $\theta$, condition을 $c$라 할 때 기본 loss는 다음과 같다.

$$
\mathcal L_{\mathrm{mask}}=
\mathbb E\left[-\frac{1}{\max(1,|\mathcal M_t|)}
\sum_{k\in\mathcal M_t}\log p_\theta(u_{0,k}\mid u_t,t,c)\right].
$$

mask가 없는 sequence는 loss contribution 0이며 강제로 mask를 추가하거나 t를 다시 뽑지 않는다. batch에서는 sequence별 loss를 평균한다. EOS/PAD를 loss에서 제외하지 않는다.

현재 [masked diffusion 코드](src/diffusion_hash_inv/discrete.py)는 유한 schedule index를 균등 표집한다. 이는 이 문서의 연속 $t\sim U(0,1)$과 정확히 같지 않다. **연속 corruption 구현·검증은 PoC 실행 전 gap**이며 이번 문서 작업에서 수정하거나 완료로 간주하지 않는다.

### 10.3 Hash-only length leakage prevention

모델 입력은 corrupted sequence, timestep, hash condition뿐이다. true message length, true EOS/PAD position, true sequence length, padding attention mask를 별도로 제공하지 않는다. 고정 32 positions를 모두 정상 model position으로 처리하며 PAD는 categorical state일 뿐 attention padding 용도가 아니다.

clean target의 EOS/PAD는 학습 정답으로 필요하고 확률적 corruption 뒤 일부 token이 남을 수 있다. 이 자연스러운 관측을 별도 length side channel과 혼동하지 않는다. EOS/PAD를 의도적으로 항상 unmasked로 유지하거나 정답 위치 mask를 전달하는 것은 금지한다. inference는 32개 모두 MASK에서 시작하며 evaluator-only source metadata에 접근하지 않는다.

### 10.4 Basic reverse sampling

primary는 기존 absorbing-mask formulation과 일치하는 **무작위 위치의 점진적 unmasking**으로 고정한다. reverse time grid를 $1=t_J>\cdots>t_0=0$, step 수를 $J$라 한다. 현재 MASK인 각 위치는 이전 time으로 갈 때 확률 $1-t_{j-1}/t_j$로 reveal하고, clean categorical prediction에서 token을 표집한다. 이미 reveal한 위치는 유지한다. confidence에 따른 순위 선택이나 재마스킹은 사용하지 않는다. $t_0=0$에서 남은 MASK도 같은 rule로 reveal한다.

sampling steps/time grid와 temperature는 제한된 Pilot validation search에서 선택해 test 전에 freeze한다. low-confidence remasking tuning, beam search, top-k 후보 선택 후 추가 search, external reranker, true-length correction은 primary에서 금지한다. 후속 exploratory ablation으로만 둔다.

### 10.5 Final sequence validity

정확히 32개의 정상 categorical token, EOS 정확히 1개, EOS 이전 payload 길이 4–31, EOS 이전 PAD 없음, EOS 이후 전부 PAD, 전체 sequence에 MASK 없음, 해당 source alphabet 준수를 모두 요구한다. 누락/중복 EOS, unknown token, trailing payload, 잘못된 shape는 invalid다. EOS 이동, PAD 채우기, truncation 등의 repair를 하지 않으며 invalid도 K를 소비한다.

## 11. Hash Conditioning

두 approach의 primary condition은 동일하다.

$$
c=(a,q,y_q),\qquad a=\mathrm{MD5}.
$$

canonical MSB-first digest bits를 사용하고 내부 embedding architecture 차이만 허용한다. 기존 [canonical condition builder](src/diffusion_hash_inv/conditioning.py)의 algorithm/q/prefix 의미를 재사용한다. 고정 폭의 unused suffix는 q에만 의존하는 상수이며 숨은 full-digest bits를 넣지 않는다. Gaussian과 Discrete의 입력 정보 동등성을 검사한다.

raw source, source ID, target index, target length, true header/mask/EOS/PAD는 condition에 넣지 않는다. source type과 고정 최대 길이는 공개 run 명세이며 target별 정보가 아니다. Gaussian text caption은 primary comparison에서 제외한다. full digest와 representative source는 evaluator metadata로 격리한다.

## 12. Core Experiment Matrix

| ID | Source | Model | Representation | Condition |
|---|---|---|---|---|
| P-G-BGV | Printable | Gaussian Diffusion | BGV | Hash-only |
| R-G-BGV | Random Bytes | Gaussian Diffusion | BGV | Hash-only |
| P-DISC | Printable | Discrete Diffusion | categorical sequence | Hash-only |
| R-DISC | Random Bytes | Discrete Diffusion | categorical byte sequence | Hash-only |

각 ID를 q=8/12/16에 대해 독립 학습한다. 한 setting은 `(source, model/representation, MD5, q, Hash-only)`이고 K는 같은 checkpoint/stream의 평가 prefix다. model seed는 setting의 반복이다. 서로 다른 q/source의 checkpoint transfer나 test 기반 fine-tuning은 primary에 포함하지 않는다.

## 13. Dataset and Split

### 13.1 Pilot-only quotas

source·q별 train/validation/test는 각각 **10,000 / 1,000 / 1,000 unique raw messages**다. 총 여섯 dataset manifest를 두 model approach와 controls가 공유한다. Main 100k/10k/10k는 사용하지 않는다.

이 숫자는 message quota다. 실제 평가 표본 수 $N_{d,q}$는 source $d$와 q의 **unique test digest condition 수**이며, 표나 수식에서 source/q가 고정되면 $N$으로 줄인다.

$$
N_{d,q}\le\min(1{,}000,2^q).
$$

train/validation에도 digest group이 배정되므로 q=8의 실제 test N은 256보다 작다. test messages=1,000을 test targets=1,000으로 기재하지 않는다. train/validation의 unique condition 수도 별도로 기록한다.

### 13.2 Pairwise split independence

split $s$의 raw message 집합을 $X_s$, condition 집합을 $Y_s^q=\{H_{\mathrm{MD5},q}(x):x\in X_s\}$라 한다. 서로 다른 모든 train/validation/test pair $s,t$에서:

$$
X_s\cap X_t=\varnothing,\qquad Y_s^q\cap Y_t^q=\varnothing.
$$

digest group 전체는 한 split에만 속한다. raw message 또는 digest group이 split을 넘으면 해당 run은 invalid다. MD5/q별 group owner, split assignment seed, audit version과 모든 pairwise overlap count를 저장한다.

low-q group 크기와 정확한 message quota를 동시에 만족하기 위해 기존 deterministic reserve-pool/exact-quota construction을 재사용한다. 부족하면 사전 고정한 draw budget 안에서 같은 D로 보충하고 surplus는 unused로 남긴다. quota를 못 채우면 construction failure이며 group을 다른 split으로 옮기거나 quota를 조용히 줄이지 않는다. construction budget과 seed는 Phase P0에서 고정한다.

### 13.3 Target representatives and provenance

collision group마다 representative source를 dataset seed로 deterministic하게 사전 고정한다. primary는 group당 한 target이며 여러 source를 독립 target처럼 세지 않는다. source recovery와 length diagnostics만 representative를 참조한다.

같은 source·q에서는 모든 model, shuffled control, source-prior와 seed가 같은 ordered target IDs를 사용한다. test quota가 1,000이므로 모든 unique target에 K=100을 적용할 수 있으며 별도 target subset을 선택하지 않는다.

각 q의 split은 독립 manifest다. base pool은 재사용할 수 있지만 서로 다른 q의 학습 경로·checkpoint를 섞지 않고 cross-q paired inference를 가정하지 않는다. 이전에 관측한 legacy/engineering raw test corpus는 새 평가 corpus로 재사용하지 않는다. 이는 모든 역사적 low-q prefix가 전 세계적으로 미관측이어야 한다는 뜻이 아니며, 해당 모델의 train/validation/test condition independence와 test 접근 이력을 감사한다.

raw draw·duplicate·unused 수, split별 길이 빈도, message quota, unique digest count, representative selection을 기록한다. **actual test N은 항상 unique digest target 기준**이다.

## 14. Candidate Budget

$$
K\in\{1,10,100\},\qquad C_1\subset C_{10}\subset C_{100}.
$$

$C_K=(\hat x_{i,1},\ldots,\hat x_{i,K})$는 target $i$의 ordered attempt stream이다. 위 포함 표기는 duplicate를 제거한 set이 아니라 prefix 관계다. 각 target에서 100개를 생성하고 앞 1/10/100개로 평가하므로 같은 target set에서 반드시 다음이 성립한다.

$$
\mathrm{PreimageSuccess@1}\le\mathrm{PreimageSuccess@10}
\le\mathrm{PreimageSuccess@100}.
$$

- invalid, malformed, duplicate candidate는 각각 attempt 하나를 소비한다.
- valid 또는 unique candidate가 나올 때까지 추가 sampling하지 않는다. 성공 후에도 100-attempt stream을 완료한다.
- denoising step은 candidate attempt가 아니다. 여러 최종 후보를 만든 뒤 budget 밖에서 골라 제출하는 것은 금지한다.
- 생성 중 hash를 조회하여 candidate를 수정하거나 성공 후보만 남기는 search는 primary sampler가 아니다.
- target·method·seed마다 $K_{\mathrm{actual}}=K_{\mathrm{declared}}$여야 한다. 중단으로 부족하면 INCOMPLETE/G2 미통과이며 누락을 실패 후보로 만들어 채우지 않는다.
- 재개할 때 저장된 attempt/RNG state를 이어 쓰고 invalid output을 교체하지 않는다. 이 재개 경로가 준비되지 않았다면 실행 전 blocker로 남긴다.

raw sampler output, decode 결과, invalid reason, duplicate flag, attempt index를 보존한다. duplicate는 추가 해시 호출까지 기록하고 성공률 분모에서 제거하지 않는다. 동일 K는 candidate-count fairness이며 동일 FLOPs나 wall-clock을 뜻하지 않는다.

## 15. Baseline

필수 primary competing baseline은 **Source-prior random search 하나**다. D에서 길이와 payload를 독립 표집하고 같은 target, q, K, verifier와 byte-domain validity로 판정한다. target length, test의 empirical length 빈도, 모델의 유리한 후보를 제공하지 않는다. direct conditional predictor는 full research로 미룬다.

source·q·target ID별 canonical evaluation seed에서 baseline 100-attempt stream을 한 번 생성한다. 이 seed namespace는 model seed와 독립이며 두 approach와 model seeds 0/1/2 모두에 같은 stream/outcome을 재사용한다. baseline을 seed마다 다시 추첨하거나 유리한 realization을 고르지 않는다.

target $y_i$의 실제 single-draw probability를 $\pi_i$라 하면:

$$
\pi_i=\sum_{x\in\operatorname{supp}(D)}D(x)
\mathbf1[H_{\mathrm{MD5},q}(x)=y_i],\qquad
P_{B,i}(K)=1-(1-\pi_i)^K.
$$

source-prior를 무조건 $\pi_i=2^{-q}$라고 가정하지 않는다. exhaustive/analytic 계산이 가능한 sanity domain은 정확히 계산하고, main D에서는 별도 preregistered Monte Carlo draw budget·seed·uncertainty procedure로 추정한다. target별 추정과 평균 $N^{-1}\sum_iP_{B,i}(K)$를 함께 기록한다. 0 hits는 resolution-limited 추정이며 진짜 확률 0으로 쓰지 않는다.

통계 비교의 baseline binary outcome은 **실제로 생성한 K개 stream**에서 계산한다. analytic/MC expectation으로 대체하지 않는다. MC/exhaustive 검증용 hash calls와 비용은 primary candidate budget 밖의 별도 진단 비용으로 공개하며 그 후보를 model/baseline primary stream에 넣지 않는다. Monte Carlo budget과 seed는 **TBD — freeze before test**다.

## 16. Positive / Negative Controls

### 16.1 G1-A — Codec Correctness

encoder/decoder를 각각 $E_r,D_r$라 하면 고정 correctness corpus의 모든 message에서 다음을 요구한다.

$$
D_{\mathrm{BGV}}(E_{\mathrm{BGV}}(x))=x,\qquad
D_{\mathrm{token}}(E_{\mathrm{token}}(x))=x.
$$

둘 모두 **100%**여야 한다. 길이 4/31과 중간 길이, 모든 source symbol, 반복·혼합 payload, `0x00`/`0xFF`, EOS 끝 위치와 PAD 경계, malformed 사례를 포함한다. 이는 actual-model conditional generation 성공과 별도 검사다.

### 16.2 G1-B — Conditional Generation Positive Control

각 source × model family × 적용 configuration에서 **held-out reversible condition**으로 실제 model training/sampling/decoding을 수행한다. 가역 condition은 raw source를 lossless하게 담되 generator가 condition을 직접 decode해 반환하는 oracle shortcut을 사용하지 않는다. Gaussian은 기존 encoded BGV spatial condition을 우선 재사용한다. Discrete의 정확한 condition 정의/embedding과 target 수는 **TBD — freeze before test**다.

$$
\mathrm{ExactRecovery@1}\ge0.99.
$$

이를 G1-B 필수 point-estimate 기준으로 채택하고 95% CI도 보고한다. 가능하면 lower bound 0.98 이상을 확인하되 사후 필수 gate로 승격하지 않는다. positive control은 의도적으로 가역 정보를 제공하는 별도 과제이며 hash-only length leakage 금지와 혼동하지 않는다.

Gaussian의 기존 [positive-control 명세](src/diffusion_hash_inv/positive_control.py)는 train ladder 1/4/16/64, held-out 단계 train 64 / validation 16 / test 16 messages, sampling seeds 0/1/2 각각 target당 K=1이다. 이 **16-target 명세는 이미 존재**하므로 임의의 TBD로 지우지 않는다. PoC의 두 source와 최종 architecture에 적용할 수 있는지는 Phase P0에서 확인하고, 변경하면 새 version·근거를 등록한다. 세 sampling 반복을 best-of-3으로 바꾸거나 독립 target 48개로 세지 않는다. seed별 K=1 recovery와 기존 반복평균, target-cluster CI를 함께 남긴다. 작은 target 수로 인해 CI가 넓을 수 있다.

| Control specification | Gaussian | Discrete |
|---|---|---|
| Reversible condition | 기존 exact BGV tensor의 spatial condition 우선 재사용; 최종 연결 방식 freeze | 정확한 가역 condition·연결 방식 TBD |
| Held-out target 수 | 기존 validation/test 각각 16; PoC 적용 확인 필요 | **TBD**, Phase P0에서 정확한 수 지정 |
| K / 반복 | K=1, sampling seeds 0/1/2; best-of 금지 | K=1; 반복 정책 TBD |
| Model seeds / coverage | 적용 configuration과 model seed coverage freeze | 적용 configuration과 model seed coverage freeze |
| PASS | ExactRecovery@1 ≥0.99, CI 병기 | ExactRecovery@1 ≥0.99, CI 병기 |

control corpus는 hash test와 분리한다. 학습된 positive-control checkpoint를 hash model 초기값으로 전이하지 않는다. 한 source/family의 성공을 다른 것으로 전용하지 않는다. q별 architecture/training 차이가 있으면 control 적용 범위를 명시하고 해당 구성을 별도로 검증한다. 복제 seed에도 적용 가능한 G1 근거가 필요하며 coverage 누락을 PASS로 처리하지 않는다.

### 16.3 Required shuffled-condition negative control

train/validation/test 각각 split 내부에서 digest condition donor를 derangement하여 hash-message correspondence를 제거한다. donor mapping·seed를 저장하고 **실제 q-bit digest가 원래 condition과 다름**을 검증한다. 단순 record index permutation은 truncated collision 때문에 충분하지 않다. 유효 derangement가 불가능하면 원래 condition을 그대로 쓰지 않고 control blocked로 기록한다.

shuffled-condition 모델을 별도로 학습한다. main에서 frozen된 architecture, selected training update budget, optimizer, sampler, decoder, model seed를 공유하고 대응만 제거한다. shuffled 성능에 맞춘 추가 tuning은 하지 않는다. 평가에서도 split-local shuffled condition을 제공하되 성공 판정 target은 원래 $y_i$로 유지해 main과 paired comparison한다. correct-trained checkpoint의 inference-only shuffle은 추가 diagnostic일 수 있으나 필수 learned shuffled control을 대체하지 않는다.

$$
\Delta_{\mathrm{shuffled}}=P_{\mathrm{main}}-P_{\mathrm{shuffled}}.
$$

$P$는 같은 unique target set과 K에서의 PreimageSuccess@K다. main이 source-prior보다 높아도 shuffled와 차이가 없으면 hash condition 사용 가능성이 입증되었다고 보지 않는다. zero-condition 등은 optional diagnostic이다.

## 17. Metrics

### 17.1 유일한 primary metric

$N$은 평가한 **unique digest targets 수**, $y_i$는 target prefix, $\hat x_{i,j}$는 attempt j의 decoded payload다. $\operatorname{Valid}$는 representation format, source alphabet, 길이 4–31을 모두 만족하는 predicate다.

$$
\boxed{\mathrm{PreimageSuccess@K}=
\frac{1}{N}\sum_{i=1}^N
\mathbf1\left[\exists j\le K:
\operatorname{Valid}(\hat x_{i,j})\land
H_{\mathrm{MD5},q}(\hat x_{i,j})=y_i\right].}
$$

원본 representative source와 같을 필요가 없다. 다른 길이의 candidate라도 source domain·format이 valid하고 prefix가 같으면 성공이다. LengthMatchRate를 primary 조건으로 넣지 않는다.

모든 attempt는 independent reference verifier 경로에 전달한다. decoded bytes가 있으면 domain-invalid/duplicate를 포함하여 `hashlib.md5`로 **full MD5를 재계산**한 뒤 q-bit prefix만 비교한다. 모델이 보고한 digest/성공 flag는 사용하지 않는다. hash 입력은 payload bytes이며 BGV header, EOS/PAD/MASK, hex 문자열은 포함하지 않는다.

bytes로 해석할 수 없는 malformed output은 rehash할 수 없으므로 `digest=null`, invalid reason, verification count 0으로 남긴다. 임의 bytes를 만들어 해시하지 않는다. 따라서 actual candidate count와 actual hash-verification count는 다를 수 있다. invalid target/attempt를 primary denominator에서 제거하지 않는다.

### 17.2 Required secondary metrics and accounting

| Metric | 정의 / 집계 |
|---|---|
| ExactSourceRecovery@K | valid candidate 중 representative source와 bytes가 같은 것이 있는 target 비율 |
| ValidDecodeRate | valid attempts / all attempts |
| InDomainPreimageSuccess@K | source alphabet·length 범위 내 성공률; 이 문서는 domain을 Valid에 포함하므로 primary와 같아야 함 |
| LengthMatchRate | valid이고 representative 길이와 같은 attempts / all attempts |
| Character Error Rate, Printable | valid decoded payload의 character edit distance / representative character length |
| Bit / Byte Error Rate | valid payload와 representative의 MSB-first bit / byte edit distance를 각각 representative 길이로 정규화 |
| Gaussian BGV decode failure reasons | invalid reason별 count와 all-attempt 비율; nonfinite/shape/length/header/mask/padding/domain 포함 |
| Discrete EOS validity | EOS 정확히 1개인 정상 shape token outputs / all attempts |
| Discrete format validity | §10.5의 sequence format 통과 attempts / all attempts; source domain 검사는 별도 flag도 보존 |
| Candidate generation time | sampling 시간과 decode/verifier 시간 분리; target·candidate latency 및 총 wall-clock |
| Training time | 선택된 run 시간과 validation/tuning 총시간을 분리 |
| Sampling steps / NFE | 실제 reverse step 수와 denoiser forward 횟수 |
| Actual candidate count | target·method·seed별 actual/declared, invalid 및 duplicate counts |
| Actual hash-verification count | 실제 full-MD5 호출 수; dataset/MC/sanity 비용과 분리 |

error rate는 insertion/deletion을 포함한 unit-cost edit distance로 정의하고 각 valid candidate의 정규화 값을 macro-average한다. best-of-K similarity selection은 하지 않는다. invalid는 error metric의 분모에서 제외하되 valid 수/coverage를 병기하고 valid=0이면 NA다. 이 제한은 secondary에만 적용한다. padding/header가 error를 낮추지 않도록 payload 기준으로 계산하고, 보조 token accuracy를 추가하면 payload-only와 EOS/PAD 포함 값을 분리한다. error rate는 insertion 때문에 1보다 클 수 있다.

secondary가 좋아도 primary에서 source-prior/shuffled 대비 advantage가 없으면 hash-conditioned feasibility evidence로 해석하지 않는다. 모든 표는 source, q, K, seed, **actual unique N**, 성공 target 수를 함께 제공한다.

## 18. G0 — Data Independence

필수 PASS 조건:

- raw message overlap = 0, validation 포함 모든 split pair.
- q-bit digest-group overlap = 0, `(MD5,q)`별 모든 split pair.
- tuning 및 checkpoint/config selection은 validation-only.
- first hash test 이전 frozen configuration/selection procedure와 timestamp·content hash 존재.
- test-set adaptation 없음; historical/engineering corpus와 접근 이력 기록.

split 생성 직후와 evaluation 직전에 감사한다. 하나라도 위반하면 해당 PoC run은 invalid다. 단순 `frozen=true` flag가 사전 freeze 시점을 입증하지는 않는다. immutable snapshot 또는 외부 preregistration을 보존한다.

사전에 고정한 Stage B 복제 여부 계산은 **실행할 seed의 allocation만** 바꾼다. q/K, hyperparameters, target set, 판정 기준을 바꾸지 않으며 post-hoc tuning으로 사용하지 않는다. 이 설계는 선택된 setting의 feasibility 복제이고 독립 confirmatory study라고 부르지 않는다.

## 19. G1 — Pipeline Correctness

필수 PASS 조건:

- BGV deterministic round-trip 100%.
- Discrete tokenizer deterministic round-trip 100%, PAD/0x00 분리 및 malformed validity 확인.
- MD5 verifier sanity PASS: 1–2 byte의 Printable/Random Bytes exhaustive domain, q=8/12/16 prefix extraction, reference MD5와 독립 prefix extraction/ground-truth verdict 일치.
- source·model-family별 actual-model reversible positive control 기준 통과.
- 공통 hash bits와 hash-only length leakage 방지 경로 검증.

Gaussian/Discrete가 각자의 해당 codec와 positive control을 통과해야 한다. 어느 family가 실패하면 해당 family의 hash experiment로 feasibility claim을 하지 않고 먼저 control 문제를 해결한다. 공통 split/verifier 오류는 영향받는 모든 family를 중단한다. 수정 후 이전 test를 보고 맞춘 결과를 새 유효 evidence로 재명명하지 않는다.

기존 unit test나 oracle sampler success는 필요 근거지만 G1-B actual-model generation을 대신하지 않는다. 최신 repository의 engineering 결과와 PoC G1 status를 별도로 기록한다.

length leakage 검사는 같은 `(a,q,y_q)`에서 evaluator-only representative length/EOS/PAD metadata와 q 이후 digest bits를 바꿔도 condition tensor와 동일 RNG의 sampler 초기 입력이 변하지 않는지 확인한다. t=1에서는 payload/EOS/PAD 모두 MASK여야 하고, 중간 t에서도 token 종류에 따른 corruption 예외가 없어야 한다. inference 입력에 target별 padding mask가 없고 최종 decoder가 true length 없이 동일 verdict를 내는지도 검사한다.

## 20. G2 — Candidate-Budget Fairness

비교되는 main, shuffled, baseline은 같은 test targets, q, source, length range, K, independent verifier, evaluation rule을 사용한다.

$$
K_{\mathrm{actual}}=K_{\mathrm{declared}}.
$$

불일치, target 누락, attempt order 변경, invalid/duplicate 제외, valid까지 추가 sampling은 G2 FAIL이다. prefix monotonicity, target별 actual count, decoded bytes에 대한 actual hash call count를 감사한다. completed run의 성공 후보만 남겨 비교하지 않는다.

G0/G1은 hash run에 진입하기 전에 확인하고, G2는 pipeline validator를 사전 검증한 뒤 실제 완료된 각 run에도 적용한다. incomplete run은 비교에서 성공/실패로 채우지 않고 미완료로 공개한다.

## 21. Statistical Validation — staged G3

### 21.1 Paired target effects

고정 setting·seed·K에서 target i의 main, random, shuffled binary outcome을 각각 $M_i,B_i,S_i$라 한다. 각 rate는 해당 binary outcome의 N-target 평균이다.

$$
\Delta_{\mathrm{random}}=P_{\mathrm{main}}-P_{\mathrm{random}},\qquad
\Delta_{\mathrm{shuffled}}=P_{\mathrm{main}}-P_{\mathrm{shuffled}}.
$$

두 비교 모두 exact one-sided McNemar test를 사용한다. 비교 상대를 $C_i\in\{B_i,S_i\}$라 하고 $n_{10}=\#\{M_i=1,C_i=0\}$, $n_{01}=\#\{M_i=0,C_i=1\}$, $m=n_{10}+n_{01}$라 하면:

$$
p=\Pr[Z\ge n_{10}],\qquad Z\sim\operatorname{Binomial}(m,1/2).
$$

$m=0$이면 p=1이다. target의 paired outcome을 unit으로 하는 **10,000-resample percentile paired bootstrap 95% CI**를 계산한다. bootstrap seed·quantile convention은 첫 test 전에 freeze한다. NK candidates나 3N seed-target rows를 독립 표본으로 삼지 않는다. 독립 binomial CI의 overlap로 paired test를 대체하지 않는다.

### 21.2 Preregistered comparison families

| Family | 구성 | 역할 |
|---|---|---|
| Primary seed-0 family | 4 core IDs × q=12/16 × K=100 × random/shuffled = **16 one-sided comparisons** | 전체 core feasibility screening에 공동 Holm correction |
| Auxiliary seed-0 family | 4 IDs × q=12/16 × K=1/10 × random/shuffled = **32 comparisons** | K 의존성의 보조 분석, 별도 공동 Holm correction |
| q=8 | 모든 K의 rate/effect/CI 및 sanity 결과 | P1 진입이나 강한 feasibility claim에 사용하지 않음 |
| Seed 1/2 | 사전 규칙으로 선택된 동일 setting의 두 비교, 모든 K | seed별 raw McNemar/effect/CI와 재현 방향 보고 |

각 seed-0 family의 모든 raw p-value에 Holm을 공동 적용한다. 불리한 setting/control을 family에서 제거하지 않는다. family 일부가 invalid/blocked이면 그 원인을 공개하고 완전한 family에 대한 adjusted superiority 판정은 보류한다. 미실행 outcome을 만들어 채우지 않는다.

G3의 **통계적 지지가 있는 signal**은 K=100에서 두 비교 모두 Holm p<0.05와 marginal 95% CI lower bound>0인 경우로 별도 표시한다. 그러나 PoC P1/복제 진입은 §22–23의 양의 효과 기준이며 CI/p 기준을 사후 필수로 바꾸지 않는다. marginal CI를 simultaneous CI라고 하지 않는다.

선택된 seed 1/2는 재현성 분석으로 보고하며 seed 중 가장 좋은 것을 primary로 바꾸지 않는다. 선택된 setting만 복제한 사실과 seed 0의 selection 영향을 공개한다. 새 independent holdout에 대한 confirmatory claim은 별도 연구가 필요하다.

Gaussian–Discrete secondary question은 동일 source/q의 paired rate difference와 CI를 기술적으로 보고한다. 이 PoC에서 별도 model-ranking significance claim을 추가하지 않는다. source 간/q 간 결과 차이는 target composition·N·source entropy와 함께 해석한다.

### 21.3 Small N and zero success

absolute gain, $N\Delta$ additional solved targets, $n_{10}/n_{01}$, raw/adjusted p, paired CI를 기록한다. baseline rate가 0이면 relative gain은 NA이며 무한 개선이라고 표현하지 않는다.

0/N 성공은 확률 0이나 이론적 불가능성을 뜻하지 않는다. target-level binomial sampling 가정 아래 one-sided 95% upper bound를 참고로 보고한다.

$$
p_{\mathrm{upper}}=1-0.05^{1/N}\approx3/N.
$$

finite digest group holdout과 target별 확률 이질성 때문에 population 해석은 제한됨을 병기한다. N=0이면 평가 불가다. 모두 동일한 paired outcome에서 bootstrap CI가 [0,0]이어도 효과의 부재나 동등성 증명이 아니다. actual unique N과 가능한 discordance 수에 따른 power/MDE 한계를 train/validation 또는 독립 simulation으로 검토하고 test를 본 뒤 N을 늘리지 않는다. PoC의 목적은 full confirmatory evidence가 아니라 feasibility signal 평가다.

## 22. Seed Strategy — staged G4

### Stage A

네 core setting을 q=8/12/16에서 모두 **model seed=0**으로 실행한다. 각 main과 matched learned shuffled control이 포함된다. G0/G1 실패 family는 blocked로 남기고 실행을 강행하지 않는다. q=8은 sanity이며 seed 확장 기준으로 사용하지 않는다.

### Stage B

복제 진입 budget은 사전 선택된 **K=100**이다. q=12 또는 q=16의 setting이 아래 조건을 모두 만족할 때만 main/shuffled를 model seed 1과 2로 확장한다.

$$
G0\land G1\land G2\ \mathrm{PASS},\qquad
\Delta_{\mathrm{random},0,100}>0,\qquad
\Delta_{\mathrm{shuffled},0,100}>0.
$$

이는 positive paired effect가 두 비교에서 관측됐다는 뜻이다. CI가 0을 걸쳐도 사전 point-estimate 규칙에 따라 복제할 수 있고 uncertainty를 함께 보고한다. K=1/10에서만 양수인 setting은 primary 복제에 진입하지 않는다.

eligibility를 모든 setting에 기계적으로 적용하고 전체 결정표를 저장한다. seed 1 결과가 불리해도 seed 2를 생략하지 않는다. target, baseline stream, configuration, training/checkpoint rule과 sampler를 변경하지 않는다. 누락·resource interruption은 INCOMPLETE이며 P2를 부여하지 않는다. model seed, sampling seed, dataset seed, canonical baseline evaluation seed, bootstrap seed는 별도 namespace로 관리한다.

## 23. P0 / P1 / P2 Evidence Levels

| Level | 사전 조건 | 허용 해석 |
|---|---|---|
| P0 — Pipeline Feasible | G0 ∧ G1 ∧ G2 PASS | hash-conditioned experiment를 유효하게 실행할 수 있음. hash advantage evidence는 아님 |
| P1 — Hash-conditioned Signal | P0 + q=12/16, K=100, seed 0에서 main−random>0 및 main−shuffled>0 | hash condition이 candidate generation에 영향을 줄 가능성이 관찰됨 |
| P2 — Replicated Feasibility | 동일 P1 setting에서 seeds 0/1/2 모두 두 효과>0, 각 run prerequisite 충족 | truncated MD5 PoC의 hash-conditioned candidate-generation advantage가 여러 model seed에서 재현됨 |

P1은 paired effects와 95% CI를 반드시 병기하고 가능하면 두 lower bound>0을 확인한다. point-estimate signal과 통계적 지지가 있는 signal은 구분한다. P2도 seed별 CI를 함께 보고하고 세 seed 모두 lower bound>0인 경우 추가로 명시할 수 있다. 이를 P2의 사후 필수조건으로 추가하지 않는다.

G0/G1/G2 실패는 INVALID이며 P0도 부여하지 않는다. prerequisite PASS 후 signal이 없으면 P0와 “해당 setting에서 signal 미관측”을 기록한다. 미실행은 level 없음/NOT_EVALUATED다. q=8의 높은 rate만으로 P1을 부여하지 않는다. 어떤 PoC level도 full MD5 inversion evidence가 아니다.

## 24. Hyperparameter Freeze and Required TBD

### 24.1 Validation-only selection policy

Phase P0에서 먼저 허용 search space와 finite maximum trials를 등록하고, train/validation development 이후 **첫 hash test 접근 전에** 전체 q=8/12/16 matrix의 configuration/selection rule을 seal한다. model family별 표에 빈 실행값이 남으면 test로 진입하지 않는다.

hash configuration의 validation metric은 **validation unique digest targets의 PreimageSuccess@100**이다. 최대값을 선택하고 동률이면 적은 sampling steps, 다시 동률이면 사전 등록 configuration ID 순서로 선택한다. checkpoint도 사전 등록 update 후보에서 같은 기준으로 선택하고 같은 config/steps의 동률이면 가장 이른 update를 선택한다. secondary similarity는 selection metric으로 바꾸지 않는다. reversible-control development는 별도 held-out validation ExactRecovery@1 기준을 사용하며 hash test에 접근하지 않는다.

validation target/stream/seed, trial 정의, candidate·training budget을 고정한다. 각 source/q의 config selection에 쓰는 allowed space와 trial cap의 적용 단위까지 등록하며 실패 trial도 cap을 소비한다. 재현 seed에서는 새 search를 하지 않는다. shuffled에는 main에서 선택된 config와 training update budget을 복제한다.

### 24.2 Model-family search specifications

| 항목 | Gaussian BGV | Discrete sequence |
|---|---|---|
| 재사용 우선순위 | 기존 U-Net/Gaussian 및 G1 검증 설정, BGV 31-byte codec | 기존 TokenCodec/SequenceDenoiser/absorbing-mask primitives; scientific 검증 완료로 간주하지 않음 |
| Search space | 기존 설정의 재사용 적합성 확인 후 architecture/parameterization/schedule/steps/optimizer 허용값 목록 **TBD — freeze before test** | architecture/embedding/steps/temperature/optimizer 허용값 목록 **TBD — freeze before test** |
| Maximum trials | 정확한 finite 수와 source/q별 cap **TBD — freeze before test** | 정확한 finite 수와 source/q별 cap **TBD — freeze before test** |
| Validation metric | PreimageSuccess@100 | PreimageSuccess@100 |
| Selection / tie-break | §24.1의 최대값 → 적은 steps → config ID → 이른 checkpoint | 동일 |
| Prediction / schedule | 최종 x0/epsilon parameterization, Gaussian noise schedule, reverse rule **TBD**; 기존 설정 근거 제시 | clean categorical prediction, $\rho(t)=t$, continuous $t\sim U(0,1)$ 고정 |
| Sampling | Gaussian initial noise; steps, 수치처리 최종값 **TBD** | all-MASK initial state, random reveal, no remasking/repair 고정; reverse grid/steps/temperature 최종값 **TBD** |
| Training / checkpoint | optimizer, lr, batch, update candidates 및 평가 간격 **TBD** | optimizer, lr, batch, update candidates 및 평가 간격 **TBD** |

TBD는 무제한 tuning 허용이 아니다. 숫자를 임의로 만들지 않은 **실행 blocker**다. sampling steps 등 search는 등록한 좁은 목록 안에서만 가능하다. 검증된 Gaussian 설정이 그대로 적용되면 불필요한 architecture search를 추가하지 않는다. positive control 미달을 본 뒤 hash test를 조회하며 구조를 바꾸는 것은 금지한다.

q=8을 configuration debugging에 사용한 run은 EXPLORATORY로 표시한다. 수정 전 실패 artifact를 보존하고, 해당 test를 보고 변경한 설정은 같은 test의 유효 PoC evidence로 재사용하지 않는다. q=12/16 test를 열기 전 새 protocol/config version과 오염되지 않은 holdout을 확보한다.

### 24.3 Phase P0에서 해결할 specification / implementation blockers

| 항목 | 현재 상태 / 결정 또는 검증할 내용 |
|---|---|
| Seeds / data | dataset/split/representative, sampling, shuffle, baseline evaluation, bootstrap, MC seed 및 construction budget TBD; 기존 fixture seed 자동 채택 금지 |
| Gaussian config | 기존 control 설정의 적용 근거 있음; hash canonical-bit 최종 config/search space/trial cap은 TBD |
| Discrete config | continuous t corruption은 현재 discrete-index 구현과 차이; optimizer/lr/steps/temperature/search cap/selection 등록 필요 |
| Positive control | Gaussian 기존 16 held-out targets의 PoC 적용 확인, Discrete condition·target 수·반복 정책 TBD; source/config/seed coverage 등록 필요 |
| Statistics | 절차·family 구조는 §21 고정; RNG 수치/quantile version와 단계적 seed 보고 경로 freeze 필요 |
| Baseline expectation | 실제 D 기반 Monte Carlo draw budget·seed·uncertainty implementation 최종 선택 필요 |
| Scientific execution path | 최신 audit의 engineering-only runner를 그대로 과학적 실행으로 사용하지 않음; external exact-quota data, G0/G1 guard, validation checkpoint selection, secondary metrics 통합 필요 |
| Stage-aware statistics / resume | 현재 all-three-seeds aggregate 경로와 조건부 복제의 차이 처리, per-attempt interrupted sampling 복구 검증 필요 |
| Compute/storage | 접근 가능한 device, training/NFE 상한, control/tuning 비용, output retention, storage 한도 TBD |

이 표는 새 구현 작업의 필요사항을 밝힌다. **이번 작업에서 이 기능을 구현하거나 검증 완료로 표시하지 않는다.** 기존 [specification template](local_experiment_archive/retired_2026-09-21/artifacts/config/CONFIRMATORY_SPECIFICATION_TEMPLATE.json)은 Full Plan용이므로 해당 null을 임의로 채우거나 그 파일을 PoC용으로 덮어쓰지 않는다.

## 25. Execution Phases

아래는 **향후 실행 순서**이며 본 문서 작성 중 실행한 작업 목록이 아니다.

| Phase | 수행할 일 | 다음 단계 조건 |
|---|---|---|
| Phase P0 — Specification Freeze | L_max=31, 두 source/length law, q, K, dataset/evaluation/model seed policy, Gaussian/Discrete config/search space/trial caps, controls, statistical procedure와 자원 envelope 등록 | §24의 필수 TBD 해소; development 허용 범위 먼저 등록하고 test 전 최종 config seal |
| Phase P1 — Pipeline Validation | Pilot dataset generation, split audit, BGV/tokenizer round-trip, MD5 verifier sanity, Gaussian/Discrete positive controls, source-prior sanity, leakage/budget validator 검증 | 해당 family의 G0/G1 PASS; 공통 실패 시 영향받는 모든 family 중단 |
| Phase P2 — q=8 Sanity | 네 core × MD5 q8 × seed 0의 main/shuffled, shared source-prior, K prefixes 평가 | end-to-end 실행, conditioning, rehash, evaluator, budget 검증; 높은 성능 자체는 강한 evidence 아님 |
| Phase P3 — q=12 / q=16 Feasibility | 네 core의 seed 0 main/shuffled 실행 및 baseline 비교, paired 통계 | q≥12/K100의 사전 P1 eligibility 결정표 작성 |
| Phase P4 — Seed Replication | eligible setting만 seed 1/2의 main/shuffled를 고정 protocol로 실행 | 두 추가 seeds 모두 완료 후 동일 K100의 P2 여부 판정 |
| Phase P5 — PoC Final Report | 모든 planned IDs의 결과/실패/미실행 사유, actual N, costs, limitations, evidence level 보고 | 성공/실패 모두 해석 제한과 후속 조건 명시 |

Phase P0는 development의 search 공간과 procedure를 먼저 고정한다. Phase P1의 train/validation 및 control 결과에 따른 최종 선택은 그 절차 안에서 수행하며 Phase P2의 첫 hash test 전에 선택 결과까지 seal한다. 이후 seed별 validation checkpoint selection이 필요하면 이미 고정한 규칙과 update 후보만 적용한다.

q=8 sanity에서 test를 본 뒤 pipeline 수정이 필요하면 해당 run은 invalid/exploratory로 보존하고 새 freeze/holdout 절차를 거친다. G0/G1 실패를 무시하고 q=12/16으로 이동하지 않는다. 모든 설정이 P1 진입에 실패하면 Phase P4는 NOT_RUN이며 Phase P5로 간다. resource interruption과 과학적 signal 미관측을 구분한다.

## 26. Resource Plan

### 26.1 Training jobs

하나의 hash training job은 한 core ID × q × model seed × main/shuffled model 학습이다. K=1/10/100은 같은 checkpoint의 prefix 평가이며 학습 세 번이 아니다. 아래는 **계획 수량**이며 완료된 jobs가 아니다.

$e_G,e_D$는 q=12/16의 eligible Gaussian/Discrete setting 수이며 각각 0–4, $e=e_G+e_D\le8$이다.

| 항목 | Gaussian jobs | Discrete jobs | 합계 |
|---|---:|---:|---:|
| Stage A main, seed 0 | 2 sources × 3 q = 6 | 2 × 3 = 6 | 12 |
| Stage A shuffled, seed 0 | 6 | 6 | 12 |
| Stage A hash subtotal | 12 | 12 | **24** |
| Stage B main, seeds 1/2 | $2e_G$ | $2e_D$ | $2e$ |
| Stage B shuffled, seeds 1/2 | $2e_G$ | $2e_D$ | $2e$ |
| Stage B hash subtotal | $4e_G$ | $4e_D$ | **$4e\le32$** |

따라서 hash jobs는 **$24+4e\le56$**이다. prerequisite 실패로 실행하지 못한 row는 planned count에서 지우지 않고 blocked로 보고한다.

positive-control 학습 수를 $J_{\mathrm{PC},0}$, replication 관련 추가 control 수를 $J_{\mathrm{PC},B}$, 선택된 hash jobs 외의 tuning/development 학습 수를 $J_{\mathrm{tune}}$라 하면:

$$
J_{\mathrm{total}}=24+4e+J_{\mathrm{PC},0}+J_{\mathrm{PC},B}+J_{\mathrm{tune}}.
$$

controls는 source × family별 최소 네 개의 최종 held-out 검증이 필요하지만 이는 training job 네 개와 같지 않다. **기존 Gaussian 전체 ladder를 재사용하면 source당 5 jobs**(train 1/4/16/64와 별도 held-out stage)다. 두 source에서 같은 q 공통 configuration의 ladder를 수행하는 경우 seed 0 Gaussian positive-control jobs는 10개다. Discrete source별 control training job 수를 $d_P,d_R$라 하면 그 경우 $J_{\mathrm{PC},0}=10+d_P+d_R$이며 $d_P,d_R$는 아직 TBD다. Discrete를 source당 한 held-out 학습으로 확정하는 경우에만 12개가 된다. 이를 현재 확정된 총수로 쓰지 않는다.

q별 configuration이 달라 control을 공유할 수 없거나 replication seed별 control이 필요하면 추가한다. 공유 가능한 source/config별 기존 Gaussian 5-job ladder를 두 추가 seed에 적용하는 경우 해당 source당 10 jobs가 추가된다. Discrete 추가 controls도 같은 coverage 원칙으로 계산한다. control coverage, search cap이 freeze되기 전 GPU-hours 전체값은 산정 완료가 아니다. 선택 trial을 최종 학습으로 재사용하면 $J_{\mathrm{tune}}$에 중복 계상하지 않는다.

### 26.2 K=100 generation volume

$$
U=\sum_{d\in\{\mathrm{Printable},\mathrm{Bytes}\}}
\sum_{q\in\{8,12,16\}}N_{d,q}.
$$

한 method/setting/seed는 $100N_{d,q}$ attempts를 만든다. baseline은 두 모델·모든 seed에 공유한다.

| 범위 | Candidate attempts |
|---|---:|
| Stage A main | $200U$ |
| Stage A shuffled | $200U$ |
| Shared source-prior, 여섯 source/q streams | $100U$ |
| Stage B main + shuffled | $400\sum_{h\in\mathcal E}N_h$ |

$\mathcal E$는 eligible core/q settings 집합이며 $N_h$는 해당 source/q의 actual unique N이다. K=1/10은 prefix 재집계이므로 $111N$으로 계산하지 않는다.

느슨한 상한은 $U\le2(256+1{,}000+1{,}000)=4{,}512$다. q=8의 train/validation group 소유권을 무시한 보수적 상한이므로 실제 test N의 예측값이 아니다.

- Stage A main+shuffled: 최대 1,804,800 attempts.
- shared baseline: 최대 451,200 attempts; seed 1/2에서 재생성하지 않음.
- Stage B main+shuffled: 최대 3,200,000 attempts.
- 위 hash evaluation+baseline 전체: 최대 **5,456,000 attempts**.

positive controls, validation selection, tuning, independent MC, exhaustive sanity는 이 합계 밖의 별도 비용이다. 각 항목의 actual generation/hash counts를 보고한다. optional q20은 별도 등록 시 seed 0 main 4+shuffled 4=8 jobs 및 $500\sum_dN_{d,20}$ evaluation attempts를 추가하며, 복제 여부/예산은 그 stretch protocol에서 별도로 정한다.

### 26.3 NFE, GPU-hours, memory

NFE는 denoiser forward 평가 횟수다. job j의 candidate 수를 $A_j$, candidate당 forward 수를 $F_j$라 하면 candidate 단위 NFE 합계는:

$$
\mathrm{NFE}_{\mathrm{candidate}}=\sum_j A_jF_j.
$$

job j의 reverse step 수를 $J_j$라 하면 각 step에 forward 한 번인 현재 기본 sampler는 $F_j=J_j$이며 실제 호출 수를 기록한다. batch 크기 $B_j$로 묶는다면 framework의 batched forward calls는 대략 $\lceil A_j/B_j\rceil F_j$이지만 target별 batching 정책·마지막 batch 때문에 actual count도 남긴다. NFE가 같아도 두 architecture의 FLOPs가 같은 것은 아니다. primary에는 추가 guidance/reranking forward를 숨기지 않는다.

Gaussian/Discrete별 학습 시간은 update 수 × 측정한 update latency, sampling 시간은 batch forward calls × 측정한 batch latency로 추정하고 실제 시간과 비교한다. controls/tuning/MC/verifier 시간을 합산한다. **steps·training budget·hardware가 미확정이므로 현재 expected NFE/GPU-hours의 수치값은 TBD**이며 위 산식에 frozen 값과 validation-only timing 측정치를 넣어 test 전에 자원 계획을 완성한다.

required GPU model/count/VRAM, 사용할 precision, batch size, peak memory, wall-clock 상한은 TBD다. 기존 보고서의 CPU-only 접근 결과는 역사적 관측이며 현재 hardware 가용성으로 단정하지 않는다. 기존 1-step smoke를 학습 수렴 시간이나 성능 추정으로 외삽하지 않는다.

### 26.4 Storage

| Raw artifact | Candidate당 uncompressed 크기 |
|---|---:|
| Gaussian BGV float32 `[2,32,128]` | 32,768 bytes |
| Discrete length-32 uint16 tokens | 64 bytes; 259 states도 보존 가능 |

두 family의 Stage A main+shuffled는 각각 $200U$ outputs다. 위 느슨한 N 상한에서 Gaussian raw outputs는 약 **29.57 GB**다. 모든 q12/16 setting을 replication하면 Gaussian 최대 1,600,000 outputs가 추가되어 약 **52.43 GB**, 합계 약 **82.00 GB**다. 대응 Discrete raw outputs는 전체 약 **0.160 GB**다. GB는 $10^9$ bytes 기준이다.

이는 hash main/shuffled raw tensors만의 상한이며 checkpoint/optimizer state, candidates/metrics JSONL, baseline, dataset, control, tuning, backups는 추가다. float tensor를 JSON pixel 배열로 저장하면 더 커진다. lossless 압축 또는 정확한 replay 정보 보존 정책을 Phase P0에서 등록하고 측정한 크기와 disk 여유로 실제 quota를 정한다. 성공 flag만 남기거나 저장 공간을 이유로 invalid 원본을 선택적으로 버리지 않는다.

전체 Full Plan의 수백 training runs를 바로 실행하지 않고, 이 작은 matrix의 prerequisite·signal·cost를 먼저 확인하는 것이 PoC의 핵심이다. same K 결과를 compute 효율 우위로 해석하지 않는다.

## 27. Interpretation Rules

결론마다 source/length law, MD5, q, K, approach, model seeds, actual unique N, gates와 control 결과를 명시한다. 아래 허용 문장은 실제 해당 기준을 충족한 뒤에만 사용할 수 있는 **보고 예시**다.

| 허용 | 조건 / 제한 |
|---|---|
| “truncated MD5 PoC에서 hash-conditioned generation signal이 관찰되었다.” | P1 setting과 effect/CI를 구체적으로 명시 |
| “source-prior와 shuffled-condition에 대한 positive effect가 관찰되었다.” | 두 paired effects 모두 양수, 불확실성 병기 |
| “동일 효과가 여러 model seeds에서 재현되었다.” | 동일 setting/K의 seeds 0/1/2, P2 충족 |

다음 표현은 금지한다: “MD5 was inverted.”, “Diffusion can invert cryptographic hashes.”, “SHA-256 inversion is feasible.”, “full-digest inversion was demonstrated.” q8/12/16 결과를 full MD5와 같은 evidence로 다루지 않는다.

q=8의 높은 성공률, valid decode 증가, source similarity, 두 approach의 순위, positive-control PASS만으로 hash advantage를 주장하지 않는다. main≈shuffled이면 condition 사용 evidence가 부족하다. P2 역시 동일 dataset에서의 model-seed robustness이며 독립 dataset replication이나 cryptographic complexity 감소 증명이 아니다.

PoC에서 재현 가능한 advantage가 없으면 다음과 같이 보고한다.

> 해당 PoC dataset size, model configuration, source distribution, q, K에서 reproducible hash-conditioned advantage를 관찰하지 못했다.

이는 일반적인 Diffusion/hash inversion 불가능성 증명이 아니다. G1 failure이면 우선 pipeline의 검증 실패이고 hash signal 부재로 해석하지 않는다. 미실행/blocked에 대해서는 관측 실패라는 문장도 사용하지 않는다.

## 28. Expansion Criteria

**P2가 관찰된 setting**을 근거로 별도 full research phase를 설계할 수 있다. 자동 실행을 뜻하지 않는다. 새 preregistration, validation-only selection, 오염되지 않은 holdout, G0–G4, 더 강한 baselines 및 자원 계획을 준비한다. PoC의 selected-setting effect를 독립 confirmatory 결과로 재사용하지 않는다.

권장 확장 순서:

1. MD5 q=20.
2. MD5 q=24.
3. MD5 q=32.
4. MD5 q=64.
5. full MD5 q=128.
6. CGGE representation ablation.
7. Known-length extension 및 length-aware baselines.
8. stronger non-diffusion baselines, direct conditional predictor 포함.
9. SHA-256 replication.
10. full SHA-256 q=256.

더 긴 message는 별도 Length Scaling Experiment로 진행하고 encoding 변경을 명시한다. PoC 실패는 이 roadmap이 불가능함을 증명하지 않으며, 새 탐색을 할 때에도 기존 결과·기준을 소급 변경하지 않는다. Full Plan의 full-digest 의무와 claim 기준은 PoC 완료만으로 충족되지 않는다.

## 29. Required Artifacts

아래는 **향후 실행에서 생성해야 할 산출물**이다. 현재 CLI가 모두 생성한다고 주장하지 않는다. 새 PoC namespace를 사용하고 기존 `output/`, Full Plan state 및 frozen artifacts를 덮어쓰지 않는다.

| Artifact | 필수 내용 |
|---|---|
| Protocol / freeze | PoC 문서 version, comparison/eligibility manifest, code commit, immutable config hash/timestamp, package/device, test access log |
| Search provenance | family별 space/trial cap, 모든 trial/실패, validation metric, 선택·동률 결정, checkpoint/update, config seal |
| Dataset / split | source/length law, raw bytes 또는 복원 가능 자료, raw draw/duplicate/unused 수, quota, group owner, 모든 pairwise overlap, seeds, representatives, **split별 unique N** |
| Model / condition / codec | core ID, q/full width/truncated label, canonical bits format, shape/token IDs/version, PAD/0x00 분리, architecture/parameter count, optimizer/schedule/loss |
| Sampler / budget | initial state, reverse grid/steps/temperature, seed/RNG namespace, decoder thresholds, declared/actual K, sampling resume provenance |
| Candidate ledger | target/attempt ID, raw output 또는 exact replay 자료, decoded bytes, invalid reason, duplicate flag, candidate length, independently rehashed full MD5/null, prefix/match flags |
| Outcome ledger | target별 ordered K-prefix success, main/random/shuffled binary outcomes, source/length diagnostics, actual attempts/hash calls |
| Baseline / controls | canonical baseline stream ID/seed, actual D expectation 및 MC uncertainty/cost, shuffle donor mapping, actual-model positive control counts/CI, leakage tests |
| Statistics / evidence | N, successes, n10/n01, deltas/CI, raw/Holm p, family IDs, bootstrap seed/version, zero-success bound, G0–G4 세부 상태, P0/P1/P2 |
| Replication / resources | 모든 setting eligibility, seed 1/2 실행/미실행 사유, per-seed result, NFE/time/memory/storage, planned/completed/blocked/invalid 상태 |
| Final report | 모든 core/q/K 결과표, matched-target K curve, actual N을 병기한 q curve, limits, 성공·실패 해석, 확장 조건 |

machine-readable JSON/JSONL/CSV와 사람이 읽는 요약을 함께 보존한다. statistical row의 N은 message quota나 candidate count가 아닌 unique target 수다. evaluator-only source/full digest 정보는 generation input과 분리한다. primary candidate 재생성 없이 prefix metric을 재집계할 수 있어야 한다.

## 30. Document Consistency Review

아래는 **문서 명세의 일관성 검토**이며 실제 PoC gate 검증 결과가 아니다.

| 검사 | 명세 상태 |
|---|---|
| 최종 연구 목표와 PoC feasibility 구분 | 유지; full inverse 주장 금지 |
| Source/length/shape | 94/256 iid, 같은 U{4,…,31}, BGV `[2,32,128]`, sequence 32 일치 |
| Discrete tokens/process | 97/259, PAD≠0x00, EOS/PAD masking, continuous t 명세와 구현 gap 명시 |
| Conditioning/leakage | 두 approach 동일 canonical bits, Hash-only, padding attention mask/true-length correction 금지 |
| Dataset/evaluation unit | Pilot quota와 actual unique N 구분; raw/digest 모든 split pair 서로소 |
| K/baseline/verifier | ordered prefix, invalid/duplicate 소비, shared canonical baseline, actual D, independent full MD5 후 prefix 판정 |
| Controls/gates | codec와 learned control 분리; G0–G2 prerequisite, old gate labels 전용 금지 |
| Statistics/seeds | paired target, 10,000 bootstrap, Holm 16/32 family, K100 사전 replication, seed pooling 금지 |
| Evidence/phase 구분 | P0–P2 feasibility와 execution Phase P0–P5 분리, q8은 sanity |
| Resource arithmetic | Stage A 24 hash jobs, Stage B ≤32, PC/tuning 별도; prefix 중복 계산 없음 |
| Implementation/status | 기존 engineering 자산, 실제 legacy 결과, 미구현 gap, 새 PoC 미실행을 구분 |
| TBD/limits | 실행 전 사양 결정과 측정이 필요한 값 공개; 예상 결과를 실제 결과로 쓰지 않음 |

## 31. Differences from the Full Plan

아래 Full Plan은 원본 목표와 **Gaussian/Discrete 확장 계획**을 함께 가리킨다. 특히 `Discrete: Yes`는 [RESEARCH_PLAN_GAUSSIAN_DISCRETE.md](RESEARCH_PLAN_GAUSSIAN_DISCRETE.md) 기준이다. 원본 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)의 Direct Bits를 categorical Discrete와 동일한 모델로 소급 해석하지 않는다.

| 항목 | 기존 Full Plan | PoC Plan |
|---|---|---|
| Algorithms | MD5 + SHA-256 | MD5 |
| q | low q ~ full digest | 8, 12, 16 (+ optional 20, 기본 비활성) |
| Data | Pilot + Main | Pilot only |
| Gaussian repr. | BGV + CGGE | BGV |
| Discrete | Yes — Gaussian/Discrete 확장 계획 기준 | Yes |
| Known-length | Yes | Deferred |
| Full digest | Mandatory | Deferred |
| Seeds | 0,1,2 | seed 0 first, conditional 1/2 at q12/16, K100 |
| Goal | Confirmatory evidence | Feasibility signal |
| Primary baseline | source-prior + 적용 가능한 direct predictor | source-prior only; shuffled는 필수 negative control |
| Evidence | full G0–G4 및 L0–L4 | G0–G2 필수, staged G3/G4, PoC P0–P2 |
| Length | 원본 31; 확장 계획의 새 L_max는 TBD | 31 고정, length scaling deferred |

## 32. Pre-run Checklist

아래 checkbox는 **새 PoC의 실행 준비 기록**이며 이 문서 작성만으로 체크하지 않는다.

- [ ] `L_max = 31` frozen.
- [ ] Printable distribution frozen: ASCII 0x21–0x7E, 94 iid symbols, space 제외.
- [ ] Random Byte distribution frozen: 0x00–0xFF iid, 동일 uniform length law.
- [ ] MD5 verifier validated.
- [ ] q = 8/12/16 frozen; optional q20의 비활성/별도 등록 상태 명시.
- [ ] dataset seed frozen; split/representative seed와 construction budget 포함.
- [ ] baseline evaluation seed frozen; model seed 독립, 두 model과 모든 seeds에 canonical stream 재사용.
- [ ] BGV codec round-trip 100%.
- [ ] Discrete tokenizer round-trip 100%.
- [ ] PAD/0x00 separation verified.
- [ ] Hash-only length leakage test passed; EOS/PAD corruption, 32 positions, padding attention mask 없음.
- [ ] Continuous t ~ U(0,1) forward와 masked-position loss/빈 mask 처리가 명세와 일치.
- [ ] Gaussian positive control passed; 정확한 target 수·condition·seed coverage 기록.
- [ ] Discrete positive control passed; target 수·condition·반복 정책 TBD 해소.
- [ ] digest-group split audit passed; raw-message 포함 validation까지 모든 pair overlap=0.
- [ ] actual test N을 unique digest target 기준으로 기록; message quota와 구분.
- [ ] candidate-budget validator passed; K1/10/100 prefix monotonicity 확인.
- [ ] invalid/duplicate candidate accounting verified; 추가 sampling/성공 후 조기종료 없음.
- [ ] shuffled-condition control configured; learned path와 digest derangement 감사 준비.
- [ ] independent rehash verifier configured; bytes 없는 invalid의 digest=null 및 실제 hash counts 기록.
- [ ] hyperparameter search space frozen; family별 finite maximum trials, validation metric/selection/tie-break 포함.
- [ ] Gaussian parameterization/noise schedule/steps와 Discrete reverse grid/steps/remasking rule/temperature/optimizer/lr/checkpoint freeze.
- [ ] statistical procedure frozen; 16/32 Holm family, bootstrap seed/quantile, paired target unit 포함.
- [ ] Stage B q12/16·K100 eligibility rule frozen; seed 1/2 모두 실행·보고, seed pooling 없음.
- [ ] source-prior expectation은 실제 D로 산정; MC budget/seed/uncertainty 별도 고정.
- [ ] scientific runner의 G0/G1 guard, external data, validation selection, secondary diagnostics, partial-seed 통계와 resume gap 해결.
- [ ] positive-control/tuning 포함 training jobs, NFE, GPU/memory/storage envelope와 retention policy 확정.
- [ ] immutable config snapshot과 test access log 준비.
- [ ] no test-set adaptation confirmed; q8 debugging은 exploratory로 구분.
- [ ] 이전 연구 문서·artifact를 보존하고 실제 실험은 별도 실행 작업으로 시작한다.
