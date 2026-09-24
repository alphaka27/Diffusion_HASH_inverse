# 연구 계획 v2: Diffusion 기반 truncated-MD5 preimage 후보 생성 PoC

**Protocol:** `dhi-poc-v2-20260923` · **작성일:** 2026-09-23  
**상태:** 설계 정의 완료 / 공학적 준비와 검증 진행 가능 / 확증적 hash 실행은 gate 충족 전 BLOCKED.

본 문서는 신규 v2 PoC의 기준 문서다. [v1](/Users/choisoonwook/Experiments_local/DHI_AI_gen/RESEARCH_PLAN.md) 및 이전 계획·실험은 이력으로 보존하며 v2의 미지정 값을 대신하지 않는다. 채택한 고정값은 [protocol JSON](/Users/choisoonwook/Experiments_local/DHI_AI_gen/examples/poc-v2-protocol.json)에 함께 기록한다. 이 JSON은 기존 experiment CLI에 바로 전달하는 실행 설정이 아니다. 문서와 JSON이 충돌하면 실행을 막고 둘을 일치시킨다.

본 문서 작성·설계 검증은 모델 학습이나 primary hash 평가를 의미하지 않는다. 측정 결과와 실행 준비도는 [v2 검증 보고서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/RESEARCH_PLAN_V2_VALIDATION.md)로 구분한다. 문서 검토 PASS는 과학적 gate PASS가 아니다.

## 1. 목적과 v1 대비 변경

핵심 질문은 다음과 같다.

> 지정한 source domain에서 학습에 사용하지 않은 12-bit MD5 digest만 조건으로 받은 모델이, 표적당 100번의 생성 기회 내에 유효한 preimage를 찾을 확률에서 source-prior random과 matched shuffled-training control을 모두 능가하고, 그 우위가 model seeds 0·1·2에서 반복되는가?

| 항목 | v2 결정 |
|---|---|
| 확증적 matrix | 다섯 pipeline × MD5 q=12 |
| Primary outcome | unique-target PreimageSuccess@100 |
| 보조 곡선 | 같은 stream의 Success@1, @10; 확증 family에 추가하지 않음 |
| Model seeds | 0, 1, 2를 모두 실행; 성능에 따른 seed 선택 없음 |
| 대조군 | source-prior random, 별도 shuffled-training model |
| 평가 표적 | source별 서로 다른 digest 2,048개; 같은 source의 pipeline은 같은 표적 사용 |
| 통계 family | pipeline별 6개 component의 max-p를 구성한 뒤 5개 composite에 Holm |
| Gaussian | T=1,000, linear beta .0001→.02, epsilon prediction, sampling 100 steps |
| Discrete | continuous uniform masking, 32-step reverse grid, temperature 1 |
| q=8 | 독립 engineering fixture와 개발 검증에 한정 |
| q=16 이상 | 별도 표본·예산·사전등록이 필요한 후속 연구 |
| 표현 비교 | BGV–CGGE 및 Gaussian–Discrete 모두 기술적·탐색적으로 보고 |

권장 수정안을 검증하는 과정에서 **두 source의 공통 ownership은 source별 ownership으로 보완**했다. 기존 MD5 q=12/16 평가·validation 이력의 12-bit prefix를 합쳐 제외하면 공통 새 표적 풀이 부족하기 때문이다. 같은 source 내부의 표적 공유는 유지한다. 배제 범위와 잔여 수는 §4 및 검증 보고서에 명시한다.

현재 범위의 성공은 해당 pipeline·source·표적 집합·학습 budget에 한정한 후보 생성 우위다. Full MD5 inversion, 임의 128-bit target, 암호학적 break, 계산량 우위, SHA-256, 알고리즘 계열 전체 우위로 확대하지 않는다. MD5 collision과 preimage는 다른 과제다.

## 2. Primary matrix와 실행 단위

| ID | Source | Representation | Model | q | Seeds |
|---|---|---|---|---:|---|
| P-G-BGV | Printable | BGV | Gaussian | 12 | 0/1/2 |
| P-G-CGGE | Printable | CGGE | Gaussian | 12 | 0/1/2 |
| P-DISC | Printable | Token | Discrete | 12 | 0/1/2 |
| R-G-BGV | Random Bytes | BGV | Gaussian | 12 | 0/1/2 |
| R-DISC | Random Bytes | Token | Discrete | 12 | 0/1/2 |

각 pipeline·seed마다 Main과 Shuffled를 학습하므로 primary learned hash runs는 5×3×2=30개다. Random은 source·seed당 한 stream을 만들어 같은 source의 pipeline에 공유한다. 학습한 positive control 15개, validation, engineering benchmark, 사전등록된 재현 점검은 이 30개에 포함하지 않는다. 실패·미완료 pipeline을 matrix나 family에서 삭제하지 않는다.

한 run은 protocol revision, source, pipeline, method, model seed, checkpoint, dataset/ownership hash, sampler config로 식별한다. 통계적 관측 단위는 **unique digest target**이다. 후보·token·pixel·seed-target row를 독립 표적 수로 늘리지 않는다.

## 3. Source와 hash

Printable alphabet은 0x21…0x7E의 94개 ASCII 문자로 space를 제외한다. Random Bytes는 0x00…0xFF 전체다. 두 source 모두 길이 L∼Uniform{4,…,31}, 길이가 주어졌을 때 payload symbol은 iid uniform이다.

$$P(x)=\frac1{28}A^{-|x|},\qquad A\in\{94,256\}.$$

Uniform lengths는 전체 가능한 문자열에 대한 uniform 분포가 아니다. NUL, 고위 byte, byte 순서를 손실 없이 보존한다. Domain validity는 원문과 동일한 길이를 요구하지 않고, 길이 4–31 및 해당 alphabet을 요구한다.

표준 MD5는 payload bytes에만 적용한다. Header, EOS, PAD, 이미지 pixel은 hash 입력에 포함하지 않는다. Standard serialized digest의 첫 12 bits를 byte 내 MSB-first로 취한다.

$$v(x)=\operatorname{Int}_{big}(\mathrm{MD5}(x))\gg116.$$

Model condition은 v의 정확히 12개 binary bits이며 leading zero를 보존한다. 로그는 lowercase hex 3자리다. 예를 들어 12-bit 경계는 첫 세 hex digits이며 내부 MD5 word endian을 재해석하지 않는다. [RFC 1321](https://www.rfc-editor.org/rfc/rfc1321)을 기준으로 독립 hashlib verifier와 알려진 test vectors를 확인한다.

## 4. 이전 노출 이력과 source별 ownership

### 4.1 배제 집합

Source s마다 `E_s`를 만든다. 알려진 이전 **학습 모델의 test/validation 평가** 중 MD5 q≥12 target은 그 첫 12 bits를 E_s에 넣는다. q<12 자료의 원문·full digest·모델 출력이 새로운 설계 선택에 사용됐는지도 별도로 감사한다. 순수 codec/verifier fixture와 실제 모델 성능 평가는 구분하고 분류 근거를 남긴다.

단순 파일 존재는 사람이 결과를 읽었다는 증거는 아니지만, 확증 holdout 선정에는 보수적으로 알려진 평가 target을 제외한다. 반대로 지정 폴더에서 찾지 못했다고 외부/삭제된 노출이 없었다고 단정하지 않는다. 기존 model weights, optimizer state, 원문 데이터, test-informed checkpoint는 v2로 이전하지 않는다.

본 검증에서 확인한 retired PoC의 MD5 q=12/16 test 및 validation metadata 범위에서는 Printable의 E 크기가 1,885, Random Bytes는 1,860이다. 남는 후보는 각각 2,211과 2,236개다. 두 source의 배제 집합 합집합은 2,915개로 공통 잔여 값은 1,181개다. 이 수치는 **제한된 알려진 이력의 audit**이며 최종 전체 접근 감사 완료를 뜻하지 않는다.

Same-source fresh target이라는 정의를 사용한다. 다른 source에서 본 동일 prefix까지 모두 배제한 project-wide novelty는 주장하지 않는다. Source별 배제로 가능한 연구 범위가 이 목적에 맞는지 freeze 기록에 명시한다. 최종 감사 후 어떤 source든 |{0,…,4095}\E_s|<2,048이면 v2의 확증 dataset 구성이 실패한다. 노출된 target을 몰래 채우거나 reseed로 해결하지 않는다.

### 4.2 결정적 ownership 알고리즘

1. 접근 감사 문서, metadata file hashes, E_s 목록을 봉인한다.
2. Available A_s={0,…,4095}\E_s를 정렬하고 source별 ownership RNG로 한 번 섞는다. 처음 2,048개를 test ownership으로 둔다.
3. Test 이외 2,048개 값을 정렬 후 별도 namespace RNG로 섞어 처음 1,536개를 train, 나머지 512개를 validation으로 둔다.
4. 모든 raw message와 digest는 정확히 하나의 owner에 속한다. 같은 source의 모든 pipeline·seed가 ownership 및 dataset을 공유한다.
5. Test와 E_s의 교집합 0, split 간 digest/raw message 교집합 0, 각 ownership 개수를 검사한다.

Train/validation에는 과거 노출 prefix가 포함될 수 있지만 원문은 새로 생성한다. 새로운 test는 가용 prefix 집합에 조건부로 선택되므로 전체 4,096개에 대한 무조건적 random sample이라고 부르지 않는다. Test 절반 확보는 train condition 다양성을 줄이는 선택이며, v1과 난이도가 동일하다고 가정하지 않는다.

## 5. Dataset 생성, seeds, 정보 경계

### 5.1 Corpus 생성

Source별로 원래 prior에서 draw를 반복한다. 길이는 `random.Random.randrange(4,32)`, Printable symbol은 `randrange(33,127)`, Random Bytes symbol은 `randrange(256)`으로 생성한다. Source RNG와 ownership RNG는 분리한다.

- Train 영역이면 최초 등장한 raw message를 draw 순으로 최대 10,000개 보관한다.
- Validation/Test 영역이면 해당 digest의 최초 원문을 representative로 보관한다. 각각 512개와 2,048개 digest를 모두 채운다.
- 중복, 이미 대표가 있는 digest의 추가 draw, train quota 초과를 각각 계수한다.
- Source별 draw cap은 1,000,000이다. 필요한 세 조건이 충족되면 중단한다. Cap 도달 실패는 construction failure다.

Train digest ownership은 1,536개이지만 실제 관측 digest 수는 이보다 작을 수 있다. 관측 수를 보고한다. 원문 10,000 / validation digest 512 / test digest 2,048은 서로 다른 단위다. Retained train은 ownership 및 unique-message 선택으로 조건화되었으며 원래 prior의 독립 draw라고 표현하지 않는다.

본실험용 corpus 생성은 protocol/노출 감사 봉인 후 수행한다. Engineering construction dry-run은 별도 seed를 쓰고 raw messages를 로그·보고서에 노출하지 않는다. Dry-run counts는 primary dataset이나 G0 certificate가 아니다.

### 5.2 Seed와 재현성

Model labels는 0/1/2, study master seed는 2026092302, engineering seed는 2026092399다. 하위 seed는 다음 UTF-8 문자열의 SHA-256 첫 8 bytes를 big-endian integer로 사용한다.

`protocol_id:master_seed:namespace:source:pipeline:method:model_seed:public_target:position`

해당하지 않는 필드는 빈 문자열로 명시한다. Namespace는 ownership-test, ownership-rest, source, train-order, train-noise, shuffle, validation-noise, generation, random-baseline, bootstrap, positive-control로 분리한다. 같은 architecture의 Main/Shuffled는 동일 초기 weight seed와 train order를 쓰되 pairing RNG는 별도다. Generation은 method별·target별 독립 stream을 사용한다. 후보의 hash 검증 결과, hidden length, 원문 record ID, 실행 순서는 seed에 포함하지 않는다.

여기서 source는 public domain label이며 public_target은 허용된 12-bit target이다. Seed에서 제외하는 hash result는 후보를 검증한 결과를 뜻한다. 원문 record ID는 사용하지 않는다. Engineering 전용 통계·timing 검사는 별도 namespace를 사용할 수 있다.

PRNG 구현·Python/NumPy/Torch 버전·precision·batch policy를 실행 manifest에 고정한다. GPU의 bitwise 재현성을 가정하지 않는다. 동일 bytes에 대한 deterministic verification과 stochastic retraining을 구분한다.

### 5.3 Model-facing contract

Generator가 받는 target-specific 정보는 12-bit vector뿐이다. Public source/shape/q, checkpoint, sampler config, 독립 난수는 허용한다. 원문, 길이, full digest suffix, EOS/PAD 위치, target image, 원문에서 만든 mask는 전달하지 않는다. Evaluator는 representative를 별도로 보관한다.

Digest·checkpoint·RNG를 고정한 상태에서 원문·길이·suffix·ID·padding metadata를 바꾸거나 제거해도 생성 결과가 같아야 한다. Batch sorting, attention mask, cache key, filename, recovery diagnostic도 감사한다. 기존 full-record 함수의 length flag를 끄는 것만으로 이 검증을 대체하지 않는다.

## 6. Representation와 decoder

| 항목 | BGV | CGGE | Token |
|---|---|---|---|
| 적용 source | 두 source | Printable | 두 source |
| Shape | [2,32,128] | [2,32,64] | [32] |
| Payload | MSB-first byte glyph | 고정 8×8 glyph | payload token |
| Length | 생성된 slot-0 byte | 생성된 contiguous validity mask | 생성된 EOS 위치 |
| Padding | decoded zero byte + invalid mask | invalid mask; unused glyph 값은 무시 | EOS 이후 PAD |

BGV는 기존 4×8 slots, 2×4 bit glyph, 각 bit의 4×4 확장을 채택한다. Header와 payload의 cell validity 평균은 ≥.5일 때 valid, bit block 평균은 ≥.5일 때 1이다. Header 길이 4–31, mask는 header와 정확히 L개 payload만 valid여야 한다. Padding glyph는 decoded byte가 0이어야 한다. 길이·mask 불일치, nonfinite, 잘못된 shape는 invalid다. Printable source membership은 별도로 검사한다.

CGGE는 `font8x8-basic-v1` embedded 94 glyphs를 채택한다. Table SHA-256은 `6ef6d0bfa6e29c823ad9ff7d6b4b29b9af5f739fee711268ff5b83ca7aeed50a`다. Runtime font rendering은 사용하지 않는다. Slot 0부터 payload를 넣고 32번째 slot은 unused다. Mask cell 평균 ≥.5, valid cells는 처음 L개 contiguous여야 한다. 유효 cell은 94 prototype과의 pixel MSE 최소값으로 decode하며 MSE>.1이면 invalid다. 정확한 tie는 ASCII 순서가 작은 prototype으로 결정한다. 이 규칙은 target 원문을 사용하지 않는다. Unused glyph를 무시하는 것은 명시적 decoder 정책이며 BGV와의 차이를 보고한다.

Gaussian clean image는 `z0=2*image-1`, inverse는 `(z+1)/2`다. 최종 sampler 출력 clipping은 [-1,1]이다. 모든 channel·header·validity·padding을 함께 noise 처리하고 생성한다.

Token은 Printable payload 0–93/EOS94/PAD95/MASK96, Random Bytes payload 0–255/EOS256/PAD257/MASK258이다. 길이 32, payload L개 뒤 EOS 하나, 그 뒤 PAD만 허용한다. MASK 잔존·다중/누락 EOS·unknown/noninteger token·suffix 불일치는 invalid다. Random payload 0x00과 PAD는 다르다. EOS/PAD 포함 전 위치를 corruption하며 true padding attention mask를 사용하지 않는다.

Correctness corpus는 모든 허용 symbol의 길이-4 반복문, 모든 길이 4–31의 최소 symbol 반복·최대 symbol 반복·mixed pattern, 잘못된 shape/nonfinite/길이/mask/padding/token 구조 fixture다. Applicable codec의 clean round-trip은 100%여야 한다. Clean round-trip은 생성 유효도나 조건 활용의 증거가 아니다.

## 7. Model, objective, sampler

### 7.1 Gaussian

ImageUNet width=32, input channels=2, condition dimension=12를 사용한다. 원문 spatial condition channel은 없다. 구조는 현재 `models.py`의 ImageUNet을 기준으로 source hash와 parameter count를 freeze한다. BGV/CGGE shape가 달라 계산량이 같다고 주장하지 않는다.

Forward는 `zt=sqrt(alpha_bar[t])*z0+sqrt(1-alpha_bar[t])*epsilon`, t는 0…999 discrete uniform, epsilon은 full-shape N(0,I)다. Linear beta .0001→.02, epsilon-prediction full-tensor mean squared error를 사용한다. Loss에 true-length mask를 적용하지 않는다.

Reverse는 현재 deterministic DDIM-style 식을 사용한다. Index grid는 `round(linspace(999,0,100))`, 전체 noise에서 시작한다. `z0_hat=(zt-sqrt(1-alpha_bar)*epsilon_hat)/sqrt(alpha_bar)`로 예측하고 다음 alpha로 이동하며, 마지막 previous-alpha는 1이다. Epsilon mode의 중간 clean prediction은 clip하지 않고 최종 출력만 [-1,1]로 clip한다. Guidance·verifier feedback은 없다.

Terminal alpha-bar≈4.03583×10^-5는 순수 잡음의 근사이며 정확히 0이 아니다. 이를 숨기지 않고 full-generation positive control로 점검한다. Zero-SNR를 쓰려면 prediction/reverse 식까지 바꾸는 새 revision이 필요하다. [Terminal mismatch 관련 연구](https://arxiv.org/abs/2305.08891)

### 7.2 Discrete

현재 SequenceDenoiser global MLP, hidden width=128, token embedding=16, condition dimension=12를 사용한다. Input은 32개 token embedding flatten, continuous t, condition이며 출력은 위치별 clean-state logits다. MASK는 output clean state에서 제외한다.

Training은 t∼U(0,1), 각 위치를 조건부 독립 확률 t로 MASK한다. Loss는 각 sequence에서 masked-position CE 합을 `max(1, masked_count)`로 나눈 후 batch 평균이다. Mask가 없으면 그 sequence contribution은 0이다. 이 가중치를 특정 likelihood ELBO와 동일하다고 주장하지 않고 **명시적 denoising surrogate**로 채택한다. Objective와 masking diffusion의 관계는 [MDLM](https://arxiv.org/abs/2406.07524)을 참조한다.

Generation은 all MASK에서 시작한다. Reverse grid t=1,31/32,…,0의 32 transitions를 사용한다. 현재 MASK인 위치를 확률 `1-t_previous/t_current`로 reveal하고 logits/temperature=1에서 categorical sample한다. Reveal된 위치는 다시 mask하지 않는다. EOS/PAD의 생성 규칙은 payload와 같고 최종 strict grammar 검사 전 수선하지 않는다.

## 8. 학습과 validation 선택

최초 v2 profile은 architecture당 search trial 1개다. Adam(lr=.001, betas=.9/.999, eps=1e-8, weight_decay=0), batch=64, float32, epochs=100을 고정한다. Train 10,000개를 epoch마다 무작위 permutation으로 한 번씩 처리하며 마지막 작은 batch를 버리지 않는다. 따라서 run당 15,700 optimizer updates다. Dropout, augmentation, mixed precision, 성능 기반 early stopping, seed hunting은 사용하지 않는다.

Epoch 10,20,…,100에서 validation loss를 계산한다. Validation representative 512개마다 4개의 고정 corruption draws를 사용한다. Time/noise/masks는 validation-noise namespace로 미리 정하고 checkpoint·Main/Shuffled 간 고정한다. Validation에는 실제 condition을 사용하며, Main/Shuffled 모두 같은 objective·선택 규칙·평가 횟수를 적용한다. 평균 validation denoising loss 최소 checkpoint를 선택하고 exact tie는 이른 epoch를 택한다. Validation hash success로 선택하지 않는다.

이 loss는 source/structure 복원 proxy이며 hash success를 보장하지 않는다. Hash test를 열기 전에 선택된 checkpoint와 모든 config를 봉인한다. Seed별로 동일 규칙을 적용하며 seed 0 결과로 seed 1/2 config를 바꾸지 않는다.

학습 budget·architecture가 positive control을 충족하지 못하면 그 pipeline은 BLOCKED다. 독립 개발 자료에서 변경할 수 있지만 v2.x amendment로 새 값을 기록하고 영향받은 control을 재검증한다. 이미 hash test를 본 후 같은 holdout에 맞춰 고친 결과를 v2 확증 결과로 재명명하지 않는다.

## 9. 대조군

### 9.1 Source-prior random

각 attempt마다 원래 source prior에서 독립적으로 길이와 payload를 draw한다. Replacement를 허용한다. Ownership, target 원문 길이, generated validity, 이전 성공으로 필터링하지 않는다. Source·seed·target별 stream을 만들어 같은 source의 pipeline이 immutable하게 재사용한다. Domain, K, verifier, target order는 Main과 일치한다.

실제 p_y=Σ_{x:H12(x)=y}P(x)에 대해 Success@K=1-(1-p_y)^K다. `p_y≈1/4096`에서 Success@100≈.02412136은 설계 근사다. 실제 paired control outcome 대신 이 수치를 넣지 않는다.

### 9.2 Shuffled training

각 epoch의 10,000 training rows에 독립 uniform permutation π를 만들고 x_i와 H12(x_π(i))를 pairing한다. Training split 밖 donor는 사용하지 않는다. Permutation을 epoch마다 갱신하고 namespace/seed/checksum과 accidental same-digest fraction을 기록한다. Fixed point나 same-digest match를 outcome에 따라 제거하거나 shuffle을 재추첨하지 않는다.

Architecture, initialization, train order, optimizer, epochs, validation rule, sampler, K는 Main과 맞춘다. Validation과 primary inference에는 **실제 target y**를 입력한다. Inference-time condition shuffling은 별도 diagnostic이며 이 대조군을 대신하지 않는다.

### 9.3 학습된 합성 positive control

12-bit y를 큰 자리부터 4-bit nibble 세 개로 나눈다. Printable에서는 `0123456789abcdef`, Random Bytes에서는 byte 0x00…0x0F에 대응시켜 첫 세 payload 위치를 정한다. L∼U{4,…,31}, 나머지 payload는 해당 alphabet에서 iid uniform으로 생성한다. 합성 과제의 첫 세 위치 분포가 primary source와 다름을 보고한다.

4096 conditions를 `(y,y XOR 4095)`의 2048 complement pair로 묶는다. Pair ID를 고정 RNG로 섞어 train 1536 pairs=3072 conditions, validation 256 pairs=512, test 256 pairs=512로 배분한다. **조건 반전 뒤에도 같은 split에 남는다.** Hash ownership과 synthetic split은 서로 다른 task/namespace다.

Train은 train condition을 균등하게 선택하여 unique message 10,000개를 생성한다. Validation은 condition당 1개 대표를 쓰고 test는 condition list만 sampler에 전달한다. 실제 12-bit encoder, architecture, training profile, sampler, codec을 모두 그대로 사용한다. Model seeds 0/1/2마다 별도 control을 학습한다. 원문 전체 condition·spatial bypass는 금지한다.

각 test condition당 K=1의 정상 generation과 bitwise 반전 condition generation을 같은 초기 RNG state로 수행한다. Synthetic verifier는 MD5 verifier와 task ID로 분리한다. 원문 전체 exact recovery는 요구하지 않는다.

각 seed에서 다음 모두를 요구한다.

1. 정상 condition에 대한 valid AND constraint success의 양측 95% Wilson lower bound≥.90.
2. 반전 condition의 생성물에 대해 반전 조건 자체의 valid AND constraint success lower bound≥.90.
3. 반전 생성물이 원래 condition 제약을 만족하는 성공률의 Wilson upper bound≤.05.

Gate는 실제 학습·held-out generation으로 측정한다. Partition 산술, codec 검사, oracle 출력만으로 통과시킬 수 없다. 이 control은 조건 경로가 작동하는 최소 증거이며 MD5 학습 가능성의 충분조건이 아니다.

## 10. 후보 생성·ledger·primary metric

각 target·method·seed에서 정확히 100개의 generation opportunity를 만든다. 첫 1/10/100개를 prefix로 평가한다. Invalid, 중복, 이미 성공한 뒤의 후보도 예산에 포함한다. 큰 pool에서 선별하거나 validity repair, rejection, beam search, hash reranking을 하지 않는다. Reverse trajectory 하나가 attempt 하나다.

$$S_i(K)=1\{\exists j\le K:\operatorname{Valid}(\hat x_{ij})\land H_{12}(\hat x_{ij})=y_i\},\quad\widehat P_K=N^{-1}\sum_iS_i(K).$$

Verifier는 decoded payload만 hash한다. Representative 길이·full digest 일치·source recovery는 성공 조건이 아니다. Candidate에 bytes가 있으면 duplicate/invalid-domain 여부와 관계없이 실제 hash 호출을 계수하고, bytes가 없는 invalid에는 hash 호출 0을 기록한다. Success는 source-domain validity를 동시에 요구한다.

Ledger는 target/source/pipeline/method/seed/position, candidate hex 또는 invalid marker, decoder reason, validity, verifier outcome, RNG/config/checkpoint identity, timing을 generation 직후 append한다. Target별 실제 100 rows 및 K-prefix monotonicity를 검사한다. Output order는 target numeric order, attempt 1…100이다.

Crash 후 completed rows를 바꾸지 않는다. Resume는 봉인된 checkpoint와 RNG/position 상태에서만 허용한다. 정확한 continuation이 불가능하면 INCOMPLETE이며 실패 attempt를 지우고 유리한 stream을 재생성하지 않는다. Zero-success completed와 missing stream은 구분한다.

## 11. 통계적 추론과 power

### 11.1 Estimand와 component test의 가정

Primary estimand는 고정된 source·ownership·학습 절차·checkpoint·seed에서 retained unique test targets를 같은 비중으로 평균한 sampling success 차이다. Source-specific exposure 제외와 current split에 조건부다. 다른 source, 새 학습 데이터, 임의 전체 digest로의 일반화를 자동으로 주장하지 않는다.

각 control과 paired binary outcomes M_i,B_i를 만들고 n10,n01,n00,n11을 기록한다. 관측 효과 Δ=(n10-n01)/N이다. One-sided exact McNemar의 계산은 `Binomial(n10+n01,.5)`의 n10 이상 tail이며 discordance 0이면 p=1이다. [Binomial test 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)

해석을 위해 **독립 target pairs와 null에서의 discordant-direction exchangeability**를 요구한다. 한 가지 충분한 working model은 각 target의 discordance 발생률은 달라도, 발생한 경우 Main-only 방향 확률 θ가 target에 공통이고 null에서 θ≤.5라는 것이다. 이 모형에서는 평균 효과의 부호와 2θ-1의 부호가 일치한다. Arbitrary target별 θ_i의 상쇄로 평균 효과가 0인 모든 경우까지 exact 보장을 확장하지 않는다.

고유 digest, 난수 namespace 분리, 공통 checkpoint만으로 이 확률 모형이 증명되는 것은 아니다. Finite target population, training/split dependence, target 난이도·효과 이질성을 제한사항에 남긴다. “Exact”는 조건부 binomial tail의 계산이며 모든 설계 의존성에 대한 distribution-free 보장을 뜻하지 않는다. 이 모형 해석을 방어할 수 없는 경우 effect/CI는 기술적으로 보고하고 **confirmatory G3를 부여하지 않는다**. 사후에 유리한 다른 검정으로 교체하지 않는다. 변경이 필요하면 test 공개 전 새 분석 revision과 power 검토를 수행한다.

### 11.2 Composite family와 판정

각 pipeline f에서 두 controls×세 seeds의 여섯 component p-value를 결합한다.

$$p_f=\max_{s\in\{0,1,2\},\ b\in\{random,shuffled\}}p_{f,s,b}.$$

Composite null은 “여섯 비교 중 하나 이상에서 우위가 없음”이다. Component p가 유효할 때 true-null component p_j에 대해 Pr(max(p)≤a)≤Pr(p_j≤a)≤a이므로 max-p는 유효하다. 이 결합 논증은 component 간 독립을 요구하지 않는다.

다섯 p_f에 Holm을 한 번 적용한다. Family membership과 tie ordering은 matrix의 ID 순서로 고정하고 α=.05, adjusted p<.05를 통과 조건으로 삼는다. Missing pipeline/component는 status를 BLOCKED/INCOMPLETE로 유지하며 보정 계산에만 p=1을 넣는다. 성능을 0으로 채우지 않는다. [Holm 공식 문서](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)

G3 통과는 measurement gates 완료, 여섯 관측 Δ>0, model-based inferential 조건의 명시, composite Holm 통과를 모두 요구한다. Family는 다섯 “반복된 양쪽 control 우위” 주장에 대한 것이며 30개 component 각각의 독립적인 project-wide 유의성 주장으로 재해석하지 않는다. K=1/10, 대표 원문 복원, representation comparisons는 exploratory다.

### 11.3 신뢰구간과 음성 결과

Seed별 control 차이에 대해 target index를 10,000번 paired resample하고 Δ를 계산한다. 같은 source에서 모든 method·seed·K의 target row를 함께 유지한다. Percentile 2.5/97.5%, NumPy quantile `method="linear"`, 봉인된 bootstrap seed를 사용한다. Interval은 marginal이며 Holm이 simultaneous coverage를 만들어 주지 않는다. CI lower bound 조건을 별도 성공 gate로 더하지 않는다.

Empirical 차이가 모두 같아 구간이 퇴화하면 `DEGENERATE_EMPIRICAL_CI`를 붙인다. 성공 0이면 독립·동일 성공확률 모형의 one-sided upper `1-.05^(1/N)`를 보조적으로 보고하되 이질적 finite-target coverage까지 주장하지 않는다. 비유의·sign-only·효과 상한·검정력 부족을 구분한다. 두 배 효과를 설계 대안으로 썼다고 통과 시 최소 두 배가 입증되는 것은 아니다.

### 11.4 Power와 민감도

Reference scenario는 p_random=p_shuffled≈.02412136, p_main=2p_random, N=2048이며 homogeneous target probabilities, 방법·seed별 독립 결과를 가정한다. 여섯 component 모두 p<.01이라는 Holm 충분조건에서 이전 100,000회 설계 simulation은 약 81.3%의 통과 확률을 보였다. 이는 한 pipeline의 통계 조건에 대한 수치이고 전체 다섯 pipeline이나 측정 gate의 통과 확률이 아니다.

v2 검증에서는 같은 reference 외에 하나의 약한 seed(1.5배), 더 강한 shuffled(1.5×random), 두 target 난이도 strata(.2/1.8), 3배 Main을 점검한다. Reference 80%는 특정 대안의 설계 목표이며 모든 대안에 대한 보장이 아니다. 민감도가 나쁘면 작은 효과·seed 변동을 배제할 수 없다는 결론 범위를 명시한다. 이 제한을 받아들이지 못하면 test 전에 표본/범위/목표 효과를 개정한다. Baseline 결과를 미리 열어 유리한 N이나 K를 선택하지 않는다.

## 12. Gate와 실행 순서

| Gate | 요구 증거 | 실패·누락 처리 |
|---|---|---|
| G0 | 전체 접근 감사, 봉인된 ownership/dataset, overlap 0, 정보 경계·mutation test | 해당 data/pipeline BLOCKED |
| G1-A | 전체 correctness corpus round-trip 및 invalid 구조 rejection | 해당 codec BLOCKED |
| G1-B | 실제 학습된 same-path synthetic control, 세 seeds | 해당 pipeline hash runs BLOCKED |
| G2 | 동일 target/order/100-prefix stream, 독립 verifier, ledger·resume·대조군 일치 | 해당 비교 INVALID/BLOCKED |
| G3 | §11의 가정·paired test·max-p·Holm와 효과 기준 | 확증적 우위 없음 |
| G4 | 지정 seeds 0/1/2의 완전한 실행, seed 선택 없음 | 미완료는 INCOMPLETE |
| CA0 | 필수 compute/environment/provenance telemetry 완료 | 별도 accounting INCOMPLETE |

P0는 G0/G1/G2를 통과한 측정 가능성이다. P1은 seed-0의 기술적 관측으로만 사용할 수 있으며 후속 seed 실행 대상 선택에 사용하지 않는다. P2/F2는 G3와 완전한 G4까지 통과한 pipeline에만 부여한다. Sign-only 관측은 “확증 기준 미충족”이다.

실행 순서는 (1) 노출 감사·과학 설계 봉인, (2) 독립 engineering 구현/fixture, (3) synthetic positive controls, (4) hardware benchmark와 실행 manifest 봉인, (5) primary dataset 생성 및 G0/G1/G2 artifact 확인, (6) 세 seed의 train/validation/checkpoint seal, (7) test stream 생성·독립 평가, (8) 통계·CA0·모든 실패 보고다.

소스 노출 감사 전 최종 test ownership을 확정하지 않는다. Measurement gate 미충족을 단지 “실험 결과가 나빴다”고 처리하지 않는다. Pipeline 일부가 blocked여도 나머지가 자신의 prerequisites를 만족하면 실행할 수 있지만 family 5는 유지하고 전체 연구를 완결됐다고 부르지 않는다. 결과는 pipeline별로 보고하며 하나의 전체 성공 label로 합치지 않는다.

## 13. 비용, 저장, 실행 manifest

Primary learned candidate opportunities는 6,144,000개다. Gaussian 18 runs×2048×100×100 NFE와 Discrete 12 runs×2048×100×32 NFE의 합은 **447,283,200 candidate-level NFE**다. 모든 sampler가 100 steps라고 가정한 614,400,000과 구분한다. Random은 source×seed 공유 시 1,228,800 attempts다. Learned 및 공유 Random ledger는 총 7,372,800 rows다.

NFE는 batched API call 수나 MD5 연산량과 같지 않다. 학습 updates, examples, parameter count, training/validation/inference/verification/IO 시간, 실제 MD5 calls, batch, peak RAM/VRAM, checkpoint/artifact 크기, device/software/precision을 분리한다. Accelerator 타이밍은 동기화 경계를 명시한다. Unavailable 값은 0 대신 이유와 함께 기록한다.

실행 전에 actual hardware, final inference batch, per-run/study wall-clock cap, storage cap, resume policy, mandatory telemetry instrumentation을 채운다. 이들은 **필수 execution record**이며 미기입이면 실행을 막는다. 현 환경의 짧은 CPU benchmark는 untrained synthetic forward 비용의 참고값일 뿐 최종 budget 승인이나 trained generation 성능을 의미하지 않는다.

Candidate bytes/invalid ledger는 전부 stream으로 보존한다. Raw tensor는 run별 첫 16개 target의 첫 candidate만 보관한다. 선택은 결과와 무관하다. Full raw tensor pool을 메모리에 누적하지 않는다. Code revision과 uncommitted patch hash, config·checkpoint·dataset·ownership·노출 inventory hash, commands, runtime, start/end, seeds를 연결한다. Binary는 hex+length로 기록한다.

경로는 `local_experiment_archive/runs/<run-id>/`를 사용한다. Raw candidates, checkpoints, datasets는 git에 넣지 않는다. Plan·schema·검증 코드와 concise report는 version 관리한다. 실제 outcome 없이 “예상 성공률”을 관측 결과 필드에 쓰지 않는다.

## 14. 최종 결과와 후속 단계

각 pipeline의 결과는 `BLOCKED/INCOMPLETE`, `VALID BUT NO CONFIRMED ADVANTAGE`, `CONFIRMED REPLICATED ADVANTAGE` 중 하나이며 CA0는 별도 열로 보고한다. Sign-only, seed별 실패, CI 퇴화, power 한계, source-specific target exclusion을 함께 명시한다.

후속 compute study는 새 사전등록과 평가 범위에서 source-prior search, 전처리 비용을 포함한 preimage table, training/tuning 상각, memory와 wall time을 비교한다. 후보 수 우위를 계산량 우위로 바꾸어 표현하지 않는다. q=16과 큰 q scaling도 별도 검정력·예산을 확보한다. Truncated 구간의 기울기로 full MD5의 복잡도를 주장하지 않는다.

허용되는 성공 문장은 “봉인한 source-specific 12-bit target 집합, 지정 model seeds와 candidate budget에서 명시한 추론 모형하에 두 대조군 대비 반복 우위가 지지되었다”다. 측정 gate가 실패했다면 “해당 pipeline으로 연구 가설을 시험하지 못했다”고 쓰고, 비유의라면 “본 budget에서 확증 기준을 충족하지 못했다”고 쓴다.

## 15. PoC 적합성 판단과 남은 실행 조건

v2는 q·endpoint·source별 표적 수·대조군·모든 seeds·statistical family·codec·초기 model/training profile을 명시한 **개발 가능한 PoC 설계**다. 낮은 q의 큰 효과를 선별하는 제한된 연구이며 작은 효과나 일반적인 불가능성을 판정하도록 설계되지 않았다.

즉시 확증 실행 준비 완료를 뜻하지 않는다. 다음 다섯 항목은 artifact로 닫혀야 한다.

1. 알려진 archive 범위 밖까지 포함한 접근 감사와 최종 source별 ownership seal.
2. Prefix-only generation, train-only shuffle, 100-prefix ledger, mutation/resume 검사까지 연결된 v2 실행 경로.
3. 다섯 pipeline·세 seed의 실제 학습 positive-control 통과.
4. §11의 inferential population/working assumptions와 검정력 민감도에 맞는 제한된 claim을 명시한 analysis seal.
5. 실제 hardware 처리량, 전체 학습·평가·저장 비용, 실행 상한을 포함한 resource manifest.

현재 legacy runner는 이 조건들을 자동으로 만족하지 않는다. Existing tests의 성공, 새 문서의 완성, 산술의 일치만으로 G0/G1-B/G2 또는 F2를 부여하지 않는다. 구체적인 관측 증거와 적합성 판정은 별도 검증 보고서에 기록한다.
