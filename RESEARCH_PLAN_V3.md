# 연구 계획 수정안 v3 — Pilot부터 본실험까지

**Protocol:** `dhi-v3-20260924` · **Revision:** 3.0 · **작성일:** 2026-09-24 KST  
**현재 상태 (2026-09-25 갱신):** Pilot CLI P0–P3 및 Pilot report 구현 / 정식 P1–P3·본실험 미실행 / 본실험 CLI 미구현  
**기계 판독 명세:** [poc-v3-protocol.json](/Users/choisoonwook/Experiments_local/DHI_AI_gen/examples/poc-v3-protocol.json)

## 1. 문서의 역할과 완료 상태

본 문서는 Pilot의 범위·단계별 규모·학습·평가·완료 기준·실패 대응과 본실험 진입·실행·보고를 연결한 v3 명세다. 과학적 고정값을 아래에 다시 적으며 v2나 v2.1의 미지정 값을 암묵적으로 상속하지 않는다. v2 계열 문서는 변경 이력으로 남긴다. 문서와 v3 JSON이 불일치하면 실행을 차단한다.

v2.1에서 제안한 **고정 test pool 안의 독립 균등 복원추출**을 v3의 본평가 방식으로 채택한다. Trial 수는 2,048로 고정한다. 검정력·자원 검사에서 부적합하면 이 문서의 값을 실행 중 바꾸지 않고 v3.1 이상의 새 revision으로 개정한다.

2026-09-25에 **`diffusion_hash_inv.study_cli pilot` 및 Pilot용 `report`를 구현했다.** 실제 명령·파라미터·검증 범위는 [PILOT_V3_CLI.md](/Users/choisoonwook/Experiments_local/DHI_AI_gen/PILOT_V3_CLI.md)를 따른다. §15의 나머지 본실험 명령은 구현 계약으로 남는다. JSON은 기존 `ExperimentConfig`가 아니며 `*_at_authoring` 필드는 2026-09-24 작성 당시 상태를 보존한다. 문서·소프트웨어 구현·실험 완료를 구분한다. 삭제된 pilot이나 그 결과를 복원하지 않았다.

| 상태 | 정의 |
|---|---|
| `PILOT_TECHNICAL_PASS` | P0·P1·P2의 환경·입력·학습·생성·저장·재개 검사가 통과 |
| `PILOT_EVALUATION_COMPLETE` | P3의 15개 합성 모델 평가와 무결한 결과 보고가 완료; 성능 통과와 별개 |
| `PILOT_QUALIFIED` | 앞의 두 상태와 P3의 15개 모델 수용 기준을 모두 충족 |
| `PILOT_PARTIAL_QUALIFICATION` | 일부 pipeline만 세 seed의 수용 기준을 충족; 적격 목록과 실패 이유를 명시 |
| `MAIN_READY` | 적격 pipeline에 대해 노출·데이터·분석·자원·구현 조건을 봉인 |
| `STUDY_COMPLETE` | 지정된 전체 matrix의 실행·무결성 검사·분석·비용 보고 완료; 우위 비유의여도 가능 |

일부 pipeline이 제외되면 `PARTIAL_STUDY_COMPLETE`로 보고한다. 전체 완료 또는 전체 성공으로 합치지 않는다. 통계적 성공은 별도로 pipeline별 판정한다.

## 2. 연구 질문·범위·고정 matrix

> 봉인한 source별 2,048개 미관측 12-bit MD5 digest pool에서 표적을 균등하게 추출했을 때, 학습 모델이 100번의 생성 기회 안에 유효 preimage를 찾을 평균 확률이 source-prior random과 shuffled-training control보다 높으며, 그 우위가 지정 model seeds 0·1·2 각각에서 성립하는가?

| 순서·ID | Source | 표현 | Model | Formal seeds |
|---|---|---|---|---|
| 1. P-G-BGV | Printable | BGV | Gaussian ImageUNet | 0/1/2 |
| 2. P-G-CGGE | Printable | CGGE | Gaussian ImageUNet | 0/1/2 |
| 3. P-DISC | Printable | Token | Discrete SequenceDenoiser | 0/1/2 |
| 4. R-G-BGV | Random Bytes | BGV | Gaussian ImageUNet | 0/1/2 |
| 5. R-DISC | Random Bytes | Token | Discrete SequenceDenoiser | 0/1/2 |

本실험은 pipeline별 Main·Shuffled 두 모델을 세 seed에서 학습한다. 전체 30 learned runs다. Random은 source·model seed별 stream을 만들어 같은 source의 pipeline이 공유한다. P3의 합성 모델 15개는 별도다. GPU 작업은 한 번에 한 run만 실행하며 pipeline 순서→model seed 오름차순→Main/Shuffled 순서로 진행한다.

성공 주장은 고정 source·pool·training data·checkpoints·지정 seeds의 평균 sampling 성능으로 제한한다. Full MD5 역산, 모든 digest에서의 우위, 새 dataset에서의 재현, 최소 두 배 효과, 계산량 우위, Gaussian/Discrete 계열 일반의 우위는 이 설계로 주장하지 않는다. 표현 비교와 @1/@10은 탐색적이다.

## 3. 공통 source·hash·표현·모델 명세

### 3.1 Source와 hash

길이 L은 정수 4…31에서 균등, 주어진 길이에서 symbol은 iid uniform이다. Printable은 byte 0x21…0x7E의 94문자이며 space를 제외한다. Random Bytes는 0x00…0xFF다. 따라서 `P(x)=A^(-|x|)/28`이고 전체 가능한 문자열에 대한 균등분포가 아니다.

`H12(x)=Int_big(MD5(payload_bytes)) >> 116`을 사용한다. Condition은 leading zero를 보존하는 정확히 12개 binary bits다. Header/EOS/PAD/이미지 pixels는 MD5 입력이 아니다. 원문의 길이·원문 자체·full digest suffix는 생성 입력이 아니다. 성공은 **source-valid payload이면서 H12(payload)=target**인 경우다. 대표 원문과 같은 문자열·길이일 필요는 없다.

### 3.2 표현과 strict decoder

| 항목 | BGV | CGGE | Token |
|---|---|---|---|
| Shape | `[2,32,128]` | `[2,32,64]` | `[32]` |
| 길이 | Slot 0의 생성된 length byte | 처음 L개 연속 valid cells | 생성된 EOS 위치 |
| Payload | 4×8 slots; byte의 2×4 bits를 각 4×4 pixels로 확장 | 고정 8×8 glyph, Printable만 | Source별 payload IDs |
| Padding | Invalid mask와 decoded byte 0 | Invalid mask, unused glyph 값 무시 | EOS 뒤 PAD만 |

BGV의 cell validity와 bit block mean은 각각 ≥.5를 참으로 해석한다. Header 길이는 4…31, header 및 정확히 L개 payload mask만 valid여야 한다. CGGE mask는 처음 L개 contiguous이고 32번째 slot은 unused다. Glyph는 prototype MSE 최소로 decode하며 MSE>.1이면 invalid, 동률이면 작은 ASCII를 고른다. 고정 font는 `font8x8-basic-v1`, SHA-256은 `6ef6d0bfa6e29c823ad9ff7d6b4b29b9af5f739fee711268ff5b83ca7aeed50a`다.

Printable tokens는 payload 0…93/EOS94/PAD95/MASK96, Random Bytes는 payload 0…255/EOS256/PAD257/MASK258이다. Payload L개, EOS 하나, 나머지 PAD만 허용한다. MASK 잔존·잘못된 token·다중 EOS·padding 오류·잘못된 shape·비유한 image는 invalid다. Random byte 0과 PAD를 구분한다.

Gaussian clean image는 `z0=2*image-1`로 변환한다. Validity·header·padding을 포함한 모든 좌표를 noise 처리한다. 최종 출력만 [-1,1]로 clip한다. True-length mask, target image, true padding attention mask, validity repair, rejection sampling은 사용하지 않는다.

### 3.3 Model·objective·sampler

| 항목 | Gaussian | Discrete |
|---|---|---|
| 구조 | ImageUNet, width32, channels2, condition12 | Global SequenceDenoiser MLP, width128, embedding16, condition12 |
| Parameters | BGV/CGGE 각각 114,978 | Printable 465,168; Random Bytes 1,136,496 |
| Corruption | T=1,000, linear beta .0001→.02, uniform integer t | t∼U(0,1), 각 위치를 확률 t로 MASK |
| Loss | Full-tensor epsilon MSE | Sequence별 masked CE 합/max(1,masked count)의 batch 평균 |
| Reverse | Deterministic DDIM-style, 100 steps | 32 intervals, temperature1, reveal 후 remask 없음 |

Gaussian은 `alpha_bar=cumprod(1-beta)`이고 `zt=sqrt(alpha_bar)*z0+sqrt(1-alpha_bar)*epsilon`이다. Sampling indices는 `round(linspace(999,0,100))`, 전체 Gaussian noise에서 시작한다. `z0_hat=(zt-sqrt(1-alpha_bar)*epsilon_hat)/sqrt(alpha_bar)`를 사용하고 다음 alpha로 이동한다. 마지막 previous-alpha는 1이며 중간 clean prediction을 clip하지 않는다. Terminal alpha-bar≈4.03583×10^-5는 정확히 0이 아닌 pure-noise 근사다.

Discrete loss에서 mask가 없는 sequence contribution은 0이다. 이를 likelihood ELBO와 동일하다고 주장하지 않는다. Reverse grid는 t=1,31/32,…,0이며 MASK 위치를 확률 `1-t_previous/t_current`로 reveal한다. Clean-state logits에서 MASK를 제외하고 categorical sampling한다. EOS/PAD도 동일하게 생성한다.

기준 구조는 현재 [models.py](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/models.py), [discrete.py](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/discrete.py)다. 작성 시점의 source hashes를 JSON에 기록했다. 구현 봉인 시 parameter count와 source hash를 검사하며 차이가 나면 변경 내용을 검토한다. 구조·objective·sampler의 과학적 동작을 바꾸면 새 scientific revision과 영향받는 control 재검증이 필요하다.

### 3.4 공통 학습 규칙

Adam(lr=.001, betas=.9/.999, eps=1e-8, weight_decay=0), batch64, float32다. Epoch마다 모든 train rows를 permutation으로 한 번씩 처리하며 마지막 작은 batch를 버리지 않는다. Augmentation, dropout, mixed precision, 성능 기반 early stopping, seed hunting을 사용하지 않는다. Profile search trial은 1개다.

Validation condition마다 고정 corruption draws 4개를 사용하고 checkpoint·Main/Shuffled·model seed 사이에 동일하게 유지한다. 실제 condition의 평균 denoising loss가 최소인 checkpoint를 선택하며 정확한 tie는 이른 epoch다. Validation hash success로 선택하지 않는다. Validation epoch 목록과 dataset 크기는 §4의 단계별 값을 따른다.

Main/Shuffled는 같은 초기 weights와 train order를 사용한다. Shuffled만 epoch마다 training rows의 uniform donor permutation으로 condition pairing을 바꾼다. Donor는 train 안에서만 뽑고 fixed point·same-digest match를 제거하거나 재추첨하지 않는다. Permutation seed/checksum·same-condition 비율을 기록한다. Validation/inference에는 실제 condition을 준다.

## 4. Pilot 범위와 단계별 고정 실행 규모

Pilot은 **독립 합성 과제로 입력·학습·생성·측정 경로와 학습 적격성을 검증하는 과정**이다. Primary MD5 test의 성능으로 개발하지 않는다. 작은 기술 검사를 위해 본실험의 전체 노출 감사를 기다릴 필요는 없다.

| 단계 | 모델·seed·methods | 학습/validation | 학습량·선택 | 생성·평가 | 완료 기준 |
|---|---|---|---|---|---|
| P0 환경·fixture | 5개 pipeline 구성 확인, 학습 없음 | 고정 codec·verifier·통계 fixture | 없음 | MPS tensor 검사, clean/negative fixtures | 모든 필수 검사가 통과 |
| P1 작은 GPU 통합 | 5 pipelines×Main/Shuffled×seed0 = 10 models | Train256, validation32 | 2 epochs; epoch1/2 validation; 8 updates/run | Validation pool32에서 16 iid trials, K10; Random은 source별 공유; inference batch4 | Finite 학습·생성, 입력 경계·예산·저장·재개 통과; 생성 유효도 문턱 없음 |
| P2 학습·자원 진단 | 5 pipelines×Main×seed0 = 5 models | Train10,000, validation512 | 10 epochs; epoch5/10 validation; 1,570 updates/run | Validation128 conditions, 정상/반전 각 K1, batch4; 별도 batch profile | 진단·측정·복구 보고 완성; 생성 품질은 진단값 |
| P3 정규 합성 검증 | 5 pipelines×Main×seeds0/1/2 = 15 models | Train10,000, validation512 | 100 epochs; 매10 epochs validation; 15,700 updates/run | 봉인된 synthetic test512 전수, 정상/반전 각 K1 | 각 seed에서 §5의 정수 문턱 통과 |

P1/P2/P3는 각각 fresh initialization으로 시작한다. P1/P2 weights·optimizer를 P3 또는 MD5 본실험으로 이어 학습하지 않는다. Formal profile의 선택은 P3 test를 보기 전에 확정한다.

P1은 @1/@10과 ledger 동작을 확인한다. K=100의 invalid·중복·성공 후 시도 계수와 @1/@10/@100 monotonicity는 P0의 고정 100-attempt fixture로 따로 확인한다. 따라서 P1의 작은 K를 본실험 K=100으로 오인하지 않는다.

### 4.1 P0의 필수 검사

- 현재 Python/NumPy/Torch·OS·device와 package 경로를 기록하고 MPS tensor 연산·동기화를 확인한다. GPU 불가 시 GPU 단계는 차단하고 CPU로 자동 변경하지 않는다.
- 모델 parameter count·input shape·condition12·CGGE font checksum을 확인한다.
- 모든 symbol의 길이-4 반복과 길이4…31의 min/max/mixed codec corpus를 검사한다. Applicable 조합의 clean round-trip 1,214개 전부 일치해야 한다. 잘못된 shape·nonfinite·header/mask·token 문법 fixtures는 거부해야 한다.
- `hashlib.md5` verifier를 알려진 vectors와 독립 구현으로 비교한다. Source validity·앞12bits·header/EOS 제외를 별도 검사한다. 합성 verifier와 MD5 verifier의 task IDs를 혼용하지 않는다.
- 원문·길이·suffix·ID·padding metadata mutation, train-only shuffle, 같은 digest의 다른 trial RNG identity, missing/duplicate rows에 대한 negative fixture를 검사한다.
- Exact McNemar 작은 정수 fixture, max-p, five-family Holm, missing=1, discordance0, CI 퇴화, 100-attempt prefix fixture를 검사한다. 이 단계는 통계적 power 검증이 아니다.

### 4.2 P1의 복구 검사와 P2의 해석

P1은 동일 환경의 uninterrupted reference와 강제 중단·재개 실행을 비교한다. 각 pipeline의 Main에 대해 optimizer update5 직후와 generation의 첫 trial attempt7 뒤의 강제 중단을 검사한다. Checkpoint 및 candidate commit 경계는 §12를 따른다. 최종 logical update 수·parameter tensors·선택 checkpoint·decoded bytes·validity·verifier 결과가 같고 ledger key 누락·중복이 없어야 한다. Timing은 비교에서 제외한다. Shuffled 경로도 epoch permutation과 실제 inference condition fixture를 통과해야 한다.

P2에서 loss가 유한하지만 joint success가 0이면 `LEARNING_SIGNAL_ABSENT` 경고를 기록한다. 이것만으로 profile을 자동 변경하거나 본실험 가설을 기각하지 않는다. P2 완료는 학습 능력 인증이 아니며, P3가 최종 적격성 검사다. P2의 진단값을 근거로 profile을 수정하려면 P3 test 공개 전에 새 revision을 기록하고 영향받는 P0/P1/P2 검사를 다시 수행한다.

## 5. 합성 자료와 P3 수용 기준

Task ID는 `synthetic_nibbles`다. 12-bit y의 큰 자리부터 세 nibble을 Printable의 `0123456789abcdef` 또는 Random Bytes의 byte0…15로 대응해 첫 세 payload 위치를 정한다. 길이4…31 및 나머지 symbol은 원래 source prior에서 생성한다. 첫 세 위치의 분포가 실제 MD5 source와 다르며 이 과제가 MD5 학습을 입증하지 않는다는 점을 보고한다.

4096 conditions를 `(y,y XOR4095)`의 2048 complement pairs로 묶고 고정 RNG로 한 번 섞는다. Train1536 pairs=3072 conditions, validation256 pairs=512, test256 pairs=512다. Source·pipeline·model seed 사이에 이 condition split을 공유한다. Split RNG는 primary ownership과 분리한다.

Source별 train condition을 균등 선택해 unique messages 10,000개를 생성한다. Validation은 condition당 대표 하나다. Source별 draw cap은 1,000,000이며 부족하면 construction failure다. P1 train은 이 corpus의 draw-order 첫256개, P1 validation은 validation pair 순서의 첫16 pairs=32개다. P2는 전체 train/validation을 사용하고, `dev-probe` RNG로 validation pairs 중64개를 비복원 선택한 128개 condition을 probe한다. P1/P2에는 synthetic test512를 generation용으로 열지 않는다.

P3는 test512 conditions만 sampler에 전달하며 case별 정상 y와 반전 y XOR4095 generation을 동일한 초기 RNG state로 수행한다. 두 생성물은 별도 rows로 남긴다. 각 model seed에 대해 다음을 모두 요구한다.

1. 정상 생성물의 **valid AND 정상 constraint** 성공 ≥475/512.
2. 반전 생성물의 **valid AND 반전 constraint** 성공 ≥475/512.
3. 반전 생성물의 **valid AND 원래 constraint** 성공 ≤15/512.

이는 고정된 512개 cases에 대한 engineering acceptance 기준이다. 전체 모집단 또는 15개 모델의 동시 95% 보장으로 표현하지 않는다. Oracle·codec 성공으로 대체하지 않는다. 세 seed 중 하나라도 문턱을 못 넘으면 해당 pipeline은 본실험 부적격이다. 성능 미달만으로 나머지 지정 control seeds를 생략하지 않는다.

P3 test를 보고 수정했다면 같은 결과를 독립 control 통과로 재명명하지 않는다. 새 revision에서 개발에 사용한 cases·변경 이유·새 검증 범위를 명시한다. Primary holdout은 계속 열지 않는다.

## 6. GPU profile·운영 예산·실측값 확정

기준 장비는 현재 보존된 검사 기록의 Apple M3 Max/MPS다. 실제 실행 시작 시 환경을 다시 기록한다. MPS는 명시적으로 요청하고 silent CPU fallback은 금지한다. 기존 GPU smoke는 v3 인증이 아니다. [PyTorch MPS 문서](https://docs.pytorch.org/docs/2.14/notes/mps.html)

P2 학습에서 첫20 updates를 timing warm-up으로 제외하고 이후100 updates씩 세 구간의 elapsed time을 기록한다. 학습 updates 자체를 추가하거나 버리지 않는다. Validation, checkpoint 저장 비용도 별도 기록한다.

P2의 선택 checkpoint로 inference batch1/4/16/64를 각각 warm-up1 batch와 측정3 batches 실행한다. 입력은 synthetic validation conditions를 고정 순서로 순환하고 profile RNG를 사용한다. 총340 candidates/pipeline, 전체1,700 candidates다. 동기화한 model 시간, CPU transfer/decode/hash/IO, peak RSS/MPS memory를 구분한다. Validity가 높은 batch를 고르지 않는다. Exception·nonfinite 연산·메모리 한도 초과가 없고 처리량이 가장 높은 batch를 선택하며 1% 이내 차이는 작은 batch를 선택한다. Valid한 최대 길이 codec fixture로 decoder·verifier 비용도 확인한다.

선택 batch로 P1의 generation 복구 검사를 추가 실행해 동일 backend에서의 continuation을 확인한다. Formal batch는 pipeline별로 `resources.json`에 고정하며 P3와 본실험 Main/Shuffled/세 seeds에 동일하게 적용한다. Batch 변경이 필요하면 formal 실행 전에 다시 profile·복구 검사한다. Run 도중 자동 변경하지 않는다.

### 6.1 숫자가 정해진 정책 상한

아래는 실측 예상 시간이 아닌 **v3의 기본 운영 상한**이다. 너무 작거나 현재 저장 공간에 맞지 않으면 본평가 결과를 보기 전에 운영 amendment를 기록한다. 무제한 실행을 기본값으로 두지 않는다.

| 범위 | 누적 active wall-clock 상한 |
|---|---:|
| P0 | 10분 |
| P1, 필수 복구 replay 포함 | 30분 |
| P2, profile·복구 재검사 포함 | 4시간 |
| P3 전체 | 48시간 |
| 통계 calibration 전체 | 4시간 |
| 본실험 prepare/train/evaluate/report 전체 | 168시간 |
| Formal learned run 하나, train+evaluation 합계 | 24시간 |

Active time은 프로세스가 실제 실행 중인 시간을 누적하며 중단·전원 꺼짐 시간은 별도 기록한다. 재개 시 budget을 초기화하지 않는다. Study 산출물64 GiB, run 산출물8 GiB, process RSS64 GiB, MPS allocated memory64 GiB가 상한이다. RSS와 MPS 수치는 unified memory에서 겹칠 수 있으므로 합산하지 않고 각각 검사한다. 디스크 여유가10 GiB 미만이면 새 batch를 시작하지 않는다.

P2 실측으로 학습·validation·선택·생성·decode·verification·IO·control 비용을 포함한 stage/run 예상 시간을 산정한다. 예상 시간×1.5와 저장량×2를 필요한 budget으로 기록한다. 이 값이 정책 상한을 넘거나 남은 디스크에 들어가지 않으면 P3 또는 본실험을 시작하지 않고 `BLOCKED_RESOURCE`로 둔다. 실측 budget이 상한 이내이면 그 budget을 실제 soft cap으로 봉인한다. 상한 도달 후 성능을 보고 연장하지 않는다.

Formal batch·실측 처리량·예상 시간은 문서 작성 시 알 수 없는 값이다. 이를 비워 둔 채 실행하는 것이 아니라 **P2가 정해진 규칙으로 산출하고 P3 진입 전에 봉인**한다. 같은 장비라도 온도·시스템 부하에 따른 추산 오차가 있으며 여유율은 완료 보장이 아니다.

## 7. 본실험의 노출 감사와 데이터

Pilot의 합성 데이터와 MD5 본실험 데이터는 서로 다른 task/RNG다. Pilot에서 primary test pool의 생성 성능을 측정하지 않는다. Fixed verifier vectors의 correctness 확인과 실제 모델 성능 평가를 audit에서 구분한다.

### 7.1 Exposure inventory

본실험 전에 source별 배제 집합 E_s를 확정한다. 이전 learned MD5 validation/test의 q≥12 target은 앞12bits를 제외한다. q<12라도 raw/full digest/model output을 설계에 사용했는지 따로 감사한다. 다른 source의 같은 prefix까지 배제한 project-wide novelty는 주장하지 않는다.

`exposure_inventory.json`에는 다음 입력을 요구한다.

- 조사한 project/archive 및 추가 선언 경로, 조사 timestamp와 작성자.
- 각 자료의 path 또는 삭제/외부 자료 식별자, source, task, q, 역할(train/validation/test/fixture), 사용·노출 분류와 근거.
- 읽을 수 있는 metadata 파일 hashes, source별 제외12-bit 목록.
- 외부·삭제 자료에 대해 알려진 이력, 조사 범위 확인문, 미해결 항목 목록.

파일이 없다는 이유만으로 미노출로 처리하지 않는다. 알려진 미해결 항목이 남으면 main audit는 미완료다. 자동 scan이 사람의 과거 접근을 완전히 증명한다고 표현하지 않는다. 현재 알려진 잔여2211/2236개는 최종 audit 값이 아니다.

### 7.2 Ownership·corpus 알고리즘

Source별로 다음을 수행한다.

1. `A_s=sorted({0,…,4095}−E_s)`를 ownership-test RNG로 한 번 섞고 첫2048개를 test pool T_s로 둔다. |A_s|<2048이면 해당 source는 BLOCKED다.
2. T_s 외의2048개 값을 정렬·ownership-rest RNG로 섞어 train1536, validation512로 나눈다.
3. 원래 source prior에서 draw하고 MD5를 계산한다. Train owner의 최초 unique raw messages 10,000개를 보관한다. Validation/test는 각 digest의 첫 대표를 evaluator 영역에만 보관한다.
4. Train10,000·validation512·test2048 조건이 모두 채워지면 중단한다. Source별 draw cap1,000,000을 넘으면 construction failure다. Duplicate/surplus·실제 train digest 수를 보고한다.
5. Split 간 digest/raw 교집합0, T_s∩E_s=∅, quota·byte 보존·hash 재계산·dataset/ownership hashes를 검사한다.

동일 source의 모든 pipeline·method·model seed가 같은 corpus를 사용한다. 과거 weights·raw datasets·test-informed checkpoints는 이전하지 않는다. Dataset 생성 뒤 test 대표를 generator에서 구조적으로 분리한다.

Train은 ownership과 uniqueness로 조건화된 분포이고, pool은 노출 제외에 조건부다. 모델 seeds 반복은 새로운 training data에서의 재현 실험이 아니다.

## 8. 본실험 실행: M0–M3

| 단계 | 실행 | 선행 조건·완료 증거 |
|---|---|---|
| M0 준비·봉인 | Audit, 데이터 생성, 분석·resource·code/config 봉인 | P0/P1/P2 PASS, source audit PASS, §10 calibration PASS, 해당 pipeline P3 세 seed PASS |
| M1 학습 | 적격 pipeline의 Main/Shuffled×seeds0/1/2, train10,000·100 epochs | 매10 epochs validation512×4, 15,700 updates/run, 최저 loss checkpoint 선택 |
| M2 평가 | 모든 적용 checkpoints 봉인 후 source별 trial 목록 생성·평가 | Trial2048×K100; Main/Shuffled와 공유 Random; 실제 ledger 무결성 검사 |
| M3 분석·보고 | Paired trial outcomes, max-p/Holm, CI·비용·실패 보고 | §10·§14·§16의 완전성 및 provenance |

다섯 pipeline 전체가 적격이면 M1은30 models다. 일부가 blocked면 적격 pipeline은 실행할 수 있으나 family는5를 유지하며 전체 완료라고 부르지 않는다. Primary 성능을 이유로 seed나 pipeline을 제외하지 않는다. 본평가가 시작되기 전에 모든 적용 M1 checkpoints를 봉인한다.

## 9. 본평가 표집·후보·estimand

T_s 크기는2048이며 R=2048회 `Y_i iid∼Uniform(T_s)`를 추출한다. Source별 sorted pool의 index를 `random.Random(evaluation-target-seed).randrange(2048)`로 R번 생성한다. Trial 목록은 모든 적용 checkpoint 봉인 뒤 단 한 번 작성해 hash를 남긴다. 반복 digest를 제거하거나 coverage가 높아질 때까지 다시 뽑지 않는다.

같은 source의 pipeline·method·model seed는 동일한 trial 목록을 사용한다. 같은 digest가 재등장해도 trial_id가 다른 새 generation RNG를 사용한다. Random은 source·model seed·trial별로 원래 prior에서 길이와 bytes를 independent replacement draw한다. Target 길이·ownership·validity·이전 성공으로 필터링하지 않는다.

각 trial·method·model seed마다 정확히100 opportunities를 생성한다. Invalid·duplicate·이미 성공한 뒤의 후보도 계수한다. Reverse trajectory 하나가 attempt 하나다. 큰 pool에서 선별, repair, rejection, beam search, guidance·verifier feedback, hash reranking을 하지 않는다.

\[
S_i(K)=1\{\exists j\le K:Valid(\hat x_{ij})\land H12(\hat x_{ij})=Y_i\},
\quad\widehat P_K=R^{-1}\sum_iS_i(K).
\]

통계 단위는 **evaluation trial**이다. Pool 크기2048·trial 수2048·실제 평가한 고유 digest 수를 각각 보고한다. 고유 평가 digest 수 기대값은 약1294.77(63.22%)이며 전수 평가가 아니다. 같은 digest의 여러 trials를 OR로 합치거나 새 digest 여러 개로 세지 않는다.

고정 T_s·training data·checkpoints·model seed 아래의 estimand는

\[
\Delta_T=|T|^{-1}\sum_{y\in T}[P(S_M(100)=1\mid y)-P(S_B(100)=1\mid y)].
\]

불확실성에는 **표적의 random draw와 generation randomness 모두**를 포함한다. 실제 뽑힌 target 목록에 다시 조건을 걸어 다른 estimand로 바꾸지 않는다. 전체4096 digest, 새로운 학습 데이터, 임의 model seed에 대한 일반화는 주장하지 않는다.

Verifier는 bytes가 있는 후보를 duplicate·invalid-domain 여부와 관계없이 실제 MD5 계산하고 호출을 계수한다. Bytes가 없는 invalid는 MD5 호출0이다. Success는 validity를 동시에 요구한다. Synthetic task는 별도 predicate이며 MD5 성공으로 기록하지 않는다.

## 10. 분석·검정력·오류율 검증

### 10.1 추론 모형과 판정

Training data·checkpoints·T를 고정하고 각 trial의 target과 생성 난수가 독립이면 `(M_i,B_i)`는 `Q_T=|T|^-1 Σ_y Q_y`에서 iid다. `π10=P(M=1,B=0)`, `π01=P(M=0,B=1)`일 때 `Δ_T=π10−π01`이다. 따라서 H0:Δ_T≤0 아래 discordant Main-only 비율은≤.5다. 표적별 난이도·우위 방향이 같을 필요는 없다.

Component p-value는 `P[Binomial(n10+n01,.5)≥n10]`인 one-sided exact McNemar tail이다. Discordance0이면 p=1이다. Random target sampling·fresh trial RNG·고정 sampler·checkpoint 선봉인 조건을 위반하면 이 추론을 적용하지 않는다. PRNG를 통계적 난수원으로 취급하는 계산 모형을 명시한다. [Binomial test 공식 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)

Pipeline별 두 controls×세 seeds의6개 p를 max-p로 결합하고, 고정한5개 composite에 Holm을 한 번 적용한다. Matrix 순서가 tie 순서이며 adjusted p<.05를 요구한다. Component가 유효하면 이 결합은 comparisons 간 독립을 요구하지 않는다. Missing은 보정에만p=1, 실제 outcome에는0을 넣지 않는다. [Holm 공식 문서](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)

확증 우위에는 measurement gates·모든6개 observed Δ>0·composite Holm 통과·세 seed 완료가 필요하다. CI 하한은 추가 gate가 아니다. Bootstrap은 source별 같은 trial index를10,000번 재표집해 모든 method·seed·K를 함께 유지한다. Percentile2.5/97.5%, NumPy quantile `linear`다. 근사 marginal CI이며 simultaneous/exact interval은 아니다. 퇴화하면 `DEGENERATE_EMPIRICAL_CI`를 붙인다. 성공0일 때 iid trial 모형의 one-sided upper `1-.05^(1/R)`를 보조 표시할 수 있다.

### 10.2 Calibration을 완료하는 구체적인 방법

Primary 자료·모델 결과를 사용하지 않고 source별 가상 pool2048에서 iid target trials2048을 생성한다. 모든 pipeline에 actual family code를 적용한다. 같은 source의 target schedule과 source·seed별 Random outcomes를 공유하고, 그 외 method·seed의 Bernoulli 결과는 지정 target 확률에 조건부 독립으로 생성한다. Target probability는 **Success@100 확률**이다. `p0=1-(1-1/4096)^100`을 사용한다.

| 시나리오 | 지정 확률 |
|---|---|
| All-null | 모든 Main/Shuffled/Random=p0 |
| Opposite-effect null | 같은 수의 두 strata에서 `(Main,Random,Shuffled)=(.08,.02,.02)` 및 `(.02,.08,.08)`; pool 평균 차이0 |
| One-null-component | 기본 Main=3p0, controls=p0; P-G-BGV의 seed2만 Main=2p0, Shuffled=2p0, Random=p0 |
| Reference twofold | 모든 Main=2p0, controls=p0 |
| One-weaker-seed | Main seeds0/1=2p0, seed2=1.5p0; controls=p0 |
| Stronger-shuffled | Main=2p0, Shuffled=1.5p0, Random=p0 |
| Shared target difficulty | 1024개씩 difficulty .2/1.8; Main=2p0×difficulty, controls=p0×difficulty; 같은 target difficulty를 공유 |

각 시나리오20,000 repetitions, engineering master의 시나리오별 namespace를 사용한다. 실제 five-composite Holm으로 참인 composite null 중 하나 이상을 기각한 FWER와 pipeline별 power, Monte Carlo SE·Wilson interval을 보고한다.

수용 규칙은 null 세 시나리오에서 FWER의 **one-sided95% Wilson upper≤.06**, reference twofold에서 각 pipeline power의 **one-sided95% Wilson lower≥.80**이다. .06은 simulation의 진단 허용치이며 본검정 α=.05를 바꾸는 값이 아니다. 수학적 유효성을 simulation으로 대신하지 않는다. 나머지 sensitivity 시나리오는80% 통과를 요구하지 않고 결과를 보고한다.

Fail이면 구현 산술·공유 구조를 먼저 검사한다. 설계 자체가 목표에 못 미치면 본실험을 차단하고 새 revision에서 R/목표효과/범위를 함께 개정한다. Seed를 바꿔 simulation이 통과할 때까지 재추첨하지 않는다. 기존 v2의80.94%나 v2.1의 단일 null 진단을 v3 calibration PASS로 옮겨 쓰지 않는다.

## 11. 난수·정보 경계·재현성

Study master는2026092403, engineering master는2026092499다. 모든 seed는 다음 배열을 UTF-8 compact JSON(`separators=(',',':')`, `ensure_ascii=False`)으로 직렬화한 SHA-256의 첫8bytes big-endian integer다. 사용하지 않는 필드는 JSON `null`이다.

`[protocol_id, master_seed, task, stage, namespace, source, pipeline, method, model_seed, epoch, unit_id, attempt]`

Trial·epoch·attempt·validation draw 번호는1부터 시작한다. Synthetic case unit_id는 `case:000`처럼 소문자3자리 hex를 쓰며, complement pair ID는 `min(y,y XOR4095)`다. Model seed labels만0/1/2다. Validation corruption의 unit_id는 공개 condition case ID, attempt 필드는1…4 draw 번호로 사용한다.

| 용도 | Master·공유 규칙 |
|---|---|
| Synthetic split/corpus | Engineering; stage=`SYN_DATA`; split은 source도 null, corpus는 source만 구분 |
| Main ownership/corpus | Study; stage=M0; source 구분, pipeline/method/model_seed null; ownership-test/rest/source namespace 분리 |
| Weight initialization | Pilot는 engineering, main은 study; stage·source·pipeline·model_seed 구분, method null |
| Train order | 같은 stage·source·pipeline·model_seed·epoch; Main/Shuffled method null |
| Train noise | Method 포함한 독립 namespace |
| Shuffle | 별도 namespace, model_seed·epoch 포함 |
| Validation noise | Stage·source·pipeline·condition unit_id·draw index 구분; method/model_seed/epoch null |
| P1 trial list | Engineering, source별 공유, pipeline/method/model_seed null |
| Main evaluation targets | Study, M2, source별 공유, pipeline/method/model_seed null |
| Learned generation | 해당 master·task·stage·source·pipeline·method·model_seed·unit_id·attempt |
| Shared Random | Source·model_seed·trial_id·attempt; pipeline null |
| Synthetic 정상/반전 | 같은 case unit_id·attempt seed/state 복제; variant를 seed에 넣지 않음 |
| Profile·calibration·bootstrap | 각각 독립 namespace; bootstrap index는 source별 comparisons에 공유 |

P3의 master는 engineering이지만 final held-out control task이며 data·config·test 접근을 봉인한다. Python `random.Random`은 corpus/ownership/target schedule, NumPy `default_rng`는 simulation/bootstrap, Torch generators는 model noise/sampling에 사용한다. 초기화의 CPU/Torch global state, device RNG·optimizer·permutation state도 checkpoint에 저장한다. Library versions와 PRNG 구현을 기록한다.

Model-facing 정보는12-bit target와 public source/shape/q, checkpoint·sampler·독립 난수뿐이다. Unit IDs는 public trial/case 번호이며 원문 record ID가 아니다. Hidden 길이·suffix·mask·candidate verifier 결과를 seed·batch sorting·cache key에 넣지 않는다. Same-backend recovery는 P1에서 실제로 검사하며 OS·Torch·device 사이의 bitwise 동일성을 보장하지 않는다. [PyTorch 재현성 문서](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

## 12. 저장·checkpoint·중단·재개

기존 단일 Python package `diffusion_hash_inv` 안에 구현한다. 별도 패키지나 외부 실험 관리 서비스가 필요하지 않다. 표준 라이브러리 JSON·hashlib·sqlite3와 기존 NumPy/Torch·codec·통계 primitive를 재사용한다.

Candidate ledger는 run별 SQLite다. WAL, synchronous FULL, 고유 key `(run_id,unit_id,condition_variant,attempt)`를 사용한다. Primary unit은 trial_id, synthetic P2/P3 unit은 case_id다. Random은 source·seed별 별도 run으로 한 번 저장하고 pipeline reports가 참조한다.

각 row는 source/pipeline/method/model_seed, task, unit_id, requested target, variant, attempt, candidate hex 또는 null, byte length, decoder valid/reason, verifier kind/outcome/calls, MD5 calls, RNG identity, checkpoint/config identity, timing을 포함한다. Case의 original target도 필요한 synthetic flip rows에 기록한다. Success는 독립 evaluator가 재계산할 수 있어야 한다.

한 inference batch의 rows를 하나의 transaction으로 commit한다. Generation 직후 insert하되 commit 전 crash rows는 replay 대상이다. 이미 commit된 rows는 바꾸거나 다른 후보로 덮어쓰지 않는다. Raw tensors는 learned run의 첫16 units에서 normal variant의 첫 attempt만 보존한다. 전체 raw pool을 메모리에 누적하지 않는다.

Training checkpoint는100 updates마다·epoch 종료·정상 중단 직전에 저장한다. Weights/Adam state·epoch/permutation·logical update·noise/global/device RNG·validation 선택 상태·config/data hashes를 포함한다. Checkpoint 파일과 checksum을 완성한 뒤 `LATEST.json` pointer를 atomic replace한다. 검증된 최신2개와 best checkpoint를 보존한다. 미완성 파일을 최신 checkpoint로 추정하지 않는다.

Resume는 같은 protocol·code/patch·data·environment·device·batch policy에서만 허용한다. Training의 checkpoint 이후 미확정 연산은 동일 state에서 replay하고 physical compute를 따로 계수한다. Generation은 commit key와 고정 RNG identity에서 이어간다. Replay가 reference와 일치하지 않거나 무결성을 판단할 수 없으면 `INCOMPLETE`이며 유리한 후보를 얻기 위한 재시도를 하지 않는다.

### 실패별 처리

| 사건 | 처리 |
|---|---|
| 일반적인 finite invalid/duplicate 후보 | 정상 attempt로 기록·계속 실행; 후보를 보충하지 않음 |
| Loss/weights의 NaN·Inf 또는 sampler numerical exception | 해당 run `FAILED_NUMERICAL`, 진단·가능한 state 보존; 자동 hyperparameter 변경 없음 |
| Decoder에 들어온 malformed/nonfinite sample | Invalid reason 기록; exception으로 run을 완주 못하면 INCOMPLETE; 누락 rows를0으로 채우지 않음 |
| Profile 중 OOM | 해당 batch 후보를 부적격으로 기록하고 사전 grid의 나머지를 측정 |
| 고정 학습 batch 또는 formal sampling의 OOM | Run 중단·INCOMPLETE; batch 축소/CPU fallback으로 같은 run을 계속하지 않음 |
| Disk·time·memory cap 도달 | 새 batch 전에 안전 중단, 상태 보존·INCOMPLETE; 누적 cap 초기화 금지 |
| 일시적인 process/IO 중단 | 원인 제거 후 같은 설정으로 최대1회 exact resume 시도; 실패하면 INCOMPLETE |
| Config/data/checkpoint checksum 불일치 | `INVALID_INTEGRITY`, 자동 교체·재학습 금지 |
| P3 성능 미달 | 정상적으로 평가를 완료할 수 있으나 해당 pipeline은 본실험 BLOCKED |
| 본실험 우위 비유의 | 정상 과학 결과; 결과를 이유로 재학습·seed 교체·trial 재추첨하지 않음 |

같은 root의 기존 run을 새 실행으로 덮어쓰지 않는다. 재개에는 명시적 `--resume`, 새 revision/독립 시도에는 새 study ID를 사용한다. 중단·실패·재개 이력을 보존한다.

## 13. 필수 산출물과 보고 형식

출력 root는 `local_experiment_archive/runs/<study-id>/`다. 바이너리·원문·candidates·checkpoints는 git에 넣지 않는다. 계획·설정 명세와 간결한 검증 보고서는 version 관리한다.

| 범위 | 필수 파일·내용 |
|---|---|
| Study | `protocol.frozen.json`, `manifest.json`, `gates.json`, `resources.json`, `analysis_validation.json`, `report.md` |
| Exposure/data | `exposure_inventory.json`, `exposure_audit.json`, source별 ownership·dataset·overlap/hash 보고서 |
| P0 | `pilot/P0/checks.json`; 항목별 PASS/FAIL과 실제 검사 수 |
| 각 learned run | `configuration.json`, `metrics.json`, `telemetry.jsonl`, `candidates.sqlite`, checkpoints·checksums·LATEST pointer |
| P2 | 학습 곡선·조건/validity 진단, batch profile, 선택 batch의 복구 검사 |
| P3 | Pipeline·seed별 세 acceptance counts와 판정; 정상/반전 모두 포함 |
| M1 | 모든 적용 best checkpoints와 selection 기록을 연결한 `checkpoints_seal.json` |
| M2 | Source별2048행 trial 목록·hash, 실제 ledger counts·budget 검사 |
| M3 | Component/pipeline 통계 표, CI·Holm·missing 상태·비용·주장 범위 |

Manifest에는 protocol/config/code revision과 uncommitted patch hash, data/ownership/audit/checkpoint hashes, command, environment, device, PRNG·precision·batch, start/end·재개·실패 이력을 기록한다.

Metrics에는 opportunity 수, valid rate·decoder 이유, duplicate rate, actual verifier/MD5 calls, @1/@10/@100, trial 수·고유 target 수, synthetic constraint accuracy, n10/n01/n00/n11·Δ·p·CI를 해당 task에 맞게 기록한다. P1은@100을 관측값으로 만들지 않는다. 측정하지 않은 값은0이 아니라 null과 이유를 기록한다.

## 14. 실행 gate와 단계 간 의존성

| Gate | 증거 | 적용 시점 |
|---|---|---|
| E: Environment | MPS·versions·codec/model signatures | P1 전에 |
| B: Boundary/accounting | 원문 정보 경계·shuffle·prefix·negative fixtures | P1 완료 시 |
| R: Recovery/resources | Same-backend 중단·재개, 선택 batch, 실측 예산 | P3 전에 |
| C: Learned control | Pipeline의 P3 세 seed acceptance 통과 | 해당 pipeline M0/M1 전에 |
| D: Data/exposure | 최종 source별 audit·ownership·corpus·overlap0 | 해당 source M0 완료 시 |
| A: Analysis | 개정 family code·모형·calibration·power 검사 | M0 완료 시 |
| S: Checkpoint seal | 모든 적용 M1 checkpoint 선택·봉인 | Trial 목록과 M2 시작 전에 |
| L: Actual ledger | 실제 예산·정렬·key·verifier·missing 검사 | 각 평가 완료 후 |
| F: Scientific finding | 여섯 효과·max-p/Holm·세 seed 완료 | M3에서만 |

P0–P2는 D/A가 아직 없어도 독립 개발자료로 수행할 수 있다. P3도 합성 과제로 독립 수행 가능하나, 본실험을 목표로 한다면 비용이 큰 P3 전에 exposure feasibility와 analysis 검사를 끝내는 것이 권장 순서다. F를 Pilot 통과나 본실험 착수의 선행 조건으로 요구하지 않는다.

## 15. CLI 구현 상태와 명령 순서

**현재 `pilot` P0/P1/P2/P3와 Pilot용 `report`가 실행 가능하다.** `audit`, 통계 `validate`, `prepare`, 본실험 `train/evaluate` 및 본실험 통계 report는 미구현이다. 이전 `experiment_cli`에 v3 JSON을 전달하지 않는다. CPU 개발 점검과 `--dry-run` 등의 운영 옵션은 별도 CLI 사용 문서에 정리했다.

| 명령 | 역할 |
|---|---|
| `pilot --stage P0/P1/P2/P3` | 지정 Pilot 단계와 그 단계의 필수 보고서 생성; 선행 gate 확인 |
| `validate --suite statistics` | §10.2의 실제 family calibration, 결과·seed 기록 |
| `audit --inventory PATH` | Exposure 입력 검증·metadata audit·source별 제외 목록 확정 |
| `prepare` | 적격 pipeline 목록과 data·analysis·resource·code/config 봉인, M0 |
| `train` | 적격 pipeline의 지정 Main/Shuffled·세 seeds 학습 및 전체 checkpoint seal, M1 |
| `evaluate` | Trial 목록 한 번 생성, 모든 적용 candidates·shared Random 평가, M2 |
| `report` | 산출물 검사, Pilot/main 상태·통계·비용 보고, M3; 실패·partial에도 사용 |

공통 인자는 `--protocol`, `--workdir`이며 GPU 실행 명령은 `--device mps`를 받는다. `--resume`은 기존 동일 run의 검증된 상태를 이어가는 명시적 옵션이다. 임의 epoch/seed/K override나 성능 기반 pipeline 선택 옵션은 제공하지 않는다. 설정 변경은 protocol revision으로 처리한다.

현재 사용할 수 있는 Pilot 명령:

```bash
# 각 단계의 완료·무결성·gate를 확인한 뒤 다음 명령을 실행한다.
DHI_ROOT="/Users/choisoonwook/Experiments_local/DHI_AI_gen"
DHI_PY="$DHI_ROOT/.venv/bin/python"
DHI_SPEC="$DHI_ROOT/examples/poc-v3-protocol.json"
DHI_RUN="$DHI_ROOT/local_experiment_archive/runs/v3-pilot-$(date +%Y%m%d-%H%M%S)"

"$DHI_PY" -m diffusion_hash_inv.study_cli pilot --stage P0 --protocol "$DHI_SPEC" --workdir "$DHI_RUN" --device mps
"$DHI_PY" -m diffusion_hash_inv.study_cli pilot --stage P1 --protocol "$DHI_SPEC" --workdir "$DHI_RUN" --device mps
"$DHI_PY" -m diffusion_hash_inv.study_cli pilot --stage P2 --protocol "$DHI_SPEC" --workdir "$DHI_RUN" --device mps
```

위 각 단계의 report와 gate를 확인한 다음 진행한다. P3는 `pilot --stage P3`으로 독립 실행할 수 있다. **아래 블록은 본실험까지 연결하는 향후 실행 순서이며 아직 통째로 실행할 수 없다.** Inventory는 §7.1의 실제 이력을 담아 준비해야 하며, 빈 목록으로 audit를 우회하지 않는다.

```bash
# NOT IMPLEMENTED: 구현 완료 후 앞 단계 통과를 확인하며 한 명령씩 실행한다.
"$DHI_PY" -m diffusion_hash_inv.study_cli audit --inventory "$DHI_RUN/exposure_inventory.json" --protocol "$DHI_SPEC" --workdir "$DHI_RUN"
"$DHI_PY" -m diffusion_hash_inv.study_cli validate --suite statistics --protocol "$DHI_SPEC" --workdir "$DHI_RUN"
"$DHI_PY" -m diffusion_hash_inv.study_cli pilot --stage P3 --protocol "$DHI_SPEC" --workdir "$DHI_RUN" --device mps
"$DHI_PY" -m diffusion_hash_inv.study_cli prepare --protocol "$DHI_SPEC" --workdir "$DHI_RUN"
"$DHI_PY" -m diffusion_hash_inv.study_cli train --protocol "$DHI_SPEC" --workdir "$DHI_RUN" --device mps
"$DHI_PY" -m diffusion_hash_inv.study_cli evaluate --protocol "$DHI_SPEC" --workdir "$DHI_RUN" --device mps
"$DHI_PY" -m diffusion_hash_inv.study_cli report --protocol "$DHI_SPEC" --workdir "$DHI_RUN"
```

Pilot의 재개 예시는 `pilot --stage P3 ... --resume`다. `train/evaluate --resume`은 향후 본실험 구현 계약이다. [기존 GPU smoke 절차](/Users/choisoonwook/Experiments_local/DHI_AI_gen/GPU_EXECUTION.md)는 v3 gate를 대체하지 않는다.

Exit codes는0=요청 단계 계약 통과 또는 유효한 report 생성,2=설정/선행 gate/적격성 미충족,3=runtime/numerical error,4=무결성·resume 불일치,5=resource cap,130=사용자 중단으로 정한다. Report의 exit0은 과학적 우위나 전체 study 완료를 의미하지 않는다. P3의 전체 적격성 미충족은 exit2와 pipeline별 상태를 남기며, 적격 subset의 후속 진행은 고정 family5와 partial 상태로 관리한다.

## 16. 실행량·보고 판정·계획 검산

### 16.1 전체 matrix의 최초 실행량

아래는 실패·재개 replay·추가 development revision을 제외한 기본 workload다. Profile은 별도 행으로 표시한다.

| 구분 | 학습 runs | Optimizer updates 합 | Learned candidates | Random candidates | Sampling NFE |
|---|---:|---:|---:|---:|---:|
| P1 | 10 | 80 | 1,600 | 320 | 116,480 |
| P2 probe | 5 | 7,850 | 1,280 | 0 | 93,184 |
| P3 | 15 | 235,500 | 15,360 | 0 | 1,118,208 |
| MD5 본실험 | 30 | 471,000 | 6,144,000 | 1,228,800 | 447,283,200 |
| 기본 합계 | **60** | **714,430** | **6,162,240** | **1,229,120** | **448,611,072** |
| P2 batch profile 추가 | 새 학습 없음 | 0 | 1,700 | 0 | 123,760 |

기본 candidate rows는7,391,360개, profile까지7,393,060개다. Validation corruption sample evaluations는944,640회이며 sampling NFE와 다른 단위다. 재개 검사의 reference/replay·추가 codec timing fixture 비용은 telemetry에 더하고 정책 cap 안에 포함한다. NFE는 GPU API call 수나 MD5 호출 수가 아니다.

### 16.2 최종 결과 문구

| 결과 | 보고 문구 |
|---|---|
| P0/P1/P2 완료, P3 미실행 | “기술 Pilot 완료; 학습 적격성은 미검증” |
| P3 평가 완료, 일부/전체 문턱 미달 | “Pilot 평가 완료; 해당 pipeline은 본실험 부적격” |
| 본실험 measurement·모든 seeds 완료, Holm 미통과 | “지정 budget에서 확증적 우위를 확인하지 못함” |
| 모든 관련 gates·여섯 양의 효과·Holm·세 seeds 통과 | “봉인한 pool·checkpoints·지정 seeds에서 양쪽 control 대비 평균 우위가 지지됨” |
| Missing·무결성 실패 | “미완료/무효”; 실제 outcomes를0으로 채워 통계적 음성 결과로 만들지 않음 |

Pilot의 기술적 성공, 학습 적격성, 본실험 완료, 가설 성공을 별도 열로 보고한다. 비유의로 작은 효과·새 데이터에서의 가능성·모델 계열 전체의 불가능성을 단정하지 않는다. 후속 compute study에서는 전처리 table·random search·학습 상각·memory·wall time을 별도 사전등록으로 비교한다.

### 16.3 이번 문서 작성에서의 검증 범위

이번 작성에서 JSON 파싱, 단계별 quota/update/candidate/NFE 산술, 현재 모델을 CPU에 생성한 parameter count, 문서의 합계·로컬 파일 참조를 검사해 PASS를 확인했다. 기본714,430 updates·7,391,360 candidate rows·944,640 validation sample evaluations 및 profile1,700 candidates를 독립 계산과 대조했다. 통계 단위·후보 예산·stage dependencies·CLI 상태 구분도 문서상 검토했다.

이 검산은 구현·GPU 학습·control·통계 calibration·본평가의 PASS를 부여하지 않는다. 실제 실험 완료 상태는 모두 `NOT_RUN` 또는 `NOT_IMPLEMENTED`에서 시작한다.

삭제된 pilot과 기존 v2 자료를 새 v3 실행 결과로 승격하지 않는다. 구현자는 이 문서와 JSON을 기준으로 최소 실행 경로를 만든 뒤 P0부터 증거를 생성해야 한다.
