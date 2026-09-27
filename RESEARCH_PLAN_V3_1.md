# v3.1 실험 계획 — PoC 검증과 본실험의 정상 완료

**Protocol:** `dhi-v3.1-20260925` · **Revision:** 3.1 · **작성일:** 2026-09-25 KST  
**상태:** 계획 작성 완료 / 수정 모델·개발 P0/P1/P2 구현 / 정식 PoC·본실험 미실행
**구현 현황(2026-09-26):** [V3_1_IMPLEMENTATION.md](V3_1_IMPLEMENTATION.md)  
**기계 판독 명세:** [poc-v3.1-protocol.json](examples/poc-v3.1-protocol.json)  
**근거:** [v3 실패 상세 분석](LATEST_EXPERIMENT_ANALYSIS_KO.md), [개선 제안](EXPERIMENT_IMPROVEMENT_PROPOSAL_KO.md)

## 1. 목표와 완료의 정의

v3.1의 목적은 **다섯 pipeline 전체에서 PoC를 검증하고, 본실험의 학습·평가·통계 분석·비용 보고를 정상적으로 끝내는 것**이다. 본실험에서 random보다 유의하게 우수해야 실험 완료로 인정하는 것은 아니다. 성능 부적격·비유의·실행 실패를 구분한다.

이 계획은 실험 시행에 필요한 구현과 검증의 계약이다. 현재 CLI는 고정된 3.0/3.1 JSON과 v3.1 개발 P0/P1/P2를 지원한다. P3 이후 및 정식 v3.1 실행은 미구현 계약이 남아 있어 차단된다. 개발 검사나 JSON 검산으로 정식 실험 gate가 PASS가 되지 않는다. 기존 v3의 결과·checkpoint·누적 시간을 수정하지 않는다.

| 상태 | v3.1 정의 |
|---|---|
| `IMPLEMENTATION_READY` | v3.1 schema, 모든 등록 모델 profile, Pilot, 실제 MD5 학습/평가, 감사, 통계 및 보고 경로의 필수 검사가 구현·통과 |
| `TECHNICAL_READY` | P0/P1 기술 검사와 선택한 profile의 P2 자원 검증 통과 |
| `DEVELOPMENT_READY` | 다섯 pipeline 모두 P2의 학습·실제 생성 기준을 만족하는 profile이 정해짐 |
| `E0_COMPLETE` | 실제 본실험 실행기로 수행한 작은 MD5 리허설이 무결하게 완료; hash 성공률은 조건 아님 |
| `POC_EVALUATION_COMPLETE` | E0 완료 및 P3의 지정 15개 모델 평가·보고 완료; 성능 미달도 기록 |
| `POC_QUALIFIED` | 앞의 기술·개발·리허설 조건과 P3의 15개 모델 성능 기준 모두 충족 |
| `MAIN_READY` | POC_QUALIFIED 및 노출·데이터·통계 calibration·코드·자원·profile 봉인 완료 |
| `STUDY_COMPLETE` | 전체 본실험 30 learned runs와 6 shared Random streams의 후보·통계·비용 보고가 무결하게 완료 |
| `POSITIVE_FINDING` | 완료와 별도로, 해당 pipeline의 여섯 효과가 모두 양수이고 고정 family의 Holm 기준 통과 |

일부 pipeline만 적격이거나 일부 run이 미완료면 `PARTIAL`/`BLOCKED`를 명시한다. **v3.1에서는 적격 subset만 실행하고 전체 완료로 처리하지 않는다.** 목표 범위를 줄이려면 별도 revision이 필요하다. 계획된 개선이 반드시 성능 gate를 통과한다고 보장하지 않는다.

## 2. v3 실패를 반영한 변경

| v3에서 확인한 문제 | v3.1 조치 |
|---|---|
| 연산 시간으로 전체 run 시간을 예측해 P-DISC 중단 | 전체 update/batch 주기 측정, 디렉터리 검사 비용 축소, soft 예상치와 hard 중단 상한 분리 |
| joint 성공 0을 `LEARNING_SIGNAL_ABSENT`로 표시 | `JOINT_SUCCESS_ZERO`, 형식 실패, 조건 반응을 별도 보고; argmax와 실제 sampling 구분 |
| P2 품질 경고 후 바로 큰 P3로 진행 | 작은 과적합 검사 P2A와 제한된 profile 개발 P2B 추가; 최종 실제 생성 기준 미달이면 P3 차단 |
| Gaussian의 길이·mask 불일치 및 고잡음 복원 취약 | 기존 모델에 좌표 입력을 추가한 profile, x0 예측 profile을 사전 등록한 순서로만 검사 |
| Discrete가 prefix를 알아도 EOS/PAD 생성 실패 | 기본 token 모델과 길이를 먼저 생성하는 factorized 모델을 구분하여 검사 |
| P3 중단 시 완료한 모델 결과가 보고서에서 누락 | 각 run의 seal을 읽어 완료·실패·미착수를 즉시 표시 |
| 본실험 실행기 및 calibration이 미구현 | 구현을 초기 필수 작업으로 포함하고 실제 MD5 리허설 E0를 P3 전에 수행 |

P0/P1은 기술 검증 역할을 유지한다. P2의 초반 성능 0만으로 학습 불가능성을 판정하지 않는다. P3의 기존 475/512 기준은 유지한다.

## 3. 유지하는 연구 범위와 본실험 고정값

| ID | Source | 표현 | 후보 profile 순서 | Formal seeds |
|---|---|---|---|---|
| P-G-BGV | Printable | BGV | G0 → G1 → G2 | 0/1/2 |
| P-G-CGGE | Printable | CGGE | G0 → G1 → G2 | 0/1/2 |
| P-DISC | Printable | Token | D0 → D1 | 0/1/2 |
| R-G-BGV | Random Bytes | BGV | G0 → G1 → G2 | 0/1/2 |
| R-DISC | Random Bytes | Token | D0 → D1 | 0/1/2 |

길이 L은 4–31에서 균등, Printable bytes는 33–126, Random Bytes는 0–255에서 주어진 길이에 조건부 iid uniform이다. `H12(x)=Int_big(MD5(payload_bytes)) >> 116`이며 입력 digest는 leading zero를 포함한 12 bits다. 성공은 source-valid payload가 target prefix를 맞히는 것이다. 대표 원문의 길이나 내용과 일치할 필요는 없다.

BGV `[2,32,128]`, CGGE `[2,32,64]`, Token `[32]`와 strict decoder는 유지한다. BGV bit/mask threshold .5, 길이 4–31, length/mask/padding 일치; CGGE 연속 mask, 32번째 cell unused, prototype MSE≤.1; Token은 정확히 한 EOS와 이후 PAD를 요구한다. CGGE font는 v3와 같은 checksum을 사용한다. Source 밖 bytes도 invalid다. 숨은 원문·길이·full digest·검증 결과를 생성 입력에 넣지 않는다.

본실험 source별 train digest slots 1,536, validation 512, test pool 2,048, train unique messages 10,000을 유지한다. 학습은 Main/Shuffled × seeds0/1/2, 전체 30 models다. 평가는 source별 독립 균등 복원추출 trial 2,048개, trial당 K=100이다. Random은 source×seed별 6 streams를 같은 source의 pipelines가 공유한다. Primary는 Success@100, @1/@10은 탐색적이다.

## 4. 구현할 최소 모델 변경과 profile 선택

새 diffusion 프레임워크나 새 외부 dependency를 도입하지 않는다. `GaussianDiffusion`, `ImageUNet`, `SequenceDenoiser`, `MaskedDiffusion`, 기존 codec·통계 primitive를 재사용한다.

### 4.1 Gaussian

모든 Gaussian profile은 width32, 출력 channels2, T=1,000, linear beta .0001→.02, sampling100이다. Clean image는 `2*image-1`, 최종 clip은 [-1,1]이다.

| Profile | 위치 정보 | 학습/validation target | Sampling의 clean 추정 |
|---|---|---|---|
| G0 | 없음 | epsilon | v3와 동일; 중간 clip 없음 |
| G1 | 고정 x/y 두 채널 | epsilon | G0와 동일 |
| G2 | 고정 x/y 두 채널 | clean image x0 | 각 step의 x0를 [-1,1]로 clip하고 epsilon을 재구성 |

좌표는 각 축의 -1…1 linspace를 모든 sample에 동일하게 붙인다. 생성 대상·noise 대상·loss 대상은 기존 두 image channels뿐이다. 원문 이미지나 길이를 spatial condition으로 넣지 않는다. 초기 noise, time grid, output shape는 유지한다.

Epsilon profile은 `x0_hat=(xt-sqrt(1-a)*eps_hat)/sqrt(a)`, G2는 `x0_hat=clip(model_output)` 및 `eps_hat=(xt-sqrt(a)*x0_hat)/sqrt(1-a)`를 사용한다. 다음 step은 기존 deterministic DDIM식이며 마지막 이전 alpha는 1이다. **모델 설정·학습 target·validation target·Pilot 및 본실험 sampler를 함께 일치시킨다.** G2는 objective와 중간 clipping을 함께 바꾼 profile이며 결과를 objective 단독 효과로 해석하지 않는다.

좌표 입력 방법은 [CoordConv 원 논문](https://arxiv.org/abs/1807.03247), sampling의 기본 배경은 [DDIM 원 논문](https://arxiv.org/abs/2010.02502)을 참고했다. 이 자료가 본 과제에서의 개선을 보장하지는 않는다.

### 4.2 Discrete

D0는 v3의 width128/embedding16, 32 intervals, temperature1, masked CE, EOS/PAD 동시 생성과 reveal 후 remask 없음 정책을 유지한다. MPS embedding은 검증된 one-hot matmul 경로를 유지한다.

D1은 `p(L|y) × p(payload|L,y)`로 구성한다. 길이 head는 `Linear(12,28)`이고 4…31 중 하나를 temperature1 categorical로 한 번 뽑는다. Payload denoiser는 기존 SequenceDenoiser를 사용하되 condition에 `L/31`을 추가한다. 기존 TokenCodec의 32개 위치와 vocabulary를 유지한다.

학습에서는 training message의 관측 길이를 길이 CE target 및 payload denoiser의 조건으로 쓴다. 첫 L개 payload 위치만 uniform-time MASK corruption을 하고 masked payload CE를 계산한다. Loss는 길이 CE와 sequence별 masked payload CE의 합이며 가중치는 각각 1이다. Masked payload가 없으면 그 항만 0이다. EOS/PAD 위치는 길이에 따라 고정 context다.

생성에서는 **모델이 뽑은 L**을 사용하고 첫 L개 위치에서 payload vocabulary만 sampling한다. EOS를 위치 L에, PAD를 이후 위치에 둔다. 이는 D1의 생성 모델 정의이며 기존 D0의 실패 출력을 사후 수선하는 절차가 아니다. 숨은 정답 길이는 쓰지 않는다. Length RNG와 payload RNG를 분리하고 둘 다 기록한다. 정상/반전은 각 RNG의 초기 상태를 공유한다.

D1에서는 형식 유효성이 대부분 구조적으로 보장되므로 유효율 상승을 학습 능력의 증거로 주장하지 않는다. 학습 적격성은 정상/반전 prefix의 실제 sampling 성공으로 검사한다. Main/Shuffled에 같은 구조를 적용하고 Random도 동일 source의 정상적인 길이·bytes 분포를 사용한다. D1은 후보당 length head 1회와 payload denoiser32회, 총 NFE33으로 계수한다. D0는32다. Profile이 다른 pipeline의 차이를 순수한 표현 효과로 해석하지 않는다.

### 4.3 선택 규칙과 공통 학습

Pipeline별로 표의 순서에 따라 P2A→P2B→자원 조건을 만족하는 **첫 profile**을 선택한다. G0가 통과하면 G1/G2는 `NOT_NEEDED`, D0가 통과하면 D1은 `NOT_NEEDED`다. 전체 후보-pipeline 조합은 최대13개다. 모든 후보가 실패하면 `BLOCKED_DEVELOPMENT`로 남기며, 새 hyperparameter/seed를 즉석에서 추가하지 않는다.

Adam lr=.001, betas=.9/.999, eps=1e-8, weight_decay0, foreach/fused false, float32, 기본 batch64다. Epoch별 train permutation을 한 번씩 사용하고 마지막 작은 batch를 유지한다. P2A만 batch16·고정 updates를 쓴다. Validation은 condition당 고정 corruption draws4개, 최소 validation loss checkpoint를 선택하며 정확한 동률은 이른 epoch다. D1 validation loss는 길이 CE+payload masked CE다. MD5 성공으로 checkpoint를 고르지 않는다.

Main/Shuffled는 같은 profile·초기 weights·train order를 사용하고 Shuffled의 condition donor만 epoch별 train 내부에서 바꾼다. Validation/inference는 실제 target을 사용한다. P1, P2A, P2B, E0, P3, 본실험 사이에 weights/optimizer를 이어 쓰지 않는다. P2B의 10→30→100 epochs만 동일 개발 run의 연속 학습이다.

## 5. Synthetic 자료와 노출 범위

Synthetic task는 12-bit y의 세 nibble을 Printable `0123456789abcdef` 또는 Random Bytes 0…15로 첫 세 payload 위치에 넣고, 길이·나머지 payload는 원래 source prior에서 생성하는 `synthetic_nibbles`다.

4096 conditions를 `(y,y XOR4095)`의 2048 complement pairs로 묶는다. v3.1 seed로 한 번 섞어 train1536 pairs=3072 conditions, validation256 pairs=512, test256 pairs=512로 분할한다. 모든 source/profile/seeds가 condition split을 공유한다. Source별 train unique messages10,000, validation 대표512개를 만들고 draw cap은1,000,000이다. Source·split별 raw/digest/condition 경계와 checksum을 검사한다.

v3 및 이번 진단에서 본 자료를 exposure inventory에 기록한다. **새 split seed가 프로젝트 전체에서 처음 보는 12-bit 조건을 보장하지 않는다.** v3.1 P3는 현재 revision의 train/validation과 condition-disjoint이며 profile 선택에 쓰지 않은 fixed-case engineering acceptance다. 프로젝트 전체의 최초 미관측 조건 검증 또는 모집단 동시 신뢰 보장을 주장하지 않는다. P3 접근 전 profile·코드·자료·자원을 봉인하고, P3 결과를 본 뒤 같은 test로 profile을 다시 선택하지 않는다.

## 6. P0–P3 및 E0 실행 절차

| 순서 | 단계 | 실행 규모 | 완료/진입 기준 |
|---|---|---|---|
| 1 | P0 | Codec1,214개, 등록 profile-pipeline13개의 shape/parameter/forward 검사; 학습 없음 | 환경·입력 경계·sampler/통계·예산 fixture 모두 PASS |
| 2 | P1 | 13조합×Main/Shuffled×seed0=26 models; train256/val32;2 epochs=8 updates/run | Finite 실행, 저장·중단/복구, ledger PASS; 품질 문턱 없음 |
| 3 | P2A | 후보별16 training cases; seed99;batch16;2,000 updates | 정상/반전 각각≥61/64, 원래 조건 오성공≤1/64 |
| 4 | P2B | P2A 통과 후보별 train10,000/val512;seed100;100 epochs | 마지막 개발 평가와 자원 기준 PASS인 첫 profile 선택 |
| 5 | E0 | 선택5 pipelines×Main/Shuffled×seed0=10 MD5 models;2 epochs | 실제 본실험 경로·재개·계수·독립 재검증 PASS; hash 성공률 기준 없음 |
| 6 | P3 | 선택5 pipelines×seeds0/1/2=15 fresh models;100 epochs | 모든 seed의 정상/반전≥475/512, 원래 조건 오성공≤15/512 |

### P0/P1: 고장과 낮은 성능 구분

P0에는 기존 clean/negative codec·MD5 vector·leading-zero·정보 경계·McNemar/max-p/Holm·missing/duplicate fixtures를 유지한다. 추가로 G2 parameterization oracle, D1 길이 입력 경계, profile별 batch/single trajectory 일치, 동일 digest의 서로 다른 trial IDs, full-loop 비용 계수, 누적 resume 및 부분 보고를 검사한다. Parameter count와 코드 hash는 실제 구현 후 freeze 파일에 기록하고 불일치 시 차단한다. 과거 source hashes는 참고값이다.

P1의 각 모델은 validation pool32에서 복원추출한16 trials×K10, inference batch4로 평가한다. Source별 pool과 trial 목록은 모든 profile/method가 공유하고 Random도 source별160개씩 공유한다. 각 profile의 Main에서 update5 직후와 첫 trial attempt7의 commit 전 중단을 검증한다. Reference와 재개의 weights/optimizer·logical updates·선택 checkpoint·decoded bytes·validity·검증 결과 및 ledger keys가 일치해야 한다. 시간은 결과 동일성 비교에서 제외하되 소비량은 계수한다. D1은 길이 sampling 상태도 비교한다.

### P2A: 작은 자료에서 실제로 생성하는가

Training split의 고정 pair 순서에서 양쪽 complement가 train corpus에 실제 존재하는 첫8쌍을 선택하고, condition별 최초 training record 하나씩16개를 고정한다. 이 선택은 모델 결과를 보지 않고 source별 한 번만 수행하여 모든 profile이 공유한다. 8쌍보다 적으면 데이터 구성 단계에서 실패 처리한다. 최종 update2,000 weights로16 conditions×고정 trajectories4개×정상/반전2개를 생성한다. 각 trajectory는 K1이며 4개 중 하나만 맞으면 성공으로 세지 않는다.

정상/반전의 각64회 중61회 이상이 strict valid이면서 조건을 맞혀야 하고 반전 시 원래 조건 오성공은1회 이하여야 한다. 이 문턱은 작은 학습 사례의 engineering 진단이며 독립 일반화 인증이 아니다. Argmax 검사·oracle 출력·정답 prefix 삽입은 통과를 대신하지 않는다.

### P2B: 개발 자료에서 생성 품질과 자원 검증

Fresh seed100으로100 epochs·15,700 updates를 실행한다. Validation loss는 매10 epochs 측정한다. Validation pairs 중 고정64쌍=128 conditions에 대해 epoch10/30/100에 best-so-far checkpoint로 정상/반전 각각 K1 생성한다. 후보당 총768개이며 모든 profile이 같은 case·generation 난수를 사용한다.

Epoch10/30에서 joint0이면 경고와 분해 지표를 남기고 등록한100 epochs까지 진행한다. **P3 진입 판정은 epoch100에서 선택된 checkpoint의 평가만 사용한다.** 정상·반전 각각128개 중 strict joint≥122, valid≥122, 반전의 원래 조건 오성공≤3이어야 한다. 이 개발 기준은 문서상 사전 고정된 engineering screen이며 P3의 별도 기준을 대체하지 않는다. 여러 checkpoint 중 generation 점수가 좋은 것을 다시 고르지 않는다.

필수 진단은 strict valid/decoder 이유, first-prefix 위치별 정확도, valid일 때의 constraint 정확도, 실제 normal/flipped joint, 원래 조건 오성공이다. Valid가0이면 conditional accuracy는 null이다. 별도로 Gaussian validation16개에서 t=0/100/300/500/700/999의 noise/x0/영역별 오차·clipping 비율, Discrete validation64개에서 mask fraction .1/.5/.9/1의 logits·정답 확률·EOS 분포를 기록한다. 진단의 argmax/복원값을 공식 candidate 성공에 넣지 않는다.

실제 sampling 품질을 통과한 후보의 batch1/4/16/64를 각각 warm-up1회·측정3회 profile한다. 예외·비유한 연산·메모리 초과가 없으며 처리량이 가장 높은 batch를 선택하고 1% 이내는 작은 batch를 택한다. 품질로 batch를 선택하지 않는다. Pipeline별로 선택한 architecture/objective/sampler/batch를 E0 전에 `profile.frozen.json`에 봉인한다. 선택 profile의 추가 생성 복구도 검사한다.

### E0: 실제 MD5 실행기의 작은 전체 경로 리허설

E0는 synthetic 평가를 MD5 평가로 잘못 기록하는 오류와 본실험 전용 코드의 미검증을 막기 위한 단계다. Production의 train/evaluate/report 함수를 그대로 사용하고 engineering task namespace·규모만 다르게 한다.

Source별 감사된 이미 노출된 prefixes에서 train64/validation32/test32를 서로 다르게 지정한다. 128개가 부족하면 부족분을 별도로 예약해 E_s에 추가하고 **그 후** main test pool의 가용성≥2048을 확인한다. Main test pool은 아직 선택·공개하지 않는다. Source prior를 draw해 train owner의 unique messages256개와 validation/test의 첫 대표를 모으며 cap은 source별1,000,000이다. E0의 train·validation·test에 사용한 모든 prefix를 main test에서 제외한다.

선택한5 profiles로 Main/Shuffled×seed0, train256·2 epochs·8 updates/run, validation32×4 draws를 사용한다. Source별 test pool32에서 iid replacement trials32개, K100을 고정하고 Random은 source별 공유한다. Checkpoints 봉인 뒤 trial 목록을 생성한다. 성능이0이어도 ledger·검증·보고가 맞으면 기술적으로 완료된다. Formal family의 세 seeds가 없으므로 과학적 판정은 `NOT_APPLICABLE_ENGINEERING_REHEARSAL`이다.

모든 선택 profile의 same-backend 학습/생성 복구, invalid·duplicate·성공 후 attempt 계수, 실제 MD5 재계산, 같은 target의 다른 trials, 실패/부분 보고를 확인한다. Pipeline/profile별 CPU decode·MD5·SQLite 비용과 최대 길이/row 크기 stress fixture를 측정해 본실험 자원 추정을 완성한다. E0 자료/weights/성과는 main 학습·선택에 재사용하지 않는다.

### P3: 최종 합성 적격성

E0, 최종 자원 봉인, source exposure feasibility, §9의 calibration이 통과한 뒤 시작한다. Fresh seeds0/1/2를 모두100 epochs 학습하고 최소 고정 validation loss로 checkpoint를 선택한다. Test512 conditions에서 정상/반전 각각 K1, 같은 초기 RNG를 사용한다. 각 seed에서 정상 joint≥475, 반전 joint≥475, 원래 조건 오성공≤15를 모두 요구한다.

Finite 성능 미달로 나머지 seeds를 생략하지 않는다. D1의 높은 형식 유효성만으로 적격성을 부여하지 않는다. 다섯 pipeline 중 하나라도 P3 미달이면 본실험은 `BLOCKED_QUALIFICATION`이며 현재 전체 범위의 목적은 아직 달성되지 않은 상태다.

## 7. 시간·메모리·저장량과 중단 정책

MPS를 명시적으로 요청하고 GPU 동시 run은1개다. Silent CPU fallback, mixed precision, run 도중 batch/seed/epochs 변경은 하지 않는다. CPU 진단 결과를 formal MPS 복구의 증거로 쓰지 않는다.

### 7.1 측정과 soft/hard 구분

Warm-up20 updates 후 연속100 updates의 전체 경과 시간을 세 구간 측정한다. Batch 준비·budget check·상태 저장·telemetry·동기화·연산을 포함하고 세 구간 중 가장 큰 update 평균을 사용한다. Validation/checkpoint는 구간에 포함했다면 다시 더하지 않으며, 제외했다면 별도로 전량 추가한다. Generation도 모델 호출부터 transfer·decode·MD5·ledger commit·검사까지 같은 원칙으로 측정한다. 각 비용의 포함 범위를 machine-readable 항목으로 남긴다.

완료 run이 쌓인 디렉터리 fixture와 최대 길이 유효 decoder/ledger fixture도 측정한다. P2에서는 P3와 main을 잠정 추정하고, E0 실측으로 main 비용을 보완해 P3 전에 최종 봉인한다. 예상 시간×1.5를 soft 예산으로, 예상 저장량×2를 필요한 저장량으로 기록한다. 실측치가 없으면 null/NOT_MEASURED이며 0으로 대체하지 않는다.

**Soft 예산 초과는 `RESOURCE_ESTIMATE_EXCEEDED` 경고와 관측값을 남기고 사전 hard cap 이내에서 계속한다.** 기존242초 같은 예상치 초과만으로 정상 run을 중단하지 않는다. Hard cap은 성능과 무관하게 유지한다. Formal 진행 전에 예상치가 hard cap 또는 남은 저장량을 넘으면 `BLOCKED_RESOURCE`다. 실행 중 성능을 보고 상한을 늘리지 않는다.

| Hard active-time cap | 값 |
|---|---:|
| P0 | 10분 |
| P1 전체, 복구 포함 | 30분 |
| P2A/P2B·후보 profile 탐색·profile/복구 전체 | 24시간 |
| E0 | 4시간 |
| P3 | 48시간 |
| Statistics calibration | 4시간 |
| PoC 전체 active time | 96시간 |
| 본실험 M0–M3 전체 | 168시간 |
| Learned run 하나 | 24시간; stage 잔여 상한도 적용 |

Study64GiB, run8GiB, process RSS64GiB, MPS allocated64GiB, 디스크 여유10GiB를 유지한다. RSS/MPS는 겹칠 수 있으므로 합산하지 않는다. 저장량은 main의 실제 예정 row 수×E0 및 큰 ledger fixture에서 구한 보수적 row 비용, index/WAL/checkpoints/raw/telemetry를 포함해 추정한다. P2의 작은 총 디스크 사용량에 단순히 seed 수만 곱하지 않는다.

### 7.2 비용을 줄이되 복구를 유지

시간·메모리의 가벼운 검사는 매 update/batch에 유지한다. 전체 파일 재귀 순회는 큰 쓰기 전·checkpoint·stage 경계에서 수행하고 예정 쓰기의 상한 크기를 사전 예약해 검사 사이에 저장 cap을 우회하지 못하게 한다. Atomic pointer, fsync, SQLite WAL/FULL은 유지한다. 복잡한 외부 실험 관리 서비스는 추가하지 않는다.

누적 active time은 재개 시 초기화하지 않으며 checkpoint 이후 replay한 물리 연산도 포함한다. 마지막 상태 저장 이후 미계수 작업의 보수적 비용을 복구 기록에 반영하고, 그 비용을 판단할 수 없으면 정확한 budget continuation으로 승인하지 않는다. Process 중단 시간과 계산 시간은 구분한다.

Run 고유의 numerical/time 오류는 해당 run을 보존하고, 공통 무결성·global cap·backend 상태가 정상일 때 독립적인 나머지 지정 run을 수행한다. Global disk/memory/stage cap 또는 공통 데이터 무결성 오류이면 stage를 중단한다. 유리한 결과를 얻기 위한 재시도는 없으며 일시 오류의 exact resume는 run당 최대1회다. Hard cap을 이미 소진한 run은 resume로 해결하지 않는다. 어느 경우든 missing outcomes는0으로 채우지 않는다.

## 8. 본실험 데이터와 M0–M3

노출 감사는 초기 구현 단계부터 시작하고 E0 후 최종 확정한다. 이전 MD5 learned validation/test의 q≥12 target은 source별 E_s에 포함하며 q<12의 raw/full digest 사용, 외부·삭제 자료도 조사한다. 파일 부재를 미노출 증거로 쓰지 않는다. 미해결 노출이 남으면 main audit는 BLOCKED다. Synthetic 결과의 노출과 MD5 target 노출을 task별로 구분한다.

Source별 `A_s={0..4095}−E_s`가2048개 이상이어야 한다. 이를 고정 RNG로 섞어 첫2048개를 test pool로 정하고, 나머지2048개 전체를 별도 RNG로 섞어 train1536/validation512에 배정한다. E_s는 main test에서 배제되며 train/validation은 별도 ownership 규칙을 따른다. Source prior draw로 train unique10,000과 validation512/test2048 대표를 수집한다. 대표는 evaluator 영역에만 보관하고 generation에는 target12 bits만 전달한다. Draw cap은 source별1,000,000이다.

| 단계 | 계약 |
|---|---|
| M0 | 다섯 pipeline의 POC_QUALIFIED, exposure·calibration·resource·code/profile를 확인하고 main ownership/corpus 및 모든 checksum 봉인 |
| M1 | Main/Shuffled×seeds0/1/2 전체30 models, 각각100 epochs·15,700 updates; 매10 epochs validation512×4; best checkpoints 전체 봉인 |
| M2 | 모든30개 checkpoint 봉인 후 source별2048 trials를 한 번 추출; learned 및6 shared Random streams에서 trial당 정확히100 attempts 생성 |
| M3 | Ledger 전수 계수·독립 verifier 재검사, paired statistics, 비용 및 완전성 보고 |

Source별 target 목록은 pool에서 iid uniform replacement로 생성해 모든 pipeline/method/seeds가 공유한다. 반복 target을 제거하거나 coverage를 보고 재추첨하지 않는다. 같은 target의 다른 trial은 다른 generation RNG를 쓴다. Trial ID가 통계 단위다. Invalid·duplicate·이미 성공한 뒤의 후보도 예산을 소비한다. 큰 후보 집합에서 선별하거나 verifier feedback, hash reranking, rejection sampling을 하지 않는다.

D1의 intrinsic 길이 factorization은 등록된 생성 과정이다. 이미 생성한 후보의 EOS/길이/header를 검증 결과에 맞춰 수정하지 않는다. Payload bytes가 있으면 duplicate나 source-invalid라도 MD5를 실제 계산·계수하고, bytes가 없는 invalid의 MD5 calls는0이다. Synthetic predicate 성공과 MD5 성공을 섞지 않는다.

## 9. 통계 및 calibration 완료 계약

고정된 source test pool·training data·checkpoints·지정 seeds 아래에서, 무작위 target trial과 generation randomness에 대한 Success@100 차이를 추정한다. Full MD5 역산, 모든 digest의 우위, 새로운 데이터/임의 seed의 일반화, 계산량 우위는 이 설계의 결론이 아니다.

Component는 pipeline별 Main 대 Random/Shuffled × seeds0/1/2의6개 one-sided exact McNemar다. Discordance0이면 p=1이다. Pipeline별 max-p를 만든 뒤 고정5개 family에 Holm을 적용한다. Missing은 보정용 p=1이며 outcome은 null이다. 유효한 완전 측정, 여섯 observed Δ>0, composite adjusted p<.05를 모두 충족해야 해당 pipeline 우위를 주장한다. CI 하한은 추가 gate가 아니다.

CI는 source별 trial index를 공유하여 모든 method/seed/K를 함께10,000회 bootstrap하고 percentile2.5/97.5%, NumPy linear quantile을 사용한다. 반복 digest의 다른 trial을 합치지 않는다. CI는 approximate marginal이며 simultaneous/exact 보장이 아니다. 퇴화·기준 성공0의 비율 미정의 등을 명시한다.

**기존 `study_statistics.analyze_family()`를 그대로 쓰지 않는다.** 해당 함수는 seed별 Holm·추가 CI 기준을 사용하므로 v3.1 계약과 다르다. `exact_mcnemar`, `holm_adjust`, `family_decision`의 primitive를 재사용하되 실제 M3 집계 함수를 calibration과 공유한다. 기존 unique-digest evaluator도 replacement trial ledger를 그대로 처리하지 못하므로 trial 기반 경로를 구현한다.

Calibration은 primary 자료 없이 가상 pool2048, iid trials2048, source별 공유 target/Random을 사용한다. Target별 Success@100 기준 `p0=1-(1-1/4096)^100`이다. 각 시나리오20,000 repetitions를 고정하고 난수 seed를 바꿔 통과할 때까지 반복하지 않는다.

| 시나리오 | 확률 지정 |
|---|---|
| All-null | Main/Shuffled/Random 모두 p0 |
| Opposite-effect null | 동일 크기 strata의 (Main,Random,Shuffled)=(.08,.02,.02)/(.02,.08,.08) |
| One-null-component | 기본 Main3p0/controls p0; P-G-BGV seed2만 Main=Shuffled=2p0, Random=p0 |
| Reference twofold | Main2p0, controls p0 |
| One-weaker-seed | Main seeds0/1=2p0, seed2=1.5p0, controls p0 |
| Stronger-shuffled | Main2p0, Shuffled1.5p0, Random p0 |
| Shared difficulty | 절반씩 difficulty .2/1.8, Main2p0×d, controls p0×d |

공유 target과 Random 이외의 method/seed Bernoulli outcomes는 target 확률에 조건부 독립으로 생성한다. 실제 family code에서 참인 null의 FWER와 pipeline별 power를 산출한다. 앞의 null3개는 one-sided95% Wilson upper≤.06, reference twofold는 모든 pipeline의 one-sided95% Wilson power lower≥.80을 요구한다. 나머지는 민감도 분석이며80% gate를 요구하지 않는다. Calibration 실패 시 M0를 차단하고 산술/설계 원인을 조사하며 R·목표효과 등을 수정하려면 새 revision을 만든다.

## 10. 재현성·저장·보고

Study master2026092531, engineering master2026092599, formal seeds0/1/2, P2A seed99, P2B seed100이다. SHA-256 첫8bytes big-endian을 사용하고 seed 배열은 JSON에 명시한 순서로 compact JSON 직렬화한다. Profile ID를 identity에 항상 포함하되 공통 data/order/noise/trial/generation 비교에서는 seed의 profile field를 null로 한다. 초기화·profile·recovery에서는 profile을 구분한다. Main/Shuffled 초기화의 method field는 null이다. Length/payload RNG는 별도 namespace이며 정상/반전 variant는 seed에 넣지 않는다.

각 모델·후보가 접근할 정보는 target12 bits와 public source/shape, 자기 checkpoint·난수다. D1은 자기 length head가 생성한 L을 내부 조건으로 추가한다. Sample IDs나 cache keys에 숨은 길이·원문을 넣지 않는다. Same-backend exact recovery만 주장한다.

Run별 SQLite WAL/FULL, `(run_id,unit_id,variant,attempt)` unique key와 batch transaction을 유지한다. Checkpoint는100 updates/epoch 경계와 정상 중단에 weights·optimizer·RNG·permutation·선택 상태·누적 비용을 저장한다. 검증된 최신2개와 best를 보관한다. Raw는 learned run의 첫16 units·normal 첫attempt만 보존한다. Full raw pool을 보관하지 않는다.

`execution_status`, `qualification_status`, `scientific_status`를 별도 필드로 보고한다. P2 profile 탈락, 선택되지 않아 미실행인 후보, P3 성능 미달, numerical failure, resource stop, 아직 평가하지 않은 run을 구분한다. Stage gate가 없어도 run seals와 partial ledger를 검증해 지금까지의 결과를 보고한다. 누락 통계는 null+reason이고 0이나 PASS가 아니다.

필수 산출물은 protocol/manifest/gates, implementation_readiness, exposure inventory/audit, profile_selection 및 profile.frozen, resources, analysis_validation, poc_status, main checkpoints_seal, trial schedules, run configuration/training/metrics/telemetry/ledger, 최종 report다. 아직 수행하지 않은 단계의 파일은 absent 또는 NOT_RUN으로 표시하며 가짜 PASS 파일을 만들지 않는다.

## 11. 구현 작업과 실행 순서

| 작업 | 완료해야 할 사항 | 현재 상태 |
|---|---|---|
| I0 | v3.1 schema/profile dispatch, G1/G2/D1, 공통 sampler 정합성, 진단, 전체 주기 예산, 부분 보고 | 일부 구현; 모델·개발 P0/P1/P2·진단·잠정 자원·부분 보고 완료; E0 최종 자원 확정과 정식 경로 오류 처리 남음 |
| I1 | Exposure/ownership, 실제 MD5 train/evaluate, trial ledger, M3 family 분석, full calibration, E0 경로 | 구현 필요 |
| I2 | 등록13조합 P0/P1, 순서가 고정된 P2A/B 선택·profile 봉인 | 개발 P0/P1 실행 및 P2A/B 구현; 원본 규모 P2와 정식 단계 미실행 |
| I3 | E0 리허설, 자원 최종 봉인, audit feasibility/calibration, P3 전체 적격성 | 미실행 |
| I4 | M0–M3 전체 실행·완전성 및 통계 보고 | 미실행 |

권장 흐름은 **I0/I1 구현 → P0 → P1 → P2 → E0 → 자원·감사·calibration 확정 → P3 → M0 → M1 → M2 → M3**다. Exposure 조사와 통계 구현/검산은 P2를 기다리지 않고 진행한다. 본실험 경로의 구현 누락을 P3 후에 발견하지 않도록 I1을 선행한다.

향후 CLI 계약은 `pilot --stage P0/P1/P2/P3`, `rehearse`, `audit --inventory`, `validate --suite statistics`, `prepare`, `train`, `evaluate`, `report`다. P2가 내부적으로 A/B와 profile 선택을 관리한다. 공통 `--protocol`, `--workdir`, 명시적 `--device mps`, 검증된 exact `--resume`, 쓰기 없는 `--dry-run`을 지원해야 한다. 임의 epochs/K/profile 선택으로 고정 계약을 우회하지 않는다. **현재는 개발 P0/P1/P2, Pilot dry-run 및 report를 지원하며 전체 v3.1 계약은 미완성이다.**

Exit0은 요청한 실행 계약 완료이며 과학적 우위를 뜻하지 않는다. 설정/선행 gate/적격성 미충족2, runtime/numerical3, 무결성4, hard resource5, 사용자 중단130을 유지한다. Soft 초과는 exit5가 아니다. 보고 명령은 부분 결과도 생성할 수 있고 전체 완료 여부는 별도 필드로 반환한다.

## 12. 실행량과 계획 검산

다음은 full matrix가 준비되는 성공 경로의 범위다. P2에서 후보가 모두 실패하면 더 적은 실행으로 BLOCKED가 될 수 있다. Forced recovery reference/replay, diagnostic forward, codec/ledger stress 및 calibration 비용은 별도 계수하고 모든 시간 상한에 포함한다.

| 구분 | 독립 learned runs | Optimizer updates | 평가 candidate rows |
|---|---:|---:|---:|
| P1, 모든 후보 기술 검사 | 26 | 208 | learned4,160 + Random320 = 4,480 |
| P2A | 5–13 | 10,000–26,000 | 640–1,664 |
| P2B, 세 probe 포함 | 5–13 | 78,500–204,100 | 3,840–9,984 |
| E0 | 10 | 80 | learned32,000 + Random6,400 = 38,400 |
| P3 | 15 | 235,500 | 15,360 |
| 본실험 | 30 | 471,000 | learned6,144,000 + Random1,228,800 = 7,372,800 |
| 합계 | 91–107 | 795,288–936,888 | 7,435,520–7,442,688 |

Batch profile은 선택5개만 측정하면1,700 candidates이고, 품질을 통과했지만 자원 조건에 실패한 후보까지13개 전부 profile하면 최대4,420이다. 이는 위 평가 ledger rows와 별도이며 실제 수행 수를 기록한다. Main sampling NFE는 D0/D1 선택에 따라447,283,200–449,740,800, P3는1,118,208–1,124,352다. D1 length head 호출을 누락하지 않는다. 실행 시간은 새 실측 전에는 제시하지 않는다.

계획 검산은 [validate_research_plan_v3_1.py](scripts/validate_research_plan_v3_1.py)로 JSON·고정 matrix·threshold·후보/update/NFE 산술·상태와 문서의 수치 일치를 확인한다. 검산 PASS와 IMPLEMENTATION_READY/POC_QUALIFIED/STUDY_COMPLETE는 다른 상태다.

2026-09-25에 `python3 scripts/validate_research_plan_v3_1.py`를 실행하여 `PLAN_CONSISTENCY_PASS`를 확인했다. 검산은 학습·후보 생성·통계 calibration을 수행하지 않으며, 모든 v3.1 정식 실험 gate는 `NOT_RUN`이다. 2026-09-26 구현 후에도 계획 검산을 통과했다.
