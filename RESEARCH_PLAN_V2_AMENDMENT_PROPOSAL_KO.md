# 연구 계획 v2.1 권장 수정안

**작성일:** 2026-09-24 KST  
**상태:** 제안 — 기존 protocol·실행 코드에 미적용  
**근거:** [현재 계획 검토 보고서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/RESEARCH_PLAN_V2_REVIEW_KO.md)

## 1. 권장 방향과 중요한 선택

권장안은 **개발 PoC에서 측정 경로를 검증한 뒤, 고정된 미노출 표적 pool에서 독립적으로 표적을 추출하는 본평가**다. 본평가의 표집 방식을 바꾸어 평균 성능 차이와 paired test의 확률 모형을 연결한다. 이는 현재 문제 목록을 반복한 체크리스트가 아니라, 채택할 수 있는 구체적인 설계 변경안이다.

| 결정 | 권장안 |
|---|---|
| 연구 목적 | 봉인한 source별 미관측 12-bit digest pool에서 평균 PreimageSuccess@100 우위 검정 |
| 개발 자료 | Primary test와 분리한 engineering fixture 및 synthetic train/validation |
| 기본 matrix | 현재 5 pipelines, Main/Shuffled/Random, model seeds 0/1/2 |
| Source·hash·학습 profile | 현재 source 정의, q=12, 100 epochs, architecture·objective·sampler를 최초 profile로 사용 |
| **본평가 표집 변경** | **2,048개 고유 digest pool에서 2,048번 독립 균등 복원추출** |
| 평가 단위 | **고유 digest 대신 독립 evaluation trial**; 각 trial에서 세 method를 같은 target에 pairing |
| 후보 예산 | Trial·method·seed당 100 attempts; @1/@10/@100은 같은 stream의 prefix |
| 검정 | One-sided exact McNemar → pipeline별 6개 max-p → 5-pipeline Holm |
| CI | 동일 trial 행을 함께 resample한 근사 paired bootstrap interval |
| 주장 범위 | 고정 pool·학습 데이터·봉인한 checkpoints·지정 seeds에 조건부인 평균 sampling 성능 |

**중요한 trade-off:** 원래 계획의 “2,048개 digest를 모두 한 번씩 평가”와 다르다. 권장안은 추론의 기준을 명확히 하는 대신, 고유 표적 전수 평가를 보장하지 않는다. 이 변경을 감춘 채 기존 unique-target 평가라고 부르면 안 된다.

2,048개 전수 평가가 연구의 필수 요구라면 이 표집 변경을 채택하지 않는다. 그 경우에는 기존 균등 가중 benchmark를 기술적 PoC로 먼저 보고하고, 고정된 이질적 표적들의 평균에 대한 확증 분석은 별도로 설계한다. 결과를 본 뒤 두 방식 중 유리한 것을 선택하지 않는다.

## 2. 연구 질문과 성공 문구 수정

### 권장 연구 질문

> 봉인한 source별 2,048개 미관측 12-bit MD5 digest pool에서 표적을 균등하게 추출했을 때, 모델이 100번의 생성 기회 안에 유효한 preimage를 찾을 평균 확률이 source-prior random과 shuffled-training control보다 높으며, 그 우위가 지정한 model seeds 0·1·2 각각에서 성립하는가?

### 허용되는 성공 문구

> 고정된 source별 평가 pool, 학습 데이터, checkpoints와 지정 model seeds에서, 독립 표적 추출·생성 절차에 따른 평균 PreimageSuccess@100의 양쪽 대조군 대비 우위가 지지되었다.

“모든 digest에서 개선”, “새 학습 데이터에서도 재현”, “최소 두 배”, “full MD5 역산”, “계산량 우위”는 이 판정으로 주장하지 않는다. Measurement gate 실패는 가설의 기각으로 처리하지 않는다.

## 3. 데이터와 평가 표집의 구체적 수정

### 3.1 Ownership과 노출 감사

현재 source별 train/validation/test ownership 1,536/512/2,048 및 train 10,000 messages를 출발점으로 삼는다. Test ownership을 본평가의 **고정 pool T**로 정의한다.

1. 알려진 archive 외의 과거 접근 이력을 포함해 source별 배제 목록을 확정한다.
2. Source별 미노출 pool이 2,048개 이상일 때만 현재 ownership 구성을 채택한다.
3. 최종 배제 목록·ownership·dataset hashes와 교집합 0 증거를 봉인한다.
4. 다른 source에서 본 같은 prefix까지 미노출이라는 project-wide novelty 주장은 하지 않는다.

가용 pool이 부족하면 N이나 제외 규칙을 실행 중에 바꾸지 않는다. 자료 범위·표본·주장을 수정한 새 protocol을 먼저 만든다. 현재 알려진 여유 163개·188개만으로 audit 완료를 선언할 수 없다.

### 3.2 Trial 목록 생성

모든 적용 checkpoint를 선택·봉인한 후, 사전 지정한 독립 RNG로 source마다 다음 목록을 한 번 생성한다.

\[
Y_i\overset{iid}{\sim}\operatorname{Uniform}(T),
\qquad i=1,\ldots,R,\quad |T|=2048,\ R=2048.
\]

- 같은 source의 모든 pipeline·method·model seed가 동일한 trial 목록을 사용한다.
- 같은 digest가 여러 번 나와도 각 trial은 **새로운 generation RNG**로 100 candidates를 만든다.
- Random은 source·model seed·trial별 stream을 만들고 같은 source의 pipeline들이 공유한다.
- 이전 trial의 candidates·success·invalid 결과를 다음 trial에 재사용하거나 sampler에 전달하지 않는다.
- Trial 목록은 중복 제거, 결과에 따른 재추첨, coverage 기준을 만족할 때까지의 재추첨을 하지 않는다.
- Ledger key는 `(source, pipeline, method, model_seed, trial_id, attempt)`로 정한다. Target digest는 별도 열이다.
- 통계 행 순서는 `trial_id`이며, 같은 digest를 합쳐 하나의 성공 여부로 만들지 않는다.

중복 digest는 독립적인 “새 digest”가 아니다. 새로운 난수로 수행한 sampling trial이며, 평가 pool 크기·trial 수·실제로 나온 고유 digest 수를 따로 보고한다.

### 3.3 Coverage와 예산

고유 표적 수의 기대값은 `|T| × [1-(1-1/|T|)^R]`다. 이번 제안에서 직접 검산한 값은 다음과 같다.

| 고정 pool | Trial 수 | 고유 표적 수 기대값 | Pool coverage 기대값 | Primary 평가 비용 비율 |
|---:|---:|---:|---:|---:|
| 2,048 | **2,048 — 최초 권장값** | 1,294.77 | 63.22% | 1배 |
| 2,048 | 4,096 | 1,770.97 | 86.47% | 2배 |
| 2,048 | 6,144 | 1,946.11 | 95.02% | 3배 |

아래 두 행은 trade-off 설명이며 자동으로 늘릴 예산이 아니다. R=2,048이면 기존의 6,144,000 learned candidates, 7,372,800 primary ledger rows, 447,283,200 sampling NFE 산술은 유지된다. 평가한 고유 표적 수는 달라진다. R의 최종 채택은 개정 설계의 검정력과 GPU 예산을 확인한 뒤 test 전에 봉인한다.

## 4. 통계 수정의 근거와 남는 조건

### 4.1 왜 표적별 효과가 달라도 연결되는가

T와 학습 데이터·checkpoints를 고정한다. 한 component comparison에서 trial i의 결과를 `Z_i=(M_i,B_i)`로 놓고, 두 method의 생성 난수까지 포함해 표적 y에서의 joint outcome 분포를 Q_y라고 하자.

표적 추출과 generation이 trial 간 독립이고, 알고리즘이 trial history에 따라 바뀌지 않으면

\[
Z_i\overset{iid}{\sim}Q_T,
\qquad Q_T=\frac1{|T|}\sum_{y\in T}Q_y.
\]

따라서 `π10=P(M=1,B=0)`, `π01=P(M=0,B=1)`에 대해

\[
\Delta_T=\frac1{|T|}\sum_y[P(M=1\mid y)-P(B=1\mid y)]
=\pi_{10}-\pi_{01}.
\]

귀무가설 `Δ_T≤0`은 `π10≤π01`과 같다. Discordant trials에 조건을 걸면 Main-only 수의 binomial parameter는 `π10/(π10+π01)≤1/2`이므로 one-sided exact McNemar tail을 사용할 수 있다. Discordance 0이면 p=1이다.

**이 정당화는 본 제안의 표집 설계에서 도출한 것이다.** 표적별 θ_y가 모두 같아야 한다는 가정을 대신해, 표적을 포함한 trial 전체가 동일 혼합분포에서 독립 추출된다는 조건을 사용한다. 표적별 난이도와 효과가 달라도 된다. 반면 고정된 실제 trial target 목록에 조건을 걸어 “이번에 뽑힌 target들만의 평균”을 검정한다고 다시 해석하면 이 논증과 달라진다.

Paired binary discordance에 대한 exact test의 계산 근거는 [McNemar 공식 문서](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)와 [binomial test 공식 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)에서 확인할 수 있다. One-sided 방향을 명시한 기존 primitive를 재사용하고, 양측 기본값을 그대로 호출하지 않는다.

### 4.2 무조건적인 독립성 보장은 아니다

다음 조건을 실행 계약으로 고정해야 한다.

- Checkpoint 선택·hyperparameter 변경·pipeline 선택에 본평가 결과를 사용하지 않음.
- Target schedule RNG가 학습·validation·generation RNG와 분리됨.
- Trial별 generation 난수가 독립이며 반복 digest에도 새 stream을 사용함.
- 후보 캐시, verifier feedback, 성능 기반 재시도·중단, trial history에 따른 sampler 변경이 없음.
- Model/precision/batch policy가 run 안에서 고정됨. PRNG를 통계적 난수 생성원으로 취급하는 일반적인 계산 모형을 명시함.

같은 source의 methods·seeds·pipelines 사이에는 pairing과 shared targets에 따른 의존성이 있다. 필요한 것은 각 comparison의 trial 간 독립성이며 모든 비교 사이의 독립성이 아니다.

### 4.3 Composite·CI·결과 상태

Pipeline별 두 controls×세 model seeds의 여섯 p-values를 max-p로 합치고, 다섯 composite에 Holm을 적용한다. 이 조합은 component가 유효하면 비교 간 독립을 요구하지 않는다. [Holm 공식 문서](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)

Bootstrap은 target ID가 아니라 **trial_id 행**을 10,000회 재표집한다. 같은 source의 모든 method·seed·K 행을 함께 유지한다. Percentile 95% interval은 iid trial 분포 아래 평균 차이에 대한 근사 marginal interval이며 exact·동시 신뢰구간이 아니다. CI 하한을 별도 성공 gate로 추가하지 않는다. 평균 비율 차이의 CI와 exact sign/McNemar test의 정합성은 별도의 통계 문제다. [Fay·Lumbard 원 논문](https://pubmed.ncbi.nlm.nih.gov/33263202/)

G3는 measurement gates, 모든 여섯 관측 차이 >0, composite Holm adjusted p<.05를 요구한다. G4는 지정 seeds 모두의 완전한 실행이다. Missing은 실제 outcomes에 0을 넣지 않고 BLOCKED/INCOMPLETE로 유지하며, 보정 계산에만 p=1을 사용한다.

### 4.4 채택 전 검정력과 오류율 재검사

기존 80.940%를 새 설계의 검정력 인증으로 옮겨 쓰지 않는다. 기존 통계 primitives와 simulation 방식을 재사용하되 다음 시나리오를 개정 trial 표집으로 확인한다.

1. Reference twofold, 한 seed만 약함, 강한 Shuffled, target 난이도 이질성.
2. Target별로 우위 방향이 반대지만 pool 평균 차이는 0인 null.
3. 여섯 비교 중 하나만 null이고 나머지가 강한 경우.
4. 같은 source의 trial 목록과 random outcomes 공유를 포함한 실제 five-pipeline family.

시나리오당 20,000회와 Monte Carlo 오차를 기록한다. 오류율 검사는 모든 경우에 대한 증명을 대신하지 않으며, reference power 목표 80%도 지정 대안의 목표다. 목표가 충족되지 않으면 R·목표 효과·주장 범위를 test 전에 개정한다.

**이번 제안에서 수행한 제한적 산술 진단:** 동일 비중의 두 strata에서 `(p_Main,p_Control)=(.08,.02),(.02,.08)`로 두면 target별 discordant Main-only 확률은 약 .8099와 .1901이지만, 혼합분포의 π10=π01=.0484다. 이 혼합분포에서 R=2,048, 20,000회 multinomial simulation을 수행했다. 기각률은 α=.05에서 4.315%(MC SE .144 pp), α=.01에서 .760%(MC SE .061 pp)였다. 한 component의 null 예시만 확인했으며 전체 family 오류율·power·실제 모델 동작을 검증한 결과는 아니다.

## 5. 구현 수정안

기존 model·codec·device·통계 primitive를 재사용하고 v2.1 입력에서 평가·보고까지 연결하는 작은 실행 경로를 만든다. 삭제된 pilot 복원이나 기존 PoC 프레임워크 전체 복제는 필요하지 않다.

| 부분 | 구체적인 수정 | 수용 검사 |
|---|---|---|
| 설정 | Pool 크기와 trial 수를 별도 필드로 정의. Protocol revision·schema 불일치 차단 | `test_pool_size=2048`, `evaluation_trials=2048`, `sampling=uniform_with_replacement`가 명확함 |
| 입력 | Model-facing target을 12-bit vector로 제한. Evaluator의 raw/length/suffix와 분리 | Hidden metadata mutation에도 고정 난수 출력 불변 |
| 학습 | Epoch별 무복원 permutation, 100 epochs, 마지막 batch 포함 | Run당 15,700 updates, 모든 training rows가 epoch마다 한 번 처리됨 |
| Shuffled | Epoch별 training donor permutation만 적용. Validation/inference는 실제 target | Split 밖 donor 0, inference condition 일치 |
| 선택 | Epoch 10·20·…·100, 고정 validation corruption, 최소 loss/이른 tie | Test 접근 전에 선택 checkpoint hash 봉인 |
| 생성 | 1,000표적 hard cap을 사용하지 않는 경로. Trial별 정확히 100 attempts | 2,048 trial IDs 각각 100 rows; @1≤@10≤@100 |
| 반복 표적 | Trial ID를 generation RNG identity에 포함 | 같은 digest의 다른 trial에 별도 RNG stream. 같은 bytes의 우연한 재생성은 허용 |
| 저장 | Candidate ledger append, raw tensor는 run별 첫 16 trials의 첫 candidate | 전체 raw pool 메모리 누적 없음. 보존 위치가 결과와 무관함 |
| 재개 | Model/optimizer·epoch/order·RNG·trial/attempt·저장 commit 상태 연결 | 강제 중단 전 rows 보존, 최종 key 누락·중복 0 |
| 분석 | Trial 단위 paired outcomes→6개 max-p→5개 Holm | Missing·discordance 0·opposite effect·CI 퇴화 fixture 검증 |

현재 source의 근거: [학습·generation 경로](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/runner.py:239), [1,000표적 제한](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/runner.py:339), [legacy family 집계](/Users/choisoonwook/Experiments_local/DHI_AI_gen/src/diffusion_hash_inv/study_statistics.py:9).

## 6. RNG 명세 보완안

Seed 문자열에 protocol revision과 task namespace를 포함하고, hash 실험과 synthetic control의 난수를 분리한다. 기존 SHA-256 기반 seed 파생을 재사용할 수 있다. Namespace별 공유 범위는 다음처럼 명시한다.

| Namespace | 공유 범위·차별화 기준 |
|---|---|
| Ownership·source corpus | Source별 공유; pipeline/method/model seed 제외 |
| Weight initialization | Source·pipeline·model seed별; Main/Shuffled 사이 method 제외 |
| Train order | Source·pipeline·model seed·epoch별; Main/Shuffled 동일 |
| Training noise | Method를 포함한 별도 stream; shuffle RNG와 분리 |
| Shuffle donors | Source·pipeline·model seed·epoch별 별도 stream |
| Validation corruption | Source·pipeline별 고정 draws; checkpoint·Main/Shuffled·model seed 사이 공유 |
| Evaluation targets | Source별; pipeline/method/model seed 제외 |
| Learned generation | Source·pipeline·method·model seed·trial_id·attempt별 |
| Random baseline | Source·model seed·trial_id·attempt별; pipeline 제외 |
| Bootstrap | Source별 동일 trial-index 재표집을 모든 comparisons에 공유 |

초기 weight seed namespace를 명시적으로 추가하고, 어떤 필드를 비우는지 protocol에 기록한다. Synthetic 정상/반전 비교는 동일한 public case ID의 초기 RNG state를 복제해 조건만 바꾼다. 반전 target 값이 달라졌다는 이유로 다른 초기 난수를 쓰지 않는다.

이 표는 하나의 구체적인 권장 공유 정책이다. 이를 채택하면 문서·JSON·seed fixtures를 함께 수정해야 하며, 기존 runner의 암묵적 seed 동작을 대신 사용할 수 없다.

## 7. Positive control과 개발 PoC 수정안

합성 과제·complement-pair split·15개 모델·K=1은 최초 profile의 acceptance 검사로 사용한다. Synthetic control의 test는 본평가의 복원추출 방식과 구분하여 **512개 고유 conditions 각각을 평가**한다.

운영 문턱은 정수로 명시한다.

- 정상 조건의 valid AND constraint 성공 ≥475/512.
- 반전 조건 자체의 valid AND constraint 성공 ≥475/512.
- 반전 생성물이 원래 조건 제약을 만족하는 수 ≤15/512.

이 문턱은 기존 Wilson 규칙과 동일하지만 **고정된 512개 condition에 대한 engineering acceptance 기준**으로 해석한다. 15개 모델이나 모든 이질적 condition에 대한 동시 95% 신뢰 보장으로 표현하지 않는다.

개발 중에는 synthetic train/validation에서 loss, valid rate, nibble accuracy, header/EOS/padding 오류를 점검한다. 최종 설정을 정한 뒤 control test를 평가한다. 같은 control test를 반복적으로 보고 수정했다면 독립 검증 자료라고 계속 부르지 말고 변경 이력·개발 사용 여부를 남긴다.

Control 실패 시 먼저 “모델 입력이 전달되는가 → clean codec가 맞는가 → 학습 loss가 유한하고 감소하는가 → full generation이 유효한가 → 조건 제약을 따르는가”를 분리한다. 실패 원인이 확인되기 전 모델 확대·schedule 변경·학습량 증가를 일괄 적용하지 않는다. 필요한 변경은 별도 revision에서 영향받는 controls를 다시 검증한다.

## 8. GPU·저장·복구 정책 수정안

학습 batch=64와 float32는 최초 과학 profile이다. 운영 profile에서는 primary 성능을 보지 않고 engineering 입력으로 inference batch 후보 1/4/16/64를 측정하는 방안을 권한다. Warm-up 이후 반복 측정하고 pipeline별 sampling, CPU decode, hash, ledger IO를 포함해 비교한다. 측정에서 안전하고 효율적인 batch를 본평가 전에 고정한다.

Resource manifest에는 다음 값을 실제 수치로 채운다.

- Pipeline별 optimizer step·validation·candidate 생성 처리량과 peak memory.
- 최초 45개 학습 runs, validation, control generation, primary 평가·저장을 포함한 전체 시간 추산.
- 실제 ledger row 크기·checkpoint·raw samples·임시 파일을 포함한 저장량.
- Run/study wall-clock 및 storage 상한, commit·checkpoint 주기, resume 조건.

초기 운영 여유로 측정 추산 시간의 1.5배, 저장량의 2배를 검토할 수 있다. 이는 권장 여유율이며 장비 측정이나 사용자의 자원 한도를 대신하는 보장은 아니다. 자원 상한은 성능을 보고 연장하는 규칙으로 만들지 않는다. 상한 도달은 사전 정의된 중단·재개 정책과 INCOMPLETE 상태로 처리한다.

MPS smoke 성공을 전체 v2.1 처리량으로 환산하지 않는다. 수백만 후보의 raw tensor를 Python list에 누적하지 않고, 일반적인 파일/transaction 기능으로 최소 streaming·resume 경로를 구현한다.

## 9. 적용 순서와 완료 기준

| 단계 | 할 일 | 다음 단계 진입 조건 |
|---|---|---|
| 1. 과학 설계 결정 | 노출 audit·연구 범위 확인, 복원추출 수정의 채택 여부 결정 | 충분한 test pool과 명확한 분석 대상 |
| 2. 문서·schema 정합화 | 연구 질문·평가 단위·RNG·통계·판정 문구를 v2.1에 반영 | 문서/JSON 일치, 변경 이력과 hashes |
| 3. 최소 구현 | 12-bit 입력·epoch 학습·train-only shuffle·validation 선택·trial ledger·통계 연결 | Leakage/negative/counting fixture 통과 |
| 4. 실행·분석 검증 | 작은 GPU profile, 강제 중단·재개, 개정 통계 null/power 검사 | 처리량·무결성·분석 동작 확인 |
| 5. 학습 controls | 최종 profile의 다섯 pipeline×세 seeds | Pipeline별 정수 acceptance 문턱 통과 |
| 6. 최종 봉인 | Resource manifest·primary data·모든 적용 checkpoints 봉인 | Test-independent 선택과 provenance 완성 |
| 7. 본평가·보고 | 고정 trial schedule로 생성, 독립 verifier, 통계·비용·실패 보고 | 실제 ledger 검사 및 모든 지정 seeds 완료 |

측정 gate를 통과하지 못한 pipeline은 본평가에 진입하지 않는다. 다른 pipeline은 자기 조건을 충족하면 진행할 수 있으나 five-pipeline family를 유지한다. 중간 성능으로 model seed나 pipeline을 선택하지 않는다.

## 10. 제안서의 검증 범위

이번 작업은 권장 수정안 작성이다. 원래 계획이나 JSON·구현을 개정하지 않았고, 모델 학습·primary 평가를 수행하지 않았다. 삭제된 pilot을 복원하거나 근거로 사용하지 않았다.

새로 수행한 계산은 coverage 기대값과 §4.4의 단일 component null 진단이다. Null 진단은 NumPy `default_rng(2026092401)`, R=2,048, 20,000 repetitions, joint cell probabilities `(00,01,10,11)=(.9016,.0484,.0484,.0016)`, 현재 `exact_mcnemar(..., alternative="greater")` 및 strict `p<alpha`를 사용했다. 단순한 이론 모형의 검산이며 v2.1 전체의 적합성 인증은 아니다.

채택 시 권장할 수정은 **평가 표집·통계 단위·RNG identity를 함께 바꾸는 것**이다. 통계 함수만 교체하거나 trial ID 없이 반복 digest를 추가하는 변경으로는 이 제안의 근거가 성립하지 않는다.
