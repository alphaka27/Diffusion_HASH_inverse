연구 가능성 검토 — 2026-09-28

검토 대상은 “해시만 조건으로 받은 diffusion이 미학습 MD5 target의 유효한 역상 후보를 source-prior random과 shuffled-condition 모델보다 높은 확률로 생성할 수 있는가”다. 원본 연구 질문과 최신 실행을 기준으로, 저장 결과의 재검증·이론적 기준 계산·기존 문헌 대조를 수행했다. 새 모델 학습·생성이나 formal test 접근은 하지 않았다.

**판정: 연구를 반증 가능한 실험으로 계속할 근거는 있다. 그러나 실제 MD5 후보 생성 우위는 아직 입증되지 않았고, full MD5/SHA-256 역상 또는 계산량 우위를 예상할 실증 근거는 없다. 현재 확실하게 개선된 것은 조건부 생성의 기술적 실행 가능성이다.**

[최신 실패 분석](/Users/choisoonwook/Experiments_local/DHI_AI_gen/V3_1_P2STRUCT_ANALYSIS_KO.md) · [검증 수치 JSON](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/research-feasibility-20260928/verification.json) · [재현 스크립트](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/research-feasibility-20260928/verify.py)

**연구 질문을 네 수준으로 나누면 판단이 명확해진다**

| 주장 | 현재 증거 | 판정 |
|---|---|---|
| 실제 모델이 공개 조건을 사용하고 유효한 메시지를 생성할 수 있다 | 최신 P-DISC 123/125, R-G-BGV 128/128, 각 분모128 | 단일 개발 seed에서 지지됨 |
| Gaussian·Discrete 다섯 pipeline이 모두 안정적으로 작동한다 | Printable Gaussian 두 개와 R-DISC가 P2B 미달 | 전체 적격성 미확보 |
| 미학습 MD5-12 target에서 Main이 Random과 Shuffled를 모두 능가한다 | 최신 실행은 synthetic이고 MD5 판정은 없음; 이전 실제 MD5 결과도 우위 없음 | 미입증 |
| 해시를 직접 시도하는 것보다 적은 시간·비용으로 역상을 찾는다 | 동등 계산량 비교, 학습비 상각, 전처리 경쟁법 비교 없음 | 미검증 |

Synthetic 적격성 실패는 현재 구현에 대한 실패다. 모든 diffusion의 불가능성을 증명하지 않는다. 반대로 synthetic 적격성 통과도 실제 해시의 역상 구조를 학습했다는 증거가 되지 않는다.

**최신 성공과 본 연구 사이에는 과제의 차이가 있다**

현재 `synthetic_nibbles`는 다음 방식으로 데이터를 만든다.

`x = encode_3_nibbles(y) || random_suffix`

따라서 y를 받으면 앞 3바이트의 정답이 직접 정해진다. 길이와 suffix는 정해진 prior에서 뽑을 수 있고, 알려진 규칙으로는 신경망 없이도 해당 조건을 만족하는 메시지를 구성할 수 있다. 이는 양성 대조군으로 적절하지만 암호학적 난이도를 갖는 문제가 아니다. 모델이 그 규칙을 학습했는지를 검사하는 engineering control이다.

본실험은 다음을 요구한다.

`y = first_12_bits(MD5(x)); model receives y only`

여기서는 y를 x의 앞 3바이트로 복사해도 성공 조건이 충족되지 않는다. MD5 출력 prefix와 입력 payload prefix는 다른 값이다. 특히 이번 p2struct의 prefix3/suffix 분리 loss는 synthetic에서 정답이 어느 위치에 존재하는지 이용한다. 실제 MD5 과제에서 첫 3바이트만 특별히 중요하다는 근거는 없다. **그 loss의 성공을 본실험용 objective의 검증으로 옮기면 안 된다.** 등록된 p2struct amendment도 synthetic 개발 전용이며 MD5 전이에 별도 개정을 요구한다.

BGV/CGGE는 메시지의 표현을 바꾼다. 가역적 인코딩 자체가 추가 역상 정보를 만드는 것은 아니다. 인코딩이 특정 신경망의 학습 편향에 유리할 가능성과, 해시 역상을 쉽게 만든다는 주장은 따로 실험해야 한다. 현재 source는 자연어·비밀번호 빈도분포가 아닌 iid Printable94/Bytes256과 균등 길이이며, 정확한 source prior는 이미 직접 sampling할 수 있다. 따라서 언어적 규칙을 학습해 얻는 이득도 이번 연구 설정의 설명으로 쓸 수 없다.

**실패 원인별로 연구 가능성에 주는 의미가 다르다**

| 관측 | 연구에 유리한 해석 | 아직 남는 장벽 |
|---|---|---|
| Gaussian 길이·mask 문제 해소, R-G-BGV 256/256 성공 | 이미지 표현의 구조 실패가 고정적인 한계는 아니었음 | Random Bytes는 모든 byte가 허용됨; suffix 분포 품질·MD5 적합성은 별도 |
| P-G-BGV prefix 256/256 정답, 전체 valid 16/256 | 공개 조건 반영은 작동함 | Printable suffix 때문에 대부분의 후보 예산을 잃음 |
| P-G-CGGE valid 86/256, 그중 조건 성공86 | 일부 정상적인 문자 생성이 가능함 | 긴 메시지의 glyph 유효성을 충분히 유지하지 못함 |
| P-DISC 통과, R-DISC 정상에서1개 부족 | Token 표현과 조건 주입이 실용적인 개발 경로임 | 확률적 오답 선택과 일부 오답 순위를 줄일 필요 |
| D1 loss 개선으로 성공률 상승 | 특정 합성 조건 학습 병목은 수정 가능함 | 실제 MD5 조건 활용을 보여 주는 증거는 아님 |

현재 결과는 “고칠 수 있는 생성 병목이 있다”는 설명을 지지한다. “병목을 고치면 무작위 탐색보다 잘한다”는 결론까지 지지하지는 않는다. 연구를 계속한다면 후자를 직접 시험하는 단계로 가는 것이 정보 가치가 높다.

**과거 실제 MD5 증거도 포함하면 긍정적 우위는 확인되지 않는다**

먼저 2026-09-17의 `ABCD^4` toy MD5-8 실험을 재검증했다. 세 방법의 저장 후보 총5,400개를 다시 해시하고 파일 seal·타깃별 K100 집계·paired bootstrap을 대조했다.

| 방법 | Success@100 | Valid 비율 |
|---|---:|---:|
| Diffusion | 3/18, 16.67% | 65.33% |
| Source-prior random | 7/18, 38.89% | 100% |
| Uniform-domain random | 9/18, 50.00% | 100% |

Source-prior 대비 차이는 −22.22%p, 저장 결과와 동일한 paired bootstrap95% CI는 [−50.00,+5.56]%p, 우위 방향 exact McNemar p=0.96484375다. 표본이 작아 전체 방법의 불가능성을 판정할 수 없지만 우위의 증거도 아니다. 이 toy의 두 random 방법은 같은 분포의 다른 RNG 실현이다.

이 toy에서는 target별 전상 개수가 달라 random 성공률을 무조건 `1−(255/256)^100`으로 놓으면 부정확하다. 완전한256개 domain을 다시 해시해 실제18개 target의 `p_y=#preimages(y)/256`을 계산한 결과, 평균 이론적 Success@100은 **40.54%**다. Source-prior의 관측38.89%와 양립한다. 작은 제한 domain에서의 우연 성공을 해시 구조 학습으로 해석하면 안 되는 사례다.

이후 2026-09-21에 확정된 Printable94/Bytes256 실제 MD5 PoC도 확인했다. 저장된 비교 JSON에서 q12 결과는 다음과 같다.

| Source | Main@100 | Random@100 | Shuffled@100 |
|---|---:|---:|---:|
| Printable | 0/305 | 9/305 | 0/305 |
| Random Bytes | 0/303 | 8/303 | 0/303 |

당시 보고서에 따르면 Gaussian Main의 q8/q12/q16 test 합계247,500개 중 valid는1개뿐이었다. 이 큰 형식 실패 때문에 MD5 신호를 제대로 분리 평가하지 못했다. 최신 G3/D1은 모델·목표가 달라 당시 실패를 그대로 현재 모델의 성능으로 옮길 수도 없다. 이번 검토에서는 이 PoC의 비교 요약과 노출 타깃을 확인했으며, 전체742,500개 test ledger의 재검증은 [당시 상세 감사 보고서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/retired_2026-09-21/artifacts/poc_md5_truncated/reports/RESULT_ANALYSIS_KO.md)의 범위다.

**이론적 기준은 “불가능 증명”이 아니라 비교할 출발점이다**

이상적인 q-bit random function에서 아직 질의하지 않은 서로 다른 입력의 출력은 독립 균등하다. 별도의 목표 역상·전처리 정보가 없으면 새 후보 K개의 성공 확률은 `1−(1−2^-q)^K`다. 학습 메시지의 해시 정보가 새로운 메시지의 출력을 예측하게 해 주지 않는다는 이상화 아래의 계산이다. 실제 고정 MD5의 모든 알고리즘에 대한 불가능 정리가 아니며, 이미 수행한 전처리·학습용 해시 질의는 비용에서 빠질 수 없다. 입력을 중복하면 이 독립 새 질의 수가 줄어든다.

| q | 후보1개 성공확률 | 후보100개 중 성공확률 |
|---|---:|---:|
| 8 | 0.390625% | 32.3884% |
| 12 | 0.0244141% | 2.41214% |
| 16 | 0.00152588% | 0.152473% |

실제 MD5/source/target pool의 `p_y`는 별도로 측정해야 하며 위 표를 관측 baseline 대신 쓰지 않는다. 출력 비트가 적다는 것은 우연한 성공을 쉽게 만든다는 뜻이지, MD5의 내부 round를 줄였다는 뜻은 아니다. MD5-12는 전체 MD5를 계산한 뒤12비트만 검사한다.

MD5의 충돌 공격과 주어진 digest의 역상 공격도 구분해야 한다. [RFC6151 §2.1–2.2](https://www.rfc-editor.org/rfc/rfc6151.html#section-2.2)는 이를 따로 다룬다. 문서의 공격 비용은2011년 당시 설명이며 현재 최선의 공격을 조사한 수치로 사용하지 않는다. [Rogaway–Shrimpton의 기본 정의 논문](https://www.iacr.org/archive/fse2004/30170373/30170373.pdf) 역시 서로 다른 보안 정의를 구분한다.

신경망을 이용한 [Goncharov의2019년 연구](https://arxiv.org/html/1901.02438v1)는 줄이거나 약화한 round의 부분 역상을 다루며, 저자도 그 결과만으로 실용 암호분석 가치를 주장하지 않는다. 이는 neural 접근을 조사할 수 있다는 선행 사례이지만 현재 full-round truncated-MD5 실험의 성공 증거는 아니다. 이 검토는 관련 문헌 전체의 최신 성과 부재를 증명한 체계적 조사도 아니다.

**형식 실패는 후보 우위를 얻는 데 필요한 효과 크기를 키운다**

후보 한 개의 성공확률은 항상 `P(valid) × P(hash hit | valid)`로 분해된다. 현재 synthetic의 valid 비율이 향후 MD5 실행에서도 같다고 가정하면, Random의 per-candidate 성공확률과 같아지기 위해 필요한 valid 후보의 조건부 lift는 다음과 같다.

| 모델 | 가정에 사용한 현재 valid | Random과 같아지는 조건부 lift |
|---|---:|---:|
| P-G-BGV | 6.25% | 16배 |
| P-G-CGGE | 33.59% | 약2.98배 |
| D1·R-G-BGV | 100% | 1배; 우위를 위해서는 그 이상 |

이는 **가정에 따른 민감도 계산**이며 현재 모델의 MD5 성능 측정이 아니다. 이 때문에 실제 해시 신호의 존재를 먼저 확인하는 데는 유효율이 높은 경로가 유리하다. Gaussian의 유효율 향상만으로 Random보다 높은 조건부 hash hit가 생기지는 않는다.

계산량 우위는 더 강한 조건이다. 동일 목표·동일 성공확률에서 후보당 비용과 학습·자료 생성·검증·중복 비용을 포함해야 한다. 현재 D1은 후보당33회, G3는101회 신경망 호출을 쓴다. 이를 MD5 호출 횟수와 바로 동일 단위로 비교할 수 없으며 실제 wall time과 처리량을 재야 한다.

또한 q12처럼 출력 공간이 작은 문제에는 전처리 lookup이라는 강한 경쟁법이 있다. 균등 digest 가정에서4096개 target 각각의 역상을 하나씩 모으는 coupon-collector 기대 draw 수는 **36,434.35회**다. 최장31바이트 payload만 저장하면126,976바이트, 약124KiB이며 index·length 등 부대 저장은 별도다. 이는 분석식으로 계산한 값으로, 실제 lookup을 만들거나 새 MD5 target을 열지 않았다. 현재 모델의 후보 우위 검정에 lookup을 몰래 넣는다는 뜻도 아니다. 향후 계산량 우위를 주장하려면 모델의 학습 전처리와 이런 경쟁 전처리를 함께 고려해야 한다.

**새 q12 검증에는 노출 타깃 수라는 설계 제약도 있다**

v3.1은 source별4096개 target 중 이전 MD5 learned validation/test에서 노출된 q≥12의12비트 prefix를 제외하고, test pool2048개를 확보하도록 정했다. 한 개의 과거 PoC만 조사해도 다음을 확인했다. 실제 validation/test 대표의 full MD5를 재계산했고 q16은 앞12비트로 투영했다. Manifest에 소유권만 예약된 미사용 target은 이 집계에 넣지 않았다.

| Source | 확인된 기존 노출 하한 | 추가 감사 전 잔여 상한 | test2048개 대비 최대 여유 |
|---|---:|---:|---:|
| Printable | 1,885 | 2,211 | 163 |
| Random Bytes | 1,860 | 2,236 | 188 |

따라서 현재 q12 holdout이 충분하다고 아직 인증할 수 없다. 다른 archive, q8 실행에서 raw/full digest를 사용한 범위, 외부·삭제 자료, E0의 추가 노출을 합치면 여유가 줄 수 있다. 이 값은 조사한 기존 실험의 노출 하한이며 완성된 exposure inventory가 아니다. 이미 노출된 target을 새 seed로 섞어 미노출이라고 할 수 없다. 풀을 확보하지 못하면 새 개정에서 평가 범위 또는 q를 재설계해야 하며, q를 늘렸다고 자동 독립성이 생기는 것도 아니다.

**연구를 계속한다면 다음 판정을 직접 겨냥하는 것이 타당하다**

목표는 남은 synthetic 점수를 끝없이 올리는 것이 아니라, 생성 가능한 모델에서 실제 MD5 조건의 이득이 관측되는지 판단하는 것이다. 우선 P-DISC/D1은 Printable source에서 실제 생성과 자원 검사를 통과한 후보이므로, 별도 exploratory 개정의 최소 신호 검증 대상으로 합리적이다. Random Bytes/Gaussian의 정보가 필요하면 이미 통과한 R-G-BGV를 별도 arm으로 둘 수 있다. 이는 결과를 보고 다섯 pipeline 중 실패 항목을 삭제해 기존 연구를 완료 처리하는 제안이 아니다.

| 검증 단계 | 필요한 증거 | 다음 판단 |
|---|---|---|
| 노출·구현 감사 | source별 가용 holdout 확인, 실제 MD5 학습/평가·후보 원장·독립 rehash 정합성 | 불충분하면 새 데이터/평가 개정부터 |
| 본 과제용 학습 목표 고정 | synthetic prefix 전용 loss의 자동 전이 방지, model/sampler/예산/선택 규칙 사전 고정 | 실제 MD5 조건의 영향만 해석 가능한 상태 확보 |
| 직접적인 효능 비교 | 동일 pipeline·source·seed의 Main/Shuffled, 동일 source-prior Random, 같은 target/K100 | Main이 두 대조군을 모두 능가하는지 확인 |
| 재현·통계 | 사전 지정 seeds, 동일 family와 분석 코드의 calibration, 유효·중복·실패 후보 모두 계수 | 양의 효과와 불확실성을 분리 판단 |
| 확대 여부 | 사전 정한 유용한 효과 크기에 대한 검정력·CI와 비용 | 양의 증거면 확장; 의미 있는 이득을 배제하면 해당 설정 중단; 정밀도 부족이면 불확정 |

원래 v3.1 전체 연구를 유지한다면 다섯 pipeline의 P2, E0, P3 및30 learned runs 등의 기존 완료 계약을 충족해야 한다. 빠른 한-pipeline 가능성 검사는 **별도 개정의 제한적 결과**다. P2 문턱을 낮추거나 기존 gate를 우회한 결과로 만들지 않는다. 새 hash 실험의 비유의 p값 하나만으로 “효과 없음”을 확정하지 말고, 사전에 정한 효과 크기를 CI가 배제하는지 확인해야 한다.

종합 판단은 **조건부 연구 지속**이다. “조건부 생성의 실패 원인을 규명하고 개선한다”는 연구는 이미 관측 가능한 성과를 갖는다. “실제 해시 target에서 후보 우위가 있다”는 가설은 유효한 MD5 대조 실험으로 남아 있다. 현재 근거만으로 full-hash 역상 응용을 전망하거나, 반대로 diffusion 전체의 가능성을 기각하는 것은 모두 근거를 넘는다.
