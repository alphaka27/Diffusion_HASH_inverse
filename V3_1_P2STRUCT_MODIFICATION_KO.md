P2 후속 권장 수정안 및 구현 기록 — 2026-09-27 작성, 2026-09-28 검증 완료

근거: [최신 P2-fix 분석](/Users/choisoonwook/Experiments_local/DHI_AI_gen/V3_1_P2FIX_ANALYSIS_KO.md). D1의 최종 실패 33개는 모두 prefix 한 바이트 오류였고, Gaussian은 최종 768개 중 767개가 형식 오류였다. 보존된 Gaussian 샘플 48개의 prefix는 모두 정답이었다.

**권장 변경: D1의 학습 목표를 조건 정확성과 맞추고, Gaussian의 길이·문법을 생성 모델 안에서 일관되게 결정한다.** 아래 변경은 합성 과제 개발용 개정 `p2struct`로 구현한다. 성능 향상은 새 P2 실행으로 검증할 가설이다.

| 대상 | 수정 | 유지할 항목 |
|---|---|---|
| D1 | length CE + prefix3 masked CE 평균 + suffix masked CE 평균, 각 가중치 1 | 구조·optimizer·학습량·temperature1·length/payload RNG |
| D1 진단 | P2B 후보별 prefix 토큰이 최초 확정되는 step, 정답/선택 토큰 확률, argmax 기록 | sampler의 토큰 선택·reveal 난수 및 gate |
| Gaussian G3 | 학습한 길이 분포에서 L을 한 번 뽑고 L에 맞는 header·mask·padding 문맥 안에서 payload glyph만 확산 | strict codec·G2 좌표/조건 출력 경로·x0 예측·100-step sampling |
| 평가 | 새 protocol·새 workdir에서 P0/P1/P2 | P2A 10k, P2B fresh seed100/epoch100, 정상·반전 각각 122/128 |

D1은 현재 전체 masked token 평균에서 prefix의 비중이 길이와 무작위 suffix에 따라 희석된다. 새 loss는 각 메시지의 가려진 prefix와 suffix를 따로 평균한 뒤 더한다. 비어 있는 영역은 0이다. `payload_ce`는 두 영역의 합이며 `prefix_ce`, `suffix_ce`, `length_ce`를 별도로 기록한다. 이 변경만으로 통과를 보장하지 않으며, 학습량이나 sampling temperature를 동시에 바꾸지 않는다. 진단은 실제로 뽑힌 후보의 ledger에 붙여 저장하므로 재시작 시 후보와 함께 복구된다. 진단용 정답 계산은 sampler 밖에서 이루어진다.

G3는 G2를 대체 수정하지 않는 새 Gaussian 프로필이다. D1과 같은 12→28 선형 length head를 학습한다. 학습 시 길이는 해당 학습 메시지에서 얻고, 생성 시에는 공개 12-bit 조건만으로 계산한 분포에서 독립 length RNG로 4~31 중 하나를 뽑는다. Denoiser에는 공개 조건과 L/31만 전달한다. 실제 평가 메시지의 길이나 prefix 정답은 전달하지 않는다.

G3의 확산 상태에서 활성 payload glyph만 잡음을 받는다. BGV 길이 header, 연속 활성 mask, 비활성 glyph padding은 뽑힌 L에서 구성한 고정 문맥이며 시작부터 매 reverse step까지 유지한다. 손실은 length CE + prefix glyph MSE 평균 + suffix glyph MSE 평균이다. 이미 고정한 header·mask·padding을 다시 예측하도록 loss를 부과하지 않는다. 마지막에 잘못된 생성물을 고치는 후처리와 구분되는, 길이 조건부 생성 공간의 변경이다. 모델 출력의 NaN/Inf는 고정 문맥으로 덮어 숨기지 않고 오류로 처리한다. Printable payload의 source domain과 CGGE glyph 품질은 계속 strict decoder가 검사하므로 구조 일관성이 곧 전체 valid나 joint 성공을 보장하지는 않는다.

G3는 length head 1회 + denoiser 100회로 NFE를 101로 기록한다. 별도 length RNG identity와 sampled length를 ledger에 보존한다. P0 파라미터 수, Torch/MLX loss·gradient, 학습/생성 복구, P2 진단, 자원 추정도 같은 경로를 사용한다. Gaussian 진단에서 관측 길이를 사용하는 복원 평가는 `teacher_forced_length`로 명시하고 실제 생성 품질과 구분한다.

기존 P2-fix 명세·결과·checkpoint는 유지한다. 새 `examples/poc-v3.1-p2struct-protocol.json`은 G3/D1만 실행하며 기존 source data·case 선택·seed namespace·품질 기준을 유지한다. G3는 프로필과 구조가 달라 초기 가중치까지 이전 G2와 같지는 않다. D1은 구조와 초기화가 동일하고 loss만 바뀌므로 같은 epoch100 결과로 비교한다. Prefix 기반 loss는 합성 과제 특성을 사용하므로 MD5 objective로 자동 전이하지 않는다. Formal P3·E0·본실험 차단은 유지한다.

**검증 및 실행**

수정 및 검증을 완료했다.

- 전체 회귀 검사 **154 passed, 5 skipped**. 건너뛴 5개는 CUDA 미지원 검사다. PyTorch CPU와 MLX CPU/Metal에서 새 loss·forward·gradient·Adam 일치, 길이 4~31의 구조 문맥, NaN/Inf 거부, prefix 진단 활성화 전후 후보 일치, checkpoint·생성 복구를 확인했다. 축소 P2에서는 최종 checkpoint 판정·진단·봉인도 확인했다. [전체 테스트 XML](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2struct-tests-20260927.xml).
- 최종 명세의 NFE 합계 및 SHA-256 등록을 반영한 뒤 명세 보존 검사를 별도로 통과했다. 최종 SHA-256은 `6e2f450767b586bba0e7069c0df9c9bc24d8a4e870cda1e2ad85308cd15d3916`이다. 원본 연구 명세 정합성 검사와 최종 P2 CLI dry-run도 통과했다.
- 등록된 규모 그대로 MLX Metal에서 개발 **P0/P1 PASS**. P0는 51개 검사와 1,214개 codec 왕복을, P1은 10개 learned run·2개 Random stream과 5개 모델의 학습/생성 복구를 통과했다. 복구 재실행을 제외한 후보 1,920개와 NFE 118,080을 확인했다. [실행 보고서](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/runs/v31-p2struct-validation-20260927/report.md).
- 최종 source 49개, P0/P1 봉인 파일 408개, 복구를 포함한 17개 ledger의 2,720행을 점검했다. G3의 길이 RNG·NFE 101 기록과 생성 길이 범위 4~31도 확인했다. [무결성 점검 결과](/Users/choisoonwook/Experiments_local/DHI_AI_gen/local_experiment_archive/analyses/v31-p2struct-validation-audit-20260928.json).

**원본 규모 P2는 실행하지 않았다.** P1은 짧은 실행·복구 검사로, learned run의 joint는 모두 0이었다. Gaussian의 남은 실패는 printable source domain 또는 CGGE glyph 오차였다. P0/P1 및 축소 P2 통과는 실행 정합성 증거이며 생성 품질 개선이나 formal 적격성 증거가 아니다. 실제 품질 판정은 아래 새 실행으로 확인한다.

```sh
DHI_RUN="local_experiment_archive/runs/v31-p2struct-$(date +%Y%m%d-%H%M%S)"
for stage in P0 P1 P2; do
  .venv/bin/python -m diffusion_hash_inv.study_cli pilot \
    --protocol examples/poc-v3.1-p2struct-protocol.json \
    --workdir "$DHI_RUN" --stage "$stage" \
    --backend mlx --device gpu --development || break
done
```

기존 run은 source hash가 달라 재개하지 않는다. 실제 validation은 이미 개발에 사용한 조건이므로 독립적인 최종 검증으로 주장하지 않는다.
