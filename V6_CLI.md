# V6 독립 MLX 구현

## 구현 결정

- Source별 token ID는 `data.TOKENS`로 관리한다. `fresh_batch`는 `(stage, source, seed_id)` 또는 CLP namespace에서 source를 읽으며, 파이프라인과 Main/Shuffled는 메시지 namespace에 넣지 않는다.
- 이미지 codec은 NumPy NCHW float32를 입출력하고, prototype 거리는 float64로 계산한다. 거리 행렬은 256개 후보씩 처리한다.
- Gaussian 재생성 감사의 마지막 묶음이 64개보다 작으면 마지막 선택 key를 반복해 batch 64를 채우고 실제 선택 후보만 비교한다.
- **Calibration의 두 용도를 구분한다.** 200회 축소 검사는 판정 경로·CP 계산·공동 중단의 회귀 검사다. 관측한 CP 한계와 기준 충족 여부는 그대로 출력하지만 `scope=regression-only`, `production=false`, `passed=false`이며 연구 통과 근거가 될 수 없다.
- **정식 calibration**은 등록된 블록 크기 8,192, 시나리오별 최소 2,000회로 수행한다. 무효과 양성률의 one-sided 95% CP 상한 ≤ 0.025, 전체 기각률 CP 하한 ≥ 0.95, +δ 검출률 CP 하한 ≥ 0.95를 모두 만족해야 `passed=true`다. 정식 게이트 실패 시 난수·반복 수·문턱을 사후 조정해 통과시키지 않는다.
- 이 구분은 2026-09-29 사용자의 “연구 최종 결론의 근거로 사용하기에 논리적으로 적절한 기준” 요청에 따른다. 과학적 문턱과 등록 JSON은 변경하지 않았다. Calibration은 판정 절차 검증이며, 실제 효과에 대한 최종 결론은 별도의 봉인된 C 자료·무결성 감사·필요한 R 재현에만 근거한다.
