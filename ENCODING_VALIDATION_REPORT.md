# 인코딩 검증 보고서

- 검증일: 2026-09-15
- 대상: BGV, CGGE, Direct Bits 인코더/디코더와 최소 실행 경로
- 판정: **결정적 인코딩·디코딩은 정상 수행 가능**. 단, 현재 diffusion 모델의 reversible positive control은 통과하지 못했으므로 hash-inversion 실험은 아직 시작 조건(G1)을 만족하지 않는다.

## 검증 범위와 결과

| 항목 | 결과 | 근거 |
| --- | --- | --- |
| 독립 oracle 검증 | PASS | `tests/test_encoding_independent.py` 결과: 2 passed |
| 전체 회귀 테스트 | PASS | `.venv/bin/python -m pytest -q` 결과: 38 passed |
| BGV | PASS | Printable/Random Bytes, 길이 4--31의 canonical 및 `[-1, 1]` 정규화 round-trip 성공; 형상 `[2, 32, 128]` |
| CGGE | PASS | Printable ASCII 전용, 길이 4--31의 canonical 및 정규화 round-trip 성공; 94 glyph 모두 고유; 형상 `[2, 32, 64]` |
| Direct Bits | PASS | Printable/Random Bytes, 길이 4--31의 canonical 및 정규화 round-trip 성공; 형상 `[32, 8]` |
| BGV/CGGE 실행 경로 | PASS | 각 representation으로 최소 diffusion run을 실행해 학습, decode, candidate budget 기록 경로를 확인 |

추가 독립 점검에서 CGGE 무작위 Printable message 896건, BGV 및 Direct Bits의 Printable/Random Bytes 경계 길이 28건씩을 encode/decode했다. 모든 case에서 원문과 복원문이 일치했고, 정규화 tensor도 동일하게 복원됐다.

## 독립 검증 방법

기존 codec helper를 재사용하지 않고 테스트에서 명세 기반 oracle을 별도로 계산했다.

- BGV: MSB-first bit, `2x4` glyph, `4x4` block, `4x8` slot 및 length/mask 위치를 직접 계산했다.
- Direct Bits: length header, payload, zero padding의 32x8 record를 직접 계산했다.
- CGGE: 고정 glyph table SHA-256을 알려진 값 `6ef6d0...eed50a`과 비교하고, 4x8 cell grid와 mask 위치를 별도로 계산했다.

이 oracle tensor가 실제 encoder 출력과 완전히 같은지, 그리고 canonical/정규화 tensor가 실제 decoder로 원문을 복원하는지를 길이 4--31 전체에서 검사했다. 독립 검증은 [tests/test_encoding_independent.py](tests/test_encoding_independent.py)에 재실행 가능하게 남겼다.

## 구현 적합성

- BGV는 첫 slot의 길이 header와 validity mask를 함께 기록한다. strict decoder는 길이 범위, length slot, 연속 payload mask를 검증한다.
- CGGE는 저장소 내 고정 8x8 glyph table을 사용한다. 시스템 font를 실행 시점에 rasterize하지 않으며, decoder는 reserve cell, 연속 mask, nearest-glyph 거리 threshold를 검증한다.
- Direct Bits는 한 byte 길이 header와 payload byte bit record를 사용한다.
- 세 representation 모두 canonical tensor를 `[-1, 1]`로 변환한 뒤에도 decoder가 원문을 복원한다.

## 현재 실험 진행 가능 여부

codec만의 G1 결정적 round-trip 요구사항은 충족한다. 그러나 전체 연구 계획의 G1은 actual-model reversible positive control도 99% 이상 요구한다. 기존 MD5/BGV pilot에서는 이 control이 `0/25` exact recovery였고, 생성 결과가 주로 length 또는 mask 검증에서 무효 처리됐다.

따라서 현재 상태에서는 다음 해석이 맞다.

1. 계획한 BGV·CGGE·Direct Bits 방식으로 data를 losslessly 표현하고 다시 읽는 것은 가능하다.
2. 현 diffusion 학습·sampling 설정으로는 그 representation을 안정적으로 생성하지 못한다.
3. hash-conditioned 본 실험 전, validation split에서 reversible positive control을 99% 이상으로 올리고 설정을 고정해야 한다.

## 범위의 한계

`run_experiment()`의 실행 중 G1 round-trip check는 대표 고정 corpus만 확인한다. 계획서에 명시한 전체 fixed corpus와 모든 94 CGGE glyph에 대한 artifact를 run마다 남기려면, 별도 G1 검증 artifact를 추가해야 한다. 이번 검증에서는 회귀 테스트와 독립 round-trip check로 해당 동작을 확인했다.

## 관련 자료

- `src/diffusion_hash_inv/encoding/bgv.py`
- `src/diffusion_hash_inv/encoding/cgge.py`
- `src/diffusion_hash_inv/encoding/bits.py`
- `tests/test_bgv_roundtrip.py`, `tests/test_cgge.py`, `tests/test_direct_bits_runner.py`
- `PILOT_RESULTS_REPORT.md`
