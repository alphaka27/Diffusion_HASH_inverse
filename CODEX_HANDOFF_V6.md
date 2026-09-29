# V6 구현 Handoff — 로컬 Codex용

**작성일:** 2026-09-29 KST · **대상:** 이 Mac(Apple M3 Max, macOS, arm64)에서 저장소 루트를 작업 디렉터리로 실행하는 Codex CLI

**목표:** [V6_IMPLEMENTATION_SPEC.md](V6_IMPLEMENTATION_SPEC.md)대로 `src/dhi_v6/` 패키지, 테스트, 사용 문서를 구현한다. **실험 실행은 목표가 아니다.** 정식 실행은 구현이 끝난 뒤 사용자가 직접 시작한다.

---

## 0. 한눈에 보기

| 항목 | 내용 |
|---|---|
| 만들 것 | `src/dhi_v6/` 9개 모듈, `tests/test_study_v6.py`, `V6_CLI.md`. `pyproject.toml`에는 console script 1줄과 pytest marker만 추가 |
| 작업 단위 | Phase 1–7(§5). **한 세션에 한 phase만** 수행하고 보고한 뒤 멈춘다 |
| 기준 문서 우선순위 | 구현 세부는 명세, 값은 [examples/v6-protocol.json](examples/v6-protocol.json), 과학적 설정은 [RESEARCH_PLAN_V6.md](RESEARCH_PLAN_V6.md) |
| 완료 정의 | §6의 최종 검증 명령이 모두 통과하고 Metal 테스트 skip이 0개 |
| 범위 밖 | 정식 실행(`dhi_v6.study run` 전부), 노출 inventory의 최종 분류, `git push` |

---

## 1. 착수 전 준비 (사용자가 한다)

1. **기준 커밋을 만든다.** V5 구현과 V6 계획·명세·등록 JSON·스크립트는 이미 커밋 `767f47d`(브랜치 `experiment-20260921`)에 들어 있다. 남은 것은 이 문서, `AGENTS.md`, 명세의 마지막 보완(§17 Metal 테스트 규칙, §18 handoff 안내)이다. 구현용 브랜치를 만들고 이것들을 커밋한 뒤 Codex를 시작한다. 이 커밋이 phase별 변경을 검토할 기준점이 된다.

   ```bash
   git switch -c v6-implementation
   git add AGENTS.md CODEX_HANDOFF_V6.md V6_IMPLEMENTATION_SPEC.md
   git commit -m "Add Codex handoff for V6 implementation"
   ```

2. **사람 결정 항목을 확인한다.** [RESEARCH_PLAN_V6.md](RESEARCH_PLAN_V6.md) 인계 안내의 결정 항목 1–5(cap, F1, δ, decoder, Stage S)다. 등록 JSON에는 권장값이 들어 있다. 권장값을 그대로 쓰면 할 일이 없고, 바꾸면 Codex를 시작하기 전에 JSON과 `.sha256`을 먼저 갱신한다.
3. **Codex 실행 설정을 정한다.**
   - 작업 디렉터리는 저장소 루트다.
   - 샌드박스는 작업 공간 쓰기를 허용한다. 네트워크는 필요 없다(§2).
   - 승인 정책은 Codex가 필요할 때 샌드박스 밖 실행을 요청할 수 있는 수준으로 둔다.
   - Metal이 필요한 명령은 샌드박스 안에서 실패할 수 있다. 이 저장소의 v3.1 작업에서도 샌드박스에서 Metal 접근이 막혀 승인된 권한으로 실행한 기록이 있다([V3_1_P2A10K_ANALYSIS_KO.md](V3_1_P2A10K_ANALYSIS_KO.md)). 이런 요청은 승인한다.
4. **커밋 방식을 고른다.** 기본값은 Codex가 phase 완료 기준을 만족하면 phase마다 한 번 커밋하고 push는 하지 않는 것이다. 직접 커밋하려면 요청 문구에 "커밋하지 마"를 넣는다(§10).
5. **잠자기를 막는다.** Metal 테스트가 길어질 수 있으므로 `caffeinate -i`를 권장한다.

---

## 2. 실행 환경 (이미 준비됨, 변경 금지)

| 항목 | 값 |
|---|---|
| Python | `.venv/bin/python` (3.12.4) |
| 패키지 | numpy 2.5.3, mlx 0.32.2, mlx-metal 0.32.2, torch 2.14.0, pytest 9.1.1 |
| 설치 방식 | 프로젝트가 editable로 설치되어 `src/`가 `sys.path`에 있다. 새 `src/dhi_v6/`는 추가 설치 없이 import된다 |
| Lock 일치 | 2026-09-29에 `uv sync --locked --extra mlx --group dev --dry-run --offline`로 확인했다. 외부 패키지 변경은 없고 프로젝트 editable 재설치만 남아 있다 |

- 명세 구현에 추가 패키지는 필요 없다. safetensors 저장은 MLX 내장 기능(`mx.save_safetensors`, `mx.load`)을 쓰고, 정규분포 분위수는 표준 라이브러리 `statistics.NormalDist`를 쓴다.
- **새 패키지 추가, 버전 변경, `uv.lock` 수정은 금지다.** V6는 환경 버전을 봉인하며, 명세의 결정성 실측도 이 버전 기준이다.
- Console script `hash-inverse-v6`는 `pyproject.toml`을 고친 뒤 사용자가 `uv sync --locked --extra mlx --group dev`를 실행해야 생긴다. Codex는 문서와 테스트에서 `.venv/bin/python -m dhi_v6.study`를 쓴다.
- **Metal 확인 명령:**

  ```bash
  .venv/bin/python -c "import mlx.core as mx; a = mx.ones((256, 256)); mx.eval(a @ a); print(mx.default_device())"
  ```

  `Device(gpu, 0)`이 출력되면 Metal을 쓸 수 있다. 오류가 나면 샌드박스 밖 실행을 요청한다.

---

## 3. 읽을 자료

| 순서 | 자료 | 목적 |
|---|---|---|
| 1 | [V6_IMPLEMENTATION_SPEC.md](V6_IMPLEMENTATION_SPEC.md) 전체 | 구현 기준. §0의 공백 해결과 §0.3의 보완을 먼저 읽는다 |
| 2 | [examples/v6-protocol.json](examples/v6-protocol.json) | `registration()`이 반환할 값 |
| 3 | [RESEARCH_PLAN_V6.md](RESEARCH_PLAN_V6.md) §4–§15 | 판정 규칙과 단계 순서의 이유 |
| 4 | `src/dhi_v5/*.py` | 복사해서 고칠 기반 코드. 난수 identity, 해시, split, stream, D1 모델·sampler, checkpoint, `Budget`, 통계 |
| 5 | `src/diffusion_hash_inv/mlx_models.py`, `encoding/{bgv,cgge,tokens}.py` | G3 U-Net, 초기화, 고정 구조, BGV·CGGE 인코더와 strict decoder, 글꼴 표 |
| 6 | [scripts/validate_research_plan_v6.py](scripts/validate_research_plan_v6.py) | `classify`와 `headline`. 생산 판정 엔진의 일치 기준 |
| 7 | [V5_CLI.md](V5_CLI.md), `tests/test_study_v5.py` | 사용 문서 형식과 Metal 테스트를 subprocess로 도는 방식 |

---

## 4. 반드시 지킬 규칙

### 4.1 수정 금지

- `src/dhi_v5/`, `src/diffusion_hash_inv/`, 기존 테스트 파일 전부
- `examples/v5-*`, `examples/v6-protocol.json`, `examples/v6-protocol.json.sha256`
- `scripts/` 전부, `RESEARCH_PLAN_*.md`, `V6_IMPLEMENTATION_SPEC.md`, 이 문서, `uv.lock`
- `local_experiment_archive/` 전체(읽기도 필요할 때만 한다)

명세나 JSON이 틀렸다고 판단되면 고치지 말고 멈춘 뒤 근거와 함께 보고한다.

### 4.2 허용되는 변경

- 새 파일: `src/dhi_v6/*.py`, `tests/test_study_v6.py`, `V6_CLI.md`
- `pyproject.toml`: `[project.scripts]`에 `hash-inverse-v6 = "dhi_v6.study:main"` 한 줄. `[tool.pytest.ini_options]`에 `markers = ["metal: requires MLX Metal"]`.

### 4.3 실행 금지

- `dhi_v6.study run …`(모든 stage와 phase), `approve-caps`, 실제 archive를 대상으로 하는 `audit`과 `inventory`. 단, 테스트에서 임시 디렉터리 fixture로 이 코드 경로를 부르는 것은 허용한다.
- `local_experiment_archive/` 아래에 쓰기
- **W3·W4의 test group, r=4 W1의 test group으로 후보를 생성하거나 평가하는 실행.** 테스트와 quick check도 포함한다. 테스트는 synthetic 과제, W1 r=64, validation group만 쓴다. 해시 값 계산과 split 생성 자체는 허용한다.
- `git push`, 새 패키지 설치, 네트워크 사용

### 4.4 구현 원칙

- 명세의 등록값, 게이트 기준, 난수 namespace, 레코드 형식, 판정 규칙, 단계 순서를 바꾸지 않는다.
- **테스트를 통과시키려고 기준을 느슨하게 하거나, 테스트를 skip·xfail로 바꾸지 않는다.**
- V5 코드를 복사할 때는 V6의 `PROTOCOL`과 `MASTER_SEED`를 쓴다. 복사 출처를 모듈 docstring에 한 줄로 남긴다.
- 명세가 정하지 않은 구현 선택(내부 함수 시그니처, JSON 산출물 필드명, 진행 출력 형식, 벡터화 방식)은 허용한다. 대신 `V6_CLI.md`의 "구현 결정" 절에 기록한다.
- 코드 스타일은 `dhi_v5`를 따른다: 간결한 함수, 명시적 key, 원자적 JSON 봉인.
- **Metal 테스트 규칙.** MLX가 필요한 테스트는 `@pytest.mark.metal`을 붙이고 V5처럼 subprocess로 실행한다. Metal을 쓸 수 없으면 skip하되, 환경변수 `DHI_V6_REQUIRE_METAL=1`이면 skip 대신 **실패**하게 만든다. Metal 테스트가 skip된 결과를 통과로 보고하지 않는다.

---

## 5. Phase별 작업

모든 phase는 같은 흐름이다. 명세의 해당 절을 읽고 → 구현하고 → 테스트를 쓰고 → 아래 명령을 실행하고 → §8 형식으로 보고한 뒤 멈춘다.

### Phase 1 — 골격, protocol, data (Metal 불필요)

- **명세:** §1, §2, §3, §4.
- **파일:**
  - `__init__.py`
  - `protocol.py`의 일부: `registration()`, `canonical`, `atomic_json`, `sealed_json`, `read_json`, `file_hash`, `source_manifest`, `environment`. `Budget`, 예산 계획, 노출 감사, 동결은 Phase 7에서 만든다.
  - `data.py` 전체
  - `pyproject.toml`의 script와 marker
- **`registration()`:** V5처럼 코드 안의 dict로 작성하고, 테스트에서 `examples/v6-protocol.json`과 같은지와 `.sha256`이 맞는지를 확인한다.
- **테스트:** `test_independence_and_registration`, `test_hash_windows_sources`, `test_token_codec_p_r`, `test_shared_streams_and_controls`(stream, permutation, derangement 부분).
- **완료 기준:**
  - 위 테스트 통과
  - 해시 게이트 축소판(두 source × 1,000 메시지, 모든 rung, W1–W4) 통과
  - `.venv/bin/python -m pytest tests/test_study_v5.py -q`가 12 passed를 유지
- **커밋:** `V6 phase 1: registration and data layer`

### Phase 2 — 이미지 codec (Metal 불필요, torch 사용)

- **명세:** §5.
- **파일:** `codecs.py`. NumPy BGV·CGGE 인코더, G3 고정 구조(NumPy), prototype decoder, strict decoder, 글꼴 표.
- **테스트:** `test_image_encoders_match_v31`, `test_prototype_and_strict_decoders`.
- **완료 기준:**
  - 두 source의 모든 기호와 길이 4–31에서 round-trip 100%
  - v3.1 인코더와 텐서 일치(무작위 1,000 메시지 + 길이 4·31 경계)
  - strict decoder가 v3.1 decoder와 일치(잡음을 넣은 fixture 이미지 200개)
  - 동점 규칙과 글꼴 SHA-256(`6ef6d0bf…ed50a`) 확인
  - §7의 성능 측정값 보고
- **커밋:** `V6 phase 2: image codecs and prototype decoders`

### Phase 3 — 모델, sampler, CLP 점수 (Metal 필요)

- **명세:** §6, §8, §9.
- **파일:** `models.py`.
- **레이어 이름:** 등가 테스트에서 가중치를 그대로 옮길 수 있도록 원본과 같은 속성 이름을 쓴다. D1-S는 V5와 같이 `length_head`, `embedding`, `hidden`, `output`, `residual`, G3-U는 v3.1과 같이 `input`, `down`, `middle`, `up`, `output`, `condition`, `condition_output`, `length_head`다.
- **테스트(`metal`):** `test_parameter_counts`, `test_d1s_parity_with_v5`(V5 모델 가중치를 V6 모델에 불러 같은 key로 비교), `test_g3u_forward_parity_with_v31`, `test_samplers_reference_and_invariance`, `test_clp_antisymmetry`.
  - `test_samplers_reference_and_invariance`는 축소 규모로 한다. D1-S는 후보 256개에서 scalar와 벡터화가 bitwise 같아야 한다. G3는 S_G=25에서 후보 64개로 scalar 허용오차를 검사하고, batch 64/128/256이 bitwise 같아야 한다.
- **완료 기준:** `DHI_V6_REQUIRE_METAL=1`로 전체 통과, skip 0. 파라미터 수가 명세 §6과 일치해야 한다: 508,668 / 1,252,572 / 910,638 / 513,326 / 6,415,308.
- **커밋:** `V6 phase 3: models, samplers and likelihood scores`

### Phase 4 — 학습 runtime (Metal 필요)

- **명세:** §7.
- **파일:** `runtime.py`의 학습 부분. contract, segment, checkpoint(V5 복사), 학습 digest 저장소, `charge_work`, `attempt.json`, `resume_from`, validation 진단.
- **테스트(`metal`):** `test_training_resume_and_continuation`.
  - 계열별(D1-S P, D1-S R, G3 BGV, G3 CGGE)로 batch 8, 4 update, 2 update마다 checkpoint를 두고 중단·재개가 bitwise 같아야 한다.
  - 4→8 이어 학습 결과가 처음부터 8 update 학습한 결과와 같아야 한다.
  - 같은 source의 두 파이프라인이 digest 저장소를 공유하고 교차 검증해야 한다.
- **완료 기준:** 위 테스트 통과, skip 0. 학습 부대비용 측정값 보고(§7).
- **커밋:** `V6 phase 4: training runtime with segments and continuation`

### Phase 5 — 평가 stream, 원장, 검증기, 재생성 감사 (일부 Metal)

- **명세:** §10.
- **파일:** `runtime.py`의 평가 부분. 36-byte 레코드, 블록 파일, commit JSON, 잘라내기 재개, 독립 검증기(hashlib 경로), trial 요약, 중복, 학습 일치, 재생성 감사, trial schedule.
- **테스트:** `test_ledger_commit_resume_tamper_regeneration`. Random stream fixture와 작은 학습 모델 stream을 쓴다. 학습 모델 부분은 `metal`로 표시한다.
- **완료 기준:**
  - 레코드 크기가 정확히 36 byte
  - 중단 후 재개한 원장이 끊김 없이 만든 원장과 byte 단위로 같음
  - flag 변조와 payload 변조를 각각 검출
  - invalid와 중복이 attempt를 소비
  - 검증기 처리량 보고
- **커밋:** `V6 phase 5: evaluation ledger, verifier and regeneration audit`

### Phase 6 — 통계·판정 엔진 (Metal 불필요)

- **명세:** §11.
- **파일:** `statistics.py`. 구간, CP 한계(V5 복사), 순차 엔진(interim·full 분류, 공동 중단, 예산 종료), 재현, P 검정, 대비, C4, 종합 판정, production calibration(trial 단위 모의).
- **테스트:** `test_sequential_engine_rules`, `test_design_script_parity`, `test_replication_p_contrasts_c4_headline`, `test_calibration_quick`.
  - `test_design_script_parity`는 `importlib`로 `scripts/validate_research_plan_v6.py`를 불러, 무작위 입력 10,000개에서 `classify`와 `headline` 결과가 생산 코드와 같은지 확인한다.
  - `test_calibration_quick`은 시나리오당 200회로 축소한다.
- **완료 기준:** z 값이 소수 4자리까지 명세와 일치(3.4808, 3.0902, 3.0233, 2.8782, 2.8653, 3.2905). 설계 스크립트와 판정 100% 일치.
- **커밋:** `V6 phase 6: sequential decision engine and calibration`

### Phase 7 — 단계 실행기, 예산, 노출 감사, CLI, 보고서, 문서

- **명세:** §0.3, §12–§16.
- **파일:**
  - `checks.py`(G1–G10과 `--quick` 축소판)
  - `study.py`(A의 phase들, C, R, P, S, 보고서, CLI)
  - `protocol.py`의 나머지(`Budget`, 예산 계획, `halt`/`approve-caps`, 노출 감사 v6, `inventory --draft`, 동결·검증)
  - `V6_CLI.md`
- **테스트:** `test_budget_plan_decisions`, `test_audit_v6_rules`, `test_stage_order_and_blinding`, `test_partial_report_is_not_rejection`, CLI 전제조건 검사(빈 디렉터리 요구, `failure.json`이 있으면 거부, `approve-caps`는 MD5 조건 데이터가 없을 때만 허용).
- **`V6_CLI.md`:** V5_CLI.md 형식을 따른다. 설치, 실행 순서, 명령, 산출물, 운영 주의를 쓰고 "구현 결정" 절을 둔다.
- **완료 기준:** §6의 최종 검증 통과.
- **커밋:** `V6 phase 7: stage runner, budget, audit, CLI and report`

---

## 6. 최종 검증 (Phase 7 끝, 샌드박스 밖에서 실행)

```bash
DHI_V6_REQUIRE_METAL=1 .venv/bin/python -m pytest tests/test_study_v6.py -q -rs
.venv/bin/python -m pytest tests/test_study_v5.py -q
DHI_V6_REQUIRE_METAL=1 .venv/bin/python -m dhi_v6.study check --root "$TMPDIR/dhi-v6-quick" --quick
.venv/bin/python -m dhi_v6.study plan
git status --short
```

| 명령 | 기대 결과 |
|---|---|
| V6 테스트 | 전부 통과, skip 0 |
| V5 테스트 | 12 passed |
| Quick check | 모든 축소 게이트 통과. 결과에 "정식 PASS 아님"이 표시됨 |
| `plan` | 출력이 `examples/v6-protocol.json`과 같음 |
| `git status` | §4.2에서 허용한 파일만 변경됨 |

---

## 7. 성능 목표 (권장, phase 보고에 측정값 포함)

계획의 예산(학습 부대비용 19 ms/update, 생성 stream 효율 0.546)보다 느려지면 예산 계획이 틀어진다.

| 항목 | 목표 | 근거 |
|---|---|---|
| 학습 부대비용(모델 계산 제외: stream·인코딩·digest·`charge_work`) | update당 5 ms 이하 | V5 추정 19 ms. SQLite commit 제거 효과 |
| BGV·CGGE 인코딩(256 메시지) | 5 ms 이하 | 학습 update마다 실행 |
| Prototype decode(후보 2,048개) | BGV·CGGE 각 0.3초 이하 | BGV 25 step 생성 시간(약 3.5초)의 10% 이하 |
| 원장 기록(해시·학습 조회 포함, 후보 2,048개) | 50 ms 이하 | 생성 대비 무시 가능해야 함 |
| 독립 검증기 | 초당 100,000행 이상 | C 원장 약 8,850만 행을 15분 안에 |

---

## 8. Phase 종료 보고 형식

```text
Phase N 완료 보고
- 변경 파일: …
- 테스트: passed a / failed b / skipped c (Metal 사용: 예/아니오, DHI_V6_REQUIRE_METAL=1: 예/아니오)
- V5 회귀 테스트: 12 passed 여부
- 성능 측정(§7 해당 항목): …
- 명세와 다르게 구현한 점과 이유: 없음 / …
- 구현 결정(V6_CLI.md에 기록할 것): …
- 남은 문제·질문: …
- 커밋: <hash 또는 "사용자 요청으로 커밋 안 함">
```

명세가 모호하거나 모순되어 진행할 수 없으면 구현하지 말고, 해당 절과 선택지를 적어 질문한다.

---

## 9. 자주 틀리는 부분

| 항목 | 올바른 구현 |
|---|---|
| Token ID 순서 | PAD = S, EOS = S+1, MASK = S+2(V5 규칙). v3.1 `tokens.py`의 순서(EOS가 먼저)와 **반대**다 |
| Synthetic P prefix | **대문자** hex(`0123456789ABCDEF`, V5). v3.1은 소문자였다 |
| Window byte | W3 = `(d8<<4)|(d9>>4)`, W4 = `(d4<<4)|(d5>>4)`. W2는 split과 stream에서 금지 |
| Namespace | split은 stage를 포함하지 않는다(C와 S가 공유). 학습 stream과 permutation은 pipeline을 포함하지 않는다(같은 source끼리 공유). 가중치와 corruption은 Main·Shuffled가 공유한다 |
| Identity 상수 | V6 `PROTOCOL`·`MASTER_SEED`를 쓴다. V5 상수를 그대로 import하면 안 된다 |
| Embedding | one-hot 행렬곱(`DenseEmbedding` 방식). MLX scatter gradient의 비결정성을 피하기 위해서다 |
| G3 loss | 활성 payload 픽셀 균일 평균. v3.1의 prefix/suffix 분리 loss를 복사하면 안 된다 |
| G3 sampler | `times = rint(linspace(999, 0, S_G, float32))`. 매 step x̂₀ clip, 마지막 ᾱ′ = 1, 최종 clip 후 decode. key draw는 길이 0, 초기 잡음 1 |
| G3 batch 1 | conv가 batch 1에서 최대 1.8×10⁻⁷ 다르다. scalar 참조 비교에만 허용오차를 쓰고, batch ≥ 64 비교는 bitwise |
| Metal 테스트 | 샌드박스에서 조용히 skip된다. `DHI_V6_REQUIRE_METAL=1`로 실패 처리 |
| 레코드 | 정확히 36 byte. Discrete와 Random의 margin은 NaN |
| `registration()` | JSON과 정확히 같아야 한다(키, 값, 타입). 값을 바꾸고 싶으면 멈추고 보고 |
| 이어 학습 | 새 run이 `resume_from`의 checkpoint hash를 검증한다. lr은 update 번호만의 함수여야 등가성이 성립한다 |

---

## 10. Codex 요청 문구 예시

Phase마다 새 세션에서 다음과 같이 요청한다(N만 바꾼다).

```text
AGENTS.md와 CODEX_HANDOFF_V6.md를 먼저 읽고 Phase N만 수행해 줘.
V6_IMPLEMENTATION_SPEC.md의 해당 절을 따르고, 완료 기준을 모두 만족하면 §8 형식으로 보고한 뒤 멈춰.
Metal이 필요한 명령이 샌드박스에서 실패하면 샌드박스 밖 실행을 요청해.
명세와 충돌하거나 모호한 점이 있으면 구현하지 말고 질문해.
```

직접 커밋하려면 마지막 줄에 "커밋하지 마"를 추가한다.

---

## 11. 구현 후 절차 (사용자)

1. Phase마다 보고와 diff를 검토한다. 커밋은 기본값대로 Codex가 하거나 직접 한다.
2. Phase 7 뒤 console script를 등록한다: `uv sync --locked --extra mlx --group dev`.
3. 노출 inventory v6 초안을 만들고(`inventory --draft`) 규칙에 맞지 않는 파일을 직접 분류한다.
4. 정식 실행은 [RESEARCH_PLAN_V6.md](RESEARCH_PLAN_V6.md) §14 순서대로 직접 시작한다. 첫 단계는 A-impl 정식 게이트이며 약 30–45분 걸린다.
