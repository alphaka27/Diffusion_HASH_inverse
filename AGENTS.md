# 에이전트 작업 규칙

이 저장소의 현재 작업은 **V6 구현**이다. 작업 단계, 완료 기준, 요청 형식은 [CODEX_HANDOFF_V6.md](CODEX_HANDOFF_V6.md)를 따른다.

- **기준 문서.** 구현 세부는 [V6_IMPLEMENTATION_SPEC.md](V6_IMPLEMENTATION_SPEC.md), 등록값은 [examples/v6-protocol.json](examples/v6-protocol.json), 과학적 설정은 [RESEARCH_PLAN_V6.md](RESEARCH_PLAN_V6.md)다.
- **작업 단위.** 한 세션에 handoff의 phase 하나만 수행하고, 보고한 뒤 멈춘다.
- **수정 금지.** `src/dhi_v5/`, `src/diffusion_hash_inv/`, 기존 테스트, `examples/`의 기존 파일, `scripts/`, 계획·명세·handoff 문서, `uv.lock`, `local_experiment_archive/`.
- **환경.** Python은 `.venv/bin/python`만 쓴다. 새 패키지 설치와 의존성 변경은 금지다.
- **실행 금지.** `dhi_v6.study run`, `approve-caps`, 실제 archive를 대상으로 하는 `audit`·`inventory`를 실행하지 않는다. W3·W4 test group과 r=4 W1 test group으로 후보를 만들거나 평가하지 않는다.
- **Metal.** MLX 명령이 샌드박스에서 실패하면 샌드박스 밖 실행을 요청한다. Metal 테스트는 `DHI_V6_REQUIRE_METAL=1`로 실행하며, skip을 통과로 보고하지 않는다.
- **정직성.** 테스트를 통과시키려고 기준을 바꾸거나 skip·xfail로 바꾸지 않는다. 명세가 틀렸거나 모호하면 멈추고 질문한다.
- **Git.** `git push`는 금지다. 원장, checkpoint, 보고서 같은 실행 산출물은 커밋하지 않는다.
- **언어.** 보고와 문서는 한국어로 쓴다. 코드, 식별자, 커밋 메시지는 영어로 쓴다.
