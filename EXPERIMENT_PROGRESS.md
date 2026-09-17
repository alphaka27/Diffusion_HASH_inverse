# Experiment progress (append only)

## 2026-09-17T13:51:39.594926+09:00

Gate: G2
Stage: INITIALIZE

Completed:
- Repository and existing reports/code/artifact inventory.
- Created persistent state and G0/G1 SHA256 manifest.

Results:
- G0 PASS; G1-A/B/C PASS for bits, BGV, CGGE. G2–G6 NOT RUN.
- Working tree initially clean.

Files changed:
- Four root state documents; prerequisite manifest.

Tests:
- Pending regression.

Artifacts:
- output/session_checkpoints/prerequisite_sha256.json

Decision: CONTINUE

Next exact action:
- Run prerequisite regression.

## 2026-09-17T13:54:20.925566+09:00

Gate: G2
Stage: IMPLEMENT

Completed:
- G2 paired evaluator and atomic recovery checkpoints implemented

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- Prerequisite regression: see saved logs

Artifacts:
- src/diffusion_hash_inv/conditional_dependence.py
- src/diffusion_hash_inv/experiment_state.py
- tests/test_conditional_dependence.py

Decision: READY

Next exact action:
- .venv/bin/python -m pytest -q

## 2026-09-17T13:54:51.687363+09:00

Gate: G2
Stage: TEST

Completed:
- G2 regression 52/52; independent encoding 2/2 PASS

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/session_checkpoints/g2_pytest.log
- output/session_checkpoints/g2_independent.log

Decision: READY

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 0

## 2026-09-17T13:54:52.589016+09:00

Gate: G2
Stage: RUN_SMALL

Completed:
- Stage started

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- 

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 0

## 2026-09-17T13:54:52.774958+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed0 train sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-0/train-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 0

## 2026-09-17T13:54:52.928677+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed0 train sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-0/train-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 0

## 2026-09-17T13:54:53.098719+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed0 train sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-0/train-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 0

## 2026-09-17T13:54:53.161272+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed0 validation sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-0/validation-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 0

## 2026-09-17T13:54:53.220645+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed0 validation sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-0/validation-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 0

## 2026-09-17T13:54:53.281200+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed0 validation sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-0/validation-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 0

## 2026-09-17T13:54:53.342523+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed0 test sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-0/test-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 0

## 2026-09-17T13:54:53.402582+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed0 test sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-0/test-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 0

## 2026-09-17T13:54:53.454826+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed0 test sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-0/test-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 0

## 2026-09-17T13:54:53.466306+09:00

Gate: G2
Stage: VALIDATE representation/model seed

Completed:
- bits seed0 PASS

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-0/metrics.json
- output/g2/bits/seed-0/complete.json

Decision: READY

Next exact action:
- Inspect completed metrics and select the next unfinished G2 run; DECIDE G2 on failure.

## 2026-09-17T13:55:11.806576+09:00

Gate: G2
Stage: RUN_SMALL

Completed:
- Stage started

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- 

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 0

## 2026-09-17T13:55:20.883060+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed0 train sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-0/train-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 0

## 2026-09-17T13:55:29.963504+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed0 train sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-0/train-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 0

## 2026-09-17T13:55:38.918648+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed0 train sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-0/train-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 0

## 2026-09-17T13:55:40.949421+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed0 validation sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-0/validation-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 0

## 2026-09-17T13:55:42.992272+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed0 validation sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-0/validation-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 0

## 2026-09-17T13:55:44.996283+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed0 validation sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-0/validation-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 0

## 2026-09-17T13:55:47.061419+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed0 test sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-0/test-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 0

## 2026-09-17T13:55:49.043716+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed0 test sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-0/test-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 0

## 2026-09-17T13:55:51.269718+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed0 test sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-0/test-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 0

## 2026-09-17T13:55:51.282314+09:00

Gate: G2
Stage: VALIDATE representation/model seed

Completed:
- bgv seed0 PASS

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-0/metrics.json
- output/g2/bgv/seed-0/complete.json

Decision: READY

Next exact action:
- Inspect completed metrics and select the next unfinished G2 run; DECIDE G2 on failure.

## 2026-09-17T13:56:27.712868+09:00

Gate: G2
Stage: RUN_SMALL

Completed:
- Stage started

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- 

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 0

## 2026-09-17T13:56:32.360664+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed0 train sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-0/train-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 0

## 2026-09-17T13:56:36.888099+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed0 train sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-0/train-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 0

## 2026-09-17T13:56:41.591837+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed0 train sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-0/train-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 0

## 2026-09-17T13:56:42.617194+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed0 validation sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-0/validation-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 0

## 2026-09-17T13:56:43.716811+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed0 validation sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-0/validation-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 0
