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

## 2026-09-17T13:56:44.891023+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed0 validation sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-0/validation-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 0

## 2026-09-17T13:56:45.981532+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed0 test sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-0/test-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 0

## 2026-09-17T13:56:47.020504+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed0 test sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-0/test-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 0

## 2026-09-17T13:56:48.082677+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed0 test sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-0/test-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 0

## 2026-09-17T13:56:48.094743+09:00

Gate: G2
Stage: VALIDATE representation/model seed

Completed:
- cgge seed0 PASS

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-0/metrics.json
- output/g2/cgge/seed-0/complete.json

Decision: READY

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --all

## 2026-09-17T13:56:48.101556+09:00

Gate: G2
Stage: RUN_FULL

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
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:48.880662+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bits seed1 training step 250

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:49.068885+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bits seed1 training step 500

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:49.232037+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bits seed1 training step 750

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:49.391816+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bits seed1 training step 1000

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:49.559179+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bits seed1 training step 1250

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:49.764777+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bits seed1 training step 1500

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:49.943643+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed1 train sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/train-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:50.086555+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed1 train sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/train-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:50.220724+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed1 train sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/train-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:50.265722+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed1 validation sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/validation-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:50.310253+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed1 validation sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/validation-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:50.356799+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed1 validation sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/validation-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:50.412497+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed1 test sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/test-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:50.466023+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed1 test sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/test-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:50.520879+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed1 test sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/test-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 1

## 2026-09-17T13:56:50.534453+09:00

Gate: G2
Stage: VALIDATE representation/model seed

Completed:
- bits seed1 PASS

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-1/metrics.json
- output/g2/bits/seed-1/complete.json

Decision: READY

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --all

## 2026-09-17T13:56:50.540247+09:00

Gate: G2
Stage: RUN_FULL

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
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:57:02.518969+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bgv seed1 training step 250

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:57:14.430215+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bgv seed1 training step 500

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:57:25.992360+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bgv seed1 training step 750

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:57:37.653827+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bgv seed1 training step 1000

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:57:49.181808+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bgv seed1 training step 1250

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:58:00.661450+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bgv seed1 training step 1500

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:58:09.785055+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed1 train sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/train-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:58:18.887895+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed1 train sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/train-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:58:27.889077+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed1 train sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/train-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:58:29.952801+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed1 validation sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/validation-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:58:31.996526+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed1 validation sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/validation-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:58:34.068772+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed1 validation sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/validation-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:58:36.096999+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed1 test sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/test-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:58:38.551158+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed1 test sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/test-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:58:40.816752+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed1 test sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/test-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 1

## 2026-09-17T13:58:40.827213+09:00

Gate: G2
Stage: VALIDATE representation/model seed

Completed:
- bgv seed1 PASS

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-1/metrics.json
- output/g2/bgv/seed-1/complete.json

Decision: READY

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --all

## 2026-09-17T13:58:40.832299+09:00

Gate: G2
Stage: RUN_FULL

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
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:58:47.018240+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- cgge seed1 training step 250

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:58:53.258967+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- cgge seed1 training step 500

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:58:59.463239+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- cgge seed1 training step 750

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:59:05.507121+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- cgge seed1 training step 1000

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:59:11.548865+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- cgge seed1 training step 1250

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:59:17.491437+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- cgge seed1 training step 1500

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:59:22.122828+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed1 train sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/train-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:59:26.637324+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed1 train sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/train-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:59:31.194411+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed1 train sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/train-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:59:32.255814+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed1 validation sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/validation-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:59:33.347066+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed1 validation sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/validation-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:59:34.402215+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed1 validation sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/validation-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:59:35.447130+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed1 test sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/test-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:59:36.482640+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed1 test sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/test-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:59:37.516743+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed1 test sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/test-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 1

## 2026-09-17T13:59:37.526516+09:00

Gate: G2
Stage: VALIDATE representation/model seed

Completed:
- cgge seed1 PASS

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-1/metrics.json
- output/g2/cgge/seed-1/complete.json

Decision: READY

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --all

## 2026-09-17T13:59:37.531827+09:00

Gate: G2
Stage: RUN_FULL

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
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:37.724462+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bits seed2 training step 250

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:37.925513+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bits seed2 training step 500

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:38.107781+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bits seed2 training step 750

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:38.277897+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bits seed2 training step 1000

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:38.434833+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bits seed2 training step 1250

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:38.615753+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bits seed2 training step 1500

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:38.811575+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed2 train sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/train-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:38.990926+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed2 train sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/train-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:39.162704+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed2 train sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/train-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:39.216419+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed2 validation sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/validation-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:39.264602+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed2 validation sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/validation-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:39.310434+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed2 validation sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/validation-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:39.359684+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed2 test sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/test-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:39.406442+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed2 test sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/test-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:39.449073+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bits seed2 test sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/test-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 2

## 2026-09-17T13:59:39.458901+09:00

Gate: G2
Stage: VALIDATE representation/model seed

Completed:
- bits seed2 PASS

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bits/seed-2/metrics.json
- output/g2/bits/seed-2/complete.json

Decision: READY

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --all

## 2026-09-17T13:59:39.464395+09:00

Gate: G2
Stage: RUN_FULL

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
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T13:59:51.175099+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bgv seed2 training step 250

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:00:02.997115+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bgv seed2 training step 500

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:00:14.774942+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bgv seed2 training step 750

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:00:27.429190+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bgv seed2 training step 1000

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:00:39.657505+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bgv seed2 training step 1250

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:00:51.436390+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- bgv seed2 training step 1500

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:01:00.902039+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed2 train sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/train-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:01:10.225487+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed2 train sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/train-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:01:19.309108+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed2 train sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/train-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:01:21.352240+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed2 validation sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/validation-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:01:23.387198+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed2 validation sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/validation-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:01:25.417010+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed2 validation sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/validation-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:01:27.466827+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed2 test sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/test-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:01:29.508000+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed2 test sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/test-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:01:31.528313+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- bgv seed2 test sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/test-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bgv --model-seed 2

## 2026-09-17T14:01:31.539704+09:00

Gate: G2
Stage: VALIDATE representation/model seed

Completed:
- bgv seed2 PASS

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/bgv/seed-2/metrics.json
- output/g2/bgv/seed-2/complete.json

Decision: READY

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --all

## 2026-09-17T14:01:31.558073+09:00

Gate: G2
Stage: RUN_FULL

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
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:01:37.756244+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- cgge seed2 training step 250

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:01:43.715887+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- cgge seed2 training step 500

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:01:49.741862+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- cgge seed2 training step 750

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:01:55.721481+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- cgge seed2 training step 1000

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:02:01.815809+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- cgge seed2 training step 1250

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:02:07.931513+09:00

Gate: G2
Stage: RUN_FULL training

Completed:
- cgge seed2 training step 1500

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/training_resume.pt

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:02:12.570107+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed2 train sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/train-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:02:17.166796+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed2 train sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/train-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:02:23.080833+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed2 train sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/train-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:02:24.461081+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed2 validation sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/validation-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:02:26.517558+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed2 validation sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/validation-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:02:29.673130+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed2 validation sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/validation-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:02:30.593003+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed2 test sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/test-sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:02:31.493257+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed2 test sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/test-sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:02:32.400623+09:00

Gate: G2
Stage: VALIDATE paired unit

Completed:
- cgge seed2 test sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/test-sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation cgge --model-seed 2

## 2026-09-17T14:02:32.410095+09:00

Gate: G2
Stage: VALIDATE representation/model seed

Completed:
- cgge seed2 PASS

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/cgge/seed-2/metrics.json
- output/g2/cgge/seed-2/complete.json

Decision: READY

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.conditional_dependence --all

## 2026-09-17T14:02:32.412711+09:00

Gate: G2
Stage: DECIDE/CHECKPOINT

Completed:
- G2 PASS: All three representations and three model seeds satisfy the frozen causal criteria.

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 52 passed; independent encoding 2 passed

Artifacts:
- output/g2/gate_summary.json
- output/g2/report.md

Decision: READY

Next exact action:
- Implement and freeze G3 information ladder before evaluation.

## 2026-09-17T14:05:23.113102+09:00

Gate: G3
Stage: IMPLEMENT

Completed:
- G2 PASS; G3 protocol frozen before test evaluation

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 55 passed; independent encoding 2 passed

Artifacts:
- output/g2/report.md
- output/g3/config_frozen.json

Decision: READY

Next exact action:
- Implement G3 frozen native-cell masking and deterministic tests.

## 2026-09-17T14:05:41.580012+09:00

Gate: G3
Stage: TEST

Completed:
- G3 masking implementation; regression 58 passed; independent 2 passed

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 58 passed; independent encoding 2 passed

Artifacts:
- output/session_checkpoints/g3_pytest.log
- output/g3/config_frozen.json

Decision: READY

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.information_ladder --representation bits --model-seed 0

## 2026-09-17T14:05:44.323212+09:00

Gate: G3
Stage: RUN_SMALL

Completed:
- Stage started

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 58 passed; independent encoding 2 passed

Artifacts:
- 

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.information_ladder --all

## 2026-09-17T14:05:44.610278+09:00

Gate: G3
Stage: VALIDATE

Completed:
- G3 bits seed0 sampling0 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 58 passed; independent encoding 2 passed

Artifacts:
- output/g3/bits/seed-0/sampling-0.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.information_ladder --all

## 2026-09-17T14:05:44.947106+09:00

Gate: G3
Stage: VALIDATE

Completed:
- G3 bits seed0 sampling1 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 58 passed; independent encoding 2 passed

Artifacts:
- output/g3/bits/seed-0/sampling-1.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.information_ladder --all

## 2026-09-17T14:05:45.275861+09:00

Gate: G3
Stage: VALIDATE

Completed:
- G3 bits seed0 sampling2 complete

Results:
- Status: RUNNING

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 58 passed; independent encoding 2 passed

Artifacts:
- output/g3/bits/seed-0/sampling-2.json

Decision: RUNNING

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.information_ladder --all

## 2026-09-17T14:05:45.292608+09:00

Gate: G3
Stage: VALIDATE

Completed:
- G3 bits seed0 PASS

Results:
- Status: READY

Files changed:
- State/progress/next-action documents and listed artifacts.

Tests:
- 58 passed; independent encoding 2 passed

Artifacts:
- output/g3/bits/seed-0/metrics.json

Decision: READY

Next exact action:
- .venv/bin/python -m diffusion_hash_inv.information_ladder --all
