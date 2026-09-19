# Next actions / exact resume point

Current phase: scientific Phase 0 freeze BLOCKED_SPECIFICATION; independent Phase 1 engineering checks complete. First unresolved job: **SPECIFICATION_FREEZE**. Do not interpret completed engineering fixtures as scientific gate passes.

## Start every session here

```bash
.venv/bin/python -m diffusion_hash_inv.automation --verify
.venv/bin/python -m diffusion_hash_inv.experiment_state --verify
```

Read `artifacts/automation/EXPERIMENT_STATE.json`, `IMPLEMENTATION_AUDIT.md`, `artifacts/config/CONFIRMATORY_SPECIFICATION_TEMPLATE.json`, and `artifacts/reports/FINAL_EXPERIMENT_REPORT.md`. Current source version is `84132ed8dd074e51d7bea8502a9fd5169f8aa22cd865d0476d517eb5c057584a`; base Git commit `9b2b69a41cf4c5a57fea21fa58af08372133a0a3` plus uncommitted changes. Code snapshot is `artifacts/automation/SOURCE_SNAPSHOT_84132ed8dd07.tar.gz`.

## Independent engineering follow-up before training

Finish a confirmatory entry point that accepts registered external exact-quota manifests, enforces complete G0/G1 before training, implements the registered validation-only checkpoint rule and emits all secondary diagnostics/compute fields. Per-attempt sampling interruption resume and direct predictor final shared specification are still pending. Reuse current primitives; do not run the engineering CLI as a substitute. Add regression checks and run:

```bash
.venv/bin/python -m pytest -q
```

## Specification-dependent next work

Resolve each null in `artifacts/config/CONFIRMATORY_SPECIFICATION_TEMPLATE.json` from an actual registered decision or a validation-only development protocol. L_max=31 is inherited under the user's explicit legacy-value fallback; fresh dataset/split seeds, hyperparameters/selection/search budget, control target counts, statistical membership/seeds and compute/stopping envelope remain unresolved. Do not fill them from fixture outcomes. Ensure all observed old/engineering test corpora are excluded. Freeze a new content-addressed scientific config; never overwrite `artifacts/config/frozen/` snapshots.

Generate the 34 scientific datasets with registered quotas and bounded construction, run G0, codec/verifier checks and actual-model reversible controls for every family. Only after full G0/G1 PASS start the first planned model job: `MD5_PRINTABLE_GAUSSIAN_BGV_Q8_HASHONLY_SEED0_TRAIN`, then its prefix evaluations. Continue MD5 q8/12/16, q20/24/32/64, full128; then SHA-256 including full256. All job IDs/dependencies remain in UNFINISHED_JOBS.json.

To verify/skip the already completed engineering preflight:

```bash
.venv/bin/python -m diffusion_hash_inv.preflight --run
```

The completed engineering model checkpoints are under `artifacts/training/`; `complete.json` checksums prevent duplicate training. `training_resume.pt` contains optimizer/model/RNG state. A changed code/config requires a new run ID. Legacy `output/` and root EXPERIMENT_STATE.json remain unchanged; do not resume their stopped scientific protocol as this new study.
