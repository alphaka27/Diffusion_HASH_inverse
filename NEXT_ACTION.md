Current gate: G5
Current stage: FINAL CHECKPOINT
Status: STOPPED — G5 INCONCLUSIVE

Next action (single, read-only): Verify persisted state and all final artifact hashes.

Command:
```bash
.venv/bin/python -m diffusion_hash_inv.experiment_state --verify
```

Expected result: integrity PASS; G0–G4 PASS, G5 INCONCLUSIVE, G6 NOT_RUN.
Then report the existing conclusion from EXPERIMENT_G2_TO_G6_REPORT.md.

Prerequisites: read EXPERIMENT_STATE.json, this file, the last EXPERIMENT_PROGRESS.md section and output/gate_summary.json. Retain completed artifacts; do not rerun training/evaluation. No unfinished experimental action remains under the frozen protocol.

Do not proceed to G6, change the completed criteria, regenerate completed candidates or repeat G0/G1. A continuation request alone does not override the fail-stop. If hashes mismatch, inspect and repair only the affected artifact/state boundary from saved checkpoints/units.
