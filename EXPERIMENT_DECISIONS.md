# Experiment decisions

## Protocol precedence
Decision: Use the requested sequential G2 conditional dependence → G3 information ladder → G4 toy hash → G5 candidate efficacy → G6 full experiment gate namespace. Preserve RESEARCH_PLAN.md and all prior G0/G1 results unchanged.
Reason: RESEARCH_PLAN.md has older G0–G4 numbering; the latest user request and DIFFUSION_GATE_REPORT.md define this continuation.

## G2 causal evaluation and seeds
Decision: Reuse each representation's G1-C checkpoint for model_seed=0. Train only G1-C-equivalent correct-conditioned models for seeds 1/2 on the SAME saved 64/16/16 message split. Evaluate train/validation/test with sampling seeds 0/1/2 and correct/deranged/zero conditions. Save each split/seed before proceeding. No retraining per intervention. Optional random diagnostic omitted.
Reason: Isolate condition intervention from training and dataset randomness. Reinitializing the sampling generator gives identical x_T; record and assert first model input equality.
Alternative rejected: Separate training for negative controls, or rerunning G1 ladders.
Seed namespace: dataset_seed=0, split_seed=0; model_seed=0/1/2. Original seed0 shares a batch/timestep/noise generator (legacy checkpoint); new runs separate these streams. Sampling and condition streams are independent.

## G2 decision frozen before evaluation
Decision: PASS requires all three representations and model seeds 0/1/2: correct exact >=0.99 on train/validation/test, strictly greater exact than deranged and zero on every split, and nonzero output bit disagreement versus both controls. Deranged donors must have no fixed points; pairs must have identical initial noise. Missing/failed integrity → INVALID; missing model coverage → PARTIAL; valid completed metric criterion failure → FAIL. Do not proceed on any non-PASS outcome.
Reason: Matches G1 near-perfect reconstruction and user's directional causal criteria; no test-driven threshold changes. Sampling seeds are repeated measurements, not independent targets.

## Later gates
Decision: Freeze each gate protocol before its first test evaluation, use K=[1,10,100] from RESEARCH_PLAN.md for G5/G6. Do not implement or execute a later gate before the preceding gate passes. Failure before candidate efficacy implies the overall hash advantage question remains INCONCLUSIVE, not NO.
