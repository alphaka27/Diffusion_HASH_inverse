# Diffusion Hash — G2 to G6 Experiment Report

Finalized: 2026-09-17T14:18:54.012092+09:00

## 1. Executive Summary

연구 질문의 답: **INCONCLUSIVE**. G2와 G3는 세 representation × model seeds 0/1/2에서 PASS, G4 toy pipeline은 PASS다. G5 primary K=100에서 Diffusion 3/18, Source Prior 7/18, Uniform Random 9/18이었다. Source Prior 대비 차이 −22.22 percentage points, paired target bootstrap 95% CI [−50.00,+5.56] percentage points로 우위를 입증하지 못했다. G5=INCONCLUSIVE에 따라 G6는 NOT RUN이다. 기준·예산을 바꾸거나 추가 seed로 유리한 결과를 탐색하지 않았다.

## 2. Repository / Environment

Python 3.12.4, PyTorch 2.14.0, CPU, experiment threads=1. Dependencies were already installed; no dependency added. Initial Git worktree was clean. Existing implementation was reused for codecs, sampler, model families, dataset splitting, verification, paired statistics and predictor training. New modules: experiment_state.py, conditional_dependence.py, information_ladder.py, toy_hash.py, candidate_budget.py.
구현 milestone commits: dcbca1b (G2 recovery/intervention), 47848ce (G2 completion/G3 freeze), acb3b95 (G3 completion/G4 freeze). Final checkpoint commit is recorded in Git history. Full environment is saved in output/session_checkpoints/environment.json. Git sandbox write failure was resolved by authorized escalation; no experiment command failed.

## 3. G0/G1 prerequisite verification

기존 G0 PASS, G1-A/B/C PASS를 실제 checkpoint/metrics/split과 대조했다. 83개 G0/G1 파일의 시작 SHA256 manifest를 만들고 실행 중·종료 시 검증했다. 기존 G0/G1과 RESEARCH_PLAN.md는 변경하지 않았다. 시작 regression 48 passed, independent encoding 2 passed. 최종 regression **65 passed**, independent encoding **2 passed**.
G2 seed0의 correct train/validation/test metrics는 모든 representation에서 원래 G1 metrics와 절대 오차 1e-7 이내로 일치한다. output/g2/seed0_validation.json 참조. 연구계획의 구 Gate 번호와 이번 요청 번호는 달라 이번 작업은 요청한 G2 conditional dependence → G6 full experiment 번호를 사용했다.

## 4. G2 Conditional Dependence

PASS. Direct Bits/BGV/CGGE × model seeds0/1/2, each 64 train +16 validation +16 unseen test messages; sampling seeds0/1/2. Fixed correct-trained checkpoint에서 correct/deranged/zero 조건만 교체했다. 모든 pair의 실제 첫 model-input x_T가 같음을 assert하고 SHA256을 저장했다. Derangement는 split 내 fixed-point-free random cycle이며 mapping을 보존했다.
모든 9개 조합, 모든 split에서 correct exact=1.0, deranged original exact=0.0, zero exact=0.0, deranged donor exact=1.0. Same dataset is reused across model seeds; dataset/split/model/batch/timestep/noise/condition/sampling seeds are recorded. Legacy seed0 shares training RNG; new seeds separate streams. No negative-condition model retraining.
Exact/valid/byte/bit-or-pixel accuracy/normalized MSE, output bit/byte/decoded disagreement, donor/original comparisons are in output/g2/<representation>/seed-<seed>/. Main report: output/g2/report.md. Sampling repetitions are not 48 independent targets; held-out target count is16 per representation/model seed.

## 5. G3 Information Ladder

PASS under the frozen diagnostic criterion. Same 9 G2 checkpoints, same16 unseen messages, 3 sampling seeds. Whole payload-cell masks expose 31/23/15/7/3/0 cells from a fixed seeded permutation. Nominal 100/75/50/25/12.5/0% levels therefore have effective fractions 100/74.19/48.39/22.58/9.68/0%. Mask selection is independent of the message. All partial levels remove header/validity metadata; L0 is all-zero for every target.
각 조합은 full exact≥0.99, nominal50% payload bit accuracy >L0, payload MSE <L0, output change >0을 충족했다. Direct Bits seed0: L0 payload accuracy0.512232/MSE1.916552 → L50 accuracy0.733370/MSE1.047723. Partial exact/valid rates are retained even when zero. No repair using true target length is performed.
Accuracy includes revealed content. 따라서 hidden-content prediction beyond prior를 입증하지 않으며, partial reconstruction improvement까지만 지지한다. 전체 ladder/seed 표와 valid candidate diversity는 output/g3/report.md와 per-run metrics에 있다.

## 6. G4 Toy Hash

PASS. MD5 first8 bits, alphabet ABCD, fixed length4, exhaustive256 messages. There are162 prefix groups, including71 collision groups. Independent hashlib/binary-string prefix oracle agreed with the production integer-shift verifier for **41,472** candidate-target comparisons. Prefix-order tests cover every q from1 to128 for MD5 and1 to256 for SHA-256, including byte boundaries.
Digest-group split:204 train messages/129 prefixes,27 validation/15 prefixes,25 test/18 prefixes. Raw-message and prefix overlaps are zero. Dense existing hash BitDenoiser (configured width32, internal128), 1500 Adam updates, batch16, lr0.001, x0 prediction,50 diffusion/sampling steps, beta_end0.4, model_seed0. First three validation targets ×10 candidates validated implementation without adaptation; full test consumes18×100=1800 attempts.
K100: codec-valid100%, in-domain valid65.3333%, unique valid candidates27.7222%, exact original recovery1/18, hash-prefix success3/18,13 successful candidate attempts. Every generated in-domain match equals oracle membership. Generation receives digest bits only. All out-of-domain candidates still consume budget. This is a toy MD5 prefix result, not full digest inversion.

## 7. G5 Candidate Budget / Baselines

INCONCLUSIVE. Same18 targets and saved G4 candidates; K=[1,10,100]. Baselines receive no target condition: uniform enumeration draw and iid four-position ABCD source-prior draw. These baselines have identical distributions and different pre-frozen RNG streams. Direct Predictor is included only at K1. Exhaustive oracle K256 coverage=1 is a separate reference.

| Method | K | Exact@K | HashMatch@K | Valid ratio | Unique ratio | Duplicate ratio | First match median (solved only) |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion | 1 | 0.000000 | 0.000000 | 0.777778 | 0.777778 | 0.000000 | None |
| diffusion | 10 | 0.055556 | 0.055556 | 0.694444 | 0.616667 | 0.083333 | 10 |
| diffusion | 100 | 0.055556 | 0.166667 | 0.653333 | 0.277222 | 0.461667 | 36 |
| direct_predictor_K1 | 1 | 0.000000 | 0.000000 | 0.222222 | 0.222222 | 0.000000 | None |
| source_prior | 1 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | None |
| source_prior | 10 | 0.055556 | 0.055556 | 1.000000 | 0.983333 | 0.016667 | 4 |
| source_prior | 100 | 0.333333 | 0.388889 | 1.000000 | 0.844444 | 0.155556 | 65 |
| uniform_random | 1 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | None |
| uniform_random | 10 | 0.000000 | 0.000000 | 1.000000 | 0.988889 | 0.011111 | None |
| uniform_random | 100 | 0.500000 | 0.500000 | 1.000000 | 0.832778 | 0.167222 | 56 |

Invalid, malformed and duplicate candidates each consume one attempt. No retry-to-valid. Unique ratio counts unique valid candidates/K; duplicate ratio counts repeated candidate values including None. Zero observed success is not zero true success probability: with18 targets the rule-of-three upper95 bound is0.1667; bootstrap [0,0] for all-zero paired outcomes is a degenerate empirical interval, not proof of equivalence.

## 8. G6 Full Experiment

**NOT RUN**. G5 did not PASS. No G6 configuration or training was executed. G6 execution_status=NOT_RUN, scientific_result=NOT_RUN (the requested VALID/INVALID and POSITIVE/NULL/NEGATIVE/INCONCLUSIVE execution axes apply only once G6 runs). Overall research_answer=INCONCLUSIVE. Main Printable94/Random Bytes distributions, full MD5 and full SHA-256 experiments remain untested.

## 9. Statistical Analysis

Target is the unit:18 binary outcomes per method/K. 10,000 paired target bootstrap replicates, bootstrap_seed4300, percentile95% CI. Exact one-sided McNemar tests superiority; Holm correction includes all7 method/K comparisons. G5 uses only model_seed0; no candidate or seed pooling. G2/G3 seeds are directional diagnostics, not a population significance test.

| Comparison | Delta HashMatch | 95% paired CI | n10/n01 | Raw p | Holm p |
|---|---:|---|---|---:|---:|
| direct_predictor_K1@1 | 0.000000 | [0.0, 0.0] | 0/0 | 1.000000 | 1.000000 |
| source_prior@1 | 0.000000 | [0.0, 0.0] | 0/0 | 1.000000 | 1.000000 |
| source_prior@10 | 0.000000 | [-0.16666666666666666, 0.16666666666666666] | 1/1 | 0.750000 | 1.000000 |
| source_prior@100 | -0.222222 | [-0.5, 0.05555555555555555] | 2/6 | 0.964844 | 1.000000 |
| uniform_random@1 | 0.000000 | [0.0, 0.0] | 0/0 | 1.000000 | 1.000000 |
| uniform_random@10 | 0.055556 | [0.0, 0.16666666666666666] | 1/0 | 0.500000 | 1.000000 |
| uniform_random@100 | -0.333333 | [-0.6666666666666666, 0.0] | 3/9 | 0.980713 | 1.000000 |

Primary source-prior@100:3 vs7 successes, −4 solved targets, relative gain0.42857, Δ−0.22222, CI[−0.5,0.05556], raw p0.96484, Holm p1.0. Uniform@100:3 vs9, Δ−0.33333, CI[−0.66667,0]. Neither comparison supports superiority. G5 primary interval crosses zero, so the frozen decision is INCONCLUSIVE rather than a definitive NO.

## 10. Leakage Audit

G2/G3 train/validation/test messages are disjoint; the reversible condition intentionally reveals full or masked record content. G3 partial masks remove separate headers, validity masks and reserved metadata. Revealed payload/padding may imply length constraints as part of the explicitly visible cells; no additional length, index or validity side channel reaches the model. Evaluation-only payload masks use the true target solely to score outputs. G3 cannot demonstrate unknown-bit inference.
G4/G5 raw messages and q8 digest groups are disjoint across splits. Model generation only receives q8 bits, Gaussian noise and timestep. Independent oracle and target messages are restricted to post-generation verification. Baseline generation APIs have no target argument. G1 source splitting is preserved, including its sorted-corpus order; G2/G3 are control diagnostics on that small corpus.
G4 diffusion outcomes were necessarily known before the G5 baseline freeze. This is an explicitly staged exploratory toy screen, not independent confirmatory preregistration. G3 cell rounding and all gate thresholds are recorded before their evaluation; none was changed to improve results.

## 11. Failure Boundary

**G5 INCONCLUSIVE**. Operational execution and integrity checks passed; the blocked boundary is statistical efficacy. Diffusion success is lower in point estimate than both random baselines at K100, while primary CI includes zero. G6 was automatically withheld. No changes to q/domain/budget/thresholds or additional exploratory reruns were made to rescue the result.

## 12. Supported Conclusions

- The tested reversible models use their conditions and follow deranged donors.
- Exposed partial record information improves payload reconstruction relative to L0 under the G3 diagnostic definition.
- The exhaustive toy hash verifier/generator pipeline is consistent and can produce valid8-bit prefix preimages.
- This fixed G5 screen does not establish candidate-generation advantage over source-prior/random baselines.

## 13. Unsupported Conclusions

No general hash inversion advantage; no hidden-content prediction advantage from G3; no full MD5/SHA preimage attack; no G6 result; no multi-seed hash efficacy claim; no claim that zero observed success implies zero success probability; no conclusion that all architectures/distributions must fail.

## 14. Reproduction Commands

Completed experiments are not rerun during continuation. Current exact next action is read-only verification:

```bash
.venv/bin/python -m diffusion_hash_inv.experiment_state --verify
```

Historical commands (do not execute past the saved fail-stop):

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_encoding_independent.py
.venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation bits --model-seed 0
.venv/bin/python -m diffusion_hash_inv.conditional_dependence --all
.venv/bin/python -m diffusion_hash_inv.information_ladder --representation bits --model-seed 0
.venv/bin/python -m diffusion_hash_inv.information_ladder --all
.venv/bin/python -m diffusion_hash_inv.toy_hash
.venv/bin/python -m diffusion_hash_inv.candidate_budget
```

## 15. Artifact Paths

- EXPERIMENT_STATE.json / NEXT_ACTION.md / EXPERIMENT_PROGRESS.md / EXPERIMENT_DECISIONS.md
- output/gate_summary.json
- output/g2/report.md and <representation>/seed-0,1,2/
- output/g3/report.md, config_frozen.json and <representation>/seed-0,1,2/
- output/g4/report.md, config_frozen.json, oracle/, experiments/seed-0/
- output/g5/report.md, config_frozen.json, diffusion/, uniform_random/, source_prior/, direct_predictor_K1/, statistics/
- output/session_checkpoints/: stage snapshots, logs, environment, prerequisite and final SHA256 manifests

## 16. Continuation State

Current gate=G5; status=STOPPED; gate result=INCONCLUSIVE; G6=NOT_RUN. A new session must read the four state documents, verify actual artifacts, then retain the fail-stop. There is no pending training/evaluation action under the current protocol. The single next action is `.venv/bin/python -m diffusion_hash_inv.experiment_state --verify` followed by review of this result; any new scientific experiment requires a separately specified protocol and cannot relabel the completed gates.
Atomic per-unit JSON writes, complete-artifact hashes, training optimizer/RNG checkpoints, and a single-writer lock support recovery. Missing derived products can be rebuilt from completed units; hash mismatches are reported for inspection instead of silently reused. G0/G1 experiments are never rerun by the new commands. All final artifacts remain on disk in this repository, including larger ignored training/candidate files. Key configs/reports/manifests and code are committed.
