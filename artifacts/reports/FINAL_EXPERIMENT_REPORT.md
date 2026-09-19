# Final Experiment Report — current session

## 1. Executive Summary

The new scientific question remains **UNANSWERED / BLOCKED_SPECIFICATION**. No new confirmatory pilot/main/full-digest experiment has run. Implemented and verified generalized image codecs, categorical tokenizer/masked diffusion, canonical conditioning, evaluator corrections, exact-quota dataset construction, gate/statistics/state components. Full regression: **88 passed** (baseline 65).

Completed 12 small engineering datasets, 2,540,548 exhaustive verifier verdict checks and 10 one-step actual-model engineering smoke jobs. All 20 smoke attempts were invalid; this verifies execution/accounting, not learned reversible control or scientific performance. No threshold or model was tuned in response. Do not infer success probability or convergence from these fixtures.

## 2. Experiment Coverage

- Completed scientific training jobs: **0/510**; scientific K evaluation rows: **0/1530**.
- Completed engineering smoke: 10 families, seed 0 only, one training update per family; saved checkpoints and candidate ledgers.
- Failed scientific jobs: none executed. Blocked: all scientific jobs (specification/G1 prerequisites). Resource planning additionally pending.
- Full MD5 q=128 and full SHA-256 q=256 remain present for all families/seeds/K in [coverage](../automation/CONFIRMATORY_COVERAGE.csv).
- Immutable engineering checks: [preflight](../automation/PREFLIGHT_RESULTS_84132ed8dd07.json), [smoke index](../automation/MATRIX_SMOKE_84132ed8dd07.json).

| Family | Train steps | Parameters | Targets | Attempts | Valid decode | Scope |
|---|---:|---:|---:|---:|---:|---|
| G-P-BGV | 1 | 13746 | 2 | 2 | 0.000 | engineering only |
| G-P-CG | 1 | 13746 | 2 | 2 | 0.000 | engineering only |
| G-R-BGV | 1 | 13746 | 2 | 2 | 0.000 | engineering only |
| D-P | 1 | 9322 | 2 | 2 | 0.000 | engineering only |
| D-R | 1 | 21310 | 2 | 2 | 0.000 | engineering only |
| G-P-BGV-L | 1 | 13770 | 2 | 2 | 0.000 | engineering only |
| G-P-CG-L | 1 | 13770 | 2 | 2 | 0.000 | engineering only |
| G-R-BGV-L | 1 | 13770 | 2 | 2 | 0.000 | engineering only |
| D-P-L | 1 | 9330 | 2 | 2 | 0.000 | engineering only |
| D-R-L | 1 | 21318 | 2 | 2 | 0.000 | engineering only |

The smoke maximum length is 7 and seed=7, matching bounded test fixtures. It is explicitly outside the scientific plan; inherited scientific L_max=31 has provenance in the old plan/config and is not inferred from fixture performance.

## 3. Dataset and Splits

Scientific pilot/main datasets were not generated because fresh uncontaminated dataset/split specifications are unregistered. Each completed sanity dataset has exact message quotas 40/10/10. Actual unique digest counts:

| Algorithm | Source | q | Train conditions | Validation | Test |
|---|---|---:|---:|---:|---:|
| md5 | printable | 8 | 32 | 8 | 9 |
| md5 | printable | 12 | 39 | 10 | 10 |
| md5 | printable | 16 | 40 | 10 | 10 |
| sha256 | printable | 8 | 37 | 9 | 10 |
| sha256 | printable | 12 | 40 | 10 | 10 |
| sha256 | printable | 16 | 40 | 10 | 10 |
| md5 | random_bytes | 8 | 37 | 6 | 9 |
| md5 | random_bytes | 12 | 40 | 10 | 10 |
| md5 | random_bytes | 16 | 40 | 10 | 10 |
| sha256 | random_bytes | 8 | 38 | 9 | 9 |
| sha256 | random_bytes | 12 | 40 | 10 | 10 |
| sha256 | random_bytes | 16 | 40 | 10 | 10 |

Manifests record raw record checksum, source/length, seed, construction budget/unused draws, digest ownership, frozen configuration and code version. No split leakage found. Seeded representatives and ordered K prefixes are saved. These small datasets are **not** the 10,000/1,000/1,000 pilot. Do not use previously observed engineering or legacy sources as new confirmatory test data.

## 4. G0 Results

12 structural engineering G0 checks PASS. Scientific G0: BLOCKED_SPECIFICATION, not PASS. [G0 summary](../gates/G0_SUMMARY.md); per-dataset `G0_SANITY_*.json` and split_validation.md exist.

## 5. G1 Results

Codec round-trips: 9,636/9,636 across family-labeled checks plus 94/94 individual glyph tests. Known-length codec checks repeat the corresponding codec, not an independent learned model verification. BGV header/padding and CGGE reserve/extra cells tested at multiple maxima. Token vocabularies 97/259 and 0x00/PAD separation verified. Exhaustive verifier: Printable 8,930 and Random Bytes 65,792 one-/two-byte messages per algorithm, all planned q, matching/nonmatching comparisons; all PASS.

**Learned reversible positive controls: NOT RUN. G1 overall PARTIAL.** Oracle sampler checks and one-update smoke are not ExactRecovery ≥0.99 controls. Old positive-control artifacts do not transfer to new representations/configurations.

## 6. Candidate Budget / G2

Invalid attempts consume K; domain-invalid decoded bytes still rehashed, undecodable outputs have null digest. Budget mismatch/duplicate target/order/version failures are tested. One evaluation pass creates summary, outcome and candidate ledger, preventing unreported duplicate verifier calls inside write_evaluation. Per-target source-prior streams preserve K prefixes; actual sanity K=1/10/100 curves are monotone. G2 scientific comparison NOT RUN. Self-comparison fixtures only validate G2 implementation.

## 7. MD5 Results

No scientific results; engineering source-prior outcomes are available only in per-dataset evaluation artifacts. No hash-conditioned advantage is estimated.

| Algorithm | q | K | Scientific status / result |
|---|---:|---|---|
| md5 | 8 | 1,10,100 | NOT RUN / N/A |
| md5 | 12 | 1,10,100 | NOT RUN / N/A |
| md5 | 16 | 1,10,100 | NOT RUN / N/A |
| md5 | 20 | 1,10,100 | NOT RUN / N/A |
| md5 | 24 | 1,10,100 | NOT RUN / N/A |
| md5 | 32 | 1,10,100 | NOT RUN / N/A |
| md5 | 64 | 1,10,100 | NOT RUN / N/A |
| md5 | 128 | 1,10,100 | NOT RUN / N/A |
| sha256 | 8 | 1,10,100 | NOT RUN / N/A |
| sha256 | 12 | 1,10,100 | NOT RUN / N/A |
| sha256 | 16 | 1,10,100 | NOT RUN / N/A |
| sha256 | 20 | 1,10,100 | NOT RUN / N/A |
| sha256 | 24 | 1,10,100 | NOT RUN / N/A |
| sha256 | 32 | 1,10,100 | NOT RUN / N/A |
| sha256 | 64 | 1,10,100 | NOT RUN / N/A |
| sha256 | 128 | 1,10,100 | NOT RUN / N/A |
| sha256 | 256 | 1,10,100 | NOT RUN / N/A |

## 8. SHA-256 Results

Scientific replication NOT RUN. The SHA-256 sanity checks above are G1 reference-verifier/baseline engineering checks, not execution of Phase 5. Full q=256 remains pending regardless of pilot performance.

## 9. Gaussian Results

BGV and CGGE canonical-bit training/sampling paths executed for all applicable one-step fixtures. See smoke table for actual parameter counts and artifacts. No trained reversible control, validation-selected checkpoint or scientific advantage result exists.

## 10. Discrete Results

Masked corruption covers payload/EOS/PAD. Categorical CE and iterative unmasking execute; neither true length nor target EOS/PAD position is used for repair. Both source alphabets and condition types ran the one-step smoke. Saved finite losses/checkpoints establish engineering execution only. Scientific architecture/embedding/schedule/temperature selection remains unregistered.

## 11. Gaussian vs Discrete

No inferential comparison. Shared canonical digest information is unit tested; fixture differences are not effects. Any eventual result is an end-to-end Gaussian-image versus Discrete-sequence approach comparison.

## 12. Known-length Effects

Known-length conditioning paths run in fixtures and preserve raw generated output. No EOS/mask/PAD/truncation repair performed. No scientific hash-only versus known-length effect is available.

## 13. Negative Controls

Zero, shuffled and length-only input paths exist; same-length hash derangement is tested and rejects impossible strata. Actual learned negative-control studies and coverage-aware singleton reporting remain pending. Codec-only legacy controls are not accepted as learned pipeline controls.

## 14. Statistical Validation

Exact one-sided McNemar for baseline superiority, exact two-sided for secondary comparisons, target-pair bootstrap (10,000), Holm over explicit registered family, seed-specific output implemented/tested. Bootstrap uses the multinomial counts of paired differences (-1/0/+1), mathematically equivalent to resampling target differences; it never resamples candidate attempts. RNG/quantile implementation is versioned.

Scientific [McNemar](../statistics/MCNEMAR_RESULTS.csv), [bootstrap](../statistics/BOOTSTRAP_RESULTS.csv) and [Holm](../statistics/HOLM_RESULTS.csv) files have **headers only**. Synthetic binary pipeline fixtures are under `statistics/PIPELINE_FIXTURE_84132ed8dd07` and cannot be cited as experiment results. Actual scientific N/power/MDE unavailable; see [power report](../statistics/POWER_MDE_ANALYSIS.md).

## 15. Seed Reproducibility

No scientific seeds 0/1/2 matrix trained. G4 NOT RUN. Synthetic seed aggregation checks do not confer Reproduced/Strongly Reproduced. Never pool seeds into independent population observations.

## 16. Evidence Classification

All 1,530 unexecuted scientific rows have `NOT_EVALUATED`, with separate blocked status. Assigning L0/L1 to missing experiments would fabricate observed gate failure. L0–L3 classification is deferred until a real run exists; L4 is unavailable. No validated hash inversion evidence from this session.

## 17. Failure Analysis

- Engineering regression failures: none in final 88-test suite; source bugs found by audit were fixed before scientific test use.
- Invalid generated fixtures: 20/20 attempts, preserved unchanged. One update is not a registered positive-control or efficacy experiment, so no scientific failure classification is inferred.
- Scientific blocker: fresh dataset/split and model/control/statistical/resource specifications are still null.
- Implementation blocker: full confirmatory orchestration (full G0/G1 training guard, external exact-quota data, registered validation selection, all requested secondary diagnostics, per-attempt interrupted sampling resume) is not finished. Current runner labels output engineering/legacy only.
- Resource blocker: only CPU accessible; no registered compute envelope; conservative raw-output storage bound exceeds current free disk. No q/family was dropped.
- Old study remains STOPPED at its own G5 INCONCLUSIVE/G6 NOT_RUN; 628 checksums and 83 prerequisite files verified unchanged. That result is not the new study's result.

## 18. Limitations

No powered efficacy experiment or positive control was run. Small sanity target counts and Monte Carlo resolution limit baseline interpretation. Gaussian image vs discrete sequence confounds representation and architecture. Same K is not same compute. Future claims remain restricted to source/length/q/K/configuration/sample size; truncated evidence never implies full-digest inversion.

For any future observed zero-success run, report N, exact one-sided upper95 = 1 - 0.05^(1/N), rule-of-three ≈3/N with target-sampling assumptions and: “No successful preimage was observed under the tested candidate budget and sample size.” Do not apply the binomial population interpretation to arbitrary engineering fixture results.

## 19. Reproduction Instructions

From repository root:

```bash
.venv/bin/python -m diffusion_hash_inv.automation --verify
.venv/bin/python -m diffusion_hash_inv.experiment_state --verify
.venv/bin/python -m pytest -q
.venv/bin/python -m diffusion_hash_inv.preflight --run
```

The last command verifies and skips completed preflight with unchanged source version. Each engineering training directory contains configuration_frozen.json, code hash, checkpoint.pt, training_resume.pt (model/optimizer/RNG), complete.json checksums and evaluation outputs. Same-ID incompatible configuration/code or corrupted completed artifact is rejected. No expensive completed study training exists to repeat.

[Source archive](../automation/SOURCE_SNAPSHOT_84132ed8dd07.tar.gz) and [code manifest](../automation/CODE_MANIFEST_84132ed8dd07.json) capture the current uncommitted implementation. Git base commit is `9b2b69a41cf4c5a57fea21fa58af08372133a0a3`; source hash `84132ed8dd074e51d7bea8502a9fd5169f8aa22cd865d0476d517eb5c057584a`. Do not confuse base commit with the dirty working-tree code version.

Do not launch the scientific matrix from legacy CLI. Resolve the [specification template](../config/CONFIRMATORY_SPECIFICATION_TEMPLATE.json), finish the explicitly listed integration gaps, freeze all settings before test, run independent actual-model positive controls, then run MD5 phases followed by SHA-256.

## 20. Artifact Index

- [Implementation audit](../automation/IMPLEMENTATION_AUDIT.md)
- [Current state](../automation/EXPERIMENT_STATE.json)
- [Resume instructions](../automation/NEXT_ACTIONS.md)
- [Resource plan](../automation/RESOURCE_PLAN.md)
- [Unfinished jobs](../automation/UNFINISHED_JOBS.json)
- [Scientific coverage](../automation/CONFIRMATORY_COVERAGE.csv)
- [Gate summary](../gates/GATE_SUMMARY.md)
- [Final tests](../automation/FINAL_TESTS.log)
- [Engineering preflight](../automation/PREFLIGHT_RESULTS_84132ed8dd07.json)
- [Engineering smoke index](../automation/MATRIX_SMOKE_84132ed8dd07.json)
- [Power/MDE status](../statistics/POWER_MDE_ANALYSIS.md)
- [MD5 pilot status](PILOT_REPORT.md)
- [MD5 main status](MD5_MAIN_REPORT.md)
- [SHA-256 status](SHA256_REPORT.md)
- [Legacy integrity](../automation/LEGACY_INTEGRITY.json)
- [Artifact checksum index](../automation/ARTIFACT_INDEX.json)
