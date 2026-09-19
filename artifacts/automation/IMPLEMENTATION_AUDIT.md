# Implementation audit — initial source inspection

Scope: new Gaussian/Discrete plan. READY requires an executed check; source existence alone is not readiness. Historical outputs are preserved, not current-study evidence.

| Component | Initial status | Evidence / gap |
|---|---|---|
| Research plans | READY | Both read; primary plan SHA stored in state; original unchanged |
| Configs/specification | PARTIAL | Legacy max_length=31 registered; new protocol has explicit TBDs |
| Dataset generator | PARTIAL | SourceSpec uniform length/iid alphabet; no exact quota reserve construction or construction limit |
| Digest split | PARTIAL | Whole groups; current default 0.8/0.1/0.1 does not meet 10000/1000/1000 quota |
| BGV | PARTIAL | Fixed-capacity grid, no non-finite or padding-glyph rejection |
| CGGE/glyph table | PARTIAL | Fixed public-domain table/checksum; grid and reserve assume exact capacity |
| Discrete tokenizer | MISSING | No categorical sequence codec in baseline repository |
| Gaussian | PARTIAL | Pixel U-Net and Gaussian diffusion; legacy sampler is DDIM-style; selected schedule still TBD |
| Discrete diffusion | MISSING | No masked categorical forward/reverse implementation |
| Conditioning | PARTIAL | Image uses caption, bits use vector; primary information path not shared |
| Training/sampling CLI | PARTIAL | runner.py lacks new model family and optimizer/RNG resume/checkpoint selection |
| Candidate evaluator/verifier | PARTIAL | hashlib rehash exists; domain validity not enforced for primary; length=31 hardcoded; repeated verifier calls undercounted |
| Baselines | PARTIAL | Prior/direct/nearest exist; MC expectation and shared new protocol absent |
| Controls | PARTIAL | Actual-model controls exist, but codec-only controls also exposed; same-length shuffle incomplete |
| G0 | PARTIAL | Pairwise overlap check; new frozen-config/checksum gate missing |
| G2 | PARTIAL | Aggregate budget check; strict ordering/version/per-target ledger checks missing |
| G3/G4 | PARTIAL | One-sided McNemar/bootstrap/Holm exist; full family aggregation and secondary two-sided path missing |
| State/resume | PARTIAL | Old G2–G6 workflow only; new namespace required |
| Prior checkpoints/outputs | READY (integrity only) | 36 checkpoints; 628 artifact hashes and 83 prerequisite files verified; no new efficacy evidence |
| Reports/manifests | PARTIAL | Legacy reports exist; new artifacts being created |
| Tests | UNVERIFIED | Baseline full suite running; results will be recorded separately |

## Specification decision

Use existing L_max=31 only because current user explicitly permits an existing registered value; evidence is RESEARCH_PLAN.md and preserved dataset manifest. Do not invent a different maximum. The new confirmatory run remains BLOCKED_SPECIFICATION: new uncontaminated dataset seed selection, discrete architecture/training/masking/selection configuration, positive-control target count, statistical family membership/seed values, and compute/stopping envelope are unregistered. Implement and execute independent deterministic engineering checks meanwhile. Unit-test parameters are not study hyperparameters.

## Post-implementation audit (2026-09-19T08:36:44.443303+00:00)

Status READY below is component readiness for the executed engineering tests, not confirmatory run readiness.

| Component | Current status | Executed evidence / remaining gap |
|---|---|---|
| BGV generalized codec | READY | Maxima 4/7/8/31/32/255, canonical/strict masks/padding/nonfinite; source domain enforced by evaluator |
| CGGE generalized codec/glyphs | READY | Same generalized shapes; 94 unique fixed glyphs/checksum and non-finite/reserve tests |
| Tokenizer | READY | Printable97/Bytes259, every symbol, EOS/PAD/MASK invalidity, 0x00 distinct from PAD |
| Masked diffusion primitives | READY | EOS/PAD corruption, masked CE gradients, oracle algorithm test and saved one-step actual-model smoke for both sources/types |
| Gaussian canonical input path | READY (engineering) | Four Printable/Random Bytes image/type combinations plus CGGE exercised in the 10-family smoke matrix |
| Dataset/source-prior/split | READY (engineering) | Deterministic exact quota construction with immutable group owner, bounded draws, 12 actual manifests and overlap checks |
| Canonical hash conditioning | READY (engineering) | Equal digest bits across representations, hidden suffix/length invariance tests, known-length field |
| Independent evaluator | READY (engineering) | Source domain in primary validity; invalid/duplicate budget; one hash pass per write; all available bytes rehashed; exact zero-success bound |
| Source-prior expectation | READY (engineering) | Actual length/source Monte Carlo with intervals, independent realized K outcomes |
| Direct predictor | PARTIAL | Existing non-diffusion predictor reused; registered shared codec/model still TBD; categorical predictor not implemented |
| Controls | PARTIAL | Actual-model zero/shuffle/length-only condition paths; same-length derangement rejects impossible strata; new learned controls not run |
| G0/G2 | READY (engineering) | Fail-closed checksum/manifest/split and per-target budget/ordering/version tests; scientific integration remains pending |
| G3/G4 primitives | READY (engineering) | Explicit complete family membership, two-sided secondary direction, seed-specific Holm/bootstrap; no scientific outcome data |
| Training resume | PARTIAL | Immutable config/code fingerprint, output lock, model/optimizer/RNG recovery, completed artifact checksum skip; validation selection and per-attempt interrupted sampling resume not complete |
| Confirmatory orchestrator | PARTIAL | 510 trainings/1,530 metric rows enumerated; runner is explicitly engineering-only until mandatory G0/G1/external dataset/validation integration |
| Secondary diagnostics | PARTIAL | Common primary/length/source/counts/timing/parameter data available; all CER/BER/token/glyph/compute reports not yet integrated |
| Old checkpoints/output | READY (integrity only) | Rechecked 628 hashes + 83 prerequisite files; unchanged; not new-study evidence |
| New full test suite | READY | FINAL_TESTS.log: 88 passed; baseline suite was 65 passed |

Current code source hash: `84132ed8dd074e51d7bea8502a9fd5169f8aa22cd865d0476d517eb5c057584a`. No research plan was relaxed or modified. Engineering fixture settings were frozen independently and are not recommended scientific hyperparameters. Confirmatory architecture/schedule/selection changes must be selected from train/validation under a registered protocol before test access.
