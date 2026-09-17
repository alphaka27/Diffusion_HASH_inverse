# G5 Matched Candidate Budget and Baselines

Gate: INCONCLUSIVE; model seed0; 18 shared target prefixes; MD5 q8 toy only.

| Method | K | HashMatch@K | Exact@K | Valid ratio | Unique ratio | Duplicate ratio |
|---|---:|---:|---:|---:|---:|---:|
| diffusion | 1 | 0.000000 | 0.000000 | 0.777778 | 0.777778 | 0.000000 |
| uniform_random | 1 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 |
| source_prior | 1 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 |
| direct_predictor_K1 | 1 | 0.000000 | 0.000000 | 0.222222 | 0.222222 | 0.000000 |
| diffusion | 10 | 0.055556 | 0.055556 | 0.694444 | 0.616667 | 0.083333 |
| uniform_random | 10 | 0.000000 | 0.000000 | 1.000000 | 0.988889 | 0.011111 |
| source_prior | 10 | 0.055556 | 0.055556 | 1.000000 | 0.983333 | 0.016667 |
| diffusion | 100 | 0.166667 | 0.055556 | 0.653333 | 0.277222 | 0.461667 |
| uniform_random | 100 | 0.500000 | 0.500000 | 1.000000 | 0.832778 | 0.167222 |
| source_prior | 100 | 0.388889 | 0.333333 | 1.000000 | 0.844444 | 0.155556 |

| Comparison | Delta HashMatch | Paired 95% CI | McNemar p | Holm p |
|---|---:|---|---:|---:|
| uniform_random@1 | 0.000000 | (0.0, 0.0) | 1.000000 | 1.000000 |
| source_prior@1 | 0.000000 | (0.0, 0.0) | 1.000000 | 1.000000 |
| direct_predictor_K1@1 | 0.000000 | (0.0, 0.0) | 1.000000 | 1.000000 |
| uniform_random@10 | 0.055556 | (0.0, 0.16666666666666666) | 0.500000 | 1.000000 |
| source_prior@10 | 0.000000 | (-0.16666666666666666, 0.16666666666666666) | 0.750000 | 1.000000 |
| uniform_random@100 | -0.333333 | (-0.6666666666666666, 0.0) | 0.980713 | 1.000000 |
| source_prior@100 | -0.222222 | (-0.5, 0.05555555555555555) | 0.964844 | 1.000000 |

Primary K=100, primary source-prior baseline. Both source-prior and uniform comparisons must pass to advance. Exact one-sided McNemar and 10,000 paired target bootstrap replicates; Holm correction across all 7 comparisons. No candidate-level inference or seed pooling.
Baseline candidate functions receive method, target count, K and seed only. Source-prior iid ABCD^4 and uniform domain sampling have the same distribution; the two reported draws are independent RNG realizations.
All attempts consume K, including malformed/invalid and repeated candidates. Unique ratio is unique valid candidates/K; duplicate ratio counts repeated candidate values including None. First-match indices and medians among solved targets are in statistics files.
Exhaustive oracle reference reaches all targets after enumerating all256 messages; it is not a same-K efficacy baseline.
G4 model outcomes were already visible before G5 baseline freeze. This is an exploratory staged screen, not a new confirmatory preregistration. Only model seed0 was evaluated for efficacy. No positive generalization beyond this toy setting follows.
Reproduce/resume: `.venv/bin/python -m diffusion_hash_inv.candidate_budget`.
