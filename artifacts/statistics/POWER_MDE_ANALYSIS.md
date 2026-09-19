# Power / MDE analysis status

**BLOCKED_SPECIFICATION for confirmatory power.** Main/pilot manifests and registered baseline expectation/discordance assumptions, Holm family and power target are absent. Actual scientific N and powered MDE cannot be computed honestly. The table below is from completed 60-message engineering sanity datasets, not the pilot or main datasets.

| Algorithm | Source | q | Unique train | Validation | Test | MC expected Success@100 | Optimistic detection floor |
|---|---|---:|---:|---:|---:|---:|---|
| md5 | printable | 8 | 32 | 8 | 9 | 0.389761 | 5/9 = 0.556 (one unadjusted hypothesis, best case only) |
| md5 | printable | 12 | 39 | 10 | 10 | 0.019042 | 5/10 = 0.500 (one unadjusted hypothesis, best case only) |
| md5 | printable | 16 | 40 | 10 | 10 | 0.000000 | 5/10 = 0.500 (one unadjusted hypothesis, best case only) |
| sha256 | printable | 8 | 37 | 9 | 10 | 0.341695 | 5/10 = 0.500 (one unadjusted hypothesis, best case only) |
| sha256 | printable | 12 | 40 | 10 | 10 | 0.019042 | 5/10 = 0.500 (one unadjusted hypothesis, best case only) |
| sha256 | printable | 16 | 40 | 10 | 10 | 0.000000 | 5/10 = 0.500 (one unadjusted hypothesis, best case only) |
| md5 | random_bytes | 8 | 37 | 6 | 9 | 0.349950 | 5/9 = 0.556 (one unadjusted hypothesis, best case only) |
| md5 | random_bytes | 12 | 40 | 10 | 10 | 0.009521 | 5/10 = 0.500 (one unadjusted hypothesis, best case only) |
| md5 | random_bytes | 16 | 40 | 10 | 10 | 0.000000 | 5/10 = 0.500 (one unadjusted hypothesis, best case only) |
| sha256 | random_bytes | 8 | 38 | 9 | 9 | 0.305628 | 5/9 = 0.556 (one unadjusted hypothesis, best case only) |
| sha256 | random_bytes | 12 | 40 | 10 | 10 | 0.019042 | 5/10 = 0.500 (one unadjusted hypothesis, best case only) |
| sha256 | random_bytes | 16 | 40 | 10 | 10 | 0.000000 | 5/10 = 0.500 (one unadjusted hypothesis, best case only) |

The floor assumes every discordance favors the model and no baseline-only wins: five model-only wins give exact one-sided p=1/32<0.05. It is NOT a power-based MDE, does NOT include Holm, and does NOT establish the paired-CI criterion. With m Holm hypotheses, replace five by the smallest d with 2^-d < 0.05/m; required effect floor is d/N. Actual power can be much lower and requires preregistered joint-outcome assumptions.

MC uses the actual uniform length/source law, 1,000 iid candidates per target, seed 8; per-target Wilson intervals and transformed K intervals are saved in each source_prior_mc.json. Zero MC hits are resolution-limited and never treated as true zero probability. No 2^-q substitution is used.

q=8 has at most 256 total digest conditions, with some assigned to train/validation. Increasing message count does not create more than 256 independent target conditions. q=12,16 also require their actual unique N. All planned q remain registered in the coverage matrix.

| Planned algorithm | q | Scientific unique counts / baseline / MDE |
|---|---:|---|
| md5 | 8 | NOT AVAILABLE — dataset/specification not frozen |
| md5 | 12 | NOT AVAILABLE — dataset/specification not frozen |
| md5 | 16 | NOT AVAILABLE — dataset/specification not frozen |
| md5 | 20 | NOT AVAILABLE — dataset/specification not frozen |
| md5 | 24 | NOT AVAILABLE — dataset/specification not frozen |
| md5 | 32 | NOT AVAILABLE — dataset/specification not frozen |
| md5 | 64 | NOT AVAILABLE — dataset/specification not frozen |
| md5 | 128 | NOT AVAILABLE — dataset/specification not frozen |
| sha256 | 8 | NOT AVAILABLE — dataset/specification not frozen |
| sha256 | 12 | NOT AVAILABLE — dataset/specification not frozen |
| sha256 | 16 | NOT AVAILABLE — dataset/specification not frozen |
| sha256 | 20 | NOT AVAILABLE — dataset/specification not frozen |
| sha256 | 24 | NOT AVAILABLE — dataset/specification not frozen |
| sha256 | 32 | NOT AVAILABLE — dataset/specification not frozen |
| sha256 | 64 | NOT AVAILABLE — dataset/specification not frozen |
| sha256 | 128 | NOT AVAILABLE — dataset/specification not frozen |
| sha256 | 256 | NOT AVAILABLE — dataset/specification not frozen |
