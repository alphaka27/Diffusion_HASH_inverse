# G2 Conditional Dependence

| Representation | Model seed | Correct exact | Deranged exact | Zero exact | Donor exact | Status |
|---|---:|---:|---:|---:|---:|---|
| bits | 0 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | PASS |
| bgv | 0 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | PASS |
| cgge | 0 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | PASS |
| bits | 1 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | PASS |
| bgv | 1 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | PASS |
| cgge | 1 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | PASS |
| bits | 2 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | PASS |
| bgv | 2 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | PASS |
| cgge | 2 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | PASS |

Gate: PASS. Each row: 16 unseen targets × 3 paired sampling seeds; train/validation metrics saved separately.
Same correct-trained checkpoint, first model-input tensor, scheduler and sampling steps; only conditions change.
Donor mapping is a seeded random cycle without fixed points within each split. Seed0 checkpoints are copied from G1-C without training.
Model seeds 1/2 use the same G1 dataset/config with separately seeded batch/timestep/noise streams.
Output byte disagreement is decoded byte mismatch versus correct output, counting an invalid intervention output as total mismatch. Bit/pixel accuracy includes padding and masks; it is a diagnostic, not a hash efficacy metric.
Repeated sampling seeds are not independent targets. No population significance or hash-advantage claim is made.
Supported conclusion on PASS: condition affects reconstruction, including donor-directed output changes.
Reproduce/resume: `.venv/bin/python -m diffusion_hash_inv.conditional_dependence --all`.
Detailed metrics, per-target outcomes, initial-noise hashes, donor mapping, configs, checkpoints and environments are in representation/seed directories.
