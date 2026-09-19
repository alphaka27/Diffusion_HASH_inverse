# Resource plan

Observed environment: `{"python": "3.12.4", "platform": "macOS-27.0-arm64-arm-64bit", "torch": "2.14.0", "cpu_count": 16, "cuda_available": false, "mps_available": false, "mps_built": true, "disk_free_bytes": 238387200000, "scope": "observed in this process; inaccessible GPU is not proof of absent hardware"}`.

- Core families: 10; model seeds: 0, 1, 2.
- MD5 q: [8, 12, 16, 20, 24, 32, 64, 128]; SHA-256 q: [8, 12, 16, 20, 24, 32, 64, 128, 256].
- Pilot messages: 10,000 / 1,000 / 1,000. Main: 100,000 / 10,000 / 10,000.
- Core diffusion training runs: **510 = 10 × 3 × 17**. MD5 240; SHA-256 270.
- Distinct candidate streams: 510. K metric rows: 1,530, sharing stream prefixes (not three separate trainings).
- Shared non-diffusion predictor: 2 sources × 2 condition types × 3 seeds × 17 q settings = **204** training runs if one registered predictor per source/condition is selected.
- Source-prior streams: 204 under the same sharing policy. Baseline/model sampling seed policy remains TBD.
- Two learned negative controls per family would add 1,020 trainings. Actual control protocol/coverage and positive-control target counts must be frozen; these are planning counts, not executed runs.
- Scientific datasets: 34 (2 sources × 17 algorithm/q settings), shared between representations/condition types. None generated yet.

For each setting, N ≤ min(test message quota, 2^q). The upper bound ignores groups reserved to train/validation, so it overestimates attainable test N, particularly q=8.
Candidate attempts per method/seed = 10N + 90 min(1000,N). Across core diffusion streams the conservative bound is **76,236,000** attempts. Baseline/controls/MC costs are additional.

At inherited L_max=31, raw BGV float32 output is 32,768 bytes, CGGE 16,384 bytes, and categorical uint16 sequence 64 bytes. Storing every raw output without compression is bounded here by approximately **1.251 TB** before metadata, checkpoints, baselines and controls. Current free disk is approximately 238.4 GB. JSON pixel serialization is larger. This is a capacity warning, not a prediction of actual unique N or compressibility; compression/replay retention and storage budget must be fixed before runs.

Training updates, architecture widths, sampling steps, compute envelope and wall-clock ceilings remain TBD. A measured one-step fixture cannot estimate full training convergence/runtime. GPU availability in this process: CUDA=false, MPS=false. No claim that the machine physically lacks a GPU.

Primary comparison matches candidate count, not FLOPs or wall-clock. Report NFE, model parameter count, training/evaluation wall-clock, memory, verification calls and storage separately. Every full MD5 q=128 and SHA-256 q=256 job remains in UNFINISHED_JOBS.json; no scientific condition was removed due to runtime.
