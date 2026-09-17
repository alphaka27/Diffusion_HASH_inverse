# G4 Toy Hash Pipeline

Gate: PASS. MD5 q=8 prefix only, alphabet ABCD, length4, 256-message exhaustive domain.

| K | Targets | Exact original@K | HashPrefixMatch@K | Valid ratio | Unique ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 0.000000 | 0.000000 | 0.777778 | 0.777778 |
| 10 | 18 | 0.055556 | 0.055556 | 0.694444 | 0.616667 |
| 100 | 18 | 0.055556 | 0.166667 | 0.653333 | 0.277222 |

Independent oracle audit: {'agreement': True, 'comparison_count': 41472, 'domain_size': 256, 'prefix_groups': 162, 'collision_groups': 71}. Split audit: {'passed': True, 'algorithm_q_consistent': True, 'pairwise': {'train_validation': {'message_overlap_count': 0, 'message_examples_hex': [], 'digest_overlap_count': 0, 'digest_examples': []}, 'train_test': {'message_overlap_count': 0, 'message_examples_hex': [], 'digest_overlap_count': 0, 'digest_examples': []}, 'validation_test': {'message_overlap_count': 0, 'message_examples_hex': [], 'digest_overlap_count': 0, 'digest_examples': []}}, 'record_counts': {'train': 204, 'validation': 27, 'test': 25}, 'unique_digest_counts': {'train': 129, 'validation': 15, 'test': 18}}.
Independent reference uses hashlib plus binary-string prefix extraction; production verifier uses big-endian integer shifts. Every enumerated message is checked against every unique target prefix.
Generator inputs: digest bits, Gaussian initial noise, timestep. Oracle and representative target messages are used only after generation for evaluation. No candidate repair or extra attempts. Out-of-domain/invalid candidates and duplicates consume budget.
Full-digest equality and exact original equality are separately saved. 8-bit prefix matches do not imply full MD5 preimage capability.
Validation smoke checks integrity only. Full configuration was frozen before training/evaluation. A valid run with zero generated held-out preimages fails the required G4 functional-success criterion.
Reproduce/resume: `.venv/bin/python -m diffusion_hash_inv.toy_hash`.
