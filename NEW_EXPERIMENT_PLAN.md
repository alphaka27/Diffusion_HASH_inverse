# New experiment

Status: planning.

1. Verify codec round-trip and valid decoding before hash-conditioned generation.
2. Verify a simple conditional positive control before a hash pilot.
3. Run a fixed-budget q=8 pilot only if both gates pass.

Store every run under `local_experiment_archive/runs/<run-id>/`. Commit only code, tests, frozen configuration, and a concise plan; do not commit raw candidates, generated arrays, databases, checkpoints, or result reports.
