"""Build truthful coverage reports from saved engineering results; performs no training."""
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path

import torch
from diffusion_hash_inv.automation import source_version,freeze_json,complete_job,verify_completed,STATE_PATH
from diffusion_hash_inv.experiment_state import write_json,sha256,verify_persisted_state
from diffusion_hash_inv.preflight import QS,FAMILIES
from diffusion_hash_inv.study_statistics import write_statistics

root=Path('artifacts');version=source_version();tag=version[:12]
summary=json.loads((root/'automation'/f'PREFLIGHT_RESULTS_{tag}.json').read_text())
smoke=json.loads((root/'automation'/f'MATRIX_SMOKE_{tag}.json').read_text())
state=json.loads(STATE_PATH.read_text());verify_completed(state)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
environment=dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,
                 cpu_count=os.cpu_count(),cuda_available=torch.cuda.is_available(),mps_available=torch.backends.mps.is_available(),
                 mps_built=torch.backends.mps.is_built(),disk_free_bytes=shutil.disk_usage('.').free,
                 scope='observed in this process; inaccessible GPU is not proof of absent hardware')
write_json(root/'automation/ENVIRONMENT.json',environment)
legacy=verify_persisted_state();write_json(root/'automation/LEGACY_INTEGRITY.json',legacy)
files=sorted([*Path('src').rglob('*.py'),*Path('tests').rglob('*.py'),Path('pyproject.toml'),Path('.python-version'),
              Path('RESEARCH_PLAN.md'),Path('RESEARCH_PLAN_GAUSSIAN_DISCRETE.md')])
code_manifest={str(p):sha256(p) for p in files}
freeze_json(root/'automation'/f'CODE_MANIFEST_{tag}.json',code_manifest)
archive=root/'automation'/f'SOURCE_SNAPSHOT_{tag}.tar.gz'
if not archive.exists():
 with tarfile.open(archive,'w:gz') as tar:
  for path in files:tar.add(path,arcname=str(path))
checkpoints=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in Path('output').rglob('*.pt')]
freeze_json(root/'automation/LEGACY_CHECKPOINT_INVENTORY.json',checkpoints)

template=dict(status='BLOCKED_SPECIFICATION_NOT_A_RUN_CONFIG',plan_sha256=state['plan_version'],
 l_max=31,l_max_provenance=state['l_max_provenance'],l_min=4,length_distribution='uniform_integer',
 sources=['printable','random_bytes'],model_seeds=[0,1,2],dataset_seed=None,split_seed=None,
 excluded_observed_corpora=['all output/ legacy corpora','artifacts/datasets/SANITY_*','all engineering smoke/test fixture corpora'],
 q_values=QS,k=[1,10,100],pilot_quotas=[10000,1000,1000],main_quotas=[100000,10000,10000],
 condition_format='canonical_bits',candidate_subset_seed=None,sampling_seeds=None,baseline_seeds=None,
 bootstrap_seed=None,monte_carlo_seed=None,monte_carlo_draws=None,power_target=None,
 family_configuration={family:None for family in FAMILIES},positive_control_protocol=None,
 direct_predictor_configuration=None,statistical_comparison_manifest=None,stopping_rule=None,
 resource_envelope=None,full_digest_reserved_budget=None,validation_selection_protocol=None)
write_json(root/'config/CONFIRMATORY_SPECIFICATION_TEMPLATE.json',template)

jobs=[];coverage=[];streams=0;candidate_upper=0;raw_bytes=0
for algorithm,qs in QS.items():
 for q in qs:
  phase='PHASE_2' if algorithm=='md5' and q<=16 else 'PHASE_4' if algorithm=='md5' and q==128 else 'PHASE_3' if algorithm=='md5' else 'PHASE_5'
  tier='pilot' if q<=16 else 'main';messages=1000 if tier=='pilot' else 10000
  n_bound=min(messages,2**q)
  attempts=10*n_bound+90*min(1000,n_bound)
  for family,(source,model,representation) in FAMILIES.items():
   condition='KNOWNLENGTH' if family.endswith('-L') else 'HASHONLY'
   for seed in (0,1,2):
    base=f'{algorithm.upper()}_{source.upper()}_{model.upper()}_{representation.upper()}_Q{q}_{condition}_SEED{seed}'
    train_id=base+'_TRAIN'
    jobs.append(dict(job_id=train_id,status='BLOCKED_SPECIFICATION',phase=phase,family=family,tier=tier,
                     algorithm=algorithm,q=q,seed=seed,depends_on=['SPECIFICATION_FREEZE','CONFIRMATORY_G0',f'G1_{family}'],
                     eventual_resource_status='PENDING_RESOURCE_UNTIL_BUDGET_REGISTERED'))
    streams+=1;candidate_upper+=attempts
    per_attempt=2*32*(128 if representation=='bgv' else 64)*4 if representation in {'bgv','cgge'} else 32*2
    raw_bytes+=attempts*per_attempt
    for k in (1,10,100):
     job_id=base+f'_K{k}'
     jobs.append(dict(job_id=job_id,status='BLOCKED_SPECIFICATION',phase=phase,family=family,tier=tier,
                      algorithm=algorithm,q=q,seed=seed,k=k,depends_on=[train_id],shared_stream=base+'_C100'))
     coverage.append(dict(algorithm=algorithm,source=source,family=family,representation=representation,
                           condition=condition,q=q,k=k,seed=seed,status='BLOCKED_SPECIFICATION',
                           target_count='',successes='',evidence_level='NOT_EVALUATED'))
state['pending_jobs']=[dict(job_id='SPECIFICATION_FREEZE',status='BLOCKED_SPECIFICATION',phase='PHASE_0')]+jobs
state['blocked_jobs']=[dict(job_id='CONFIRMATORY_SPECIFICATION',status='BLOCKED_SPECIFICATION',
 reason='Required choices remain null in CONFIRMATORY_SPECIFICATION_TEMPLATE.json. No scientific defaults inferred.',
 affected_diffusion_training_jobs=510,affected_evaluation_rows=1530),
 dict(job_id='CONFIRMATORY_PIPELINE_INTEGRATION',status='BLOCKED_IMPLEMENTATION',
 reason='Engineering runner does not yet enforce full learned G1 prerequisite, accept registered external exact-quota datasets, perform validation checkpoint selection, or aggregate all requested representation-specific diagnostics. Do not launch it as a confirmatory runner.'),
 dict(job_id='FULL_MATRIX_RESOURCE',status='PENDING_RESOURCE',
 reason='CPU-only available in this process. Training/sampling budgets not registered. Float32 canonical image storage upper estimate exceeds currently free disk; no matrix conditions removed.')]
state['current_phase']='PHASE_0_SCIENTIFIC_FREEZE_BLOCKED_PHASE_1_ENGINEERING_COMPLETE'
state['gates']=summary['scientific_gates']
state['last_successful_checkpoint']=str(Path(smoke[-1]['artifact'])/'training_resume.pt')
state['last_checkpoint_scope']='engineering fixture only; not a scientific checkpoint'
state['source_version']=version
state['code_manifest']=str(root/'automation'/f'CODE_MANIFEST_{tag}.json')
state['source_snapshot']=str(archive)
state['updated_at']=now
state['failed_jobs']=[]
state['next_job']='SPECIFICATION_FREEZE'
for path in (root/'training').glob('*/configuration_frozen.json'):
 item={'path':str(path),'sha256':sha256(path),'scope':'engineering'}
 if not any(v['path']==str(path) for v in state['config_snapshots']):state['config_snapshots'].append(item)
write_json(STATE_PATH,state)
with (root/'automation/CONFIRMATORY_COVERAGE.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
write_json(root/'automation/UNFINISHED_JOBS.json',state['pending_jobs'])
write_statistics(root/'statistics',[]) # Header-only: there are no scientific comparison results.

resource=f'''# Resource plan

Observed environment: `{json.dumps(environment,ensure_ascii=False)}`.

- Core families: 10; model seeds: 0, 1, 2.
- MD5 q: {QS['md5']}; SHA-256 q: {QS['sha256']}.
- Pilot messages: 10,000 / 1,000 / 1,000. Main: 100,000 / 10,000 / 10,000.
- Core diffusion training runs: **{streams} = 10 × 3 × 17**. MD5 240; SHA-256 270.
- Distinct candidate streams: 510. K metric rows: 1,530, sharing stream prefixes (not three separate trainings).
- Shared non-diffusion predictor: 2 sources × 2 condition types × 3 seeds × 17 q settings = **204** training runs if one registered predictor per source/condition is selected.
- Source-prior streams: 204 under the same sharing policy. Baseline/model sampling seed policy remains TBD.
- Two learned negative controls per family would add 1,020 trainings. Actual control protocol/coverage and positive-control target counts must be frozen; these are planning counts, not executed runs.
- Scientific datasets: 34 (2 sources × 17 algorithm/q settings), shared between representations/condition types. None generated yet.

For each setting, N ≤ min(test message quota, 2^q). The upper bound ignores groups reserved to train/validation, so it overestimates attainable test N, particularly q=8.
Candidate attempts per method/seed = 10N + 90 min(1000,N). Across core diffusion streams the conservative bound is **{candidate_upper:,}** attempts. Baseline/controls/MC costs are additional.

At inherited L_max=31, raw BGV float32 output is 32,768 bytes, CGGE 16,384 bytes, and categorical uint16 sequence 64 bytes. Storing every raw output without compression is bounded here by approximately **{raw_bytes/10**12:.3f} TB** before metadata, checkpoints, baselines and controls. Current free disk is approximately {environment['disk_free_bytes']/10**9:.1f} GB. JSON pixel serialization is larger. This is a capacity warning, not a prediction of actual unique N or compressibility; compression/replay retention and storage budget must be fixed before runs.

Training updates, architecture widths, sampling steps, compute envelope and wall-clock ceilings remain TBD. A measured one-step fixture cannot estimate full training convergence/runtime. GPU availability in this process: CUDA=false, MPS=false. No claim that the machine physically lacks a GPU.

Primary comparison matches candidate count, not FLOPs or wall-clock. Report NFE, model parameter count, training/evaluation wall-clock, memory, verification calls and storage separately. Every full MD5 q=128 and SHA-256 q=256 job remains in UNFINISHED_JOBS.json; no scientific condition was removed due to runtime.
'''
(root/'automation/RESOURCE_PLAN.md').write_text(resource)

power=['# Power / MDE analysis status','','**BLOCKED_SPECIFICATION for confirmatory power.** Main/pilot manifests and registered baseline expectation/discordance assumptions, Holm family and power target are absent. Actual scientific N and powered MDE cannot be computed honestly. The table below is from completed 60-message engineering sanity datasets, not the pilot or main datasets.','','| Algorithm | Source | q | Unique train | Validation | Test | MC expected Success@100 | Optimistic detection floor |','|---|---|---:|---:|---:|---:|---:|---|']
for row in summary['datasets']:
 n=row['unique_test'];power.append(f"| {row['algorithm']} | {row['source']} | {row['q']} | {row['unique_train']} | {row['unique_validation']} | {n} | {row['mc_expected_at_100']:.6f} | 5/{n} = {5/n:.3f} (one unadjusted hypothesis, best case only) |")
power+=['','The floor assumes every discordance favors the model and no baseline-only wins: five model-only wins give exact one-sided p=1/32<0.05. It is NOT a power-based MDE, does NOT include Holm, and does NOT establish the paired-CI criterion. With m Holm hypotheses, replace five by the smallest d with 2^-d < 0.05/m; required effect floor is d/N. Actual power can be much lower and requires preregistered joint-outcome assumptions.','','MC uses the actual uniform length/source law, 1,000 iid candidates per target, seed 8; per-target Wilson intervals and transformed K intervals are saved in each source_prior_mc.json. Zero MC hits are resolution-limited and never treated as true zero probability. No 2^-q substitution is used.','','q=8 has at most 256 total digest conditions, with some assigned to train/validation. Increasing message count does not create more than 256 independent target conditions. q=12,16 also require their actual unique N. All planned q remain registered in the coverage matrix.','', '| Planned algorithm | q | Scientific unique counts / baseline / MDE |','|---|---:|---|']
for algorithm,qs in QS.items():
 for q in qs:power.append(f'| {algorithm} | {q} | NOT AVAILABLE — dataset/specification not frozen |')
(root/'statistics/POWER_MDE_ANALYSIS.md').write_text('\n'.join(power)+'\n')

scope='Engineering validators passed; this is not scientific gate approval.'
(root/'gates/G0_SUMMARY.md').write_text('# G0 summary\n\n'+scope+'\n\n12 exact-quota sanity datasets (40/10/10) have zero message/digest overlaps, valid independent dataset hashes, frozen engineering configuration checksums and per-dataset JSON/Markdown audits. Scientific pilot/main G0 remains BLOCKED_SPECIFICATION.\n')
(root/'gates/GATE_SUMMARY.md').write_text('''# Scientific gate status

| Gate | Current scientific status | Actual completed checks |
|---|---|---|
| G0 | BLOCKED_SPECIFICATION | 12 engineering sanity manifests/splits PASS; no scientific dataset frozen |
| G1 | PARTIAL | 9,636 codec round-trips plus 94 single glyph tests PASS; verifier exhaustive PASS; learned reversible controls NOT RUN |
| G2 | NOT_RUN | Exact-budget, ordering, version and negative-case unit tests PASS; sanity self-comparisons PASS |
| G3 | NOT_RUN | One-/two-sided exact McNemar, 10,000-resample paired bootstrap and Holm exercised on synthetic binary fixture only |
| G4 | NOT_RUN | Seed aggregation implementation checked on synthetic fixture; no scientific model seeds trained |

The legacy G0–G6 gate namespace is unrelated and its old PASS labels are not imported. No L0–L4 efficacy evidence is assigned to unexecuted experiments.
''')
for gate in ('G3','G4'):
 write_json(root/'gates'/f'{gate}_SUMMARY.json',{'gate':gate,'status':'NOT_RUN','reason':'No new-protocol scientific model/baseline paired outcomes; fixture results are separate.'})

# Efficacy report stubs deliberately contain no invented values.
for filename,title in [('PILOT_REPORT.md','MD5 pilot'),('MD5_MAIN_REPORT.md','MD5 main'),('SHA256_REPORT.md','SHA-256 replication')]:
 (root/'reports'/filename).write_text(f'# {title}\n\nStatus: BLOCKED_SPECIFICATION / NOT RUN. No confirmatory model efficacy, McNemar p-value or G4 result exists. See [final report](FINAL_EXPERIMENT_REPORT.md), [gate status](../gates/GATE_SUMMARY.md), and [coverage](../automation/CONFIRMATORY_COVERAGE.csv). Engineering checks involving SHA-256 are verifier/baseline sanity checks, not a replication phase started before MD5.\n')

audit=root/'automation/IMPLEMENTATION_AUDIT.md'
initial=audit.read_text().split('\n## Post-implementation audit')[0]
audit.write_text(initial+f'''
## Post-implementation audit ({now})

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

Current code source hash: `{version}`. No research plan was relaxed or modified. Engineering fixture settings were frozen independently and are not recommended scientific hyperparameters. Confirmatory architecture/schedule/selection changes must be selected from train/validation under a registered protocol before test access.
''')

smoke_table='\n'.join(['| Family | Train steps | Parameters | Targets | Attempts | Valid decode | Scope |','|---|---:|---:|---:|---:|---:|---|']+[f"| {r['family']} | {r['training_steps']} | {r['parameters']} | {r['targets']} | {r['attempts']} | {r['valid_decode_rate']:.3f} | engineering only |" for r in smoke])
data_table='\n'.join(['| Algorithm | Source | q | Train conditions | Validation | Test |','|---|---|---:|---:|---:|---:|']+[f"| {r['algorithm']} | {r['source']} | {r['q']} | {r['unique_train']} | {r['unique_validation']} | {r['unique_test']} |" for r in summary['datasets']])
q_table='\n'.join(['| Algorithm | q | K | Scientific status / result |','|---|---:|---|---|']+[f'| {algorithm} | {q} | 1,10,100 | NOT RUN / N/A |' for algorithm,qs in QS.items() for q in qs])
report=f'''# Final Experiment Report — current session

## 1. Executive Summary

The new scientific question remains **UNANSWERED / BLOCKED_SPECIFICATION**. No new confirmatory pilot/main/full-digest experiment has run. Implemented and verified generalized image codecs, categorical tokenizer/masked diffusion, canonical conditioning, evaluator corrections, exact-quota dataset construction, gate/statistics/state components. Full regression: **88 passed** (baseline 65).

Completed 12 small engineering datasets, {sum(r['verdict_checks'] for r in summary['verifier']):,} exhaustive verifier verdict checks and 10 one-step actual-model engineering smoke jobs. All 20 smoke attempts were invalid; this verifies execution/accounting, not learned reversible control or scientific performance. No threshold or model was tuned in response. Do not infer success probability or convergence from these fixtures.

## 2. Experiment Coverage

- Completed scientific training jobs: **0/510**; scientific K evaluation rows: **0/1530**.
- Completed engineering smoke: 10 families, seed 0 only, one training update per family; saved checkpoints and candidate ledgers.
- Failed scientific jobs: none executed. Blocked: all scientific jobs (specification/G1 prerequisites). Resource planning additionally pending.
- Full MD5 q=128 and full SHA-256 q=256 remain present for all families/seeds/K in [coverage](../automation/CONFIRMATORY_COVERAGE.csv).
- Immutable engineering checks: [preflight](../automation/PREFLIGHT_RESULTS_{tag}.json), [smoke index](../automation/MATRIX_SMOKE_{tag}.json).

{smoke_table}

The smoke maximum length is 7 and seed=7, matching bounded test fixtures. It is explicitly outside the scientific plan; inherited scientific L_max=31 has provenance in the old plan/config and is not inferred from fixture performance.

## 3. Dataset and Splits

Scientific pilot/main datasets were not generated because fresh uncontaminated dataset/split specifications are unregistered. Each completed sanity dataset has exact message quotas 40/10/10. Actual unique digest counts:

{data_table}

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

{q_table}

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

Scientific [McNemar](../statistics/MCNEMAR_RESULTS.csv), [bootstrap](../statistics/BOOTSTRAP_RESULTS.csv) and [Holm](../statistics/HOLM_RESULTS.csv) files have **headers only**. Synthetic binary pipeline fixtures are under `statistics/PIPELINE_FIXTURE_{tag}` and cannot be cited as experiment results. Actual scientific N/power/MDE unavailable; see [power report](../statistics/POWER_MDE_ANALYSIS.md).

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

[Source archive](../automation/SOURCE_SNAPSHOT_{tag}.tar.gz) and [code manifest](../automation/CODE_MANIFEST_{tag}.json) capture the current uncommitted implementation. Git base commit is `{state['code_commit']}`; source hash `{version}`. Do not confuse base commit with the dirty working-tree code version.

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
- [Engineering preflight](../automation/PREFLIGHT_RESULTS_{tag}.json)
- [Engineering smoke index](../automation/MATRIX_SMOKE_{tag}.json)
- [Power/MDE status](../statistics/POWER_MDE_ANALYSIS.md)
- [MD5 pilot status](PILOT_REPORT.md)
- [MD5 main status](MD5_MAIN_REPORT.md)
- [SHA-256 status](SHA256_REPORT.md)
- [Legacy integrity](../automation/LEGACY_INTEGRITY.json)
- [Artifact checksum index](../automation/ARTIFACT_INDEX.json)
'''
(root/'reports/FINAL_EXPERIMENT_REPORT.md').write_text(report)
(root/'automation/NEXT_ACTIONS.md').write_text(f'''# Next actions / exact resume point

Current phase: scientific Phase 0 freeze BLOCKED_SPECIFICATION; independent Phase 1 engineering checks complete. First unresolved job: **SPECIFICATION_FREEZE**. Do not interpret completed engineering fixtures as scientific gate passes.

## Start every session here

```bash
.venv/bin/python -m diffusion_hash_inv.automation --verify
.venv/bin/python -m diffusion_hash_inv.experiment_state --verify
```

Read `artifacts/automation/EXPERIMENT_STATE.json`, `IMPLEMENTATION_AUDIT.md`, `artifacts/config/CONFIRMATORY_SPECIFICATION_TEMPLATE.json`, and `artifacts/reports/FINAL_EXPERIMENT_REPORT.md`. Current source version is `{version}`; base Git commit `{state['code_commit']}` plus uncommitted changes. Code snapshot is `artifacts/automation/SOURCE_SNAPSHOT_{tag}.tar.gz`.

## Independent engineering follow-up before training

Finish a confirmatory entry point that accepts registered external exact-quota manifests, enforces complete G0/G1 before training, implements the registered validation-only checkpoint rule and emits all secondary diagnostics/compute fields. Per-attempt sampling interruption resume and direct predictor final shared specification are still pending. Reuse current primitives; do not run the engineering CLI as a substitute. Add regression checks and run:

```bash
.venv/bin/python -m pytest -q
```

## Specification-dependent next work

Resolve each null in `artifacts/config/CONFIRMATORY_SPECIFICATION_TEMPLATE.json` from an actual registered decision or a validation-only development protocol. L_max=31 is inherited under the user's explicit legacy-value fallback; fresh dataset/split seeds, hyperparameters/selection/search budget, control target counts, statistical membership/seeds and compute/stopping envelope remain unresolved. Do not fill them from fixture outcomes. Ensure all observed old/engineering test corpora are excluded. Freeze a new content-addressed scientific config; never overwrite `artifacts/config/frozen/` snapshots.

Generate the 34 scientific datasets with registered quotas and bounded construction, run G0, codec/verifier checks and actual-model reversible controls for every family. Only after full G0/G1 PASS start the first planned model job: `MD5_PRINTABLE_GAUSSIAN_BGV_Q8_HASHONLY_SEED0_TRAIN`, then its prefix evaluations. Continue MD5 q8/12/16, q20/24/32/64, full128; then SHA-256 including full256. All job IDs/dependencies remain in UNFINISHED_JOBS.json.

To verify/skip the already completed engineering preflight:

```bash
.venv/bin/python -m diffusion_hash_inv.preflight --run
```

The completed engineering model checkpoints are under `artifacts/training/`; `complete.json` checksums prevent duplicate training. `training_resume.pt` contains optimizer/model/RNG state. A changed code/config requires a new run ID. Legacy `output/` and root EXPERIMENT_STATE.json remain unchanged; do not resume their stopped scientific protocol as this new study.
''')
reports=[str(p) for folder in ('automation','gates','reports','statistics') for p in (root/folder).glob('*.md')]
state=json.loads(STATE_PATH.read_text());state['generated_reports']=sorted(reports);write_json(STATE_PATH,state)
print(json.dumps({'reports':len(reports),'diffusion_training_jobs':streams,'K_rows':len(coverage),
                  'candidate_upper_bound':candidate_upper,'raw_storage_upper_bytes':raw_bytes,'status':'BLOCKED_SPECIFICATION'}))
