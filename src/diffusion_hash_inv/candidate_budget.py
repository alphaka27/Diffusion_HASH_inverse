"""G5 fixed-budget, target-paired toy efficacy screen."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import statistics
import traceback
import torch
from .conditional_dependence import validate_completed
from .dataset import build_digest_records,select_digest_representatives,split_digest_groups
from .evaluation import paired_comparison,holm_adjust,verify_candidate
from .experiment_state import checkpoint,experiment_lock,record_gate,require_previous,sha256,verify_prerequisites,write_json
from .models import DirectPredictor
from .runner import ExperimentConfig,_codec,_conditions,_train_predictor
from .toy_hash import toy_domain,oracle_sets

COMMAND='.venv/bin/python -m diffusion_hash_inv.candidate_budget'


def baseline_candidates(method,n,k,seed):
    if method not in ('uniform_random','source_prior') or min(n,k)<1:raise ValueError('Invalid baseline configuration')
    rng=random.Random(seed);domain=toy_domain()
    return [[rng.choice(domain) if method=='uniform_random' else bytes(rng.choice(b'ABCD') for _ in range(4)) for _ in range(k)] for _ in range(n)]


def outcomes(rows,targets,k):
    result=[]
    for target in targets:
        group=[r for r in rows if r['target_prefix']==target and r['candidate_index']<k]
        if len(group)!=k or {r['candidate_index'] for r in group}!=set(range(k)):
            raise ValueError('Every target must consume exactly K distinct attempt indices')
        valid=[r for r in group if r['valid']]
        successes=[r['candidate_index']+1 for r in valid if r['hash_prefix_match']]
        result.append(dict(target_prefix=target,success=bool(successes),exact=any(r['exact_original'] for r in valid),
            valid_ratio=len(valid)/k,unique_ratio=len({r['candidate_hex'] for r in valid})/k,
            duplicate_ratio=(len(group)-len({r['candidate_hex'] for r in group}))/k,
            candidates_to_first_match=min(successes) if successes else None))
    if set(r['target_prefix'] for r in rows)!=set(targets):raise ValueError('Target sets differ')
    return result


def compare_target_rows(model,baseline,seed=4300):
    if [r['target_prefix'] for r in model]!=[r['target_prefix'] for r in baseline]:raise ValueError('Unpaired target accounting')
    return asdict(paired_comparison([r['success'] for r in model],[r['success'] for r in baseline],bootstrap_seed=seed,bootstrap_samples=10000))


def aggregate(values):
    n=len(values)
    result={key:sum(r[key] for r in values)/n for key in ('success','exact','valid_ratio','unique_ratio','duplicate_ratio')}
    first=[r['candidates_to_first_match'] for r in values if r['candidates_to_first_match'] is not None]
    result.update(target_count=n,candidates_to_first_match_median_solved=statistics.median(first) if first else None,
                  zero_success_upper95=min(1.0,3/n) if not first else None)
    return result


def run():
    require_previous('G5');verify_prerequisites()
    state=json.loads(Path('EXPERIMENT_STATE.json').read_text())
    if state['gates']['G5'] not in ('NOT_RUN','PASS'):raise RuntimeError('G5 already reached fail-stop')
    source=Path('output/g4/experiments/seed-0')
    if not validate_completed(source):raise RuntimeError('G4 artifact incomplete')
    freeze=json.loads(Path('output/g5/config_frozen.json').read_text())
    protocol_sha=sha256('output/g5/config_frozen.json')
    config=ExperimentConfig(**json.loads(Path('output/g4/config_frozen.json').read_text())['experiment'])
    records=build_digest_records(toy_domain(),source='printable',algorithm='md5',q=8)
    split=split_digest_groups(records,seed=config.split_seed);targets=select_digest_representatives(split['test'])
    prefixes=[t.prefix for t in targets];domain=set(toy_domain());oracles=oracle_sets(domain,'md5',8)
    rows={'diffusion':[json.loads(line) for line in (source/'per_target.jsonl').read_text().splitlines()]}
    for method in ('uniform_random','source_prior','direct_predictor_K1'):
        output=Path('output/g5')/method;output.mkdir(parents=True,exist_ok=True)
        artifact=output/'per_target.jsonl'
        if artifact.exists() and (output/'complete.json').exists() and validate_completed(output):
            rows[method]=[json.loads(line) for line in artifact.read_text().splitlines()];continue
        checkpoint('RUN_SMALL' if method=='direct_predictor_K1' else 'RUN_FULL',f'Generate {method}',COMMAND,gate='G5',status='RUNNING',active_config='output/g5/config_frozen.json')
        if method=='direct_predictor_K1':
            encoder,decoder,shape=_codec('bits')
            with torch.random.fork_rng():torch.manual_seed(0);model=DirectPredictor(config.condition_dim,256,width=max(64,config.width*4))
            path=output/'checkpoint.pt'
            if path.exists():model.load_state_dict(torch.load(path,weights_only=False)['model_state'])
            else:
                loss=_train_predictor(model,split['train'],config,encoder,shape,torch.device('cpu'))
                torch.save(dict(model_state=model.state_dict(),config=asdict(config),loss=loss),path)
            model.eval()
            with torch.no_grad():values=model(_conditions(targets,config),shape).clamp(-1,1)
            candidates=[[decoder.decode((v+1)/2).message] for v in values]
            k=1
        else:
            k=100;candidates=baseline_candidates(method,len(targets),k,freeze['baseline_seeds'][method])
        generated=[]
        for target,group in zip(targets,candidates):
            for index,candidate in enumerate(group):
                valid=candidate in domain
                verified=verify_candidate(candidate,target) if candidate is not None else None
                matched=valid and verified.prefix_match
                assert bool(matched)==bool(valid and candidate in oracles[int(target.prefix,16)])
                generated.append(dict(target_id=target.id,target_prefix=target.prefix,target_hex=target.message.hex(),candidate_index=index,
                    candidate_hex=candidate.hex() if candidate is not None else None,valid=valid,
                    exact_original=valid and candidate==target.message,hash_prefix_match=bool(matched),
                    actual_digest=verified.digest if verified else None))
        artifact.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in generated))
        write_json(output/'config.json',dict(method=method,K=k,protocol_sha256=protocol_sha,model_seed=0,baseline_seed=freeze['baseline_seeds'].get(method)))
        write_json(output/'environment.json',dict(torch=torch.__version__,device='cpu'))
        write_json(output/'diagnostics.json',dict(candidate_count=len(generated),target_count=len(targets),oracle_agreement=True,target_blind=method!='direct_predictor_K1'))
        (output/'command.txt').write_text(COMMAND+'\n')
        write_json(output/'complete.json',dict(sha256={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='complete.json'}))
        rows[method]=generated
        checkpoint('VALIDATE',f'{method}: exact budget and oracle agreement validated',COMMAND,gate='G5',status='RUNNING',artifacts=[str(artifact)])
    # Save a provenance reference rather than regenerating completed diffusion attempts.
    write_json('output/g5/diffusion/source.json',dict(path=str(source),per_target_sha256=sha256(source/'per_target.jsonl'),checkpoint_sha256=sha256(source/'checkpoint.pt')))
    metrics={};comparisons={};per_target={}
    for k in freeze['K']:
        for method,attempts in rows.items():
            if method=='direct_predictor_K1' and k!=1:continue
            values=outcomes(attempts,prefixes,k);per_target[f'{method}@{k}']=values
            metrics[f'{method}@{k}']=aggregate(values)
            if method!='diffusion':comparisons[f'{method}@{k}']=compare_target_rows(per_target[f'diffusion@{k}'],values,freeze['bootstrap_seed'])
    adjusted=holm_adjust({name:c['mcnemar_pvalue'] for name,c in comparisons.items()})
    for name,c in comparisons.items():c['holm_adjusted_pvalue']=adjusted[name]
    primary=comparisons['source_prior@100']
    passed=all(comparisons[f'{name}@100']['delta_ci95'][0]>0 and comparisons[f'{name}@100']['holm_adjusted_pvalue']<.05 for name in ('source_prior','uniform_random'))
    status='PASS' if passed else 'FAIL' if primary['delta_ci95'][1]<0 else 'INCONCLUSIVE'
    write_json('output/g5/statistics/comparisons.json',comparisons)
    write_json('output/g5/statistics/per_target_outcomes.json',per_target)
    write_json('output/g5/statistics/metrics.json',metrics)
    write_json('output/g5/statistics/oracle_reference.json',dict(domain_size=256,target_preimage_counts={t.prefix:len(oracles[int(t.prefix,16)]) for t in targets},exhaustive_K256_success=1.0))
    for method in rows:
        write_json(f'output/g5/{method}/metrics.json',{k:v for k,v in metrics.items() if k.startswith(method+'@')})
    lines=['# G5 Matched Candidate Budget and Baselines','',f'Gate: {status}; model seed0; 18 shared target prefixes; MD5 q8 toy only.','',
        '| Method | K | HashMatch@K | Exact@K | Valid ratio | Unique ratio | Duplicate ratio |','|---|---:|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():
        method,k=name.split('@');lines.append(f"| {method} | {k} | {m['success']:.6f} | {m['exact']:.6f} | {m['valid_ratio']:.6f} | {m['unique_ratio']:.6f} | {m['duplicate_ratio']:.6f} |")
    lines+=['','| Comparison | Delta HashMatch | Paired 95% CI | McNemar p | Holm p |','|---|---:|---|---:|---:|']
    for name,c in comparisons.items():lines.append(f"| {name} | {c['absolute_gain']:.6f} | {c['delta_ci95']} | {c['mcnemar_pvalue']:.6f} | {c['holm_adjusted_pvalue']:.6f} |")
    lines+=['','Primary K=100, primary source-prior baseline. Both source-prior and uniform comparisons must pass to advance. Exact one-sided McNemar and 10,000 paired target bootstrap replicates; Holm correction across all 7 comparisons. No candidate-level inference or seed pooling.',
        'Baseline candidate functions receive method, target count, K and seed only. Source-prior iid ABCD^4 and uniform domain sampling have the same distribution; the two reported draws are independent RNG realizations.',
        'All attempts consume K, including malformed/invalid and repeated candidates. Unique ratio is unique valid candidates/K; duplicate ratio counts repeated candidate values including None. First-match indices and medians among solved targets are in statistics files.',
        'Exhaustive oracle reference reaches all targets after enumerating all256 messages; it is not a same-K efficacy baseline.',
        'G4 model outcomes were already visible before G5 baseline freeze. This is an exploratory staged screen, not a new confirmatory preregistration. Only model seed0 was evaluated for efficacy. No positive generalization beyond this toy setting follows.',
        'Reproduce/resume: `'+COMMAND+'`.']
    Path('output/g5/report.md').write_text('\n'.join(lines)+'\n')
    reason='Frozen paired efficacy criteria met.' if passed else 'Source-prior difference CI wholly below zero.' if status=='FAIL' else 'Frozen efficacy criteria not met; interval includes zero or evidence lacks precision.'
    record_gate('G5',status,reason,'Freeze G6 configuration and implement full experiment.' if passed else 'STOP. G6 NOT RUN. Read EXPERIMENT_G2_TO_G6_REPORT.md.')
    print(json.dumps(dict(status=status,primary=primary),sort_keys=True),flush=True)


def main():
    torch.set_num_threads(1)
    try:run()
    except Exception:
        write_json('output/g5/error.json',dict(traceback=traceback.format_exc()))
        checkpoint('ERROR','G5 error saved; inspect before retry',COMMAND,gate='G5',status='ERROR',last_failed_command=COMMAND,artifacts=['output/g5/error.json'])
        raise

if __name__=='__main__':
    with experiment_lock():main()
