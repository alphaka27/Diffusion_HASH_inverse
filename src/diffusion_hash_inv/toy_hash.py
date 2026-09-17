"""G4 exhaustive toy oracle and independent verification of generated candidates."""
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import traceback
import torch
from .conditional_dependence import validate_completed
from .dataset import build_digest_records,select_digest_representatives,split_digest_groups,split_validation_report
from .evaluation import verify_candidate
from .experiment_state import checkpoint,experiment_lock,record_gate,require_previous,sha256,verify_prerequisites,write_json
from .models import GaussianDiffusion
from .runner import ExperimentConfig,_build_model,_codec,_conditions,_training_data

COMMAND='.venv/bin/python -m diffusion_hash_inv.toy_hash'


def reference_prefix(message,algorithm,q):
    digest={'md5':hashlib.md5,'sha256':hashlib.sha256}[algorithm](message).digest()
    if not 1<=q<=len(digest)*8:raise ValueError('Invalid prefix width')
    binary=''.join(format(byte,'08b') for byte in digest)
    return int(binary[:q],2)


def toy_domain():
    return tuple(bytes(values) for values in itertools.product(b'ABCD',repeat=4))


def oracle_sets(domain,algorithm,q):
    result={}
    for message in domain: result.setdefault(reference_prefix(message,algorithm,q),set()).add(message)
    return result


def audit_oracle(domain,oracles,algorithm,q):
    records=build_digest_records(domain,source='printable',algorithm=algorithm,q=q)
    targets=select_digest_representatives(records)
    for target in targets:
        assert int(target.prefix,16)==reference_prefix(target.message,algorithm,q)
        for message in domain:
            assert verify_candidate(message,target).prefix_match == (message in oracles[int(target.prefix,16)])
    return dict(agreement=True,comparison_count=len(targets)*len(domain),domain_size=len(domain),prefix_groups=len(targets),
                collision_groups=sum(len(v)>1 for v in oracles.values()))


def train_model(config,records,output):
    encoder,decoder,shape=_codec('bits')
    with torch.random.fork_rng():
        torch.manual_seed(config.model_seed);model=_build_model(config,shape)
    diffusion=GaussianDiffusion(config.diffusion_steps,beta_end=config.beta_end,prediction_type=config.prediction_type,device=torch.device('cpu'))
    final=output/'checkpoint.pt'
    if final.exists():
        data=torch.load(final,weights_only=False);assert data['config']==asdict(config)
        model.load_state_dict(data['model_state'])
    else:
        clean,conditions=_training_data(records,config,encoder,torch.device('cpu'))
        optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
        streams={name:torch.Generator().manual_seed(config.model_seed+offset) for name,offset in [('batch',4400),('timestep',4500),('noise',4600)]}
        resume=output/'training_resume.pt';start=0
        if resume.exists():
            data=torch.load(resume,weights_only=False);assert data['config']==asdict(config)
            start=data['step'];model.load_state_dict(data['model_state']);optimizer.load_state_dict(data['optimizer'])
            for name,g in streams.items():g.set_state(data['rng'][name])
        model.train()
        for step in range(start+1,config.training_steps+1):
            ids=torch.randint(len(clean),(config.batch_size,),generator=streams['batch']);values=clean[ids]
            index=torch.randint(diffusion.steps,(len(values),),generator=streams['timestep'])
            noise=torch.randn(values.shape,generator=streams['noise'])
            noisy=diffusion.add_noise(values,noise,index)
            loss=torch.nn.functional.mse_loss(model(noisy,index.float()/(diffusion.steps-1),conditions[ids]),values)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step%250==0 or step==config.training_steps:
                temp=resume.with_suffix('.tmp')
                torch.save(dict(config=asdict(config),step=step,model_state=model.state_dict(),optimizer=optimizer.state_dict(),rng={name:g.get_state() for name,g in streams.items()}),temp);temp.replace(resume)
                with (output/'training.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=loss.item()))+'\n')
                checkpoint('RUN_SMALL training',f'G4 seed0 training step{step}',COMMAND,gate='G4',status='RUNNING',artifacts=[str(resume)])
        torch.save(dict(config=asdict(config),model_state=model.state_dict()),final)
    model.eval()
    return model,diffusion,decoder,shape


@torch.no_grad()
def generate(model,diffusion,condition,shape,steps,k,seed):
    # This function receives only the digest condition and fixed sampling settings.
    return diffusion.sample(model,condition[None].expand(k,-1),shape,sampling_steps=steps,generator=torch.Generator().manual_seed(seed))


def evaluate(targets,model,diffusion,decoder,shape,config,oracles,domain,output,k,split,seed):
    rows=[]
    for target,condition in zip(targets,_conditions(targets,config)):
        unit=output/f'{split}-target-{target.prefix}.json'
        provenance=dict(checkpoint_sha256=sha256(output/'checkpoint.pt'),k=k,split=split,seed=seed,
                        target_prefix=target.prefix,protocol_sha256=sha256('output/g4/config_frozen.json'))
        if unit.exists():
            data=json.loads(unit.read_text());assert data['provenance']==provenance and len(data['rows'])==k
        else:
            values=generate(model,diffusion,condition,shape,config.sampling_steps,k,seed)
            attempts=[]
            for index,value in enumerate(values):
                decoded=decoder.decode((value+1)/2)
                candidate=decoded.message
                in_domain=decoded.valid and candidate in domain
                verified=verify_candidate(candidate,target) if decoded.valid else None
                matched=in_domain and verified.prefix_match
                membership=in_domain and candidate in oracles[int(target.prefix,16)]
                assert matched==membership
                attempts.append(dict(target_id=target.id,target_prefix=target.prefix,target_hex=target.message.hex(),candidate_index=index,
                    candidate_hex=candidate.hex() if candidate is not None else None,codec_valid=decoded.valid,valid=in_domain,
                    reason=decoded.reason if not decoded.valid else None if in_domain else 'outside_toy_domain',
                    exact_original=in_domain and candidate==target.message,hash_prefix_match=bool(matched),oracle_membership=bool(membership),
                    raw_prefix_match=verified.prefix_match if verified else False,actual_digest=verified.digest if verified else None,
                    full_digest_match=verified.digest==target.digest.hex() if verified else False))
            data=dict(provenance=provenance,rows=attempts);write_json(unit,data)
        rows.extend(data['rows'])
        checkpoint('RUN_SMALL' if split=='validation' else 'RUN_FULL',f'G4 {split} target {target.prefix} complete',COMMAND,gate='G4',status='RUNNING',artifacts=[str(unit)])
    return rows


def metric_summary(rows,k):
    targets=sorted({r['target_prefix'] for r in rows});n=len(targets)
    assert len(rows)==n*k
    return dict(target_count=n,candidate_count=len(rows),K=k,
        valid_decode=sum(r['valid'] for r in rows)/len(rows),codec_valid_decode=sum(r['codec_valid'] for r in rows)/len(rows),
        exact_original_recovery=sum(any(r['exact_original'] for r in rows if r['target_prefix']==t) for t in targets)/n,
        hash_prefix_match=sum(any(r['hash_prefix_match'] for r in rows if r['target_prefix']==t) for t in targets)/n,
        oracle_preimage_membership=sum(any(r['oracle_membership'] for r in rows if r['target_prefix']==t) for t in targets)/n,
        unique_candidate_ratio=sum(len({r['candidate_hex'] for r in rows if r['target_prefix']==t and r['valid']}) for t in targets)/len(rows),
        generated_match_count=sum(r['hash_prefix_match'] for r in rows),oracle_agreement=all(r['hash_prefix_match']==r['oracle_membership'] for r in rows))


def run():
    require_previous('G4');verify_prerequisites()
    state=json.loads(Path('EXPERIMENT_STATE.json').read_text())
    if state['gates']['G4'] not in ('NOT_RUN','PASS'):raise RuntimeError('G4 already reached fail-stop')
    freeze=json.loads(Path('output/g4/config_frozen.json').read_text());config=ExperimentConfig(**freeze['experiment'])
    output=Path('output/g4/experiments/seed-0');output.mkdir(parents=True,exist_ok=True)
    domain=toy_domain();oracles=oracle_sets(domain,config.algorithm,config.q)
    audit_path=Path('output/g4/oracle/audit.json')
    if audit_path.exists():audit=json.loads(audit_path.read_text())
    else:
        audit=audit_oracle(domain,oracles,config.algorithm,config.q)
        write_json(audit_path,audit)
        write_json('output/g4/oracle/preimages.json',{str(k):[v.hex() for v in sorted(values)] for k,values in oracles.items()})
    records=build_digest_records(domain,source='printable',algorithm=config.algorithm,q=config.q)
    splits=split_digest_groups(records,seed=config.split_seed)
    leakage=split_validation_report(splits);assert leakage['passed']
    write_json('output/g4/oracle/split_audit.json',leakage)
    write_json('output/g4/oracle/split.json',{name:[r.to_json(name) for r in values] for name,values in splits.items()})
    checkpoint('VALIDATE','G4 exhaustive oracle and split audit PASS',COMMAND,gate='G4',status='RUNNING',artifacts=[str(audit_path)],active_model_seed=0,active_config='output/g4/config_frozen.json')
    model,diffusion,decoder,shape=train_model(config,splits['train'],output)
    write_json(output/'config.json',dict(**freeze,seed_namespace=dict(dataset_seed=4000,split_seed=4001,model_seed=0,batch_seed=4400,timestep_seed=4500,diffusion_noise_seed=4600,sampling_seed=4100,condition_seed=None,baseline_seed=4200,bootstrap_seed=4300)))
    write_json(output/'environment.json',dict(torch=torch.__version__,threads=torch.get_num_threads(),device='cpu'))
    (output/'command.txt').write_text(COMMAND+'\n')
    if not (output/'smoke_metrics.json').exists():
        smoke=evaluate(select_digest_representatives(splits['validation'])[:3],model,diffusion,decoder,shape,config,oracles,set(domain),output,10,'validation',4101)
        write_json(output/'smoke_metrics.json',metric_summary(smoke,10))
        checkpoint('VALIDATE','G4 validation smoke integrity PASS; no parameter adaptation',COMMAND,gate='G4',status='RUNNING',artifacts=[str(output/'smoke_metrics.json')])
    targets=select_digest_representatives(splits['test'])
    rows=evaluate(targets,model,diffusion,decoder,shape,config,oracles,set(domain),output,100,'test',4100)
    metrics={str(k):metric_summary([r for r in rows if r['candidate_index']<k],k) for k in (1,10,100)}
    status='PASS' if audit['agreement'] and leakage['passed'] and metrics['100']['oracle_agreement'] and metrics['100']['generated_match_count']>0 else 'FAIL'
    write_json(output/'metrics.json',dict(status=status,metrics=metrics))
    write_json(output/'diagnostics.json',dict(oracle_audit=audit,leakage=leakage,no_target_side_inputs=True,checkpoint_sha256=sha256(output/'checkpoint.pt')))
    (output/'per_target.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    write_json(output/'complete.json',dict(sha256={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name not in ('complete.json','training_resume.pt')}))
    lines=['# G4 Toy Hash Pipeline','',f'Gate: {status}. MD5 q=8 prefix only, alphabet ABCD, length4, 256-message exhaustive domain.','',
           '| K | Targets | Exact original@K | HashPrefixMatch@K | Valid ratio | Unique ratio |','|---:|---:|---:|---:|---:|---:|']
    for k,m in metrics.items():lines.append(f"| {k} | {m['target_count']} | {m['exact_original_recovery']:.6f} | {m['hash_prefix_match']:.6f} | {m['valid_decode']:.6f} | {m['unique_candidate_ratio']:.6f} |")
    lines.extend(['',f'Independent oracle audit: {audit}. Split audit: {leakage}.',
        'Independent reference uses hashlib plus binary-string prefix extraction; production verifier uses big-endian integer shifts. Every enumerated message is checked against every unique target prefix.',
        'Generator inputs: digest bits, Gaussian initial noise, timestep. Oracle and representative target messages are used only after generation for evaluation. No candidate repair or extra attempts. Out-of-domain/invalid candidates and duplicates consume budget.',
        'Full-digest equality and exact original equality are separately saved. 8-bit prefix matches do not imply full MD5 preimage capability.',
        'Validation smoke checks integrity only. Full configuration was frozen before training/evaluation. A valid run with zero generated held-out preimages fails the required G4 functional-success criterion.',
        'Reproduce/resume: `'+COMMAND+'`.'])
    Path('output/g4/report.md').write_text('\n'.join(lines)+'\n')
    record_gate('G4',status,'Oracle agreement, no leakage and at least one held-out preimage.' if status=='PASS' else 'No valid held-out preimage generated at the frozen K=100 budget despite verified oracle/pipeline.',
        'Freeze and implement G5 matched-budget baseline comparison.' if status=='PASS' else 'STOP. G5/G6 NOT RUN. Read EXPERIMENT_G2_TO_G6_REPORT.md.')
    print(json.dumps(dict(status=status,metrics=metrics),sort_keys=True),flush=True)


def main():
    torch.set_num_threads(1)
    try:run()
    except Exception:
        write_json('output/g4/error.json',dict(traceback=traceback.format_exc()))
        checkpoint('ERROR','G4 error saved; inspect before retry',COMMAND,gate='G4',status='ERROR',last_failed_command=COMMAND,artifacts=['output/g4/error.json'])
        raise

if __name__=='__main__':
    with experiment_lock():main()
