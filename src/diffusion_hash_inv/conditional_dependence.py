"""G2 paired interventions; run units and training checkpoints are resumable."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import sys
import traceback

import torch
from .dataset import build_digest_records
from .experiment_state import checkpoint, record_gate, require_previous, sha256, verify_prerequisites, write_json
from .models import GaussianDiffusion
from .positive_control import _byte_score
from .runner import ExperimentConfig, _build_model, _codec, _training_data


def interventions(condition, seed=0):
    if len(condition) < 2:
        raise ValueError('Derangement requires at least two targets')
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(condition), generator=generator)
    mapping = torch.empty_like(order)
    mapping[order] = order.roll(1)
    assert (mapping != torch.arange(len(condition))).all()
    return {'correct': condition.clone(), 'deranged': condition[mapping], 'zero': torch.zeros_like(condition)}, mapping


@torch.no_grad()
def paired_sample(model, diffusion, conditions, shape, steps, seed):
    outputs, initials = {}, {}
    for name, condition in conditions.items():
        captured = []
        def capture(module, args):
            if not captured:
                captured.append(args[0].detach().clone())
        hook = model.register_forward_pre_hook(capture)
        try:
            outputs[name] = diffusion.sample(model, condition, shape, sampling_steps=steps,
                                             generator=torch.Generator(device=diffusion.device).manual_seed(seed))
        finally:
            hook.remove()
        initials[name] = captured[0]
    reference = next(iter(initials.values()))
    if not all(torch.equal(reference, value) for value in initials.values()):
        raise RuntimeError('Unpaired initial noise')
    return outputs, reference


def summarize(rows):
    n = len(rows)
    return dict(sample_count=n,
                exact_recovery_rate=sum(r['original_exact'] for r in rows)/n,
                valid_decode_rate=sum(r['valid'] for r in rows)/n,
                byte_accuracy=sum(r['original_byte_correct'] for r in rows)/sum(r['original_byte_total'] for r in rows),
                bit_accuracy=sum(r['bit_accuracy'] for r in rows)/n,
                reconstruction_mse=sum(r['mse'] for r in rows)/n,
                donor_exact=sum(r['donor_exact'] for r in rows)/n,
                donor_byte_accuracy=sum(r['donor_byte_correct'] for r in rows)/sum(r['donor_byte_total'] for r in rows),
                output_bit_disagreement=sum(r['output_bit_disagreement'] for r in rows)/n,
                output_byte_disagreement=sum(r['output_byte_disagreement'] for r in rows)/n,
                decoded_message_disagreement=sum(r['decoded_message_disagreement'] for r in rows)/n)


def rows_for_outputs(outputs, records, donors, clean, decoder, split, sampling_seed):
    decoded = {name: [decoder.decode((v+1)/2) for v in values] for name, values in outputs.items()}
    rows = []
    reference = outputs['correct']
    for name, values in outputs.items():
        for i, (value, result) in enumerate(zip(values, decoded[name])):
            candidate = result.message if result.valid else None
            original = records[i].message
            donor = records[int(donors[i])].message
            oc, ot = _byte_score(candidate, original)
            dc, dt = _byte_score(candidate, donor)
            changed = (value >= 0) != (reference[i] >= 0)
            ref = decoded['correct'][i]
            bc, bt = _byte_score(candidate, ref.message) if ref.valid else (0, 1)
            rows.append(dict(split=split, sampling_seed=sampling_seed, condition=name, target_index=i,
                target_hex=original.hex(), donor_index=int(donors[i]), donor_hex=donor.hex(),
                candidate_hex=candidate.hex() if candidate is not None else None, valid=result.valid, reason=result.reason,
                original_exact=candidate == original, donor_exact=candidate == donor,
                original_byte_correct=oc, original_byte_total=ot, original_byte_accuracy=oc/ot,
                donor_byte_correct=dc, donor_byte_total=dt, donor_byte_accuracy=dc/dt,
                bit_accuracy=((value >= 0) == (clean[i] >= 0)).float().mean().item(),
                mse=(value-clean[i]).square().mean().item(),
                output_bit_disagreement=changed.float().mean().item(),
                output_byte_disagreement=1-bc/bt if name != 'correct' else 0.0,
                decoded_message_disagreement=(result.valid, result.message) != (ref.valid, ref.message)))
    return rows


def load_setup(representation, seed, output, command):
    original = Path(f'output/g1-{representation}-seed0')
    saved = torch.load(original/'g1-c-n64-seed0/checkpoint.pt', map_location='cpu', weights_only=False)
    config = replace(ExperimentConfig(**saved['config']), model_seed=seed)
    sets = {split: [] for split in ('train','validation','test')}
    for line in (original/'split.jsonl').read_text().splitlines():
        row = json.loads(line)
        sets[row['split']].append(bytes.fromhex(row['message_hex']))
    assert all(not set(sets[a]) & set(sets[b]) for a,b in [('train','test'),('train','validation'),('validation','test')])
    sets = {k:build_digest_records(v, source=config.source, algorithm='md5', q=8) for k,v in sets.items()}
    encoder, decoder, shape = _codec(representation)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = _build_model(config, shape)
    diffusion = GaussianDiffusion(config.diffusion_steps, beta_end=config.beta_end,
                                  prediction_type=config.prediction_type, device=torch.device('cpu'))
    path = output/'checkpoint.pt'
    if path.exists():
        saved_model = torch.load(path, weights_only=False)
        if saved_model['config'] != asdict(config):
            raise RuntimeError('Checkpoint config mismatch')
        model.load_state_dict(saved_model['model_state'])
    elif seed == 0:
        model.load_state_dict(saved['model_state'])
        torch.save(saved, path)
    else:
        values, conditions = _training_data(sets['train'], config, encoder, torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        streams = {name:torch.Generator().manual_seed(seed+offset) for name,offset in [('batch',1000),('timestep',2000),('noise',3000)]}
        resume = output/'training_resume.pt'
        start = 0
        if resume.exists():
            data = torch.load(resume, weights_only=False)
            model.load_state_dict(data['model_state']); optimizer.load_state_dict(data['optimizer'])
            for name in streams: streams[name].set_state(data['rng'][name])
            start = data['step']
        model.train()
        for step in range(start+1, config.training_steps+1):
            ids = torch.randint(len(values), (config.batch_size,), generator=streams['batch'])
            clean = values[ids]
            index = torch.randint(diffusion.steps, (len(clean),), generator=streams['timestep'])
            noise = torch.randn(clean.shape, generator=streams['noise'])
            noisy = diffusion.add_noise(clean, noise, index)
            prediction = model(noisy, index.float()/(diffusion.steps-1), conditions[ids])
            loss = torch.nn.functional.mse_loss(prediction, clean if config.prediction_type=='sample' else noise)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step%250 == 0 or step == config.training_steps:
                temporary = resume.with_suffix('.tmp')
                torch.save(dict(step=step, model_state=model.state_dict(), optimizer=optimizer.state_dict(),
                                rng={name:g.get_state() for name,g in streams.items()}), temporary)
                temporary.replace(resume)
                with (output/'training.jsonl').open('a') as stream: stream.write(json.dumps(dict(step=step,loss=loss.item()))+'\n')
                checkpoint('RUN_FULL training', f'{representation} seed{seed} training step {step}', command,
                           status='RUNNING', artifacts=[str(resume)], active_model_seed=seed)
        torch.save(dict(config=asdict(config), model_state=model.state_dict()), path)
    model.eval()
    return config, sets, encoder, decoder, shape, model, diffusion


def validate_completed(output):
    marker = output/'complete.json'
    if not marker.exists(): return False
    for name, digest in json.loads(marker.read_text())['sha256'].items():
        if not (output/name).is_file() or sha256(output/name) != digest:
            raise RuntimeError(f'Completed artifact corrupt: {output/name}')
    return True


def run(representation, seed):
    require_previous('G2'); verify_prerequisites()
    output = Path(f'output/g2/{representation}/seed-{seed}'); output.mkdir(parents=True,exist_ok=True)
    command = f'.venv/bin/python -m diffusion_hash_inv.conditional_dependence --representation {representation} --model-seed {seed}'
    if validate_completed(output):
        print(f'SKIP verified {output}'); return
    checkpoint('RUN_SMALL' if seed==0 else 'RUN_FULL', '', command, status='RUNNING',
               active_model_seed=seed, active_config=str(output/'config.json'))
    config, sets, encoder, decoder, shape, model, diffusion = load_setup(representation,seed,output,command)
    write_json(output/'config.json', dict(experiment=asdict(config), conditions=['correct','deranged','zero'],
        sampling_seeds=[0,1,2], seed_namespace=dict(dataset_seed=0,split_seed=0,model_seed=seed,
            batch_seed=seed+1000 if seed else 0,timestep_seed=seed+2000 if seed else 0,
            diffusion_noise_seed=seed+3000 if seed else 0,condition_seed=100,
            sampling_seed=[0,1,2],baseline_seed=None,bootstrap_seed=None),
        legacy_seed0_shared_training_rng=seed==0,checkpoint_sha256=sha256(output/'checkpoint.pt'),
        split_sha256=sha256(f'output/g1-{representation}-seed0/split.jsonl')))
    (output/'command.txt').write_text(command+'\n')
    write_json(output/'environment.json',dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),threads=torch.get_num_threads()))
    all_rows=[]; diagnostics={}
    for split, records in sets.items():
        clean, condition = _training_data(records, config, encoder, torch.device('cpu'))
        variants, donors = interventions(condition,100)
        for sampling_seed in (0,1,2):
            unit=output/f'{split}-sampling-{sampling_seed}.json'
            if unit.exists():
                data=json.loads(unit.read_text())
                assert len(data['rows']) == len(records)*3
                assert data['checkpoint_sha256'] == sha256(output/'checkpoint.pt')
            else:
                outputs, initial = paired_sample(model,diffusion,variants,shape,config.sampling_steps,sampling_seed)
                data=dict(rows=rows_for_outputs(outputs,records,donors,clean,decoder,split,sampling_seed),
                          donor_mapping=donors.tolist(),paired_x_T=True,
                          initial_noise_sha256=__import__('hashlib').sha256(initial.numpy().tobytes()).hexdigest(),
                          checkpoint_sha256=sha256(output/'checkpoint.pt'))
                write_json(unit,data)
            all_rows.extend(data['rows'])
            diagnostics[f'{split}-{sampling_seed}']={k:v for k,v in data.items() if k!='rows'}
            checkpoint('VALIDATE paired unit',f'{representation} seed{seed} {split} sampling{sampling_seed} complete',command,
                       status='RUNNING',active_sampling_seed=sampling_seed,artifacts=[str(unit)])
    metrics={split:{name:summarize([r for r in all_rows if r['split']==split and r['condition']==name])
                    for name in variants} for split in sets}
    passed=all(m['correct']['exact_recovery_rate']>=.99 and all(
        m['correct']['exact_recovery_rate']>m[n]['exact_recovery_rate'] and m[n]['output_bit_disagreement']>0
        for n in ('deranged','zero')) for m in metrics.values())
    write_json(output/'metrics.json',dict(status='PASS' if passed else 'FAIL',metrics=metrics))
    write_json(output/'diagnostics.json',diagnostics)
    (output/'per_target.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in all_rows))
    write_json(output/'complete.json',dict(sha256={p.name:sha256(p) for p in output.iterdir()
        if p.is_file() and p.name not in ('complete.json','training_resume.pt')}))
    checkpoint('VALIDATE representation/model seed', f'{representation} seed{seed} '+('PASS' if passed else 'FAIL'),
               '.venv/bin/python -m diffusion_hash_inv.conditional_dependence --all',
               status='READY',last_successful_command=command,artifacts=[str(output/'metrics.json'),str(output/'complete.json')])
    print(json.dumps(dict(representation=representation,model_seed=seed,status='PASS' if passed else 'FAIL',test=metrics['test']),sort_keys=True))


def run_all():
    require_previous('G2'); verify_prerequisites()
    state=json.loads(Path('EXPERIMENT_STATE.json').read_text())
    if state['gates']['G2'] not in ('NOT_RUN','PASS'):
        raise RuntimeError('G2 already reached fail-stop; do not rerun')
    results=[]
    for seed in (0,1,2):
        for representation in ('bits','bgv','cgge'):
            run(representation,seed)
            path=Path(f'output/g2/{representation}/seed-{seed}/metrics.json')
            result=json.loads(path.read_text())
            results.append((representation,seed,result))
            if result['status']!='PASS': break
        if results[-1][2]['status']!='PASS': break
    status='PASS' if len(results)==9 and all(r[2]['status']=='PASS' for r in results) else 'FAIL'
    lines=['# G2 Conditional Dependence', '', '| Representation | Model seed | Correct exact | Deranged exact | Zero exact | Donor exact | Status |',
           '|---|---:|---:|---:|---:|---:|---|']
    for rep,seed,result in results:
        m=result['metrics']['test']
        lines.append(f"| {rep} | {seed} | {m['correct']['exact_recovery_rate']:.6f} | {m['deranged']['exact_recovery_rate']:.6f} | {m['zero']['exact_recovery_rate']:.6f} | {m['deranged']['donor_exact']:.6f} | {result['status']} |")
    lines.extend(['',f'Gate: {status}. Each row: 16 unseen targets × 3 paired sampling seeds; train/validation metrics saved separately.',
                  'Same correct-trained checkpoint, first model-input tensor, scheduler and sampling steps; only conditions change.',
                  'Donor mapping is a seeded random cycle without fixed points within each split. Seed0 checkpoints are copied from G1-C without training.',
                  'Model seeds 1/2 use the same G1 dataset/config with separately seeded batch/timestep/noise streams.',
                  'Output byte disagreement is decoded byte mismatch versus correct output, counting an invalid intervention output as total mismatch. Bit/pixel accuracy includes padding and masks; it is a diagnostic, not a hash efficacy metric.',
                  'Repeated sampling seeds are not independent targets. No population significance or hash-advantage claim is made.',
                  'Supported conclusion on PASS: condition affects reconstruction, including donor-directed output changes.',
                  'Reproduce/resume: `.venv/bin/python -m diffusion_hash_inv.conditional_dependence --all`.',
                  'Detailed metrics, per-target outcomes, initial-noise hashes, donor mapping, configs, checkpoints and environments are in representation/seed directories.'])
    Path('output/g2/report.md').write_text('\n'.join(lines)+'\n')
    record_gate('G2',status,'All three representations and three model seeds satisfy the frozen causal criteria.' if status=='PASS' else 'A representation/model seed failed frozen reconstruction or intervention criteria.',
                'Implement and freeze G3 information ladder before evaluation.' if status=='PASS' else 'STOP. Review output/g2/report.md; G3–G6 are NOT RUN.')
    print('G2 '+status)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--representation',choices=['bits','bgv','cgge'])
    parser.add_argument('--model-seed',type=int,choices=[0,1,2])
    parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    if not args.all and (args.representation is None or args.model_seed is None):
        parser.error('Specify --all or both --representation and --model-seed')
    torch.set_num_threads(1)
    try:
        if args.all: run_all()
        else: run(args.representation,args.model_seed)
    except Exception:
        error=Path('output/g2/error.log');error.parent.mkdir(parents=True,exist_ok=True)
        error.write_text(traceback.format_exc())
        command='.venv/bin/python -m diffusion_hash_inv.conditional_dependence --all'
        checkpoint('ERROR','Command failed; inspect error log before retry',command,status='ERROR',
                   last_failed_command=command,artifacts=[str(error)])
        raise


if __name__=='__main__': main()
