"""G3 evaluation-time information ladder on immutable G2 checkpoints."""
import argparse
import json
from pathlib import Path
import traceback
import torch
from .conditional_dependence import load_setup, paired_sample, rows_for_outputs, summarize, validate_completed
from .experiment_state import checkpoint, experiment_lock, record_gate, require_previous, sha256, verify_prerequisites, write_json
from .runner import _codec, _training_data

COMMAND='.venv/bin/python -m diffusion_hash_inv.information_ladder --all'
LEVELS={'correct':1,'L75':.75,'L50':.5,'L25':.25,'L12.5':.125,'L0':0}


def native_mask(representation, fraction, seed=3100):
    if fraction not in LEVELS.values(): raise ValueError('Unsupported frozen information level')
    _,_,shape=_codec(representation)
    mask=torch.zeros(shape,dtype=torch.bool)
    if fraction==1: return torch.ones(shape,dtype=torch.bool)
    slots=torch.randperm(31,generator=torch.Generator().manual_seed(seed))[:int(31*fraction)]
    for slot in slots.tolist():
        if representation=='bits': mask[slot+1,:]=True
        else:
            row,col=divmod(slot+(representation=='bgv'),8)
            height,width=(8,16) if representation=='bgv' else (8,8)
            mask[0,row*height:(row+1)*height,col*width:(col+1)*width]=True
    return mask


def payload_mask(representation,length):
    _,_,shape=_codec(representation)
    mask=torch.zeros(shape,dtype=torch.bool)
    for slot in range(length):
        if representation=='bits': mask[slot+1]=True
        else:
            row,col=divmod(slot+(representation=='bgv'),8)
            width=16 if representation=='bgv' else 8
            mask[0,row*8:(row+1)*8,col*width:(col+1)*width]=True
    return mask


def run(rep,seed):
    require_previous('G3'); verify_prerequisites()
    output=Path(f'output/g3/{rep}/seed-{seed}');output.mkdir(parents=True,exist_ok=True)
    if validate_completed(output): print(f'SKIP verified {output}',flush=True);return
    checkpoint('RUN_SMALL' if seed==0 else 'RUN_FULL','',COMMAND,gate='G3',status='RUNNING',active_model_seed=seed,active_config='output/g3/config_frozen.json')
    source=Path(f'output/g2/{rep}/seed-{seed}')
    if not validate_completed(source): raise RuntimeError('Incomplete G2 model artifact')
    config,sets,encoder,decoder,shape,model,diffusion=load_setup(rep,seed,source,COMMAND)
    records=sets['test']; clean,conditions=_training_data(records,config,encoder,torch.device('cpu'))
    masks={name:native_mask(rep,f) for name,f in LEVELS.items()}
    variants={name:conditions*mask.flatten() for name,mask in masks.items()}
    assert not variants['L0'].any() and torch.equal(variants['correct'],conditions)
    payload=torch.stack([payload_mask(rep,len(r.message)) for r in records])
    provenance=dict(checkpoint_sha256=sha256(source/'checkpoint.pt'),protocol_sha256=sha256('output/g3/config_frozen.json'))
    all_rows=[];diagnostics={}
    for sampling_seed in (0,1,2):
        unit=output/f'sampling-{sampling_seed}.json'
        if unit.exists():
            data=json.loads(unit.read_text());assert data['provenance']==provenance
            assert len(data['rows'])==len(records)*len(LEVELS)
        else:
            outputs,initial=paired_sample(model,diffusion,variants,shape,config.sampling_steps,sampling_seed)
            rows=rows_for_outputs(outputs,records,torch.arange(len(records)),clean,decoder,'test',sampling_seed)
            for row in rows:
                i=row['target_index']; value=outputs[row['condition']][i]; selection=payload[i]
                row.update(payload_bit_accuracy=((value[selection]>=0)==(clean[i][selection]>=0)).float().mean().item(),
                    payload_mse=(value[selection]-clean[i][selection]).square().mean().item(),
                    disagreement_from_L0=((value>=0)!=(outputs['L0'][i]>=0)).float().mean().item())
            data=dict(provenance=provenance,rows=rows,paired_x_T=True,
                initial_noise_sha256=__import__('hashlib').sha256(initial.numpy().tobytes()).hexdigest())
            write_json(unit,data)
        all_rows.extend(data['rows']);diagnostics[str(sampling_seed)]={k:v for k,v in data.items() if k!='rows'}
        checkpoint('VALIDATE',f'G3 {rep} seed{seed} sampling{sampling_seed} complete',COMMAND,gate='G3',status='RUNNING',active_sampling_seed=sampling_seed,artifacts=[str(unit)])
    metrics={}
    for name in LEVELS:
        rows=[r for r in all_rows if r['condition']==name]
        metrics[name]=summarize(rows)
        for key in ('payload_bit_accuracy','payload_mse','disagreement_from_L0'):
            metrics[name][key]=sum(r[key] for r in rows)/len(rows)
        metrics[name]['candidate_diversity']=sum(len({r['candidate_hex'] for r in rows if r['target_index']==i and r['valid']})/3 for i in range(len(records)))/len(records)
        metrics[name]['effective_payload_fraction']=int(31*LEVELS[name])/31
    passed=(metrics['correct']['exact_recovery_rate']>=.99 and
            metrics['L50']['payload_bit_accuracy']>metrics['L0']['payload_bit_accuracy'] and
            metrics['L50']['payload_mse']<metrics['L0']['payload_mse'] and
            metrics['L50']['disagreement_from_L0']>0)
    write_json(output/'metrics.json',dict(status='PASS' if passed else 'FAIL',metrics=metrics))
    write_json(output/'diagnostics.json',dict(pairs=diagnostics,
        masks={name:mask.flatten().nonzero().flatten().tolist() for name,mask in masks.items()},
        metadata_removed_at_partial_levels=True,L0_constant=True,no_additional_target_inputs=True))
    write_json(output/'config.json',dict(**provenance,representation=rep,model_seed=seed,sampling_seeds=[0,1,2],mask_seed=3100,checkpoint=str(source/'checkpoint.pt')))
    (output/'environment.json').write_bytes((source/'environment.json').read_bytes())
    (output/'command.txt').write_text(COMMAND+'\n')
    (output/'per_target.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in all_rows))
    write_json(output/'complete.json',dict(sha256={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='complete.json'}))
    checkpoint('VALIDATE',f'G3 {rep} seed{seed} '+('PASS' if passed else 'FAIL'),COMMAND,gate='G3',status='READY',artifacts=[str(output/'metrics.json')],last_successful_command=COMMAND)
    print(f'{rep} seed{seed}: '+('PASS' if passed else 'FAIL'),flush=True)


def run_all():
    require_previous('G3')
    state=json.loads(Path('EXPERIMENT_STATE.json').read_text())
    if state['gates']['G3'] not in ('NOT_RUN','PASS'): raise RuntimeError('G3 already stopped')
    results=[]
    for seed in (0,1,2):
        for rep in ('bits','bgv','cgge'):
            run(rep,seed)
            result=json.loads(Path(f'output/g3/{rep}/seed-{seed}/metrics.json').read_text())
            results.append((rep,seed,result))
            if result['status']!='PASS': break
        if results[-1][2]['status']!='PASS': break
    status='PASS' if len(results)==9 and all(r[2]['status']=='PASS' for r in results) else 'FAIL'
    lines=['# G3 Information Ladder','',f'Gate: {status}', '', '| Representation | Seed | Level | Exact | Valid | Byte accuracy | Payload bit accuracy | Payload MSE | Diversity |', '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for rep,seed,result in results:
        for name,m in result['metrics'].items():
            lines.append(f"| {rep} | {seed} | {name} | {m['exact_recovery_rate']:.6f} | {m['valid_decode_rate']:.6f} | {m['byte_accuracy']:.6f} | {m['payload_bit_accuracy']:.6f} | {m['payload_mse']:.6f} | {m['candidate_diversity']:.6f} |")
    lines.extend(['','Protocol: config_frozen.json. Masks expose 31,23,15,7,3,0 of 31 cells (100%,74.19%,48.39%,22.58%,9.68%,0%).',
      'Partial conditions remove all length headers and validity channels; L0 is identical zero for every target. No target index, length, padding mask or other side input is passed to the model. Visible byte/glyph content and visible padding can imply constraints on length; these are part of the declared revealed cells, not an extra metadata channel.',
      'Payload metrics use true payload positions only in the evaluator. Accuracy includes exposed cells and does not establish hidden-content prediction above prior. The frozen 50% test is directional across all model seeds, without a population significance claim.',
      'Invalid decoding from missing metadata remains invalid. No target-provided repair is applied. Candidate diversity is the number of unique valid decoded candidates per target divided by 3 attempts.',
      'Full bit/pixel accuracy and MSE, byte accuracy, exact/valid recovery, paired differences and per-target results are saved in each run directory.',
      'Reproduce/resume: `'+COMMAND+'`.'])
    Path('output/g3/report.md').write_text('\n'.join(lines)+'\n')
    record_gate('G3',status,'All representations/seeds meet the frozen full/partial/L0 criteria.' if status=='PASS' else 'Frozen full/partial/L0 criterion failed.',
        'Implement and freeze G4 toy hash oracle and pipeline.' if status=='PASS' else 'STOP. Review output/g3/report.md; G4–G6 NOT RUN.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');parser.add_argument('--representation',choices=['bits','bgv','cgge']);parser.add_argument('--model-seed',type=int,choices=[0,1,2]);args=parser.parse_args()
    if not args.all and (args.representation is None or args.model_seed is None): parser.error('Specify --all or representation and seed')
    torch.set_num_threads(1)
    try:
        if args.all: run_all()
        else: run(args.representation,args.model_seed)
    except Exception:
        write_json('output/g3/error.json',dict(traceback=traceback.format_exc()))
        checkpoint('ERROR','G3 error saved; inspect before retry',COMMAND,gate='G3',status='ERROR',last_failed_command=COMMAND,artifacts=['output/g3/error.json'])
        raise

if __name__=='__main__':
    with experiment_lock(): main()
