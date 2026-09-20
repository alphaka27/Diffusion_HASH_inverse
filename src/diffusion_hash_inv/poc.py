"""Sealed truncated-MD5 PoC. No legacy experiment state is read or modified.

Run --prepare, --validate, --run in order. --run resumes durable jobs/attempts.
SQLite transactions retain every invalid/duplicate attempt; raw sampler batches
are committed before evaluation, so recovery never resamples a saved output.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
import platform
import random
import resource
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np
import torch

from .baselines import _sample_message
from .conditioning import digest_condition, shuffled_donors
from .dataset import (DigestRecord, SourceSpec, quota_digest_split, select_digest_representatives,
                      split_validation_report, digest_prefix_hex)
from .discrete import MaskedDiffusion, SequenceDenoiser
from .encoding.bgv import BGVEncoder, BGVDecoder, DecodeResult
from .encoding.tokens import TokenCodec
from .evaluation import verify_candidate, paired_comparison, holm_adjust, binomial_ci95
from .models import ImageUNet, GaussianDiffusion, parameter_count

ROOT = Path('artifacts/poc_md5_truncated')
CORES = {'P-G-BGV': ('printable', 'gaussian'), 'R-G-BGV': ('random_bytes', 'gaussian'),
         'P-DISC': ('printable', 'discrete'), 'R-DISC': ('random_bytes', 'discrete')}
STATES = {'PLANNED','READY','RUNNING','COMPLETED','PASS','FAIL','INVALID','BLOCKED',
          'INCOMPLETE','NOT_RUN','NOT_EVALUATED','EXPLORATORY'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write(path, value):
    atomic(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def read(path):
    return json.loads(Path(path).read_text())


def seal(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'immutable artifact mismatch: {path}')
    else:
        atomic(path, data)
        path.chmod(0o444)
    return digest(path)


def seed(namespace, *keys):
    return int.from_bytes(hashlib.sha256((':'.join(map(str, (namespace, *keys)))).encode()).digest()[:8], 'big') % (2**63-1)


def save_torch(path, obj):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    atomic(path, buffer.getvalue())


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()


def state(phase=None, job=None, status=None, **details):
    s = read(ROOT/'RUN_STATE.json')
    if status is not None and status not in STATES:
        raise ValueError(status)
    if phase:
        s['phase'] = phase
        s['phases'][phase] = status or 'RUNNING'
    if job:
        s['jobs'][job] = dict(status=status, updated=now(), **details)
    s['updated'] = now()
    s['next_jobs'] = [k for k,v in s['jobs'].items() if v['status'] in {'PLANNED','READY','INCOMPLETE'}]
    write(ROOT/'RUN_STATE.json', s)
    if job or phase:
        message = dict(time=now(), phase=phase or s['phase'], job=job, status=status, **details)
        print(json.dumps(message), flush=True)
        with (ROOT/'RUNBOOK.md').open('a') as f:
            f.write('\n- ' + json.dumps(message) + '\n')


def code_manifest():
    files = sorted(Path('src/diffusion_hash_inv').rglob('*.py')) + sorted(Path('tests').glob('*.py'))
    files += [Path('pyproject.toml'), Path('RESEARCH_PLAN_POC.md')]
    return {str(p): digest(p) for p in files}


def verify_code(spec):
    if code_manifest() != spec['provenance']['files']:
        raise RuntimeError('code/specification changed after seal; refusing primary continuation')


def config(family):
    common = dict(optimizer='Adam', lr=.001, batch=16, updates=[1500], checkpoint_interval=500,
                  trial_cap_per_source_q=1, temperature=1.0, precision='float32')
    if family == 'gaussian':
        return dict(common, id='G01', architecture='ImageUNet', width=8, steps=50,
                    sampling_steps=50, beta_start=.0001, beta_end=.4, prediction='sample',
                    reverse='existing DDIM-style; clamp clean/output [-1,1]')
    return dict(common, id='D01', architecture='SequenceDenoiser', width=128, embedding_dim=16,
                steps=32, sampling_steps=32, grid=[i/32 for i in range(33)],
                forward='continuous per-sequence Uniform(0,1); iid Bernoulli(t) all positions',
                objective='per-sequence masked CE / max(1,masked count); batch mean; empty=0',
                reverse='reveal 1-t_previous/t_current; categorical; retain revealed; no repair')


def prepare():
    path = ROOT/'protocol/FROZEN_POC_SPEC.json'
    if path.exists():
        verify_code(read(path))
        return
    timing = read(ROOT/'protocol/TIMING.json')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    # Device must be explicit; no fallback during a run.
    if device != 'mps':
        raise RuntimeError('Preparation expects measured MPS device; run outside sandbox')
    namespaces = {name: 2026092000+i for i,name in enumerate(
        ['dataset','split','representative','sampling','shuffle','baseline','bootstrap','monte_carlo','control_data','training'],1)}
    spec = dict(version=1, created=now(), run_id='poc_md5_truncated_v1', algorithm='md5', qs=[8,12,16],
        q20_enabled=False, cores=CORES, min_length=4, max_length=31,
        sources={'printable':dict(alphabet=[33,126],size=94),'random_bytes':dict(alphabet=[0,255],size=256)},
        source_law='length first uniform 4..31 then iid uniform payload; 0x00 is payload',
        quotas=[10000,1000,1000], max_draws=240000, seeds=namespaces, model_seeds=[0,1,2],
        seed_derivation='SHA256 colon-separated(namespace,keys), first8 big-endian modulo 2**63-1',
        datasets='fresh source/q seeds; immutable first-seen weighted group owners; surplus unused; failure never relaxes quota',
        representatives='SHA256(representative seed colon + message), minimum per group; lexicographic prefix order',
        gaussian=config('gaussian'), discrete=config('discrete'),
        rationale={'gaussian':'Reuse examples/g1-bgv.json: no new architecture or sampler; actual fresh held-out controls required.',
                   'discrete':'Reuse existing global SequenceDenoiser. Width128 embedding16 is a modest explicit capacity (1.17M random-byte hash parameters); singleton finite trial, not claimed validated.',
                   'search':'One registered configuration and terminal update1500 per source/q; validation PreimageSuccess@100 still measured. No extra trials after failure.',
                   'control':'Direct held-out64/16/16 stage, retaining historical Gaussian held-out sizes/seeds; optional memorization ladder omitted because it does not establish held-out generalization.'},
        selection=dict(metric='validation unique-target PreimageSuccess@100',
            order=['highest score','fewer sampling steps','config ID order G01/D01','earliest update'],
            config_order=['G01','D01'], updates=[1500], all_validation_targets=True, k=100,
            sampling_namespace='sampling:validation:core:q:seed:method:target:batch',
            failed_trials_consume_cap=True, replicate_search=False, shuffled='exact selected main config/update/seed'),
        positive_control=dict(train=64,validation=16,test=16,model_seeds=[0,1,2],
            replication_controls='seeds1/2 only if Stage B eligible; same-source/config reused across q',
            sampling_seeds=[0,1,2],k=1,updates=1500,validation_intervals=[500,1000,1500],
            selection='fixed terminal checkpoint; validation monitoring only',
            gaussian='exact BGV [0,1] flatten passed through existing spatial-condition U-Net',
            discrete='32 tokens encoded as 9 MSB-first bits each (288 float bits) into existing global condition input; no output copying',
            pass_rule='held-out repetition-average ExactRecovery@1 >= .99 (48 draws clustered into16 targets); per-seed rates reported; no best-of',
            ci='10000 target-cluster bootstrap lower order statistic percentiles; Wilson per repetition; small-N caution'),
        condition='259 floats: MD5 onehot2,q/256,canonical firstq bits,constant zeros; no evaluator metadata',
        codec=dict(bgv_shape=[2,32,128],bgv_threshold=.5,bgv_tie='>=',token_length=32,
                   tokens={'printable':[94,95,96,97],'random_bytes':[256,257,258,259]},repair=False),
        budget=dict(ks=[1,10,100],attempts=100,invalid_consumes=True,duplicate_consumes=True,
                    initial_gaussian='normal float32',initial_discrete='32 MASK',no_early_stop=True,
                    generation_batch=100,raw_retention='all outputs lossless npz; raw batch committed before attempt transactions',
                    resume='model/Adam/local and global RNG checkpoint; atomic batch raw file; SQLite attempt PK/transactions; no replacing rows'),
        baseline=dict(shared_across=['models','seeds'],stream='one source/q/target100 stream',
                      mc_draws_per_source=1000000,mc='one independent source-prior histogram per source for all q; Wilson95 transformed to K expectation; correlated target intervals not a mean CI'),
        statistics=dict(unit='unique matched digest targets',bootstrap_samples=10000,
            quantile='sorted samples[floor(p*(B-1))], p=.025/.975; numpy PCG64 multinomial paired differences',
            mcnemar='one-sided exact Binomial(n10+n01,.5) upper tail; zero discordance p=1',
            primary_family=16,auxiliary_family=32,holm='joint complete family only; missing rows => adjusted p null',
            eligibility='q12/16 K100 seed0 G0/G1/G2 PASS and both deltas>0',
            signal='both K100 Holm p<.05 and both marginal CI lower>0; distinct from eligibility',
            evidence='P0 G0/G1/G2; P1 eligible seed0; P2 both positive effects in each of0/1/2; no pooling'),
        resources=dict(device=device,cpu_threads=1,precision='float32',unified_memory_bytes=128*2**30,
            gpu='Apple M3 Max 40-core Metal',gpu_count=1,cuda=None,cpu='Apple M3 Max 16-core',
            disk_free=shutil.disk_usage(ROOT).free,storage_quota_bytes=150*10**9,wall_policy='24 hours active execution per invocation; checkpoint and INCOMPLETE_RESOURCE on limit; resume same protocol',
            wall_seconds=86400,oom='INCOMPLETE_RESOURCE, preserve checkpoint; no silent device/config fallback',
            timing=timing,hash_training_jobs_stage_a=24,hash_training_jobs_stage_b_max=32,
            controls_stage_a=4,controls_stage_b_max=8,
            estimate='batch16 MPS timing conservative for batch100; StageA Gaussian<=902400*0.00852s; Discrete<=902400*0.00907s plus validation and controls',
            raw_max_gaussian_bytes=82000000000,retention='all failed/success/invalid raw/checkpoints retained; no selective deletion'),
        provenance=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                        files=code_manifest(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                        historical='Legacy reports read, prior engineering/PoC corpora not reused; no hash PoC test access before seal'))
    check_spec(spec)
    patch = subprocess.check_output(['git','diff','--binary'])
    atomic(ROOT/'protocol/tracked_dirty.patch',patch)
    # Include untracked scientific implementation in the immutable code snapshot.
    import tarfile
    with tarfile.open(ROOT/'protocol/SOURCE_SNAPSHOT.tar.gz','w:gz') as tar:
        for name in spec['provenance']['files']:
            tar.add(name)
    spec['provenance']['dirty_patch_sha256'] = digest(ROOT/'protocol/tracked_dirty.patch')
    spec['provenance']['snapshot_sha256'] = digest(ROOT/'protocol/SOURCE_SNAPSHOT.tar.gz')
    assert not (ROOT/'TEST_ACCESS_LOG.jsonl').read_text().strip()
    h=seal(path,spec)
    seal(ROOT/'protocol/SEAL.json',dict(sha256=h,created=now(),test_access_rows_before_freeze=0))
    atomic(ROOT/'protocol/FROZEN_POC_SPEC.md', ('# Frozen PoC v1\n\nSHA256: '+h+'\n\n```json\n'+json.dumps(spec,indent=2)+'\n```\n').encode())
    jobs={f'{core}-q{q}-s0-{method}':dict(status='PLANNED') for core in CORES for q in (8,12,16) for method in ('main','shuffled')}
    jobs.update({f'PC-{core}-s0':dict(status='PLANNED') for core in CORES})
    s=read(ROOT/'RUN_STATE.json');s['jobs']=jobs;write(ROOT/'RUN_STATE.json',s)
    write(ROOT/'RUN_MANIFEST.json',dict(run_id=spec['run_id'],created=spec['created'],protocol_sha256=h,
          resume_command='.venv/bin/python -m diffusion_hash_inv.poc --run',planned_hash_jobs=24,
          conditional_additional_jobs_max=32,scientific_test_access='only guarded test evaluation after selection seal'))
    state('P0',status='COMPLETED',spec_sha256=h)


def check_spec(spec):
    if spec['qs'] != [8,12,16] or spec['q20_enabled'] or spec['quotas'] != [10000,1000,1000]:
        raise ValueError('fixed matrix/quota')
    if len(set(spec['seeds'].values())) != len(spec['seeds']):
        raise ValueError('seed namespaces collide')
    if set(spec['cores']) != set(CORES) or spec['budget']['attempts'] != 100:
        raise ValueError('core matrix/budget')
    if 'TBD' in json.dumps(spec):
        raise ValueError('unresolved specification')
    for family in ('gaussian','discrete'):
        c=spec[family]
        if c['updates'] != [1500] or c['trial_cap_per_source_q'] != 1 or c['batch']!=16:
            raise ValueError('unregistered search')


def load_spec():
    spec=read(ROOT/'protocol/FROZEN_POC_SPEC.json')
    if digest(ROOT/'protocol/FROZEN_POC_SPEC.json') != read(ROOT/'protocol/SEAL.json')['sha256']:
        raise RuntimeError('frozen protocol integrity failure')
    check_spec(spec);verify_code(spec)
    return spec


def bundle(source, family, cfg, device, *, positive=False):
    if family=='gaussian':
        encoder,decoder,shape=BGVEncoder(),BGVDecoder(),(2,32,128)
        model=ImageUNet(2,8192 if positive else 259,width=cfg['width'],condition_shape=shape if positive else None)
        diffusion=GaussianDiffusion(cfg['steps'],beta_start=cfg['beta_start'],beta_end=cfg['beta_end'],prediction_type='sample',device=device)
    else:
        encoder=decoder=TokenCodec(source,31);shape=(32,)
        model=SequenceDenoiser(encoder.vocabulary_size,32,288 if positive else 259,width=cfg['width'],embedding_dim=cfg['embedding_dim'])
        diffusion=MaskedDiffusion(encoder.mask,cfg['grid'],device=device)
    return model.to(device),diffusion,encoder,decoder,shape


def hash_condition(q, prefix):
    # Only a q-bit prefix can cross this generator boundary.
    if not 1<=q<=128 or not 0<=int(prefix,16)<2**q:
        raise ValueError('noncanonical prefix')
    return digest_condition('md5',q,(int(prefix,16)<<(128-q)).to_bytes(16,'big'))


def clean_condition(values, family):
    if family=='gaussian':
        return ((values+1)/2).flatten(1)
    shifts=torch.arange(8,-1,-1,device=values.device)
    return ((values[:,:,None] >> shifts)&1).float().flatten(1)


def sample(model,diffusion,conditions,shape,cfg,generator):
    kwargs=dict(sampling_steps=cfg['sampling_steps'],generator=generator)
    if isinstance(diffusion,MaskedDiffusion):
        kwargs['temperature']=cfg['temperature']
    model.eval()
    return diffusion.sample(model,conditions,shape,**kwargs)


def train(folder, model, diffusion, values, conditions, cfg, model_seed, spec, *, callback=None):
    folder.mkdir(parents=True,exist_ok=True)
    identity=dict(cfg=cfg,model_seed=model_seed,protocol=digest(ROOT/'protocol/FROZEN_POC_SPEC.json'),
                  values_sha256=hashlib.sha256(values.cpu().numpy().tobytes()).hexdigest(),
                  conditions_sha256=hashlib.sha256(conditions.cpu().numpy().tobytes()).hexdigest())
    seal(folder/'identity.json',identity)
    device=values.device
    generator=torch.Generator(device=device).manual_seed(seed(spec['seeds']['training'],model_seed))
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    start=0; elapsed=0.;last_loss=None
    checkpoint=folder/'training_resume.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=True)
        if saved['identity']!=identity:raise RuntimeError('training identity mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['rng'].cpu());torch.set_rng_state(saved['global_rng'].cpu())
        start=saved['step'];elapsed=saved['elapsed'];last_loss=saved['loss']
    if (folder/'COMPLETE.json').exists():
        done=read(folder/'COMPLETE.json')
        if digest(checkpoint)!=done['checkpoint_sha256']:raise RuntimeError('checkpoint checksum')
        return done
    maximum=max(cfg['updates']);wall=time.perf_counter()
    for step in range(start+1,maximum+1):
        model.train()
        indices=torch.randint(len(values),(cfg['batch'],),device=device,generator=generator)
        loss=diffusion.loss(model,values[indices],conditions[indices],generator=generator)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%cfg['checkpoint_interval']==0 or step==maximum:
            sync(device);last_loss=float(loss.item());elapsed+=time.perf_counter()-wall
            save_torch(checkpoint,dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
                       rng=generator.get_state(),global_rng=torch.get_rng_state(),step=step,loss=last_loss,elapsed=elapsed))
            with (folder/'training.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=step,loss=last_loss,elapsed=elapsed,time=now()))+'\n')
            if callback:callback(step)
            print(json.dumps(dict(job=str(folder),update=step,loss=last_loss,seconds=elapsed)),flush=True)
            wall=time.perf_counter()
    done=dict(status='COMPLETED',updates=maximum,loss=last_loss,seconds=elapsed,parameter_count=parameter_count(model),
              checkpoint_sha256=digest(checkpoint),checkpoint=str(checkpoint),config_id=cfg['id'],
              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              mps_driver_bytes=torch.mps.driver_allocated_memory() if device.type=='mps' else None)
    write(folder/'COMPLETE.json',done)
    return done


def dataset(spec,source,q):
    folder=ROOT/'datasets'/f'{source}-q{q}'
    if not (folder/'manifest.json').exists():
        ds_seed=seed(spec['seeds']['dataset'],source,q);sp_seed=seed(spec['seeds']['split'],source,q)
        parts,construction=quota_digest_split(SourceSpec(source,12000,seed=ds_seed),algorithm='md5',q=q,
            split_seed=sp_seed,quotas=tuple(spec['quotas']),max_draws=spec['max_draws'])
        folder.mkdir(parents=True,exist_ok=True)
        hashes={}
        for split,records in parts.items():
            path=folder/f'{split}.jsonl'
            atomic(path,(''.join(json.dumps(r.to_json(split),sort_keys=True)+'\n' for r in records)).encode())
            hashes[split]=digest(path)
        audit=split_validation_report(parts)
        assert audit['passed'] and list(audit['record_counts'].values())==spec['quotas']
        representatives={split:[r.to_json(split) for r in select_digest_representatives(records,
            seed=seed(spec['seeds']['representative'],source,q))] for split,records in parts.items()}
        seal(folder/'representatives.json',representatives)
        seal(folder/'manifest.json',dict(source=source,q=q,dataset_seed=ds_seed,split_seed=sp_seed,
             construction=construction,audit=audit,sha256=hashes,representatives_sha256=digest(folder/'representatives.json'),
             length_counts={s:dict(Counter(len(r.message) for r in rs)) for s,rs in parts.items()},
             protocol_sha256=digest(ROOT/'protocol/FROZEN_POC_SPEC.json')))
        access('construction_and_G0_audit',source,q,'No model output/selection; evaluator-only split integrity')
    manifest=read(folder/'manifest.json')
    for split,h in manifest['sha256'].items():
        if digest(folder/f'{split}.jsonl')!=h:raise RuntimeError('dataset integrity failure')
    if digest(folder/'representatives.json')!=manifest['representatives_sha256']:raise RuntimeError('representative integrity')
    return folder,manifest


def records_from(path):
    return tuple(DigestRecord(row['id'],row['source'],bytes.fromhex(row['message_hex']),row['algorithm'],row['q'],bytes.fromhex(row['full_digest']))
                 for row in (json.loads(line) for line in Path(path).read_text().splitlines()))


def access(kind,source,q,reason):
    if not (ROOT/'protocol/SEAL.json').exists():raise RuntimeError('test access before freeze')
    with (ROOT/'TEST_ACCESS_LOG.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=now(),kind=kind,source=source,q=q,reason=reason,
                   protocol_sha256=digest(ROOT/'protocol/FROZEN_POC_SPEC.json')))+'\n')
        f.flush();os.fsync(f.fileno())


def audit_dataset(folder):
    parts={s:records_from(folder/f'{s}.jsonl') for s in ('train','validation','test')}
    a=split_validation_report(parts)
    if not a['passed'] or list(a['record_counts'].values())!=[10000,1000,1000]:
        raise RuntimeError('G0 overlap/quota failure')
    for records in parts.values():
        for r in records:
            if hashlib.md5(r.message).digest()!=r.digest:raise RuntimeError('dataset independent digest failure')
    return a


def decode(raw,decoder,family,source):
    decoded=decoder.decode((raw+1)/2) if family=='gaussian' else decoder.decode(raw)
    if decoded.valid and (not 4<=len(decoded.message)<=31 or
       source=='printable' and any(not 33<=v<=126 for v in decoded.message)):
        return DecodeResult(decoded.message,False,decoded.length,'source_domain')
    return decoded


def edit_distance(a,b):
    # Exact Myers bit-vector Levenshtein: keeps per-candidate bit diagnostics tractable.
    if not a:return len(b)
    masks={}
    for i,value in enumerate(a):masks[value]=masks.get(value,0)|(1<<i)
    positive,negative,score,last=~0,0,len(a),1<<(len(a)-1)
    for value in b:
        equal=masks.get(value,0);vertical=equal|negative
        horizontal=(((equal&positive)+positive)^positive)|equal
        plus=negative|~(horizontal|positive);minus=positive&horizontal
        score+=bool(plus&last)-bool(minus&last)
        plus=(plus<<1)|1;minus<<=1
        positive=minus|~(vertical|plus);negative=plus&vertical
    return score


class Ledger:
    """Durable attempt rows. A primary key is a target/attempt, not successful bytes."""
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS attempts (target TEXT, attempt INTEGER, row TEXT NOT NULL, PRIMARY KEY(target,attempt))')
        self.db.commit()

    def rows(self,target=None):
        query='SELECT row FROM attempts'+(' WHERE target=?' if target is not None else '')+' ORDER BY target,attempt'
        return [json.loads(r[0]) for r in self.db.execute(query,() if target is None else (target,))]

    def put(self,row):
        with self.db:
            self.db.execute('INSERT INTO attempts VALUES (?,?,?)',(row['target_prefix'],row['attempt_index'],json.dumps(row,allow_nan=False)))

    def close(self):
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');self.db.close()


def evaluate_raw(raw, target, decoder, family, *, index, seen, metadata):
    started=time.perf_counter()
    decoded=decode(raw,decoder,family,target.source) if family!='random' else DecodeResult(raw,True,len(raw),None)
    message=decoded.message
    if decoded.valid and (message is None or not 4<=len(message)<=31 or
        target.source=='printable' and any(not 33<=v<=126 for v in message)):
        decoded=DecodeResult(message,False,len(message) if message is not None else None,'source_domain')
    checked=verify_candidate(message,target) if message is not None else None
    # Repeated malformed raw outputs are also flagged; distinct invalids are not all one duplicate.
    key=message.hex() if message is not None else 'raw:'+hashlib.sha256(raw.numpy().tobytes()).hexdigest()
    duplicate=key in seen;seen.add(key)
    valid=bool(decoded.valid)
    row=dict(metadata,run_id='poc_md5_truncated_v1',source=target.source,q=target.q,
         target_id=f'{target.source}:md5:{target.q}:{target.prefix}',target_prefix=target.prefix,
         attempt_index=index,decoded_hex=message.hex() if message is not None else None,
         candidate_length=len(message) if message is not None else None,validity=valid,
         invalid_reason=decoded.reason,duplicate=duplicate,duplicate_key=key,actual_hash_call=message is not None,
         full_md5=checked.digest if checked else None,
         candidate_prefix=digest_prefix_hex(bytes.fromhex(checked.digest),target.q) if checked else None,
         prefix_match=bool(checked and checked.prefix_match),match=bool(valid and checked and checked.prefix_match),
         exact_source=bool(valid and message==target.message),length_match=bool(valid and len(message)==len(target.message)),
         byte_error=None,bit_error=None,character_error=None,timestamp=now())
    if valid:
        byte_error=edit_distance(message,target.message)/len(target.message)
        bits=lambda x: ''.join(f'{b:08b}' for b in x)
        row.update(byte_error=byte_error,character_error=byte_error if target.source=='printable' else None,
                   bit_error=edit_distance(bits(message),bits(target.message))/(8*len(target.message)))
    if family=='discrete':
        row['eos_validity']=tuple(raw.shape)==(32,) and int((raw==decoder.eos).sum())==1
        row['format_validity']=decoded.valid
    row['evaluation_seconds']=time.perf_counter()-started
    return row


def generate_stream(folder, targets, source, family, spec, *, model=None,diffusion=None,decoder=None,
                    shape=None,cfg=None,conditions=None,core='SHARED',q=8,model_seed=0,method='random',split='test',k=100):
    folder.mkdir(parents=True,exist_ok=True)
    identity=dict(target_order=[t.prefix for t in targets],core=core,q=q,seed=model_seed,method=method,
         split=split,k=k,protocol_sha256=digest(ROOT/'protocol/FROZEN_POC_SPEC.json'),
         config_id=cfg['id'] if cfg else 'source-prior',
         checkpoint_id=read(folder/'checkpoint.json')['sha256'] if (folder/'checkpoint.json').exists() else 'validation-training-checkpoint' if model is not None else 'source-prior',
         condition_sha256=hashlib.sha256(conditions.cpu().numpy().tobytes()).hexdigest() if conditions is not None else None)
    seal(folder/'stream_identity.json',identity)
    ledger=Ledger(folder/'candidates.sqlite')
    try:
        for target_index,target in enumerate(targets):
            existing=ledger.rows(target.prefix)
            if [r['attempt_index'] for r in existing]!=list(range(1,len(existing)+1)) or len(existing)>k:
                raise RuntimeError('attempt ordering/budget corruption')
            if len(existing)==k:
                raw_path=Path(existing[0]['raw_output'])
                if digest(raw_path)!=existing[0]['raw_sha256']:raise RuntimeError('completed raw artifact corrupted')
                continue
            rng_seed=seed(spec['seeds']['baseline'] if family=='random' else spec['seeds']['sampling'],
                          source,q,target.prefix) if family=='random' else seed(spec['seeds']['sampling'],split,core,q,model_seed,method,target.prefix)
            raw_path=folder/'raw'/f'{target.prefix}.npz'
            if raw_path.exists():
                with np.load(raw_path,allow_pickle=False) as z:
                    raw_values=z['values'];generation_seconds=float(z['seconds']);stored_seed=int(z['seed'])
                if stored_seed!=rng_seed or len(raw_values)!=k:raise RuntimeError('raw identity mismatch')
            else:
                started=time.perf_counter()
                if family=='random':
                    rng=random.Random(rng_seed)
                    messages=[_sample_message(rng,source,min_length=4,max_length=31,length=None) for _ in range(k)]
                    raw_values=np.zeros((k,32),dtype=np.uint8)
                    for i,x in enumerate(messages):raw_values[i,0]=len(x);raw_values[i,1:len(x)+1]=list(x)
                else:
                    device=next(model.parameters()).device
                    cond=conditions[target_index:target_index+1].expand(k,-1)
                    outputs=sample(model,diffusion,cond,shape,cfg,torch.Generator(device=device).manual_seed(rng_seed))
                    sync(device)
                    raw_values=outputs.cpu().numpy().astype(np.float32 if family=='gaussian' else np.uint16)
                generation_seconds=time.perf_counter()-started
                buffer=io.BytesIO();np.savez_compressed(buffer,values=raw_values,seconds=generation_seconds,seed=np.int64(rng_seed))
                atomic(raw_path,buffer.getvalue())
            raw_hash=digest(raw_path)
            seen={r['duplicate_key'] for r in existing}
            for i in range(len(existing),k):
                raw=bytes(raw_values[i,1:int(raw_values[i,0])+1]) if family=='random' else torch.from_numpy(raw_values[i].copy()).to(torch.float32 if family=='gaussian' else torch.long)
                row=evaluate_raw(raw,target,decoder,family,index=i+1,seen=seen,metadata=dict(
                    phase='P1' if split=='validation' else 'P2' if q==8 else 'P3' if model_seed==0 else 'P4',
                    core_id=core,model_family=family,model_seed=None if family=='random' else model_seed,
                    sampling_seed=rng_seed,method=method,raw_output=str(raw_path),raw_sha256=raw_hash,raw_index=i,
                    config_id=cfg['id'] if cfg else 'source-prior',checkpoint_id=identity.get('checkpoint_id'),
                    generation_seconds=generation_seconds/k,nfe=0 if family=='random' else cfg['sampling_steps'],
                    resume_provenance='durable raw batch + attempt transaction'))
                ledger.put(row)
            if target_index%25==0:print(json.dumps(dict(stream=str(folder),target=target_index+1,N=len(targets))),flush=True)
        rows=ledger.rows()
        if len(rows)!=len(targets)*k:raise RuntimeError('G2 budget mismatch')
        summary=aggregate(rows,[t.prefix for t in targets],k)
        write(folder/'metrics.json',summary)
        atomic(folder/'candidates.jsonl',(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)).encode())
        write(folder/'COMPLETE.json',dict(status='COMPLETED',candidate_count=len(rows),k=k,
            ledger_jsonl_sha256=digest(folder/'candidates.jsonl'),metrics_sha256=digest(folder/'metrics.json'),target_count=len(targets)))
        return summary
    finally:
        ledger.close()


def aggregate(rows,target_order,k=100):
    grouped={key:[] for key in target_order}
    for r in rows:grouped[r['target_prefix']].append(r)
    if any([r['attempt_index'] for r in rs]!=list(range(1,k+1)) for rs in grouped.values()):
        raise ValueError('incomplete or reordered stream')
    result=dict(status='PASS',N=len(target_order),candidate_count=len(rows),actual_hash_calls=sum(r['actual_hash_call'] for r in rows),
                target_order=target_order,ks={},nfe=sum(r['nfe'] for r in rows),
                generation_seconds=sum(r['generation_seconds'] for r in rows),evaluation_seconds=sum(r['evaluation_seconds'] for r in rows))
    for cutoff in [v for v in (1,10,100) if v<=k]:
        prefix=[r for rs in grouped.values() for r in rs[:cutoff]]
        outcomes=[any(r['match'] for r in grouped[t][:cutoff]) for t in target_order]
        valid=[r for r in prefix if r['validity']];n=len(outcomes)
        result['ks'][str(cutoff)]=dict(successes=sum(outcomes),rate=sum(outcomes)/n,outcomes=outcomes,
            N=n,valid_decode_rate=len(valid)/len(prefix),invalid_rate=1-len(valid)/len(prefix),
            duplicate_rate=sum(r['duplicate'] for r in prefix)/len(prefix),
            exact_source_recovery=sum(any(r['exact_source'] for r in grouped[t][:cutoff]) for t in target_order)/n,
            length_match_rate=sum(r['length_match'] for r in prefix)/len(prefix),
            length_histogram=dict(Counter(r['candidate_length'] for r in prefix if r['candidate_length'] is not None)),
            invalid_reasons=dict(Counter(r['invalid_reason'] for r in prefix if not r['validity'])),
            error_valid_count=len(valid),
            **{name:sum(r[name] for r in valid if r[name] is not None)/sum(r[name] is not None for r in valid)
                 if any(r[name] is not None for r in valid) else None for name in ('byte_error','bit_error','character_error')},
            eos_validity_rate=sum(r.get('eos_validity',False) for r in prefix)/len(prefix) if 'eos_validity' in prefix[0] else None,
            format_validity_rate=sum(r.get('format_validity',False) for r in prefix)/len(prefix) if 'format_validity' in prefix[0] else None,
            zero_success_upper95=1-.05**(1/n) if not any(outcomes) else None)
    rates=[r['rate'] for r in result['ks'].values()]
    if rates!=sorted(rates):raise RuntimeError('prefix monotonicity')
    return result


def positive_control(spec,core,model_seed=0):
    source,family=CORES[core];cfg=spec[family];job=f'PC-{core}-s{model_seed}';folder=ROOT/'controls'/job
    if (folder/'result.json').exists():
        result=read(folder/'result.json')
        if digest(folder/'training_resume.pt')!=result['checkpoint_sha256']:raise RuntimeError('control checkpoint integrity')
        return result
    state(job=job,status='RUNNING')
    device=torch.device(spec['resources']['device']);torch.manual_seed(model_seed)
    model,diff,encoder,decoder,shape=bundle(source,family,cfg,device,positive=True)
    rng=random.Random(seed(spec['seeds']['control_data'],source));messages=[]
    while len(messages)<96:
        value=_sample_message(rng,source,min_length=4,max_length=31,length=None)
        if value not in messages:messages.append(value)
    seal(folder/'corpus.json',dict(train=[x.hex() for x in messages[:64]],validation=[x.hex() for x in messages[64:80]],test=[x.hex() for x in messages[80:]]))
    values=torch.stack([encoder.encode(x) for x in messages]).to(device)
    if family=='gaussian':values=values*2-1
    cond=clean_condition(values,family)
    def evaluate(start,stop,sampling_seeds,stage):
        rows=[]
        for ss in sampling_seeds:
            path=folder/f'{stage}-sampling{ss}.npz'
            if path.exists():
                with np.load(path) as z:outputs=torch.from_numpy(z['values'])
            else:
                sample_started=time.perf_counter()
                outputs=sample(model,diff,cond[start:stop],shape,cfg,torch.Generator(device=device).manual_seed(ss)).cpu()
                b=io.BytesIO();np.savez_compressed(b,values=outputs.numpy(),seconds=time.perf_counter()-sample_started);atomic(path,b.getvalue())
            for index,value in enumerate(outputs,start):
                decoded=decode(value,decoder,family,source)
                rows.append(dict(target=index-start,sampling_seed=ss,valid=decoded.valid,reason=decoded.reason,
                    exact=bool(decoded.valid and decoded.message==messages[index]),target_hex=messages[index].hex(),
                    decoded_hex=decoded.message.hex() if decoded.message is not None else None,raw=str(path)))
        return rows
    def monitor(step):
        rows=evaluate(64,80,[0],f'validation-u{step}')
        write(folder/f'validation-u{step}.json',dict(rows=rows,rate=sum(r['exact'] for r in rows)/16))
    done=train(folder,model,diff,values[:64],cond[:64],cfg,model_seed,spec,callback=monitor)
    rows=evaluate(80,96,[0,1,2],'heldout-final')
    clusters=np.array([sum(r['exact'] for r in rows if r['target']==i)/3 for i in range(16)])
    rng=np.random.default_rng(seed(spec['seeds']['bootstrap'],'PC',core,model_seed))
    samples=np.sort(clusters[rng.integers(16,size=(10000,16))].mean(1))
    rate=float(clusters.mean())
    result=dict(status='PASS' if rate>=.99 else 'FAIL',core=core,source=source,family=family,
          model_seed=model_seed,config_id=cfg['id'],N=16,attempts=48,exact_successes=sum(r['exact'] for r in rows),rate=rate,
          ci95=[float(samples[int(.025*9999)]),float(samples[int(.975*9999)])],
          per_sampling_seed={str(ss):dict(successes=sum(r['exact'] for r in rows if r['sampling_seed']==ss),
               rate=sum(r['exact'] for r in rows if r['sampling_seed']==ss)/16,
               wilson_ci95=binomial_ci95(sum(r['exact'] for r in rows if r['sampling_seed']==ss),16)) for ss in (0,1,2)},
          rows=rows,checkpoint_sha256=done['checkpoint_sha256'],training=done,
          scope='actual trained reversible model; no hash advantage; 16 clusters not48 independent targets',
          all_success_small_n_note='Empirical cluster bootstrap can be degenerate; per-repetition Wilson intervals report small-N uncertainty')
    write(folder/'result.json',result)
    state(job=job,status=result['status'],exact=rate,N=16,artifact=str(folder/'result.json'))
    return result


def pipeline_validation(spec):
    path=ROOT/'gates/PIPELINE.json'
    if path.exists():return read(path)
    started=time.perf_counter();counts={}
    for source in ('printable','random_bytes'):
        alphabet=range(33,127) if source=='printable' else range(256)
        enc,dec=BGVEncoder(),BGVDecoder();token=TokenCodec(source,31)
        corpus=[bytes([s])*n for s in alphabet for n in (4,17,31)]
        corpus += [bytes(list(alphabet)[i%len(alphabet)] for i in range(n)) for n in (4,9,31)]
        for message in corpus:
            assert dec.decode(enc.encode(message)).message==message
            assert token.decode(token.encode(message)).message==message
        counts[source]=dict(messages=len(corpus),bgv_roundtrip=1.,token_roundtrip=1.)
    from .hashing.md5 import MD5Tracer
    tracer=MD5Tracer();verifier=[]
    for source in ('printable','random_bytes'):
        checks=messages=0
        alphabet=range(33,127) if source=='printable' else range(256)
        for length in (1,2):
            for symbols in product(alphabet,repeat=length):
                x=bytes(symbols);d=hashlib.md5(x).digest()
                if tracer.trace(x)['digest']!=d.hex():raise RuntimeError('independent MD5 implementation mismatch')
                other=hashlib.md5(x+b'\0').digest()
                for q in (8,12,16,128):
                    ref=''.join(f'{v:08b}' for v in d)[:q]
                    r=DigestRecord(0,source,x,'md5',q,d)
                    assert int(r.prefix,16)==int(ref,2)
                    assert verify_candidate(x,r).prefix_match
                    alt=replace(r,digest=other)
                    assert verify_candidate(x,alt).prefix_match==(ref==''.join(f'{v:08b}' for v in other)[:q])
                    checks+=2
                messages+=1
        verifier.append(dict(source=source,messages=messages,verdict_checks=checks,status='PASS'))
    result=dict(status='PASS',codecs=counts,verifier=verifier,seconds=time.perf_counter()-started,
                hash_calls=sum(v['messages']*10 for v in verifier),
                leakage_and_budget='tests/test_poc.py; full pytest log includes negative cases, exact resume, continuous t and metadata isolation',
                learned_controls='separate actual model gates; this alone does not grant G1')
    write(path,result)
    return result


def trained_hash(spec,core,q,model_seed,method):
    source,family=CORES[core];cfg=spec[family];folder,_=dataset(spec,source,q)
    records=records_from(folder/'train.jsonl')
    torch.manual_seed(model_seed);device=torch.device(spec['resources']['device'])
    model,diff,encoder,decoder,shape=bundle(source,family,cfg,device)
    values=torch.stack([encoder.encode(r.message) for r in records]).to(device)
    if family=='gaussian':values=values*2-1
    conditions=torch.stack([hash_condition(q,r.prefix) for r in records]).to(device)
    job=f'{core}-q{q}-s{model_seed}-{method}';out=ROOT/'models'/job
    if method=='shuffled':
        donors=shuffled_donors(records,seed=seed(spec['seeds']['shuffle'],source,q,'train'),same_length=False)
        seal(out/'train_donors.json',[dict(recipient=r.id,recipient_prefix=r.prefix,donor=records[d].id,donor_prefix=records[d].prefix) for r,d in zip(records,donors)])
        conditions=conditions[donors]
    state(job=job,status='RUNNING',stage='training',artifact=str(out))
    done=train(out,model,diff,values,conditions,cfg,model_seed,spec)
    del values,conditions
    return model,diff,decoder,shape,cfg,out,done


def evaluation_conditions(spec,records,source,q,split,method,folder,device):
    conditions=torch.stack([hash_condition(q,r.prefix) for r in records]).to(device)
    if method=='shuffled':
        donors=shuffled_donors(records,seed=seed(spec['seeds']['shuffle'],source,q,split),same_length=False)
        seal(folder/f'{split}_donors.json',[dict(recipient=r.id,recipient_prefix=r.prefix,donor=records[d].id,donor_prefix=records[d].prefix) for r,d in zip(records,donors)])
        conditions=conditions[donors]
    return conditions


def validation_search(spec,controls):
    selected={}
    for core,(source,family) in CORES.items():
        for q in spec['qs']:
            key=f'{core}-q{q}'
            if controls[core]['status']!='PASS':
                selected[key]=dict(status='BLOCKED',reason='G1-B actual-model control failed',config_id=spec[family]['id'])
                continue
            result_path=ROOT/'search'/key/'selected.json'
            if result_path.exists():selected[key]=read(result_path);continue
            seal(ROOT/'search'/key/'TRIAL_REGISTRATION.json',dict(config_id=spec[family]['id'],trial_index=1,cap=1,update=1500,registered_before_training=True))
            model,diff,decoder,shape,cfg,folder,done=trained_hash(spec,core,q,0,'main')
            data,_=dataset(spec,source,q)
            targets=select_digest_representatives(records_from(data/'validation.jsonl'),seed=seed(spec['seeds']['representative'],source,q))
            cond=evaluation_conditions(spec,targets,source,q,'validation','main',folder,next(model.parameters()).device)
            seal(ROOT/'search'/key/'validation/checkpoint.json',dict(path=done['checkpoint'],sha256=done['checkpoint_sha256']))
            metric=generate_stream(ROOT/'search'/key/'validation',targets,source,family,spec,model=model,diffusion=diff,
                decoder=decoder,shape=shape,cfg=cfg,conditions=cond,core=core,q=q,model_seed=0,method='main',split='validation')
            result=dict(status='COMPLETED',config_id=cfg['id'],updates=1500,sampling_steps=cfg['sampling_steps'],
                trial_cap=1,trials=[dict(id=cfg['id'],status='COMPLETED',validation_N=len(targets),
                preimage_success_at100=metric['ks']['100']['rate'],checkpoint=done['checkpoint'],checkpoint_sha256=done['checkpoint_sha256'])],
                selection_reason='single preregistered eligible trial; ties follow frozen steps/config/update order')
            seal(result_path,result);selected[key]=result
            del model,diff
    seal(ROOT/'protocol/SELECTION_SEAL.json',dict(selected=selected,protocol_sha256=digest(ROOT/'protocol/FROZEN_POC_SPEC.json')))
    return selected


def monte_carlo(spec):
    for source in ('printable','random_bytes'):
        path=ROOT/'baseline'/f'MC-{source}.json'
        if path.exists():continue
        started=time.perf_counter();rng=random.Random(seed(spec['seeds']['monte_carlo'],source))
        hist=Counter();lengths=Counter();draws=spec['baseline']['mc_draws_per_source']
        for _ in range(draws):
            x=_sample_message(rng,source,min_length=4,max_length=31,length=None)
            hist[int.from_bytes(hashlib.md5(x).digest()[:2],'big')]+=1;lengths[len(x)]+=1
        write(path,dict(source=source,draws=draws,actual_hash_calls=draws,
            seed=seed(spec['seeds']['monte_carlo'],source),histogram16=dict(hist),length_histogram=dict(lengths),seconds=time.perf_counter()-started,
            diagnostic='separate independent draws, never primary candidates'))


def baseline(spec,source,q,targets):
    path=ROOT/'baseline'/f'{source}-q{q}'
    result=generate_stream(path,targets,source,'random',spec,q=q,method='random')
    hist=read(ROOT/'baseline'/f'MC-{source}.json');draws=hist['draws']
    counts=Counter()
    for key,count in hist['histogram16'].items():counts[int(key)>>(16-q)]+=count
    rows=[]
    for target in targets:
        hits=counts[int(target.prefix,16)];ci=binomial_ci95(hits,draws)
        rows.append(dict(target_prefix=target.prefix,hits=hits,pi=hits/draws,pi_ci95=ci,zero_hits_resolution_limited=hits==0,
            expectation={str(k):1-(1-hits/draws)**k for k in (1,10,100)},
            expectation_ci95={str(k):[1-(1-p)**k for p in ci] for k in (1,10,100)}))
    write(path/'MC_DIAGNOSTIC.json',dict(N=len(targets),rows=rows,draws=draws,actual_hash_calls=0,
        mean_expectation={str(k):sum(r['expectation'][str(k)] for r in rows)/len(rows) for k in (1,10,100)},
        note='Wilson target intervals; mean expectation has no simultaneous interval; shared histogram cost counted once per source'))
    return result


def test_setting(spec,core,q,model_seed,controls):
    source,family=CORES[core]
    if controls[core]['status']!='PASS':raise RuntimeError('G1 guard')
    if not (ROOT/'protocol/SELECTION_SEAL.json').exists():raise RuntimeError('selection not sealed')
    selected=read(ROOT/'protocol/SELECTION_SEAL.json')['selected'][f'{core}-q{q}']
    if selected['status']!='COMPLETED':raise RuntimeError('selection blocked')
    data,manifest=dataset(spec,source,q)
    audit=audit_dataset(data)
    write(ROOT/'gates'/f'G0-{core}-q{q}-s{model_seed}.json',dict(status='PASS',audit=audit,pretest_seal=True,validation_only=True))
    access('hash_test_evaluation',source,q,f'{core} seed{model_seed}; all source/q selections sealed; G0/G1 passed')
    targets=select_digest_representatives(records_from(data/'test.jsonl'),seed=seed(spec['seeds']['representative'],source,q))
    results={'random':baseline(spec,source,q,targets)}
    for method in ('main','shuffled'):
        model,diff,decoder,shape,cfg,folder,done=trained_hash(spec,core,q,model_seed,method)
        cond=evaluation_conditions(spec,targets,source,q,'test',method,folder,next(model.parameters()).device)
        # Save split-local validation donor map too; never optimize the shuffled model.
        val=select_digest_representatives(records_from(data/'validation.jsonl'),seed=seed(spec['seeds']['representative'],source,q))
        evaluation_conditions(spec,val,source,q,'validation',method,folder,next(model.parameters()).device)
        output=ROOT/'evaluation'/f'{core}-q{q}-s{model_seed}-{method}'
        seal(output/'checkpoint.json',dict(path=done['checkpoint'],sha256=done['checkpoint_sha256'],config_id=cfg['id']))
        results[method]=generate_stream(output,targets,source,family,spec,model=model,diffusion=diff,decoder=decoder,
                  shape=shape,cfg=cfg,conditions=cond,core=core,q=q,model_seed=model_seed,method=method)
        state(job=f'{core}-q{q}-s{model_seed}-{method}',status='COMPLETED',G2='PASS',N=len(targets),artifact=str(output))
        del model,diff
    compared=[]
    for k in (1,10,100):
        main=results['main']['ks'][str(k)]
        for method in ('random','shuffled'):
            other=results[method]['ks'][str(k)]
            if results['main']['target_order']!=results[method]['target_order']:raise RuntimeError('unpaired target order')
            pair=asdict(paired_comparison(main['outcomes'],other['outcomes'],bootstrap_seed=seed(spec['seeds']['bootstrap'],core,q,model_seed,k,method)))
            compared.append(dict(core=core,source=source,q=q,seed=model_seed,k=k,comparator=method,
                comparison_id=f'{core}-q{q}-s{model_seed}-k{k}-{method}',**pair,holm_adjusted_pvalue=None))
    result=dict(core=core,source=source,q=q,seed=model_seed,N=len(targets),G0='PASS',G1='PASS',G2='PASS',
          metrics=results,comparisons=compared,
          eligible=q in (12,16) and all(r['absolute_gain']>0 for r in compared if r['k']==100))
    write(ROOT/'results'/f'{core}-q{q}-s{model_seed}.json',result)
    return result


def report(spec,controls):
    results=[read(p) for p in sorted((ROOT/'results').glob('*-q*-s*.json'))] if (ROOT/'results').exists() else []
    rows=[r for result in results for r in result['comparisons']]
    for name,ks,expected in [('primary',[100],16),('auxiliary',[1,10],32)]:
        family=[r for r in rows if r['q'] in (12,16) and r['seed']==0 and r['k'] in ks]
        adjustment=holm_adjust({r['comparison_id']:r['mcnemar_pvalue'] for r in family}) if len(family)==expected else {}
        for r in family:r.update(family_id=name,holm_adjusted_pvalue=adjustment.get(r['comparison_id']),family_complete=len(family)==expected)
    matrix=[]
    for core,(source,family) in CORES.items():
        for q in (8,12,16):
            found=[r for r in results if r['core']==core and r['q']==q]
            seed0=next((r for r in found if r['seed']==0),None)
            blocked=controls.get(core,{}).get('status')=='FAIL'
            level='none'
            if seed0:
                level='P1' if seed0['eligible'] else 'P0'
                if seed0['eligible'] and {r['seed'] for r in found}=={0,1,2} and all(r['eligible'] for r in found):level='P2'
            primary=[r for r in rows if r['core']==core and r['q']==q and r['seed']==0 and r['k']==100]
            signal=len(primary)==2 and all(r['holm_adjusted_pvalue'] is not None and r['holm_adjusted_pvalue']<.05 and r['delta_ci95'][0]>0 for r in primary)
            matrix.append(dict(core=core,source=source,q=q,status='COMPLETED' if seed0 else 'BLOCKED' if blocked else 'INCOMPLETE',
                G0='PASS' if (ROOT/'datasets'/f'{source}-q{q}'/'manifest.json').exists() else 'NOT_EVALUATED',
                G1=controls.get(core,{}).get('status','NOT_EVALUATED'),G2='PASS' if seed0 else 'NOT_EVALUATED',
                G3='COMPLETED' if seed0 else 'NOT_RUN',G4='COMPLETED' if level=='P2' else 'NOT_RUN' if not seed0 or not seed0['eligible'] else 'COMPLETED' if len(found)==3 else 'INCOMPLETE',
                evidence=level,statistically_supported_signal=signal,model_seeds=[r['seed'] for r in found],
                reason='G1-B reversible control below .99' if blocked else None))
    state_data=read(ROOT/'RUN_STATE.json')
    sizes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    completions=[read(p) for p in (ROOT/'models').glob('*/COMPLETE.json')] if (ROOT/'models').exists() else []
    completions += [read(p) for p in (ROOT/'controls').glob('*/COMPLETE.json')]
    metric_paths=list((ROOT/'evaluation').glob('*/metrics.json')) if (ROOT/'evaluation').exists() else []
    metric_paths+=list((ROOT/'baseline').glob('*/metrics.json'))
    metric_paths+=list((ROOT/'search').glob('*/validation/metrics.json')) if (ROOT/'search').exists() else []
    metrics=[read(p) for p in metric_paths]
    resources=dict(training_jobs_completed=len(completions),training_jobs_planned_stage_a=28,tuning_trials=sum(1 for _ in (ROOT/'search').glob('*/TRIAL_REGISTRATION.json')),training_seconds=sum(r['seconds'] for r in completions),
         control_attempts=sum(48 for p in (ROOT/'controls').glob('*/result.json')),candidate_count=sum(m['candidate_count'] for m in metrics),actual_candidate_hash_calls=sum(m['actual_hash_calls'] for m in metrics),
         nfe=sum(m['nfe'] for m in metrics),sampling_seconds=sum(m['generation_seconds'] for m in metrics),
         evaluation_seconds=sum(m['evaluation_seconds'] for m in metrics),storage_bytes=sizes,
         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         monte_carlo_hash_calls=sum(read(p)['actual_hash_calls'] for p in (ROOT/'baseline').glob('MC-*.json')),
         verifier=read(ROOT/'gates/PIPELINE.json') if (ROOT/'gates/PIPELINE.json').exists() else None)
    final=dict(status='COMPLETED_WITH_BLOCKED_BRANCHES' if all(r['status'] in ('COMPLETED','BLOCKED') for r in matrix) and any(r['status']=='BLOCKED' for r in matrix) else 'COMPLETED' if all(r['status']=='COMPLETED' for r in matrix) else 'INCOMPLETE',
         updated=now(),matrix=matrix,comparisons=rows,positive_controls=controls,resources=resources,
         protocol_sha256=digest(ROOT/'protocol/FROZEN_POC_SPEC.json'),states=state_data)
    write(ROOT/'reports/FINAL_POC_RESULTS.json',final)
    columns=['core','source','q','seed','k','comparator','target_count','model_successes','baseline_successes','n10','n01','absolute_gain','delta_ci95','mcnemar_pvalue','holm_adjusted_pvalue','family_id','family_complete']
    f=io.StringIO();writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    atomic(ROOT/'reports/COMPARISONS.csv',f.getvalue().encode())
    lines=['# Truncated MD5 PoC final report', '',f'Updated: {now()}. Execution: **{final["status"]}**.',
      '','No legacy gate PASS is imported. Software checks and actual learned controls are separate.',
      '', '## Frozen protocol', '',f'Protocol SHA256 `{final["protocol_sha256"]}`. See ../protocol/FROZEN_POC_SPEC.json and SELECTION_SEAL.json for exact seeds, configurations, provenance and selection.',
      '','Pilot exact message quotas 10,000/1,000/1,000; unique digest targets are the evaluation unit. MD5 q8/12/16, uniform lengths4–31, iid ASCII94/bytes256. One100-attempt stream per target; K1/10/100 are prefixes. G01: width8 x0 DDIM50 beta_end.4; D01: global MLP128 embedding16 continuous masking and32-step random reveal. Adam1e-3, batch16,1500 updates. No repairs or test adaptation.',
      '','## Matrix and gates','','|Core|q|Status|G0|G1|G2|G3|G4|Evidence|Statistical signal|','|---|---:|---|---|---|---|---|---|---|---|']
    for r in matrix:lines.append('|'+ '|'.join(str(r[k]) for k in ('core','q','status','G0','G1','G2','G3','G4','evidence','statistically_supported_signal'))+'|')
    lines+=['','## Actual learned controls','','|Core|Seed|N clusters|Exact attempts/48|Rate|Cluster CI95|Gate|','|---|---:|---:|---:|---:|---|---|']
    for c,r in controls.items():lines.append(f'|{c}|{r.get("model_seed",0)}|16|{r.get("exact_successes")}|{r.get("rate")}|{r.get("ci95")}|{r["status"]}|')
    lines+=['','Repetitions use sampling seeds0/1/2 with K1 each; no best-of and no48-independent-target inference. Degenerate empirical bootstrap intervals are not proof of certainty; per-repetition Wilson intervals are in JSON.',
        '', '## K/q curves and paired results','','|Core|q|Seed|K|N|Main|Random|Shuffled|Delta random|Delta shuffled|','|---|---:|---:|---:|---:|---|---|---|---:|---:|']
    for r in results:
        for k in (1,10,100):
            m=[r['metrics'][x]['ks'][str(k)] for x in ('main','random','shuffled')]
            lines.append(f'|{r["core"]}|{r["q"]}|{r["seed"]}|{k}|{r["N"]}|'+ '|'.join(f'{x["successes"]}/{r["N"]} ({x["rate"]:.6f})' for x in m)+f'|{m[0]["rate"]-m[1]["rate"]:.6f}|{m[0]["rate"]-m[2]["rate"]:.6f}|')
    if not results:lines.append('\nNo hash experiment outcomes are available; do not interpret blocked branches as null effects.')
    lines+=['','All n10/n01, raw exact one-sided McNemar p, paired10,000-bootstrap CI and Holm p are in COMPARISONS.csv/FINAL_POC_RESULTS.json. Holm is null when the registered16/32-comparison family is incomplete. Missing comparisons were not dropped or imputed.',
       '', '## Validity, accounting and replication','',
       'Each evaluation/*/metrics.json retains invalid/duplicate/validity rates, failure reasons, length histograms, length/source recovery, payload byte/bit/character edit distances and EOS/format validity. Candidate JSONL and SQLite plus lossless raw NPZ permit K-prefix reaggregation without generation. Eligible seed1/2 results are individual rows; no seed pooling.',
       '', '## Resources','', '```json',json.dumps({k:v for k,v in resources.items() if k!='verifier'},indent=2),'```',
       '', '## Limitations','',
       '- q8/12/16 are truncated MD5; no full MD5 inversion evidence.',
       '- Sources are iid ASCII94/bytes256 and lengths4–31 only.',
       '- Candidate-count fairness is not compute fairness.',
       '- P2 is same-dataset model-seed robustness; selected-setting replication is not an independent confirmatory study.',
       '- Gaussian versus Discrete compares end-to-end approaches, not purely noise formulations.',
       '- Singleton finite search and small held-out control sets limit scope. G1 failures diagnose this pipeline/configuration, not absence of hash signal.',
       '- Bootstrap intervals are marginal; small-N/zero-success bounds rely on target-level sampling assumptions. Test groups have heterogeneous probabilities.',
       '', '## Resume / reproduce','', '```bash','.venv/bin/python -m diffusion_hash_inv.poc --verify','.venv/bin/python -m diffusion_hash_inv.poc --run','```',
       '', 'Run with MPS access outside the sandbox. Completed checksum-valid training and attempts are reused. The fixed protocol rejects code changes. Resource interruptions remain INCOMPLETE with reason INCOMPLETE_RESOURCE. All progress and next jobs are in RUN_STATE.json and RUNBOOK.md.']
    atomic(ROOT/'reports/FINAL_POC_REPORT.md',('\n'.join(lines)+'\n').encode())
    return final


def run(spec):
    if read(ROOT/'RUN_STATE.json')['status']=='COMPLETED':
        verify()
        return
    started=time.perf_counter();state('P1',status='RUNNING')
    pipeline_validation(spec)
    for source in ('printable','random_bytes'):
        for q in spec['qs']:dataset(spec,source,q)
    controls={}
    for core in CORES:
        try:controls[core]=positive_control(spec,core)
        except (MemoryError,TimeoutError) as error:
            controls[core]=dict(status='INCOMPLETE',reason='INCOMPLETE_RESOURCE',error=str(error))
            state(job=f'PC-{core}-s0',status='INCOMPLETE',reason='INCOMPLETE_RESOURCE')
    write(ROOT/'gates/G1.json',controls)
    monte_carlo(spec)
    validation_search(spec,controls)
    state('P1',status='COMPLETED',gates={k:v['status'] for k,v in controls.items()})
    for phase,qs in [('P2',[8]),('P3',[12,16])]:
        state(phase,status='RUNNING')
        for core in CORES:
            if controls[core]['status']!='PASS':
                for q in qs:
                    for method in ('main','shuffled'):state(job=f'{core}-q{q}-s0-{method}',status='BLOCKED',reason='G1-B control FAIL')
                continue
            for q in qs:
                if time.perf_counter()-started>spec['resources']['wall_seconds']:raise TimeoutError('INCOMPLETE_RESOURCE wall policy')
                test_setting(spec,core,q,0,controls)
        state(phase,status='COMPLETED' if any(c['status']=='PASS' for c in controls.values()) else 'BLOCKED')
    eligible=[]
    for core in CORES:
        for q in (12,16):
            p=ROOT/'results'/f'{core}-q{q}-s0.json'
            eligible.append(dict(core=core,q=q,eligible=read(p)['eligible'] if p.exists() else False,
                                 reason='registered paired-effect rule' if p.exists() else 'prerequisites blocked'))
    write(ROOT/'results/ELIGIBILITY.json',dict(settings=eligible))
    chosen=[r for r in eligible if r['eligible']]
    state('P4',status='RUNNING' if chosen else 'NOT_RUN')
    for r in chosen:
        for model_seed in (1,2):
            pc=positive_control(spec,r['core'],model_seed)
            if pc['status']=='PASS':test_setting(spec,r['core'],r['q'],model_seed,{r['core']:pc})
            else:
                for method in ('main','shuffled'):state(job=f'{r["core"]}-q{r["q"]}-s{model_seed}-{method}',status='BLOCKED',reason='replication G1 control FAIL')
    if chosen:state('P4',status='COMPLETED')
    state('P5',status='RUNNING')
    final=report(spec,controls)
    state('P5',status='COMPLETED',result=final['status'])
    s=read(ROOT/'RUN_STATE.json');s['status']='INCOMPLETE' if final['status']=='INCOMPLETE' else 'COMPLETED';s['final_result']=final['status'];write(ROOT/'RUN_STATE.json',s)
    report(spec,controls)


def verify():
    spec=load_spec();checked=0
    for marker in ROOT.rglob('COMPLETE.json'):
        r=read(marker)
        if 'metrics_sha256' in r and digest(marker.parent/'metrics.json')!=r['metrics_sha256']:raise RuntimeError(str(marker))
        if 'checkpoint_sha256' in r:
            if digest(marker.parent/'training_resume.pt')!=r['checkpoint_sha256']:raise RuntimeError(str(marker))
        if 'ledger_jsonl_sha256' in r:
            if digest(marker.parent/'candidates.jsonl')!=r['ledger_jsonl_sha256']:raise RuntimeError(str(marker))
        checked+=1
    for path in (ROOT/'datasets').glob('*/manifest.json'):
        m=read(path);dataset(spec,m['source'],m['q']);audit_dataset(path.parent)
    print(json.dumps(dict(status='PASS',completed_artifacts=checked,scope='integrity only; not a new scientific gate')),flush=True)


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--validate',action='store_true');parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    with (ROOT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare:prepare()
        if args.validate:
            proc=subprocess.run([sys.executable,'-m','pytest','-q'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            atomic(ROOT/'logs/FULL_TESTS.log',proc.stdout.encode())
            if proc.returncode:raise RuntimeError('software tests failed; scientific execution forbidden')
            write(ROOT/'gates/SOFTWARE.json',dict(status='PASS',time=now(),log='logs/FULL_TESTS.log',code=code_manifest()))
        if args.verify:verify()
        if args.run:
            spec=load_spec()
            software=read(ROOT/'gates/SOFTWARE.json')
            if software['status']!='PASS' or software['code']!=code_manifest():raise RuntimeError('fresh full software suite required')
            if spec['resources']['device']=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no silent fallback')
            try:run(spec)
            except BaseException as e:
                s=read(ROOT/'RUN_STATE.json');s.update(status='INCOMPLETE',error=repr(e))
                for v in s['jobs'].values():
                    if v['status']=='RUNNING':v.update(status='INCOMPLETE',reason='INCOMPLETE_RESOURCE' if isinstance(e,(TimeoutError,MemoryError,KeyboardInterrupt)) or 'out of memory' in str(e).lower() else repr(e))
                write(ROOT/'RUN_STATE.json',s)
                with (ROOT/'RUNBOOK.md').open('a') as f:f.write(f'\nInterruption {now()}: {e!r}. Resume same command; no missing attempts imputed.\n')
                raise


if __name__=='__main__':main()
