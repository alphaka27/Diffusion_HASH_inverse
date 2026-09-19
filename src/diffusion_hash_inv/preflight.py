"""Executable engineering checks only; never starts a confirmatory hash experiment."""
import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
from time import perf_counter

import torch

from .automation import STATE_PATH, complete_job, freeze_json, source_version, verify_completed
from .experiment_state import sha256, write_json
from .dataset import SourceSpec, DigestRecord, quota_digest_split, select_digest_representatives
from .encoding.bgv import BGVConfig, BGVEncoder, BGVDecoder
from .encoding.cgge import CGGEConfig, CGGEEncoder, CGGEDecoder, PRINTABLE94, glyph_for_character, glyph_table_checksum
from .encoding.tokens import TokenCodec
from .evaluation import verify_candidate, score_attempts, write_evaluation, EVALUATOR_VERSION, VERIFIER_VERSION
from .baselines import source_prior_random_search, source_prior_expectation
from .protocol_gates import validate_dataset, validate_comparison
from .study_statistics import analyze_family, write_statistics

QS = {'md5': [8,12,16,20,24,32,64,128], 'sha256': [8,12,16,20,24,32,64,128,256]}
FAMILIES = {'G-P-BGV': ('printable','gaussian','bgv'), 'G-P-CG': ('printable','gaussian','cgge'),
            'G-R-BGV': ('random_bytes','gaussian','bgv'), 'D-P': ('printable','discrete','tokens'),
            'D-R': ('random_bytes','discrete','tokens')}
FAMILIES.update({key+'-L': value for key,value in list(FAMILIES.items())})


def _codec_checks():
    result = {}
    for name, (source, family, representation) in FAMILIES.items():
        count = 0
        for maximum in (4,7,8,31,32,255):
            if representation == 'bgv':
                config = BGVConfig(max_message_length=maximum)
                encoder, decoder = BGVEncoder(config), BGVDecoder(config)
            elif representation == 'cgge':
                config = CGGEConfig(max_message_length=maximum)
                encoder, decoder = CGGEEncoder(config), CGGEDecoder(config)
            else:
                encoder = decoder = TokenCodec(source,maximum)
            alphabet = range(0x21,0x7f) if source == 'printable' else range(256)
            corpus = [bytes([symbol])*4 for symbol in alphabet]
            corpus += [(b'!Az9' * maximum)[:maximum]]
            if source == 'random_bytes':
                corpus += [bytes(i % 256 for i in range(maximum)), b'\x00' * maximum]
            for message in corpus:
                output = decoder.decode(encoder.encode(message))
                if not output.valid or output.message != message:
                    raise RuntimeError(f'roundtrip failure {name} length {len(message)}')
                count += 1
        result[name] = dict(gate='G1', scope='engineering', status='PARTIAL',
                            codec_status='PASS', roundtrip_count=count, roundtrip_rate=1.0,
                            positive_control_status='BLOCKED_SPECIFICATION',
                            reason='No new-protocol learned reversible positive control has been executed.')
    # Check every prototype itself, including one-character unit-test configuration.
    config = CGGEConfig(min_message_length=1)
    encoder, decoder = CGGEEncoder(config), CGGEDecoder(config)
    glyphs = set()
    for character in PRINTABLE94:
        if decoder.decode(encoder.encode(character)).message != character.encode():
            raise RuntimeError('single-character glyph failure')
        glyphs.add(bytes(map(int,glyph_for_character(character).flatten().tolist())))
    if len(glyphs) != 94 or glyph_table_checksum() != '6ef6d0bfa6e29c823ad9ff7d6b4b29b9af5f739fee711268ff5b83ca7aeed50a':
        raise RuntimeError('glyph table integrity failure')
    return result


def _exhaustive_verifier():
    rows = []
    for source in ('printable','random_bytes'):
        alphabet = range(0x21,0x7f) if source == 'printable' else range(256)
        for algorithm, qs in QS.items():
            checks = messages = 0
            for length in (1,2):
                for symbols in product(alphabet,repeat=length):
                    message = bytes(symbols)
                    digest = hashlib.new(algorithm,message).digest()
                    reference_bits = ''.join(f'{byte:08b}' for byte in digest)
                    # Check a self target and an independently hashed non-self target.
                    other = hashlib.new(algorithm,message+b'\x00').digest()
                    other_bits = ''.join(f'{byte:08b}' for byte in other)
                    for q in qs:
                        target = DigestRecord(0,source,message,algorithm,q,digest)
                        verified = verify_candidate(message,target)
                        if not verified.prefix_match or verified.digest != digest.hex() or not verified.full_digest_match:
                            raise RuntimeError('self rehash mismatch')
                        other_target = DigestRecord(1,source,message+b'\x00',algorithm,q,other)
                        checked = verify_candidate(message,other_target)
                        if checked.prefix_match != (reference_bits[:q] == other_bits[:q]):
                            raise RuntimeError('non-self prefix mismatch')
                        checks += 2
                    messages += 1
            rows.append(dict(source=source,algorithm=algorithm,lengths=[1,2],messages=messages,
                             verdict_checks=checks,status='PASS'))
    return rows


def run():
    torch.set_num_threads(1)
    state = json.loads(STATE_PATH.read_text())
    verify_completed(state)
    version = source_version()
    tag = version[:12]
    job_id = 'ENGINEERING_PREFLIGHT_' + tag
    if any(job['job_id'] == job_id for job in state['completed_jobs']):
        print(json.dumps({'status':'SKIPPED_VERIFIED_COMPLETE','job_id':job_id}))
        return
    root = Path('artifacts')
    config_path = root/'config/frozen'/f'engineering_{tag}.json'
    config = dict(scope='engineering_only_not_preregistered_scientific_run', source_version=version,
                  dataset_seed=7, split_seed=11, representative_seed=7, maximum_length=31,
                  codec_test_maxima=[4,7,8,31,32,255], sanity_quotas=[40,10,10], max_draws=3000,
                  algorithms=QS, sanity_q=[8,12,16], ks=[1,10,100], baseline_seed=7,
                  monte_carlo_draws=1000, monte_carlo_seed=8, bootstrap_seed=7, bootstrap_samples=10000)
    checksum = freeze_json(config_path,config)
    if not any(item['path']==str(config_path) for item in state['config_snapshots']):
        state['config_snapshots'].append({'path':str(config_path),'sha256':checksum,'scope':'engineering'})
    state['current_phase']='PHASE_1_ENGINEERING_PREFLIGHT'
    write_json(STATE_PATH,state)
    artifacts = [config_path]
    started = perf_counter()
    codecs = _codec_checks()
    for name, result in codecs.items():
        path = root/'gates'/f'G1_{name}_{tag}.json'
        write_json(path,result); artifacts.append(path)
    verifier = _exhaustive_verifier()
    path=root/'gates'/f'VERIFIER_EXHAUSTIVE_{tag}.json'
    write_json(path,{'scope':'engineering','rows':verifier,'verifier_version':VERIFIER_VERSION});artifacts.append(path)
    data_rows, g0_rows = [], []
    for source in ('printable','random_bytes'):
        for algorithm in QS:
            for q in (8,12,16):
                dataset_id=f'SANITY_{algorithm}_{source}_Q{q}_{tag}'
                folder=root/'datasets'/dataset_id;folder.mkdir(parents=True,exist_ok=True)
                spec=SourceSpec(source,60,seed=7,max_length=31)
                split,construction=quota_digest_split(spec,algorithm=algorithm,q=q,split_seed=11,
                                                     quotas=(40,10,10),max_draws=3000)
                records_path=folder/'records.jsonl'
                records_path.write_text(''.join(json.dumps(record.to_json(name),sort_keys=True)+'\n'
                                               for name,records in split.items() for record in records))
                manifest=dict(dataset_id=dataset_id,seed=7,split_seed=11,source=source,l_min=4,l_max=31,
                              size=60,algorithm=algorithm,q=q,scope='engineering_sanity_not_pilot',
                              counts={name:len(records) for name,records in split.items()},
                              records_sha256=sha256(records_path),generation_code_version=version,
                              frozen_config_sha256=checksum,construction=construction)
                freeze_json(folder/'manifest.json',manifest)
                audit=validate_dataset(folder,config_path)
                if audit['status']!='PASS':raise RuntimeError(f'G0 failed: {dataset_id}')
                g0=root/'gates'/f'G0_{dataset_id}.json';write_json(g0,audit)
                markdown=folder/'split_validation.md'
                markdown.write_text(f'# {dataset_id}\n\nEngineering sanity only. G0 structural/integrity status: PASS.\n\n'
                                    f'```json\n{json.dumps(audit["audit"],indent=2)}\n```\n')
                g0_rows.append(dataset_id)
                counts=audit['audit']['unique_digest_counts']
                targets=select_digest_representatives(split['test'],seed=7)
                stream=source_prior_random_search(targets,k=100,seed=7,max_length=31)
                scores={k:asdict(score_attempts(targets,[row[:k] for row in stream],k=k)) for k in (1,10,100)}
                if not scores[1]['preimage_success_at_k']<=scores[10]['preimage_success_at_k']<=scores[100]['preimage_success_at_k']:
                    raise RuntimeError('K-prefix monotonicity failure')
                eval_dir=root/'evaluation'/dataset_id
                write_evaluation(targets,stream,eval_dir,method='source_prior_engineering_sanity',k=100,
                                 metadata={'scope':'engineering','dataset_id':dataset_id,'seed':7})
                write_json(eval_dir/'prefix_metrics.json',scores)
                mc=source_prior_expectation(targets,draws=1000,seed=8,min_length=4,max_length=31,length_aware=False)
                write_json(eval_dir/'source_prior_mc.json',mc)
                paired=dict(source=source,algorithm=algorithm,q=q,k=100,l_min=4,l_max=31,condition_type='hash-only',
                            verifier_version=VERIFIER_VERSION,evaluator_version=EVALUATOR_VERSION,
                            validity_rule='format-and-source-domain-v2',dataset_id=dataset_id,
                            target_order=[target.prefix for target in targets])
                ledger=[json.loads(line) for line in (eval_dir/'candidates.jsonl').read_text().splitlines()]
                g2=validate_comparison(paired,paired,ledger,ledger)
                # Self-comparison exercises validator only; it is not model-vs-baseline evidence.
                g2.update(scope='engineering_self_comparison_only')
                if g2['status']!='PASS':raise RuntimeError('G2 engineering validator failed')
                g2_path=root/'gates'/f'G2_{dataset_id}.json';write_json(g2_path,g2)
                artifacts.extend([records_path,folder/'manifest.json',markdown,g0,g2_path,*eval_dir.glob('*')])
                data_rows.append(dict(dataset_id=dataset_id,source=source,algorithm=algorithm,q=q,
                                      unique_train=counts['train'],unique_validation=counts['validation'],unique_test=counts['test'],
                                      baseline_at_100=scores[100]['preimage_success_at_k'],
                                      mc_expected_at_100=sum(r['at_k_estimate']['100'] for r in mc)/len(mc)))
                progress=root/'automation'/f'preflight_progress_{tag}.json'
                write_json(progress,{'completed_datasets':data_rows,'elapsed_seconds':perf_counter()-started})
    ids=list(range(8));pairs={s:(ids,[True]*8,[False]*8) for s in (0,1,2)}
    rows,reproduction=analyze_family({'fixture_prior':'greater','fixture_approach':'two-sided'},
                                    {'fixture_prior':pairs,'fixture_approach':pairs},bootstrap_seed=7)
    fixture_dir=root/'statistics'/f'PIPELINE_FIXTURE_{tag}'
    write_statistics(fixture_dir,rows)
    write_json(fixture_dir/'scope.json',{'scope':'synthetic_binary_fixture_not_model_results','reproduction':reproduction})
    artifacts.extend(fixture_dir.glob('*'))
    summary=root/'automation'/f'PREFLIGHT_RESULTS_{tag}.json'
    write_json(summary,dict(scope='engineering_only',source_version=version,codecs=codecs,verifier=verifier,
                            datasets=data_rows,elapsed_seconds=perf_counter()-started,
                            scientific_gates={'G0':'BLOCKED_SPECIFICATION','G1':'PARTIAL_POSITIVE_CONTROLS_NOT_RUN',
                                              'G2':'NOT_RUN','G3':'NOT_RUN','G4':'NOT_RUN'}))
    artifacts.append(summary)
    complete_job(job_id,artifacts,code_version=version,config_checksum=checksum)
    print(json.dumps({'status':'COMPLETED','scope':'engineering_only','summary':str(summary),
                      'dataset_count':len(data_rows),'elapsed_seconds':perf_counter()-started}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args()
    run()
