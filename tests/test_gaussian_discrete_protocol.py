"""New-plan regression tests; all numerical model values here are engineering fixtures."""
import hashlib
import json
import math
from dataclasses import asdict, replace
from unittest.mock import patch

import pytest
import torch

from diffusion_hash_inv.encoding.bgv import BGVConfig, BGVEncoder, BGVDecoder
from diffusion_hash_inv.encoding.cgge import CGGEConfig, CGGEEncoder, CGGEDecoder, glyph_table_checksum
from diffusion_hash_inv.encoding.tokens import TokenCodec
from diffusion_hash_inv.discrete import MaskedDiffusion, SequenceDenoiser
from diffusion_hash_inv.conditioning import digest_condition, shuffled_donors
from diffusion_hash_inv.dataset import SourceSpec, build_digest_records, quota_digest_split, split_validation_report, select_digest_representatives
from diffusion_hash_inv.baselines import source_prior_random_search, source_prior_expectation
from diffusion_hash_inv.evaluation import CandidateAttempt, score_attempts, write_evaluation, exact_mcnemar, verify_candidate, paired_comparison
from diffusion_hash_inv.automation import freeze_json, complete_job, verify_completed
from diffusion_hash_inv.runner import ExperimentConfig, _conditions, run_experiment
from diffusion_hash_inv.protocol_gates import validate_dataset, validate_comparison


@pytest.mark.parametrize('maximum', [4, 7, 8, 31, 32, 64, 255])
def test_generalized_shapes_roundtrip_padding_nonfinite(maximum):
    for config, encoder, decoder, width in ((BGVConfig, BGVEncoder, BGVDecoder, 128), (CGGEConfig, CGGEEncoder, CGGEDecoder, 64)):
        settings = config(max_message_length=maximum)
        enc, dec = encoder(settings), decoder(settings)
        for length in (4, maximum):
            value = enc.encode(b'!' * length)
            assert value.shape == (2, 8 * math.ceil((maximum + 1) / 8), width)
            assert dec.decode(value).message == b'!' * length
            damaged = value.clone(); damaged[0, 0, 0] = float('nan')
            assert not dec.decode(damaged).valid
        if settings.slot_count > maximum + 1:
            value = enc.encode(b'!!!!'); cell_width = 16 if width == 128 else 8
            row, col = divmod(maximum + 1, 8)
            value[1, row*8:(row+1)*8, col*cell_width:(col+1)*cell_width] = 1
            assert not dec.decode(value).valid
    if maximum > 4:
        enc, dec = BGVEncoder(BGVConfig(max_message_length=maximum)), BGVDecoder(BGVConfig(max_message_length=maximum))
        value = enc.encode(b'!!!!'); value[0, :8, 5*16:6*16] = 1
        assert dec.decode(value).reason == 'padding_inconsistent'


def test_checksum_and_header_limit():
    assert glyph_table_checksum() == '6ef6d0bfa6e29c823ad9ff7d6b4b29b9af5f739fee711268ff5b83ca7aeed50a'
    with pytest.raises(ValueError): BGVConfig(max_message_length=256)


@pytest.mark.parametrize('source,vocab', [('printable',97),('random_bytes',259)])
def test_tokens_all_symbols_and_formats(source,vocab):
    codec=TokenCodec(source,31)
    assert codec.vocabulary_size==vocab
    alphabet=range(0x21,0x7f) if source=='printable' else range(256)
    for symbol in alphabet:
        for length in (4,31):
            message=bytes([symbol])*length
            assert codec.decode(codec.encode(message)).message==message
    message=b'ABCD' if source=='printable' else b'\x00\x00\xff\x00'
    value=codec.encode(message)
    if source=='random_bytes': assert value[0]==0 and codec.pad!=0
    bad=value.clone();bad[0]=codec.mask
    assert codec.decode(bad).reason=='mask_remaining'
    bad=value.clone();bad[0]=codec.eos
    assert codec.decode(bad).reason=='eos_count'
    bad=value.clone();bad[4]=codec.pad
    assert codec.decode(bad).reason=='eos_count'
    bad=value.clone();bad[0]=codec.pad
    assert codec.decode(bad).reason=='invalid_payload'
    bad=value.clone();bad[5]=0
    assert codec.decode(bad).reason=='non_pad_after_eos'
    assert not codec.decode(value.float()).valid
    assert not codec.decode(value[:-1]).valid


def test_digest_information_and_no_length_leakage():
    records=build_digest_records((b'ABCD',b'ABCDEFGHI'),source='printable',algorithm='sha256',q=12)
    target=records[0]
    suffix_changed=replace(target,digest=target.digest[:2]+b'\xff'*30)
    assert torch.equal(digest_condition('sha256',12,target.digest),digest_condition('sha256',12,suffix_changed.digest))
    for rep in ('bgv','cgge','tokens'):
        kwargs=dict(masking_schedule=(0,0.5,1),token_embedding_dim=2,token_temperature=1.) if rep=='tokens' else {}
        config=ExperimentConfig(rep,'printable','sha256',12,condition_format='canonical_bits',condition_dim=259,**kwargs)
        assert torch.equal(_conditions([target],config)[0],digest_condition('sha256',12,target.digest))
        assert torch.equal(_conditions([target],config),_conditions([replace(target,message=b'Z'*31)],config))
    length=digest_condition('sha256',12,target.digest,length=4)
    assert len(length)==260 and length[-1]==4/255
    bits=''.join(f'{byte:08b}' for byte in target.digest)[:12]
    assert length[3:15].tolist()==list(map(int,bits))


def test_discrete_forward_loss_sampler_and_gradient():
    torch.set_num_threads(1)
    codec=TokenCodec('random_bytes',7)
    clean=torch.stack([codec.encode(b'\x00\xffAB'),codec.encode(b'ABCDEFG')])
    condition=torch.zeros(2,259)
    diffusion=MaskedDiffusion(codec.mask,[0,.5,1],device=torch.device('cpu'))
    noisy,mask=diffusion.corrupt(clean,torch.tensor([2,2]),generator=torch.Generator().manual_seed(0))
    assert mask.all() and (noisy==codec.mask).all()  # Includes EOS and PAD.
    model=SequenceDenoiser(codec.vocabulary_size,8,259,width=8,embedding_dim=2)
    loss=diffusion.loss(model,clean,condition,generator=torch.Generator().manual_seed(0))
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    class Oracle(torch.nn.Module):
        def forward(self,value,time,cond):
            return torch.nn.functional.one_hot(clean,codec.mask).float()*100
    sample=diffusion.sample(Oracle(),condition,(8,),sampling_steps=2,generator=torch.Generator().manual_seed(1),temperature=1.)
    assert torch.equal(sample,clean)
    # Oracle checks algorithm correctness only, never model G1 positive-control efficacy.
    with pytest.raises(ValueError): MaskedDiffusion(codec.mask,[0,0.5,0.4,1],device=torch.device('cpu'))


def test_exact_quota_and_seeded_representative_selection():
    spec=SourceSpec('random_bytes',60,seed=7,max_length=8)
    split,construction=quota_digest_split(spec,algorithm='md5',q=8,split_seed=11,quotas=(40,10,10),max_draws=2000)
    again,_=quota_digest_split(spec,algorithm='md5',q=8,split_seed=11,quotas=(40,10,10),max_draws=2000)
    assert split==again and split_validation_report(split)['passed']
    assert list(map(len,split.values()))==[40,10,10]
    assert construction['draw_count']>=60
    records=split['test']
    assert select_digest_representatives(records,seed=7)==select_digest_representatives(tuple(reversed(records)),seed=7)
    with pytest.raises(RuntimeError): quota_digest_split(spec,algorithm='md5',q=8,split_seed=11,quotas=(40,10,10),max_draws=60)


@pytest.mark.parametrize('source',['printable','random_bytes'])
def test_prior_length_alphabet_prefix_expectation(source):
    targets=build_digest_records((b'ABCD',b'ABCDAB'),source=source,algorithm='md5',q=8)
    short=source_prior_random_search(targets,k=10,seed=7,max_length=8)
    long=source_prior_random_search(targets,k=100,seed=7,max_length=8)
    assert short==tuple(row[:10] for row in long)
    draws=source_prior_random_search(targets[:1],k=5000,seed=7,max_length=8)[0]
    frequencies=[sum(len(a.message)==length for a in draws) for length in range(4,9)]
    assert all(800<n<1200 for n in frequencies)
    if source=='printable': assert all(0x21<=v<=0x7e for a in draws for v in a.message)
    else: assert {v for a in draws for v in a.message}==set(range(256))
    known=source_prior_random_search(targets,k=10,seed=7,max_length=8,length_aware=True)
    assert all(len(a.message)==len(t.message) for t,row in zip(targets,known) for a in row)
    mc=source_prior_expectation(targets,draws=100,seed=8,min_length=4,max_length=8,length_aware=False)
    assert all(row['single_draw_ci95'][1]>0 for row in mc)


def test_domain_invalid_hashing_and_single_pass_accounting(tmp_path):
    targets=build_digest_records((b'\x00'*4,),source='printable',algorithm='md5',q=128)
    attempts=((CandidateAttempt(None,False,'bad'),CandidateAttempt(b'\x00'*4,True),CandidateAttempt(b'!!!!',False,'format')),)
    with patch('diffusion_hash_inv.evaluation.verify_candidate',wraps=verify_candidate) as verifier:
        result=write_evaluation(targets,attempts,tmp_path,method='fixture',k=3)
        assert verifier.call_count==result.hash_verification_count==2
    assert result.candidate_attempt_count==3 and result.preimage_success_at_k==0
    assert result.valid_decode_rate==0 and result.zero_success_upper95==pytest.approx(.95)
    ledger=[json.loads(line) for line in (tmp_path/'candidates.jsonl').read_text().splitlines()]
    assert ledger[1]['prefix_match'] and not ledger[1]['valid']
    assert ledger[2]['actual_digest'] is not None
    with pytest.raises(ValueError):score_attempts(targets,attempts,k=2)
    target=build_digest_records((b'!'*32,),source='printable',algorithm='md5',q=128)
    assert score_attempts(target,((CandidateAttempt(b'!'*32,True),),),k=1,max_length=32).preimage_success_at_k==1
    with pytest.raises(ValueError):score_attempts(target*2,((CandidateAttempt(None,False),),)*2,k=1)


def test_statistics_two_sided_and_no_discordance():
    assert exact_mcnemar(6,0)==1/64
    assert exact_mcnemar(6,0,alternative='two-sided')==2/64
    assert exact_mcnemar(0,0)==1
    assert exact_mcnemar(0,6)==1
    pair=paired_comparison([True]*6,[False]*6,alternative='two-sided',bootstrap_samples=10000,bootstrap_seed=7)
    assert pair.delta_ci95==(1.,1.) and pair.mcnemar_pvalue==2/64


def test_shuffled_hash_donors_and_impossible_strata():
    records=build_digest_records((b'ABCD',b'BCDE',b'CDEF',b'DEFG'),source='printable',algorithm='md5',q=128)
    donors=shuffled_donors(records,seed=7,same_length=True)
    assert sorted(donors)==list(range(4)) and all(i!=d for i,d in enumerate(donors))
    with pytest.raises(ValueError):shuffled_donors(records[:1],seed=7,same_length=True)


def test_snapshot_state_integrity_and_resume(tmp_path):
    config=tmp_path/'config.json';checksum=freeze_json(config,{'k':1})
    assert freeze_json(config,{'k':1})==checksum
    with pytest.raises(RuntimeError):freeze_json(config,{'k':10})
    artifact=tmp_path/'result.json';artifact.write_text('{}')
    state_path=tmp_path/'state.json'
    state_path.write_text(json.dumps({'completed_jobs':[],'pending_jobs':[{'job_id':'fixture'}],'config_snapshots':[{'path':str(config),'sha256':checksum}]}))
    assert complete_job('fixture',[artifact],code_version='v1',config_checksum=checksum,state_path=state_path)
    assert not complete_job('fixture',[artifact],code_version='v1',config_checksum=checksum,state_path=state_path)
    assert verify_completed(json.loads(state_path.read_text()))==1
    with pytest.raises(RuntimeError):complete_job('fixture',[artifact],code_version='v2',config_checksum=checksum,state_path=state_path)
    artifact.write_text('tampered')
    with pytest.raises(RuntimeError):verify_completed(json.loads(state_path.read_text()))


def test_discrete_runner_engineering_smoke(tmp_path):
    config=ExperimentConfig('tokens','random_bytes','md5',8,dataset_size=60,data_seed=7,split_seed=11,
        max_length=7,condition_dim=259,condition_format='canonical_bits',masking_schedule=(0,.5,1),
        token_embedding_dim=2,token_temperature=1.,training_steps=1,batch_size=2,width=8,
        sampling_steps=2,k=1,test_limit=2)
    result=run_experiment(config,tmp_path)
    assert result.summary.candidate_attempt_count==result.summary.target_count
    assert result.parameter_count>0 and math.isfinite(result.training_loss)
    manifest=json.loads((tmp_path/'run_manifest.json').read_text())
    assert manifest['condition_format']=='canonical_bits'
    assert manifest['scope']=='engineering_or_legacy_only_not_confirmatory'


def test_g0_integrity_and_g2_per_target_budget_fail_closed(tmp_path):
    config=tmp_path/'frozen.json';checksum=freeze_json(config,{'scope':'engineering'})
    records=build_digest_records((b'ABCD',b'BCDE',b'CDEF'),source='printable',algorithm='md5',q=128)
    root=tmp_path/'data';root.mkdir()
    text=''.join(json.dumps(record.to_json(split))+'\n' for record,split in zip(records,('train','validation','test')))
    (root/'records.jsonl').write_text(text)
    manifest={'frozen_config_sha256':checksum,'records_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'source':'printable','algorithm':'md5','q':128,'l_min':4,'l_max':31,
        'counts':{'train':1,'validation':1,'test':1},'scope':'engineering'}
    (root/'manifest.json').write_text(json.dumps(manifest))
    assert validate_dataset(root,config)['status']=='PASS'
    (root/'records.jsonl').write_text(text+'\n')
    assert validate_dataset(root,config)['status']=='FAIL'
    from diffusion_hash_inv.evaluation import EVALUATOR_VERSION,VERIFIER_VERSION
    paired={'source':'printable','algorithm':'md5','q':128,'k':1,'l_min':4,'l_max':31,'condition_type':'hash-only',
        'verifier_version':VERIFIER_VERSION,'evaluator_version':EVALUATOR_VERSION,'validity_rule':'domain-v2',
        'dataset_id':'fixture','target_order':[records[0].prefix]}
    row={'target_prefix':records[0].prefix,'k_position':1,'algorithm':'md5','q':128}
    assert validate_comparison(paired,paired,[row],[row])['status']=='PASS'
    assert validate_comparison(paired,paired,[row,row],[])['status']=='FAIL'
    changed={**paired,'verifier_version':'different'}
    assert validate_comparison(paired,changed,[row],[row])['status']=='FAIL'


def test_registered_family_and_seed_reproduction():
    from diffusion_hash_inv.study_statistics import analyze_family
    ids=list(range(8));paired={s:(ids,[True]*8,[False]*8) for s in (0,1,2)}
    rows,reproduction=analyze_family({'prior':'greater','approach':'two-sided'},
                                    {'prior':paired,'approach':paired},bootstrap_seed=7)
    assert len(rows)==6 and reproduction['prior']=='Strongly Reproduced'
    assert all(row['holm_adjusted_pvalue']>=row['mcnemar_pvalue'] for row in rows)
    with pytest.raises(ValueError):analyze_family({'prior':'greater'},{},bootstrap_seed=7)


def test_completed_training_is_checked_and_not_repeated(tmp_path):
    config=ExperimentConfig('tokens','printable','md5',8,dataset_size=60,data_seed=7,split_seed=11,
        max_length=7,condition_dim=259,condition_format='canonical_bits',masking_schedule=(0,.5,1),
        token_embedding_dim=2,token_temperature=1.,training_steps=1,batch_size=2,width=8,
        sampling_steps=2,k=1,test_limit=1)
    first=run_experiment(config,tmp_path)
    checkpoint=(tmp_path/'checkpoint.pt').read_bytes()
    with patch('diffusion_hash_inv.runner._train',side_effect=AssertionError('must skip completed training')):
        again=run_experiment(config,tmp_path)
    assert first.training_loss==again.training_loss
    assert checkpoint==(tmp_path/'checkpoint.pt').read_bytes()
    with pytest.raises(RuntimeError):run_experiment(replace(config,width=16),tmp_path)
    (tmp_path/'checkpoint.pt').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError):run_experiment(config,tmp_path)
