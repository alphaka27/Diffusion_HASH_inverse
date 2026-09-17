import pytest
import torch
from diffusion_hash_inv.conditional_dependence import interventions, paired_sample, rows_for_outputs, summarize
from diffusion_hash_inv.dataset import build_digest_records
from diffusion_hash_inv.models import GaussianDiffusion
from diffusion_hash_inv.runner import _codec

@pytest.mark.parametrize('representation',['bits','bgv','cgge'])
def test_intervention_and_paired_oracle(representation):
    encoder,decoder,shape=_codec(representation)
    records=build_digest_records([b'ABCD',b'wxyz',b'1234'],source='printable',algorithm='md5',q=8)
    clean=torch.stack([encoder.encode(r.message)*2-1 for r in records])
    condition=(clean.flatten(1)+1)/2
    original=condition.clone()
    variants,donors=interventions(condition)
    assert torch.equal(condition,original) and torch.equal(variants['correct'],original)
    assert all(v.shape==condition.shape for v in variants.values())
    assert not variants['zero'].any()
    assert sorted(donors.tolist())==list(range(3))
    assert (donors!=torch.arange(3)).all()
    class Oracle(torch.nn.Module):
        def forward(self,value,time,cond): return cond.reshape_as(value)*2-1
    diffusion=GaussianDiffusion(3,beta_end=.4,prediction_type='sample',device=torch.device('cpu'))
    outputs,initial=paired_sample(Oracle(),diffusion,variants,shape,3,0)
    assert torch.equal(initial,torch.randn(clean.shape,generator=torch.Generator().manual_seed(0)))
    rows=rows_for_outputs(outputs,records,donors,clean,decoder,'test',0)
    correct=summarize([r for r in rows if r['condition']=='correct'])
    deranged=summarize([r for r in rows if r['condition']=='deranged'])
    assert correct['exact_recovery_rate']==1 and correct['reconstruction_mse']==0
    assert deranged['exact_recovery_rate']==0 and deranged['donor_exact']==1
    assert deranged['donor_byte_accuracy']==1 and deranged['decoded_message_disagreement']==1
    assert deranged['output_bit_disagreement']>0


def test_derangement_requires_two():
    with pytest.raises(ValueError): interventions(torch.zeros(1,8))


def test_completed_artifact_integrity_and_missing_product_recovery(tmp_path):
    import json
    from diffusion_hash_inv.conditional_dependence import validate_completed
    from diffusion_hash_inv.experiment_state import sha256
    product=tmp_path/'metrics.json'
    product.write_text('{}')
    (tmp_path/'complete.json').write_text(json.dumps({'sha256':{'metrics.json':sha256(product)}}))
    assert validate_completed(tmp_path)
    product.unlink()
    assert not validate_completed(tmp_path)
    product.write_text('{"tampered":true}')
    with pytest.raises(RuntimeError,match='corrupt'): validate_completed(tmp_path)


def test_gate_fail_stop(tmp_path,monkeypatch):
    import json
    from diffusion_hash_inv.experiment_state import require_previous
    monkeypatch.chdir(tmp_path)
    (tmp_path/'EXPERIMENT_STATE.json').write_text(json.dumps({'gates':{'G0':'PASS','G1':'PASS','G2':'FAIL'}}))
    with pytest.raises(RuntimeError,match='blocked by G2'): require_previous('G3')
    require_previous('G2')


def test_recovery_lock_rejects_concurrent_writer(tmp_path,monkeypatch):
    from diffusion_hash_inv.experiment_state import experiment_lock
    monkeypatch.chdir(tmp_path)
    with experiment_lock():
        with pytest.raises(RuntimeError,match='active'):
            with experiment_lock(): pass
    with experiment_lock(): pass
