"""Scientific PoC engineering checks. Synthetic outputs never confer learned G1."""
import hashlib
import json
import random
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from diffusion_hash_inv import poc
from diffusion_hash_inv.discrete import MaskedDiffusion, SequenceDenoiser
from diffusion_hash_inv.encoding.tokens import TokenCodec
from diffusion_hash_inv.dataset import DigestRecord


def test_continuous_corruption_loss_empty_and_special_tokens():
    torch.set_num_threads(1)
    codec=TokenCodec('random_bytes',31)
    clean=torch.stack([codec.encode(b'\x00\xffAB')]*4096)
    d=MaskedDiffusion(codec.mask,[0,.5,1],device=torch.device('cpu'))
    g=torch.Generator().manual_seed(20260920)
    corrupt,mask=d.corrupt_at_time(clean,torch.full((4096,),.37),generator=g)
    assert all(.33<float(mask[:,i].float().mean())<.41 for i in range(32))
    assert d.corrupt_at_time(clean[:2],torch.ones(2),generator=g)[1].all()
    assert not d.corrupt_at_time(clean[:2],torch.zeros(2),generator=g)[1].any()
    with pytest.raises(ValueError):d.corrupt_at_time(clean[:2],torch.tensor([float('nan'),0.]),generator=g)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.logits=torch.nn.Parameter(torch.zeros(2,32,258));self.time=None
        def forward(self,value,time,condition):self.time=time;return self.logits
    m=Model();c=torch.zeros(2,259);g=torch.Generator().manual_seed(20)
    expected_g=torch.Generator().manual_seed(20)
    times=torch.rand(2,generator=expected_g);_,mask=d.corrupt_at_time(clean[:2],times,generator=expected_g)
    loss=d.loss(m,clean[:2],c,generator=g)
    assert torch.equal(m.time,times) and not torch.isin(m.time,torch.tensor([0,.5,1])).any()
    expected=(torch.full((2,32),np.log(258))*mask).sum(1)/mask.sum(1).clamp_min(1)
    assert loss.item()==pytest.approx(expected.mean().item())
    with patch.object(d,'corrupt_at_time',return_value=(clean[:2],torch.zeros_like(clean[:2],dtype=torch.bool))):
        loss=d.loss(m,clean[:2],c,generator=g);loss.backward()
        assert loss.item()==0 and m.logits.grad.abs().sum()==0


def test_forward_no_padding_mask_and_condition_metadata_isolation():
    for q in (8,12,16):
        digest=hashlib.md5(b'ABCD').digest();prefix=poc.digest_prefix_hex(digest,q)
        expected=poc.hash_condition(q,prefix)
        for n in (4,31):
            fake=replace(DigestRecord(0,'printable',b'A'*n,'md5',q,digest),id=199)
            changed=((int.from_bytes(digest,'big')>>(128-q))<<(128-q)) | ((1<<(128-q))-1)
            fake=replace(fake,digest=changed.to_bytes(16,'big'))
            assert torch.equal(expected,poc.digest_condition('md5',q,fake.digest))
        class Spy(torch.nn.Module):
            def __init__(self):super().__init__();self.initial=None
            def forward(self,value,time,condition):
                if self.initial is None:self.initial=value.clone()
                return torch.zeros(len(value),32,258)
        model=Spy();diff=MaskedDiffusion(258,[0,.5,1],device=torch.device('cpu'))
        diff.sample(model,expected[None],(32,),sampling_steps=2,generator=torch.Generator().manual_seed(1),temperature=1.)
        assert (model.initial==258).all()


def test_reverse_retains_revealed_tokens_nonuniform_time():
    seen=[]
    class Spy(torch.nn.Module):
        def forward(self,value,time,condition):
            seen.append((value.clone(),time.clone()))
            logits=torch.full((*value.shape,4),-100.)
            logits[:,:,len(seen)%4]=100
            return logits
    diff=MaskedDiffusion(4,[0,.1,.7,1],device=torch.device('cpu'))
    result=diff.sample(Spy(),torch.zeros(100,1),(32,),sampling_steps=3,generator=torch.Generator().manual_seed(4),temperature=1.)
    assert seen[0][1][0]==1 and seen[1][1][0]==pytest.approx(.7) and seen[2][1][0]==pytest.approx(.1)
    for (old,_),(new,_) in zip(seen,seen[1:]):assert torch.equal(old[old!=4],new[old!=4])
    assert not (result==4).any()


def fixture_root(tmp_path,monkeypatch):
    monkeypatch.setattr(poc,'ROOT',tmp_path)
    poc.write(tmp_path/'protocol/FROZEN_POC_SPEC.json',{'fixture':True})
    return {'seeds':{'sampling':345,'baseline':123,'training':456}}


def test_attempt_resume_no_regeneration_invalid_duplicate_and_rehash(tmp_path,monkeypatch):
    spec=fixture_root(tmp_path,monkeypatch);codec=TokenCodec('random_bytes',31)
    target=DigestRecord(0,'random_bytes',b'ABCD','md5',12,hashlib.md5(b'ABCD').digest())
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(1))
    model=Tiny();values=torch.stack([codec.encode(b'ABCD')]*100);values[0,0]=codec.mask
    cfg={'id':'fixture','sampling_steps':2}
    folder=tmp_path/'eval'
    original_put=poc.Ledger.put
    def interrupted(self,row):
        if row['attempt_index']==8:raise KeyboardInterrupt()
        original_put(self,row)
    kwargs=dict(model=model,diffusion=None,decoder=codec,shape=(32,),cfg=cfg,conditions=torch.zeros(1,259),
                core='fixture',q=12,method='main')
    with patch.object(poc,'sample',return_value=values) as sampler,patch.object(poc.Ledger,'put',interrupted):
        with pytest.raises(KeyboardInterrupt):poc.generate_stream(folder,[target],'random_bytes','discrete',spec,**kwargs)
        assert sampler.call_count==1
    ledger=poc.Ledger(folder/'candidates.sqlite');assert len(ledger.rows())==7;ledger.close()
    with patch.object(poc,'sample',side_effect=AssertionError('saved candidate regenerated')):
        result=poc.generate_stream(folder,[target],'random_bytes','discrete',spec,**kwargs)
        again=poc.generate_stream(folder,[target],'random_bytes','discrete',spec,**kwargs)
    assert result==again and result['candidate_count']==100 and result['actual_hash_calls']==99
    assert result['ks']['1']['successes']==0 and result['ks']['10']['successes']==result['ks']['100']['successes']==1
    assert result['ks']['100']['duplicate_rate']==.98
    ledger=poc.Ledger(folder/'candidates.sqlite');rows=ledger.rows();ledger.close()
    assert rows[0]['full_md5'] is None and not rows[0]['actual_hash_call']
    assert rows[-1]['attempt_index']==100
    with pytest.raises(ValueError):poc.aggregate(rows[:-1],[target.prefix])
    # Invalid interpretable bytes are still independently hashed, never repaired.
    row=poc.evaluate_raw(b'\x00'*4,replace(target,source='printable'),None,'random',index=1,seen=set(),metadata={})
    assert row['actual_hash_call'] and not row['validity']


def test_checkpoint_resume_matches_uninterrupted(tmp_path,monkeypatch):
    spec=fixture_root(tmp_path,monkeypatch)
    cfg={'id':'fixture','lr':.001,'batch':2,'updates':[4],'checkpoint_interval':2}
    clean=torch.tensor([[0,1,2,3],[1,2,3,0]],dtype=torch.long);cond=torch.zeros(2,3)
    def build():
        torch.manual_seed(0)
        return SequenceDenoiser(5,4,3,width=8,embedding_dim=2),MaskedDiffusion(4,[0,.5,1],device=torch.device('cpu'))
    model,diff=build()
    def interrupt(step):raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):poc.train(tmp_path/'resume',model,diff,clean,cond,cfg,0,spec,callback=interrupt)
    resumed,diff=build();poc.train(tmp_path/'resume',resumed,diff,clean,cond,cfg,0,spec)
    full,diff=build();poc.train(tmp_path/'full',full,diff,clean,cond,cfg,0,spec)
    for a,b in zip(resumed.parameters(),full.parameters()):assert torch.equal(a,b)
    with patch.object(diff,'loss',side_effect=AssertionError('retrained completed job')):
        poc.train(tmp_path/'full',full,diff,clean,cond,cfg,0,spec)


def test_edit_distance_exact_against_dynamic_programming():
    def reference(a,b):
        row=list(range(len(b)+1))
        for i,x in enumerate(a,1):
            new=[i]
            for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
            row=new
        return row[-1]
    rng=random.Random(5)
    for _ in range(200):
        a=bytes(rng.randrange(4) for _ in range(rng.randrange(257)))
        b=bytes(rng.randrange(4) for _ in range(rng.randrange(257)))
        assert poc.edit_distance(a,b)==reference(a,b)


def test_immutable_config_and_statistics_direction(tmp_path):
    p=tmp_path/'frozen.json';poc.seal(p,{'k':100});poc.seal(p,{'k':100})
    with pytest.raises(RuntimeError):poc.seal(p,{'k':10})
    result=poc.paired_comparison([True]*6,[False]*6,bootstrap_seed=1)
    assert result.mcnemar_pvalue==1/64 and result.delta_ci95==(1.,1.)
    assert poc.holm_adjust({'a':.01,'b':.04})=={'a':.02,'b':.04}


def test_gaussian_initial_input_unaffected_by_evaluator_metadata():
    from diffusion_hash_inv.models import GaussianDiffusion
    conditions=poc.hash_condition(12,'abc')[None]
    initial=[]
    class Spy(torch.nn.Module):
        def forward(self,value,time,condition):
            if len(initial)<2:initial.append(value.clone())
            return torch.zeros_like(value)
    diffusion=GaussianDiffusion(2,beta_end=.4,prediction_type='sample',device=torch.device('cpu'))
    for representative_length in (4,31):
        diffusion.sample(Spy(),conditions,(2,32,128),sampling_steps=1,generator=torch.Generator().manual_seed(51))
    assert torch.equal(initial[0],initial[1])
