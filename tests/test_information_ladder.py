import pytest
import torch
from diffusion_hash_inv.information_ladder import LEVELS,native_mask,payload_mask
from diffusion_hash_inv.runner import _codec

@pytest.mark.parametrize('rep',['bits','bgv','cgge'])
def test_native_masks_deterministic_fraction_metadata_and_counterfactual(rep):
    encoder,_,shape=_codec(rep)
    condition=encoder.encode(b'ABCD').flatten()
    assert native_mask(rep,1).all() and not native_mask(rep,0).any()
    previous=torch.zeros(shape,dtype=torch.bool)
    for fraction in sorted(LEVELS.values()):
        mask=native_mask(rep,fraction)
        assert torch.equal(mask,native_mask(rep,fraction))
        assert not (previous & ~mask).any()
        previous=mask
        if fraction==1:continue
        size=8 if rep=='bits' else 128 if rep=='bgv' else 64
        assert mask.sum()==int(31*fraction)*size
        if rep=='bits':assert not mask[0].any()
        else:
            assert not mask[1].any()
            if rep=='bgv':assert not mask[0,:8,:16].any()
            else:assert not mask[0,24:32,56:64].any()
        changed=condition.clone();changed[~mask.flatten()]=1-changed[~mask.flatten()]
        assert torch.equal(condition*mask.flatten(),changed*mask.flatten())
    for message in (b'ABCD',b'x'*31):
        assert not (encoder.encode(message)*native_mask(rep,0)).any()
    assert payload_mask(rep,4).sum()==4*size
