import hashlib
import pytest
from diffusion_hash_inv.toy_hash import reference_prefix,toy_domain,oracle_sets,audit_oracle
from diffusion_hash_inv.dataset import digest_prefix_hex

@pytest.mark.parametrize('algorithm,width',[('md5',128),('sha256',256)])
def test_independent_prefix_bit_order_all_widths(algorithm,width):
    for message in (b'',b'ABCD',b'\x00\xff',b'\x80\x01'):
        digest=hashlib.new(algorithm,message).digest()
        for q in range(1,width+1):
            assert reference_prefix(message,algorithm,q)==int(digest_prefix_hex(digest,q),16)
        assert reference_prefix(message,algorithm,1)==digest[0]>>7
        assert reference_prefix(message,algorithm,9)==digest[0]*2+(digest[1]>>7)
        assert reference_prefix(message,algorithm,width)==int.from_bytes(digest,'big')
    with pytest.raises(ValueError):reference_prefix(b'ABCD',algorithm,width+1)


def test_exhaustive_oracle_collision_membership_agrees_with_verifier():
    domain=toy_domain();assert len(domain)==len(set(domain))==256
    oracle=oracle_sets(domain,'md5',8)
    audit=audit_oracle(domain,oracle,'md5',8)
    assert audit['agreement'] and audit['collision_groups']>0
    assert set().union(*oracle.values())==set(domain)
    assert sum(map(len,oracle.values()))==len(domain)
