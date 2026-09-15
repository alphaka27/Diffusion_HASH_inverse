"""Exhaustive one-byte verifier sanity check."""

from diffusion_hash_inv.dataset import build_digest_records
from diffusion_hash_inv.sanity import exhaustive_sanity_search


def test_exhaustive_printable_one_byte_search_recovers_its_target() -> None:
    target = build_digest_records((b"A",), source="printable", algorithm="md5", q=128)
    result = exhaustive_sanity_search(target, max_length=1)[0]
    assert result.candidates_examined == 94
    assert result.matching_messages == (b"A",)
