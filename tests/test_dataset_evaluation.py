"""Dataset split and fixed-budget verifier checks."""

from diffusion_hash_inv.dataset import (
    SourceSpec,
    build_digest_records,
    digest_prefix_hex,
    generate_source_messages,
    select_digest_representatives,
    split_digest_groups,
    split_validation_report,
)
from diffusion_hash_inv.evaluation import CandidateAttempt, holm_adjust, paired_comparison, score_attempts


def test_digest_group_split_is_deterministic_and_has_no_leakage() -> None:
    spec = SourceSpec("printable", 300, seed=7)
    records = build_digest_records(generate_source_messages(spec), source="printable", algorithm="md5", q=8)
    split = split_digest_groups(records, seed=11)
    seen = {}
    for name, partition in split.items():
        for record in partition:
            assert seen.setdefault(record.prefix, name) == name
    assert sum(map(len, split.values())) == len(records)
    assert split == split_digest_groups(records, seed=11)
    assert split_validation_report(split)["passed"]
    representatives = select_digest_representatives(split["test"])
    assert len(representatives) == len({record.prefix for record in split["test"]})
    assert len(representatives) < len(split["test"])


def test_prefix_supports_non_nibble_lengths_and_invalid_attempts_consume_k() -> None:
    records = build_digest_records((b"ABCD",), source="printable", algorithm="md5", q=7)
    target = records[0]
    assert digest_prefix_hex(target.digest, 7) == target.prefix
    result = score_attempts(
        records,
        ((CandidateAttempt(None, False, "invalid_decode"), CandidateAttempt(b"ABCD", True)),),
        k=2,
    )
    assert result.hash_verification_count == 1
    assert result.valid_decode_rate == 0.5
    assert result.preimage_success_at_k == result.exact_source_recovery_at_k == 1.0
    assert result.length_matched_preimage_success_at_k == 1.0


def test_paired_statistics_keep_target_pairing() -> None:
    result = paired_comparison((True,) * 6, (False,) * 6, bootstrap_samples=100)
    assert result.n10 == 6 and result.n01 == 0
    assert result.mcnemar_pvalue == 1 / 64
    assert result.delta_ci95 == (1.0, 1.0)
    assert holm_adjust({"random": result.mcnemar_pvalue}) == {"random": result.mcnemar_pvalue}
