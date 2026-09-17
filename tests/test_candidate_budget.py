import pytest
from diffusion_hash_inv.candidate_budget import baseline_candidates,outcomes,compare_target_rows


def test_budget_invalid_and_duplicate_consumption():
    rows=[dict(target_prefix='x',candidate_index=i,candidate_hex='41414141' if i<2 else None,valid=i<2,hash_prefix_match=False,exact_original=False) for i in range(3)]
    values=outcomes(rows,['x'],3)
    assert values[0]['valid_ratio']==2/3 and values[0]['unique_ratio']==1/3
    assert values[0]['duplicate_ratio']==1/3 and not values[0]['success']
    with pytest.raises(ValueError,match='exactly K'):outcomes(rows,['x'],4)
    with pytest.raises(ValueError):outcomes(rows+rows[:1],['x'],3)


def test_baseline_target_blind_and_fixed_budget():
    import inspect
    assert list(inspect.signature(baseline_candidates).parameters)==['method','n','k','seed']
    for method in ('uniform_random','source_prior'):
        a=baseline_candidates(method,3,10,42)
        assert a==baseline_candidates(method,3,10,42)
        assert len(a)==3 and all(len(group)==10 for group in a)
        assert all(len(x)==4 and set(x)<=set(b'ABCD') for group in a for x in group)


def test_paired_target_accounting():
    a=[dict(target_prefix='a',success=True),dict(target_prefix='b',success=False)]
    with pytest.raises(ValueError,match='Unpaired'):compare_target_rows(a,list(reversed(a)))
    result=compare_target_rows(a,a)
    assert result['absolute_gain']==0 and result['delta_ci95']==(0,0) and result['mcnemar_pvalue']==1


def test_zero_success_bound_is_not_zero_probability():
    from diffusion_hash_inv.candidate_budget import aggregate
    row=dict(success=False,exact=False,valid_ratio=1,unique_ratio=1,duplicate_ratio=0,candidates_to_first_match=None)
    assert aggregate([row]*18)['zero_success_upper95']==3/18
