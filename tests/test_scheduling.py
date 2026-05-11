import numpy as np

from diffusion_hash_inv.scheduling import BetaScheduler


def test_base_betas_matches_notebook_linspace() -> None:
    scheduler = BetaScheduler(beta_min=1e-4, beta_max=2e-2)

    betas = scheduler.base_betas(4)

    np.testing.assert_allclose(
        betas,
        np.linspace(start=1e-4, stop=2e-2, num=4, dtype=np.float64),
    )


def test_approach1_multiplies_base_betas_and_rescales() -> None:
    scheduler = BetaScheduler(beta_min=1e-4, beta_max=2e-2)
    mean = np.asarray([1.0, 3.0, 6.0, 10.0])
    base = np.linspace(start=1e-4, stop=2e-2, num=4, dtype=np.float64)
    raw = np.multiply(mean, base)
    expected_rescaled = (raw - np.min(raw)) / (np.max(raw) - np.min(raw))
    expected_rescaled = expected_rescaled * (scheduler.beta_max - scheduler.beta_min)
    expected_rescaled = expected_rescaled + scheduler.beta_min
    expected_rescaled = np.clip(expected_rescaled, scheduler.beta_min, scheduler.beta_max)

    result = scheduler.approach1(mean)

    np.testing.assert_allclose(result.base_betas, base)
    np.testing.assert_allclose(result.raw_candidate, raw)
    np.testing.assert_allclose(result.rescaled_candidate, expected_rescaled)


def test_approach2_maps_mean_sn_values_with_linear_equation() -> None:
    scheduler = BetaScheduler(beta_min=1e-4, beta_max=2e-2)
    mean = np.asarray([1.0, 3.0, 6.0, 10.0])
    slope = (scheduler.beta_max - scheduler.beta_min) / (np.max(mean) - np.min(mean))
    expected = slope * (mean - np.min(mean)) + scheduler.beta_min

    result = scheduler.approach2(mean)

    assert result.slope == slope
    assert result.sn_min == np.min(mean)
    assert result.sn_max == np.max(mean)
    np.testing.assert_allclose(result.candidate, expected)


def test_constant_values_map_to_beta_midpoint() -> None:
    scheduler = BetaScheduler(beta_min=1e-4, beta_max=2e-2)
    mean = np.asarray([5.0, 5.0, 5.0])
    midpoint = (scheduler.beta_min + scheduler.beta_max) / 2

    approach1 = scheduler.approach1(mean, base_betas=np.ones_like(mean))
    approach2 = scheduler.approach2(mean)

    np.testing.assert_allclose(approach1.rescaled_candidate, np.full_like(mean, midpoint))
    np.testing.assert_allclose(approach2.candidate, np.full_like(mean, midpoint))
    assert approach2.slope == 0.0
