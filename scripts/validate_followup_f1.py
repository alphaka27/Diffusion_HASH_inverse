"""F1 design arithmetic and synthetic calibration only; no model or MD5 evaluation.

Run: .venv/bin/python scripts/validate_followup_f1.py
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/followup-f1-protocol.json"


def interval(n, total, squares, tail_alpha=.0125):
    """Maurer-Pontil Thm 4, rescaled from [0,1] to [-1,1]."""
    if n < 2 or not 0 < tail_alpha < 1:
        raise ValueError("Invalid count or tail probability")
    estimate = np.asarray(total, dtype=float) / n
    variance = np.maximum(0, (np.asarray(squares) - np.asarray(total)**2 / n) / (n - 1))
    log_factor = math.log(2 / tail_alpha)
    radius = np.sqrt(2 * variance * log_factor / n) + 14 * log_factor / (3 * (n - 1))
    return np.maximum(-1, estimate - radius), np.minimum(1, estimate + radius)


def decision(intervals, delta=.0025):
    lr, ur = intervals["Random"]
    ls, us = intervals["Shuffled"]
    if lr > delta and ls > delta:
        return "CONFIRMED_MEANINGFUL_GAIN"
    if ur < delta and us < delta:
        return "EXCLUDED_BOTH"
    if us < delta:
        return "EXCLUDED_CONDITION_GAIN"
    if ur < delta:
        return "EXCLUDED_RANDOM_ADVANTAGE"
    return "INCONCLUSIVE"


def main():
    config = json.loads(CONFIG.read_text())
    n, k = config["design"]["trials"], config["design"]["k"]
    alpha, delta = config["inference"]["tail_alpha"], config["inference"]["delta"]
    assert n == 393216 and k == 100 and config["design"]["looks"] == 1
    assert config["design"]["extensions"] == 0 and config["rng"]["evaluation_seed"] is None
    assert 4 * alpha == config["inference"]["family_alpha"] == .05
    lo, hi = interval(10, 0., 0., alpha)
    assert lo < 0 < hi  # Zero observed variance must not yield a zero-width interval.
    assert decision({"Random": (-.001, .001), "Shuffled": (-.001, .001)}) == "EXCLUDED_BOTH"
    assert decision({"Random": (0., delta), "Shuffled": (0., delta)}) == "INCONCLUSIVE"
    assert decision({"Random": (.003, .006), "Shuffled": (.003, .006)}) == "CONFIRMED_MEANINGFUL_GAIN"
    fixture = np.array([-1., 0., 1., 0., 1.])
    log_factor = math.log(2 / alpha)
    radius = math.sqrt(2 * fixture.var(ddof=1) * log_factor / len(fixture)) + 14 * log_factor / (3 * (len(fixture)-1))
    lo, hi = interval(len(fixture), fixture.sum(), fixture @ fixture, alpha)
    assert np.allclose([lo, hi], [max(-1, fixture.mean()-radius), min(1, fixture.mean()+radius)])

    provenance = {}
    for item in config["provenance"]["checkpoints"]:
        path = ROOT / item["path"]
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == item["sha256"], path
            provenance[item["path"]] = "VERIFIED"
        else:
            provenance[item["path"]] = "NOT_AVAILABLE_LOCAL"
    groups = ROOT / config["provenance"]["groups_path"]
    if groups.exists():
        assert hashlib.sha256(groups.read_bytes()).hexdigest() == config["provenance"]["groups_sha256"]

    p0 = 1 - (1 - 2**-12)**k
    # Exact eight-state multinomial draws retain shared-Main comparator covariance.
    states = np.indices((2, 2, 2)).reshape(3, -1).T  # Main, Shuffled, Random

    def independent(p):
        return np.prod(np.where(states, p, 1-np.asarray(p)), axis=1)

    scenarios = {
        "zero_gain": independent([p0]*3),
        "small_gain_0.125pp": independent([p0+.00125, p0, p0]),
        "boundary_0.25pp": independent([p0+delta, p0, p0]),
        "gain_0.5pp": independent([p0+.005, p0, p0]),
        "prior_only_gain": independent([p0+.005, p0+.005, p0]),
        "one_checkpoint_gain_boundary": (independent([p0+3*delta,p0,p0]) + 2*independent([p0]*3))/3,
        "heterogeneous_targets_zero_gain": .9*independent([.005]*3) + .1*independent([(p0-.9*.005)/.1]*3),
    }
    exclusive = np.zeros(8)
    exclusive[0] = 1 - 3*p0
    exclusive[[1, 2, 4]] = p0
    scenarios["mutually_exclusive_hits_zero_gain"] = exclusive
    generator = np.random.default_rng(2026092901)  # Calibration namespace only, never the test seed.
    reps, simulations = 20000, {}
    for name, probabilities in scenarios.items():
        assert np.isclose(probabilities.sum(), 1) and (probabilities >= 0).all()
        counts = generator.multinomial(n, probabilities, size=reps)
        limits, truth = {}, {}
        miss = np.zeros(reps, dtype=bool)
        for method, column in (("Random", 2), ("Shuffled", 1)):
            differences = states[:, 0] - states[:, column]
            lo, hi = interval(n, counts @ differences, counts @ differences**2, alpha)
            true_effect = float(probabilities @ differences)
            truth[method], limits[method] = true_effect, (lo, hi)
            miss |= (lo > true_effect) | (hi < true_effect)
        lr, ur = limits["Random"]
        ls, us = limits["Shuffled"]
        verdicts = [decision({m: (bounds[0][i], bounds[1][i]) for m, bounds in limits.items()}, delta) for i in range(reps)]
        unique, tally = np.unique(verdicts, return_counts=True)
        simulations[name] = {"true_effects": truth, "joint_interval_miss_count": int(miss.sum()),
            "decisions": dict(zip(unique.tolist(), tally.tolist())),
            "positive_vs_zero_count": int(((lr > 0) & (ls > 0)).sum()),
            "both_upper_below_delta_count": int(((ur < delta) & (us < delta)).sum())}
        assert miss.mean() <= .06, (name, "Implementation diagnostic failed; do not change alpha")
    assert simulations["zero_gain"]["both_upper_below_delta_count"] / reps >= .95
    assert simulations["gain_0.5pp"]["decisions"].get("CONFIRMED_MEANINGFUL_GAIN", 0) / reps >= .95

    # Completed V5 end-to-end stream observations, not the optimistic sampler profile.
    learned_cps, random_cps, bytes_per_row = 748.0409428951313, 126700.09833688081, 63.355
    learned_rows, random_rows = 2*n*k, n*k
    hours = (learned_rows/learned_cps + random_rows/random_cps)/3600
    variance = 2*p0*(1-p0)
    nominal_radius = math.sqrt(2*variance*log_factor/n) + 14*log_factor/(3*(n-1))
    result = {"scope": "PLAN_DESIGN_ONLY_NO_TEST_SCHEDULE_OR_MODEL_SAMPLING",
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(), "p0": p0,
        "nominal_null_radius_percentage_points": 100*nominal_radius,
        "candidates": {"learned": learned_rows, "random": random_rows, "total": learned_rows+random_rows},
        "recorded_v5_cost_inputs": {"learned_cps": learned_cps, "random_cps": random_cps, "bytes_per_ledger_row": bytes_per_row},
        "estimated_evaluation_hours": hours, "estimate_times_1_5_hours": 1.5*hours,
        "estimated_ledger_gib": (learned_rows+random_rows)*bytes_per_row/1024**3,
        "repetitions_per_scenario": reps, "calibration": simulations,
        "provenance_checks": provenance,
        "pending": ["production evaluator implementation", "end-to-end preflight", "independent review", "immutable registration receipt", "fresh evaluation root seed"]}
    output = ROOT / "local_experiment_archive/analyses/followup-f1-design-20260929/design.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "provenance_checks"}, indent=2))


if __name__ == "__main__":
    main()
