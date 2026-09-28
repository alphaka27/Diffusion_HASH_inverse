"""v4 simultaneous exact paired intervals and production calibration."""
import json
import math
from functools import lru_cache

import numpy as np

TRIALS = 16_384
REPETITIONS = 20_000
DELTA = .01
TAIL = .05 / 24  # Six differences, two discordance probabilities, two tails.
P0 = -math.expm1(100 * math.log1p(-1 / 4096))


@lru_cache(maxsize=None)
def coefficients(n):
    k = np.arange(n + 1, dtype=float)
    log_choose = np.array([math.lgamma(n + 1) - math.lgamma(x + 1)
                           - math.lgamma(n - x + 1) for x in range(n + 1)])
    return k, log_choose


@lru_cache(maxsize=None)
def cp(x, n, tail=TAIL):
    """Invert the binomial CDF; return conservative Clopper-Pearson bounds."""
    if not (isinstance(x, int) and isinstance(n, int)
            and n > 0 and 0 <= x <= n and 0 < tail < .5):
        raise ValueError("invalid binomial count or tail probability")
    if x > n // 2:
        lo, hi = cp(n - x, n, tail)
        return 1 - hi, 1 - lo
    k, log_choose = coefficients(n)

    def invert(last, target):
        left, right = 0., 1.
        indices = k[:last + 1]
        for _ in range(48):
            p = (left + right) / 2
            cdf = np.exp(log_choose[:last + 1] + indices * math.log(p)
                         + (n - indices) * math.log1p(-p)).sum()
            if cdf > target:
                left = p
            else:
                right = p
        return (left + right) / 2

    lower = 0. if x == 0 else invert(x - 1, 1 - tail)
    upper = -math.expm1(math.log(tail) / n) if x == 0 else invert(x, tail)
    return lower, upper


def intervals(n10, n01, n=TRIALS):
    lookup = {int(x): cp(int(x), n) for x in np.unique([n10, n01])}
    a = np.array([lookup[int(x)] for x in n10.flat]).reshape(*n10.shape, 2)
    b = np.array([lookup[int(x)] for x in n01.flat]).reshape(*n01.shape, 2)
    return a[..., 0] - b[..., 1], a[..., 1] - b[..., 0]


def decisions(lower, upper):
    go = np.all(lower > DELTA, axis=1)
    stop = np.all(upper < DELTA, axis=1)
    unstable = np.any(upper < DELTA, axis=1) & ~stop
    return {"GO": go, "NO_GO_SMALL": stop, "NO_GO_REPRODUCIBILITY": unstable,
            "INCONCLUSIVE": ~(go | stop | unstable)}


def self_check():
    # NIST's published n=30, x=8, 95% exact interval, plus boundary cases.
    assert np.allclose(cp(8, 30, .025), (.122795, .458894), atol=5e-7)
    assert np.allclose(cp(0, 30, .025), (0, 1 - .025 ** (1 / 30)))
    assert np.allclose(cp(30, 30, .025), (.025 ** (1 / 30), 1))
    for bad in ((-1, 30), (31, 30), (0, 0)):
        try:
            cp(*bad)
        except ValueError:
            continue
        raise AssertionError("invalid count accepted")
    lo, hi = intervals(np.array([[0, 10]]), np.array([[0, 10]]), 30)
    assert np.allclose(lo, -hi) and np.all(lo < 0)
    lower = np.array([[.02] * 6, [-.01] * 6, [.02] * 5 + [-.01], [0.] * 6])
    upper = np.array([[.03] * 6, [.005] * 6, [.03] * 5 + [.005], [.02] * 6])
    assert all(value.sum() == 1 for value in decisions(lower, upper).values())


def simulate(strata, rng, repetitions=REPETITIONS):
    # Joint nine outcomes preserve shared Main and target difficulty across seeds.
    bits = ((np.arange(512)[:, None] >> np.arange(9)) & 1).astype(bool)
    probabilities = sum(weight * np.prod(np.where(bits, p, 1 - p), axis=1)
                        for weight, p in strata)
    assert np.isclose(probabilities.sum(), 1.)
    counts = rng.multinomial(TRIALS, probabilities, size=repetitions)
    n10, n01 = [], []
    for seed in range(3):
        main = bits[:, 3 * seed]
        for control in (3 * seed + 1, 3 * seed + 2):
            n10.append(counts[:, main & ~bits[:, control]].sum(axis=1))
            n01.append(counts[:, ~main & bits[:, control]].sum(axis=1))
    lower, upper = intervals(np.array(n10).T, np.array(n01).T)
    truth = sum(weight * np.array([p[3 * s] - p[3 * s + c]
                                  for s in range(3) for c in (1, 2)])
                for weight, p in strata)
    result = {name: float(value.mean()) for name, value in decisions(lower, upper).items()}
    result["simultaneous_coverage"] = float(np.all((lower <= truth) & (truth <= upper), axis=1).mean())
    result["true_differences"] = truth.tolist()
    result["GO_probability_lower95"] = cp(round(result["GO"] * repetitions), repetitions, .05)[0]
    result["NO_GO_SMALL_probability_lower95"] = cp(round(result["NO_GO_SMALL"] * repetitions), repetitions, .05)[0]
    covered = round(result["simultaneous_coverage"] * repetitions)
    result["coverage_lower95"] = cp(covered, repetitions, .05)[0]
    result["false_GO_upper95"] = cp(round(result["GO"] * repetitions), repetitions, .05)[1]
    wrong = decisions(lower, upper)
    wrong_count = int((wrong["GO"] | wrong["NO_GO_SMALL"] | wrong["NO_GO_REPRODUCIBILITY"]).sum())
    result["boundary_error_upper95"] = cp(wrong_count, repetitions, .05)[1]
    return result



def scenarios():
    base = np.array([P0, P0, P0] * 3)
    reference = np.array([2 * P0, P0, P0] * 3)
    weak = reference.copy()
    weak[6] = P0 + .005
    one_null = reference.copy()
    one_null[8] = 2 * P0
    scenarios = {
        "all_null": [(1., base)],
        "minimum_effect_boundary": [(1., np.array([P0 + DELTA, P0, P0] * 3))],
        "reference_twofold": [(1., reference)],
        "one_weak_seed": [(1., weak)],
        "one_null_comparison": [(1., one_null)],
        "stronger_shuffled": [(1., np.array([2 * P0, P0, 1.5 * P0] * 3))],
        "shared_difficulty": [(.5, reference * .2), (.5, reference * 1.8)],
        "opposite_effect_null": [(.5, np.array([.08, .02, .02] * 3)),
                                 (.5, np.array([.02, .08, .08] * 3))],
    }
    return scenarios


def analyze_counts(counts, n=TRIALS):
    """One production decision path, also exercised by ledger/calibration fixtures."""
    expected = {f"{s}/{c}" for s in range(3) for c in ("random", "shuffled")}
    if set(counts) != expected:
        return {"scientific_decision": "INVALID_OR_INCOMPLETE", "comparisons": {},
                "reason": "six registered comparisons required"}
    result, lower, upper = {}, [], []
    for key in sorted(expected):
        row = counts[key]
        if set(row) != {"n11", "n10", "n01", "n00"} or any(type(x) is not int or x < 0 for x in row.values()) or sum(row.values()) != n:
            raise ValueError("invalid paired outcome counts")
        a, b = cp(row["n10"], n), cp(row["n01"], n)
        lo, hi = a[0] - b[1], a[1] - b[0]
        lower.append(lo)
        upper.append(hi)
        main, control = (row["n11"] + row["n10"]) / n, (row["n11"] + row["n01"]) / n
        result[key] = {**row, "delta": (row["n10"] - row["n01"]) / n,
                       "lower": lo, "upper": hi, "main_success_at_100": main,
                       "control_success_at_100": control, "relative_lift": main / control if control else None}
    decision = next(k for k, v in decisions(np.array([lower]), np.array([upper])).items() if v[0])
    return {"scientific_decision": decision, "comparisons": result, "n": n,
            "delta_threshold": DELTA, "simultaneous_confidence_at_least": .95, "tail_error": TAIL}


def calibrate():
    self_check()
    results = {name: simulate(strata, np.random.default_rng(2026092804 + i))
               for i, (name, strata) in enumerate(scenarios().items())}
    checks = {name + "/coverage": row["coverage_lower95"] >= .94 for name, row in results.items()}
    for name, row in results.items():
        if min(row["true_differences"]) <= DELTA + 1e-12:
            checks[name + "/false_go"] = row["false_GO_upper95"] <= .06
    checks["boundary_error"] = results["minimum_effect_boundary"]["boundary_error_upper95"] <= .06
    checks["reference_power"] = results["reference_twofold"]["GO_probability_lower95"] >= .80
    checks["null_exclusion"] = results["all_null"]["NO_GO_SMALL_probability_lower95"] >= .80
    # The scalar report path must agree with the vectorized simulation path.
    for i in range(30):
        rng = np.random.default_rng(2026092804 + i)
        values = rng.multinomial(TRIALS, [.001, .04, .02, .939], size=6)
        counts = {f"{s}/{c}": dict(zip(("n11", "n10", "n01", "n00"), map(int, values[2*s+j])))
                  for s in range(3) for j, c in enumerate(("random", "shuffled"))}
        lo, hi = intervals(values[None, :, 1], values[None, :, 2])
        expected = next(k for k, v in decisions(lo, hi).items() if v[0])
        assert analyze_counts(counts)["scientific_decision"] == expected
    return {"status": "PASS" if all(checks.values()) else "FAIL", "repetitions": REPETITIONS,
            "scenarios": results, "checks": checks}
