"""V5 decisions shared by production outcomes and count-based calibration."""
import math
from statistics import NormalDist

import numpy as np

DELTA = .0025
Z_C = NormalDist().inv_cdf(1 - .05 / 8)
Z_R = NormalDist().inv_cdf(.975)
Z_B = NormalDist().inv_cdf(1 - .01 / 9)
P0 = 1 - (1 - 2**-12)**100


def _beta_cdf(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1 - _beta_cdf(1 - x, b, a)
    tiny = 1e-300
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / max(d, tiny)
    h = d
    for m in range(1, 10001):
        for numerator in (m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m)),
                          -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))):
            d = 1 + numerator * d
            c = 1 + numerator / c
            d = 1 / (d if abs(d) > tiny else tiny)
            c = c if abs(c) > tiny else tiny
            change = d * c
            h *= change
        if abs(change - 1) < 3e-14:
            return math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)) * h / a
    raise ArithmeticError("Incomplete beta did not converge")


def cp_bound(successes, total, tail=.05, *, upper=False):
    """One-sided Clopper-Pearson bound, including the zero/all edge cases."""
    if not isinstance(successes, (int, np.integer)) or not isinstance(total, (int, np.integer)) or not 0 <= successes <= total or total <= 0 or not 0 < tail < 1:
        raise ValueError("Invalid binomial counts/confidence")
    if upper and successes == total:
        return 1.0
    if not upper and successes == 0:
        return 0.0
    if upper and successes == 0:
        return -math.expm1(math.log(tail) / total)
    if not upper and successes == total:
        return math.exp(math.log(tail) / total)
    a, b, target = (successes + 1, total - successes, 1 - tail) if upper else (successes, total - successes + 1, tail)
    lo, hi = 0.0, 1.0
    for _ in range(65):
        mid = (lo + hi) / 2
        if _beta_cdf(mid, a, b) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def interval_from_moments(total, sum_d, sum_d2, z=Z_C):
    if total < 2 or not np.isfinite([sum_d, sum_d2]).all():
        raise ValueError("At least two complete, finite trials required")
    estimate = sum_d / total
    variance = max(0.0, (sum_d2 - sum_d**2 / total) / (total - 1))
    se = math.sqrt(variance / total)
    return {"estimate": estimate, "se": se, "lower": estimate - z * se, "upper": estimate + z * se, "trials": total}


def compare(main, control, z=Z_C):
    main, control = np.asarray(main), np.asarray(control)
    if main.ndim != 2 or main.shape != control.shape or not np.isin(main, [0, 1]).all() or not np.isin(control, [0, 1]).all():
        raise ValueError("Outcomes must be matching, complete seed x trial binary matrices")
    difference = (main.astype(float) - control).mean(axis=0)
    result = interval_from_moments(len(difference), float(difference.sum()), float(difference @ difference), z)
    result["seeds"] = [{"estimate": float((m.astype(float) - c).mean()),
                        "main": int(m.sum()), "control": int(c.sum()),
                        "n11": int(((m == 1) & (c == 1)).sum()), "n10": int(((m == 1) & (c == 0)).sum()),
                        "n01": int(((m == 0) & (c == 1)).sum()), "n00": int(((m == 0) & (c == 0)).sum())}
                       for m, c in zip(main, control, strict=True)]
    return result


def decide(comparisons, look=1, replication=False):
    if set(comparisons) != {"Random", "Shuffled"} or look not in (1, 2):
        raise ValueError("Both registered controls and a valid look required")
    if all(c["lower"] > 0 for c in comparisons.values()):
        return "SUPPORTED" if replication else "POSITIVE"
    if replication:
        return "NOT_ESTABLISHED_NOT_REPLICATED"
    bounded = {k: c["upper"] < DELTA for k, c in comparisons.items()}
    if bounded["Shuffled"]:
        return "REJECTED_BOUNDED" if bounded["Random"] else "REJECTED_NO_CONDITION_GAIN"
    if bounded["Random"]:
        return "REJECTED_NO_RANDOM_ADVANTAGE"
    return "EXTEND" if look == 1 else "NOT_ESTABLISHED_UNRESOLVED"


def stage_c(main, random, shuffled, look=1, replication=False):
    expected = 3 if look == 1 or replication else 6
    if len(np.shape(main)) != 2 or np.shape(main)[0] != expected:
        raise ValueError("Stage C/R seed count does not match registered look")
    comparisons = {name: compare(main, arr, Z_R if replication else Z_C) for name, arr in (("Random", random), ("Shuffled", shuffled))}
    return {"decision": decide(comparisons, look, replication), "look": look, "comparisons": comparisons}


def probe(values, threshold=Z_B):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Incomplete CLP probe")
    interval = interval_from_moments(len(values), float(values.sum()), float(values @ values), threshold)
    z = interval["estimate"] / interval["se"] if interval["se"] else None
    return {**interval, "z": z, "positive": interval["lower"] > 0, "threshold": threshold}


def ladder_cell(main, random, mc, clp=None):
    comparisons = {name: compare(np.asarray(main)[None, :], np.asarray(arr)[None, :], Z_B) for name, arr in (("Random", random), ("MC", mc))}
    return {"GEN": all(v["lower"] > 0 for v in comparisons.values()), "INFO": clp["positive"] if clp else None,
            "comparisons": comparisons, "clp": clp}


def compute_advantage(c3, main_hits, main_count, random_hits, random_count, md5_throughput, model_throughput):
    arithmetic = {"training_md5_calls": 40000 * 256 / .6875, "lookup_all": 4096 * sum(1/i for i in range(1, 4097)),
                  "lookup_test": 4096 * sum(1/i for i in range(1, 1025)), "amortized_advantage": False}
    if c3 != "SUPPORTED":
        return {**arithmetic, "decision": "NO_ADVANTAGE"}
    if min(md5_throughput, model_throughput, main_count, random_count) <= 0:
        raise ValueError("C4 requires measured throughput and candidate counts")
    lower = cp_bound(main_hits, main_count, .025)
    rho = md5_throughput / model_throughput
    threshold = rho * random_hits / random_count
    return {**arithmetic, "decision": "PER_QUERY_ADVANTAGE" if lower > threshold else "NO_ADVANTAGE",
            "rho": rho, "candidate_lower": lower, "threshold": threshold}


def final_decision(c1, c3):
    if c1 == "PASS" and c3 == "SUPPORTED":
        return "FINAL_SUPPORTED"
    if c1 == "PASS" and c3.startswith("REJECTED_"):
        return "FINAL_REJECTED"
    return "FINAL_NOT_ESTABLISHED"


def _sum_distribution(probabilities):
    result = np.array([1.0])
    for p in probabilities:
        result = np.convolve(result, [1 - p, p])
    return result


def calibrate(repetitions=20000, trials=65536):
    """Exact multinomial sufficient statistics, including paired extension trials."""
    from .data import rng
    generator = rng("production-calibration")
    states = np.indices((4, 4, 4)).reshape(3, -1).T
    vectors = {"Random": (states[:, 0] - states[:, 2]) / 3, "Shuffled": (states[:, 0] - states[:, 1]) / 3}
    scenarios = {
        "null": ([P0]*3, [P0]*3, [P0]*3),
        "half_delta": ([P0+.00125]*3, [P0]*3, [P0]*3),
        "delta": ([P0+DELTA]*3, [P0]*3, [P0]*3),
        "double_delta": ([P0+.005]*3, [P0]*3, [P0]*3),
        "one_seed_null": ([P0+.005, P0+.005, P0], [P0]*3, [P0]*3),
        "prior_gain": ([P0+DELTA]*3, [P0+DELTA]*3, [P0]*3),
        "memorization": ([P0-.003]*3, [P0]*3, [P0]*3),
    }
    results = {}
    for name, probabilities in scenarios.items():
        distributions = [_sum_distribution(p) for p in probabilities]
        probability = np.prod(np.stack([d[states[:, i]] for i, d in enumerate(distributions)]), axis=0)
        probability /= probability.sum()
        tally, extensions, replicated = {}, 0, 0
        for _ in range(repetitions):
            counts = generator.multinomial(trials, probability)
            intervals = {k: interval_from_moments(trials, float(counts @ v), float(counts @ (v*v))) for k, v in vectors.items()}
            decision = decide(intervals)
            if decision == "EXTEND":
                extensions += 1
                joint = np.stack([generator.multinomial(int(n), probability) for n in counts])
                intervals = {}
                for key, v in vectors.items():
                    combined = (v[:, None] + v[None, :]) / 2
                    intervals[key] = interval_from_moments(trials, float((joint * combined).sum()), float((joint * combined**2).sum()))
                decision = decide(intervals, 2)
            tally[decision] = tally.get(decision, 0) + 1
            if decision == "POSITIVE":
                rcounts = generator.multinomial(trials, probability)
                ri = {k: interval_from_moments(trials, float(rcounts @ v), float(rcounts @ (v*v)), Z_R) for k, v in vectors.items()}
                replicated += decide(ri, replication=True) == "SUPPORTED"
        pos = tally.get("POSITIVE", 0)
        bounded = tally.get("REJECTED_BOUNDED", 0)
        results[name] = {"counts": tally, "extensions": extensions, "replicated": replicated,
                         "positive_lower": cp_bound(pos, repetitions), "positive_upper": cp_bound(pos, repetitions, upper=True),
                         "bounded_lower": cp_bound(bounded, repetitions)}
    passed = results["null"]["positive_upper"] <= .01 and results["delta"]["positive_lower"] >= .95 and results["null"]["bounded_lower"] >= .95
    return {"passed": passed and repetitions >= 20000 and trials == 65536, "repetitions": repetitions, "trials": trials, "scenarios": results}
