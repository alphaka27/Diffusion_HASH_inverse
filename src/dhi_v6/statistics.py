"""V6 순차 판정. 구간·CP 한계·CLP 통계는 dhi_v5/statistics.py에서 복사했다."""
import math
from statistics import NormalDist

import numpy as np

from .data import rng
from .protocol import registration

DELTA = .005
P0 = 1 - (1 - 2**-12)**100
Z_LOOK = tuple(NormalDist().inv_cdf(1 - .0025 * share) for share in (.1, .4, .5))
Z_C = Z_LOOK[0]
Z_P = NormalDist().inv_cdf(1 - .01 / 5)
Z_CONTRAST = NormalDist().inv_cdf(1 - .05 / 24)
Z_S = NormalDist().inv_cdf(1 - .0005)
PIPELINES = tuple(registration()["pipeline_order"])



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


def probe(values, threshold=Z_P):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Incomplete CLP probe")
    interval = interval_from_moments(len(values), float(values.sum()), float(values @ values), threshold)
    z = interval["estimate"] / interval["se"] if interval["se"] else None
    return {**interval, "z": z, "positive": interval["lower"] > 0, "threshold": threshold}


def classify(est, se, z, final):
    est, se = np.asarray(est), np.asarray(se)
    if est.ndim != 2 or est.shape[1] != 2 or se.shape != est.shape or not np.isfinite([est, se]).all() or np.any(se < 0):
        raise ValueError("Two finite comparison estimates and standard errors required")
    lo, hi = est - z * se, est + z * se
    positive, bounded = (lo > 0).all(axis=1), hi < DELTA
    verdict = np.where(positive, "POSITIVE", np.where(bounded.all(axis=1), "REJECTED_BOUNDED", ""))
    if final:
        verdict = np.where(verdict != "", verdict, np.where(bounded[:, 1], "REJECTED_NO_CONDITION_GAIN",
                            np.where(bounded[:, 0], "REJECTED_NO_RANDOM_ADVANTAGE", "NOT_ESTABLISHED_UNRESOLVED")))
    return verdict, hi


def joint_decision(est, se, look, *, budget_stop=False):
    """Shared production/calibration rule; arrays are pipeline x control."""
    if look not in (1, 2, 3) or not len(est):
        raise ValueError("Active pipelines and a registered look required")
    interim, _ = classify(est, se, Z_LOOK[look - 1], False)
    stop = budget_stop or look == 3 or np.all(interim != "")
    final, _ = classify(est, se, Z_LOOK[look - 1], True)
    decisions = final if (budget_stop or look == 3) else interim
    if budget_stop:
        decisions = np.where(decisions == "NOT_ESTABLISHED_UNRESOLVED", "NOT_ESTABLISHED_BY_BUDGET", decisions)
    return bool(stop), decisions.tolist()


def stage_c(outcomes, look, *, budget_stop=False):
    if not outcomes:
        raise ValueError("No active pipelines")
    shapes = {np.asarray(v["Main"]).shape for v in outcomes.values()}
    if len(shapes) != 1 or next(iter(shapes))[0] != 3:
        raise ValueError("Stage C requires three seeds and common trial counts")
    comparisons = {p: {c: compare(v["Main"], v[c], Z_LOOK[look - 1]) for c in ("Random", "Shuffled")}
                   for p, v in outcomes.items()}
    est = np.array([[v[c]["estimate"] for c in ("Random", "Shuffled")] for v in comparisons.values()])
    se = np.array([[v[c]["se"] for c in ("Random", "Shuffled")] for v in comparisons.values()])
    stop, decisions = joint_decision(est, se, look, budget_stop=budget_stop)
    return {"look": look, "action": "stop" if stop else "continue", "budget_stop": budget_stop,
            "pipelines": {p: {"decision": decision or "UNDECIDED", "comparisons": comparisons[p]}
                          for p, decision in zip(outcomes, decisions, strict=True)}}


def replication(main, random, shuffled, entrants):
    if entrants not in range(1, 6) or np.asarray(main).shape[0] != 3:
        raise ValueError("Replication requires 1..5 entrants and three seeds")
    z = NormalDist().inv_cdf(1 - .025 / entrants)
    comparisons = {c: compare(main, arr, z) for c, arr in (("Random", random), ("Shuffled", shuffled))}
    decision = "SUPPORTED" if all(v["lower"] > 0 for v in comparisons.values()) else "NOT_ESTABLISHED_NOT_REPLICATED"
    return {"decision": decision, "z": z, "comparisons": comparisons}


def positive_control(main, random, mc, clp):
    comparisons = {c: compare(np.asarray(main)[None], np.asarray(arr)[None], Z_P)
                   for c, arr in (("Random", random), ("MC", mc))}
    return {"GEN_4": all(v["lower"] > 0 for v in comparisons.values()), "INFO_4": bool(clp["z"] is not None and clp["z"] > Z_P),
            "comparisons": comparisons, "clp": clp}


def contrasts(outcomes):
    results = []
    for a, b in registration()["contrasts"]:
        for control in ("Random", "Shuffled"):
            if a not in outcomes or b not in outcomes:
                results.append({"pair": [a, b], "control": control, "decision": "UNAVAILABLE"})
                continue
            differences = []
            for p in (a, b):
                main, other = np.asarray(outcomes[p]["Main"]), np.asarray(outcomes[p][control])
                compare(main, other)
                differences.append((main.astype(float) - other).mean(axis=0))
            if differences[0].shape != differences[1].shape:
                raise ValueError("Contrasts require common trials")
            d = differences[0] - differences[1]
            interval = interval_from_moments(len(d), float(d.sum()), float(d @ d), Z_CONTRAST)
            lo, hi = interval["lower"], interval["upper"]
            decision = "DIFFERENT" if lo > 0 or hi < 0 else ("EQUIVALENT_WITHIN_DELTA" if lo > -DELTA and hi < DELTA else "UNDETERMINED")
            results.append({"pair": [a, b], "control": control, "decision": decision, **interval})
    return results


def compute_advantage(c3, main_hits, main_count, random_hits, random_count, md5_throughput, model_throughput):
    if min(md5_throughput, model_throughput, main_count, random_count) <= 0:
        raise ValueError("C4 requires measured throughput and candidate counts")
    rho = md5_throughput / model_throughput
    arithmetic = {"training_md5_calls": 40000 * 256 / .6875, "lookup_all": 4096 * sum(1 / i for i in range(1, 4097)),
                  "amortized_advantage": False, "rho": rho, "break_even_prior": rho * 2**-12,
                  "arithmetic": "ARITHMETICALLY_IMPOSSIBLE" if rho * 2**-12 >= 1 else "POSSIBLE"}
    if c3 != "SUPPORTED":
        return {**arithmetic, "decision": "NO_ADVANTAGE"}
    lower, threshold = cp_bound(main_hits, main_count, .025), rho * random_hits / random_count
    return {**arithmetic, "candidate_lower": lower, "threshold": threshold,
            "decision": "PER_QUERY_ADVANTAGE" if lower > threshold else "NO_ADVANTAGE"}


def headline(verdicts):
    verdicts = list(verdicts)
    if any(v == "SUPPORTED" for v in verdicts):
        return "FINAL_SUPPORTED"
    rejected = [v.startswith("REJECTED_") for v in verdicts]
    if len(verdicts) == 5 and all(rejected):
        return "FINAL_REJECTED"
    return "FINAL_REJECTED_WITH_EXCEPTIONS" if any(rejected) else "FINAL_NOT_ESTABLISHED"


def calibrate(repetitions=2000, *, block=8192):
    if repetitions < 1 or block < 2:
        raise ValueError("Invalid calibration size")
    results = {}
    for scenario in ("all_null", "P-DISC_+delta", "R-G-BGV_prior_gain", "budget_after_look2"):
        generator = rng("production-calibration", scenario)
        positives = all_rejected = planted = replicated = 0
        stops = {}
        for _ in range(repetitions):
            sums = np.zeros((5, 2))
            squares = np.zeros_like(sums)
            max_looks = 2 if scenario == "budget_after_look2" else 3
            for look in range(1, max_looks + 1):
                random = {src: generator.random((3, block)) < P0 for src in ("P", "R")}
                for i, p in enumerate(PIPELINES):
                    gain = DELTA if ((scenario == "P-DISC_+delta" and p == "P-DISC") or
                                     (scenario == "R-G-BGV_prior_gain" and p == "R-G-BGV")) else 0
                    main = generator.random((3, block)) < P0 + gain
                    shuffled = generator.random((3, block)) < P0 + (gain if scenario == "R-G-BGV_prior_gain" else 0)
                    for j, control in enumerate((random[p[0]], shuffled)):
                        d = (main.astype(float) - control).mean(axis=0)
                        sums[i, j] += d.sum()
                        squares[i, j] += d @ d
                total = look * block
                est = sums / total
                se = np.sqrt(np.maximum(squares - sums**2 / total, 0) / (total - 1) / total)
                stop, decisions = joint_decision(est, se, look, budget_stop=look == max_looks and max_looks < 3)
                if stop:
                    break
            stops[str(look)] = stops.get(str(look), 0) + 1
            positives += "POSITIVE" in decisions
            all_rejected += all(v.startswith("REJECTED_") for v in decisions)
            planted += decisions[PIPELINES.index("P-DISC")] == "POSITIVE"
            entrants = sum(v == "POSITIVE" for v in decisions)
            for p, decision in zip(PIPELINES, decisions):
                if decision != "POSITIVE":
                    continue
                gain = DELTA if ((scenario == "P-DISC_+delta" and p == "P-DISC") or
                                 (scenario == "R-G-BGV_prior_gain" and p == "R-G-BGV")) else 0
                m = generator.random((3, 16384)) < P0 + gain
                r = generator.random((3, 16384)) < P0
                s = generator.random((3, 16384)) < P0 + (gain if scenario == "R-G-BGV_prior_gain" else 0)
                replicated += replication(m, r, s, entrants)["decision"] == "SUPPORTED"
        results[scenario] = {"any_positive": positives, "all_rejected": all_rejected, "planted_positive": planted,
                             "replicated": replicated, "stopping_look": stops,
                             "positive_upper": cp_bound(positives, repetitions, upper=True),
                             "all_rejected_lower": cp_bound(all_rejected, repetitions),
                             "planted_lower": cp_bound(planted, repetitions)}
    criteria = {"null_positive": results["all_null"]["positive_upper"] <= .025,
                "null_rejected": results["all_null"]["all_rejected_lower"] >= .95,
                "delta_positive": results["P-DISC_+delta"]["planted_lower"] >= .95}
    production = repetitions >= registration()["calibration_repetitions"] and block == registration()["stage_c"]["block"]
    return {"passed": production and all(criteria.values()), "production": production,
            "criteria_passed": all(criteria.values()), "scope": "production" if production else "regression-only",
            "repetitions": repetitions, "block": block, "criteria": criteria, "scenarios": results}
