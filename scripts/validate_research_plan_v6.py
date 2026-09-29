"""Design checks for RESEARCH_PLAN_V6.md.

Trains no model, generates no candidate and opens no MD5 test pool. It reads two local, gitignored inputs
when present and otherwise uses the values recorded on 2026-09-29 (RECORDED below, quoted in the plan):
  * local_experiment_archive/analyses/v6-design-20260929/throughput.json (random-weight MLX proxy),
  * local_experiment_archive/runs/v5-study-certified/C/... result.json files (V5 partial Stage C, W2).
It computes:
1. descriptive V5 partial Stage C summary and the observed V5 stream / training cost,
2. Monte Carlo operating characteristics of the V6 Stage C rule (5 pipelines, 3 looks, per-pipeline stopping),
   including the W4 replication branch and the overall headline verdict,
3. precision of the pre-specified pipeline contrasts and of the r=4 positive control,
4. compute-advantage arithmetic per pipeline,
5. run / candidate / time arithmetic per stage for Gaussian sampling steps 25 / 50 / 100.
"""
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "local_experiment_archive/analyses/v6-design-20260929"
V5_C = ROOT / "local_experiment_archive/runs/v5-study-certified/C/W2-r64/D1-T"
Q, K = 12, 100
P1 = 2 ** -Q
P0 = 1 - (1 - P1) ** K
DELTA = 0.005
SEEDS = 3
BLOCK = 8192
LOOK_TRIALS = (BLOCK, 2 * BLOCK, 3 * BLOCK)
ALPHA_SHARE = (0.1, 0.4, 0.5)
FALLBACK_BLOCK = 6144
PIPELINES = ("P-G-BGV", "P-G-CGGE", "P-DISC", "R-G-BGV", "R-DISC")
SOURCE = {name: name[0] for name in PIPELINES}
TAIL = 0.05 / (len(PIPELINES) * 2 * 2)  # 5 pipelines x 2 controls x 2 sides, Bonferroni
Z_LOOK = tuple(NormalDist().inv_cdf(1 - TAIL * share) for share in ALPHA_SHARE)
TRIALS_R = 2 * BLOCK
TRIALS_P = 4096
Z_P = NormalDist().inv_cdf(1 - 0.01 / len(PIPELINES))  # one-sided, Bonferroni over 5 pipelines
CONTRASTS = (("P-G-BGV", "P-G-CGGE"), ("P-G-BGV", "P-DISC"), ("P-G-CGGE", "P-DISC"),
             ("R-G-BGV", "R-DISC"), ("P-G-BGV", "R-G-BGV"), ("P-DISC", "R-DISC"))
Z_CONTRAST = NormalDist().inv_cdf(1 - 0.05 / (2 * 2 * len(CONTRASTS)))  # delta_R and delta_S contrasts
UPDATES, UPDATES_L, BATCH = 40000, 160000, 256
STREAM_EFFICIENCY = 748.04 / 1369.36  # slowest completed V5 learned stream / V5 A-prof D1-T sampler
TRAIN_OVERHEAD_S = 0.019  # V5 C: (active cap - evaluation stream time) / 6 runs - D1-T update time
RANDOM_CPS = 120000  # V5 real Random streams: 6,553,600 candidates in about 52 s
MD5_PER_S = 2104757  # V5 plan: hashlib + first 12 bits, Python, one core

RECORDED = {
    "throughput": {  # MLX 0.32.2, Apple M3 Max, float32, random weights (2026-09-29)
        "gaussian": {
            "bgv-w32": {"best_rows_per_second": 15267.05, "update": {"update_seconds": 0.11348}, "parameters": 910638},
            "bgv-w64": {"best_rows_per_second": 6094.76, "update": {"update_seconds": 0.22357}, "parameters": 2035950},
            "cgge-w32": {"best_rows_per_second": 30370.85, "update": {"update_seconds": 0.05739}, "parameters": 513326},
            "cgge-w64": {"best_rows_per_second": 11988.74, "update": {"update_seconds": 0.11597}, "parameters": 1245422},
        },
        "discrete": {"D1-S": [{"candidates_per_second": 20344.46, "parameters": 508668}],
                     "D1-T": [{"candidates_per_second": 1233.97, "parameters": 1866316}]},
    },
    "v5_a_prof": {"D1-S": {"update_seconds": 0.0017918, "candidates_per_second": 20732.78},
                  "D1-T": {"update_seconds": 0.0303812, "candidates_per_second": 1369.36},
                  "D1-T-L": {"update_seconds": 0.1027278, "candidates_per_second": 380.30}},
    "v5_partial_c": {  # method-seed: (success_at_100, trials, elapsed_seconds, hits, candidates)
        "Main-0": (1538, 65536, 7975.00, 1554, 6553600), "Main-1": (1626, 65536, 7880.55, 1643, 6553600),
        "Main-2": (1598, 65536, 7670.55, 1615, 6553600), "Shuffled-0": (1542, 65536, 7498.37, 1559, 6553600),
        "Shuffled-1": (1540, 65536, 8761.02, 1552, 6553600), "Random-0": (1601, 65536, 51.61, 1624, 6553600),
        "Random-1": (1597, 65536, 51.73, 1621, 6553600), "MC-0": (105, 4096, 453.39, 107, 409600),
        "MC-1": (99, 4096, 444.80, 99, 409600),
    },
    "v5_budget_c_seconds": 57600.88,
    "v5_shuffled2_partial_attempts": 4046848,
}
SOURCES = {}


def throughput():
    path = OUT / "throughput.json"
    SOURCES["throughput"] = "archive" if path.exists() else "recorded_2026-09-29"
    return json.loads(path.read_text()) if path.exists() else RECORDED["throughput"]


def v5_partial():
    rows = {}
    for key in RECORDED["v5_partial_c"]:
        path = V5_C / f"eval-{key}" / "result.json"
        if path.exists():
            r = json.loads(path.read_text())
            rows[key] = (r["success_at_100"], r["rows"] // K, r["elapsed_seconds"], r["hits"], r["rows"])
        else:
            rows[key] = RECORDED["v5_partial_c"][key]
    SOURCES["v5_partial_c"] = "archive" if (V5_C / "eval-Main-0/result.json").exists() else "recorded_2026-09-29"
    rate = {k: v[0] / v[1] for k, v in rows.items()}
    pooled = lambda keys: sum(rows[k][0] for k in keys) / sum(rows[k][1] for k in keys)  # noqa: E731
    main01, shuf01, rand01 = pooled(["Main-0", "Main-1"]), pooled(["Shuffled-0", "Shuffled-1"]), pooled(["Random-0", "Random-1"])
    n = 2 * 65536
    se = math.sqrt(2 * P0 * (1 - P0) / n)  # descriptive, independence approximation
    learned = [k for k in rows if k.split("-")[0] in ("Main", "Shuffled")]
    stream_seconds = sum(rows[k][2] for k in rows)
    shuffled2_seconds = RECORDED["v5_shuffled2_partial_attempts"] / (
        sum(rows[k][4] for k in learned) / sum(rows[k][2] for k in learned))
    training_per_run = (RECORDED["v5_budget_c_seconds"] - stream_seconds - shuffled2_seconds) / 6
    return {
        "success_at_100": {k: round(v, 5) for k, v in rate.items()},
        "main_all_three_seeds": round(pooled(["Main-0", "Main-1", "Main-2"]), 5),
        "paired_seeds_0_1": {"main": round(main01, 5), "shuffled": round(shuf01, 5), "random": round(rand01, 5),
                             "main_minus_random": round(main01 - rand01, 5), "main_minus_shuffled": round(main01 - shuf01, 5),
                             "descriptive_se": round(se, 5)},
        "hit_per_candidate": {k: round(v[3] / v[4], 7) for k, v in rows.items()},
        "learned_stream_candidates_per_second": {k: round(rows[k][4] / rows[k][2], 1) for k in learned},
        "random_stream_candidates_per_second": {k: round(rows[k][4] / rows[k][2]) for k in rows if k.startswith("Random")},
        "estimated_d1t_training_seconds_per_run": round(training_per_run),
        "estimated_training_overhead_seconds_per_update": round(training_per_run / UPDATES - RECORDED["v5_a_prof"]["D1-T"]["update_seconds"], 4),
        "note": "W2, descriptive only; Shuffled-2, Random-2, MC-2 incomplete when the 16 h cap ended V5 Stage C",
    }


def bern(rng, probs, reps, trials):
    probs = np.asarray(probs, dtype=np.float64)[None, :, None]
    return (rng.random((reps, len(probs[0]), trials)) < probs).astype(np.float32)


def classify(est, se, z, final):
    lo, hi = est - z * se, est + z * se  # arrays (reps, 2): column 0 Random, 1 Shuffled
    positive = (lo > 0).all(1)
    bounded = hi < DELTA
    both = bounded.all(1)
    verdict = np.where(positive, "POSITIVE", np.where(both, "REJECTED_BOUNDED", ""))
    if final:
        verdict = np.where(verdict != "", verdict, np.where(bounded[:, 1], "REJECTED_NO_CONDITION_GAIN",
                           np.where(bounded[:, 0], "REJECTED_NO_RANDOM_ADVANTAGE", "NOT_ESTABLISHED_UNRESOLVED")))
    return verdict, hi


def replicate(rng, spec, z):
    m = bern(rng, spec["Main"], 1, TRIALS_R)[0]
    for control in (bern(rng, spec["Shuffled"], 1, TRIALS_R)[0], bern(rng, spec["Random"], 1, TRIALS_R)[0]):
        d = (m - control).mean(0)
        if d.mean() - z * d.std(ddof=1) / math.sqrt(d.size) <= 0:
            return False
    return True


def headline(verdicts):
    if any(v == "SUPPORTED" for v in verdicts):
        return "FINAL_SUPPORTED"
    rejected = [v.startswith("REJECTED") for v in verdicts]
    if all(rejected):
        return "FINAL_REJECTED"
    return "FINAL_REJECTED_WITH_EXCEPTIONS" if any(rejected) else "FINAL_NOT_ESTABLISHED"


def stage_c(rng, spec, reps, block=BLOCK, max_looks=3, chunk=200):
    """Global stopping: all pipelines share the trial count. Stop at an interim look only when every pipeline
    is POSITIVE or REJECTED_BOUNDED; verdicts are those of the stopping look. max_looks < 3 emulates a budget
    stop after that look (full classification order at the last completed look).
    spec[pipeline] = {"Main": [3 probs], "Shuffled": [3 probs]}; spec["Random"] = {"P": [...], "R": [...]}."""
    per = {p: {} for p in PIPELINES}
    heads, stops, false_positive = {}, {}, 0
    null = [p for p in PIPELINES if not (np.mean(spec[p]["Main"]) > np.mean(spec["Random"][SOURCE[p]])
                                         and np.mean(spec[p]["Main"]) > np.mean(spec[p]["Shuffled"]))]
    for start in range(0, reps, chunk):
        n = min(chunk, reps - start)
        s1 = np.zeros((n, len(PIPELINES), 2))
        s2 = np.zeros((n, len(PIPELINES), 2))
        final = np.full((n, len(PIPELINES)), "", dtype=object)
        stop_look = np.zeros(n, dtype=int)
        for look in range(max_looks):
            z, t = Z_LOOK[look], block * (look + 1)
            rnd = {src: bern(rng, spec["Random"][src], n, block) for src in "PR"}
            interim = np.full((n, len(PIPELINES)), "", dtype=object)
            full = np.full((n, len(PIPELINES)), "", dtype=object)
            for i, p in enumerate(PIPELINES):
                m = bern(rng, spec[p]["Main"], n, block)
                for j, control in enumerate((rnd[SOURCE[p]], bern(rng, spec[p]["Shuffled"], n, block))):
                    d = (m - control).mean(1)
                    s1[:, i, j] += d.sum(1)
                    s2[:, i, j] += (d * d).sum(1)
                est = s1[:, i] / t
                se = np.sqrt(np.maximum(s2[:, i] - t * est ** 2, 0) / (t - 1) / t)
                interim[:, i], _ = classify(est, se, z, final=False)
                full[:, i], _ = classify(est, se, z, final=True)
            open_ = stop_look == 0
            if look < max_looks - 1:
                done = open_ & (interim != "").all(1)
                final[done] = interim[done]
            else:
                done = open_
                final[done] = full[done]
            stop_look[done] = look + 1
        for r in range(n):
            stops[stop_look[r]] = stops.get(stop_look[r], 0) + 1
            positives = [i for i in range(len(PIPELINES)) if final[r, i] == "POSITIVE"]
            z_r = NormalDist().inv_cdf(1 - 0.025 / max(len(positives), 1))
            verdicts = []
            for i, p in enumerate(PIPELINES):
                v = final[r, i]
                if v == "POSITIVE":
                    false_positive += p in null
                    rspec = {"Main": spec[p]["Main"], "Shuffled": spec[p]["Shuffled"], "Random": spec["Random"][SOURCE[p]]}
                    v = "SUPPORTED" if replicate(rng, rspec, z_r) else "NOT_ESTABLISHED_NOT_REPLICATED"
                if v == "NOT_ESTABLISHED_UNRESOLVED" and max_looks < 3:
                    v = "NOT_ESTABLISHED_BY_BUDGET"
                per[p][v] = per[p].get(v, 0) + 1
                verdicts.append(v)
            h = headline(verdicts)
            heads[h] = heads.get(h, 0) + 1
    fmt = lambda d: {str(k): round(v / reps, 4) for k, v in sorted(d.items())}  # noqa: E731
    mean_trials = sum(block * look * count for look, count in stops.items()) / reps
    return {"per_pipeline": {p: fmt(per[p]) for p in PIPELINES}, "headline": fmt(heads),
            "stopping_look": fmt(stops), "mean_trials_per_pipeline": round(mean_trials),
            "positive_rate_in_null_pipelines": round(false_positive / reps, 4), "null_pipelines": null}


def scenario(main=None, shuffled=None, random=P0):
    main, shuffled = main or {}, shuffled or {}
    spec = {"Random": {"P": [random] * SEEDS, "R": [random] * SEEDS}}
    for p in PIPELINES:
        m = main.get(p, P0)
        spec[p] = {"Main": m if isinstance(m, list) else [m] * SEEDS,
                   "Shuffled": [shuffled.get(p, P0)] * SEEDS}
    return spec


def operating_characteristics(rng, reps):
    gaussian = [p for p in PIPELINES if "-G-" in p]
    waste = {p: P0 - (0.001 if p in gaussian else 0.0001) for p in PIPELINES}  # duplicates / training matches
    cases = {
        "all_null": scenario(),
        "realistic_null_duplicate_waste": scenario(main=waste, shuffled=waste),
        "P-DISC_+delta": scenario(main={"P-DISC": P0 + DELTA}),
        "P-G-BGV_+delta": scenario(main={"P-G-BGV": P0 + DELTA}),
        "P-DISC_+half_delta": scenario(main={"P-DISC": P0 + DELTA / 2}),
        "R-G-BGV_prior_gain_only": scenario(main={"R-G-BGV": P0 + DELTA}, shuffled={"R-G-BGV": P0 + DELTA}),
        "P-G-CGGE_one_seed_+3delta": scenario(main={"P-G-CGGE": [P0 + 3 * DELTA, P0, P0]}),
        "all_+delta": scenario(main={p: P0 + DELTA for p in PIPELINES}),
    }
    result = {name: stage_c(rng, spec, reps) for name, spec in cases.items()}
    result["budget_stop_after_look2/all_null"] = stage_c(rng, cases["all_null"], reps, max_looks=2)
    result["budget_stop_after_look1/all_null"] = stage_c(rng, cases["all_null"], reps, max_looks=1)
    for name in ("all_null", "P-DISC_+delta"):
        result[f"fallback_block{FALLBACK_BLOCK}/{name}"] = stage_c(rng, cases[name], reps, block=FALLBACK_BLOCK)
    return result


def precision():
    var1 = P0 * (1 - P0)
    hw = lambda var, t, z: z * math.sqrt(var / SEEDS / t)  # noqa: E731
    looks = {}
    for block in (BLOCK, FALLBACK_BLOCK):
        for i, z in enumerate(Z_LOOK):
            t = block * (i + 1)
            looks[f"block{block}_look{i + 1}_T{t}"] = {
                "z": round(z, 4), "half_width": round(hw(2 * var1, t, z), 5),
                "null_exclusion_probability_per_comparison": round(
                    NormalDist().cdf(DELTA / math.sqrt(2 * var1 / SEEDS / t) - z), 4)}
    t = LOOK_TRIALS[-1]
    se_one = math.sqrt(2 * var1 / TRIALS_P)
    return {
        "p0": P0, "delta": DELTA, "delta_relative_per_candidate_lift": round((1 - (1 - P0 - DELTA) ** (1 / K)) / P1, 4),
        "tail_per_comparison": TAIL, "alpha_share_by_look": ALPHA_SHARE, "looks": looks,
        "replication_trials": TRIALS_R,
        "contrasts": {"z": round(Z_CONTRAST, 4), "pairs": [list(c) for c in CONTRASTS],
                      **{f"T{tt}": {"delta_S_half_width": round(hw(4 * var1, tt, Z_CONTRAST), 5),
                                    "delta_R_same_source_half_width": round(hw(2 * var1, tt, Z_CONTRAST), 5),
                                    "delta_R_cross_source_half_width": round(hw(4 * var1, tt, Z_CONTRAST), 5)}
                         for tt in (2 * BLOCK, t)}},
        "positive_control_r4": {"trials": TRIALS_P, "seeds": 1, "z_one_sided": round(Z_P, 4),
                                "mde_90pct_power": round((Z_P + NormalDist().inv_cdf(0.9)) * se_one, 5)},
    }


def effective_rates(tp):
    g = tp["gaussian"]
    d = tp["discrete"]
    rates = {}
    for steps in (25, 50, 100):
        rates[steps] = {
            "P-G-BGV": g["bgv-w32"]["best_rows_per_second"] / (steps + 1) * STREAM_EFFICIENCY,
            "R-G-BGV": g["bgv-w32"]["best_rows_per_second"] / (steps + 1) * STREAM_EFFICIENCY,
            "P-G-CGGE": g["cgge-w32"]["best_rows_per_second"] / (steps + 1) * STREAM_EFFICIENCY,
            "P-DISC": d["D1-S"][0]["candidates_per_second"] * STREAM_EFFICIENCY,
            "R-DISC": d["D1-S"][0]["candidates_per_second"] * STREAM_EFFICIENCY / 2,  # 259-state output; assumption
        }
    update = {"P-G-BGV": g["bgv-w32"]["update"]["update_seconds"], "R-G-BGV": g["bgv-w32"]["update"]["update_seconds"],
              "P-G-CGGE": g["cgge-w32"]["update"]["update_seconds"],
              "P-DISC": RECORDED["v5_a_prof"]["D1-S"]["update_seconds"],
              "R-DISC": 2 * RECORDED["v5_a_prof"]["D1-S"]["update_seconds"]}
    return rates, update


def compute_advantage(rates):
    acceptance = (4096 - 1024 - 256) / 4096
    md5_per_run = UPDATES * BATCH / acceptance
    coupon_all = 4096 * sum(1 / i for i in range(1, 4097))
    rows = {}
    for steps in (25, 100):
        rows[steps] = {}
        for p, cps in rates[steps].items():
            raw = cps / STREAM_EFFICIENCY  # model-only proxy throughput: the comparison most favourable to the model
            rho = MD5_PER_S / raw
            rows[steps][p] = {"proxy_candidates_per_second": round(raw, 1), "rho_md5_over_model": round(rho, 1),
                              "break_even_per_candidate_success": round(rho * P1, 4)}
    return {"md5_calls_per_training_run": round(md5_per_run), "coupon_collector_all_4096": round(coupon_all, 1),
            "training_over_full_lookup": round(md5_per_run / coupon_all, 1), "per_pipeline_by_gaussian_steps": rows,
            "note": "break-even > 1 means a per-query advantage is arithmetically impossible"}


def arithmetic(tp):
    rates, update = effective_rates(tp)
    hours = lambda s: round(s / 3600, 2)  # noqa: E731
    train = {p: UPDATES * (u + TRAIN_OVERHEAD_S) for p, u in update.items()}
    learned_block = 2 * SEEDS * BLOCK * K  # Main + Shuffled
    random_block = 2 * SEEDS * BLOCK * K  # two sources, shared across pipelines of a source
    clp_rows = SEEDS * 65536 * 4 * 8
    gaussian_rows = {"P-G-BGV": tp["gaussian"]["bgv-w32"]["best_rows_per_second"],
                     "R-G-BGV": tp["gaussian"]["bgv-w32"]["best_rows_per_second"],
                     "P-G-CGGE": tp["gaussian"]["cgge-w32"]["best_rows_per_second"]}
    clp_seconds = sum(clp_rows / gaussian_rows.get(p, 1e6) for p in PIPELINES)
    by_steps = {}
    for steps, r in rates.items():
        block = {p: learned_block / cps for p, cps in r.items()}
        block_total = sum(block.values()) + random_block / RANDOM_CPS
        a_q = SEEDS * sum(train.values())
        a_other = 2 * 3600 + 2.5 * 3600  # A-impl + A-prof (planning allowance)
        c_train = 2 * SEEDS * sum(train.values())
        p_train = sum(train.values())
        p_gen = sum(2 * TRIALS_P * K / cps for cps in r.values()) + TRIALS_P * K * 2 / RANDOM_CPS
        l_upd = RECORDED["v5_a_prof"]["D1-T-L"]["update_seconds"]
        s_total = UPDATES_L * (l_upd + TRAIN_OVERHEAD_S) + 2 * 4096 * K / (
            RECORDED["v5_a_prof"]["D1-T-L"]["candidates_per_second"] * STREAM_EFFICIENCY)
        worst_r = max(2 * SEEDS * train[p] + 2 * block[p] for p in PIPELINES)
        by_steps[steps] = {
            "planning_candidates_per_second": {p: round(v, 1) for p, v in r.items()},
            "C_block_hours_by_pipeline": {p: hours(v) for p, v in block.items()},
            "C_block_hours_total": hours(block_total),
            "A_hours": hours(a_q + a_other), "A_qualification_training_hours": hours(a_q),
            "C_training_hours": hours(c_train), "C_clp_hours": hours(clp_seconds),
            "C_to_look2_hours": hours(c_train + 2 * block_total + clp_seconds),
            "C_worst_case_hours": hours(c_train + 3 * block_total + clp_seconds),
            "P_hours": hours(p_train + p_gen), "S_optional_hours": hours(s_total),
            "R_worst_single_pipeline_hours": hours(worst_r),
            "required_path_worst_hours": hours(a_q + a_other + c_train + 3 * block_total + clp_seconds + p_train + p_gen),
            "repair_extra_40k_updates_A_hours_by_pipeline": {p: hours(SEEDS * train[p]) for p in PIPELINES},
            "repair_extra_C_training_hours_by_pipeline": {p: hours(2 * SEEDS * train[p]) for p in PIPELINES},
        }
    return {"stream_efficiency": round(STREAM_EFFICIENCY, 4), "training_overhead_seconds_per_update": TRAIN_OVERHEAD_S,
            "training_hours_per_run": {p: hours(v) for p, v in train.items()},
            "learned_candidates_per_block_per_pipeline": learned_block, "by_gaussian_steps": by_steps,
            "note": "planning figures from random-weight proxies; A-prof re-measures end to end and seals the budget"}


def main():
    rng = np.random.default_rng(2026092907)
    tp = throughput()
    rates, _ = effective_rates(tp)
    oc = operating_characteristics(rng, reps=2000)
    budget = arithmetic(tp)
    blocks_null = oc["all_null"]["mean_trials_per_pipeline"] / BLOCK
    for row in budget["by_gaussian_steps"].values():
        c_null = row["C_training_hours"] + blocks_null * row["C_block_hours_total"] + row["C_clp_hours"]
        row["C_expected_null_hours"] = round(c_null, 2)
        row["required_path_expected_null_hours"] = round(row["A_hours"] + c_null + row["P_hours"], 2)
    budget["expected_blocks_under_null"] = round(blocks_null, 3)
    result = {
        "protocol": "dhi-v6-20260929 (plan only)",
        "z_by_look": [round(z, 4) for z in Z_LOOK],
        "precision": precision(),
        "v5_partial_stage_c": v5_partial(),
        "throughput_proxy": tp,
        "compute_advantage": compute_advantage(rates),
        "arithmetic": budget,
        "stage_c_operating_characteristics": oc,
        "measurement_sources": SOURCES,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "design_calculation.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "throughput_proxy"}, indent=1))


if __name__ == "__main__":
    main()
