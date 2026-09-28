"""Design checks for RESEARCH_PLAN_V5.md.

Trains no model and opens no MD5 test pool. It reads the small measurement files written by the V5 D0
diagnostic (local_experiment_archive/analyses/v5-d0-20260928). That archive is gitignored, so in a
clean checkout the values recorded on 2026-09-28 (RECORDED below, also quoted in the plan §1.2-§1.3) are
used instead and the output says which source was used. It computes:
1. D0 verdicts against the v4 V1 thresholds,
2. Monte Carlo operating characteristics of the Stage C rule (T=65,536, delta=0.25pp, one extension look),
3. operating characteristics of the positive-branch replication on a fresh digest window,
4. Stage B ladder sensitivity,
5. compute-advantage arithmetic (coupon collector, MD5 calls spent by training, break-even lift),
6. run / candidate / time arithmetic from the measured throughputs.
"""
import json
import math
from pathlib import Path
import re
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D0 = ROOT / "local_experiment_archive/analyses/v5-d0-20260928"
OUT = ROOT / "local_experiment_archive/analyses/v5-design-20260928"
Q = 12
P1 = 2 ** -Q
K = 100
P0 = 1 - (1 - P1) ** K
DELTA = 0.0025
TRIALS_C = 65536
TRIALS_B = 16384
SEEDS_FIRST, SEEDS_EXT = 3, 3
Z_LOOK = NormalDist().inv_cdf(1 - 0.05 / 8)  # 2 looks x 2 controls x 2 sides
Z_REPL = NormalDist().inv_cdf(1 - 0.025)  # one-sided per control, intersection-union
LADDER = (4, 5, 6, 7, 8, 10, 12, 16, 32)
Z_LADDER = NormalDist().inv_cdf(1 - 0.01 / len(LADDER))
GROUPS = {"test": 1024, "validation": 256, "train": 4096 - 1024 - 256}
UPDATES, BATCH = 40000, 256
UPDATES_L = 160000
# Measurements recorded on 2026-09-28 (MLX/Metal, float32); used when the local archive is absent.
RECORDED = {
    "d0_summary": {"thresholds": {"joint_min": 461, "of": 512, "wrong_max": 25}, "results": {
        f"{seed}/{tag}": {"normal_joint": n, "flipped_joint": f, "normal_valid": 512, "flipped_valid": 512,
                          "wrong_original": 0, "md5_calls": 0}
        for seed, tag, n, f in [(0, "epoch11", 176, 167), (0, "epoch100", 469, 460), (1, "epoch11", 160, 151),
                                (1, "epoch100", 469, 469), (2, "epoch11", 125, 134), (2, "epoch100", 462, 466)]}},
    "throughput": {"md5_h12_per_second_python_single_core": 2104757,
                   "random_method_candidates_per_second_python_single_core": 86768,
                   "d1s_candidates_per_second_batch_64": 414, "d1s_candidates_per_second_batch_256": 476,
                   "d1s_candidates_per_second_batch_1024": 406, "d1s_candidates_per_second_batch_4096": 398},
    "vectorized_sampler_check": {"bitwise_parity_256": True, "batch_composition_invariance": True,
                                 "vectorized_d1s_candidates_per_second_batch_1024": 17565},
    "transformer_timing": ("d=192 layers=4 params=1823714 update_s_batch256=0.0199 forward_s_batch4096=0.0876 "
                           "est_candidates_per_s_33nfe=1418\n"
                           "d=256 layers=8 params=6372962 update_s_batch256=0.0592 forward_s_batch4096=0.2726 "
                           "est_candidates_per_s_33nfe=455"),
}
SOURCES = {}


def measured(name, suffix=".json"):
    path = D0 / (name + suffix)
    SOURCES[name] = "archive" if path.exists() else "recorded_2026-09-28"
    if not path.exists():
        return RECORDED[name]
    return json.loads(path.read_text()) if suffix == ".json" else path.read_text()


def d0_verdicts():
    summary = measured("d0_summary")
    t = summary["thresholds"]
    rows = {}
    for key, r in summary["results"].items():
        rows[key] = {**r, "passes_v4_v1": r["normal_joint"] >= t["joint_min"] and r["flipped_joint"] >= t["joint_min"]
                     and r["normal_valid"] == r["flipped_valid"] == t["of"] and r["wrong_original"] <= t["wrong_max"]}
    return {"thresholds": t, "rows": rows,
            "epoch100_min_joint": min(min(r["normal_joint"], r["flipped_joint"]) for k, r in rows.items() if k.endswith("epoch100")),
            "epoch11_max_joint": max(max(r["normal_joint"], r["flipped_joint"]) for k, r in rows.items() if k.endswith("epoch11"))}


def look(main, controls):
    out = {}
    for name, control in controls.items():
        d = (main - control).mean(0)
        est, se = d.mean(), d.std(ddof=1) / math.sqrt(d.size)
        out[name] = (est - Z_LOOK * se, est + Z_LOOK * se)
    signal = all(lo > 0 for lo, _ in out.values())
    bounded = {k: hi < DELTA for k, (_, hi) in out.items()}
    if signal:
        return "POSITIVE", out
    if bounded["shuffled"]:
        return ("REJECTED_BOUNDED" if bounded["random"] else "REJECTED_NO_CONDITION_GAIN"), out
    if bounded["random"]:
        return "REJECTED_NO_RANDOM_ADVANTAGE", out
    return "INCONCLUSIVE", out


def replication(rng, p_main, p_null, trials=TRIALS_C, seeds=3):
    draw = lambda p: (rng.random((seeds, trials)) < p).astype(np.float32)  # noqa: E731
    m = draw(p_main)
    for control in (draw(p_null), draw(p_null)):
        d = (m - control).mean(0)
        if d.mean() - Z_REPL * d.std(ddof=1) / math.sqrt(trials) <= 0:
            return False
    return True


def simulate(rng, p_main, p_shuffled, p_random, reps, with_replication=False):
    """p_* are per-seed Success@100 lists of length 6 (first look seeds 0-2, extension seeds 3-5)."""
    tally = {}
    for _ in range(reps):
        draw = lambda p: (rng.random((len(p), TRIALS_C)) < np.asarray(p)[:, None]).astype(np.float32)  # noqa: E731
        m, sh, ra = draw(p_main), draw(p_shuffled), draw(p_random)
        verdict, _ = look(m[:SEEDS_FIRST], {"random": ra[:SEEDS_FIRST], "shuffled": sh[:SEEDS_FIRST]})
        if verdict == "INCONCLUSIVE":
            second, _ = look(m, {"random": ra, "shuffled": sh})
            verdict = "ext->" + ("NOT_ESTABLISHED_UNRESOLVED" if second == "INCONCLUSIVE" else second)
        if with_replication and verdict.endswith("POSITIVE"):
            verdict += "+REPLICATED" if replication(rng, p_main[0], P0) else "+NOT_REPLICATED"
        tally[verdict] = tally.get(verdict, 0) + 1
    return {k: round(v / reps, 4) for k, v in sorted(tally.items())}


def operating_characteristics(rng, reps=1000):
    null = [P0] * 6
    plus = lambda d: [P0 + d] * 6  # noqa: E731
    return {
        "all_null": simulate(rng, null, null, null, reps),
        "main_+0.125pp": simulate(rng, plus(.00125), null, null, reps),
        "main_+0.25pp": simulate(rng, plus(.0025), null, null, reps, with_replication=True),
        "main_+0.5pp": simulate(rng, plus(.005), null, null, reps, with_replication=True),
        "main_+0.5pp_one_seed_null": simulate(rng, [P0 + .005, P0 + .005, P0] * 2, null, null, reps),
        "shuffled_also_+0.25pp": simulate(rng, plus(.0025), plus(.0025), null, reps),
        "memorization_main_-0.3pp": simulate(rng, plus(-.003), null, null, reps),
    }


def replication_null(rng, reps=2000):
    """False replication rate when the true effect is zero (independent of the first-window result)."""
    return {"reps": reps, "pass_rate": sum(replication(rng, P0, P0) for _ in range(reps)) / reps,
            "intersection_union_upper": 0.025}


def half_width(trials, seeds, z):
    return z * math.sqrt(2 * P0 * (1 - P0) / seeds) / math.sqrt(trials)


def ladder_sensitivity():
    # One seed, paired Main vs control; 90% power at one-sided z = Z_LADDER.
    sd = math.sqrt(2 * P0 * (1 - P0))
    mde = (Z_LADDER + NormalDist().inv_cdf(.9)) * sd / math.sqrt(TRIALS_B)
    return {"cells": len(LADDER), "z_per_cell": round(Z_LADDER, 4), "trials": TRIALS_B,
            "detectable_success_at_100_difference_90pct_power": round(mde, 5)}


def throughput():
    tp = measured("throughput")
    vec = measured("vectorized_sampler_check")
    rows = {}
    for line in measured("transformer_timing", ".txt").splitlines():
        m = re.search(r"d=(\d+) layers=(\d+) params=(\d+) update_s_batch256=([\d.]+) forward_s_batch4096=([\d.]+) "
                      r"est_candidates_per_s_33nfe=(\d+)", line)
        if m:
            d, layers, params, upd, fwd, cps = m.groups()
            rows[f"d{d}_l{layers}"] = {"params": int(params), "update_seconds": float(upd), "candidates_per_second": int(cps),
                                       "forward_rows_per_second": round(4096 / float(fwd))}
    return {"md5_h12_per_second": tp["md5_h12_per_second_python_single_core"],
            "random_method_per_second_python": tp["random_method_candidates_per_second_python_single_core"],
            "d1s_current_sampler_per_second": max(v for k, v in tp.items() if k.startswith("d1s_candidates")),
            "d1s_vectorized_per_second": vec["vectorized_d1s_candidates_per_second_batch_1024"],
            "vectorized_bitwise_parity": vec["bitwise_parity_256"],
            "vectorized_batch_invariance": vec["batch_composition_invariance"],
            "transformer_proxy": rows}


def compute_advantage(tp):
    acceptance = GROUPS["train"] / 4096
    md5_per_run = UPDATES * BATCH / acceptance
    coupon_all = 4096 * sum(1 / i for i in range(1, 4097))
    coupon_test = 4096 * sum(1 / i for i in range(1, GROUPS["test"] + 1))
    t = tp["transformer_proxy"]["d192_l4"]["candidates_per_second"]
    s = tp["d1s_vectorized_per_second"]
    return {
        "train_acceptance": acceptance,
        "md5_calls_per_training_run": round(md5_per_run),
        "coupon_collector_all_4096_targets": round(coupon_all, 1),
        "coupon_collector_1024_test_targets": round(coupon_test, 1),
        "training_md5_over_full_lookup": round(md5_per_run / coupon_all, 1),
        "cost_ratio_md5_over_d1t": round(tp["md5_h12_per_second"] / t, 1),
        "cost_ratio_md5_over_d1s": round(tp["md5_h12_per_second"] / s, 1),
        "break_even_per_candidate_success_d1t": round(tp["md5_h12_per_second"] / t * P1, 4),
        "break_even_per_candidate_success_d1s": round(tp["md5_h12_per_second"] / s * P1, 4),
        "max_success_at_100_lift": round(1 / P0, 2),
    }


def arithmetic(tp):
    t_cps = tp["transformer_proxy"]["d192_l4"]["candidates_per_second"]
    l_cps = tp["transformer_proxy"]["d256_l8"]["candidates_per_second"]
    t_upd = tp["transformer_proxy"]["d192_l4"]["update_seconds"]
    l_upd = tp["transformer_proxy"]["d256_l8"]["update_seconds"]
    c_learned = SEEDS_FIRST * (2 * TRIALS_C + TRIALS_C // 4) * K  # Main + Shuffled + MC(1/4)
    b_learned = len(LADDER) * 2 * TRIALS_B * K
    b_repl = 4 * 2 * TRIALS_B * K
    s_learned = 2 * 4096 * K  # GEN at the edge rung only (Main + MC)
    hours = lambda seconds: round(seconds / 3600, 2)  # noqa: E731
    stages = {
        "A": {"runs": 6 + 6, "note": "6 qualification + <=6 short lr-dev runs (10k updates)"},
        "C_first": {"runs": 6, "learned_candidates": c_learned, "random_candidates": SEEDS_FIRST * TRIALS_C * K,
                    "train_hours_d1t": hours(6 * UPDATES * t_upd), "generation_hours_d1t": hours(c_learned / t_cps)},
        "B": {"runs": len(LADDER) + 4, "learned_candidates": b_learned + b_repl,
              "train_hours_d1t": hours((len(LADDER) + 4) * UPDATES * t_upd),
              "generation_hours_d1t": hours((b_learned + b_repl) / t_cps)},
        "S": {"runs": 2, "updates": UPDATES_L, "train_hours_large": hours(2 * UPDATES_L * l_upd),
              "generation_hours_large": hours(s_learned / l_cps)},
    }
    stages["C_extension"] = dict(stages["C_first"])
    stages["R_positive_only"] = dict(stages["C_first"])
    return {"updates_per_run": UPDATES, "batch": BATCH, "fresh_pairs_per_run": UPDATES * BATCH,
            "fresh_pairs_large_run": UPDATES_L * BATCH, "stages": stages,
            "note": "proxy timings (generic MLX Transformer, float32); A-prof re-measures the real models"}


def main():
    rng = np.random.default_rng(2026092806)
    tp = throughput()
    result = {
        "p0_success_at_100": P0, "delta": DELTA, "trials_stage_c": TRIALS_C, "z_per_look": round(Z_LOOK, 4),
        "expected_null_half_width_first_look": round(half_width(TRIALS_C, SEEDS_FIRST, Z_LOOK), 5),
        "expected_null_half_width_extension": round(half_width(TRIALS_C, SEEDS_FIRST + SEEDS_EXT, Z_LOOK), 5),
        "d0": d0_verdicts(),
        "throughput": tp,
        "stage_c_operating_characteristics": operating_characteristics(rng),
        "replication_null": replication_null(rng),
        "ladder": ladder_sensitivity(),
        "compute_advantage": compute_advantage(tp),
        "arithmetic": arithmetic(tp),
        "measurement_sources": SOURCES,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "design_calculation.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
