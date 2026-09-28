"""Design checks for RESEARCH_PLAN_V4_CLAUDE.md.

Reads no archive, trains no model and opens no MD5 test pool. It computes:
1. a step-reduced MD5 reference (r=64 must equal hashlib) and model-free difficulty profiles per rung,
2. the exact-null behaviour of the paired CLP statistic,
3. Monte Carlo operating characteristics of the Stage C decision rule,
4. fixed run/candidate/NFE arithmetic.
"""
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "local_experiment_archive/analyses/v4-claude-design-20260928"
LADDER = (4, 5, 6, 7, 8, 10, 12, 16, 32)  # Stage B; r=64 is Stage C only
RUNGS = LADDER + (64,)
P0 = 1 - (1 - 2 ** -12) ** 100
DELTA = 0.005
Z_LOOK = 2.4977054744  # one-sided tail .05 / (2 looks x 2 controls x 2 directions) = .00625
S = [7, 12, 17, 22] * 4 + [5, 9, 14, 20] * 4 + [4, 11, 16, 23] * 4 + [6, 10, 15, 21] * 4
K = [int(abs(math.sin(i + 1)) * 2 ** 32) & 0xFFFFFFFF for i in range(64)]
IV = (0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476)
M32 = 0xFFFFFFFF


def md5_steps(message, steps):
    """First `steps` of the MD5 compression on one padded block, feed-forward, standard serialization."""
    if not 0 <= steps <= 64 or len(message) > 55:
        raise ValueError("single-block messages and 0..64 steps only")
    block = message + b"\x80" + b"\x00" * (55 - len(message)) + struct.pack("<Q", 8 * len(message))
    words = struct.unpack("<16I", block)
    a, b, c, d = IV
    for i in range(steps):
        if i < 16:
            f, g = (b & c) | (~b & d), i
        elif i < 32:
            f, g = (d & b) | (~d & c), (5 * i + 1) % 16
        elif i < 48:
            f, g = b ^ c ^ d, (3 * i + 5) % 16
        else:
            f, g = c ^ (b | (~d & M32)), (7 * i) % 16
        t = (a + (f & M32) + K[i] + words[g]) & M32
        a, d, c, b = d, c, b, (b + ((t << S[i]) | (t >> (32 - S[i]))) ) & M32
    return struct.pack("<4I", *((x + y) & M32 for x, y in zip(IV, (a, b, c, d))))


def h12(message, steps=64):
    return int.from_bytes(md5_steps(message, steps), "big") >> 116


def check_reference(rng):
    vectors = [b"", b"a", b"abc", b"message digest", b"abcdefghijklmnopqrstuvwxyz"]
    randoms = [bytes(rng.integers(33, 127, size=int(rng.integers(4, 32)), dtype=np.uint8)) for _ in range(2000)]
    mismatches = sum(md5_steps(m, 64) != hashlib.md5(m).digest() for m in vectors + randoms)
    return {"rfc_vectors": len(vectors), "random_printable": len(randoms), "mismatches": mismatches}


def prior(rng, n):
    lengths = rng.integers(4, 32, size=n)
    return [bytes(rng.integers(33, 127, size=int(n_), dtype=np.uint8)) for n_ in lengths]


def mutual_information(a, b, na, nb):
    joint = np.zeros((na, nb))
    np.add.at(joint, (a, b), 1)
    joint /= joint.sum()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (pa @ pb)[nz])).sum())


def difficulty_profile(rng, n_avalanche=20000, n_mi=200000):
    """Model-free structure left in y=H12_r(x) under the Printable prior."""
    base = prior(rng, n_avalanche)
    mi_messages = prior(rng, n_mi)
    profile = {}
    for r in RUNGS:
        flips, affected = [], np.zeros(31, dtype=np.int64)
        for message in base:
            pos = int(rng.integers(len(message)))
            new = int(rng.integers(33, 126))
            new += new >= message[pos]  # a different printable byte
            changed = bytearray(message); changed[pos] = new
            diff = h12(message, r) ^ h12(bytes(changed), r)
            flips.append(bin(diff).count("1") / 12)
            affected[pos] += diff != 0
        top4 = np.array([h12(m, r) >> 8 for m in mi_messages])
        # MI(top 4 bits of y ; byte k) over messages with length > k; plug-in bias ~ (15*93)/(2 n ln2) bits.
        mi = []
        for k in range(16):
            keep = np.array([len(m) > k for m in mi_messages])
            byte = np.array([m[k] - 33 for m in mi_messages if len(m) > k])
            mi.append(mutual_information(top4[keep], byte, 16, 94))
        profile[str(r)] = {
            "mean_output_bit_flip_rate": round(float(np.mean(flips)), 4),
            "p_y_unchanged_after_one_byte_change": round(float(np.mean(np.array(flips) == 0)), 4),
            "max_mi_top4_vs_single_byte_bits": round(max(mi), 4),
            "argmax_byte": int(np.argmax(mi)),
            "plugin_bias_bits_upper": round(15 * 93 / (2 * n_mi * 12 / 28 * math.log(2)), 5),
        }
    return profile


def clp_null_check(rng, pairs=20000, reps=2000):
    """Paired CLP statistic under x independent of y: symmetric about 0 even with marginal y effects."""
    rejections = 0
    for _ in range(reps):
        x = rng.normal(size=(pairs, 2))
        y = rng.integers(0, 4096, size=(pairs, 2))
        g = np.sin(y / 97.0) * 3  # arbitrary condition-only score component
        score = lambda xi, yi: g[:, yi] + x[:, xi] * np.cos(y[:, yi] / 13.0)  # noqa: E731
        d = score(0, 0) + score(1, 1) - score(0, 1) - score(1, 0)
        z = d.mean() / (d.std(ddof=1) / math.sqrt(pairs))
        rejections += z > 2.3263  # one-sided .01
    return {"reps": reps, "one_sided_alpha": .01, "rejection_rate": rejections / reps}


def stage_c_look(main, controls, trials):
    """main: [seeds, T]; controls: {name: [seeds, T]} of 0/1 outcomes. Returns per-control bounds."""
    out = {}
    for name, control in controls.items():
        per_trial = (main - control).mean(0)
        est = per_trial.mean()
        se = per_trial.std(ddof=1) / math.sqrt(trials)
        out[name] = {"est": est, "lower": est - Z_LOOK * se, "upper": est + Z_LOOK * se,
                     "per_seed": (main - control).mean(1).tolist()}
    return classify(out), out


def classify(out):
    """Stage C rule of §7.3: CONTINUE needs both lower bounds > 0; any bounded control ends the study."""
    signal = {k: v["lower"] > 0 for k, v in out.items()}
    bounded = {k: v["upper"] < DELTA for k, v in out.items()}
    if signal["random"] and signal["shuffled"]:
        return "CONTINUE"
    if bounded["shuffled"]:
        return "CONCLUDE_BOUNDED" if bounded["random"] else "CONCLUDE_NO_CONDITION_GAIN"
    if bounded["random"]:
        return "CONCLUDE_NO_RANDOM_ADVANTAGE"
    return "INCONCLUSIVE"


def simulate_stage_c(rng, p_main, p_shuffled, p_random, trials=16384, reps=1000):
    """p_* are per-seed Success@100 vectors (length 6: seeds 0-2 first look, 3-5 extension)."""
    tally = {}
    for _ in range(reps):
        draw = lambda p: (rng.random((len(p), trials)) < np.asarray(p)[:, None]).astype(float)  # noqa: E731
        m, sh, ra = draw(p_main), draw(p_shuffled), draw(p_random)
        first, _ = stage_c_look(m[:3], {"random": ra[:3], "shuffled": sh[:3]}, trials)
        if first == "INCONCLUSIVE":
            second, _ = stage_c_look(m, {"random": ra, "shuffled": sh}, trials)
            key = "extension->" + ("CONCLUDE_UNRESOLVED" if second == "INCONCLUSIVE" else second)
        else:
            key = first
        tally[key] = tally.get(key, 0) + 1
    return {k: round(v / reps, 4) for k, v in sorted(tally.items())}


def scenarios(rng):
    null = [P0] * 6
    plus = lambda d: [P0 + d] * 6  # noqa: E731
    return {
        "all_null": simulate_stage_c(rng, null, null, null),
        "main_+0.25pp": simulate_stage_c(rng, plus(.0025), null, null),
        "main_+0.5pp": simulate_stage_c(rng, plus(.005), null, null),
        "main_+1pp": simulate_stage_c(rng, plus(.01), null, null),
        "main_2p0": simulate_stage_c(rng, [2 * P0] * 6, null, null),
        "main_+1pp_one_seed_null": simulate_stage_c(rng, [P0 + .01, P0 + .01, P0] * 2, null, null),
        "shuffled_also_+0.5pp": simulate_stage_c(rng, plus(.005), plus(.005), null),
        "memorization_main_-0.3pp": simulate_stage_c(rng, plus(-.003), null, null),
    }


def arithmetic():
    ladder_rungs = len(LADDER)
    s0 = {"archs": 2, "seeds": 3, "runs": 6, "candidates": 6 * 1024}
    ladder = {"rungs": list(LADDER), "cells": ladder_rungs * 2, "runs_seed0": ladder_rungs * 2,
              "replication_runs_max": 2 * 2 * 2,
              "trials": 4096, "k": 100}
    ladder["learned_candidates_seed0"] = ladder["cells"] * 2 * ladder["trials"] * ladder["k"]  # true + mismatched
    ladder["learned_candidates_replication_max"] = ladder["replication_runs_max"] * 2 * ladder["trials"] * ladder["k"]
    ladder["random_candidates"] = ladder_rungs * ladder["trials"] * ladder["k"]
    stage_c = {"runs_first_look": 6, "runs_extension": 6, "trials": 16384, "mismatched_trials": 4096, "k": 100}
    stage_c["learned_candidates_first_look"] = 3 * (2 * stage_c["trials"] + stage_c["mismatched_trials"]) * stage_c["k"]
    stage_c["random_candidates_first_look"] = 3 * stage_c["trials"] * stage_c["k"]
    updates = 40000
    total_runs_first_pass = s0["runs"] + ladder["runs_seed0"] + stage_c["runs_first_look"]
    total_runs_max = total_runs_first_pass + ladder["replication_runs_max"] + stage_c["runs_extension"]
    return {"updates_per_run": updates, "batch": 256, "fresh_pairs_per_run": updates * 256, "s0": s0,
            "ladder": ladder, "stage_c": stage_c, "learned_runs_first_pass": total_runs_first_pass,
            "learned_runs_max": total_runs_max, "updates_first_pass": total_runs_first_pass * updates,
            "nfe_per_candidate": 33,
            "learned_nfe_first_pass": 33 * (s0["candidates"] + ladder["learned_candidates_seed0"]
                                            + stage_c["learned_candidates_first_look"])}


def main():
    rng = np.random.default_rng(2026092805)
    result = {"p0_success_at_100": P0, "delta": DELTA, "z_per_look": Z_LOOK,
              "reference": check_reference(rng)}
    if result["reference"]["mismatches"]:
        raise SystemExit("step-reduced MD5 reference mismatch at r=64")
    result["difficulty_profile"] = difficulty_profile(rng)
    result["clp_null"] = clp_null_check(rng)
    result["stage_c_operating_characteristics"] = scenarios(rng)
    result["arithmetic"] = arithmetic()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "design_calculation.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
