"""Prospective v4 design calculation; no model, archive, or MD5 holdout access.

Run: .venv/bin/python scripts/validate_research_plan_v4.py
Only NumPy and the standard library are needed. This is not a production gate.
"""
import json
import math
from functools import lru_cache

import numpy as np

TRIALS = 16_384
REPETITIONS = 20_000
DELTA = .01
TAIL = .05 / 24  # Six differences, two discordance probabilities, two tails.
P0 = -math.expm1(100 * math.log1p(-1 / 4096))


from diffusion_hash_inv.study_v4_statistics import cp, intervals, decisions, self_check, simulate


def main():
    self_check()
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
    results = {name: simulate(strata, np.random.default_rng(2026092804 + i))
               for i, (name, strata) in enumerate(scenarios.items())}
    assert results["reference_twofold"]["GO_probability_lower95"] >= .8
    assert results["all_null"]["NO_GO_SMALL_probability_lower95"] >= .8
    learned, random = 6 * TRIALS * 100, 3 * TRIALS * 100
    assert learned + random == 14_745_600
    assert learned * 33 == 324_403_200
    print(json.dumps({
        "status": "PLAN_CALCULATION_PASS_NOT_PRODUCTION_CALIBRATION",
        "trials": TRIALS, "simulations_per_scenario": REPETITIONS,
        "minimum_useful_absolute_gain": DELTA, "one_tail_error": TAIL,
        "ideal_random_success_at_100": P0,
        "expected_successes_per_stream_under_ideal_null": TRIALS * P0,
        "training_updates_per_main_run": math.ceil(10_000 / 64) * 100,
        "main_learned_rows": learned, "main_random_rows": random,
        "main_total_rows": learned + random, "main_sampling_nfe": learned * 33,
        "scenarios": results,
    }, indent=2))


if __name__ == "__main__":
    main()
