"""Read-only plan/implementation diagnostics; no training or primary evaluation.

Writes engineering evidence to the ignored local archive. Does not certify
scientific gates, inspect archived success rates, or emit source messages.
"""
import hashlib
import json
import math
import platform
import random
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from diffusion_hash_inv.dataset import build_digest_records
from diffusion_hash_inv.discrete import MaskedDiffusion, SequenceDenoiser
from diffusion_hash_inv.encoding.bgv import BGVEncoder, BGVDecoder
from diffusion_hash_inv.encoding.cgge import CGGEEncoder, CGGEDecoder, glyph_table_checksum
from diffusion_hash_inv.encoding.tokens import TokenCodec
from diffusion_hash_inv.evaluation import exact_mcnemar, holm_adjust, binomial_ci95
from diffusion_hash_inv.models import GaussianDiffusion, ImageUNet, parameter_count
from diffusion_hash_inv.runner import ExperimentConfig, _conditions, _build_model, _codec

CONFIG_PATH = ROOT / "examples/poc-v2-protocol.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
OUTPUT = ROOT / "local_experiment_archive/runs/2026-09-23-plan-v2-validation"
OUTPUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def namespace_seed(*parts):
    assert len(parts) <= 7
    fields = (*parts, *("" for _ in range(7 - len(parts))))
    value = ":".join(map(str, (CONFIG["protocol_id"], CONFIG["dataset"]["engineering_seed"], *fields)))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def exposure_inventory():
    root = ROOT / "local_experiment_archive/retired_2026-09-21/artifacts/poc_md5_truncated"
    excluded = {source: set() for source in CONFIG["sources"]}
    records = []
    for path in sorted(root.rglob("metrics.json")):
        relative = path.relative_to(root).as_posix()
        if not (relative.startswith("evaluation/") or
                (relative.startswith("search/") and "/validation/" in relative)):
            continue
        match = re.search(r"q(12|16|20|24|32|64|128)(?:-|/)", relative)
        source = "printable" if "/P-G-" in relative else "random_bytes" if "/R-G-" in relative else None
        if match is None or source is None:
            continue
        data = json.loads(path.read_text())
        # Metadata only: neither outcome rates nor candidate messages are emitted.
        order = data.get("target_order", [])
        assert all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{3,32}", value) for value in order)
        excluded[source].update(int(value[:3], 16) for value in order)
        records.append({"file": str(path.relative_to(ROOT)), "sha256": sha(path),
                        "source": source, "q": int(match[1]), "target_count": len(order)})
    assert records, "known prior evaluation metadata missing"
    union = set().union(*excluded.values())
    return excluded, {"scope": "known retired PoC evaluation and validation metadata; broader audit still required",
                      "records": records, "shared_pool_remaining": 4096 - len(union),
                      "per_source": {s: {"excluded": len(v), "available": 4096 - len(v),
                                         "excluded_prefixes": [f"{x:03x}" for x in sorted(v)]}
                                     for s, v in excluded.items()}}


def construction_check(source, excluded):
    rng = random.Random(namespace_seed("ownership-test", source))
    available = sorted(set(range(4096)) - excluded)
    assert len(available) >= 2048
    rng.shuffle(available)
    test = set(available[:2048])
    remaining = sorted(set(range(4096)) - test)
    rng = random.Random(namespace_seed("ownership-rest", source))
    rng.shuffle(remaining)
    train, validation = set(remaining[:1536]), set(remaining[1536:])
    assert not (test & excluded or train & validation or train & test or validation & test)
    rng = random.Random(namespace_seed("source", source))
    training, val_reps, test_reps = set(), {}, {}
    duplicates = surplus = 0
    for draw in range(1, CONFIG["dataset"]["draw_cap_per_source"] + 1):
        length = rng.randint(4, 31)
        message = (bytes(rng.randrange(33, 127) for _ in range(length))
                   if source == "printable" else bytes(rng.randrange(256) for _ in range(length)))
        prefix = int.from_bytes(hashlib.md5(message).digest()[:2], "big") >> 4
        if prefix in train:
            if message in training:
                duplicates += 1
            elif len(training) < 10000:
                training.add(message)
            else:
                surplus += 1
        else:
            representatives = val_reps if prefix in validation else test_reps
            if prefix in representatives:
                surplus += 1
            else:
                representatives[prefix] = message
        if (len(training), len(val_reps), len(test_reps)) == (10000, 512, 2048):
            break
    assert (len(training), len(val_reps), len(test_reps)) == (10000, 512, 2048)
    assert not (training & set(val_reps.values()) or training & set(test_reps.values())
                or set(val_reps.values()) & set(test_reps.values()))
    return {"kind": "engineering_only_not_primary_dataset", "draws": draw,
            "train_messages": len(training), "validation_targets": len(val_reps),
            "test_targets": len(test_reps), "duplicate_train_draws": duplicates,
            "surplus_draws": surplus, "primary_study_seed_used": False,
            "source_messages_written": False}


def codecs_and_controls():
    results = {}
    for name in CONFIG["pipelines"]:
        source = "printable" if name.startswith("P-") else "random_bytes"
        if name.endswith("DISC"):
            encoder = decoder = TokenCodec(source, 31)
        elif name.endswith("CGGE"):
            encoder, decoder = CGGEEncoder(), CGGEDecoder()
        else:
            encoder, decoder = BGVEncoder(), BGVDecoder()
        alphabet = list(range(33, 127)) if source == "printable" else list(range(256))
        corpus = [bytes([x]) * 4 for x in alphabet]
        for length in range(4, 32):
            corpus.extend((bytes([alphabet[0]]) * length, bytes([alphabet[-1]]) * length,
                           bytes(alphabet[j % len(alphabet)] for j in range(length))))
        for message in corpus:
            decoded = decoder.decode(encoder.encode(message))
            assert decoded.valid and decoded.message == message, name
        results[name] = len(corpus)
    pairs = list(range(2048))  # y and y^4095 form one pair.
    random.Random(namespace_seed("positive_partition")).shuffle(pairs)
    groups = [{v for a in group for v in (a, a ^ 4095)}
              for group in (pairs[:1536], pairs[1536:1792], pairs[1792:])]
    assert list(map(len, groups)) == [3072, 512, 512]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert all((y ^ 4095) in group for group in groups for y in group)
    return {"clean_codec_roundtrips": results, "glyph_checksum": glyph_table_checksum(),
            "synthetic_partition_sizes": list(map(len, groups)), "complement_closed": True,
            "synthetic_min_correct_of_512": next(n for n in range(513) if binomial_ci95(n, 512)[0] >= .9),
            "synthetic_max_wrong_of_512": max(n for n in range(513) if binomial_ci95(n, 512)[1] <= .05),
            "learned_positive_control_status": "NOT_RUN"}


def statistics_check():
    for wins in range(15):
        for losses in range(15):
            n = wins + losses
            reference = sum(math.comb(n, r) for r in range(wins, n + 1)) / 2**n
            assert math.isclose(exact_mcnemar(wins, losses), reference, abs_tol=1e-15)
    assert holm_adjust(dict(a=.001, b=.009, c=.02, d=.04, e=1)) == dict(a=.005, b=.036, c=.06, d=.08, e=1)
    assert max([.001] * 5 + [1]) == 1
    # Exact binomial critical values; avoid asymptotic approximations in power.
    cutoffs = np.empty(2049, dtype=int)
    for n in range(2049):
        coefficient = total = 1
        cutoff = n + 1
        for failures in range(n + 1):
            if failures:
                coefficient = coefficient * (n - failures + 1) // failures
                total += coefficient
            if total / 2**n < .01:
                cutoff = n - failures
            else:
                break
        cutoffs[n] = cutoff
    p = -math.expm1(100 * math.log1p(-2**-12))
    scenarios = [
        ("reference_twofold", [2, 2, 2], 1, [(2048, 1)]),
        ("one_weaker_seed_1_5fold", [2, 2, 1.5], 1, [(2048, 1)]),
        ("shuffled_1_5_times_random", [2, 2, 2], 1.5, [(2048, 1)]),
        ("shared_target_difficulty_0_2_1_8", [2, 2, 2], 1, [(1024, .2), (1024, 1.8)]),
        ("reference_threefold", [3, 3, 3], 1, [(2048, 1)]),
    ]
    cells = [(m, r, s) for m in (0, 1) for r in (0, 1) for s in (0, 1)]
    rows = []
    for name, factors, shuffled_factor, strata in scenarios:
        rng = np.random.default_rng(namespace_seed("power", name))
        passed = np.ones(20000, dtype=bool)
        for factor in factors:
            counts = np.zeros((len(passed), 8), dtype=int)
            for n, difficulty in strata:
                probabilities = [p * difficulty * x for x in (factor, 1, shuffled_factor)]
                cell_p = [math.prod(prob if bit else 1 - prob for prob, bit in zip(probabilities, cell)) for cell in cells]
                counts += rng.multinomial(n, cell_p, size=len(passed))
            for control in (1, 2):
                wins = counts[:, [j for j, c in enumerate(cells) if c[0] == 1 and c[control] == 0]].sum(1)
                losses = counts[:, [j for j, c in enumerate(cells) if c[0] == 0 and c[control] == 1]].sum(1)
                passed &= wins >= cutoffs[wins + losses]
        estimate = float(passed.mean())
        rows.append({"scenario": name, "pass_probability": estimate,
                     "monte_carlo_se": math.sqrt(estimate * (1 - estimate) / len(passed))})
    return {"exact_test_fixture_pairs": 225, "holm_fixture": "PASS", "missing_component_fixture": "PASS",
            "reference_baseline": p, "simulation_repetitions": 20000,
            "scope": "one pipeline, all six component p<.01; sufficient Holm cutoff, not full Holm",
            "assumptions": "independent target sampling randomness and seed outcomes conditional on specified difficulty",
            "sensitivity": rows, "inferential_assumptions_certified": False}


def implementation_and_timing():
    records = build_digest_records((b"ABCD", b"EFGH"), source="printable", algorithm="md5", q=12)
    normal = ExperimentConfig("bgv", "printable", "md5", 12, condition_format="canonical_bits", condition_dim=259)
    shuffled = ExperimentConfig("bgv", "printable", "md5", 12, condition_format="canonical_bits", condition_dim=259, condition_mode="shuffled_hash")
    wrong_inference = not torch.equal(_conditions(records, normal), _conditions(records, shuffled))
    assert wrong_inference
    _, _, shape = _codec("bgv")
    control = ExperimentConfig("bgv", "printable", "md5", 8, condition_mode="reversible_record", condition_dim=math.prod(shape))
    assert _build_model(normal, shape).input.in_channels == 2
    assert _build_model(control, shape).input.in_channels == 4
    rows = []
    for name, shape, vocabulary in (("BGV", (2, 32, 128), None), ("CGGE", (2, 32, 64), None),
                                     ("P-DISC", (32,), 97), ("R-DISC", (32,), 259)):
        torch.manual_seed(namespace_seed("benchmark", name))
        model = ImageUNet(2, 12, width=32) if vocabulary is None else SequenceDenoiser(vocabulary, 32, 12, width=128, embedding_dim=16)
        model.eval()
        sampler = (GaussianDiffusion(1000, device=torch.device("cpu")) if vocabulary is None
                   else MaskedDiffusion(vocabulary - 1, [i / 32 for i in range(33)], device=torch.device("cpu")))
        steps = 100 if vocabulary is None else 32
        condition = torch.zeros((4, 12))
        times = []
        for repetition in range(4):
            generator = torch.Generator().manual_seed(namespace_seed("benchmark_noise", name, repetition))
            start = perf_counter()
            sampler.sample(model, condition, shape, sampling_steps=steps, generator=generator,
                           **({} if vocabulary is None else {"temperature": 1.0}))
            elapsed = perf_counter() - start
            if repetition:
                times.append(elapsed / 4)
        rows.append({"name": name, "parameters": parameter_count(model), "batch_size": 4,
                     "sampling_steps": steps, "median_seconds_per_candidate": statistics.median(times),
                     "untrained_synthetic_cpu_only": True})
    projected = sum(row["median_seconds_per_candidate"] * 2048 * 100 * 6 * (2 if row["name"] == "BGV" else 1) for row in rows)
    return {"legacy_shuffled_inference_mismatch_reproduced": wrong_inference,
            "legacy_positive_control_changes_input_channels": True,
            "gaussian_terminal_alpha_bar": math.prod(1 - (.0001 + .0199 * i / 999) for i in range(1000)),
            "timing": rows, "same_unoptimized_cpu_policy_inference_hours_projection": projected / 3600,
            "projection_excludes": "training, validation, decoding, ledger IO, verification, controls and interruptions",
            "execution_readiness": "BLOCKED"}


assert len(CONFIG["pipelines"]) == 5 and CONFIG["q"] == 12 and CONFIG["primary_k"] == 100
assert sum(CONFIG["dataset"][key] for key in ("train_digest_slots", "validation_digest_slots", "test_digest_slots")) == 4096
assert CONFIG["learned_runs"] == len(CONFIG["pipelines"]) * 2 * len(CONFIG["model_seeds"]) == 30
excluded, exposure = exposure_inventory()
result = {"protocol_sha256": sha(CONFIG_PATH), "validator_sha256": sha(Path(__file__)),
          "plan_sha256": sha(ROOT / "RESEARCH_PLAN_V2.md"),
          "validated_at_utc": datetime.now(timezone.utc).isoformat(),
          "core_source_sha256": {name: sha(ROOT / "src/diffusion_hash_inv" / name)
                                 for name in ("dataset.py", "models.py", "discrete.py", "evaluation.py", "runner.py",
                                              "encoding/bgv.py", "encoding/cgge.py", "encoding/tokens.py")},
          "kind": "engineering_validation_not_scientific_gate_certification",
          "environment": {"python": platform.python_version(), "machine": platform.machine(),
                          "torch": torch.__version__, "numpy": np.__version__, "cpu_threads": 2,
                          "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available()},
          "exposure": exposure,
          "construction": {s: construction_check(s, excluded[s]) for s in CONFIG["sources"]},
          "codecs_and_controls": codecs_and_controls(), "statistics": statistics_check(),
          "implementation": implementation_and_timing(),
          "learned_candidate_count": 30 * 2048 * 100,
          "learned_candidate_nfe": 18 * 2048 * 100 * 100 + 12 * 2048 * 100 * 32}
(OUTPUT / "validation_results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"output": str(OUTPUT / "validation_results.json"),
                  "construction": result["construction"], "codec_cases": result["codecs_and_controls"]["clean_codec_roundtrips"],
                  "power": result["statistics"]["sensitivity"],
                  "estimated_unoptimized_cpu_inference_hours": result["implementation"]["same_unoptimized_cpu_policy_inference_hours_projection"],
                  "execution_readiness": "BLOCKED"}, indent=2))
