"""Registered v3.1 development P2A/B, with provisional (never formal) resources."""
from contextlib import closing
import math
import random
import shutil
import sqlite3
import time

import numpy as np
import torch

from . import study_pilot as pilot
from .study_profiles import profile_ids


def cases(p, data):
    """Choose only from training records, before inspecting any model output."""
    result = {}
    for source, corpus in data["sources"].items():
        first = {}
        for row in corpus["train"]:
            first.setdefault(row[0], row)
        pairs = [pair for pair in data["pairs"]["train"] if all(y in first for y in pair)]
        count = p["pilot"]["P2A"]["train_complement_pairs"]
        if len(pairs) < count:
            raise pilot.PilotError(f"P2A needs {count} observed training complement pairs for {source}", 2)
        result[source] = [first[y] for pair in pairs[:count] for y in pair]
    return result


def targets(p, data, source, stage, chosen):
    if stage == "P2A":
        return [(f"case:{y:03x}:trajectory:{draw}", y) for y, _ in chosen[source]
                for draw in range(1, p["pilot"][stage]["trajectories_per_condition_variant"] + 1)]
    pairs = random.Random(pilot.seed(p, stage, "trial-list", source=source)).sample(
        data["pairs"]["validation"], p["pilot"][stage]["probe_conditions"] // 2)
    return [(f"case:{y:03x}", y) for pair in pairs for y in pair]


def quality_gate(cfg, metrics):
    variants = metrics["variants"]
    denominator = cfg["gate_denominator_per_variant"]
    return (all(variants[v]["candidates"] == denominator for v in ("normal", "flipped"))
            and variants["normal"]["joint"] >= cfg["normal_joint_min"]
            and variants["flipped"]["joint"] >= cfg["flipped_joint_min"]
            and variants["flipped"]["wrong_original"] <= cfg["wrong_original_max"]
            and all(variants[v]["valid"] >= cfg.get("valid_min_each_variant", 0) for v in ("normal", "flipped")))


def quality_warnings(cfg, metrics):
    warnings = []
    if not metrics["normal_joint"] or not metrics["flipped_joint"]:
        warnings.append("JOINT_SUCCESS_ZERO")
    if any(metrics["variants"][v]["valid"] < cfg.get("valid_min_each_variant", cfg[v + "_joint_min"])
           for v in ("normal", "flipped")):
        warnings.append("FORMAT_FAILURE")
    if not quality_gate(cfg, metrics):
        warnings.append("CONDITION_RESPONSE_WEAK")
    return warnings


def completed(directory):
    path = directory / "complete.json"
    if not path.exists():
        return None
    record = pilot.read_json(path)
    pilot.verify_seal(directory, record["sha256"])
    if record["summary"] != pilot.read_json(directory / "summary.json"):
        raise pilot.PilotError("P2 completion summary changed")
    return record["summary"]


def finish(directory, summary):
    pilot.atomic_json(directory / "summary.json", summary)
    pilot.atomic_json(directory / "complete.json", {"summary": summary, "sha256": pilot.seal_directory(directory)})
    return summary


def selected_model(p, pipeline, profile_id, directory, device, backend, *, best=True):
    state, checksum = pilot.load_checkpoint(directory, best=best)
    model, diffusion = pilot.model_and_diffusion(p, pipeline, device, 0, profile_id=profile_id, backend=backend)
    if backend == "mlx":
        from .mlx_backend import load_weights
        load_weights(model, state["model"])
    else:
        model.load_state_dict(state["model"])
    model.eval()
    return model, diffusion, checksum, state["best_epoch"]


@torch.no_grad()
def diagnostics(p, pipeline, profile_id, model, diffusion, data, device, budget, directory, *, records=None, stage="P2B"):
    """Teacher-forced diagnostics; none of these predictions enters a success gate."""
    native = budget.backend == "mlx"
    if native:
        from . import mlx_backend as mlx
        mx = mlx.mx
    source = p["pipelines"][pipeline]["source"]
    discrete = p["pipelines"][pipeline]["model"] == "discrete"
    if records is None:
        records = data["sources"][source]["validation"][:64 if discrete else 16]
    encoder, _, _ = pilot.codecs(p["pipelines"][pipeline])
    clean = (mlx.clean_batch(encoder, records, diffusion) if native
             else pilot.clean_batch(encoder, records, device, discrete))
    cond = mlx.models.condition([r[0] for r in records]) if native else pilot.condition([r[0] for r in records], device)
    array = lambda value: np.array(value) if native else value.detach().cpu().numpy()
    original = array(clean)
    result = {"scope": "diagnostic_only_not_candidate_success", "stage": stage,
              "split": "training" if stage == "P2A" else "validation",
              "records": len(records), "records_sha256": pilot.digest(records),
              "conditions": [r[0] for r in records], "rows": []}
    model.eval()
    seed = pilot.seed(p, stage, "validation-noise", source=source, pipeline=pipeline, unit_id="diagnostics")
    if not discrete:
        noise = mx.random.normal(clean.shape, key=mx.random.key(seed)) if native else torch.randn(clean.shape, device=device, generator=pilot.generator(device, seed))
        for timestep in sorted({min(t, diffusion.steps - 1) for t in (0, 100, 300, 500, 700, 999)}):
            budget.check(directory)
            index = mx.full((len(clean),), timestep, dtype=mx.int32) if native else torch.full((len(clean),), timestep, device=device, dtype=torch.long)
            clock = mx.full((len(clean),), timestep / (diffusion.steps - 1)) if native else torch.full((len(clean),), timestep / (diffusion.steps - 1), device=device)
            noisy = diffusion.add_noise(clean, noise, index)
            output = model(noisy, clock, cond)
            reconstructed = diffusion.predicted_clean(output, noisy, index)
            alpha = array(diffusion.alpha_bar)[timestep]
            epsilon = output if diffusion.prediction_type == "epsilon" else (noisy - math.sqrt(alpha) * reconstructed) / math.sqrt(1 - alpha)
            x0 = array(reconstructed)
            error = (x0 - original) ** 2
            if not np.isfinite(x0).all() or not np.isfinite(array(epsilon)).all():
                raise FloatingPointError("Non-finite P2 Gaussian diagnostics")
            active = original[:, 1] > 0
            slot_width = original.shape[-1] // 8
            payload_active = active.copy()
            bgv = p["pipelines"][pipeline]["representation"] == "bgv"
            if bgv:
                payload_active[:, :8, :slot_width] = False
            result["rows"].append({"timestep": timestep, "noise_mse": float(((array(epsilon) - array(noise)) ** 2).mean()),
                                   "x0_mse": float(error.mean()), "mask_mse": float(error[:, 1].mean()),
                                   "active_glyph_mse": float(error[:, 0][active].mean()),
                                   "payload_glyph_mse": float(error[:, 0][payload_active].mean()),
                                   "padding_glyph_mse": float(error[:, 0][~active].mean()) if (~active).any() else None,
                                   "first_slot_mse": float(error[:, 0, :8, :slot_width].mean()),
                                   "length_header_mse": float(error[:, 0, :8, :slot_width].mean()) if bgv else None,
                                   "clipping_fraction": float((np.abs(x0) > 1).mean())})
    else:
        factorized = profile_id == "D1"
        lengths = diffusion.lengths(clean) if factorized else None
        if factorized:
            length_logits = array(model.length_head(cond))
            if not np.isfinite(length_logits).all():
                raise FloatingPointError("Non-finite P2 length diagnostics")
            shifted = length_logits - length_logits.max(-1, keepdims=True)
            log_distribution = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
            distribution = np.exp(log_distribution)
            observed = array(lengths).astype(int)
            length_ce = -log_distribution[np.arange(len(records)), observed - 4]
            result["length_head"] = [
                {"condition": record[0], "observed_training_length" if stage == "P2A" else "observed_length": int(observed[i]),
                 "argmax_length": int(distribution[i].argmax()) + 4,
                 "observed_length_probability": float(distribution[i, observed[i] - 4]),
                 "length_ce": float(length_ce[i]), "probabilities_lengths_4_to_31": distribution[i].tolist()}
                for i, record in enumerate(records)]
            if stage == "P2A" and (directory / "candidates.sqlite").exists():
                training = dict(records)
                result["generated_length_and_prefix"] = [
                    {"unit_id": row["unit_id"], "variant": row["variant"], "condition": row["requested_target"],
                     "sampled_length": row["sampled_length"], "joint_success": row["success"],
                     "matches_target_training_length": row["byte_length"] == len(bytes.fromhex(training[row["requested_target"]])),
                     "matches_target_training_message": row["candidate_hex"] == training[row["requested_target"]],
                     "matches_any_training_message": row["candidate_hex"] in training.values()}
                    for row in pilot.logical_ledger(directory)]
        for fraction in (.1, .5, .9, 1.):
            budget.check(directory)
            if native:
                mask = mx.random.uniform(shape=clean.shape, key=mx.random.key(seed)) < fraction
                if factorized:
                    mask = mask & (mx.arange(32)[None] < lengths[:, None])
                logits = model(mx.where(mask, diffusion.mask_token, clean), mx.full((len(clean),), fraction),
                               diffusion.payload_condition(cond, lengths) if factorized else cond)
            else:
                mask = torch.rand(clean.shape, device=device, generator=pilot.generator(device, seed)) < fraction
                if factorized:
                    mask &= torch.arange(32, device=device)[None] < lengths[:, None]
                logits = model(clean.masked_fill(mask, diffusion.mask_token), torch.full((len(clean),), fraction, device=device),
                               diffusion.payload_condition(cond, lengths) if factorized else cond)
            values = array(logits)[..., :diffusion.mask_token - 2] if factorized else array(logits)
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite P2 discrete diagnostics")
            shifted = values - values.max(-1, keepdims=True)
            log_probabilities = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
            probabilities = np.exp(log_probabilities)
            target_log_probability = np.take_along_axis(log_probabilities, np.minimum(original, values.shape[-1] - 1)[..., None], -1)[..., 0]
            correct = np.exp(target_log_probability)
            selected = array(mask).astype(bool)
            row = {"mask_fraction": fraction, "masked_positions": int(selected.sum()),
                   "logits_min": float(values.min()), "logits_max": float(values.max()),
                   "correct_probability_masked": float(correct[selected].mean()) if selected.any() else None,
                   "prefix_correct_probability_masked": [float(correct[:, i][selected[:, i]].mean()) if selected[:, i].any() else None for i in range(3)],
                   "argmax_accuracy_masked": float((values.argmax(-1) == original)[selected].mean()) if selected.any() else None}
            if factorized:
                row["eos_position_mean_probability"] = ([0.] * 4 + distribution.mean(0).tolist())
                row["length_ce"] = float(length_ce.mean())
                row["payload_ce"] = float(((-target_log_probability * selected).sum(1) / np.maximum(selected.sum(1), 1)).mean())
            else:
                row["eos_position_mean_probability"] = probabilities[..., diffusion.mask_token - 2].mean(0).tolist()
            result["rows"].append(row)
    budget.reserve(directory, len(pilot.canonical(result)) + 1024)
    pilot.atomic_json(directory / "diagnostics.json", result)
    return result


def probe(p, pipeline, profile_id, data, chosen, directory, epoch, device, budget):
    cfg = p["pilot"]["P2B"]
    if epoch not in cfg["probe_epochs"]:
        return
    output = directory / "probes" / f"epoch-{epoch:04d}"
    if completed(output) is not None:
        return
    model, diffusion, checksum, best_epoch = selected_model(p, pipeline, profile_id, directory, device, budget.backend)
    source = p["pipelines"][pipeline]["source"]
    metrics = pilot.evaluate(p, "P2B", pipeline, "main", cfg["model_seeds"][0], targets(p, data, source, "P2B", chosen),
                             output, model, diffusion, checksum, cfg["probe_inference_batch"], device, budget, profile_id=profile_id)
    diagnostics(p, pipeline, profile_id, model, diffusion, data, device, budget, output)
    warnings = quality_warnings(cfg, metrics)
    finish(output, {"epoch": epoch, "best_epoch": best_epoch, "checkpoint_sha256": checksum,
                    "metrics": metrics, "warnings": warnings, "passed": quality_gate(cfg, metrics)})
    print(f"[P2B] {pipeline}/{profile_id} epoch {epoch}: normal={metrics['normal_joint']}, flipped={metrics['flipped_joint']}; {warnings}", flush=True)


def overfit_probe(p, pipeline, profile_id, data, chosen, directory, update, device, budget):
    cfg = p["pilot"]["P2A"]
    # The final update is evaluated in the run root and is the only gate input.
    if update not in cfg.get("diagnostic_updates", []) or update == cfg["optimizer_updates_per_run"]:
        return
    output = directory / "probes" / f"update-{update:08d}"
    if completed(output) is not None:
        return
    model, diffusion, checksum, _ = selected_model(p, pipeline, profile_id, directory, device, budget.backend, best=False)
    source = p["pipelines"][pipeline]["source"]
    metrics = pilot.evaluate(p, "P2A", pipeline, "main", cfg["model_seeds"][0], targets(p, data, source, "P2A", chosen),
                             output, model, diffusion, checksum, p["pilot"]["P2B"]["probe_inference_batch"], device, budget,
                             profile_id=profile_id, settings={"condition_variants": cfg["condition_variants"], "k": cfg["k_per_trajectory"]})
    diagnostics(p, pipeline, profile_id, model, diffusion, data, device, budget, output, records=chosen[source], stage="P2A")
    finish(output, {"scope": "diagnostic_only_not_selection", "update": update, "checkpoint_sha256": checksum,
                    "metrics": metrics, "warnings": quality_warnings(cfg, metrics)})
    print(f"[P2A diagnostic] {pipeline}/{profile_id} update {update}: normal={metrics['normal_joint']}, flipped={metrics['flipped_joint']}", flush=True)


def resource_profile(p, pipeline, profile_id, model, diffusion, state, checksum, data, directory, device, budget):
    """Include preparation, budget, transfer, decode and SQLite in measured loops."""
    cfg = p["execution"]
    updates = state["full_loop_update_seconds"]
    if len(updates) < 320:
        raise pilot.PilotError("P2 resource profile requires 20 warm-up + three 100-update windows", 2)
    windows = [sum(updates[start:start + 100]) / 100 for start in (20, 120, 220)]
    source = p["pipelines"][pipeline]["source"]
    pool = data["sources"][source]["validation"]
    grid = []
    for batch in cfg["profile_batch_candidates"]:
        batch_directory = directory / "profile" / f"batch-{batch}"
        saved_batch = completed(batch_directory)
        if saved_batch is not None:
            grid.append(saved_batch)
            continue
        measured = []
        try:
            for repetition in range(cfg["profile_warmup_batches_per_size"] + cfg["profile_measured_batches_per_size"]):
                output = batch_directory / str(repetition)
                saved = completed(output)
                if saved is None:
                    batch_targets = [(f"profile:{batch}:{repetition}:{i}", pool[i % len(pool)][0]) for i in range(batch)]
                    budget.check(output)
                    started = time.monotonic()
                    pilot.evaluate(p, "P2B", pipeline, "main", 100, batch_targets, output, model, diffusion, checksum,
                                   batch, device, budget, profile_id=profile_id, settings={"condition_variants": ["normal"], "k": 1, "rng_namespace": "profile"})
                    elapsed = time.monotonic() - started
                    budget.tick()
                    elapsed = max(elapsed, budget.state.get("run_active_seconds", {}).get(str(output), 0.))
                    saved = finish(output, {"seconds": elapsed, "candidates": batch})
                if repetition >= cfg["profile_warmup_batches_per_size"]:
                    measured.append(saved["seconds"])
            row = {"batch": batch, "eligible": True, "seconds": measured, "candidates_per_second": len(measured) * batch / sum(measured)}
        except (FloatingPointError, RuntimeError) as error:
            if not isinstance(error, FloatingPointError) and "out of memory" not in str(error).lower():
                raise
            if budget.backend == "mlx":
                from .mlx_backend import mx
                mx.clear_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            pilot.synchronize(device)
            budget.check()
            row = {"batch": batch, "eligible": False, "reason": str(error)}
        grid.append(finish(batch_directory, row))
    eligible = [row for row in grid if row["eligible"]]
    if not eligible:
        return {"status": "NO_ELIGIBLE_BATCH", "grid": grid, "profile_id": profile_id}
    fastest = max(row["candidates_per_second"] for row in eligible)
    selected = min((row for row in eligible if row["candidates_per_second"] >= fastest * (1 - cfg["profile_tie_relative_throughput"])), key=lambda row: row["batch"])
    # Stress the complete row/index/WAL path with maximum-length valid payloads.
    stress = directory / "profile" / "stress"
    fixture = completed(stress)
    if fixture is None:
        budget.check(directory)
        budget.reserve(directory, 2 ** 20, refresh=True)
        stress.mkdir(parents=True, exist_ok=True)
        encoder, decoder, _ = pilot.codecs(p["pipelines"][pipeline])
        payload = b"f" * 31 if source == "printable" else b"\x0f" * 31
        encoded = encoder.encode(payload)
        template = pilot.logical_ledger(directory / "profile" / f"batch-{selected['batch']}" / "0")[0]
        template.update(candidate_hex=payload.hex(), byte_length=31, valid=True, reason=None,
                        requested_target=4095, original_target=4095, success=True, wrong_original=True,
                        sampled_length=31 if profile_id == "D1" else None, md5_calls=1, verifier_calls=1)
        started = time.monotonic()
        with closing(pilot.ledger_open(stress / "candidates.sqlite")) as connection:
            for i in range(100):
                budget.check(directory)
                decoded = decoder.decode(encoded)
                if not decoded.valid or not pilot.verify(decoded.message, source, 4095)[1]:
                    raise pilot.PilotError("P2 maximum-length fixture failed", 3)
                pilot.hashlib.md5(decoded.message).digest()
                row = dict(template, unit_id=f"stress:{i}", attempt=1)
                with connection:
                    connection.execute("INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?)", ("stress", row["unit_id"], "normal", 1, pilot.canonical(row).decode()))
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        elapsed = time.monotonic() - started
        fixture = finish(stress, {"seconds_per_candidate": elapsed / 100,
                                  "row_bytes": (stress / "candidates.sqlite").stat().st_size / 100,
                                  "directory_files_at_measurement": sum(f.is_file() for f in budget.root.rglob('*'))})
    result = {"grid": grid, "selected_batch": selected["batch"], "update_seconds_windows": windows,
              "update_seconds": max(windows), "validation_seconds": max(state["validation_seconds"]),
              "checkpoint_seconds": max(state["checkpoint_seconds"], default=0),
              "candidate_seconds": max(max(selected["seconds"]) / selected["batch"], fixture["seconds_per_candidate"]),
              "row_bytes": fixture["row_bytes"], "stress": fixture,
              "training_timing_excludes": ["validation", "checkpoint", "probes"],
              "generation_timing_includes": ["budget", "sample", "transfer", "decode", "synthetic_verify", "ledger_commit", "ledger_reverify", "metrics_write"],
              "main_cost_status": "PROVISIONAL_REQUIRES_E0", "profile_id": profile_id,
              "checkpoint_bytes": sum(f.stat().st_size for f in (directory / 'checkpoints').iterdir()),
              "raw_bytes_per_candidate": 4 * math.prod(pilot.codecs(p["pipelines"][pipeline])[2])}
    pilot.atomic_json(directory / "profile.json", result)
    return result


def estimates(p, root, profiles):
    cfg = p["execution"]
    runs = {}
    for pipeline, row in profiles.items():
        def seconds(spec, candidates):
            checkpoints = math.ceil(spec["optimizer_updates_per_run"] / p["training"]["checkpoint_every_optimizer_updates"]) + spec["epochs"] + 1
            return cfg["time_budget_multiplier"] * (spec["optimizer_updates_per_run"] * row["update_seconds"]
                    + len(spec["validation_epochs"]) * row["validation_seconds"] + checkpoints * row["checkpoint_seconds"]
                    + candidates * row["candidate_seconds"])
        p3 = seconds(p["pilot"]["P3"], 2 * p["pilot"]["P3"]["evaluation_unique_conditions"])
        main_candidates = p["main"]["evaluation_trials"] * p["main"]["k"]
        main = seconds(p["main"], main_candidates)
        run_storage = cfg["storage_budget_multiplier"] * (main_candidates * row["row_bytes"] + row["checkpoint_bytes"]
                      + 16 * row["raw_bytes_per_candidate"] + 4096 * p["main"]["optimizer_updates_per_run"])
        runs[pipeline] = {"profile_id": row["profile_id"], "batch_size": row["selected_batch"], "p3_run_seconds": p3,
                          "main_run_seconds_provisional": main, "main_run_storage_bytes_provisional": run_storage}
    p3_seconds = sum(r["p3_run_seconds"] * 3 for r in runs.values())
    main_seconds = sum(r["main_run_seconds_provisional"] * 6 for r in runs.values())
    max_row = max((r["row_bytes"] for r in profiles.values()), default=0)
    # Include all planned rows, SQLite/index overhead from stress, checkpoints,
    # raw samples, and a conservative per-update telemetry allowance.
    metadata = sum(r["checkpoint_bytes"] * 9 + r["raw_bytes_per_candidate"] * 16 * 9 for r in profiles.values())
    telemetry = len(profiles) * 9 * p["pilot"]["P3"]["optimizer_updates_per_run"] * 4096
    future = cfg["storage_budget_multiplier"] * (max_row * (p["main"]["candidate_rows_full_matrix"] + p["pilot"]["P3"]["candidates"]) + metadata + telemetry)
    stored = sum(f.stat().st_size for f in root.rglob('*') if f.is_file())
    ready = (p3_seconds <= cfg["hard_stage_active_wall_seconds"]["P3"]
             and main_seconds <= cfg["hard_stage_active_wall_seconds"]["MAIN"]
             and all(max(r["p3_run_seconds"], r["main_run_seconds_provisional"]) <= cfg["hard_formal_run_active_wall_seconds"] for r in runs.values())
             and all(r["main_run_storage_bytes_provisional"] <= cfg["hard_run_storage_gib"] * pilot.GIB for r in runs.values())
             and stored + future <= cfg["hard_study_storage_gib"] * pilot.GIB
             and future + cfg["minimum_disk_free_gib"] * pilot.GIB <= shutil.disk_usage(root).free)
    return {"status": "PROVISIONAL_PASS" if ready else "BLOCKED_RESOURCE", "candidate_resources_pass": ready,
            "pipelines": runs, "p3_soft_wall_seconds": p3_seconds, "main_learned_seconds_provisional": main_seconds,
            "future_storage_bytes_provisional": future, "final_sealed": False, "main_ready": False,
            "requires": ["E0 production MD5 costs and row sizes", "exposure audit", "statistical calibration"],
            "missing_costs": {"E0": None, "main_random_generation": None, "main_analysis": None}}


def phase(p, stage, pipeline, profile_id, data, chosen, directory, device, budget, selected_profiles):
    saved = completed(directory)
    if saved is not None:
        return saved
    cfg = p["pilot"][stage]
    source = p["pipelines"][pipeline]["source"]
    callback_fn = overfit_probe if stage == "P2A" else probe
    callback = lambda epoch: callback_fn(p, pipeline, profile_id, data, chosen, directory, epoch, device, budget)
    model, diffusion, state, checksum = pilot.train_model(
        p, stage, pipeline, "main", cfg["model_seeds"][0], data, directory, device, budget, profile_id=profile_id,
        training_rows=chosen[source] if stage == "P2A" else None, epoch_callback=callback)
    pilot.atomic_json(directory / "training.json", {k: state[k] for k in ("update", "best_epoch", "best_loss", "curve", "full_loop_update_seconds")})
    if stage == "P2A":
        metrics = pilot.evaluate(p, stage, pipeline, "main", cfg["model_seeds"][0], targets(p, data, source, stage, chosen),
                                 directory, model, diffusion, checksum, p["pilot"]["P2B"]["probe_inference_batch"], device, budget,
                                 profile_id=profile_id, settings={"condition_variants": cfg["condition_variants"], "k": cfg["k_per_trajectory"]})
        diagnostics(p, pipeline, profile_id, model, diffusion, data, device, budget, directory, records=chosen[source], stage=stage)
        warnings = quality_warnings(cfg, metrics)
    else:
        final = completed(directory / "probes" / f"epoch-{cfg['gate_epoch']:04d}")
        if final is None:
            raise pilot.PilotError("P2B final probe is missing")
        metrics, warnings = final["metrics"], final["warnings"]
        pilot.atomic_json(directory / "metrics.json", metrics)
    passed = quality_gate(cfg, metrics)
    summary = {"status": "PASS" if passed else "FAIL_QUALITY", "passed": passed, "metrics": metrics,
               "profile_id": profile_id, "stage": stage, "updates": state["update"],
               "best_epoch": state["best_epoch"], "checkpoint_sha256": checksum, "warnings": warnings}
    if stage == "P2B" and passed:
        resource = resource_profile(p, pipeline, profile_id, model, diffusion, state, checksum, data, directory, device, budget)
        if resource.get("status") == "NO_ELIGIBLE_BATCH":
            summary.update(status="BLOCKED_RESOURCE", passed=False, resources=resource)
            return finish(directory, summary)
        preview = estimates(p, budget.root, {**selected_profiles, pipeline: resource})
        summary["resources"] = resource
        summary["resource_estimate"] = preview
        summary["passed"] = preview["candidate_resources_pass"]
        if summary["passed"]:
            summary["recovery"] = pilot.recovery_check(p, pipeline, data, directory / "selected_batch_recovery", directory,
                                                       model, diffusion, checksum, state, device, budget,
                                                       resource["selected_batch"], train=False, profile_id=profile_id)
        else:
            summary["status"] = "BLOCKED_RESOURCE"
    return finish(directory, summary)


def execute(p, root, data, device, budget):
    chosen = cases(p, data)
    path = root / "pilot/P2/cases.json"
    if path.exists() and pilot.read_json(path) != chosen:
        raise pilot.PilotError("P2A case selection changed")
    pilot.atomic_json(path, chosen)
    selections, profiles, runs = {}, {}, {}
    for pipeline in p["pipeline_order"]:
        selected = None
        for profile_id in profile_ids(p, pipeline):
            for suffix, stage in (("A", "P2A"), ("B", "P2B")):
                name = f"{pipeline}/{profile_id}/{suffix}"
                directory = root / "pilot/P2" / name
                if selected:
                    runs[name] = {"status": "NOT_NEEDED"}
                    continue
                if suffix == "B" and not runs[f"{pipeline}/{profile_id}/A"]["passed"]:
                    runs[name] = {"status": "NOT_RUN_P2A_FAILED"}
                    continue
                print(f"[{stage}] {pipeline}/{profile_id}", flush=True)
                try:
                    runs[name] = phase(p, stage, pipeline, profile_id, data, chosen, directory, device, budget, profiles)
                except (FloatingPointError, pilot.PilotError) as error:
                    local = isinstance(error, FloatingPointError) or error.code == 3 or (error.code == 5 and str(error) == "Per-run active wall-clock cap reached")
                    if not local:
                        raise
                    # Continue only after the backend and global limits are healthy.
                    pilot.synchronize(device)
                    budget.check()
                    directory.mkdir(parents=True, exist_ok=True)
                    pilot.atomic_json(directory / "failure.json", {"error": str(error), "stage": stage})
                    runs[name] = finish(directory, {"status": "FAILED_RUNTIME", "passed": False, "metrics": None, "error": str(error)})
                if suffix == "B" and runs[name]["passed"]:
                    selected = profile_id
                    profiles[pipeline] = runs[name]["resources"]
            selections[pipeline] = {"profile_id": selected, "status": "SELECTED" if selected else "BLOCKED_DEVELOPMENT"}
            pilot.atomic_json(root / "profile_selection.json", {"development_only": True, "pipelines": selections, "runs": runs})
    ready = all(value["profile_id"] for value in selections.values())
    resources = estimates(p, root, profiles) if profiles else {"status": "NOT_MEASURED", "pipelines": {}, "final_sealed": False, "main_ready": False}
    if ready and not resources["candidate_resources_pass"]:
        ready = False
    frozen = {"development_only": True, "protocol_sha256": pilot.digest(p), "complete": ready,
              "pipelines": {name: {"profile_id": row["profile_id"], "profile": p["model_profiles"][row["profile_id"]],
                                    "batch_size": row["selected_batch"]} for name, row in profiles.items()}}
    pilot.atomic_json(root / "profile.frozen.json", frozen)
    pilot.atomic_json(root / "resources.json", resources)
    return {"stage": "P2", "status": "PASS" if ready else "BLOCKED_DEVELOPMENT", "exit_code": 0 if ready else 2,
            "runs": runs, "selected_profiles": selections, "resources": resources, "formal_qualified": False}


def report_rows(root, p, lines, progress):
    directory = root / "pilot/P2"
    if not directory.exists():
        return
    selection_path = root / "profile_selection.json"
    selection = pilot.read_json(selection_path) if selection_path.exists() else {"runs": {}}
    lines += ["", "## P2A/B 개발 검사", "", "| Run | 실행/선택 상태 | 정상 joint | 반전 joint |", "|---|---|---:|---:|"]
    for pipeline in p["pipeline_order"]:
        for profile_id in profile_ids(p, pipeline):
            for suffix in ("A", "B"):
                name = f"{pipeline}/{profile_id}/{suffix}"
                path = directory / name
                summary = completed(path)
                if summary is None:
                    status = "INCOMPLETE" if (path / "configuration.json").exists() else selection["runs"].get(name, {}).get("status", "NOT_RUN")
                    metrics = {}
                else:
                    status, metrics = summary["status"], summary.get("metrics") or {}
                progress[f"P2/{name}"] = status
                lines.append(f"| {name} | {status} | {metrics.get('normal_joint', '—')} | {metrics.get('flipped_joint', '—')} |")
                for probe_path in sorted((path / "probes").glob("update-*" if suffix == "A" else "epoch-*")):
                    saved = completed(probe_path)
                    if saved is not None:
                        m = saved["metrics"]
                        progress[f"P2/{name}/{probe_path.name}"] = "COMPLETE"
                        detail = "diagnostic only" if suffix == "A" else f"best epoch {saved['best_epoch']}"
                        lines.append(f"| {name}/{probe_path.name} | COMPLETE ({detail}) | {m['normal_joint']} | {m['flipped_joint']} |")
    lines += ["", "P2 resources are provisional. E0/final resource seal and formal qualification remain required."]
