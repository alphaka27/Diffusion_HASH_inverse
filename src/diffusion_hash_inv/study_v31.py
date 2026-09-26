"""v3.1 implementation readiness and development checks (no main-study gate)."""
import torch

from . import study_pilot as pilot
from .study_profiles import LengthMaskedDiffusion


def readiness():
    return {"status": "IMPLEMENTATION_IN_PROGRESS", "implementation_ready": False,
            "model_profiles": {name: "IMPLEMENTED" for name in ("G0", "G1", "G2", "D0", "D1")},
            "development_pilot": {"P0": "IMPLEMENTED", "P1": "IMPLEMENTED"},
            "formal_pilot": "BLOCKED_IMPLEMENTATION",
            "remaining": ["P0 main-boundary/budget fixtures", "P2A/P2B selection and diagnostics",
                          "resource profiling and final seal", "exposure audit", "E0 production MD5 rehearsal",
                          "P3 qualification", "main M0-M3", "full statistical calibration",
                          "independent run failure continuation"],
            "poc_qualified": False, "main_ready": False, "study_complete": False}


def plan(p, stage, device, development):
    return {"protocol": p["protocol_id"], "stage": stage,
            "mode": "DEVELOPMENT_ONLY" if development else "V31_PILOT",
            "device": device, "dry_run": True, "device_checked": False,
            "settings": ({key: p["pilot"][key] for key in ("P2A", "P2B")} if stage == "P2" else p["pilot"][stage]),
            "pipelines": p["pipeline_order"], "model_profiles": p["model_profiles"],
            "executable": development and stage in {"P0", "P1"},
            "readiness": readiness(), "wall_cap_seconds": p["execution"]["hard_stage_active_wall_seconds"][stage],
            "note": "Only development P0/P1 are implemented. No experiment was run; formal gates remain blocked."}


@torch.no_grad()
def profile_checks(p, pipeline, profile_id, model, diffusion, device):
    """Actual profile forwards plus deterministic single-trajectory reference."""
    encoder, decoder, shape = pilot.codecs(p["pipelines"][pipeline])
    source = p["pipelines"][pipeline]["source"]
    targets, seeds = [0, 4095], [81, 82]
    lengths = [181, 182] if profile_id == "D1" else None
    one = pilot.sample(p, pipeline, model, diffusion, targets[:1], seeds[:1], device,
                       profile_id=profile_id, length_seeds=lengths[:1] if lengths else None)
    many = pilot.sample(p, pipeline, model, diffusion, targets, seeds, device,
                        profile_id=profile_id, length_seeds=lengths)
    options = {"temperature": 1.} if isinstance(diffusion, pilot.MaskedDiffusion) else {}
    if lengths:
        options["length_generator"] = pilot.generator(device, lengths[0])
    reference = diffusion.sample(model, pilot.condition(targets[:1], device), shape,
                                 sampling_steps=p["model_profiles"][profile_id]["sampling_steps"],
                                 generator=pilot.generator(device, seeds[0]), **options).cpu()
    single_agrees = torch.equal(one, reference)
    # Float32 convolution kernels differ by batch size; epsilon-to-x0 amplifies
    # their rounding through 100 steps. Recovery at a fixed batch stays exact.
    batch_agrees = (torch.equal(one[0], many[0]) if one.dtype == torch.long
                    else torch.allclose(one[0], many[0], atol=1e-3, rtol=1e-4))
    decoded_one = decoder.decode(one[0]) if one.dtype == torch.long else decoder.decode(one[0], normalized=True)
    decoded_many = decoder.decode(many[0]) if many.dtype == torch.long else decoder.decode(many[0], normalized=True)
    decoder_agrees = (decoded_one.valid, decoded_one.message, decoded_one.reason) == (decoded_many.valid, decoded_many.message, decoded_many.reason)
    fixture = encoder.encode(b"000!" if source == "printable" else b"\x00\x00\x00\x00")[None].to(device)
    if isinstance(diffusion, LengthMaskedDiffusion):
        intrinsic_valid = all(decoder.decode(row).valid for row in many)
        loss = diffusion.loss(model, fixture, pilot.condition([0], device), generator=pilot.generator(device, 83))
        target_matches = intrinsic_valid and torch.isfinite(loss).item()
    elif isinstance(diffusion, pilot.GaussianDiffusion):
        clean = fixture * 2 - 1
        indices = torch.tensor([diffusion.steps - 1], device=device)
        noise = torch.ones_like(clean) * .125
        noisy = diffusion.add_noise(clean, noise, indices)
        output = noise if diffusion.prediction_type == "epsilon" else clean
        target_matches = torch.allclose(diffusion.predicted_clean(output, noisy, indices), clean, atol=2e-5, rtol=2e-5)
    else:
        target_matches = torch.isfinite(diffusion.loss(model, fixture, pilot.condition([0], device),
                                                      generator=pilot.generator(device, 83))).item()
    return {"passed": bool(single_agrees and batch_agrees and decoder_agrees and target_matches),
            "single_reference_exact": bool(single_agrees), "batch_numerically_equivalent": bool(batch_agrees),
            "batch_decoder_equal": decoder_agrees, "training_parameterization": bool(target_matches),
            "sampling_nfe": 4 * p["model_profiles"][profile_id]["sampling_nfe_per_candidate"],
            "batch_max_absolute_difference": float((one[0] - many[0]).abs().max()),
            "gaussian_batch_atol": 1e-3, "gaussian_batch_rtol": 1e-4,
            "scope": "engineering_fixtures_not_generation_quality_qualification"}
