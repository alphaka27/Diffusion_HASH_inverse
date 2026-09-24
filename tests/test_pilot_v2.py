"""Small engineering checks, including injected interruption; no scientific gate claims."""
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from diffusion_hash_inv import pilot_v2 as pilot


def fixture_spec(pipeline="P-DISC", device="cpu"):
    protocol = json.loads((Path(__file__).resolve().parents[1] / "examples/poc-v2-protocol.json").read_text())
    return dict(scope="development_synthetic_pilot_not_g1_certification", pipeline=pipeline, model_seed=0,
                source="printable" if pipeline.startswith("P-") else "random_bytes", method="main",
                protocol_id=protocol["protocol_id"], master_seed=protocol["dataset"]["engineering_seed"],
                train_messages=10, validation_conditions=4, test_conditions=4, checkpoint_every_updates=1,
                prefix_diagnostic_targets=2, prefix_diagnostic_k=100, device=device,
                gaussian=protocol["gaussian"], discrete=protocol["discrete"],
                training={**protocol["training"], "batch_size": 4, "epochs": 2, "validation_every_epochs": 1})


def test_synthetic_partition_prefix_and_narrow_conditions():
    spec = fixture_spec()
    data = pilot.synthetic_dataset(spec)
    assert data == pilot.synthetic_dataset(spec)
    groups = [data["train_condition_pool"], [r["condition"] for r in data["validation"]], data["test_conditions"]]
    assert not (set(groups[0]) & set(groups[1]) or set(groups[0]) & set(groups[2]) or set(groups[1]) & set(groups[2]))
    assert all(y ^ 4095 in group for group in groups for y in group)
    for row in data["train"] + data["validation"]:
        assert pilot.synthetic_verifier(bytes.fromhex(row["message_hex"]), row["condition"], "printable") == (True, True)
    assert pilot.synthetic_verifier(b"abc!", 0xABC, "printable") == (True, True)
    assert pilot.synthetic_verifier(b"abc ", 0xABC, "printable") == (False, False)
    assert pilot.synthetic_verifier(b"\x0a\x0b\x0c\xff", 0xABC, "random_bytes") == (True, True)
    assert pilot.condition_bits([0, 4095], "cpu").tolist() == [[0.] * 12, [1.] * 12]
    for invalid in ([4096], [-1], [True], [{"condition": 1, "message": "hidden"}]):
        with pytest.raises(ValueError):
            pilot.condition_bits(invalid, "cpu")
    with pytest.raises(ValueError):
        pilot.check_conditions(torch.zeros(1, 259))
    metadata = [{"condition": 123, "message": b"ABCD", "length": 4, "suffix": "00", "id": 1}]
    first = pilot.condition_bits([r["condition"] for r in metadata], "cpu")
    metadata[0].update(message=b"Z" * 31, length=31, suffix="ff", id=999)
    assert torch.equal(first, pilot.condition_bits([r["condition"] for r in metadata], "cpu"))


@pytest.mark.parametrize("pipeline", ["P-G-BGV", "P-G-CGGE", "P-DISC", "R-G-BGV", "R-DISC"])
def test_all_components_use_twelve_bits_and_existing_shapes(pipeline):
    torch.set_num_threads(1)
    spec = fixture_spec(pipeline)
    model, diffusion, encoder, _, shape = pilot.components(spec, torch.device("cpu"))
    data = pilot.synthetic_dataset(spec)
    clean, conditions = pilot.encode_rows(data["train"][:2], encoder, "cpu")
    assert clean.shape == (2, *shape) and conditions.shape == (2, 12)
    loss = diffusion.loss(model, clean, conditions, generator=torch.Generator().manual_seed(7))
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("pipeline", ["P-DISC", "P-G-BGV"])
def test_mid_epoch_resume_and_validation_selection(tmp_path, device, pipeline):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    torch.set_num_threads(1)
    spec = fixture_spec(pipeline, device=device)
    data = pilot.synthetic_dataset(spec)
    model, diffusion, encoder, _, shape = pilot.components(spec, torch.device(device))
    clean, conditions = pilot.encode_rows(data["train"], encoder, device)
    validation = pilot.encode_rows(data["validation"], encoder, device)
    original_save = pilot.save_torch

    def interrupt_after_commit(path, payload):
        original_save(path, payload)
        if path.name == "training_resume.pt" and payload["state"]["updates"] == 2:
            raise KeyboardInterrupt("injected interruption after durable batch 2")

    with patch.object(pilot, "save_torch", side_effect=interrupt_after_commit):
        with pytest.raises(KeyboardInterrupt):
            pilot.train(model, diffusion, clean, conditions, validation, spec, tmp_path / "resumed")
    model, diffusion, _, _, _ = pilot.components(spec, torch.device(device))
    resumed = pilot.train(model, diffusion, clean, conditions, validation, spec, tmp_path / "resumed")
    full_model, full_diffusion, _, _, _ = pilot.components(spec, torch.device(device))
    full = pilot.train(full_model, full_diffusion, clean, conditions, validation, spec, tmp_path / "full")
    assert resumed["updates"] == full["updates"] == 6  # The final batch of 2 is retained.
    assert resumed["history"] == full["history"]
    assert resumed["best_epoch"] == min(full["history"], key=lambda row: row["validation_loss"])["epoch"]
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), full_model.parameters()))
    with patch("torch.optim.Adam.step", side_effect=AssertionError("must not retrain")):
        again = pilot.train(model, diffusion, clean, conditions, validation, spec, tmp_path / "resumed")
    assert again == resumed
    with pytest.raises(RuntimeError, match="identity"):
        pilot.train(model, diffusion, clean + 1, conditions, validation, spec, tmp_path / "resumed")
    options = dict(generator=torch.Generator(device=device).manual_seed(3))
    a = pilot.generate(model, diffusion, conditions[:1], shape, spec, **options)
    b = pilot.generate(model, diffusion, conditions[:1], shape, spec,
                       generator=torch.Generator(device=device).manual_seed(3))
    assert torch.equal(a, b)


def test_shuffle_training_only_and_fixed_validation_noise():
    spec = fixture_spec()
    model, diffusion, encoder, _, _ = pilot.components(spec, torch.device("cpu"))
    data = pilot.synthetic_dataset(spec)
    clean, conditions = pilot.encode_rows(data["train"], encoder, "cpu")
    shuffled = {**spec, "method": "shuffled"}
    order, paired, _ = pilot.epoch_pairing(conditions, spec, 0)
    shuffled_order, shuffled_conditions, _ = pilot.epoch_pairing(conditions, shuffled, 0)
    assert torch.equal(order, shuffled_order) and torch.equal(paired, conditions)
    assert not torch.equal(shuffled_conditions, conditions)
    assert not torch.equal(shuffled_conditions, pilot.epoch_pairing(conditions, shuffled, 1)[1])
    assert sorted(map(tuple, shuffled_conditions.tolist())) == sorted(map(tuple, conditions.tolist()))
    a = pilot.validation_loss(model, diffusion, clean, conditions, spec)
    b = pilot.validation_loss(model, diffusion, clean, conditions, shuffled)
    assert a == b  # Validation does not use method-dependent shuffling/noise.
    with patch.object(diffusion, "sample", return_value=clean[:1]) as sample:
        pilot.generate(model, diffusion, conditions[:1], (32,), shuffled, generator=torch.Generator().manual_seed(3))
    assert torch.equal(sample.call_args.args[1], conditions[:1])  # No inference-time shuffle.


def test_candidate_commit_resume_prefix_budget_and_ledger_audit(tmp_path):
    spec = fixture_spec("P-G-BGV")
    model, diffusion, _, decoder, shape = pilot.components(spec, torch.device("cpu"))
    targets = pilot.synthetic_dataset(spec)["test_conditions"]
    calls = []

    def invalid_sample(model, diffusion, conditions, shape, spec, *, generator):
        calls.append(conditions.clone())
        if len(calls) == 3:
            raise KeyboardInterrupt("injected before third commit")
        return torch.zeros((len(conditions), *shape))  # Repeated invalid attempts must still consume K.

    with patch.object(pilot, "generate", side_effect=invalid_sample):
        with pytest.raises(KeyboardInterrupt):
            pilot.candidate_stream(model, diffusion, decoder, shape, spec, tmp_path, targets, variant="normal", k=1)
    with sqlite3.connect(tmp_path / "candidates.sqlite") as db:
        before = db.execute("SELECT * FROM attempts ORDER BY target,position").fetchall()
    assert len(before) == 2
    with patch.object(pilot, "generate", return_value=torch.zeros((1, *shape))) as generate:
        pilot.candidate_stream(model, diffusion, decoder, shape, spec, tmp_path, targets, variant="normal", k=1)
        assert generate.call_count == 2
        pilot.candidate_stream(model, diffusion, decoder, shape, spec, tmp_path, targets, variant="flipped", k=1)
        pilot.candidate_stream(model, diffusion, decoder, shape, spec, tmp_path, targets[:2], variant="normal", k=100)
    with sqlite3.connect(tmp_path / "candidates.sqlite") as db:
        after = db.execute("SELECT * FROM attempts WHERE variant='normal' AND position=1 ORDER BY target").fetchall()
        assert after[:2] == before
        # Same initial seeds for normal/flip; exactly 4 retained raw samples, not one per candidate.
        assert db.execute("SELECT count(*) FROM attempts WHERE raw IS NOT NULL").fetchone()[0] == 4
        seeds = db.execute("SELECT count(*) FROM attempts a JOIN attempts b ON a.target=b.target AND a.position=b.position WHERE a.variant='normal' AND b.variant='flipped' AND a.rng_seed=b.rng_seed").fetchone()[0]
        assert seeds == 4
    result = pilot.summarize(tmp_path, spec, targets, {})
    assert result["candidate_count"] == 206 and result["pilot_thresholds"] == "FAIL"
    assert all(r["successes"] == 0 for r in result["prefix_diagnostic"].values())
    with sqlite3.connect(tmp_path / "candidates.sqlite") as db:
        db.execute("UPDATE attempts SET correct=1 WHERE variant='normal' AND position=1")
    with pytest.raises(RuntimeError, match="verifier mismatch"):
        pilot.summarize(tmp_path, spec, targets, {})
