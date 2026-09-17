"""Small deterministic checks at every reversible diffusion boundary."""

import hashlib

import pytest
import torch
from torch import nn

from diffusion_hash_inv.dataset import build_digest_records
from diffusion_hash_inv.encoding import BGVDecoder, BGVEncoder, CGGEDecoder, CGGEEncoder, DirectBitsDecoder, DirectBitsEncoder
from diffusion_hash_inv.evaluation import verify_candidate
from diffusion_hash_inv.models import GaussianDiffusion
from diffusion_hash_inv.runner import ExperimentConfig, _codec, _conditions


@pytest.mark.parametrize(
    ("encoder", "decoder", "message"),
    (
        (BGVEncoder(), BGVDecoder(), b"\x00\xff\xaa\x55"),
        (CGGEEncoder(), CGGEDecoder(), b"!~A0"),
        (DirectBitsEncoder(), DirectBitsDecoder(), bytes(range(31))),
    ),
)
def test_encoder_determinism_and_normalization_roundtrip(encoder, decoder, message) -> None:
    encoded = encoder.encode(message)
    assert torch.equal(encoded, encoder.encode(message))
    normalized = encoded * 2 - 1
    assert torch.equal((normalized + 1) / 2, encoded)
    assert decoder.decode(normalized, normalized=True).message == message


@pytest.mark.parametrize("representation", ("bgv", "cgge", "bits"))
def test_condition_target_alignment(representation: str) -> None:
    message = b"ABCD"
    encoder, _, shape = _codec(representation)
    record = build_digest_records((message,), source="printable", algorithm="md5", q=8)
    config = ExperimentConfig(
        representation=representation,
        source="printable",
        algorithm="md5",
        q=8,
        dataset_size=1,
        condition_mode="reversible_record",
        condition_dim=int(torch.tensor(shape).prod()),
    )
    assert torch.equal(_conditions(record, config)[0], encoder.encode(message).flatten())


def test_hash_reference_and_determinism() -> None:
    target = build_digest_records((b"ABCD",), source="printable", algorithm="md5", q=128)[0]
    first = verify_candidate(b"ABCD", target)
    assert first == verify_candidate(b"ABCD", target)
    assert first.digest == hashlib.md5(b"ABCD").hexdigest()
    assert first.prefix_match


def test_forward_diffusion_shape_range_and_finite_values() -> None:
    diffusion = GaussianDiffusion(10, beta_end=0.4, prediction_type="sample", device=torch.device("cpu"))
    clean = torch.tensor([[[-1.0, 1.0]]])
    noise = torch.zeros_like(clean)
    noisy = diffusion.add_noise(clean, noise, torch.tensor([9]))
    assert noisy.shape == clean.shape
    assert torch.isfinite(noisy).all()
    assert -1 <= noisy.min() <= noisy.max() <= 1
    assert 0 < diffusion.alpha_bar[-1] < diffusion.alpha_bar[0] <= 1


class _ConditionOracle(nn.Module):
    def forward(self, value, time, condition):
        return condition.reshape_as(value)


class _EpsilonOracle(nn.Module):
    def __init__(self, diffusion):
        super().__init__()
        self.diffusion = diffusion

    def forward(self, value, time, condition):
        index = (time * (self.diffusion.steps - 1)).round().long()
        alpha = self.diffusion.alpha_bar[index].reshape((len(value),) + (1,) * (value.ndim - 1))
        clean = condition.reshape_as(value)
        return (value - alpha.sqrt() * clean) / (1 - alpha).sqrt()


def test_sampling_shape_finite_values_candidate_decode_and_hash_verification() -> None:
    message = b"ABCD"
    encoder, decoder = DirectBitsEncoder(), DirectBitsDecoder()
    clean = encoder.encode(message) * 2 - 1
    diffusion = GaussianDiffusion(10, beta_end=0.4, prediction_type="sample", device=torch.device("cpu"))
    sample = diffusion.sample(
        _ConditionOracle(),
        clean.flatten()[None],
        tuple(clean.shape),
        sampling_steps=10,
        generator=torch.Generator().manual_seed(0),
    )[0]
    decoded = decoder.decode(sample, normalized=True)
    target = build_digest_records((message,), source="printable", algorithm="md5", q=128)[0]
    assert sample.shape == clean.shape and torch.isfinite(sample).all()
    assert decoded.valid and decoded.message == message
    assert verify_candidate(decoded.message, target).prefix_match


def test_epsilon_training_target_and_sampler_use_the_same_parameterization() -> None:
    clean = torch.tensor([[[-1.0, 1.0]]])
    condition = clean.flatten(1)
    diffusion = GaussianDiffusion(10, beta_end=0.4, prediction_type="epsilon", device=torch.device("cpu"))
    oracle = _EpsilonOracle(diffusion)
    assert diffusion.loss(oracle, clean, condition, generator=torch.Generator().manual_seed(0)).item() < 1e-10
    sample = diffusion.sample(
        oracle,
        condition,
        tuple(clean.shape[1:]),
        sampling_steps=10,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.allclose(sample, clean, atol=1e-5)
