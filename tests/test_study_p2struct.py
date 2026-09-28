"""Structural generation and objective checks; no quality gate is relaxed in the CLI."""
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_hash_inv import study_pilot as pilot, study_p2, study_cli

SPEC = Path(__file__).parents[1] / 'examples/poc-v3.1-p2struct-protocol.json'


def small_protocol():
    p = deepcopy(pilot.load_protocol(SPEC))
    p['synthetic'].update(train_conditions=8, validation_conditions=8, train_unique_messages=16)
    p['training'].update(batch_size=2, validation_draws_per_condition=1)
    p['pilot']['P1'].update(train_messages=8, validation_conditions=2, evaluation_trials=2)
    p['execution']['minimum_disk_free_gib'] = 0
    for name in ('G3', 'D1'):
        p['model_profiles'][name].update(sampling_steps=2, sampling_nfe_per_candidate=3)
    p['model_profiles']['G3']['diffusion_steps'] = 10
    return p


def runtime(backend):
    if backend == 'torch':
        torch.set_num_threads(1)
        return torch.device('cpu'), None
    mlx = pytest.importorskip('diffusion_hash_inv.mlx_backend', exc_type=ImportError)
    if backend == 'mlx_gpu' and not mlx.mx.metal.is_available():
        pytest.skip('native Metal access required')
    return mlx.resolve_device('gpu' if backend == 'mlx_gpu' else 'cpu'), mlx


def test_registered_amendment_preserves_thresholds_and_data(tmp_path, capsys):
    p = pilot.load_protocol(SPEC)
    old = pilot.load_protocol(SPEC.with_name('poc-v3.1-p2fix-protocol.json'))
    for key in ('codecs', 'synthetic', 'seeds', 'training'):
        assert p[key] == old[key]
    assert p['pilot']['P2A'] == old['pilot']['P2A']
    assert {k: v for k, v in p['pilot']['P2B'].items() if k != 'record_prefix_reveals'} == old['pilot']['P2B']
    assert p['model_profiles']['G3']['sampling_nfe_per_candidate'] == 101
    nfe = 3 * 101 + 2 * 33
    assert p['pilot']['P1']['sampling_nfe'] == nfe * 2 * 16 * 10
    assert p['execution']['profile_sampling_nfe_min'] == nfe * 4 * (1 + 4 + 16 + 64)
    assert p['pilot']['P3']['sampling_nfe_min'] == nfe * 3 * 2 * 512
    assert p['rehearsal']['sampling_nfe_min'] == nfe * 2 * 32 * 100
    assert p['main']['sampling_nfe_min'] == nfe * 6 * p['main']['evaluation_trials'] * p['main']['k']
    assert study_cli.main(['pilot', '--protocol', str(SPEC), '--workdir', str(tmp_path / 'unused'),
                           '--stage', 'P2', '--development', '--dry-run']) == 0
    assert 'G3' in capsys.readouterr().out and not (tmp_path / 'unused').exists()
    p['model_profiles']['D1']['prefix_balanced_loss'] = False
    pilot.atomic_json(tmp_path / 'modified.json', p)
    with pytest.raises(pilot.PilotError, match='Unsupported/modified'):
        pilot.load_protocol(tmp_path / 'modified.json')


@pytest.mark.parametrize('backend', ['torch', 'mlx_cpu', 'mlx_gpu'])
def test_length_context_loss_and_trace_do_not_change_candidates(backend):
    device, mlx = runtime(backend)
    p = small_protocol()
    for pipeline in ('P-G-BGV', 'P-G-CGGE', 'R-G-BGV'):
        encoder, decoder, shape = pilot.codecs(p['pipelines'][pipeline])
        model, diffusion = pilot.model_and_diffusion(p, pipeline, device, 8, profile_id='G3', backend='mlx' if mlx else 'torch')
        messages = [b'A' * n for n in range(4, 32)]
        clean_t = torch.stack([encoder.encode(x) for x in messages]) * 2 - 1
        clean = mlx.mx.array(clean_t.numpy()) if mlx else clean_t
        if mlx:
            diffusion.validate_clean(clean)
        lengths = diffusion.lengths(clean)
        fixed, payload = diffusion.structure(lengths)
        array = lambda x: np.array(x) if mlx else x.detach().numpy()
        assert array(lengths).tolist() == list(range(4, 32))
        np.testing.assert_array_equal(array(fixed)[~array(payload)], array(clean)[~array(payload)])
        # An oracle supplies only payload glyphs. The sampled length must own all
        # header/mask/padding, including at the shortest/longest valid lengths.
        for length in (4, 31):
            if mlx:
                class Oracle:
                    def eval(self): pass
                    def length_head(self, cond):
                        return mlx.mx.broadcast_to(mlx.mx.where(mlx.mx.arange(28) == length - 4, 0., -1000.), (len(cond), 28))
                    def __call__(self, x, t, cond):
                        assert cond.shape[1] == 13
                        context, active = diffusion.structure(mlx.mx.full((len(x),), length))
                        np.testing.assert_array_equal(array(x)[~array(active)], array(context)[~array(active)])
                        return mlx.mx.broadcast_to(clean[-1:], x.shape)
            else:
                class Oracle(torch.nn.Module):
                    def length_head(self, cond):
                        return torch.where(torch.arange(28) == length - 4, 0., -1000.).expand(len(cond), -1)
                    def forward(self, x, t, cond):
                        assert cond.shape[1] == 13
                        context, active = diffusion.structure(torch.full((len(x),), length))
                        assert torch.equal(x[~active], context[~active])
                        return clean[-1:].expand_as(x)
            values = pilot.sample(p, pipeline, Oracle(), diffusion, [0, 4095], [4, 5], device,
                                   profile_id='G3', length_seeds=[6, 7])
            assert all(decoder.decode(row, normalized=True).message == b'A' * length for row in values)
        corrupt = clean_t[:1].clone()
        corrupt[:, 1, -1, -1] = 1
        with pytest.raises(ValueError, match='invalid length/header/mask/padding'):
            diffusion.validate_clean(mlx.mx.array(corrupt.numpy())) if mlx else diffusion.lengths(corrupt)
        with pytest.raises(ValueError, match='separate length'):
            pilot.sample(p, pipeline, model, diffusion, [0], [4], device, profile_id='G3')
        # A non-finite structural output must not disappear behind fixed context.
        if mlx:
            model.output.bias = mlx.mx.array([0., float('nan')])
        else:
            with torch.no_grad(): model.output.bias[1] = float('nan')
        with pytest.raises(FloatingPointError, match='Gaussian prediction'):
            pilot.sample(p, pipeline, model, diffusion, [0], [4], device, profile_id='G3', length_seeds=[6])

    pipeline = 'P-DISC'
    model, diffusion = pilot.model_and_diffusion(p, pipeline, device, 8, profile_id='D1', backend='mlx' if mlx else 'torch')
    seen = []
    a = pilot.sample(p, pipeline, model, diffusion, [0, 4095], [4, 5], device,
                     profile_id='D1', length_seeds=[6, 7])
    b = pilot.sample(p, pipeline, model, diffusion, [0, 4095], [4, 5], device,
                     profile_id='D1', length_seeds=[6, 7], trace=lambda *args: seen.append(args))
    assert torch.equal(a, b) and len(seen) == 2


def test_d1_region_means_and_empty_masks():
    p = small_protocol()
    model, diffusion = pilot.model_and_diffusion(p, 'P-DISC', torch.device('cpu'), 8, profile_id='D1')
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
    encoder = pilot.codecs(p['pipelines']['P-DISC'])[0]
    clean = torch.stack([encoder.encode(b'000!'), encoder.encode(b'f' * 31)])
    for end, regions in ((0, 0), (3, 1), (32, 2)):
        mask = (torch.arange(32)[None] < end).expand_as(clean)
        losses, parts = diffusion.losses(model, clean, pilot.condition([0, 4095], torch.device('cpu')),
                                         torch.ones(2), mask, return_components=True)
        torch.testing.assert_close(losses, torch.full((2,), math.log(28) + regions * math.log(94)))
        torch.testing.assert_close(parts['payload_ce'], parts['prefix_ce'] + parts['suffix_ce'])


def test_g3_loss_only_counts_payload_regions_and_length():
    p = small_protocol()
    for pipeline in ('P-G-BGV', 'P-G-CGGE'):
        encoder = pilot.codecs(p['pipelines'][pipeline])[0]
        clean = torch.stack([encoder.encode(b'000!'), encoder.encode(b'f' * 31)]) * 2 - 1
        _, diffusion = pilot.model_and_diffusion(p, pipeline, torch.device('cpu'), 8, profile_id='G3')
        class UnitError(torch.nn.Module):
            def length_head(self, condition): return torch.zeros(len(condition), 28)
            def forward(self, value, time, condition): return clean + 1
        actual = diffusion.losses(UnitError(), clean, pilot.condition([0, 4095], torch.device('cpu')),
                                  torch.tensor([0, 9]), torch.zeros_like(clean))
        torch.testing.assert_close(actual, torch.full((2,), math.log(28) + 2))


@pytest.mark.parametrize('backend', ['torch', 'mlx_cpu', 'mlx_gpu'])
def test_new_profiles_p0_p1_recovery_and_gaussian_diagnostics(tmp_path, backend):
    device, mlx = runtime(backend)
    p = small_protocol()
    args = SimpleNamespace(workdir=tmp_path, stage='P0', device='gpu' if backend == 'mlx_gpu' else 'cpu',
                           backend='mlx' if mlx else 'torch', development=True, threads=1, resume=False)
    assert pilot.run_pilot(p, args)['status'] == 'PASS'
    args.stage = 'P1'
    result = pilot.run_pilot(p, args)
    assert result['status'] == 'PASS'
    for name, row in pilot.read_json(tmp_path / 'pilot/P1/summary.json')['runs'].items():
        if name.endswith('/main'):
            assert row['recovery']['status'] == 'PASS'
            ledger = pilot.logical_ledger(tmp_path / 'pilot/P1/runs' / name)
            assert all(4 <= x['sampled_length'] <= 31 and x['sampling_nfe'] == 3 for x in ledger)
    data = pilot.make_data(p)
    limits = pilot.Budget(tmp_path, 'P2', {'active_seconds': 0.}, p, device, backend=args.backend)
    for pipeline in ('P-G-BGV', 'P-G-CGGE', 'R-G-BGV'):
        model, diffusion = pilot.model_and_diffusion(p, pipeline, device, 2, profile_id='G3', backend=args.backend)
        d = study_p2.diagnostics(p, pipeline, 'G3', model, diffusion, data, device, limits, tmp_path / pipeline)
        assert d['teacher_forced_length'] and len(d['length_head']) == 8
        assert all(x['mask_mse'] == 0 and x['padding_glyph_mse'] == 0 for x in d['rows'])
    # Exercise A/B dispatch and final probe seals with tiny, explicitly failing
    # quality fixtures so this test never starts resource benchmarking.
    p['pilot']['P2A'].update(train_complement_pairs=2, train_messages=4, batch_size=4,
                            optimizer_updates_per_run=3, diagnostic_updates=[1],
                            trajectories_per_condition_variant=2, gate_denominator_per_variant=8,
                            normal_joint_min=9, flipped_joint_min=9)
    p['pilot']['P2B'].update(train_messages=16, validation_conditions=8, epochs=2,
                            optimizer_updates_per_run=16, validation_epochs=[1, 2], probe_epochs=[1, 2],
                            gate_epoch=2, probe_conditions=4, gate_denominator_per_variant=4,
                            normal_joint_min=5, flipped_joint_min=5)
    chosen = study_p2.cases(p, data)
    for pipeline in ('P-G-BGV', 'P-G-CGGE'):
        for suffix, stage in (('A', 'P2A'), ('B', 'P2B')):
            directory = tmp_path / 'pilot/P2' / pipeline / 'G3' / suffix
            saved = study_p2.phase(p, stage, pipeline, 'G3', data, chosen, directory, device, limits, {})
            assert saved['status'] == 'FAIL_QUALITY' and study_p2.completed(directory) == saved
            if suffix == 'B':
                assert saved['selected_epoch'] == 2
                assert saved['checkpoint_sha256'] == pilot.load_checkpoint(directory)[1]
