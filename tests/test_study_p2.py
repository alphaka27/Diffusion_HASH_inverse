"""Short real P2 fixtures; amended sizes/thresholds never confer qualification."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from diffusion_hash_inv import study_pilot as pilot, study_p2 as p2, study_cli

SPEC = Path(__file__).parents[1] / 'examples/poc-v3.1-protocol.json'


def protocol():
    p = deepcopy(pilot.load_protocol(SPEC))
    p['pipeline_order'] = ['P-DISC']
    p['profile_selection']['discrete_order'] = ['D1']
    p['synthetic'].update(train_conditions=8, validation_conditions=8, train_unique_messages=16)
    p['training'].update(batch_size=2, validation_draws_per_condition=1)
    p['pilot']['P1'].update(train_messages=8, validation_conditions=2, evaluation_trials=2)
    p['pilot']['P2A'].update(train_complement_pairs=2, train_messages=4, batch_size=4,
                            optimizer_updates_per_run=6, evaluation_conditions=4, trajectories_per_condition_variant=2,
                            gate_denominator_per_variant=8, normal_joint_min=0, flipped_joint_min=0, wrong_original_max=8)
    p['pilot']['P2B'].update(train_messages=16, validation_conditions=8, epochs=40, validation_epochs=[10, 30, 40],
                            probe_epochs=[10, 30, 40], gate_epoch=40, probe_conditions=4, gate_denominator_per_variant=4,
                            optimizer_updates_per_run=320, normal_joint_min=0, flipped_joint_min=0,
                            wrong_original_max=4, valid_min_each_variant=0)
    p['execution'].update(minimum_disk_free_gib=0, profile_batch_candidates=[1, 4])
    for name, spec in p['model_profiles'].items():
        spec.update(sampling_steps=2, sampling_nfe_per_candidate=3 if name == 'D1' else 2)
        if spec['family'] == 'gaussian':
            spec['diffusion_steps'] = 10
    return p


def budget(root, p, backend):
    root.mkdir(parents=True, exist_ok=True)
    if backend == 'mlx':
        mlx = pytest.importorskip('diffusion_hash_inv.mlx_backend')
        if not mlx.mx.metal.is_available():
            pytest.skip('native Metal required')
        device = mlx.resolve_device('gpu')
    else:
        device = torch.device('cpu')
    return device, pilot.Budget(root, 'P2', {'active_seconds': 0.}, p, device, backend=backend)


def test_case_selection_gate_denominators_and_cli(tmp_path, capsys):
    p = protocol()
    data = pilot.make_data(p)
    chosen = p2.cases(p, data)
    for source, rows in chosen.items():
        assert len(rows) == 4
        first = {}
        for row in data['sources'][source]['train']:
            first.setdefault(row[0], row)
        assert all(first[y] == [y, message] for y, message in rows)
        assert all(rows[i][0] ^ rows[i + 1][0] == 4095 for i in (0, 2))
        expected = p2.targets(p, data, source, 'P2A', chosen)
        assert len(expected) == len(set(expected)) == 8
    changed = deepcopy(data)
    changed['sources']['printable']['train'] = changed['sources']['printable']['train'][:1]
    with pytest.raises(pilot.PilotError, match='observed training complement'):
        p2.cases(p, changed)
    cfg = pilot.load_protocol(SPEC)['pilot']['P2A']
    metrics = {'variants': {'normal': {'candidates': 64, 'valid': 61, 'joint': 61},
                            'flipped': {'candidates': 64, 'valid': 61, 'joint': 61, 'wrong_original': 1}}}
    assert p2.quality_gate(cfg, metrics)
    for key, value in [('joint', 60), ('wrong_original', 2), ('candidates', 63)]:
        bad = deepcopy(metrics)
        bad['variants']['flipped'][key] = value
        assert not p2.quality_gate(cfg, bad)
    assert study_cli.main(['pilot', '--protocol', str(SPEC), '--workdir', str(tmp_path / 'unused'),
                           '--stage', 'P2', '--backend', 'mlx', '--development', '--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['executable']
    assert not (tmp_path / 'unused').exists()


def test_10k_amendment_preserves_data_rng_and_final_gate(tmp_path, capsys):
    original = pilot.load_protocol(SPEC)
    amended_path = SPEC.with_name('poc-v3.1-p2a10k-protocol.json')
    amended = pilot.load_protocol(amended_path)
    assert original['pilot']['P2A']['optimizer_updates_per_run'] == 2000
    assert amended['pilot']['P2A']['optimizer_updates_per_run'] == 10000
    assert amended['pilot']['P2A']['diagnostic_updates'] == [2000, 5000, 10000]
    assert original['protocol_id'] != amended['protocol_id']
    for key in ('normal_joint_min', 'flipped_joint_min', 'wrong_original_max', 'selection_checkpoint'):
        assert amended['pilot']['P2A'][key] == original['pilot']['P2A'][key]
    for stage, namespace in [('SYN_DATA', 'split'), ('P2A', 'initialization'), ('P2A', 'train-noise'),
                             ('P2A', 'generation-length'), ('P2A', 'generation-payload')]:
        assert pilot.seed(original, stage, namespace, profile_id='D1') == pilot.seed(amended, stage, namespace, profile_id='D1')
    small = protocol()
    comparison = deepcopy(small)
    comparison['protocol_id'] = amended['protocol_id']
    comparison['seeds']['protocol_namespace'] = amended['seeds']['protocol_namespace']
    assert pilot.make_data(small) == pilot.make_data(comparison)
    assert study_cli.main(['pilot', '--protocol', str(amended_path), '--workdir', str(tmp_path / 'unused'),
                           '--stage', 'P2', '--backend', 'mlx', '--development', '--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['settings']['P2A']['optimizer_updates_per_run'] == 10000
    assert not (tmp_path / 'unused').exists()
    amended['pilot']['P2A']['normal_joint_min'] = 0
    pilot.atomic_json(tmp_path / 'modified.json', amended)
    with pytest.raises(pilot.PilotError, match='Unsupported/modified'):
        pilot.load_protocol(tmp_path / 'modified.json')


def test_p2fix_protocol_and_region_loss(tmp_path):
    p = pilot.load_protocol(SPEC.with_name('poc-v3.1-p2fix-protocol.json'))
    old = pilot.load_protocol(SPEC.with_name('poc-v3.1-p2a10k-protocol.json'))
    assert p['profile_selection']['gaussian_order'] == ['G2']
    assert p['profile_selection']['discrete_order'] == ['D1']
    assert p['pilot']['P2B']['selection_checkpoint'] == 'final_epoch'
    assert p['codecs'] == old['codecs'] and p['synthetic'] == old['synthetic']
    for stage in ('P2A', 'P2B'):
        for key in ('normal_joint_min', 'flipped_joint_min', 'wrong_original_max', 'optimizer_updates_per_run'):
            assert p['pilot'][stage][key] == old['pilot'][stage][key]
    from diffusion_hash_inv.study_profiles import RegionGaussianDiffusion
    for pipeline in ('P-G-BGV', 'P-G-CGGE'):
        encoder = pilot.codecs(p['pipelines'][pipeline])[0]
        clean = torch.stack([encoder.encode(b'000!'), encoder.encode(b'f' * 31)]) * 2 - 1
        model, diffusion = pilot.model_and_diffusion(p, pipeline, torch.device('cpu'), 0, profile_id='G2')
        assert model.condition_output is not None
        class Fixed(torch.nn.Module):
            def forward(self, value, time, condition):
                return clean + 1
        loss = diffusion.losses(Fixed(), clean, pilot.condition([0, 4095], torch.device('cpu')),
                                torch.tensor([0, 999]), torch.zeros_like(clean))
        # Unit squared error in every nonempty region, independent of its area.
        expected = [5., 4.] if pipeline == 'P-G-BGV' else [4., 4.]
        assert loss.tolist() == expected
        with pytest.raises(ValueError, match='region loss'):
            RegionGaussianDiffusion(10, device=torch.device('cpu'), prediction_type='epsilon', loss_regions='bgv')
    p['model_profiles']['D1']['condition_output'] = False
    pilot.atomic_json(tmp_path / 'modified.json', p)
    with pytest.raises(pilot.PilotError, match='Unsupported/modified'):
        pilot.load_protocol(tmp_path / 'modified.json')


@pytest.mark.parametrize('backend', ['torch', 'mlx'])
@pytest.mark.parametrize('final_epoch', [False, True])
def test_real_phases_probe_resume_resources_and_seals(tmp_path, monkeypatch, backend, final_epoch):
    torch.set_num_threads(1)
    p = protocol()
    if final_epoch:
        p['model_profiles']['D1']['condition_output'] = True
        p['model_profiles']['D1']['prefix_balanced_loss'] = True
        p['pilot']['P2B']['selection_checkpoint'] = 'final_epoch'
        p['pilot']['P2B']['record_prefix_reveals'] = True
    p['pilot']['P2A']['diagnostic_updates'] = [3, 5, 6]
    data = pilot.make_data(p)
    chosen = p2.cases(p, data)
    device, limits = budget(tmp_path, p, backend)
    if backend == 'mlx':
        def forbidden(*args, **kwargs):
            raise AssertionError('P2 MLX must not call PyTorch model/optimizer operations')
        monkeypatch.setattr(torch.nn.Module, '_call_impl', forbidden)
        monkeypatch.setattr(torch.optim, 'Adam', forbidden)
    a = tmp_path / 'pilot/P2/P-DISC/D1/A'
    original_train = pilot.train_model

    def interrupted_train(*args, **kwargs):
        return original_train(*args, **kwargs, pause_update=5)

    monkeypatch.setattr(pilot, 'train_model', interrupted_train)
    original_diagnostics = p2.diagnostics

    def interrupted_diagnostics(*args, **kwargs):
        if args[8].name == 'update-00000003':
            raise KeyboardInterrupt()
        return original_diagnostics(*args, **kwargs)

    monkeypatch.setattr(p2, 'diagnostics', interrupted_diagnostics)
    with pytest.raises(KeyboardInterrupt):
        p2.phase(p, 'P2A', 'P-DISC', 'D1', data, chosen, a, device, limits, {})
    assert pilot.load_checkpoint(a)[0]['update'] == 3
    assert (a / 'probes/update-00000003/candidates.sqlite').exists()
    assert not (a / 'probes/update-00000003/complete.json').exists()
    monkeypatch.setattr(p2, 'diagnostics', original_diagnostics)
    with pytest.raises(pilot.RecoveryPause):
        p2.phase(p, 'P2A', 'P-DISC', 'D1', data, chosen, a, device, limits, {})
    probe_hash = pilot.file_hash(a / 'probes/update-00000003/complete.json')
    monkeypatch.setattr(pilot, 'train_model', original_train)
    result = p2.phase(p, 'P2A', 'P-DISC', 'D1', data, chosen, a, device, limits, {})
    assert result['passed'] and result['updates'] == 6 and result['best_epoch'] is None
    assert not (a / 'checkpoints/BEST.json').exists()
    assert pilot.file_hash(a / 'probes/update-00000003/complete.json') == probe_hash
    for update in (3, 5):
        saved = p2.completed(a / 'probes' / f'update-{update:08d}')
        assert saved['scope'] == 'diagnostic_only_not_selection' and 'passed' not in saved
        assert saved['update'] == update
    assert not (a / 'probes/update-00000006').exists()
    assert result['checkpoint_sha256'] == pilot.load_checkpoint(a)[1]
    diagnostic = pilot.read_json(a / 'diagnostics.json')
    assert diagnostic['split'] == 'training' and diagnostic['records_sha256'] == pilot.digest(chosen['printable'])
    assert len(diagnostic['length_head']) == 4
    assert len(diagnostic['generated_length_and_prefix']) == 16
    assert all(math.isclose(sum(row['probabilities_lengths_4_to_31']), 1., abs_tol=1e-6) for row in diagnostic['length_head'])
    assert all(row['length_ce'] >= 0 and row['payload_ce'] >= 0 for row in diagnostic['rows'])
    updates = [json.loads(line) for line in (a / 'telemetry.jsonl').read_text().splitlines() if json.loads(line)['kind'] == 'update']
    assert all(math.isclose(row['loss'], row['length_ce'] + row['payload_ce'], rel_tol=1e-6) for row in updates)
    reference = tmp_path / 'reference-A'
    _, _, state, _ = original_train(p, 'P2A', 'P-DISC', 'main', 99, data, reference, device, limits,
                                     profile_id='D1', training_rows=chosen['printable'])
    recovered, _ = pilot.load_checkpoint(a)
    assert pilot.equal_tensors(recovered['model'], state['model'])
    assert pilot.equal_tensors(recovered['optimizer'], state['optimizer'])
    rows = pilot.logical_ledger(a)
    assert len(rows) == 16
    lines, progress = [], {}
    p2.report_rows(tmp_path, p, lines, progress)
    assert progress['P2/P-DISC/D1/A/update-00000003'] == 'COMPLETE'
    assert 'diagnostic only' in '\n'.join(lines)
    pairs = {(r['unit_id'], r['variant']): r for r in rows}
    for unit, _ in p2.targets(p, data, 'printable', 'P2A', chosen):
        normal, flipped = pairs[unit, 'normal'], pairs[unit, 'flipped']
        assert normal['rng_identity'] == flipped['rng_identity']
        assert normal['length_rng_identity'] == flipped['length_rng_identity']
        assert normal['requested_target'] ^ flipped['requested_target'] == 4095
    b = tmp_path / 'pilot/P2/P-DISC/D1/B'
    if final_epoch:
        # Fixed increasing validation values force BEST to stay at epoch10.
        # Checkpoint/resume must still evaluate the latest completed epoch.
        values = iter((1., 2., 3.))
        monkeypatch.setattr(pilot, 'validation_loss', lambda *args: next(values))
    original_probe = p2.probe

    def interrupted_probe(*args):
        original_probe(*args)
        if args[6] == 10:
            raise KeyboardInterrupt()

    monkeypatch.setattr(p2, 'probe', interrupted_probe)
    with pytest.raises(KeyboardInterrupt):
        p2.phase(p, 'P2B', 'P-DISC', 'D1', data, chosen, b, device, limits, {})
    sealed_probe = pilot.file_hash(b / 'probes/epoch-0010/complete.json')
    monkeypatch.setattr(p2, 'probe', original_probe)
    result = p2.phase(p, 'P2B', 'P-DISC', 'D1', data, chosen, b, device, limits, {})
    assert result['updates'] == 320 and result['passed']
    if final_epoch:
        assert result['best_epoch'] == 10 and result['selected_epoch'] == 40
        assert result['checkpoint_sha256'] == pilot.load_checkpoint(b)[1]
        assert result['checkpoint_sha256'] != pilot.load_checkpoint(b, best=True)[1]
    assert pilot.file_hash(b / 'probes/epoch-0010/complete.json') == sealed_probe
    assert result['resources']['update_seconds'] == max(result['resources']['update_seconds_windows'])
    assert result['recovery']['status'] == 'PASS'
    assert not result['resource_estimate']['final_sealed']
    for epoch in (10, 30, 40):
        saved = p2.completed(b / 'probes' / f'epoch-{epoch:04d}')
        assert saved['best_epoch'] <= epoch
        if final_epoch:
            assert saved['selected_epoch'] == epoch and saved['checkpoint_selection'] == 'final_epoch'
        assert saved['metrics']['candidates'] == 8
        diagnostics = pilot.read_json(b / 'probes' / f'epoch-{epoch:04d}' / 'diagnostics.json')
        assert len(diagnostics['rows']) == 4
        if final_epoch:
            for row in pilot.logical_ledger(b / 'probes' / f'epoch-{epoch:04d}'):
                trace = row['prefix_reveals']
                assert sorted(r['position'] for r in trace) == [0, 1, 2]
                for r in trace:
                    assert bytes.fromhex(row['candidate_hex'])[r['position']] == r['sampled_byte']
                    assert 1 <= r['step'] <= p['model_profiles']['D1']['sampling_steps']
                    assert 0 <= r['correct_probability'] <= 1
    assert p2.completed(b) == result
    execution = p2.execute(p, tmp_path, data, device, limits)
    assert execution['status'] == 'PASS'
    assert pilot.read_json(tmp_path / 'profile.frozen.json')['complete']
    assert pilot.read_json(tmp_path / 'profile.frozen.json')['pipelines']['P-DISC']['profile_id'] == 'D1'
    pilot.atomic_json(b / 'training.json', {})
    with pytest.raises(pilot.PilotError, match='modified'):
        p2.completed(b)


@pytest.mark.parametrize('backend', ['torch', 'mlx'])
def test_diagnostics_use_native_backend_and_do_not_create_candidates(tmp_path, backend):
    p = protocol()
    p['profile_selection']['discrete_order'] = ['D0', 'D1']
    data = pilot.make_data(p)
    device, limits = budget(tmp_path, p, backend)
    torch.set_num_threads(1)
    for pipeline, profile in [('P-G-BGV', 'G2'), ('R-G-BGV', 'G0'), ('P-G-CGGE', 'G1'), ('P-DISC', 'D0')]:
        model, diffusion = pilot.model_and_diffusion(p, pipeline, device, 12, profile_id=profile, backend=backend)
        output = tmp_path / pipeline / profile
        result = p2.diagnostics(p, pipeline, profile, model, diffusion, data, device, limits, output)
        assert result['scope'] == 'diagnostic_only_not_candidate_success'
        assert not (output / 'candidates.sqlite').exists()
        if profile.startswith('G'):
            assert [r['timestep'] for r in result['rows']] == [0, 9]
            assert all(0 <= r['clipping_fraction'] <= 1 for r in result['rows'])
            assert all(r['payload_glyph_mse'] >= 0 for r in result['rows'])
            assert all((r['length_header_mse'] is None) == (pipeline == 'P-G-CGGE') for r in result['rows'])
        else:
            assert [r['mask_fraction'] for r in result['rows']] == [.1, .5, .9, 1.]
            assert all(len(r['eos_position_mean_probability']) == 32 for r in result['rows'])


def test_sequential_selection_failure_continuation_and_partial_report(tmp_path, monkeypatch):
    p = protocol()
    p['profile_selection']['discrete_order'] = ['D0', 'D1']
    p['pipeline_order'] = ['P-DISC', 'R-DISC']
    data = pilot.make_data(p)
    device, limits = budget(tmp_path, p, 'torch')
    calls = []

    def fake_phase(p, stage, pipeline, profile_id, data, chosen, directory, device, budget, profiles):
        calls.append((pipeline, profile_id, stage))
        if pipeline == 'P-DISC' and profile_id == 'D0':
            raise FloatingPointError('fixture numerical failure')
        passed = pipeline == 'P-DISC'
        return {'passed': passed, 'status': 'PASS' if passed else 'FAIL_QUALITY',
                'resources': {'profile_id': profile_id, 'selected_batch': 4}}

    monkeypatch.setattr(p2, 'phase', fake_phase)
    monkeypatch.setattr(p2, 'estimates', lambda *args: {'candidate_resources_pass': True, 'final_sealed': False})
    result = p2.execute(p, tmp_path, data, device, limits)
    assert result['status'] == 'BLOCKED_DEVELOPMENT' and result['exit_code'] == 2
    assert ('P-DISC', 'D0', 'P2B') not in calls
    assert ('P-DISC', 'D1', 'P2B') in calls
    assert all(stage == 'P2A' for pipeline, profile_id, stage in calls if pipeline == 'R-DISC')
    assert result['selected_profiles']['P-DISC']['profile_id'] == 'D1'
    assert not pilot.read_json(tmp_path / 'profile.frozen.json')['complete']
    lines, progress = [], {}
    p2.report_rows(tmp_path, p, lines, progress)
    assert progress['P2/P-DISC/D0/A'] == 'FAILED_RUNTIME'
    assert progress['P2/R-DISC/D1/B'] == 'NOT_RUN_P2A_FAILED'


def test_stage_dispatch_partial_result_and_external_seals(tmp_path):
    """Prerequisite seals are fixtures; exercise real P2A through run_pilot."""
    p = protocol()
    p['pilot']['P2A'].update(normal_joint_min=8, flipped_joint_min=8, wrong_original_max=0)
    device = torch.device('cpu')
    data = pilot.make_data(p)
    pilot.atomic_json(tmp_path / 'protocol.frozen.json', p)
    pilot.atomic_json(tmp_path / 'data/synthetic.json', data)
    pilot.atomic_json(tmp_path / 'manifest.json', {
        'identity': {'protocol_sha256': pilot.digest(p), 'environment': pilot.environment(device, 1), 'development': True},
        'commands': [], 'data_sha256': pilot.file_hash(tmp_path / 'data/synthetic.json')})
    pilot.atomic_json(tmp_path / 'gates.json', {s: {'status': 'PASS', 'sha256': {}} for s in ('P0', 'P1')})
    args = SimpleNamespace(workdir=tmp_path, stage='P2', device='cpu', backend='torch', development=True, threads=1, resume=False)
    result = pilot.run_pilot(p, args)
    assert result['status'] == 'BLOCKED_DEVELOPMENT' and result['exit_code'] == 2
    assert pilot.read_json(tmp_path / 'pilot/P2/state.json')['status'] == 'COMPLETE'
    summary = pilot.read_json(tmp_path / 'pilot/P2/P-DISC/D1/A/summary.json')
    assert 'CONDITION_RESPONSE_WEAK' in summary['warnings']
    assert (tmp_path / 'pilot/P2/P-DISC/D1/A/diagnostics.json').exists()
    report = pilot.write_report(tmp_path, p)
    assert report['run_progress']['P2/P-DISC/D1/A'] == 'FAIL_QUALITY'
    assert report['run_progress']['P2/P-DISC/D1/B'] == 'NOT_RUN_P2A_FAILED'
    assert not report['main_ready']
    pilot.atomic_json(tmp_path / 'profile.frozen.json', {})
    with pytest.raises(pilot.PilotError, match='selection checksum'):
        pilot.write_report(tmp_path, p)
