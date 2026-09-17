"""Atomic recovery records for the sequential G2–G6 experiment."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(stage, completed, command, *, gate='G2', artifacts=(), **updates):
    state = json.loads(Path('EXPERIMENT_STATE.json').read_text())
    now = datetime.datetime.now().astimezone().isoformat()
    state.update(current_gate=gate, current_stage=stage, updated_at=now, **updates)
    if completed and completed not in state['completed_actions']:
        state['completed_actions'].append(completed)
    state['latest_artifacts'] = list(artifacts)
    state['resume_instruction'] = command
    write_json('EXPERIMENT_STATE.json', state)
    Path('NEXT_ACTION.md').write_text(
        f'Current gate: {gate}\nCurrent stage: {stage}\nStatus: {state["status"]}\n\n'
        f'Next exact action:\n```bash\n{command}\n```\n\n'
        'Prerequisites: read EXPERIMENT_STATE.json and verify artifacts. Completed run units are validated and skipped. '
        'Do not overwrite G0/G1 or proceed past a non-PASS gate.\n')
    with Path('EXPERIMENT_PROGRESS.md').open('a') as stream:
        stream.write(f'\n## {now}\n\nGate: {gate}\nStage: {stage}\n\nCompleted:\n- {completed or "Stage started"}\n\n'
                     f'Results:\n- Status: {state["status"]}\n\nFiles changed:\n- State/progress/next-action documents and listed artifacts.\n\n'
                     f'Tests:\n- {state["latest_test_result"]}\n\nArtifacts:\n- ' + '\n- '.join(map(str, artifacts)) +
                     f'\n\nDecision: {state["status"]}\n\nNext exact action:\n- {command}\n')
    write_json(Path('output/session_checkpoints') / (now.replace(':', '-') + '.json'), state)


def verify_prerequisites():
    manifest = json.loads(Path('output/session_checkpoints/prerequisite_sha256.json').read_text())
    for name, digest in manifest.items():
        if not Path(name).is_file() or sha256(name) != digest:
            raise RuntimeError(f'Preserved prerequisite changed or missing: {name}')


def require_previous(gate):
    state = json.loads(Path('EXPERIMENT_STATE.json').read_text())
    for index in range(int(gate[1:])):
        if state['gates'][f'G{index}'] != 'PASS':
            raise RuntimeError(f'{gate} blocked by G{index}={state["gates"][f"G{index}"]}')


def record_gate(gate, status, reason, next_action):
    state = json.loads(Path('EXPERIMENT_STATE.json').read_text())
    state['gates'][gate] = status
    write_json('EXPERIMENT_STATE.json', state)
    root = json.loads(Path('output/gate_summary.json').read_text())
    for row in root['gates']:
        if row['gate'] == gate:
            row.update(status=status, artifact=f'output/{gate.lower()}/gate_summary.json')
    write_json('output/gate_summary.json', root)
    write_json(f'output/{gate.lower()}/gate_summary.json',dict(gate=gate,status=status,reason=reason))
    checkpoint('DECIDE/CHECKPOINT', f'{gate} {status}: {reason}', next_action, gate=gate,
               status='READY' if status=='PASS' else 'STOPPED',
               failure_boundary=None if status=='PASS' else f'{gate}: {reason}',
               artifacts=[f'output/{gate.lower()}/gate_summary.json',f'output/{gate.lower()}/report.md'],
               pending_actions=[next_action])


from contextlib import contextmanager


@contextmanager
def experiment_lock():
    """Keep resumed sessions from writing the same run/state concurrently."""
    import fcntl
    import os
    path = Path('output/session_checkpoints/experiment.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another experiment process is active; read experiment.lock and its logs.') from None
        stream.seek(0); stream.truncate()
        stream.write(str(os.getpid())+'\n'); stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def verify_persisted_state():
    verify_prerequisites()
    state=json.loads(Path('EXPERIMENT_STATE.json').read_text())
    root=json.loads(Path('output/gate_summary.json').read_text())
    for row in root['gates']:
        name=row['gate']
        if name in state['gates']:
            assert row['status'].replace(' ','_')==state['gates'][name], f'Gate state mismatch: {name}'
    checked=0
    for folder in ('g2','g3','g4','g5'):
        for marker in Path('output',folder).rglob('complete.json'):
            for name,digest in json.loads(marker.read_text())['sha256'].items():
                assert sha256(marker.parent/name)==digest, f'Run artifact mismatch: {marker.parent/name}'
                checked+=1
    final=Path('output/session_checkpoints/final_artifact_sha256.json')
    if final.exists():
        for name,digest in json.loads(final.read_text()).items():
            assert sha256(name)==digest, f'Final artifact mismatch: {name}'
            checked+=1
    for gate,status in state['gates'].items():
        if int(gate[1:])>=2 and status!='NOT_RUN':
            summary=json.loads(Path(f'output/{gate.lower()}/gate_summary.json').read_text())
            assert summary['status']==status
    if state['gates']['G5']!='PASS':
        assert state['gates']['G6']=='NOT_RUN'
    return dict(status='PASS',artifact_hash_checks=checked,prerequisite_files=len(json.loads(Path('output/session_checkpoints/prerequisite_sha256.json').read_text())),gates=state['gates'],experiment_status=state['status'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Read-only verification of persistent experiment state and artifacts.')
    parser.add_argument('--verify',action='store_true',required=True)
    parser.parse_args()
    print(json.dumps(verify_persisted_state(),sort_keys=True))
