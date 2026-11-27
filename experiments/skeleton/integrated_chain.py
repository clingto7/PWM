"""Run or check the existing-image observation → scene → live proxy chain.

Run: python integrated_chain.py run IMAGE OUTPUT --depth-source DIR --segmentation-source DIR --mesh GLB
Check: python integrated_chain.py check OUTPUT
Replay downstream: python integrated_chain.py replay OUTPUT NEW_OUTPUT
The run command accepts --generate-mesh instead of --mesh when local weights exist.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from stage_contracts import check as check_stage, digest
from check import check as check_saved_run


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def inside(root, name):
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'missing or outside chain artifact: {name}')
    return path


def check_live_artifacts(root):
    session = json.loads((root/'session.json').read_text())
    if session.get('inputs_unchanged') is not True:
        raise ValueError('live session changed upstream artifacts')
    steps = [json.loads(line) for line in (root/'steps.jsonl').read_text().splitlines()]
    trajectory = json.loads((root/'trajectory.json').read_text())
    contacts = json.loads((root/'contacts.json').read_text())
    timings = json.loads((root/'timings.json').read_text())
    if not steps or len(trajectory) != len(steps)+1 or len(contacts) != len(steps) or len(timings['steps']) != len(steps):
        raise ValueError('live step, trajectory, contact, or timing count mismatch')
    for index, (step, contact, timing) in enumerate(zip(steps, contacts, timings['steps'])):
        if step['index'] != index or step['state'] != trajectory[index+1] or step['contacts'] != contact:
            raise ValueError(f'live step {index} does not match solved state')
        for name in ('depth', 'instance_id'):
            inside(root, step['frame_paths'][name])
        if any(timing[key] < 0 for key in ('physics_s', 'raster_s', 'writing_s', 'response_latency_s')):
            raise ValueError(f'negative live latency at step {index}')
        if timing['response_latency_s'] + 1e-6 < timing['physics_s'] + timing['raster_s']:
            raise ValueError(f'live latency excludes solving or raster at step {index}')
    inside(root, 'proxy.mp4')
    return {'steps': len(steps), 'initialization_s': timings['initialization_s'],
            'latency_s': [item['response_latency_s'] for item in timings['steps']]}


def check_chain(root):
    root = root.resolve()
    chain = json.loads((root/'chain.json').read_text())
    if chain['schema'] != 1 or chain['status'] != 'complete':
        raise ValueError('chain is incomplete')
    if digest(inside(root, 'observation/inputs/input.jpg')) != chain['input_sha256']:
        raise ValueError('original image hash mismatch')
    observation, scene, live = (root/name for name in ('observation', 'scene', 'live'))
    for stage in ('depth', 'segmentation', 'observation', 'downstream_bundle'):
        check_stage(observation, stage)
    selection = json.loads((observation/'selection.json').read_text())
    selected = selection['selected_index']
    if selected is None or not any(c['index'] == selected and c.get('contact', {}).get('supported')
                                   for c in selection['candidates']):
        raise ValueError('missing supported subject selection evidence')
    for stage in ('observation', 'subject_mesh', 'scene', 'actions', 'physics', 'raster'):
        check_stage(scene, stage)
    run = json.loads((scene/'run.json').read_text())
    fit_text = (scene/'fit_diagnostics.json').read_text().lower()
    if any(term in fit_text for term in ('mini', 'category_size', 'class_size')) or \
       any(term in json.dumps(run.get('parameters', {})).lower() for term in ('mini', 'category_size', 'class_size')):
        raise ValueError('category size prior in scene parameters')
    saved_result = check_saved_run(scene)
    if not saved_result['passed']:
        raise ValueError('saved-run semantic gates failed: '+', '.join(
            name for name, passed in saved_result['checks'].items() if not passed))
    session = json.loads((live/'session.json').read_text())
    for name, expected in session['scene_and_observation_sha256'].items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'upstream changed since live session: {name}')
    result = check_live_artifacts(live)
    actual_files = {str(path.relative_to(root)) for directory in (observation, scene, live)
                    for path in directory.rglob('*') if path.is_file()}
    if actual_files != set(chain['artifact_sha256']):
        raise ValueError('chain output file ledger differs')
    for relative, expected in chain['artifact_sha256'].items():
        if digest(inside(root, relative)) != expected:
            raise ValueError(f'chain output hash mismatch: {relative}')
    return {'status': 'complete', 'selected_index': selected, 'rejected_indices':
            [c['index'] for c in selection['candidates'] if c['index'] != selected],
            'live': result, 'replay': 'python integrated_chain.py replay CHAIN NEW_OUTPUT'}


def execute(command):
    start = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter()-start


def ledger(root):
    return {str(path.relative_to(root)): digest(path) for directory in ('observation', 'scene', 'live')
            for path in sorted((root/directory).rglob('*')) if path.is_file()}


def replay_downstream(source, output):
    check_chain(source)
    source = source.resolve(); output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('replay output directory is not empty')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); stage = 'copy observation'
    times = {}
    script = Path(__file__).parent
    try:
        shutil.copytree(source/'observation', output/'observation')
        stage = 'scene replay'
        seconds = json.loads((source/'scene/run.json').read_text())['parameters']['seconds']
        times[stage] = execute([sys.executable, str(script/'first_image_pipeline.py'),
            str(output/'scene'), '--bundle', str(output/'observation'),
            '--resume-scene', str(source/'scene'), '--seconds', str(seconds)])
        stage = 'live replay'
        controls = [json.loads(line)['control_delta_velocity_m_s'] for line in
                    (source/'live/steps.jsonl').read_text().splitlines()]
        write(output/'controls.json', controls)
        times[stage] = execute([sys.executable, str(script/'live_session.py'),
            str(output/'scene'), str(output/'live'), '--controls', str(output/'controls.json'), '--video'])
        stage = 'check'
        write(output/'chain.json', {'schema': 1, 'status': 'complete',
            'implementation': 'integrated_chain.py replay', 'replayed_from': str(source),
            'input_sha256': digest(output/'observation/inputs/input.jpg'),
            'artifact_sha256': ledger(output),
            'timings_s': {**times, 'total': time.perf_counter()-started}})
        check_chain(output)
    except Exception as exc:
        write(output/'chain.json', {'schema': 1, 'status': 'failed', 'failure_stage': stage,
            'reason': str(exc), 'error_type': type(exc).__name__,
            'timings_s': {**times, 'elapsed': time.perf_counter()-started}})
        raise


def run(args):
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('output directory is not empty')
    root.mkdir(parents=True, exist_ok=True)
    stage = 'observation'
    started = time.perf_counter()
    times = {}
    script = Path(__file__).parent
    try:
        command = [sys.executable, str(script/'observation.py'), str(args.image), str(root/'observation')]
        for flag, value in (('--depth-source', args.depth_source), ('--segmentation-source', args.segmentation_source), ('--mesh', args.mesh)):
            if value:
                command.extend([flag, str(value)])
        if args.generate_mesh:
            command.append('--generate-mesh')
        times[stage] = execute(command)
        stage = 'scene'
        times[stage] = execute([sys.executable, str(script/'first_image_pipeline.py'),
                               str(root/'scene'), '--bundle', str(root/'observation'), '--seconds', str(args.seconds)])
        stage = 'live'
        controls = root/'controls.json'
        write(controls, args.controls)
        times[stage] = execute([sys.executable, str(script/'live_session.py'), str(root/'scene'),
                               str(root/'live'), '--controls', str(controls), '--video'])
        stage = 'check'
        chain = {'schema': 1, 'status': 'complete', 'implementation': 'integrated_chain.py',
                 'input_sha256': digest(args.image), 'artifact_sha256': ledger(root), 'timings_s':
                 {**times, 'total': time.perf_counter()-started},
                 'baseline': {'source': 'spec14-live-original-8963c94', 'total_s': 477.87195,
                              'comparison': 'compare only matching cold observation/mesh/scene/physics/raster definitions; live is additional and saved-source copies are not inference'}}
        write(root/'chain.json', chain)
        check_chain(root)
    except Exception as exc:
        status_path = root/stage/'stage_status.json'
        detail = json.loads(status_path.read_text()) if status_path.is_file() else {}
        write(root/'chain.json', {'schema': 1, 'status': 'failed', 'failure_stage': stage,
              'reason': detail.get('reason', str(exc)), 'error_type': detail.get('error_type', type(exc).__name__),
              'timings_s': {**times, 'elapsed': time.perf_counter()-started}})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('run')
    build.add_argument('image', type=Path)
    build.add_argument('output', type=Path)
    build.add_argument('--depth-source', type=Path)
    build.add_argument('--segmentation-source', type=Path)
    build.add_argument('--mesh', type=Path)
    build.add_argument('--generate-mesh', action='store_true')
    build.add_argument('--seconds', type=float, default=5)
    build.add_argument('--controls', type=json.loads, default=[[1, 0, 0], [0, 0, 0]])
    verify = sub.add_parser('check'); verify.add_argument('output', type=Path)
    replay = sub.add_parser('replay'); replay.add_argument('source', type=Path); replay.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        if bool(args.mesh) == bool(args.generate_mesh):
            parser.error('choose exactly one of --mesh or --generate-mesh')
        run(args)
    elif args.command == 'check':
        print(json.dumps(check_chain(args.output), indent=2))
    else:
        replay_downstream(args.source, args.output)


if __name__ == '__main__':
    main()
