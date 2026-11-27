"""Original-image chain: run IMAGE CONFIG OUTPUT; check OUTPUT; replay OUTPUT NEW_OUTPUT.

Config schema 1 selects saved model sources or installed local weights and a proxy
renderer. No model download or per-image script is performed.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from stage_contracts import digest, check as check_stage

STAGES = ('observation', 'mesh', 'scene_physics', 'proxy_render')
SOURCE_KEYS = ('depth_source', 'segmentation_source', 'saved_mesh', 'da3_weights', 'sam_weights', 'mesh_weights')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def config_read(path):
    data = json.loads(path.read_text())
    allowed = {'schema', 'renderer', 'seconds', 'mesh_provider', 'use_weak_prior', *SOURCE_KEYS}
    if not isinstance(data, dict) or set(data) - allowed or data.get('schema') != 1:
        raise ValueError('unsupported configuration schema or keys')
    if data.get('renderer', 'raster') not in ('raster', 'gaussian'):
        raise ValueError('renderer must be raster or gaussian')
    seconds = data.get('seconds', 5)
    if not isinstance(seconds, (int, float)) or not 0 < seconds <= 60:
        raise ValueError('seconds must be in (0, 60]')
    if data.get('mesh_provider', 'hunyuan3d21') != 'hunyuan3d21':
        raise ValueError('unsupported mesh provider')
    for key in SOURCE_KEYS:
        if key in data and not isinstance(data[key], str):
            raise ValueError(f'{key} must be a path string')
    if not isinstance(data.get('use_weak_prior', False), bool):
        raise ValueError('use_weak_prior must be boolean')
    return data


def invoke(command):
    subprocess.run([sys.executable, *map(str, command)], check=True)


def run(image, config, output, resume_observation=None, resume_scene=None):
    started = time.perf_counter()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory is not empty')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 1, 'status': 'running', 'stage': 'config', 'stages': {},
                'implementations': {}, 'timings_s': {}, 'inputs': {}, 'outputs': {}}
    try:
        data = config_read(config) if isinstance(config, Path) else config
        manifest['config'] = data
        manifest['inputs']['config'] = digest(config) if isinstance(config, Path) else None
        manifest['stage'] = 'input'
        if not image.is_file():
            raise FileNotFoundError(str(image))
        manifest['inputs']['image'] = digest(image)
        manifest['stage'] = 'observation'
        tick = time.perf_counter()
        if resume_scene:
            source = resume_scene.resolve()
            if not resume_observation:
                raise ValueError('scene replay needs saved observation')
            for name in ('depth', 'segmentation', 'observation', 'downstream_bundle'):
                check_stage(resume_observation, name)
            if digest(image) != digest(resume_observation/'inputs/input.jpg'):
                raise ValueError('replay image differs from saved observation')
            destination = output/'automatic'
            destination.mkdir()
            shutil.copytree(resume_observation, destination/'observation')
            command = [Path(__file__).with_name('first_image_pipeline.py'), destination/'run',
                       '--bundle', destination/'observation', '--resume-scene', source,
                       '--seconds', data.get('seconds', 5)]
            if data.get('use_weak_prior'):
                command.append('--use-weak-prior')
        else:
            command = [Path(__file__).with_name('original_image_run.py'), image, output/'automatic',
                       '--seconds', data.get('seconds', 5), '--mesh-provider', data.get('mesh_provider', 'hunyuan3d21')]
            if data.get('use_weak_prior'):
                command.append('--use-weak-prior')
            for key in SOURCE_KEYS:
                if key in data:
                    command.extend(['--' + key.replace('_', '-'), data[key]])
            if resume_observation:
                command.extend(['--resume-observation', resume_observation])
        invoke(command)
        manifest['timings_s']['observation_through_raster'] = time.perf_counter() - tick
        source = output/'automatic/observation'
        run_root = output/'automatic/run'
        for name in ('depth', 'segmentation', 'observation', 'downstream_bundle'):
            check_stage(source, name)
        for name in ('observation', 'subject_mesh', 'scene', 'actions', 'physics', 'raster'):
            checked = check_stage(run_root, name)
            manifest['stages'][name] = {'contract': f'automatic/run/stages/{name}.json',
                                         'seconds': checked['seconds']}
            manifest['implementations'][name] = {'name': checked['implementation'], 'version': checked['version'],
                                                  'parameters': checked['parameters']}
        manifest['stage'] = 'proxy_render'
        tick = time.perf_counter()
        invoke([Path(__file__).with_name('render_adapters.py'), run_root, output/'proxy',
                '--renderer', data.get('renderer', 'raster')])
        manifest['timings_s']['proxy_render'] = time.perf_counter() - tick
        render = json.loads((output/'proxy/render.json').read_text())
        manifest['implementations']['proxy_render'] = {'name': data.get('renderer', 'raster'),
                                                         'version': '1', 'parameters': {'nearby_yaw_deg': 5}}
        manifest['stages']['proxy_render'] = {'contract': 'proxy/render.json', 'seconds': render['seconds']}
        manifest['outputs'] = {str(path.relative_to(output)): digest(path) for path in output.rglob('*')
                               if path.is_file() and path.name != 'chain.json'}
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['failure'] = {'stage': manifest['stage'], 'error_type': type(exc).__name__, 'reason': str(exc)}
        raise
    finally:
        manifest['timings_s']['total'] = time.perf_counter() - started
        write(output/'chain.json', manifest)


def check(root):
    manifest = json.loads((root/'chain.json').read_text())
    if manifest.get('status') != 'complete':
        return {'status': 'failed', 'failure': manifest.get('failure')}
    integrity = {}
    for name, expected in manifest['outputs'].items():
        path = (root/name).resolve()
        integrity[name] = path.is_relative_to(root.resolve()) and path.is_file() and digest(path) == expected
    for name in ('depth', 'segmentation', 'observation', 'downstream_bundle'):
        try:
            check_stage(root/'automatic/observation', name)
            integrity['observation:'+name] = True
        except (OSError, ValueError, KeyError):
            integrity['observation:'+name] = False
    for name in ('observation', 'subject_mesh', 'scene', 'actions', 'physics', 'raster'):
        try:
            check_stage(root/'automatic/run', name)
            integrity['stage:'+name] = True
        except (OSError, ValueError, KeyError):
            integrity['stage:'+name] = False
    try:
        render = json.loads((root/'proxy/render.json').read_text())
        integrity['proxy_status'] = render['status'] == 'complete'
        integrity['proxy_inputs'] = all(
            digest(root/'automatic/run'/name) == expected
            for name, expected in render['inputs'].items())
    except (OSError, ValueError, KeyError):
        integrity['proxy_status'] = False
    return {'status': 'complete' if integrity and all(integrity.values()) else 'failed',
            'integrity': integrity, 'timings_s': manifest['timings_s']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('run')
    create.add_argument('image', type=Path)
    create.add_argument('config', type=Path)
    create.add_argument('output', type=Path)
    create.add_argument('--resume-observation', type=Path)
    verify = sub.add_parser('check')
    verify.add_argument('output', type=Path)
    replay = sub.add_parser('replay')
    replay.add_argument('saved', type=Path)
    replay.add_argument('output', type=Path)
    replay.add_argument('--from-stage', choices=('observation', 'scene', 'physics', 'render'), default='observation')
    replay.add_argument('--renderer', choices=('raster', 'gaussian'))
    args = parser.parse_args()
    if args.command == 'check':
        result = check(args.output.resolve())
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(0 if result['status'] == 'complete' else 1)
    if args.command == 'run':
        run(args.image, args.config, args.output, args.resume_observation)
        return
    source = args.saved.resolve()
    checked = check(source)
    if checked['status'] != 'complete':
        raise ValueError('saved chain failed read-only check')
    data = json.loads((source/'chain.json').read_text())['config']
    if args.renderer:
        data['renderer'] = args.renderer
    if args.from_stage == 'render':
        invoke([Path(__file__).with_name('render_adapters.py'), source/'automatic/run', args.output,
                '--renderer', data.get('renderer', 'raster')])
        return
    if args.from_stage == 'physics':
        # Physics outputs are immutable upstream inputs for either proxy renderer.
        invoke([Path(__file__).with_name('render_adapters.py'), source/'automatic/run', args.output,
                '--renderer', data.get('renderer', 'raster')])
        return
    run(source/'automatic/observation/inputs/input.jpg', data, args.output,
        source/'automatic/observation', source/'automatic/run' if args.from_stage == 'scene' else None)


if __name__ == '__main__':
    main()
