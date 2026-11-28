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
RENDERERS = ('raster', 'gpu_raster', 'gaussian')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def config_read(path):
    data = json.loads(path.read_text())
    allowed = {'schema', 'renderer', 'seconds', 'mesh_provider', 'gsplat_cache_dir', *SOURCE_KEYS}
    if not isinstance(data, dict) or set(data) - allowed or data.get('schema') != 1:
        raise ValueError('unsupported configuration schema or keys')
    if data.get('renderer', 'raster') not in RENDERERS:
        raise ValueError('renderer must be raster, gpu_raster or gaussian')
    if 'gsplat_cache_dir' in data and not isinstance(data['gsplat_cache_dir'], str):
        raise ValueError('gsplat_cache_dir must be a path string')
    seconds = data.get('seconds', 5)
    if not isinstance(seconds, (int, float)) or not 0 < seconds <= 60:
        raise ValueError('seconds must be in (0, 60]')
    if data.get('mesh_provider', 'hunyuan3d21') != 'hunyuan3d21':
        raise ValueError('unsupported mesh provider')
    for key in SOURCE_KEYS:
        if key in data and not isinstance(data[key], str):
            raise ValueError(f'{key} must be a path string')
    return data


def invoke(command):
    subprocess.run([sys.executable, *map(str, command)], check=True)


def renderer_command(source, output, renderer, cache_dir=None):
    command = [Path(__file__).with_name('render_adapters.py'), source, output,
               '--renderer', renderer]
    if renderer == 'gaussian' and cache_dir:
        command.extend(['--gsplat-cache-dir', cache_dir])
    return command



def semantic_check(run_root):
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('check.py')),
                             str(run_root)], capture_output=True, text=True)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f'saved-run checker produced no JSON: {result.stderr}') from exc
    return {'passed': result.returncode == 0 and report.get('passed') is True,
            'checks': report.get('checks', {}), 'metrics': report.get('metrics', {})}

def run(image, config, output, resume_observation=None, resume_scene=None, resume_perception=None):
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
            if data.get('renderer') == 'gpu_raster':
                command.extend(['--main-renderer', 'gpu_raster'])
        else:
            if resume_perception:
                source = resume_perception.resolve()
                for name in ('depth', 'segmentation', 'observation'):
                    check_stage(source, name)
                if digest(image) != digest(source/'inputs/input.jpg'):
                    raise ValueError('replay image differs from saved perception')
                adapters = output/'replay_sources'
                for kind, filename, original in (('depth', 'prediction.npz', 'inputs/prediction.npz'),
                                                 ('segmentation', 'sam2_masks.npz', 'sam2_masks.npz')):
                    folder = adapters/kind
                    folder.mkdir(parents=True)
                    shutil.copyfile(source/original, folder/filename)
                    stage = check_stage(source, kind)
                    metadata = {'model': stage['implementation'], 'input_sha256': digest(image),
                                'files': {filename: {'sha256': digest(folder/filename)}}}
                    if kind == 'depth':
                        metadata['is_metric'] = json.loads((source/'depth_capability.json').read_text())['scale'] == 'metric_m'
                    write(folder/'run.json', metadata)
                data = {**data, 'depth_source': str(adapters/'depth'),
                        'segmentation_source': str(adapters/'segmentation')}
                data.pop('saved_mesh', None)
            command = [Path(__file__).with_name('original_image_run.py'), image, output/'automatic',
                       '--seconds', data.get('seconds', 5), '--mesh-provider', data.get('mesh_provider', 'hunyuan3d21')]
            if data.get('renderer') == 'gpu_raster':
                command.extend(['--main-renderer', 'gpu_raster'])
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
        invoke(renderer_command(run_root, output/'proxy', data.get('renderer', 'raster'),
                                data.get('gsplat_cache_dir')))
        manifest['timings_s']['proxy_render'] = time.perf_counter() - tick
        render = json.loads((output/'proxy/render.json').read_text())
        manifest['implementations']['proxy_render'] = {'name': data.get('renderer', 'raster'),
                                                         'version': '1', 'parameters': {'nearby_yaw_deg': 5}}
        manifest['stages']['proxy_render'] = {'contract': 'proxy/render.json', 'seconds': render['seconds']}
        manifest['stage'] = 'semantic_check'
        semantic = semantic_check(run_root)
        manifest['semantic_check'] = semantic
        if not semantic['passed']:
            raise ValueError('saved-run physical or visual acceptance failed')
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
    try:
        semantic = semantic_check(root/'automatic/run')
        integrity['semantic_check'] = semantic['passed']
    except (OSError, ValueError) as exc:
        semantic = {'passed': False, 'reason': str(exc)}
        integrity['semantic_check'] = False
    return {'status': 'complete' if integrity and all(integrity.values()) else 'failed',
            'integrity': integrity, 'semantic_check': semantic, 'timings_s': manifest['timings_s']}



def compare(saved, output, alternate, cache_dir=None):
    saved, output = saved.resolve(), output.resolve()
    if output == saved or output in saved.parents or saved in output.parents:
        raise ValueError('comparison output must be separate from saved chain')
    if output.exists() and any(output.iterdir()):
        raise ValueError('comparison output is not empty')
    output.mkdir(parents=True, exist_ok=True)
    result = {'schema': 1, 'status': 'failed',
              'alternate_renderer': alternate,
              'artifact_roots': {'source_chain': 'caller-supplied saved chain',
                                 'comparison': 'directory containing comparison.json'}}
    try:
        if check(saved)['status'] != 'complete':
            raise ValueError('saved chain failed read-only check')
        baseline_path = saved/'proxy/render.json'
        baseline = json.loads(baseline_path.read_text())
        if baseline.get('status') != 'complete' or baseline.get('renderer') == alternate:
            raise ValueError('alternate renderer must differ from completed baseline')
        if baseline.get('inputs') != baseline.get('inputs_after'):
            raise ValueError('baseline upstream changed during rendering')
        invoke(renderer_command(saved/'automatic/run', output/'alternate', alternate, cache_dir))
        alternate_path = output/'alternate/render.json'
        other = json.loads(alternate_path.read_text())
        if other.get('status') != 'complete' or other.get('renderer') != alternate:
            raise ValueError('alternate renderer did not complete')
        if baseline['inputs'] != other.get('inputs') or other.get('inputs') != other.get('inputs_after'):
            raise ValueError('fixed upstream SHA mismatch between renderers')
        result['source_chain_sha256'] = digest(saved/'chain.json')
        result['upstream_sha256'] = baseline['inputs']
        result['renderers'] = {
            row['renderer']: {'root': root_name, 'manifest': path, 'manifest_sha256': digest(file),
                              'seconds': row.get('seconds'), 'diagnostics': row.get('diagnostics'),
                              'views': row.get('views'), 'outputs': row.get('outputs')}
            for row, root_name, path, file in ((baseline, 'source_chain', 'proxy/render.json', baseline_path),
                                               (other, 'comparison', 'alternate/render.json', alternate_path))}
        result['status'] = 'complete'
        return result
    except Exception as exc:
        result['failure'] = {'type': type(exc).__name__, 'reason': str(exc)}
        raise
    finally:
        write(output/'comparison.json', result)

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
    paired = sub.add_parser('compare')
    paired.add_argument('saved', type=Path)
    paired.add_argument('output', type=Path)
    paired.add_argument('--alternate-renderer', choices=RENDERERS, required=True)
    paired.add_argument('--gsplat-cache-dir', type=Path)
    replay = sub.add_parser('replay')
    replay.add_argument('saved', type=Path)
    replay.add_argument('output', type=Path)
    replay.add_argument('--from-stage', choices=('observation', 'mesh', 'scene', 'physics', 'render'), default='observation')
    replay.add_argument('--renderer', choices=RENDERERS)
    replay.add_argument('--gsplat-cache-dir', type=Path)
    args = parser.parse_args()
    if args.command == 'check':
        result = check(args.output.resolve())
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(0 if result['status'] == 'complete' else 1)
    if args.command == 'run':
        run(args.image, args.config, args.output, args.resume_observation)
        return
    if args.command == 'compare':
        result = compare(args.saved, args.output, args.alternate_renderer, args.gsplat_cache_dir)
        print(json.dumps(result, sort_keys=True))
        return
    source = args.saved.resolve()
    checked = check(source)
    if checked['status'] != 'complete':
        raise ValueError('saved chain failed read-only check')
    data = json.loads((source/'chain.json').read_text())['config']
    if args.renderer:
        data['renderer'] = args.renderer
    if args.from_stage == 'render':
        invoke(renderer_command(source/'automatic/run', args.output,
                                data.get('renderer', 'raster'), args.gsplat_cache_dir or data.get('gsplat_cache_dir')))
        return
    if args.from_stage == 'physics':
        # Physics outputs are immutable upstream inputs for either proxy renderer.
        invoke(renderer_command(source/'automatic/run', args.output,
                                data.get('renderer', 'raster'), args.gsplat_cache_dir or data.get('gsplat_cache_dir')))
        return
    # The observation bundle contains the independently validated mesh contract.
    # Mesh replay revalidates it in original_image_run before scene fitting.
    run(source/'automatic/observation/inputs/input.jpg', data, args.output,
        source/'automatic/observation' if args.from_stage in ('mesh', 'scene') else None,
        source/'automatic/run' if args.from_stage == 'scene' else None,
        source/'automatic/observation' if args.from_stage == 'observation' else None)


if __name__ == '__main__':
    main()
