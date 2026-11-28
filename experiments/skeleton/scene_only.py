"""Build the checked scene and passive fit trial for the online chain."""
import json
import time
from pathlib import Path

from artifacts import SceneArtifacts
from bundle import sources, copy_inputs, revision
from first_image_pipeline import digest, write_json, fit_settings
from physics import PyBulletSolver
from scene import reconstruct
from stage_contracts import check as check_stage, record as record_stage, validate_resume


def build(args):
    start = time.perf_counter()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'output directory is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    try:
        write_json(out/'stage_status.json', {'stage': 'scene', 'status': 'running'})
        bundled = copy_inputs(sources(args.bundle), out)
        inputs = {name: out/path for name, path in bundled.items()}
        hashes = {name: digest(path) for name, path in inputs.items()}
        write_json(out/'bundle.json', {'schema': 1, 'inputs': bundled,
            'source_sha256': hashes, 'revision': revision()})
        if args.resume_scene:
            import shutil
            saved = args.resume_scene.resolve()
            parameters = {'radius_m': args.radius, 'angle_deg': args.angle,
                'cell_m': args.cell, 'near_depth_m': args.near_depth,
                'ground_inlier_m': args.ground_inlier,
                'obstacle_height_m': args.obstacle_height,
                'obstacle_min_pixels': args.obstacle_min_pixels,
                'obstacle_min_cells': args.obstacle_min_cells,
                'obstacles_render_only': args.obstacles_render_only, **fit_settings(args)}
            validate_resume(saved, parameters)
            prior = json.loads((saved/'bundle.json').read_text())
            if prior['source_sha256'] != hashes:
                raise ValueError('saved scene inputs differ from current bundle')
            for name in (*SceneArtifacts.NAMES, 'fit_diagnostics.json',
                         'fit_visuals/before.png', 'fit_visuals/after.png',
                         'fit_visuals/before_depth_error.png', 'fit_visuals/after_depth_error.png'):
                target = out/name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(saved/name, target)
            ground_time = scene_time = writing = 0.0
        else:
            result = reconstruct(inputs, args, out, start)
            ground_time, scene_time, writing = result.ground_time, result.scene_time, result.writing_seconds
        scene = json.loads((out/'scene.json').read_text())
        fit = json.loads((out/'fit_diagnostics.json').read_text())
        if not fit.get('fit_trusted') or not all(fit.get('trust_checks', {}).values()):
            raise ValueError('untrusted image fit')
        import pybullet as bullet
        initialization_start = time.perf_counter()
        client = bullet.connect(bullet.DIRECT)
        try:
            solver = PyBulletSolver(bullet)
            ground, body, _ = solver.create_world(SceneArtifacts(out).load(), args, out)
            initialization_s = time.perf_counter()-initialization_start
            trial_start = time.perf_counter()
            trial = solver.trial_without_impulse(body, ground)
            trial['seconds'] = time.perf_counter()-trial_start
        finally:
            bullet.disconnect(client)
        write_json(out/'fit_trial.json', trial)
        if not trial['trusted']:
            raise ValueError('untrusted initial subject fit: see fit_trial.json')
        params = {'radius_m': args.radius, 'angle_deg': args.angle, 'cell_m': args.cell,
                  'friction': args.friction, 'density_kg_m3': args.density,
                  'near_depth_m': args.near_depth, 'ground_inlier_m': args.ground_inlier,
                  'obstacle_height_m': args.obstacle_height,
                  'obstacle_min_pixels': args.obstacle_min_pixels,
                  'obstacle_min_cells': args.obstacle_min_cells,
                  'obstacles_render_only': args.obstacles_render_only,
                  'seconds': args.seconds, 'fit_evidence': 'image_depth_support'}
        fit_files = ['fit_diagnostics.json', 'fit_visuals/before.png',
                     'fit_visuals/after.png', 'fit_visuals/before_depth_error.png',
                     'fit_visuals/after_depth_error.png']
        observation = check_stage(args.bundle, 'observation')
        mesh = check_stage(args.bundle, 'downstream_bundle')
        record_stage(out, 'observation', list(bundled.values()), list(bundled.values()),
                     observation['implementation'], observation['version'],
                     {'selection_sha256': digest(args.bundle/'selection.json')}, 0.0)
        record_stage(out, 'subject_mesh', [bundled['hunyuan_mesh']], [bundled['hunyuan_mesh']],
                     mesh['implementation'], mesh['version'], {}, 0.0)
        record_stage(out, 'scene', list(bundled.values()),
                     [*SceneArtifacts.NAMES, *fit_files, 'fit_trial.json'],
                     'scene_only.reconstruct_and_passive_trial', '1', parameters if args.resume_scene else
                     {key: params[key] for key in ('radius_m', 'angle_deg', 'cell_m', 'near_depth_m',
                       'ground_inlier_m', 'obstacle_height_m', 'obstacle_min_pixels',
                       'obstacle_min_cells', 'obstacles_render_only', 'fit_evidence')},
                     ground_time + scene_time + trial['seconds'])
        write_json(out/'run.json', {'schema': 3, 'mode': 'scene_only',
            'resume_scene_provenance': {'source': str(args.resume_scene.resolve()),
                'source_fit_timings_s': json.loads((args.resume_scene/'run.json').read_text())['timings_s']}
                if args.resume_scene else None,
            'parameters': params, 'inputs': {name: {'path': bundled[name], 'sha256': value}
                                           for name, value in hashes.items()},
            'scene_artifacts': list(SceneArtifacts.NAMES),
            'stage_contracts': {name: f'stages/{name}.json' for name in
                                ('observation', 'subject_mesh', 'scene')},
            'timings_s': {'ground_fit': ground_time, 'scene_build': scene_time,
                **({} if args.resume_scene else scene.get('timings_s', {})),
                'passive_world_initialization': initialization_s,
                'physical_trial': trial['seconds'],
                'writing': writing, 'total': time.perf_counter()-start}})
        write_json(out/'stage_status.json', {'stage': 'complete', 'status': 'complete'})
    except Exception as exc:
        write_json(out/'stage_status.json', {'stage': 'scene', 'status': 'failed',
            'error_type': type(exc).__name__, 'reason': str(exc)})
        raise
