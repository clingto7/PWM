"""Validate the 3-development/3-holdout cross-game pilot and score frozen runs.

Input JSON: {schema:1, games:[{id,split,scenes:[{id,image,eligible,
annotation,run}]}]}. Annotation is scoring-only and never passed to the runner.
Freeze JSON: {schema:1, algorithm_revision, config_sha256, failure_conditions,
thresholds:{detection_iou_min, geometry_depth_mae_m_max,
contact_penetration_m_max, render_iou_min, total_seconds_max},
aggregate_success_min}. Freeze before opening holdout images.
"""
import argparse
import hashlib
import math
import json
from pathlib import Path

from stage_contracts import digest

METRICS = ('detection_iou', 'geometry_contact_fit_rms_m', 'contact_penetration_m',
           'render_iou', 'total_seconds')
LIMITS = ('detection_iou_min', 'geometry_contact_fit_rms_m_max',
          'contact_penetration_m_max', 'render_iou_min', 'total_seconds_max')


def validate(dataset):
    if dataset.get('schema') != 1:
        raise ValueError('unsupported evaluation schema')
    games = dataset.get('games', [])
    if len(games) != 6 or len({g['id'] for g in games}) != 6:
        raise ValueError('require six distinct games')
    for split in ('development', 'holdout'):
        group = [g for g in games if g['split'] == split]
        if len(group) != 3:
            raise ValueError(f'require three {split} games')
        for game in group:
            scenes = game['scenes']
            eligible = [s for s in scenes if s['eligible']]
            if len(eligible) < 5 or len({s['id'] for s in eligible}) != len(eligible):
                raise ValueError(f'{game["id"]}: require five distinct eligible scenes')
            if any(not s.get('image') or not s.get('annotation') for s in eligible):
                raise ValueError(f'{game["id"]}: image and scoring annotation required')
    return games


def score(dataset, freeze, split):
    games = validate(dataset)
    if freeze.get('schema') != 1 or not freeze.get('algorithm_revision') or not freeze.get('config_sha256') or not freeze.get('failure_conditions'):
        raise ValueError('frozen algorithm, config and failure conditions required')
    thresholds = freeze.get('thresholds', {})
    if set(thresholds) != set(LIMITS) or any(not isinstance(thresholds[k], (int, float)) for k in LIMITS):
        raise ValueError('all numeric thresholds must be frozen')
    sealed = freeze.get('sealed_sha256')
    canonical = json.dumps({k: v for k, v in freeze.items() if k != 'sealed_sha256'},
                           sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if not sealed or hashlib.sha256(canonical).hexdigest() != sealed or dataset.get('freeze_sha256') != sealed:
        raise ValueError('dataset must pin an intact sealed freeze manifest')
    target = freeze.get('aggregate_success_min')
    if not isinstance(target, (int, float)) or not 0 <= target <= 1:
        raise ValueError('aggregate success target must be frozen')
    rows = []
    for game in games:
        if game['split'] != split:
            continue
        for scene in game['scenes']:
            if not scene['eligible']:
                continue
            run_path = Path(scene.get('run', ''))
            try:
                run = json.loads((run_path/'evaluation.json').read_text())
                if run.get('provenance', {}).get('config_sha256') != freeze['config_sha256'] or run.get('provenance', {}).get('revision') != freeze['algorithm_revision']:
                    raise ValueError('run revision or config differs from freeze')
                if digest(Path(scene['image'])) != run['provenance']['image_sha256'] or digest(Path(scene['annotation'])) != run['provenance']['annotation_sha256']:
                    raise ValueError('run image or annotation differs from dataset')
                if run['status'] != 'complete':
                    raise ValueError(run.get('reason', 'pipeline failed'))
                values = run['metrics']
                passed = (values['detection_iou'] >= thresholds['detection_iou_min'] and
                          values['geometry_contact_fit_rms_m'] <= thresholds['geometry_contact_fit_rms_m_max'] and
                          values['contact_penetration_m'] <= thresholds['contact_penetration_m_max'] and
                          values['render_iou'] >= thresholds['render_iou_min'] and
                          values['total_seconds'] <= thresholds['total_seconds_max'])
                reason = None if passed else 'metric threshold failed'
            except (OSError, ValueError, KeyError, TypeError) as exc:
                passed, reason, values = False, str(exc), None
            rows.append({'game': game['id'], 'scene': scene['id'], 'passed': passed,
                         'reason': reason, 'metrics': values})
    successes = sum(row['passed'] for row in rows)
    return {'split': split, 'eligible': len(rows), 'successes': successes,
            'success_rate': successes/len(rows), 'target_met': successes/len(rows) >= target,
            'scenes': rows}



def pixels(runs):
    return {(y, x) for y, x0, x1 in runs for x in range(x0, x1)}


def iou(a, b):
    left, right = pixels(a), pixels(b)
    union = left | right
    return len(left & right)/len(union) if union else 0.0


def score_run(root, annotation, output):
    from automatic_chain import check as chain_check
    root = root.resolve()
    result = {'schema': 1, 'status': 'failed', 'provenance': {},
              'annotation': str(annotation)}
    try:
        chain = json.loads((root/'chain.json').read_text())
        checked = chain_check(root)
        result['chain_check'] = checked
        original = root/'automatic/observation/inputs/input.jpg'
        bundle = json.loads((root/'automatic/run/bundle.json').read_text())
        result['provenance'] = {
            'image_sha256': digest(original), 'config_sha256': chain['inputs']['config'],
            'revision': bundle['revision'], 'chain_sha256': digest(root/'chain.json'),
            'annotation_sha256': digest(annotation)}
        if checked['status'] != 'complete':
            raise ValueError('chain read-only check failed')
        reference = json.loads(annotation.read_text())
        runs = reference['subject_runs_y_x0_x1']
        if reference.get('schema') != 1 or not runs:
            raise ValueError('annotation requires subject pixel runs')
        projection = json.loads((root/'automatic/run/initial_projection.json').read_text())
        metrics = checked['semantic_check']['metrics']
        values = {'detection_iou': iou(projection['reference_runs_y_x0_x1'], runs),
                  'render_iou': iou(projection['rendered_runs_y_x0_x1'], runs),
                  'geometry_contact_fit_rms_m': metrics['contact_fit_rms_m'],
                  'contact_penetration_m': metrics['max_penetration_m'],
                  'total_seconds': chain['timings_s']['total']}
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values.values()):
            raise ValueError('non-finite or missing evaluation metric')
        result['metrics'] = values
        result['status'] = 'complete'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['reason'] = str(exc)
    output.mkdir(parents=True, exist_ok=True)
    (output/'evaluation.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, nargs='?')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--annotation', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--split', choices=('development', 'holdout'))
    args = parser.parse_args()
    if args.run:
        if not args.annotation or not args.output:
            parser.error('--run requires --annotation and --output')
        result = score_run(args.run, args.annotation, args.output)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result['status'] == 'complete' else 1)
    if not args.dataset:
        parser.error('dataset required')
    dataset = json.loads(args.dataset.read_text())
    if not args.freeze:
        validate(dataset)
        print(json.dumps({'valid': True, 'status': 'protocol only; trials not run'}))
        return
    if not args.split:
        parser.error('--split required with --freeze')
    print(json.dumps(score(dataset, json.loads(args.freeze.read_text()), args.split), indent=2))


if __name__ == '__main__':
    main()
