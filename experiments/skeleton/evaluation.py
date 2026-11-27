"""Validate the 3-development/3-holdout cross-game pilot and score frozen runs.

Input JSON: {schema:1, games:[{id,split,scenes:[{id,image,eligible,
annotation,run}]}]}. Annotation is scoring-only and never passed to the runner.
Freeze JSON: {schema:1, algorithm_revision, config_sha256, failure_conditions,
thresholds:{detection_iou_min, geometry_depth_mae_m_max,
contact_penetration_m_max, render_iou_min, total_seconds_max},
aggregate_success_min}. Freeze before opening holdout images.
"""
import argparse
import json
from pathlib import Path

METRICS = ('detection_iou', 'geometry_depth_mae_m', 'contact_penetration_m',
           'render_iou', 'total_seconds')
LIMITS = ('detection_iou_min', 'geometry_depth_mae_m_max',
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
                if run['status'] != 'complete':
                    raise ValueError(run.get('reason', 'pipeline failed'))
                values = run['metrics']
                passed = (values['detection_iou'] >= thresholds['detection_iou_min'] and
                          values['geometry_depth_mae_m'] <= thresholds['geometry_depth_mae_m_max'] and
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--split', choices=('development', 'holdout'))
    args = parser.parse_args()
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
