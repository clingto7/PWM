"""Write a hashed first-round conclusion into an existing run directory."""
import argparse
import hashlib
import json
from pathlib import Path

from check import check
from gates import THRESHOLDS


def decision(value, limit, *, minimum=False, evidence=True):
    if limit is None or value is None:
        return '未设'
    if not evidence:
        return False
    return value >= limit if minimum else value <= limit


def conclude(root, thresholds=None):
    limits = {**THRESHOLDS, **(thresholds or {})}
    record = json.loads((root / 'run.json').read_text())
    scene = json.loads((root / 'scene.json').read_text())
    checked = check(root)
    metrics = checked.get('metrics', {})
    checks = checked['checks']
    sound = checks.get('hashes') is True
    projection = metrics.get('initial_projection_mask_iou')
    ground = metrics.get('ground_fit_rms_m')
    road = metrics.get('road_false_positive_ratio')
    fallback = scene.get('obstacles', {}).get('render_only', record.get('parameters', {}).get('obstacles_render_only'))
    proxy_keys = ['frame_count']
    if 'obstacles' in scene:
        proxy_keys.append('obstacle_render_ids')
    proxy_checks = [checks.get(key) for key in proxy_keys]
    proxy_usable = bool(proxy_checks) and all(v is True for v in proxy_checks) and sound
    road_gate = decision(road, limits['road_false_positive_ratio_max'], evidence=sound and checks.get('obstacle_road_ratio') is True)
    if road_gate is False and fallback and proxy_usable:
        road_gate = True
    cases = metrics.get('impulse_cases', {})
    solver = {
        'invariants': {k: {'value': metrics.get(v), 'decision': checks.get(k, '未设')}
                       for k, v in (('penetration', 'max_penetration_m'), ('resting_height', 'final_resting_height_error_m'),
                                    ('final_speed', 'final_speed_m_s'), ('no_energy_gain', 'max_step_specific_energy_gain_j_kg'),
                                    ('replay', 'replay_trajectory_identical'))},
        'analytic_errors': {name: {k: v for k, v in case['metrics'].items() if 'error' in k or 'residual' in k}
                            for name, case in cases.items()},
        'case_decisions': {name: case['checks'] for name, case in cases.items()},
    }
    solver_decision = '未设' if not cases else (sound and checks.get('impulse_cases') is True and
                                                all(checks.get(k) is True for k in ('penetration', 'resting_height', 'final_speed', 'no_energy_gain', 'replay')))
    fit = scene.get('car', {}).get('mesh_fit', {})
    size = fit.get('size_check', {})
    conditions = {
        'vehicle_silhouette': {'iou': projection, 'threshold_min': limits['initial_mask_iou_min'],
                               'decision': decision(projection, limits['initial_mask_iou_min'], minimum=True, evidence=sound and checks.get('initial_projection_record') is True)},
        'ground_fit': {'rms_m': ground, 'threshold_max_m': limits['ground_fit_rms_m'],
                       'decision': decision(ground, limits['ground_fit_rms_m'], evidence=sound and checks.get('ground_fit') is True)},
        'road_obstacles': {'false_positive_ratio': road, 'threshold_max': limits['road_false_positive_ratio_max'],
                           'render_only_fallback': fallback, 'proxy_signal_usable': proxy_usable,
                           'decision': road_gate},
        'solver': {**solver, 'decision': solver_decision},
    }
    decisions = [entry['decision'] for entry in conditions.values()]
    report = {'schema': 1, 'source_run_sha256': hashlib.sha256((root / 'run.json').read_bytes()).hexdigest(),
              'source_integrity_passed': sound, 'conditions': conditions,
              'vehicle_size_check': {'reasonable': size.get('reasonable', '未设'), 'fit_to_reference_ratio': size.get('fit_to_reference_ratio'),
                                     'note': '车型尺寸核对独立于四条失败条件'},
              'timings_s': {'total': '未设', **record.get('timings_s', {})},
              'overall_decision': False if not sound else (True if all(v is True for v in decisions) else ('未设' if '未设' in decisions and all(v is True or v == '未设' for v in decisions) else False))}
    # The digest covers every report field except the digest itself.
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    report['content_sha256'] = hashlib.sha256(canonical).hexdigest()
    target = root / 'first_round_conclusion.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--gates', type=Path, help='JSON overrides; null means 未设')
    args = parser.parse_args()
    overrides = json.loads(args.gates.read_text()) if args.gates else None
    if overrides and (set(overrides) - set(THRESHOLDS) or any(v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0) for v in overrides.values())):
        parser.error('gates must be known names with nonnegative numeric values or null')
    result = conclude(args.run_dir.resolve(), overrides)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result['overall_decision'] is True else 1)
