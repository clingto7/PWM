"""Compare Bullet contact settings on one saved scene without reconstruction.

Run with the pinned physics environment:
  python experiments/skeleton/diagnose_contact_energy.py RUN_DIRECTORY
The saved scene is read only; results are printed as JSON.
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

from artifacts import SceneArtifacts
from physics import PyBulletSolver


def specific_energy(row, inertia, mass):
    x, y, z, w = row[3:7]
    rotation = ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
    local = [sum(rotation[j][i]*row[10+j] for j in range(3)) for i in range(3)]
    return (sum(v*v for v in row[7:10])/2
            + sum(inertia[i]*local[i]**2 for i in range(3))/(2*mass)
            + 9.81*row[2])


def run_case(geometry, settings, velocity, seconds, variant, target_id=None):
    import pybullet as p
    client = p.connect(p.DIRECT)
    try:
        solver = PyBulletSolver(p)
        ground, body, obstacles = solver.create_world(geometry, settings, Path(settings.root),
                                               solver_parameters=variant['parameters'])
        effective_parameters = p.getPhysicsEngineParameters()
        inertia = p.getDynamicsInfo(body, -1)[2]
        mass = geometry.mass
        launch = variant.get('velocity', velocity)
        p.resetBaseVelocity(body, linearVelocity=[v*variant.get('speed_scale', 1.) for v in launch])
        rows = []
        target_contact_samples = 0
        max_penetration = 0.0
        for frame in range(int(seconds*solver.hz)):
            if frame:
                p.stepSimulation()
            pos, quat = p.getBasePositionAndOrientation(body)
            vel, omega = p.getBaseVelocity(body)
            row = [*pos, *quat, *vel, *omega]
            contacts = p.getContactPoints(bodyA=body)
            target_contact_samples += any(obstacles.get(c[2]) == target_id for c in contacts) if target_id is not None else 0
            max_penetration = max(max_penetration, max((-float(c[8]) for c in contacts), default=0.0))
            rows.append((specific_energy(row, inertia, mass), row,
                         min((float(c[8]) for c in contacts), default=None), len(contacts)))
        # Match check.py: omit stationary frame, then compare consecutive steps.
        gains = [(rows[i][0]-rows[i-1][0], i) for i in range(2, len(rows))]
        gain, frame = max(gains)
        before, after = rows[frame-1], rows[frame]
        final = rows[-1][1]
        return {'variant': variant['name'], 'parameters': variant['parameters'],
                'speed_scale': variant.get('speed_scale', 1.),
                'launch_velocity_m_s': [v*variant.get('speed_scale', 1.) for v in launch],
                'target_contact_samples': target_contact_samples,
                'max_contact_penetration_m': max_penetration,
                'final_position_m': final[:3],
                'final_linear_speed_m_s': math.sqrt(sum(v*v for v in final[7:10])),
                'final_angular_speed_rad_s': math.sqrt(sum(v*v for v in final[10:13])),
                'effective_engine_parameters': effective_parameters,
                'max_step_specific_energy_gain_j_kg': max(0., gain),
                'peak_frame': frame, 'peak_time_s': frame/solver.hz,
                'horizontal_distance_m': math.hypot(rows[-1][1][0]-rows[0][1][0],
                                                    rows[-1][1][1]-rows[0][1][1]),
                'vertical_displacement_m': rows[-1][1][2]-rows[0][1][2],
                'before': {'z_m': before[1][2], 'linear_velocity_m_s': before[1][7:10],
                           'angular_velocity_rad_s': before[1][10:13],
                           'minimum_contact_distance_m': before[2], 'contact_count': before[3]},
                'after': {'z_m': after[1][2], 'linear_velocity_m_s': after[1][7:10],
                          'angular_velocity_rad_s': after[1][10:13],
                          'minimum_contact_distance_m': after[2], 'contact_count': after[3]}}
    finally:
        p.disconnect(client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--action', choices=('coast', 'ballistic', 'obstacle'), default='ballistic')
    args = parser.parse_args()
    root = args.run_directory
    geometry = SceneArtifacts(root).load()
    run = json.loads((root/'run.json').read_text())
    impulses = json.loads((root/'impulses.json').read_text())
    params = run['parameters']
    settings = SimpleNamespace(root=str(root), friction=params['friction'],
                               radius=params['radius_m'])
    action = args.action
    velocity = impulses[action]['delta_velocity_m_s']
    trajectory = root/'trajectory.json' if action == 'coast' else root/action/'trajectory.json'
    saved_rows = json.loads(trajectory.read_text())
    inertia = run['impulse_cases'][action]['inertia_diagonal_kg_m2']
    saved_energy = [specific_energy(row, inertia, geometry.mass) for row in saved_rows]
    saved_gain, saved_frame = max((saved_energy[i]-saved_energy[i-1], i)
                                  for i in range(2, len(saved_energy)))
    session_path = root/'physics_session.json'
    metadata = (json.loads(session_path.read_text()).get('physics_solver')
                if session_path.is_file() else geometry.scene.get('physics_solver')) or {}
    if 'contact_erp' in metadata:
        baseline_parameters = {'contactERP': metadata['contact_erp']}
        if metadata.get('split_impulse') is True:
            baseline_parameters.update(useSplitImpulse=1,
                splitImpulsePenetrationThreshold=metadata['split_impulse_penetration_threshold_m'])
    else:
        # The integration #19 world left contact ERP and split settings unset.
        baseline_parameters = {}
    variants = [
        {'name': 'saved_baseline', 'parameters': baseline_parameters},
        {'name': 'unsplit', 'parameters': {'useSplitImpulse': 0}},
        {'name': 'current_split', 'parameters': {'useSplitImpulse': 1,
                                                'splitImpulsePenetrationThreshold': 0.0}},
        {'name': 'split_default_threshold', 'parameters': {'useSplitImpulse': 1,
                                                           'splitImpulsePenetrationThreshold': -0.04}},
        {'name': 'split_low_erp', 'parameters': {'useSplitImpulse': 1,
                                               'splitImpulsePenetrationThreshold': 0.0,
                                               'contactERP': 0.1}},
        {'name': 'default_split_low_erp', 'parameters': {'useSplitImpulse': 1,
                                                       'splitImpulsePenetrationThreshold': -0.04,
                                                       'contactERP': 0.1}},
        {'name': 'default_split_erp_0_05', 'parameters': {'useSplitImpulse': 1,
                                                       'splitImpulsePenetrationThreshold': -0.04,
                                                       'contactERP': 0.05}},
        {'name': 'default_split_erp_0_025', 'parameters': {'useSplitImpulse': 1,
                                                        'splitImpulsePenetrationThreshold': -0.04,
                                                        'contactERP': 0.025}},
    ]
    if action == 'obstacle':
        variants = [variants[0]] + [
            {'name': f'obstacle_speed_{scale:.2f}', 'parameters': {'contactERP':0.025},
             'speed_scale':scale} for scale in (1.0, 1.1, 1.15, 1.2, 1.3)]
        import numpy as np
        with np.load(root/'obstacles.npz') as obstacle_data:
            cells = np.argwhere(obstacle_data['instance_id'] == impulses[action]['target_obstacle_id'])
        cell_m = geometry.cell_m
        centroid = [geometry.origin_x + float(cells[:, 1].mean())*cell_m,
                    float(cells[:, 0].mean())*cell_m]
        origin = geometry.car_pos
        direction = [centroid[0]-origin[0], centroid[1]-origin[1]]
        length = math.hypot(*direction)
        speed = math.hypot(*velocity[:2])
        centroid_velocity = [speed*direction[0]/length, speed*direction[1]/length, 0.0]
        variants += [
            {'name': f'obstacle_centroid_{scale:.2f}', 'parameters': {'contactERP':0.025},
             'velocity':centroid_velocity, 'speed_scale':scale} for scale in (1.0, 1.1, 1.2)]
    target_id = impulses[action]['target_obstacle_id']
    results = [run_case(geometry, settings, velocity, params['seconds'], v, target_id)
               for v in variants]
    baseline = results[0]
    reproduced = abs(baseline['max_step_specific_energy_gain_j_kg']-saved_gain) <= 1e-9
    print(json.dumps({'action': action, 'saved_baseline_gain_j_kg': saved_gain,
                      'saved_baseline_peak_frame': saved_frame,
                      'saved_baseline_reproduced': reproduced,
                      'variants': results}, indent=2))
    if not reproduced:
        raise SystemExit('saved solver settings did not reproduce baseline; do not interpret sweep')


if __name__ == '__main__':
    main()
