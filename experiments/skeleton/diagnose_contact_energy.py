"""Compare Bullet contact settings on one saved scene without reconstruction.

Run with the pinned physics environment:
  python experiments/skeleton/diagnose_contact_energy.py RUN_DIRECTORY
The saved scene is read only; results are printed as JSON.
"""
import argparse
import json
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


def run_case(geometry, settings, velocity, seconds, variant):
    import pybullet as p
    client = p.connect(p.DIRECT)
    try:
        solver = PyBulletSolver(p)
        ground, body, _ = solver.create_world(geometry, settings, Path(settings.root))
        p.setPhysicsEngineParameter(**variant['parameters'])
        effective_parameters = p.getPhysicsEngineParameters()
        inertia = p.getDynamicsInfo(body, -1)[2]
        mass = geometry.mass
        p.resetBaseVelocity(body, linearVelocity=velocity)
        rows = []
        for frame in range(int(seconds*solver.hz)):
            if frame:
                p.stepSimulation()
            pos, quat = p.getBasePositionAndOrientation(body)
            vel, omega = p.getBaseVelocity(body)
            row = [*pos, *quat, *vel, *omega]
            contacts = p.getContactPoints(bodyA=body)
            rows.append((specific_energy(row, inertia, mass), row,
                         min((float(c[8]) for c in contacts), default=None), len(contacts)))
        # Match check.py: omit stationary frame, then compare consecutive steps.
        gains = [(rows[i][0]-rows[i-1][0], i) for i in range(2, len(rows))]
        gain, frame = max(gains)
        before, after = rows[frame-1], rows[frame]
        return {'variant': variant['name'], 'parameters': variant['parameters'],
                'effective_engine_parameters': effective_parameters,
                'max_step_specific_energy_gain_j_kg': max(0., gain),
                'peak_frame': frame, 'peak_time_s': frame/solver.hz,
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
    args = parser.parse_args()
    root = args.run_directory
    geometry = SceneArtifacts(root).load()
    run = json.loads((root/'run.json').read_text())
    impulses = json.loads((root/'impulses.json').read_text())
    params = run['parameters']
    settings = SimpleNamespace(root=str(root), friction=params['friction'],
                               radius=params['radius_m'])
    velocity = impulses['ballistic']['delta_velocity_m_s']
    saved_rows = json.loads((root/'ballistic'/'trajectory.json').read_text())
    inertia = run['impulse_cases']['ballistic']['inertia_diagonal_kg_m2']
    saved_energy = [specific_energy(row, inertia, geometry.mass) for row in saved_rows]
    saved_gain, saved_frame = max((saved_energy[i]-saved_energy[i-1], i)
                                  for i in range(2, len(saved_energy)))
    variants = [
        {'name': 'unsplit', 'parameters': {'useSplitImpulse': 0}},
        {'name': 'current_split', 'parameters': {'useSplitImpulse': 1,
                                                'splitImpulsePenetrationThreshold': 0.0}},
        {'name': 'split_default_threshold', 'parameters': {'useSplitImpulse': 1,
                                                           'splitImpulsePenetrationThreshold': -0.04}},
        {'name': 'split_low_erp', 'parameters': {'useSplitImpulse': 1,
                                               'splitImpulsePenetrationThreshold': 0.0,
                                               'contactERP': 0.1}},
    ]
    results = [run_case(geometry, settings, velocity, params['seconds'], v)
               for v in variants]
    current = next(r for r in results if r['variant'] == 'current_split')
    reproduced = abs(current['max_step_specific_energy_gain_j_kg']-saved_gain) <= 1e-9
    print(json.dumps({'saved_baseline_gain_j_kg': saved_gain,
                      'saved_baseline_peak_frame': saved_frame,
                      'current_split_reproduces_saved_baseline': reproduced,
                      'variants': results}, indent=2))
    if not reproduced:
        raise SystemExit('current solver settings did not reproduce saved baseline; do not interpret sweep')


if __name__ == '__main__':
    main()
