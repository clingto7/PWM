"""Portable center-of-mass impulse actions, independent of any solver."""
import math


def action_records(velocities, mass, initial_position):
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('dynamic body mass must be positive and finite')
    result = {}
    for name, vector in velocities.items():
        if not isinstance(name, str) or not name or '/' in name or name in ('.', '..'):
            raise ValueError(f'invalid action name: {name}')
        if len(vector) != 3 or not all(math.isfinite(float(v)) for v in vector):
            raise ValueError(f'action {name}: expected finite 3D delta velocity')
        values = [float(v) for v in vector]
        speed = math.sqrt(sum(v*v for v in values))
        if speed <= 0:
            raise ValueError(f'action {name}: zero impulse')
        result[name] = {'delta_velocity_m_s': values,
                        'magnitude_n_s': mass * speed,
                        'direction': [v/speed for v in values],
                        'application_point': 'center_of_mass',
                        'application_point_world_m': list(initial_position),
                        'target_obstacle_id': None}
    if not result:
        raise ValueError('at least one action is required')
    return result
