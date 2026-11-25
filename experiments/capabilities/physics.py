"""Independent PyBullet heightfield, convex body, static obstacle and replay probe."""
import hashlib
import argparse
from importlib.metadata import version
import json
import platform
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    import pybullet as p

    started = time.perf_counter()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    def run():
        client = p.connect(p.DIRECT)
        try:
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1 / 240)
            p.setPhysicsEngineParameter(fixedTimeStep=1 / 240, numSolverIterations=100)
            heights = [0.0] * (16 * 16)
            ground_shape = p.createCollisionShape(p.GEOM_HEIGHTFIELD, meshScale=[0.25, 0.25, 1], heightfieldTextureScaling=1, heightfieldData=heights, numHeightfieldRows=16, numHeightfieldColumns=16)
            ground = p.createMultiBody(0, ground_shape)
            obstacle_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
            obstacle = p.createMultiBody(0, obstacle_shape, basePosition=[1.5, 1.5, 0.5])
            vertices = [[x, y, z] for x in (-0.2, 0.2) for y in (-0.2, 0.2) for z in (-0.2, 0.2)]
            shape = p.createCollisionShape(p.GEOM_MESH, vertices=vertices)
            body = p.createMultiBody(1, shape, basePosition=[0, 0, 1])
            trajectory = []
            resting_contact_distances = []
            for step in range(1200):
                p.stepSimulation()
                pos, quat = p.getBasePositionAndOrientation(body)
                vel, angular = p.getBaseVelocity(body)
                trajectory.append([*pos, *quat, *vel, *angular])
                if step >= 960:
                    resting_contact_distances.extend(c[8] for c in p.getContactPoints(bodyA=body, bodyB=ground))
            static_before = p.getBasePositionAndOrientation(obstacle)[0]
            p.applyExternalForce(obstacle, -1, [1000, 0, 0], [0, 0, 0], p.WORLD_FRAME)
            p.stepSimulation()
            static_after = p.getBasePositionAndOrientation(obstacle)[0]
            contacts = p.getContactPoints(bodyA=body, bodyB=ground)
            return trajectory, static_before == static_after, len(contacts) > 0, min(resting_contact_distances) if resting_contact_distances else None
        finally:
            p.disconnect(client)

    first, stationary, contact, min_contact_distance = run()
    second, _, _, _ = run()
    data = json.dumps(first, separators=(",", ":")).encode()
    path = out / "trajectory.json"
    path.write_bytes(data)
    final = first[-240:]
    drift = max(max(abs(row[i] - final[0][i]) for i in range(3)) for row in final)
    speed = max(sum(row[i] ** 2 for i in range(7, 10)) ** 0.5 for row in final)
    expected_resting_height = 0.2  # Flat heightfield at z=0; hull half-height is 0.2 m.
    resting_height_tolerance = 0.01
    penetration_tolerance = 0.01
    resting_height_error = max(abs(row[2] - expected_resting_height) for row in final)
    penetration = max(0.0, -min_contact_distance) if min_contact_distance is not None else None
    result = {"command": [sys.executable, str(Path(__file__).resolve()), str(out.resolve())], "python": platform.python_version(), "pybullet_package_version": version("pybullet"), "pybullet_api_version": p.getAPIVersion(), "seconds": time.perf_counter() - started, "outputs": {"trajectory.json": hashlib.sha256(data).hexdigest()}, "replay_identical": first == second, "static_immobile": stationary, "ground_contact": contact, "expected_resting_height_m": expected_resting_height, "resting_height_error_m": resting_height_error, "resting_height_tolerance_m": resting_height_tolerance, "minimum_ground_contact_distance_m": min_contact_distance, "max_ground_penetration_m": penetration, "ground_penetration_tolerance_m": penetration_tolerance, "final_height": first[-1][2], "final_window_drift_m": drift, "final_window_max_speed_m_s": speed, "passed": stationary and contact and first == second and drift < 0.001 and speed < 0.001 and resting_height_error <= resting_height_tolerance and penetration is not None and penetration <= penetration_tolerance}
    (out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
