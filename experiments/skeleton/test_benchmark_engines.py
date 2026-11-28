import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_engines import box_proxy, parse_actions, trajectory_metrics


class BoxProxyTests(unittest.TestCase):
    def test_scene_converts_to_identical_axis_aligned_proxy_data(self):
        scene = {"ground": {"flat_world_z_m": 1.25},
                 "car": {"size_m": [2, 4, 1], "mass_kg": 800,
                         "initial_position_m": [1, 2, 3],
                         "initial_orientation_xyzw": [0, 0, 0, 1]},
                 "obstacles": {"components": [
                     {"instance_id": 8, "collision_enabled": True},
                     {"instance_id": 9, "collision_enabled": False}]}}
        meshes = [([[0, 1, 2], [2, 5, 6], [1, 4, 3]], []), ([], [])]
        proxy = box_proxy(scene, meshes)
        self.assertEqual(proxy["ground_z"], 1.25)
        self.assertEqual(proxy["car"]["size"], [2, 4, 1])
        self.assertEqual(proxy["car"]["mass"], 800)
        self.assertEqual(proxy["obstacles"], [{"instance_id": 8, "center": [1, 3, 4], "size": [2, 4, 4]}])

    def test_rejects_bad_scene_proxy_inputs(self):
        scene = {"ground": {"flat_world_z_m": 0},
                 "car": {"size_m": [1, 0, 1]}, "obstacles": {"components": []}}
        with self.assertRaisesRegex(ValueError, "car size"):
            box_proxy(scene, [])

    def test_skips_collision_disabled_obstacle(self):
        scene = {"ground": {"flat_world_z_m": 0},
                 "car": {"size_m": [1, 1, 1], "mass_kg": 1,
                         "initial_position_m": [0, 0, 1], "initial_orientation_xyzw": [0, 0, 0, 1]},
                 "obstacles": {"components": [{"instance_id": 2, "collision_enabled": False}]}}
        self.assertEqual(box_proxy(scene, [([], [])])["obstacles"], [])


class BenchmarkMetricTests(unittest.TestCase):
    def test_metrics_report_penetration_contact_rest_and_trajectory(self):
        rows = [[0, 0, 0], [0, 0, 1], [0, 0, 2]]
        m = trajectory_metrics(rows, .02, 3, [[0, 0, 0], [0, 0, 1.5], [0, 0, 2]])
        self.assertEqual(m["max_penetration_m"], .02)
        self.assertEqual(m["contact_steps"], 3)
        self.assertAlmostEqual(m["trajectory_rms_difference_m"], (.25/3)**.5)
        self.assertEqual(m["trajectory_max_difference_m"], .5)
        self.assertEqual(m["final_position_m"], [0, 0, 2])

    def test_action_parser_requires_finite_three_vector(self):
        self.assertEqual(parse_actions(["push=1,2,3"]), {"push": [1., 2., 3.]})
        with self.assertRaisesRegex(ValueError, "invalid"):
            parse_actions(["push=1,nan,3"])


if __name__ == "__main__":
    unittest.main()
