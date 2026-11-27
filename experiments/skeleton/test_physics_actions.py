"""Solver-neutral action record seam."""
import unittest
from physics_actions import action_records, obstacle_launch_speed


class ActionRecordTest(unittest.TestCase):
    def test_default_obstacle_action_keeps_reach_margin_and_speed_cap(self):
        # Measured current-image scene: nominal point-mass estimate is below
        # the 6 m/s floor; a 6 m/s rigid-body action stopped short of contact.
        self.assertAlmostEqual(obstacle_launch_speed(.6,2.0662169425,6),6.9)
        self.assertEqual(obstacle_launch_speed(.6,100,6),15)

    def test_external_impulse_is_normalized_and_invalid_action_is_diagnostic(self):
        actions = action_records({'push': [3, 4, 0]}, 2, [1, 2, 3])
        self.assertEqual(actions['push']['magnitude_n_s'], 10)
        self.assertEqual(actions['push']['direction'], [0.6, 0.8, 0.0])
        with self.assertRaisesRegex(ValueError, 'action push'):
            action_records({'push': [0, 0, 0]}, 2, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
