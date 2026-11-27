"""Solver-neutral action record seam."""
import unittest
from physics_actions import action_records


class ActionRecordTest(unittest.TestCase):
    def test_external_impulse_is_normalized_and_invalid_action_is_diagnostic(self):
        actions = action_records({'push': [3, 4, 0]}, 2, [1, 2, 3])
        self.assertEqual(actions['push']['magnitude_n_s'], 10)
        self.assertEqual(actions['push']['direction'], [0.6, 0.8, 0.0])
        with self.assertRaisesRegex(ValueError, 'action push'):
            action_records({'push': [0, 0, 0]}, 2, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
