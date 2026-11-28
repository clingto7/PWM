"""Public physics adapter trial rejects unsupported initial placement."""
import unittest

from physics import PyBulletSolver


class PassiveWorld:
    def __init__(self, gap, final):
        self.gap = gap
        self.final = final
        self.steps = 0

    def getBasePositionAndOrientation(self, body):
        return ((0, 0, 1) if self.steps == 0 else self.final, (0, 0, 0, 1))

    def getClosestPoints(self, body, support, distance):
        return [(None,) * 8 + (self.gap,)]

    def getContactPoints(self, bodyA):
        return []

    def stepSimulation(self):
        self.steps += 1


class FitTrialTest(unittest.TestCase):
    def test_stable_contact_is_trusted(self):
        result = PyBulletSolver(PassiveWorld(.01, (0, 0, .99))).trial_without_impulse(1, 2)
        self.assertTrue(result['trusted'])

    def test_floating_body_is_untrusted(self):
        result = PyBulletSolver(PassiveWorld(.4, (0, 0, .6))).trial_without_impulse(1, 2)
        self.assertFalse(result['trusted'])
        self.assertEqual(result['steps'], 120)


if __name__ == '__main__':
    unittest.main()
