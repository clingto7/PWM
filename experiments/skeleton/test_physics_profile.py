"""Focused checks for the disjoint solve-loop profiling contract."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from physics import PyBulletSolver


class TimedBullet:
    """Deterministic API stub; delays make each measured operation observable."""
    def getBasePositionAndOrientation(self, body):
        return ((0., 0., 0.), (0., 0., 0., 1.))

    def getBaseVelocity(self, body):
        return ((0., 0., 0.), (0., 0., 0.))

    def resetBaseVelocity(self, body, linearVelocity):
        pass

    def stepSimulation(self):
        pass

    def getContactPoints(self, bodyA):
        return []


class PhysicsProfileTest(unittest.TestCase):
    def test_solve_timing_buckets_are_disjoint_and_cover_operations(self):
        solver = PyBulletSolver(TimedBullet())
        rows, penetration, hits = solver.solve(3, {}, [0., 1., 0.], 5/240)

        self.assertEqual(len(rows), 5)
        self.assertEqual(penetration, 0.)
        self.assertEqual(hits, {})
        timing = solver._last_solve_timings
        self.assertEqual(set(timing), {'stepping', 'state_reads', 'contact_queries'})
        self.assertTrue(all(value >= 0 for value in timing.values()))
        self.assertGreater(timing['stepping'], 0.)
        self.assertGreater(timing['state_reads'], 0.)
        self.assertGreater(timing['contact_queries'], 0.)
        self.assertLessEqual(sum(timing.values()), solver._last_solve_elapsed)
