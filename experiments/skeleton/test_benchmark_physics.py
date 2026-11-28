import unittest

from benchmark_physics import (boundaries_equivalent, merge_collinear_segments,
                               percentile)


class BenchmarkPhysicsTest(unittest.TestCase):
    def test_merges_contiguous_collinear_edges_without_gaps(self):
        source=[((0,0),(1,0)),((1,0),(2,0)),((2,0),(3,0)),
                ((3,0),(3,1))]
        merged=merge_collinear_segments(source)
        self.assertEqual(len(merged),2)
        self.assertTrue(boundaries_equivalent(source,merged))

    def test_keeps_disconnected_collinear_edges_separate(self):
        source=[((0,0),(1,0)),((2,0),(3,0))]
        merged=merge_collinear_segments(source)
        self.assertEqual(len(merged),2)
        self.assertTrue(boundaries_equivalent(source,merged))

    def test_step_p95_uses_nearest_rank(self):
        self.assertEqual(percentile([4,1,3,2],.5),2)
        self.assertEqual(percentile([4,1,3,2],.95),4)


if __name__=='__main__':
    unittest.main()
