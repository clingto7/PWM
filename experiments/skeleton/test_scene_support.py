"""Saved scene support contract across reconstruction and physics adapters."""
import unittest
import numpy as np
from scene_support import classify_support


class SupportContractTest(unittest.TestCase):
    def test_unknown_is_not_traversable_and_occlusion_is_distinct(self):
        observed=np.zeros((7,7),bool)
        observed[2:5,2:5]=True
        occluded=np.zeros((7,7),bool)
        occluded[3,3]=True
        observed[3,3]=False
        possible=np.ones((7,7),bool)
        z=np.full((7,7),2.0)
        z[2,2]=2.2
        provenance, traversable, heights=classify_support(observed,occluded,possible,z,2.0)
        self.assertEqual(provenance[3,3],2)  # interpolated
        self.assertEqual(provenance[0,0],0)  # unknown
        self.assertFalse(traversable[0,0])
        self.assertTrue(traversable[3,3])
        self.assertGreater(heights[2,2],heights[2,3])


if __name__=='__main__': unittest.main()
