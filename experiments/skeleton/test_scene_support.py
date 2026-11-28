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

class SupportHeightTest(unittest.TestCase):
    def test_isolated_depth_spike_is_removed_without_flattening_slope(self):
        from scene_support import regularize_support_heights
        y,x=np.indices((7,7))
        heights=1+.01*x+.02*y
        heights[3,3]+=.05
        observed=np.ones((7,7),bool)
        result=regularize_support_heights(heights,observed)
        self.assertAlmostEqual(result[3,3],1.09,places=2)
        self.assertAlmostEqual(result[1,1],1.03,places=3)
        self.assertAlmostEqual(result[5,5],1.15,places=3)
