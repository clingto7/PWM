"""Saved physics/check seam: oriented collision hull on local support."""
import unittest
import numpy as np
from rest_support import hull_support_clearance


class RestSupportTest(unittest.TestCase):
    def test_rotated_hull_on_local_slope_passes_existing_tolerance(self):
        heights=np.tile(np.arange(6)*.01,(6,1))
        valid=np.ones((6,6),bool)
        hull=np.array([[-.5,0,-.5],[.5,0,-.5],[-.5,0,.5],[.5,0,.5]])
        # 90 degree yaw swaps the hull's horizontal extent; a flat-plane,
        # unrotated half-box comparison reports a false 12 mm miss.
        row=[.5,2.,.535,0,0,2**-.5,2**-.5]
        clearance=hull_support_clearance(row,hull,heights,valid,1.)
        self.assertIsNotNone(clearance)
        self.assertLess(abs(clearance),.01)
        self.assertGreater(abs(row[2]-.5),.01)


if __name__=='__main__':unittest.main()
