"""Image-wide support selection for the scene stage."""
import unittest
import numpy as np
from scene_support import select_support_plane


class SupportSelectionTest(unittest.TestCase):
    def test_support_above_midpoint_beats_lower_foreground(self):
        h,w=72,96
        y,x=np.indices((h,w))
        xx=(x-48)*.06
        yy=8+y*.08
        zz=np.where(y<34, -.3*xx+.12*yy-3, 1.5+.02*x+.01*y)
        subject=np.zeros((h,w),bool)
        subject[20:30,43:55]=True
        coeff, support, rms=select_support_plane(xx,yy,zz,~subject,subject,.12)
        self.assertGreater(np.count_nonzero(support[:34]),1000)
        self.assertLess(np.count_nonzero(support[42:]),10)
        self.assertAlmostEqual(coeff[0],-.3,places=2)
        self.assertAlmostEqual(coeff[1],.12,places=2)
        self.assertLess(rms,.02)


if __name__=='__main__': unittest.main()
