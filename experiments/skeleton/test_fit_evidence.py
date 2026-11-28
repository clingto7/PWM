import unittest
import numpy as np
from car_fit import raster_depth, support_gap, visible_fit_score


class FitEvidenceTest(unittest.TestCase):
    def test_triangle_z_buffer_reports_visible_pixel_depth(self):
        # Two overlapping camera-facing triangles: the nearer one must win.
        near = np.array([[-1, 2, 1], [1, 2, 1], [0, 2, -1]], dtype=float)
        far = near.copy(); far[:, 1] = 4
        vertices = np.vstack([far, near])
        k = np.array([[4, 0, 4], [0, 4, 4], [0, 0, 1]], dtype=float)
        depth = raster_depth(vertices, np.array([[0, 1, 2], [3, 4, 5]]), k,
                             np.array([1, 0, 0]), np.array([0, 1, 0]),
                             np.array([0, 0, 1]), 1, 0, 0, 0, 0, (8, 8))
        self.assertAlmostEqual(depth[4, 4], 2)
        self.assertTrue(np.isinf(depth[0, 0]))

    def test_contact_gap_reads_reconstructed_support_height(self):
        bottom = np.array([[0, 0, 0], [.1, 0, 0]])
        support = (np.array([[.12]]), np.array([[True]]), 0, 1)
        self.assertAlmostEqual(support_gap(bottom, (1, 0, 0, 0), 0, support), -.12)

    def test_visible_candidate_prefers_depth_evidence_when_silhouettes_match(self):
        close = visible_fit_score(.85, .2, .02, 3, 3)
        far = visible_fit_score(.85, .6, .02, 3, 3)
        self.assertLess(close, far)
        self.assertEqual(visible_fit_score(.85, None, .02, 3, 3), float('inf'))


if __name__ == '__main__':
    unittest.main()
