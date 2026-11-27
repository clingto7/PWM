import unittest
import numpy as np

from car_fit import eligible_prior, raster_depth, support_gap


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

    def test_prior_needs_semantic_classifier_evidence_above_frozen_gate(self):
        self.assertFalse(eligible_prior({'class': 'car', 'confidence': .99, 'source': 'classifier'}))
        self.assertFalse(eligible_prior({'class': 'mini_cooper_s_convertible_2021', 'confidence': .89, 'source': 'classifier'}))
        self.assertFalse(eligible_prior({'class': 'mini_cooper_s_convertible_2021', 'confidence': .99, 'source': 'manual'}))
        self.assertTrue(eligible_prior({'class': 'mini_cooper_s_convertible_2021', 'confidence': .9, 'source': 'classifier'}))

    def test_contact_gap_reads_reconstructed_support_height(self):
        bottom = np.array([[0, 0, 0], [.1, 0, 0]])
        support = (np.array([[.12]]), np.array([[True]]), 0, 1)
        self.assertAlmostEqual(support_gap(bottom, (1, 0, 0, 0), 0, support), -.12)


if __name__ == '__main__':
    unittest.main()
