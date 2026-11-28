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

class SceneEvidenceTest(unittest.TestCase):
    def test_local_variation_is_retained_outside_planar_core(self):
        h,w=60,80
        y,x=np.indices((h,w))
        xx=(x-w/2)*.05; yy=5+y*.08
        zz=.1*yy-2 + np.where(x<30, .08, 0.)
        subject=np.zeros((h,w),bool);subject[25:35,67:73]=True
        coeff,support,rms=select_support_plane(xx,yy,zz,~subject,subject,.12)
        self.assertGreater(support.sum(),2500)
        self.assertGreater(rms,.02)
        from scene_support import contact_support_fit
        fit=contact_support_fit(xx,yy,zz,support,(1.5,7.,0.),2.)
        self.assertLess(fit['rms_m'],.01)
        self.assertGreater(fit['sample_count'],100)
        self.assertGreater(fit['coverage_fraction'],.05)

    def test_uncertain_small_candidates_do_not_get_entity_ids(self):
        from scene_support import static_components
        candidate=np.zeros((8,8),bool);candidate[1:3,1:3]=True;candidate[5,5:7]=True
        votes=np.full((8,8),3)
        ids,entities=static_components(candidate,votes,2,1)
        self.assertEqual(len(entities),1)
        self.assertEqual(entities[0]['instance_id'],3)
        self.assertTrue(np.all(ids[5,5:7]==0))
        self.assertTrue(np.all(candidate[5,5:7]))

    def test_ungrounded_major_component_remains_candidate(self):
        from scene_support import static_components
        candidate=np.zeros((10,10),bool)
        candidate[1:4,1:4]=True
        candidate[6:9,6:9]=True
        votes=np.full((10,10),5)
        support=np.zeros((10,10),bool);support[1:4,1:4]=True
        ids,entities=static_components(candidate,votes,2,1,support)
        self.assertEqual([entity['cells'] for entity in entities],[9])
        self.assertEqual(int(ids[7,7]),0)
        self.assertTrue(candidate[7,7])

    def test_single_supported_cell_cannot_promote_static_entity(self):
        from scene_support import static_components
        candidate=np.zeros((5,5),bool);candidate[1:4,1:4]=True
        votes=np.full((5,5),5)
        support=np.zeros((5,5),bool);support[2,2]=True
        ids,entities=static_components(candidate,votes,2,1,support)
        self.assertEqual(entities,[])
        self.assertFalse(ids.any())
