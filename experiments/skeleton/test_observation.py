"""Original-image observation and saved-stage compatibility seam."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = Path(__file__).with_name('observation.py')


class ObservationCliTest(unittest.TestCase):
    def test_distant_selected_mask_island_does_not_expand_mesh_crop(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = ROOT/'first_image/input.jpg'
            source = root/'sam'
            source.mkdir()
            mask = np.asarray(Image.open(ROOT/'first_image/car_input/car_mask.png').convert('L').resize(
                (256, 256), Image.Resampling.NEAREST)) > 0
            mask[180, 240] = True
            np.savez_compressed(source/'sam2_masks.npz', masks_256=mask[None],
                points_xy=np.array([[128, 180]]), scores=np.array([.99]), selected=np.array([0]))
            (source/'run.json').write_text(json.dumps({'model': 'synthetic SAM'}))
            out = root/'observation'
            result = subprocess.run([sys.executable, str(CLI), str(image), str(out),
                '--depth-source', str(ROOT/'first_image/da3nested_output'),
                '--segmentation-source', str(source),
                '--mesh', str(ROOT/'first_image/hunyuan21_output/car_shape.glb')],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads((out/'selection.json').read_text())
            cleanup = evidence['cleanup']
            self.assertGreater(cleanup['removed_pixels'], 0)
            self.assertEqual(cleanup['removed_components'], 1)
            self.assertGreater(cleanup['raw_bbox_xyxy'][2], 1300)
            self.assertLess(cleanup['clean_bbox_xyxy'][2], 1100)
            crop = json.loads((out/'mesh/subject_mesh.json').read_text())['provenance']['crop']
            self.assertLess(crop['crop_xyxy'][2] - crop['crop_xyxy'][0], 600)
            self.assertTrue((out/'raw_selected_mask.png').is_file())
            for stage in ('segmentation', 'observation', 'downstream_bundle'):
                checked = subprocess.run([sys.executable, str(CLI), 'check', str(out), stage], capture_output=True)
                self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_missing_image_preserves_original_failure_in_stage_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp)/'missing.jpg'
            out = Path(tmp)/'observation'
            result = subprocess.run([sys.executable, str(CLI), str(missing), str(out)],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            status = json.loads((out/'stage_status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['error_type'], 'FileNotFoundError')
            self.assertIn('missing.jpg', status['reason'])
            self.assertNotIn('stage_status.json', status['reason'])

    def test_image_only_produces_checked_observation_without_downstream_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'observation'
            result = subprocess.run([sys.executable, str(CLI), str(ROOT/'first_image/input.jpg'),
                str(out), '--depth-source', str(ROOT/'first_image/da3nested_output'),
                '--segmentation-source', str(ROOT/'first_image/sam21large_output')],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((out/'bundle.json').exists())
            self.assertFalse((out/'inputs/car_shape.glb').exists())
            self.assertEqual(json.loads((out/'stage_status.json').read_text())['status'], 'complete')
            for stage in ('depth', 'segmentation', 'observation'):
                checked = subprocess.run([sys.executable, str(CLI), 'check', str(out), stage], capture_output=True)
                self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_saved_models_produce_checked_bundle_and_selection_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'observation'
            result = subprocess.run([sys.executable, str(CLI), str(ROOT/'first_image/input.jpg'),
                str(out), '--depth-source', str(ROOT/'first_image/da3nested_output'),
                '--segmentation-source', str(ROOT/'first_image/sam21large_output'),
                '--mesh', str(ROOT/'first_image/hunyuan21_output/car_shape.glb')],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads((out/'selection.json').read_text())
            self.assertGreater(len(evidence['candidates']), 1)
            self.assertIn('score', evidence['candidates'][0])
            self.assertEqual(evidence['selected_index'], evidence['candidates'][0]['index'])
            import numpy as np
            from PIL import Image
            chosen = np.asarray(Image.open(out/'inputs/car_mask.png').convert('L').resize((256,256))) > 127
            reference = np.asarray(Image.open(ROOT/'first_image/car_input/car_mask.png').convert('L').resize((256,256))) > 127
            self.assertGreater((chosen & reference).sum() / (chosen | reference).sum(), .9)
            depth = json.loads((out/'depth_capability.json').read_text())
            self.assertEqual(depth['scale'], 'metric_m')
            self.assertTrue(depth['camera_intrinsics'])
            for stage in ('depth', 'segmentation', 'observation'):
                checked = subprocess.run([sys.executable, str(CLI), 'check', str(out), stage], capture_output=True)
                self.assertEqual(checked.returncode, 0, checked.stderr)
            segmentation = json.loads((out/'stages/segmentation.json').read_text())
            self.assertEqual(segmentation['parameters']['model'], 'facebook/sam2.1-hiera-large')
            bundle_check = subprocess.run([sys.executable, str(CLI), 'check', str(out),
                'downstream_bundle'], capture_output=True)
            self.assertEqual(bundle_check.returncode, 0, bundle_check.stderr)

    def test_relative_depth_fails_before_bundle_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'depth'
            source.mkdir()
            original = ROOT/'first_image/da3nested_output'
            (source/'prediction.npz').write_bytes((original/'prediction.npz').read_bytes())
            (source/'run.json').write_text(json.dumps({'is_metric': False, 'model': 'relative'}))
            out = Path(tmp)/'out'
            result = subprocess.run([sys.executable, str(CLI), str(ROOT/'first_image/input.jpg'),
                str(out), '--depth-source', str(source),
                '--segmentation-source', str(ROOT/'first_image/sam21large_output'),
                '--mesh', str(ROOT/'first_image/hunyuan21_output/car_shape.glb')],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('metric', (out/'stage_status.json').read_text())


if __name__ == '__main__':
    unittest.main()
