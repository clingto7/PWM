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
    def test_saved_depth_selects_small_upper_subject_and_records_contact(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/'image.jpg'
            Image.new('RGB', (256, 256)).save(image)
            depth = np.full((1, 256, 256), 12, dtype=np.float32)
            depth[0, 45:85, 100:140] = 4
            depth[0, 85:91, 100:140] = 4.4
            depth[0, 170:235, 80:165] = 9
            da3 = root/'depth'; da3.mkdir()
            np.savez_compressed(da3/'prediction.npz', depth=depth, intrinsics=np.eye(3)[None])
            (da3/'run.json').write_text(json.dumps({'is_metric': True}))
            masks = np.zeros((2, 256, 256), dtype=bool)
            masks[0, 45:85, 100:140] = True
            masks[1, 170:235, 80:165] = True
            sam = root/'sam'; sam.mkdir()
            np.savez_compressed(sam/'sam2_masks.npz', masks_256=masks,
                points_xy=np.array([[120, 60], [120, 200]]), scores=np.array([.95, .95]), selected=np.array([0, 1]))
            (sam/'run.json').write_text('{}')
            out = root/'observation'
            result = subprocess.run([sys.executable, str(CLI), str(image), str(out),
                '--depth-source', str(da3), '--segmentation-source', str(sam)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            selection = json.loads((out/'selection.json').read_text())
            self.assertEqual(selection['selected_index'], 0)
            self.assertIn('relative_depth', selection['candidates'][0])
            self.assertIn('contact', selection['candidates'][0])

    def test_unsupported_foreground_abstains_before_mesh(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/'image.jpg'; Image.new('RGB', (256, 256)).save(image)
            depth = np.full((1, 256, 256), 20, dtype=np.float32)
            depth[0, 30:65, 90:130] = 3
            da3 = root/'depth'; da3.mkdir()
            np.savez_compressed(da3/'prediction.npz', depth=depth, intrinsics=np.eye(3)[None])
            (da3/'run.json').write_text(json.dumps({'is_metric': True}))
            mask = np.zeros((1, 256, 256), dtype=bool); mask[0, 30:65, 90:130] = True
            sam = root/'sam'; sam.mkdir()
            np.savez_compressed(sam/'sam2_masks.npz', masks_256=mask,
                points_xy=np.array([[110, 45]]), scores=np.array([.99]), selected=np.array([0]))
            (sam/'run.json').write_text('{}')
            out = root/'observation'
            result = subprocess.run([sys.executable, str(CLI), str(image), str(out),
                '--depth-source', str(da3), '--segmentation-source', str(sam)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            selection = json.loads((out/'selection.json').read_text())
            self.assertIsNone(selection['selected_index'])
            self.assertIn('contact', selection['reason'])
            self.assertFalse((out/'mesh').exists())

    def test_distinct_supported_tie_abstains(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/'image.jpg'; Image.new('RGB', (256, 256)).save(image)
            depth = np.full((1, 256, 256), 12, dtype=np.float32)
            masks = np.zeros((2, 256, 256), dtype=bool)
            for index, left in enumerate((50, 166)):
                masks[index, 60:100, left:left+40] = True
                depth[0, 60:100, left:left+40] = 4
                depth[0, 100:106, left:left+40] = 4.4
            da3 = root/'depth'; da3.mkdir()
            np.savez_compressed(da3/'prediction.npz', depth=depth, intrinsics=np.eye(3)[None])
            (da3/'run.json').write_text(json.dumps({'is_metric': True}))
            sam = root/'sam'; sam.mkdir()
            np.savez_compressed(sam/'sam2_masks.npz', masks_256=masks,
                points_xy=np.array([[70, 80], [186, 80]]), scores=np.array([.95, .95]), selected=np.array([0, 1]))
            (sam/'run.json').write_text('{}')
            out = root/'observation'
            result = subprocess.run([sys.executable, str(CLI), str(image), str(out),
                '--depth-source', str(da3), '--segmentation-source', str(sam)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            selection = json.loads((out/'selection.json').read_text())
            self.assertIsNone(selection['selected_index'])
            self.assertIn('near-tied', selection['reason'])
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
            self.assertIn(evidence['selected_index'], [c['index'] for c in evidence['candidates']])
            self.assertTrue(next(c for c in evidence['candidates'] if c['index'] == evidence['selected_index'])['contact']['supported'])
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
