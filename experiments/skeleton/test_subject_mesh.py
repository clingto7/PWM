"""Saved mesh artifact and observation-to-mesh CLI seam."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = Path(__file__).with_name('subject_mesh.py')


class SubjectMeshCliTest(unittest.TestCase):
    def test_saved_mesh_produces_checked_artifact_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'mesh'
            command = [sys.executable, str(CLI), str(ROOT/'first_image/input.jpg'),
                       str(ROOT/'first_image/car_input/car_mask.png'), str(out),
                       '--saved-mesh', str(ROOT/'first_image/hunyuan21_output/car_shape.glb')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            artifact = json.loads((out/'subject_mesh.json').read_text())
            self.assertEqual(artifact['schema'], 1)
            self.assertEqual(artifact['validity']['valid'], True)
            self.assertEqual(artifact['units'], 'normalized_model_units')
            self.assertIn('orientation', artifact)
            self.assertIn('coordinate_frame', artifact)
            self.assertIn('original_image_sha256', artifact['input_hashes'])
            self.assertIn('selected_mask_sha256', artifact['input_hashes'])
            checked = subprocess.run([sys.executable, str(CLI), 'check', str(out)], capture_output=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_invalid_saved_mesh_preserves_failure_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root/'bad.glb'
            bad.write_bytes(b'bad mesh')
            out = root/'mesh'
            result = subprocess.run([sys.executable, str(CLI), str(ROOT/'first_image/input.jpg'),
                str(ROOT/'first_image/car_input/car_mask.png'), str(out), '--saved-mesh', str(bad)],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            status = json.loads((out/'stage_status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['stage'], 'subject_mesh')
            self.assertTrue((out/'inputs/input.jpg').is_file())
            self.assertTrue((out/'inputs/car_mask.png').is_file())


if __name__ == '__main__':
    unittest.main()
