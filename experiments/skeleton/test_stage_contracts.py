"""Saved stage artifact compatibility seam."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CONTRACT = Path(__file__).with_name('stage_contracts.py')


class StageContractCliTest(unittest.TestCase):
    def test_saved_stage_validates_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'input.bin').write_bytes(b'observation')
            (root/'output.bin').write_bytes(b'scene')
            create = subprocess.run([sys.executable, str(CONTRACT), 'record', str(root),
                'scene', '--input', 'input.bin', '--output', 'output.bin',
                '--implementation', 'first_image', '--version', '1', '--seconds', '1.25'],
                capture_output=True, text=True)
            self.assertEqual(create.returncode, 0, create.stderr)
            manifest = json.loads((root/'stages/scene.json').read_text())
            self.assertEqual(manifest['schema'], 1)
            self.assertEqual(manifest['seconds'], 1.25)
            valid = subprocess.run([sys.executable, str(CONTRACT), 'check', str(root), 'scene'], capture_output=True)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            (root/'output.bin').write_bytes(b'altered')
            invalid = subprocess.run([sys.executable, str(CONTRACT), 'check', str(root), 'scene'], capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)


if __name__ == '__main__':
    unittest.main()

class ResumeAndContactCliTest(unittest.TestCase):
    def test_saved_scene_rejects_changed_fit_evidence_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'scene.json').write_text('{}')
            from stage_contracts import record, validate_resume
            record(root, 'scene', ['scene.json'], ['scene.json'], 'fit', '1',
                   {'fit_evidence': 'image_depth_support'}, 0)
            self.assertEqual(validate_resume(root, {'fit_evidence': 'image_depth_support'})
                             ['parameters']['fit_evidence'], 'image_depth_support')
            with self.assertRaisesRegex(ValueError, 'fit_evidence'):
                validate_resume(root, {'fit_evidence': 'changed'})

    def test_saved_scene_rejects_changed_geometry_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'scene.json').write_text('{}')
            subprocess.run([sys.executable, str(CONTRACT), 'record', str(root), 'scene',
                '--input', 'scene.json', '--output', 'scene.json',
                '--implementation', 'first_image', '--version', '1', '--seconds', '1',
                '--parameters', '{"cell_m": 0.5, "radius_m": 27}'], check=True, capture_output=True)
            result = subprocess.run([sys.executable, str(CONTRACT), 'check-resume', str(root),
                '--parameters', '{"cell_m": 1.0, "radius_m": 27}'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('cell_m', result.stderr)
            compatible = subprocess.run([sys.executable, str(CONTRACT), 'check-resume', str(root),
                '--parameters', '{"cell_m": 0.5, "radius_m": 27}'], capture_output=True)
            self.assertEqual(compatible.returncode, 0, compatible.stderr)

    def test_physics_contract_requires_contact_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'scene.json').write_text('{}')
            (root/'trajectory.json').write_text('[]')
            (root/'contacts.json').write_text('{"schema": 1, "cases": {}}')
            subprocess.run([sys.executable, str(CONTRACT), 'record', str(root), 'physics',
                '--input', 'scene.json', '--output', 'trajectory.json',
                '--output', 'contacts.json', '--implementation', 'PyBulletSolver',
                '--version', '1', '--seconds', '1'], check=True, capture_output=True)
            self.assertEqual(subprocess.run([sys.executable, str(CONTRACT), 'check', str(root),
                'physics'], capture_output=True).returncode, 0)
            (root/'contacts.json').unlink()
            self.assertEqual(subprocess.run([sys.executable, str(CONTRACT), 'check', str(root),
                'physics'], capture_output=True).returncode, 2)
