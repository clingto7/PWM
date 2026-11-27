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
