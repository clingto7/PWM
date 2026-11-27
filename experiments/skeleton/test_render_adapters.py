"""Saved renderer CLI contract seam."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CLI = Path(__file__).with_name('render_adapters.py')


class RenderAdapterCliTest(unittest.TestCase):
    def test_tampered_upstream_fails_before_renderer_and_records_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'saved'
            root.mkdir()
            (root/'scene.json').write_text('{}')
            (root/'trajectory.json').write_text('[]')
            (root/'run.json').write_text(json.dumps({'outputs': {'scene.json': '0'*64}}))
            out = Path(tmp)/'render'
            process = subprocess.run([sys.executable, str(CLI), str(root), str(out),
                                      '--renderer', 'raster'], capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            result = json.loads((out/'render.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertIn('hash', result['reason'])
            self.assertEqual(list(out.iterdir()), [out/'render.json'])


if __name__ == '__main__':
    unittest.main()
