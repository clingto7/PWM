import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CLI = Path(__file__).with_name('evaluation.py')


class EvaluationCliTest(unittest.TestCase):
    def test_every_eligible_missing_run_counts_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = {'schema': 1, 'games': [
                {'id': f'{split}-{i}', 'split': split, 'scenes': [
                    {'id': f's{j}', 'image': f'{j}.png', 'annotation': f'{j}.json',
                     'eligible': True, 'run': str(root/'missing')} for j in range(5)]}
                for split in ('development', 'holdout') for i in range(3)]}
            freeze = {'schema': 1, 'algorithm_revision': 'abc', 'config_sha256': '123',
                      'failure_conditions': ['pipeline failure'],
                      'thresholds': {'detection_iou_min': .5, 'geometry_depth_mae_m_max': 1,
                                     'contact_penetration_m_max': .01, 'render_iou_min': .5,
                                     'total_seconds_max': 600}, 'aggregate_success_min': .8}
            (root/'dataset.json').write_text(json.dumps(dataset))
            (root/'freeze.json').write_text(json.dumps(freeze))
            result = subprocess.run([sys.executable, str(CLI), str(root/'dataset.json'),
                                     '--freeze', str(root/'freeze.json'), '--split', 'holdout'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report['eligible'], 15)
            self.assertEqual(report['successes'], 0)
            self.assertEqual(len(report['scenes']), 15)
            self.assertFalse(report['target_met'])
