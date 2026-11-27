import json
import hashlib
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
                      'thresholds': {'detection_iou_min': .5, 'geometry_contact_fit_rms_m_max': 1,
                                     'contact_penetration_m_max': .01, 'render_iou_min': .5,
                                     'total_seconds_max': 600}, 'aggregate_success_min': .8}
            canonical = json.dumps(freeze, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            freeze['sealed_sha256'] = hashlib.sha256(canonical).hexdigest()
            dataset['freeze_sha256'] = freeze['sealed_sha256']
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


class ScoreRunTest(unittest.TestCase):
    def test_saved_projection_and_semantic_metrics_produce_scored_success(self):
        from unittest.mock import patch
        import importlib.util
        spec = importlib.util.spec_from_file_location('evaluation', CLI)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'chain'
            run = root/'automatic/run'
            image = root/'automatic/observation/inputs/input.jpg'
            run.mkdir(parents=True)
            image.parent.mkdir(parents=True)
            image.write_bytes(b'image')
            (root/'chain.json').write_text(json.dumps({'inputs': {'config': 'config-hash'},
                'timings_s': {'total': 12}}))
            (run/'bundle.json').write_text(json.dumps({'revision': 'abc'}))
            (run/'initial_projection.json').write_text(json.dumps({
                'reference_runs_y_x0_x1': [[0, 0, 2]],
                'rendered_runs_y_x0_x1': [[0, 0, 1]]}))
            annotation = Path(tmp)/'annotation.json'
            annotation.write_text(json.dumps({'schema': 1, 'subject_runs_y_x0_x1': [[0, 0, 2]]}))
            checked = {'status': 'complete', 'semantic_check': {'metrics': {
                'contact_fit_rms_m': .02, 'max_penetration_m': .003}}}
            with patch('automatic_chain.check', return_value=checked):
                result = module.score_run(root, annotation, Path(tmp)/'scored')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['metrics']['detection_iou'], 1)
            self.assertEqual(result['metrics']['render_iou'], .5)
            self.assertEqual(result['metrics']['geometry_contact_fit_rms_m'], .02)
            self.assertEqual(result['provenance']['revision'], 'abc')
            self.assertTrue((Path(tmp)/'scored/evaluation.json').is_file())

    def test_missing_chain_or_annotation_still_writes_failure_record(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('evaluation', CLI)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root/'score'
            result = module.score_run(root/'missing-chain', root/'missing-annotation.json', out)
            self.assertEqual(result['status'], 'failed')
            self.assertTrue((out/'evaluation.json').is_file())
            self.assertIn('reason', json.loads((out/'evaluation.json').read_text()))

    def test_stale_annotation_is_counted_as_failure(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('evaluation', CLI)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/'image.png'; image.write_bytes(b'image')
            annotation = root/'annotation.json'; annotation.write_text('{}')
            scored = root/'scored'; scored.mkdir()
            (scored/'evaluation.json').write_text(json.dumps({'status': 'complete',
                'provenance': {'revision': 'abc', 'config_sha256': '123',
                               'image_sha256': hashlib.sha256(b'image').hexdigest(),
                               'annotation_sha256': 'stale'},
                'metrics': {'detection_iou': 1, 'render_iou': 1,
                            'geometry_contact_fit_rms_m': 0, 'contact_penetration_m': 0,
                            'total_seconds': 1}}))
            freeze = {'schema': 1, 'algorithm_revision': 'abc', 'config_sha256': '123',
                      'failure_conditions': ['pipeline failure'],
                      'thresholds': {'detection_iou_min': .5, 'geometry_contact_fit_rms_m_max': 1,
                                     'contact_penetration_m_max': .01, 'render_iou_min': .5,
                                     'total_seconds_max': 600}, 'aggregate_success_min': .8}
            freeze['sealed_sha256'] = hashlib.sha256(json.dumps(freeze, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            dataset = {'schema': 1, 'freeze_sha256': freeze['sealed_sha256'], 'games': [
                {'id': f'{split}-{i}', 'split': split, 'scenes': [
                    {'id': f's{j}', 'image': str(image), 'annotation': str(annotation),
                     'eligible': True, 'run': str(scored)} for j in range(5)]}
                for split in ('development', 'holdout') for i in range(3)]}
            report = module.score(dataset, freeze, 'holdout')
            self.assertEqual(report['successes'], 0)
            self.assertIn('annotation differs', report['scenes'][0]['reason'])
