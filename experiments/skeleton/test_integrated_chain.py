"""Saved chain is checked through its public artifact boundary."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from integrated_chain import check_live_artifacts, check_chain, ledger, digest


class IntegratedChainTest(unittest.TestCase):
    def test_chain_rejects_tampered_ledger_and_failed_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ('observation/inputs', 'scene', 'live/0000'):
                (root/directory).mkdir(parents=True)
            (root/'observation/inputs/input.jpg').write_bytes(b'image')
            (root/'observation/selection.json').write_text(json.dumps({
                'selected_index': 1, 'candidates': [{'index': 1, 'contact': {'supported': True}},
                                                    {'index': 2, 'contact': {'supported': False}}]}))
            (root/'scene/run.json').write_text('{"parameters": {}}')
            (root/'scene/fit_diagnostics.json').write_text('{}')
            (root/'live/session.json').write_text(json.dumps({'inputs_unchanged': True,
                'scene_and_observation_sha256': {}}))
            (root/'live/steps.jsonl').write_text(json.dumps({'index': 0, 'state': [1],
                'contacts': {}, 'frame_paths': {'instance_id': '0000/instance_id.png',
                'depth': '0000/depth.png'}})+'\n')
            (root/'live/trajectory.json').write_text('[[0], [1]]')
            (root/'live/contacts.json').write_text('[{}]')
            (root/'live/timings.json').write_text(json.dumps({'initialization_s': 1,
                'steps': [{'physics_s': .1, 'raster_s': .2, 'writing_s': .1,
                           'response_latency_s': .4}]}))
            (root/'live/0000/instance_id.png').write_bytes(b'frame')
            (root/'live/0000/depth.png').write_bytes(b'depth')
            (root/'live/proxy.mp4').write_bytes(b'video')
            chain = {'schema': 1, 'status': 'complete',
                     'input_sha256': digest(root/'observation/inputs/input.jpg'),
                     'artifact_sha256': ledger(root)}
            (root/'chain.json').write_text(json.dumps(chain))
            with patch('integrated_chain.check_stage'), patch('integrated_chain.timing_evidence', return_value={}), \
                 patch('integrated_chain.check_saved_run', return_value={'passed': True}):
                self.assertEqual(check_chain(root)['rejected_indices'], [2])
                (root/'live/0000/instance_id.png').write_bytes(b'altered')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    check_chain(root)
                chain['status'] = 'failed'; chain['failure_stage'] = 'live'; chain['reason'] = 'bad frame'
                (root/'chain.json').write_text(json.dumps(chain))
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    check_chain(root)

    def test_step_frame_and_latency_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'0000').mkdir()
            (root/'0000/instance_id.png').write_bytes(b'frame')
            (root/'0000/depth.png').write_bytes(b'depth')
            (root/'proxy.mp4').write_bytes(b'video')
            (root/'steps.jsonl').write_text(json.dumps({'index': 0, 'state': [1],
                'contacts': {}, 'frame_paths': {'instance_id': '0000/instance_id.png',
                'depth': '0000/depth.png'}})+'\n')
            (root/'trajectory.json').write_text(json.dumps([[0], [1]]))
            (root/'contacts.json').write_text('[{}]')
            (root/'timings.json').write_text(json.dumps({'initialization_s': 1,
                'steps': [{'physics_s': .1, 'raster_s': .2, 'writing_s': .1,
                           'response_latency_s': .4}]}))
            (root/'session.json').write_text(json.dumps({'inputs_unchanged': True,
                'scene_and_observation_sha256': {}}))
            self.assertEqual(check_live_artifacts(root)['steps'], 1)
            (root/'0000/instance_id.png').unlink()
            with self.assertRaises(ValueError):
                check_live_artifacts(root)
