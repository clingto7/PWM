"""Saved chain is checked through its public artifact boundary."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from integrated_chain import check_live_artifacts


class IntegratedChainTest(unittest.TestCase):
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
