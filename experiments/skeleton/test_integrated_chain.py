"""Saved chain is checked through its public artifact boundary."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from integrated_chain import check_live_artifacts, check_chain, ledger, digest, run, record_live_stage, LIVE_IMPLEMENTATION
from types import SimpleNamespace


class IntegratedChainTest(unittest.TestCase):
    def test_original_image_run_uses_scene_only_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/'original.jpg'; image.write_bytes(b'original')
            output = root/'chain'
            commands = []
            def execute(command):
                commands.append(command)
                if command[1].endswith('observation.py'):
                    (output/'observation/inputs').mkdir(parents=True)
                    (output/'observation/inputs/input.jpg').write_bytes(image.read_bytes())
                elif command[1].endswith('first_image_pipeline.py'):
                    self.assertIn('--scene-only', command)
                    (output/'scene').mkdir()
                    (output/'scene/run.json').write_text('{"mode":"scene_only"}')
                else:
                    (output/'live').mkdir()
                return .01
            args = SimpleNamespace(image=image, output=output, depth_source=None,
                segmentation_source=None, mesh=None, da3_weights=None, sam_weights=None,
                mesh_weights=None, generate_mesh=False, seconds=1, controls=[[0, 0, 0]])
            with patch('integrated_chain.execute', side_effect=execute), \
                 patch('integrated_chain.record_live_stage'), \
                 patch('integrated_chain.check_chain'):
                run(args)
            self.assertEqual([Path(command[1]).name for command in commands],
                             ['observation.py', 'first_image_pipeline.py', 'live_session.py'])
            self.assertEqual(json.loads((output/'chain.json').read_text())['schema'], 2)

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

    def test_scene_only_checker_rejects_fit_tamper_even_with_updated_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ('observation/inputs', 'scene', 'live/0000'):
                (root/directory).mkdir(parents=True)
            (root/'observation/inputs/input.jpg').write_bytes(b'image')
            (root/'observation/selection.json').write_text(json.dumps({
                'selected_index': 1, 'reason': 'strongest supported candidate',
                'candidates': [{'index': 1, 'contact': {'supported': True}}]}))
            (root/'scene/run.json').write_text(json.dumps({'schema': 3, 'mode': 'scene_only',
                'parameters': {}, 'timings_s': {}}))
            fit = {'fit_trusted': True, 'trust_checks': {'silhouette': True},
                'diagnostics': {'after': {'rendered_silhouette_iou': .9,
                    'visible_depth_pixel_count': 10, 'visible_depth_mae_m': .2,
                    'support_contact_gap_m': .01}}}
            (root/'scene/fit_diagnostics.json').write_text(json.dumps(fit))
            (root/'scene/fit_trial.json').write_text(json.dumps({'trusted': True,
                'initial_support_gap_m': .01, 'max_penetration_m': .01, 'passive_drift_m': .01}))
            (root/'live/session.json').write_text(json.dumps({'inputs_unchanged': True,
                'implementation': LIVE_IMPLEMENTATION, 'steps_per_control': 8,
                'physics_hz': 240, 'scene_and_observation_sha256': {
                    'scene/run.json': digest(root/'scene/run.json'),
                    'observation/inputs/input.jpg': digest(root/'observation/inputs/input.jpg')}}))
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
            (root/'controls.json').write_text('[[0, 0, 0]]')
            record_live_stage(root, .5)
            chain = {'schema': 2, 'status': 'complete',
                'input_sha256': digest(root/'observation/inputs/input.jpg'),
                'artifact_sha256': ledger(root), 'timings_s': {'live': .5}}
            (root/'chain.json').write_text(json.dumps(chain))
            from stage_contracts import check as real_check_stage
            with patch('integrated_chain.check_stage', side_effect=lambda path, stage:
                       real_check_stage(path, stage) if stage == 'live' else {}), \
                 patch('integrated_chain.timing_evidence', return_value={}):
                self.assertEqual(check_chain(root)['selected_index'], 1)
                fit['diagnostics']['after']['rendered_silhouette_iou'] = .1
                (root/'scene/fit_diagnostics.json').write_text(json.dumps(fit))
                chain['artifact_sha256'] = ledger(root)
                (root/'chain.json').write_text(json.dumps(chain))
                with self.assertRaisesRegex(ValueError, 'fit or passive contact gates'):
                    check_chain(root)

    def test_live_handoff_detects_tamper_even_if_chain_ledger_is_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ('scene', 'observation', 'live/0000'):
                (root/directory).mkdir(parents=True)
            (root/'scene/run.json').write_text('{}')
            (root/'observation/input.jpg').write_bytes(b'input')
            (root/'controls.json').write_text('[[1,0,0]]')
            (root/'live/session.json').write_text(json.dumps({
                'implementation': LIVE_IMPLEMENTATION, 'steps_per_control': 8,
                'physics_hz': 240, 'scene_and_observation_sha256': {
                    'scene/run.json': digest(root/'scene/run.json'),
                    'observation/input.jpg': digest(root/'observation/input.jpg')}}))
            (root/'live/timings.json').write_text('{"initialization_s":0.1,"steps":[]}')
            (root/'live/contacts.json').write_text('[]')
            (root/'live/0000/depth.png').write_bytes(b'frame')
            record_live_stage(root, .5)
            from integrated_chain import check_live_stage
            chain = {'timings_s': {'live': .5}}
            self.assertEqual(check_live_stage(root, chain)['implementation'], LIVE_IMPLEMENTATION)
            (root/'controls.json').write_text('[[2,0,0]]')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                check_live_stage(root, chain)
            (root/'controls.json').write_text('[[1,0,0]]')
            (root/'live/0000/depth.png').write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                check_live_stage(root, chain)
