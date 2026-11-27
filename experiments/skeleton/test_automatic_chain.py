"""External CLI contract for the original-image chain."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CLI = Path(__file__).with_name('automatic_chain.py')


class AutomaticChainTest(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, str(CLI), *map(str, args)], capture_output=True, text=True)

    def test_missing_image_is_counted_as_diagnostic_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root/'config.json'
            config.write_text(json.dumps({'schema': 1, 'renderer': 'raster', 'seconds': 1}))
            out = root/'run'
            result = self.invoke('run', root/'missing.jpg', config, out)
            self.assertNotEqual(result.returncode, 0)
            manifest = json.loads((out/'chain.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['failure']['stage'], 'input')
            self.assertIn('missing.jpg', manifest['failure']['reason'])
            self.assertGreaterEqual(manifest['timings_s']['total'], 0)
            checked = self.invoke('check', out)
            self.assertNotEqual(checked.returncode, 0)
            self.assertEqual(json.loads(checked.stdout)['status'], 'failed')

    def test_invalid_config_fails_before_any_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root/'config.json'
            config.write_text(json.dumps({'schema': 1, 'renderer': 'invalid'}))
            result = self.invoke('run', root/'image.jpg', config, root/'run')
            self.assertNotEqual(result.returncode, 0)
            manifest = json.loads((root/'run/chain.json').read_text())
            self.assertEqual(manifest['failure']['stage'], 'config')


if __name__ == '__main__':
    unittest.main()

class ReplayRoutingTest(unittest.TestCase):
    def test_scene_reuses_scene_contract_and_render_uses_saved_trajectory(self):
        from unittest.mock import patch
        import importlib.util
        spec = importlib.util.spec_from_file_location('automatic_chain', CLI)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = root/'saved'
            saved.mkdir()
            (saved/'chain.json').write_text(json.dumps({'config': {'schema': 1, 'renderer': 'raster'}}))
            with patch.object(module, 'check', return_value={'status': 'complete'}), \
                 patch.object(module, 'run') as run, patch.object(module, 'invoke') as invoke:
                for stage in ('observation', 'mesh', 'scene', 'physics', 'render'):
                    with patch.object(sys, 'argv', ['automatic_chain.py', 'replay', str(saved),
                                                    str(root/stage), '--from-stage', stage]):
                        module.main()
                self.assertEqual(run.call_count, 3)
                self.assertEqual(run.call_args.args[4], (saved/'automatic/run').resolve())
                self.assertEqual(run.call_args_list[0].args[5], (saved/'automatic/observation').resolve())
                self.assertIsNone(run.call_args_list[0].args[3])
                self.assertEqual(run.call_args_list[1].args[3], (saved/'automatic/observation').resolve())
                self.assertIsNone(run.call_args_list[1].args[4])
                self.assertEqual(invoke.call_count, 2)
                for call in invoke.call_args_list:
                    self.assertEqual(call.args[0][1], (saved/'automatic/run').resolve())
            self.assertFalse(list(root.glob('*.replay-config.json')))

class SemanticGateTest(unittest.TestCase):
    def test_saved_run_checker_failure_blocks_chain(self):
        from unittest.mock import patch
        import importlib.util
        spec = importlib.util.spec_from_file_location('automatic_chain', CLI)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        complete = subprocess.CompletedProcess([], 1, stdout=json.dumps({
            'passed': False, 'checks': {'energy': False}, 'metrics': {'energy_gain': 1}}))
        with patch.object(module.subprocess, 'run', return_value=complete):
            result = module.semantic_check(Path('/saved/run'))
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['energy'])

class ObservationReplayTest(unittest.TestCase):
    def test_observation_replay_reuses_perception_and_regenerates_mesh(self):
        from unittest.mock import patch
        import importlib.util
        spec = importlib.util.spec_from_file_location('automatic_chain', CLI)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'saved'
            (source/'inputs').mkdir(parents=True)
            (source/'inputs/input.jpg').write_bytes(b'image')
            (source/'inputs/prediction.npz').write_bytes(b'depth')
            (source/'sam2_masks.npz').write_bytes(b'masks')
            (source/'depth_capability.json').write_text(json.dumps({'scale': 'metric_m'}))
            image = source/'inputs/input.jpg'
            config = {'schema': 1, 'renderer': 'raster', 'saved_mesh': '/old/mesh.glb'}
            with patch.object(module, 'check_stage', return_value={'implementation': 'saved model'}), \
                 patch.object(module, 'invoke', side_effect=RuntimeError('stop after routing')) as invoke:
                with self.assertRaisesRegex(RuntimeError, 'stop after routing'):
                    module.run(image, config, root/'new', resume_perception=source)
            command = list(map(str, invoke.call_args.args[0]))
            self.assertIn('--depth-source', command)
            self.assertIn('--segmentation-source', command)
            self.assertNotIn('--saved-mesh', command)
            depth_meta = json.loads((root/'new/replay_sources/depth/run.json').read_text())
            self.assertEqual(depth_meta['is_metric'], True)
            import hashlib
            self.assertEqual(depth_meta['files']['prediction.npz']['sha256'], hashlib.sha256(b'depth').hexdigest())
            self.assertEqual(json.loads((root/'new/chain.json').read_text())['failure']['stage'], 'observation')
