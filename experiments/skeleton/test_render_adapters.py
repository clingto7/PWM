"""Saved renderer CLI contract seam."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import importlib.util
import types
import hashlib

CLI = Path(__file__).with_name('render_adapters.py')


class RendererBoundaryTest(unittest.TestCase):
    def test_gaussian_external_output_hashes_are_relative_to_output(self):
        spec = importlib.util.spec_from_file_location('gaussian_outputs', CLI.with_name('gaussian_outputs.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'saved'; root.mkdir()
            out = Path(tmp)/'external'; (out/'depth').mkdir(parents=True)
            (out/'depth/0000.png').write_bytes(b'frame')
            actual = module.hash_outputs(root, out, [out/'depth'], external=True)
            self.assertEqual(actual, {'depth/0000.png': hashlib.sha256(b'frame').hexdigest()})
            legacy = root/'gaussian_replay'; (legacy/'depth').mkdir(parents=True)
            (legacy/'depth/0000.png').write_bytes(b'frame')
            self.assertEqual(module.hash_outputs(root, legacy, [legacy/'depth']),
                             {'gaussian_replay/depth/0000.png': hashlib.sha256(b'frame').hexdigest()})

    def test_raster_replay_uses_frame_source_camera_method(self):
        calls = []
        class Source:
            def __init__(self, *args): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def set_camera_yaw(self, yaw): calls.append(yaw)
        class Artifacts:
            def __init__(self, root): self.scene = {'camera': {'intrinsics': [[100,0,252],[0,100,140],[0,0,1]]}}
            def load(self): return self
        class Renderer:
            def __init__(self, *args): pass
            def render(self, *args): return None, [0.1]
        stubs = {'artifacts': types.SimpleNamespace(SceneArtifacts=Artifacts),
                 'physics': types.SimpleNamespace(BulletRasterSession=Source),
                 'raster': types.SimpleNamespace(RasterRenderer=Renderer)}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, stubs):
            root = Path(tmp); (root/'trajectory.json').write_text('[]')
            spec = importlib.util.spec_from_file_location('raster_replay_test', CLI.with_name('raster_replay.py'))
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            module.render_saved(root, root/'output', {'radius_m': 27, 'friction': .6,
                'obstacles_render_only': False, 'seconds': .1}, yaw_deg=5)
        self.assertEqual(calls, [5])


class RenderAdapterCliTest(unittest.TestCase):
    def test_collision_hull_hash_is_checked_before_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'saved'; root.mkdir()
            names = ('scene.json', 'heightmap.npz', 'obstacles.npz',
                     'fitted_car.obj', 'car_collision_hull.obj', 'trajectory.json')
            outputs = {}
            for name in names:
                (root/name).write_bytes(b'original')
                outputs[name] = hashlib.sha256(b'original').hexdigest()
            (root/'car_collision_hull.obj').write_bytes(b'changed')
            (root/'run.json').write_text(json.dumps({'outputs': outputs}))
            out = Path(tmp)/'render'
            process = subprocess.run([sys.executable, str(CLI), str(root), str(out),
                                      '--renderer', 'raster'], capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('car_collision_hull.obj', json.loads((out/'render.json').read_text())['reason'])

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
