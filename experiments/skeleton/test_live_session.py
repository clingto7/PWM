"""Public live-session boundary, with a small deterministic Bullet stand-in."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).parent))
from live_session import LiveSession, hashes, save_raster, encode_proxy_video
from physics import BulletRasterSession
import numpy as np
from PIL import Image
import subprocess


class FakeBullet:
    def __init__(self):
        self.position = 0.0
        self.velocity = 0.0
        self.steps = 0
    def getBasePositionAndOrientation(self, body):
        return ((self.position, 0, 0), (0, 0, 0, 1))
    def getBaseVelocity(self, body):
        return ((self.velocity, 0, 0), (0, 0, 0))
    def resetBaseVelocity(self, body, linearVelocity, angularVelocity):
        self.velocity = linearVelocity[0]
    def stepSimulation(self):
        self.position += self.velocity / 240
        self.steps += 1
    def getContactPoints(self, bodyA):
        return [(None, None, 7, None, None, None, None, None, -.01)]


class FakeSource:
    def __init__(self, bullet):
        self.p = bullet
        self.body = 2
        self.obstacles = {7: 4}
        self.calls = 0
    def __enter__(self):
        self.calls += 1
        return self
    def __exit__(self, *args):
        pass
    def frame_current(self):
        return self.p.position, self.p.steps


class LiveSessionTest(unittest.TestCase):
    def test_rgb_proxy_video_is_visible_and_semantic_frames_remain_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = np.zeros((32, 32), dtype=np.uint16)
            ids[:, :8] = 1
            ids[:, 8:24] = 2
            ids[:, 24:] = 4
            ids[0, :] = 0
            frame = (np.full(ids.shape, 2.0, dtype=np.float32), ids)
            paths = save_raster(frame, root/'0000')
            self.assertEqual(set(paths), {'depth', 'instance_id', 'proxy_rgb'})
            np.testing.assert_array_equal(np.asarray(Image.open(root/paths['instance_id'])), ids)
            rgb = np.asarray(Image.open(root/paths['proxy_rgb']))
            self.assertEqual(rgb.shape, (32, 32, 3))
            np.testing.assert_array_equal(rgb[16, 30], [58, 166, 105])
            np.testing.assert_array_equal(rgb[0, 30], [0, 0, 0])
            np.testing.assert_array_equal(rgb[16, 2], [145, 151, 142])
            self.assertGreater(int(rgb[16, 16].max()), 100)
            self.assertFalse(np.array_equal(rgb[16, 2], rgb[16, 16]))
            self.assertFalse(np.array_equal(rgb[16, 16], rgb[16, 28]))
            encode_proxy_video(root)
            decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(root/'proxy.mp4'),
                '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                check=True, stdout=subprocess.PIPE).stdout
            pixels = np.frombuffer(decoded, dtype=np.uint8).reshape(32, 32, 3)
            self.assertGreater(int(pixels[16, 16].max()), 80)
            self.assertFalse(np.array_equal(pixels[16, 2], pixels[16, 16]))

    def test_live_raster_rejects_visible_unmapped_collision_body(self):
        class Camera:
            ER_TINY_RENDERER = 0
            ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX = 0

            def getCameraImage(self, *args, **kwargs):
                return 2, 2, None, np.full(4, .5), np.array([1, 9, -1, 0])

        source = BulletRasterSession(None, None, 2, 2, .1, 100, [])
        source.p = Camera()
        source.view = []
        source.ground = 0
        source.body = 1
        source.obstacles = {}
        with self.assertRaisesRegex(ValueError, 'unmapped visual body'):
            source.frame_current()

    def test_snapshot_includes_bundle_and_saved_observation_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; (root/'inputs').mkdir(parents=True)
            (root/'bundle.json').write_text('{}')
            (root/'inputs'/'prediction.npz').write_bytes(b'depth')
            with patch('live_session.SceneArtifacts') as artifacts:
                artifacts.return_value.files.return_value = {}
                (root/'run.json').write_text('{}')
                snapshot = hashes(root)
            self.assertIn('run/inputs/prediction.npz', snapshot)
            self.assertIn('run/bundle.json', snapshot)

    def test_sequential_controls_render_solved_state_and_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'saved'; root.mkdir()
            (root/'scene.json').write_text('{}')
            output = Path(tmp)/'live'
            bullet = FakeBullet(); source = FakeSource(bullet)
            def save(frame, destination):
                destination.write_text(json.dumps(frame))
            with LiveSession(root, output, source=source, save_frame=save, steps_per_control=2) as session:
                first = session.step([2, 0, 0])
                second = session.step([0, 0, 0])
                self.assertEqual(first['frame'], (2/240*2, 2))
                self.assertEqual(second['frame'], (2/240*4, 4))
                self.assertEqual(second['state'][0], second['frame'][0])
                self.assertEqual(first['contacts']['obstacle_contact_samples_by_id'], {'4': 2})
                self.assertEqual(source.calls, 1)
            self.assertEqual((root/'scene.json').read_text(), '{}')
            self.assertEqual(len(json.loads((output/'trajectory.json').read_text())), 3)
            self.assertEqual(len(json.loads((output/'contacts.json').read_text())), 2)
            implementation = json.loads((output/'session.json').read_text())['implementation']
            self.assertEqual(implementation, {'session': 'live_session.LiveSession',
                'solver': f'{FakeSource.__module__}.FakeSource',
                'renderer': f'{FakeSource.__module__}.FakeSource.frame_current'})
            timing = json.loads((output/'timings.json').read_text())
            self.assertEqual(len(timing['steps']), 2)
            self.assertEqual(timing['distribution_s']['response_latency_s']['count'], 2)
            self.assertGreaterEqual(timing['steps'][0]['response_latency_s'],
                                    timing['steps'][0]['physics_s'] + timing['steps'][0]['raster_s'])

    def test_invalid_control_does_not_advance_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'saved'; root.mkdir(); (root/'scene.json').write_text('{}')
            bullet = FakeBullet()
            with LiveSession(root, Path(tmp)/'live', source=FakeSource(bullet),
                             save_frame=lambda f,p: p.write_text('frame')) as session:
                with self.assertRaises(ValueError):
                    session.step([float('nan'), 0, 0])
                with self.assertRaises(ValueError):
                    session.step(1)
                self.assertEqual(bullet.steps, 0)
