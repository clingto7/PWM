"""Persistent saved-scene Bullet/raster session. Controls are delta velocity (m/s).

CLI: python live_session.py SAVED_RUN OUTPUT --controls controls.json [--video]
Omit --controls to send one finite 3-vector per stdin JSONL line. Results flush
on stdout after each step; EOF closes the session.
"""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from artifacts import SceneArtifacts
from physics import BulletRasterSession


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def hashes(root):
    files = list(SceneArtifacts(root).files().values())
    files.append(root/'run.json')
    if (root/'bundle.json').is_file():
        files.append(root/'bundle.json')
    inputs = root/'inputs'
    if inputs.is_dir():
        files.extend(p for p in inputs.rglob('*') if p.is_file())
    observation = root.parent/'observation'
    if observation.is_dir():
        files.extend(p for p in observation.rglob('*') if p.is_file())
    return {str(p.resolve().relative_to(root.parent.resolve())): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def implementation_identity(session, source):
    def name(value):
        kind = type(value)
        return f'{kind.__module__}.{kind.__qualname__}'
    return {'session': name(session), 'solver': name(source),
            'renderer': f'{name(source)}.frame_current'}


def raster_source(root):
    from artifacts import SceneArtifacts
    artifacts = SceneArtifacts(root)
    scene = artifacts.load().scene
    params = json.loads((root/'run.json').read_text())['parameters']
    radius = params['radius_m']; width, height = 504, 280
    near, far = .1, max(100, radius*4)
    k = scene['camera']['intrinsics']
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    projection = [2*fx/width,0,0,0, 0,2*fy/height,0,0,
                  1-2*cx/width,2*cy/height-1,(far+near)/(near-far),-1,
                  0,0,2*far*near/(near-far),0]
    args = SimpleNamespace(radius=radius, friction=params['friction'],
                           obstacles_render_only=params['obstacles_render_only'])
    return BulletRasterSession(artifacts,args,width,height,near,far,projection)


def save_raster(frame, destination):
    import numpy as np
    from PIL import Image
    metric, ids = frame
    destination.mkdir()
    depth = np.where(ids>0,np.clip(metric*1000,1,65535),0).astype(np.uint16)
    Image.fromarray(depth).save(destination/'depth.png')
    Image.fromarray(ids).save(destination/'instance_id.png')
    colors = np.empty((*ids.shape, 3), dtype=np.uint8)
    colors[ids == 0] = (0, 0, 0)           # no reconstructed geometry
    colors[ids == 1] = (145, 151, 142)    # support surface
    colors[ids == 2] = (244, 153, 55)     # controlled body
    static = ids >= 3
    colors[static] = (58, 166, 105)       # static objects
    Image.fromarray(colors, 'RGB').save(destination/'proxy_rgb.png')
    return {'depth': str(destination.name+'/depth.png'),
            'instance_id': str(destination.name+'/instance_id.png'),
            'proxy_rgb': str(destination.name+'/proxy_rgb.png')}


def encode_proxy_video(output):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '30',
        '-i', str(output/'%04d/proxy_rgb.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        str(output/'proxy.mp4')], check=True)


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {'count': 0}
    return {'count': len(ordered), 'min': ordered[0], 'median': statistics.median(ordered),
            'p95': ordered[math.ceil(.95*len(ordered))-1], 'max': ordered[-1]}


class LiveSession:
    """One world, sequential bounded controls, immediate current-state raster frames."""
    def __init__(self, root, output, *, source=None, save_frame=save_raster, steps_per_control=8):
        self.root = Path(root).resolve(); self.output = Path(output).resolve()
        if self.root == self.output or self.root in self.output.parents or self.output in self.root.parents:
            raise ValueError('output must be separate from saved run')
        if not isinstance(steps_per_control, int) or not 1 <= steps_per_control <= 240:
            raise ValueError('steps_per_control must be 1..240')
        self.steps_per_control = steps_per_control
        self.source = source
        self.save_frame = save_frame
        self.rows = []; self.contacts = []; self.timings = []

    def __enter__(self):
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError('output directory is not empty')
        started = time.perf_counter()
        self.before = hashes(self.root) if self.source is None else {str((self.root/'scene.json').relative_to(self.root.parent)): hashlib.sha256((self.root/'scene.json').read_bytes()).hexdigest()}
        self.output.mkdir(parents=True, exist_ok=True)
        self.source = (self.source or raster_source(self.root)).__enter__()
        self.implementation = implementation_identity(self, self.source)
        self.initialization_s = time.perf_counter()-started
        self.rows.append(self._state())
        self.last_return = time.perf_counter()
        return self

    def _state(self):
        p = self.source.p; body = self.source.body
        pos, quat = p.getBasePositionAndOrientation(body)
        vel, omega = p.getBaseVelocity(body)
        return [*pos, *quat, *vel, *omega]

    def step(self, delta_velocity):
        if (not isinstance(delta_velocity, (list, tuple)) or len(delta_velocity) != 3
                or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in delta_velocity)):
            raise ValueError('control must be a finite 3D delta velocity')
        waiting = time.perf_counter()-self.last_return
        p = self.source.p; body = self.source.body
        start = time.perf_counter()
        response_start = start
        linear, angular = p.getBaseVelocity(body)
        p.resetBaseVelocity(body, linearVelocity=[linear[i]+delta_velocity[i] for i in range(3)], angularVelocity=angular)
        penetration = 0.; counts = {}
        for _ in range(self.steps_per_control):
            p.stepSimulation()
            for contact in p.getContactPoints(bodyA=body):
                penetration = max(penetration, -float(contact[8]))
                if contact[2] in self.source.obstacles:
                    key = str(self.source.obstacles[contact[2]])
                    counts[key] = counts.get(key, 0)+1
        state = self._state()
        physics_s = time.perf_counter()-start
        start = time.perf_counter()
        frame = self.source.frame_current()
        raster_s = time.perf_counter()-start
        index = len(self.contacts)
        contact = {'max_contact_penetration_m': penetration,
                   'obstacle_contact_samples_by_id': counts}
        start = time.perf_counter()
        frame_paths = self.save_frame(frame, self.output/f'{index:04d}')
        self.rows.append(state); self.contacts.append(contact)
        record = {'index': index, 'control_delta_velocity_m_s': list(delta_velocity),
                  'state': state, 'contacts': contact, 'frame_paths': frame_paths}
        with (self.output/'steps.jsonl').open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True)+'\n')
        writing_s = time.perf_counter()-start
        response_latency_s = time.perf_counter()-response_start
        self.timings.append({'waiting_s': waiting, 'physics_s': physics_s,
                             'raster_s': raster_s, 'writing_s': writing_s,
                             'response_latency_s': response_latency_s})
        self.last_return = time.perf_counter()
        return {**record, 'frame': frame, 'latency_s': self.timings[-1]}

    def __exit__(self, *_):
        try:
            self.source.__exit__(*_)
        finally:
            write_json(self.output/'trajectory.json', self.rows)
            write_json(self.output/'contacts.json', self.contacts)
            write_json(self.output/'timings.json', {'initialization_s': self.initialization_s,
                'steps': self.timings, 'distribution_s': {key: distribution([row[key] for row in self.timings])
                for key in ('waiting_s','physics_s','raster_s','writing_s','response_latency_s')}})
            after = {name: hashlib.sha256((self.root.parent/name).read_bytes()).hexdigest() for name in self.before}
            write_json(self.output/'session.json', {'schema': 1, 'scene_and_observation_sha256': self.before,
                'implementation': self.implementation,
                'inputs_unchanged': after == self.before, 'steps_per_control': self.steps_per_control,
                'physics_hz': 240, 'state_format': 'position_xyz,orientation_xyzw,linear_velocity_xyz,angular_velocity_xyz',
                'trajectory': 'trajectory.json', 'contacts': 'contacts.json', 'steps': 'steps.jsonl'})
            if after != self.before:
                raise RuntimeError('saved scene or observation changed during live session')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved_run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--controls', type=Path, help='JSON list; omit for stdin JSONL')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    controls = json.loads(args.controls.read_text()) if args.controls else None
    if controls is not None and not isinstance(controls, list):
        parser.error('controls must be a list')
    stream = controls if controls is not None else (json.loads(line) for line in sys.stdin if line.strip())
    count = 0
    with LiveSession(args.saved_run, args.output) as session:
        for control in stream:
            result = session.step(control)
            print(json.dumps({k:v for k,v in result.items() if k != 'frame'}), flush=True)
            count += 1
    if args.video and count:
        started = time.perf_counter()
        encode_proxy_video(args.output)
        timing_path = args.output/'timings.json'
        timing = json.loads(timing_path.read_text())
        timing['optional_video_writing_s'] = time.perf_counter()-started
        write_json(timing_path, timing)
    print(json.dumps(json.loads((args.output/'timings.json').read_text())['distribution_s'],sort_keys=True))


if __name__ == '__main__':
    main()
