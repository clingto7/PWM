"""Replay one proxy renderer against an immutable saved scene and trajectory.

Usage: render_adapters.py RUN_DIR OUTPUT_DIR --renderer raster|gaussian
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

from stage_contracts import digest

UPSTREAM = ('scene.json', 'heightmap.npz', 'obstacles.npz',
            'fitted_car.obj', 'trajectory.json')


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def render(root, out, renderer, nearby_deg):
    root, out = root.resolve(), out.resolve()
    if out == root or root in out.parents or out in root.parents:
        raise ValueError('render output must be separate from saved run')
    if out.exists() and any(out.iterdir()):
        raise ValueError('render output is not empty')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = {'schema': 1, 'renderer': renderer, 'status': 'failed',
              'camera': 'saved scene original', 'nearby_yaw_deg': nearby_deg,
              'diagnostics': {'depth': 'available, millimeters PNG',
                              'entity_id': 'available, uint16 PNG',
                              'rgb_appearance': 'unavailable; proxy colors encode entity IDs'}}
    try:
        run = json.loads((root/'run.json').read_text())
        expected = run['outputs']
        before = {}
        for name in UPSTREAM:
            before[name] = digest(root/name)
            if expected.get(name) != before[name]:
                raise ValueError(f'upstream hash mismatch: {name}')
        parameters = run['parameters']
        seconds = float(parameters['seconds'])
        fps = int(parameters['video_fps'])
        if fps != 30 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('unsupported saved video timebase')
        frames = int(seconds*fps)
        rows = json.loads((root/'trajectory.json').read_text())
        if len(rows) < frames*8 - 7:
            raise ValueError('saved trajectory does not cover video timeline')
        result['inputs'] = before
        result['timebase'] = {'fps': fps, 'frames': frames, 'physics_samples_per_frame': 8}
        view_names = [('source', 0)] + ([('nearby', nearby_deg)] if nearby_deg else [])
        result['views'] = {}
        for view_name, yaw in view_names:
            view_out = out/view_name
            if renderer == 'raster':
                from raster_replay import render_saved
                frame_times = render_saved(root, view_out, parameters, yaw)
                details = {'frame_times_s': frame_times}
            else:
                from gaussian import render as render_gaussian
                details = render_gaussian(root, output=view_out, yaw_deg=yaw)
            from PIL import Image
            import numpy as np
            video = view_out/'proxy.mp4'
            encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '504x280', '-r', str(fps),
                '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)],
                stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            for frame in range(frames):
                ids = np.asarray(Image.open(view_out/'instance_id'/f'{frame:04d}.png'))
                colors = np.stack(((ids*53)%256, (ids*97)%256, (ids*193)%256), axis=-1).astype(np.uint8)
                encoder.stdin.write(colors.tobytes())
            encoder.stdin.close()
            error = encoder.stderr.read().decode()
            if encoder.wait() != 0:
                raise RuntimeError(f'ffmpeg proxy video failed: {error}')
            result['views'][view_name] = {**details, 'video': str(video.relative_to(out))}
        after = {name: digest(root/name) for name in UPSTREAM}
        if before != after:
            raise ValueError('upstream hash changed during rendering')
        result['inputs_after'] = after
        result['outputs'] = {str(path.relative_to(out)): digest(path)
                             for path in out.rglob('*') if path.is_file()}
        result['status'] = 'complete'
        return result
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        result['reason'] = str(exc)
        raise
    finally:
        result['seconds'] = time.perf_counter() - started
        write(out/'render.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--renderer', choices=('raster', 'gaussian'), required=True)
    parser.add_argument('--nearby-yaw-deg', type=float, default=5)
    args = parser.parse_args()
    if not math.isfinite(args.nearby_yaw_deg) or abs(args.nearby_yaw_deg) > 10:
        parser.error('nearby yaw must be within 10 degrees')
    render(args.run_dir, args.output_dir, args.renderer, args.nearby_yaw_deg)


if __name__ == '__main__':
    main()
