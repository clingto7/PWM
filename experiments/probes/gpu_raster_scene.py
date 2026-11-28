"""Render a saved PWM scene and trajectory with CUDA triangle rasterization.

Experimental adapter: consumes immutable scene artifacts, retains entity IDs and
metric depth on GPU, and writes PNGs/MP4 for human comparison. It does not
replace the accepted raster renderer or modify the source run.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skeleton"))
from artifacts import SceneArtifacts


INPUTS = ("scene.json", "heightmap.npz", "obstacles.npz", "fitted_car.obj",
          "car_collision_hull.obj", "trajectory.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry_arrays(root, geometry):
    import numpy as np
    import trimesh

    vertex_parts = [np.asarray(geometry.vertices, dtype=np.float32)]
    triangle_parts = [np.asarray(geometry.indices, dtype=np.int32).reshape(-1, 3)]
    entity_parts = [np.full(len(triangle_parts[0]), 1, dtype=np.int16)]
    offset = len(vertex_parts[0])
    for component, (vertices, indices) in zip(geometry.components, geometry.obstacle_meshes):
        if not indices:
            continue
        part = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
        vertex_parts.append(part)
        triangle_parts.append(faces + offset)
        entity_parts.append(np.full(len(faces), component["instance_id"], dtype=np.int16))
        offset += len(part)
    visual = trimesh.load(root / "fitted_car.obj", force="mesh", process=False)
    car_vertices = np.asarray(visual.vertices, dtype=np.float32)
    car_faces = np.asarray(visual.faces, dtype=np.int32)
    if len(car_vertices) == 0 or len(car_faces) == 0:
        raise ValueError("empty controlled-body visual mesh")
    triangle_parts.append(car_faces + offset)
    entity_parts.append(np.full(len(car_faces), 2, dtype=np.int16))
    return (np.concatenate(vertex_parts), car_vertices,
            np.concatenate(triangle_parts), np.concatenate(entity_parts))


def run(root, output, frames):
    import numpy as np
    import torch
    import nvdiffrast.torch as dr
    from PIL import Image

    root, output = root.resolve(), output.resolve()
    if output == root or output in root.parents or root in output.parents:
        raise ValueError("output must be separate from source run")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory is not empty")
    before = {name: digest(root / name) for name in INPUTS}
    record = json.loads((root / "run.json").read_text())
    for name, value in before.items():
        if record["outputs"].get(name) != value:
            raise ValueError(f"source run hash mismatch: {name}")
    rows = json.loads((root / "trajectory.json").read_text())
    available = int(float(record["parameters"]["seconds"]) * int(record["parameters"]["video_fps"]))
    if not 1 <= frames <= available or len(rows) < (frames - 1) * 8 + 1:
        raise ValueError("requested frames exceed saved trajectory")
    geometry = SceneArtifacts(root).load()
    static, car, triangles, entities = geometry_arrays(root, geometry)
    camera = geometry.scene["camera"]
    width, height = 504, 280
    near, far = .1, max(100.0, float(record["parameters"]["radius_m"]) * 4)
    forward = np.asarray(camera["forward"], dtype=np.float32)
    up = np.asarray(camera["up"], dtype=np.float32)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    basis = torch.as_tensor(np.stack((right, up, forward)), device="cuda")
    intrinsic = camera["intrinsics"]
    fx, fy, cx, cy = (float(intrinsic[0][0]), float(intrinsic[1][1]),
                      float(intrinsic[0][2]), float(intrinsic[1][2]))
    static_gpu = torch.as_tensor(static, device="cuda")
    car_gpu = torch.as_tensor(car, device="cuda")
    faces_gpu = torch.as_tensor(triangles, device="cuda")
    lookup = torch.as_tensor(np.r_[0, entities], device="cuda")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("instance_id", "depth", "proxy_rgb"):
        (output / name).mkdir()
    context_start = time.perf_counter()
    context = dr.RasterizeCudaContext()
    torch.cuda.synchronize()
    context_s = time.perf_counter() - context_start
    render_samples, readback_samples, write_samples = [], [], []
    started_total = time.perf_counter()

    def clip_positions(world):
        view = world @ basis.T
        x, y, z = view.unbind(-1)
        return torch.stack((2 * fx / width * x + (2 * cx / width - 1) * z,
                            2 * fy / height * y + (1 - 2 * cy / height) * z,
                            (far + near) / (far - near) * z - 2 * far * near / (far - near),
                            z), dim=-1)

    for frame in range(frames):
        row = rows[frame * 8]
        torch.cuda.synchronize()
        started = time.perf_counter()
        q = torch.as_tensor(row[3:7], dtype=torch.float32, device="cuda")
        qv = q[:3].expand_as(car_gpu)
        t = 2 * torch.cross(qv, car_gpu, dim=-1)
        car_world = car_gpu + q[3] * t + torch.cross(qv, t, dim=-1)
        car_world = car_world + torch.as_tensor(row[:3], dtype=torch.float32, device="cuda")
        clip = clip_positions(torch.cat((static_gpu, car_world)))[None]
        raster, _ = dr.rasterize(context, clip, faces_gpu, [height, width], grad_db=False)
        raster = raster[0].flip(0)
        triangle = raster[..., 3].to(torch.long)
        ids = lookup[triangle]
        z_ndc = raster[..., 2]
        depth = torch.where(triangle > 0,
            2 * far * near / (far + near - z_ndc * (far - near)), 0.0)
        torch.cuda.synchronize()
        render_samples.append(time.perf_counter() - started)
        started = time.perf_counter()
        ids_cpu = ids.cpu().numpy().astype(np.uint16)
        depth_cpu = (depth.cpu().numpy() * 1000).clip(0, 65535).astype(np.uint16)
        readback_samples.append(time.perf_counter() - started)
        started = time.perf_counter()
        name = f"{frame:04d}.png"
        Image.fromarray(ids_cpu).save(output / "instance_id" / name)
        Image.fromarray(depth_cpu).save(output / "depth" / name)
        colors = np.zeros((height, width, 3), dtype=np.uint8)
        colors[ids_cpu == 1] = (145, 151, 142)
        colors[ids_cpu == 2] = (244, 153, 55)
        colors[ids_cpu >= 3] = (58, 166, 105)
        Image.fromarray(colors, "RGB").save(output / "proxy_rgb" / name)
        write_samples.append(time.perf_counter() - started)
    subprocess.run(("ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-framerate", "30", "-i", str(output / "proxy_rgb" / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output / "proxy.mp4")), check=True)
    after = {name: digest(root / name) for name in INPUTS}
    if before != after:
        raise RuntimeError("source artifacts changed during rendering")
    result = {"schema": 1, "renderer": "nvdiffrast CUDA experimental",
              "source_sha256": before, "inputs_unchanged": True,
              "frames": frames, "image_wh": [width, height],
              "triangles": len(triangles), "gpu": torch.cuda.get_device_name(),
              "context_s": context_s,
              "first_render_ms": render_samples[0] * 1000,
              "render_median_ms": statistics.median(render_samples) * 1000,
              "readback_median_ms": statistics.median(readback_samples) * 1000,
              "png_write_median_ms": statistics.median(write_samples) * 1000,
              "total_s": time.perf_counter() - started_total}
    (output / "measurements.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=150)
    args = parser.parse_args()
    if not 1 <= args.frames <= 1800:
        parser.error("frames must be 1..1800")
    print(json.dumps(run(args.source_run, args.output, args.frames), sort_keys=True))


if __name__ == "__main__":
    main()
