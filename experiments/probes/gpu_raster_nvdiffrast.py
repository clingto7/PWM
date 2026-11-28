"""Probe GPU-resident instance ID/depth rasterization with a synthetic grid.

Requires an existing torch+nvdiffrast environment. This tests the renderer API,
readback, and PNG writing; synthetic geometry cannot establish PWM scene quality.
"""

import argparse
import json
import statistics
import time
from pathlib import Path


def grid_geometry(cells, optical_depth_m=2.0, near=.1, far=100.0):
    import numpy as np

    coordinates = np.linspace(-.9, .9, cells + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coordinates, coordinates)
    z_ndc = (far + near - 2 * far * near / optical_depth_m) / (far - near)
    vertices = np.stack((xx, yy, np.full_like(xx, z_ndc), np.ones_like(xx)), -1)
    faces = []
    entity_ids = []
    stride = cells + 1
    for y in range(cells):
        for x in range(cells):
            a = y * stride + x
            faces.extend(((a, a + 1, a + stride),
                          (a + 1, a + stride + 1, a + stride)))
            entity_ids.extend((1 if x < cells // 2 else 2,) * 2)
    return vertices.reshape(-1, 4), np.asarray(faces, dtype=np.int32), entity_ids


def run(output, cells, frames):
    import torch
    import nvdiffrast.torch as dr
    import numpy as np
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    output.mkdir(parents=True, exist_ok=False)
    vertices, faces, entities = grid_geometry(cells)
    pos = torch.as_tensor(vertices, device="cuda")
    tri = torch.as_tensor(faces, device="cuda")
    lookup = torch.as_tensor([0, *entities], dtype=torch.int16, device="cuda")
    started = time.perf_counter()
    context = dr.RasterizeCudaContext()
    torch.cuda.synchronize()
    context_s = time.perf_counter() - started

    def render():
        raster, _ = dr.rasterize(context, pos[None], tri, [280, 504], grad_db=False)
        raster = raster[0].flip(0)  # nvdiffrast uses bottom-up image rows.
        triangle = raster[..., 3].to(torch.long)
        ids = lookup[triangle]
        normalized_depth = raster[..., 2]
        depth = torch.where(triangle > 0,
            2 * 100.0 * .1 / (100.0 + .1 - normalized_depth * (100.0 - .1)), 0.0)
        return ids, depth

    started = time.perf_counter()
    ids, depth = render()
    torch.cuda.synchronize()
    first_s = time.perf_counter() - started
    samples = []
    for _ in range(frames):
        started = time.perf_counter()
        ids, depth = render()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    started = time.perf_counter()
    ids_cpu = ids.cpu().numpy().astype(np.uint16)
    depth_cpu = (depth.cpu().numpy() * 1000).clip(0, 65535).astype(np.uint16)
    readback_s = time.perf_counter() - started
    Image.fromarray(ids_cpu).save(output / "instance_id.png")
    Image.fromarray(depth_cpu).save(output / "depth.png")
    colors = np.zeros((280, 504, 3), dtype=np.uint8)
    colors[ids_cpu == 1] = (145, 151, 142)
    colors[ids_cpu == 2] = (244, 153, 55)
    Image.fromarray(colors, "RGB").save(output / "proxy_rgb.png")
    write_s = time.perf_counter() - started - readback_s
    result = {"schema": 1, "scope": "synthetic GPU raster API probe, not PWM scene",
              "renderer": "nvdiffrast CUDA", "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "image_wh": [504, 280],
              "triangles": len(faces), "context_s": context_s, "first_render_s": first_s,
              "steady_median_ms": statistics.median(samples) * 1000,
              "readback_ms": readback_s * 1000, "three_png_write_ms": write_s * 1000,
              "visible_pixels_by_id": {str(i): int(np.count_nonzero(ids_cpu == i)) for i in (1, 2)}}
    (output / "measurements.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    if not 2 <= args.cells <= 256 or args.cells % 2 or not 1 <= args.frames <= 10000:
        parser.error("cells must be even and 2..256; frames must be 1..10000")
    print(json.dumps(run(args.output, args.cells, args.frames), sort_keys=True))


if __name__ == "__main__":
    main()
