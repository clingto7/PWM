"""Measure gsplat extension cold/warm process startup on one synthetic scene.

Run twice in separate processes with the same extension cache. This code does
not call torch.compile: gsplat already JIT builds its CUDA extension.
"""

import argparse
import json
import statistics
import time
from pathlib import Path


def run(output, frames):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    import torch
    from gsplat.rendering import rasterization
    import gsplat
    import numpy as np
    from PIL import Image
    import_time_s = time.perf_counter() - started
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    device = "cuda"
    means = torch.tensor([[0., 0., 3.], [-.5, 0., 3.], [.5, 0., 3.]], device=device)
    quats = torch.tensor([[1., 0., 0., 0.]] * 3, device=device)
    scales = torch.tensor([[.2, .2, .02]] * 3, device=device)
    opacity = torch.full((3,), .98, device=device)
    features = torch.eye(3, 4, device=device)
    view = torch.eye(4, device=device)[None]
    intrinsic = torch.tensor([[[200., 0., 252.], [0., 200., 140.], [0., 0., 1.]]], device=device)

    def render():
        return rasterization(means, quats, scales, opacity, features, view,
            intrinsic, 504, 280, render_mode="RGB+ED", packed=False,
            near_plane=.1, far_plane=100)

    started = time.perf_counter()
    image, alpha, _ = render()
    torch.cuda.synchronize()
    first_s = time.perf_counter() - started
    samples = []
    for _ in range(frames):
        started = time.perf_counter()
        image, alpha, _ = render()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    started = time.perf_counter()
    ids = image[0, :, :, :3].argmax(-1).to(torch.int16) + 1
    ids = torch.where(alpha[0, :, :, 0] >= .2, ids, 0)
    depth = torch.where(ids > 0, image[0, :, :, -1], 0)
    ids_cpu = ids.cpu().numpy().astype(np.uint16)
    depth_cpu = (depth.cpu().numpy() * 1000).clip(0, 65535).astype(np.uint16)
    readback_s = time.perf_counter() - started
    Image.fromarray(ids_cpu).save(output / "instance_id.png")
    Image.fromarray(depth_cpu).save(output / "depth.png")
    write_s = time.perf_counter() - started - readback_s
    result = {"schema": 1, "scope": "synthetic gsplat extension startup, not PWM scene",
              "gsplat": getattr(gsplat, "__version__", "unknown"),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "image_wh": [504, 280], "gaussians": 3,
              "import_s": import_time_s, "first_render_s": first_s,
              "steady_median_ms": statistics.median(samples) * 1000,
              "readback_ms": readback_s * 1000, "two_png_write_ms": write_s * 1000}
    (output / "measurements.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.frames <= 10000:
        parser.error("frames must be 1..10000")
    print(json.dumps(run(args.output, args.frames), sort_keys=True))


if __name__ == "__main__":
    main()
