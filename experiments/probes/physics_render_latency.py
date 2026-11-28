"""Measure online neural-step overhead and GPU frame readback without training.

This is a latency floor, not a physics-quality test or a rasterizer benchmark.
Run in an existing PyTorch environment; it installs nothing.
"""

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path


def summarize(samples):
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "median_ms": statistics.median(ordered) * 1000,
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1] * 1000,
        "max_ms": ordered[-1] * 1000,
    }


def wall_samples(action, synchronize, *, warmup, repeats):
    for _ in range(warmup):
        action()
    synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        action()
        synchronize()
        samples.append(time.perf_counter() - started)
    return summarize(samples)


def run(output, warmup, repeats):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for this probe")
    from PIL import Image

    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device("cuda:0")
    torch.empty(1, device=device)
    torch.cuda.synchronize()
    initialization_s = time.perf_counter() - started
    result = {
        "schema": 1,
        "scope": "untrained MLP inference and GPU-to-CPU frame transfer only",
        "machine": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "parameters": {"warmup": warmup, "repeats": repeats,
                       "frame_wh": [504, 280], "state_features": 32,
                       "mlp_hidden": 64, "mlp_output": 13},
        "initialization_s": initialization_s,
    }
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(),
                                torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                torch.nn.Linear(64, 13)).eval()
    with torch.inference_mode():
        for batch in (1, 256):
            cpu_input = torch.randn(batch, 32)
            gpu_input = cpu_input.to(device)
            cpu_model = model.cpu()
            result[f"mlp_cpu_batch_{batch}"] = wall_samples(
                lambda: cpu_model(cpu_input), lambda: None,
                warmup=warmup, repeats=repeats)
            gpu_model = model.to(device)
            result[f"mlp_gpu_batch_{batch}"] = wall_samples(
                lambda: gpu_model(gpu_input), torch.cuda.synchronize,
                warmup=warmup, repeats=repeats)

        height, width = 280, 504
        rgb = torch.full((height, width, 3), 128, dtype=torch.uint8, device=device)
        depth = torch.full((height, width), 2.0, dtype=torch.float32, device=device)
        ids = torch.full((height, width), 2, dtype=torch.int16, device=device)

        def readback():
            return rgb.cpu(), depth.cpu(), ids.cpu()

        result["frame_readback"] = wall_samples(
            readback, torch.cuda.synchronize, warmup=warmup, repeats=repeats)
        write_start = time.perf_counter()
        rgb_cpu, depth_cpu, ids_cpu = readback()
        Image.fromarray(rgb_cpu.numpy(), "RGB").save(output / "proxy_rgb.png")
        Image.fromarray((depth_cpu.numpy() * 1000).astype("uint16")).save(output / "depth.png")
        Image.fromarray(ids_cpu.numpy().astype("uint16")).save(output / "instance_id.png")
        result["one_frame_readback_and_png_s"] = time.perf_counter() - write_start

    (output / "measurements.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.warmup < 0 or not 1 <= args.repeats <= 10000:
        parser.error("warmup must be nonnegative; repeats must be 1..10000")
    print(json.dumps(run(args.output, args.warmup, args.repeats), sort_keys=True))


if __name__ == "__main__":
    main()
