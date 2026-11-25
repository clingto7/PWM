"""Independent gsplat depth and four-channel feature rendering probe."""
import hashlib
import argparse
import json
import platform
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    import gsplat
    import torch
    from gsplat.rendering import rasterization

    started = time.perf_counter()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("gsplat capability probe requires CUDA")
    means = torch.tensor([[-0.2, 0, 2], [0.2, 0, 3]], device=device, dtype=torch.float32)
    quats = torch.tensor([[1, 0, 0, 0]] * 2, device=device, dtype=torch.float32)
    scales = torch.tensor([[0.2, 0.2, 0.02]] * 2, device=device, dtype=torch.float32)
    opacities = torch.tensor([0.95, 0.95], device=device)
    features = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 1]], device=device, dtype=torch.float32)
    view = torch.eye(4, device=device)[None]
    K = torch.tensor([[[100, 0, 32], [0, 100, 32], [0, 0, 1]]], device=device, dtype=torch.float32)
    rendered, alpha, _ = rasterization(means, quats, scales, opacities, features, view, K, 64, 64, render_mode="RGB+ED", packed=False)
    torch.cuda.synchronize()
    array = rendered.detach().cpu().numpy()
    path = out / "depth_features.npy"
    import numpy as np
    np.save(path, array)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    occupied = alpha.detach().cpu().numpy()[0, :, :, 0] > 0.01
    depth = array[0, :, :, -1]
    result = {"command": [sys.executable, str(Path(__file__).resolve()), str(out.resolve())], "python": platform.python_version(), "gsplat": getattr(gsplat, "__version__", "unknown"), "torch": torch.__version__, "cuda": torch.version.cuda, "seconds": time.perf_counter() - started, "outputs": {path.name: digest}, "shape": list(array.shape), "occupied_pixels": int(occupied.sum()), "depth_min_max_m": [float(depth[occupied].min()), float(depth[occupied].max())] if occupied.any() else None, "fourth_feature_max": float(array[..., 3].max()), "passed": array.shape == (1, 64, 64, 5) and occupied.any() and float(array[..., 3].max()) > 0 and bool(np.isfinite(depth[occupied]).all())}
    (out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
