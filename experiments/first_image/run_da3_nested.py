"""Run Depth Anything 3 Nested Giant 1.1 on the full selected scene."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.glb import export_to_glb
from depth_anything_3.utils.export.gs import export_to_gs_ply


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--no-gs", action="store_true", help="Skip the Gaussian branch if memory is insufficient")
args = parser.parse_args()
input_path = root / "input.jpg"
model_path = root / "models/da3nested"
output_dir = root / "da3nested_output"
output_dir.mkdir(exist_ok=True)

torch.manual_seed(42)
start = time.monotonic()
model = DepthAnything3.from_pretrained(str(model_path)).to("cuda").eval()
load_seconds = time.monotonic() - start
torch.cuda.reset_peak_memory_stats()
start = time.monotonic()
prediction = model.inference([str(input_path)], infer_gs=not args.no_gs, process_res=504)
inference_seconds = time.monotonic() - start

arrays = {"depth": prediction.depth}
for name in ("conf", "sky", "intrinsics", "extrinsics", "processed_images"):
    value = getattr(prediction, name, None)
    if value is not None:
        arrays[name] = np.asarray(value)
np.savez_compressed(output_dir / "prediction.npz", **arrays)

depth = prediction.depth[0]
valid = depth[np.isfinite(depth) & (depth > 0)]
lo, hi = np.percentile(valid, [1, 99])
preview = np.uint8(np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1) * 255)
Image.fromarray(preview).save(output_dir / "depth_preview.png")

exports = {}
for name, action in (
    ("glb", lambda: export_to_glb(prediction, str(output_dir), show_cameras=False)),
    ("gs_ply", lambda: export_to_gs_ply(prediction, str(output_dir)) if prediction.gaussians is not None else None),
):
    try:
        start = time.monotonic()
        result = action()
        exports[name] = {
            "status": "not_available" if name == "gs_ply" and prediction.gaussians is None else "ok",
            "seconds": round(time.monotonic() - start, 2),
        }
    except Exception as error:
        exports[name] = {"status": "error", "error": f"{type(error).__name__}: {error}"}

files = {}
for path in sorted(output_dir.rglob("*")):
    if path.is_file() and path.name != "run.json":
        files[str(path.relative_to(output_dir))] = {"size_bytes": path.stat().st_size, "sha256": file_hash(path)}
record = {
    "input": "input.jpg",
    "input_sha256": file_hash(input_path),
    "model": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "implementation": "https://github.com/ByteDance-Seed/Depth-Anything-3",
    "implementation_head_at_download": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
    "weight_sha256": file_hash(model_path / "model.safetensors"),
    "process_res": 504,
    "infer_gs": not args.no_gs,
    "load_seconds": round(load_seconds, 2),
    "inference_seconds": round(inference_seconds, 2),
    "max_cuda_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    "is_metric": bool(prediction.is_metric),
    "depth_shape": list(prediction.depth.shape),
    "depth_meters_min": float(valid.min()) if prediction.is_metric else None,
    "depth_meters_median": float(np.median(valid)) if prediction.is_metric else None,
    "depth_meters_max": float(valid.max()) if prediction.is_metric else None,
    "preview_depth_p01_p99": [float(lo), float(hi)],
    "exports": exports,
    "files": files,
}
(output_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
