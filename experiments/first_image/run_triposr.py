"""Run the official TripoSR model on the same automatic car cutout."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tsr.system import TSR
from tsr.utils import resize_foreground


root = Path(__file__).resolve().parent
input_path = root / "car_input/car_rgba.png"
model_path = root / "models/triposr"
output_dir = root / "triposr_output"
output_dir.mkdir(exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


torch.manual_seed(42)
start = time.monotonic()
model = TSR.from_pretrained(str(model_path), config_name="config.yaml", weight_name="model.ckpt")
model.renderer.set_chunk_size(8192)
model.to("cuda:0")
model.eval()
load_seconds = time.monotonic() - start

foreground = resize_foreground(Image.open(input_path).convert("RGBA"), 0.85)
rgba = np.asarray(foreground, dtype=np.float32) / 255.0
rgb = rgba[..., :3] * rgba[..., 3:4] + 0.5 * (1.0 - rgba[..., 3:4])
processed_path = output_dir / "processed_input.png"
image = Image.fromarray((rgb * 255).astype(np.uint8))
image.save(processed_path)

torch.cuda.reset_peak_memory_stats()
start = time.monotonic()
with torch.no_grad():
    scene_codes = model([image], device="cuda:0")
    mesh = model.extract_mesh(scene_codes, True, resolution=256)[0]
inference_seconds = time.monotonic() - start
output_path = output_dir / "car_shape.glb"
mesh.export(output_path)

metadata = {
    "input": str(input_path),
    "input_sha256": sha256_file(input_path),
    "processed_input": processed_path.name,
    "processed_input_sha256": sha256_file(processed_path),
    "preprocess": "Official resize_foreground 0.85, alpha composite against gray 0.5",
    "model": "stabilityai/TripoSR",
    "implementation": "https://github.com/VAST-AI-Research/TripoSR",
    "implementation_head_at_download": "107cefdc244c39106fa830359024f6a2f1c78871",
    "weight_sha256": sha256_file(model_path / "model.ckpt"),
    "seed": 42,
    "mc_resolution": 256,
    "chunk_size": 8192,
    "vertex_colors": True,
    "output": output_path.name,
    "output_sha256": sha256_file(output_path),
    "load_seconds": round(load_seconds, 2),
    "inference_seconds": round(inference_seconds, 2),
    "vertices": len(mesh.vertices),
    "faces": len(mesh.faces),
    "watertight": bool(mesh.is_watertight),
    "bounds": mesh.bounds.tolist(),
    "max_cuda_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
}
(output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
