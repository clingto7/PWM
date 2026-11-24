"""Generate a car mesh with the official Hunyuan3D 2.1 shape pipeline."""

import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


root = Path(__file__).resolve().parent
input_path = root / "car_input/car_rgba.png"
model_path = root / "models/hunyuan21"
output_dir = root / "hunyuan21_output"
output_dir.mkdir(exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


start = time.monotonic()
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(str(model_path))
load_seconds = time.monotonic() - start
torch.manual_seed(42)
start = time.monotonic()
mesh = pipeline(
    image=Image.open(input_path).convert("RGBA"),
    num_inference_steps=30,
    guidance_scale=5.0,
    octree_resolution=256,
)[0]
inference_seconds = time.monotonic() - start
output_path = output_dir / "car_shape.glb"
mesh.export(output_path)
component_face_counts = sorted(
    (len(component.faces) for component in mesh.split(only_watertight=False)),
    reverse=True,
)
metadata = {
    "input": str(input_path),
    "input_sha256": sha256_file(input_path),
    "model": "tencent/Hunyuan3D-2.1, hunyuan3d-dit-v2-1, shape only",
    "implementation": "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1",
    "implementation_head_at_download": "82920d643c0dc2f7bfd7255f45f62d386edfe60c",
    "weight_sha256": sha256_file(model_path / "hunyuan3d-dit-v2-1/model.fp16.ckpt"),
    "seed": 42,
    "steps": 30,
    "guidance_scale": 5.0,
    "octree_resolution": 256,
    "output": "car_shape.glb",
    "output_sha256": sha256_file(output_path),
    "load_seconds": round(load_seconds, 2),
    "inference_seconds": round(inference_seconds, 2),
    "vertices": len(mesh.vertices),
    "faces": len(mesh.faces),
    "watertight": bool(mesh.is_watertight),
    "component_face_counts": component_face_counts,
    "bounds": mesh.bounds.tolist(),
    "max_cuda_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
}
(output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
