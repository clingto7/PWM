"""Repeat the original uniform-point mask experiment with SAM 2.1 Large."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import Sam2Model


root = Path(__file__).resolve().parent
input_path = root / "input.jpg"
model_path = root / "models/sam21large"
output_dir = root / "sam21large_output"
output_dir.mkdir(exist_ok=True)
image = Image.open(input_path).convert("RGB")
width, height = image.size

rgb = np.asarray(image.resize((1024, 1024), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
pixels = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0).cuda()
xs = np.linspace(0.04, 0.96, 16) * width
ys = np.linspace(0.04, 0.96, 9) * height
points = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
scaled = points * np.array([1024 / width, 1024 / height], dtype=np.float32)
prompts = torch.from_numpy(scaled).cuda().unsqueeze(0).unsqueeze(2)

torch.cuda.reset_peak_memory_stats()
start = time.monotonic()
model = Sam2Model.from_pretrained(model_path, local_files_only=True).cuda().eval()
with torch.inference_mode():
    output = model(pixel_values=pixels, input_points=prompts, multimask_output=True)
seconds = time.monotonic() - start
raw_scores = output.iou_scores[0].float().cpu().numpy()
best = raw_scores.argmax(axis=1)
scores = raw_scores[np.arange(len(points)), best]
logits = output.pred_masks[0, torch.arange(len(points)), torch.from_numpy(best).cuda()]
masks_256 = (logits > 0).cpu().numpy()
areas = masks_256.mean(axis=(1, 2))
candidates = np.where((scores > 0.8) & (areas > 0.0005) & (areas < 0.7))[0]
selected = []
for idx in candidates[np.argsort(scores[candidates])[::-1]]:
    mask = masks_256[idx]
    if all((mask & masks_256[prior]).sum() / (mask | masks_256[prior]).sum() < 0.8 for prior in selected):
        selected.append(int(idx))

np.savez_compressed(
    output_dir / "sam2_masks.npz",
    masks_256=masks_256,
    points_xy=points,
    scores=scores,
    selected=np.array(selected),
)
colors = np.random.default_rng(0).integers(32, 256, size=(len(selected), 3), dtype=np.uint8)
overlay = np.asarray(image).copy()
for idx, color in zip(selected, colors):
    small = torch.from_numpy(masks_256[idx].astype(np.float32))[None, None]
    mask = F.interpolate(small, size=(height, width), mode="nearest")[0, 0].numpy().astype(bool)
    overlay[mask] = np.uint8(0.55 * overlay[mask] + 0.45 * color)
Image.fromarray(overlay).save(output_dir / "sam2_overlay.png")

def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


record = {
    "input": "input.jpg",
    "input_sha256": file_hash(input_path),
    "model": "facebook/sam2.1-hiera-large",
    "weight_sha256": file_hash(model_path / "model.safetensors"),
    "method": "uniform 16x9 point prompts; best IoU mask per point; IoU deduplication",
    "preprocessing": "PIL bilinear resize to 1024x1024; ImageNet mean/std; point coordinates scaled independently to 1024",
    "count": len(selected),
    "raw_count": len(masks_256),
    "shape": list(masks_256.shape),
    "load_and_inference_seconds": round(seconds, 2),
    "max_cuda_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    "files": {
        name: {"size_bytes": (output_dir / name).stat().st_size, "sha256": file_hash(output_dir / name)}
        for name in ("sam2_masks.npz", "sam2_overlay.png")
    },
}
(output_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
