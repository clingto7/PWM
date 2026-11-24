"""Automatically isolate the red foreground car for object-level 3D models."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label
from torch.nn import functional as F
from transformers import Sam2Model


ROOT = Path(__file__).resolve().parent
image_path = ROOT / "input.jpg"
model_path = ROOT / "models/sam2"
out = ROOT / "car_input"
out.mkdir(exist_ok=True)
image = Image.open(image_path).convert("RGB")
rgb = np.asarray(image, dtype=np.float32)
height, width = rgb.shape[:2]
red = (rgb[:, :, 0] > 1.55 * rgb[:, :, 1]) & (rgb[:, :, 0] > 1.35 * rgb[:, :, 2]) & (rgb[:, :, 0] > 90)
red[: height // 2] = False
components, count = label(red)
sizes = np.bincount(components.ravel())
sizes[0] = 0
largest = sizes.argmax()
yy, xx = np.nonzero(components == largest)
x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
box = np.array([
    max(0, x0 - 0.06 * (x1 - x0)),
    max(0, y0 - 0.22 * (y1 - y0)),
    min(width, x1 + 0.06 * (x1 - x0)),
    min(height, y1 + 0.08 * (y1 - y0)),
], dtype=np.float32)
resized = np.asarray(image.resize((1024, 1024), Image.Resampling.BILINEAR), dtype=np.float32) / 255
normalized = (resized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
device = "cuda" if torch.cuda.is_available() else "cpu"
pixels = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)
scaled_box = box * np.array([1024 / width, 1024 / height, 1024 / width, 1024 / height], dtype=np.float32)
model = Sam2Model.from_pretrained(model_path, local_files_only=True).to(device).eval()
with torch.inference_mode():
    prediction = model(pixel_values=pixels, input_boxes=torch.from_numpy(scaled_box)[None, None].to(device))
scores = prediction.iou_scores[0, 0].float().cpu().numpy()
chosen = int(scores.argmax())
logits = prediction.pred_masks[0, 0, chosen][None, None]
mask = (F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)[0, 0] > 0).cpu().numpy()
Image.fromarray(mask.astype(np.uint8) * 255).save(out / "car_mask.png")
yy, xx = np.nonzero(mask)
mx0, mx1, my0, my1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
side = int(max(mx1 - mx0, my1 - my0) * 1.15)
cx, cy = (mx0 + mx1) / 2, (my0 + my1) / 2
canvas = Image.new("RGBA", (side, side))
masked = np.dstack([np.asarray(image), mask.astype(np.uint8) * 255])
crop = Image.fromarray(masked).crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))
canvas.alpha_composite(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
canvas.save(out / "car_rgba.png")
metadata = {
    "source_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
    "method": "largest saturated-red component in lower half -> padded box prompt to SAM 2.1 Tiny",
    "red_component_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
    "sam_box_xyxy": box.tolist(),
    "sam_scores": scores.tolist(),
    "selected_mask": chosen,
    "mask_bbox_xyxy": [int(mx0), int(my0), int(mx1), int(my1)],
    "car_rgba_size": list(canvas.size),
}
(out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
