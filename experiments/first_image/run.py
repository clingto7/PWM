"""Run depth estimation and automatic masks on the selected OSDS frame.

Usage: python run.py --input input.jpg --depth-model models/depth --sam-model models/sam2
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, Sam2Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--depth-model", type=Path, required=True)
    parser.add_argument("--sam-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.input).convert("RGB")
    width, height = image.size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    record = {
        "input": str(args.input),
        "source_url": "https://simcruisegame.github.io/osds/images/release-2.1.0/01-mini-nebraska-roadside-chase.jpg",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "image_size": [width, height],
        "device": device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "depth_model": "depth-anything/Depth-Anything-V2-Small-hf",
        "depth_weight_sha256": hashlib.sha256((args.depth_model / "model.safetensors").read_bytes()).hexdigest(),
        "sam_model": "facebook/sam2.1-hiera-tiny",
        "sam_weight_sha256": hashlib.sha256((args.sam_model / "model.safetensors").read_bytes()).hexdigest(),
        "outputs": {},
    }

    start = time.monotonic()
    processor = AutoImageProcessor.from_pretrained(args.depth_model, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(args.depth_model, local_files_only=True).to(device).eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        prediction = model(**inputs)
        depth = processor.post_process_depth_estimation(prediction, target_sizes=[(height, width)])[0]["predicted_depth"]
    depth = depth.float().cpu().numpy()
    np.save(args.output / "relative_depth.npy", depth)
    lo, hi = np.percentile(depth, [1, 99])
    display = np.uint8(np.clip((depth - lo) / (hi - lo), 0, 1) * 255)
    Image.fromarray(display).save(args.output / "relative_depth.png")
    record["outputs"]["depth"] = {
        "raw": "relative_depth.npy", "preview": "relative_depth.png",
        "shape": list(depth.shape), "min": float(depth.min()), "max": float(depth.max()),
        "p01": float(lo), "p99": float(hi), "seconds": round(time.monotonic() - start, 2),
    }
    del model, prediction, inputs
    torch.cuda.empty_cache()

    start = time.monotonic()
    # The installed environment lacks torchvision, required by the SAM2 HF
    # image processor. Reproduce its documented resize/normalization in PIL.
    rgb = np.asarray(image.resize((1024, 1024), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    pixels = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)
    xs = np.linspace(0.04, 0.96, 16) * width
    ys = np.linspace(0.04, 0.96, 9) * height
    points = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    scaled = points * np.array([1024 / width, 1024 / height], dtype=np.float32)
    prompts = torch.from_numpy(scaled).to(device).unsqueeze(0).unsqueeze(2)
    sam = Sam2Model.from_pretrained(args.sam_model, local_files_only=True).to(device).eval()
    with torch.inference_mode():
        output = sam(pixel_values=pixels, input_points=prompts, multimask_output=True)
    raw_scores = output.iou_scores[0].float().cpu().numpy()
    best = raw_scores.argmax(axis=1)
    scores = raw_scores[np.arange(len(points)), best]
    logits = output.pred_masks[0, torch.arange(len(points)), torch.from_numpy(best).to(device)]
    masks_256 = (logits > 0).cpu().numpy()
    areas = masks_256.mean(axis=(1, 2))
    candidates = np.where((scores > 0.8) & (areas > 0.0005) & (areas < 0.7))[0]
    selected = []
    for idx in candidates[np.argsort(scores[candidates])[::-1]]:
        mask = masks_256[idx]
        if all((mask & masks_256[prior]).sum() / (mask | masks_256[prior]).sum() < 0.8 for prior in selected):
            selected.append(int(idx))
    np.savez_compressed(args.output / "sam2_masks.npz", masks_256=masks_256,
                        points_xy=points, scores=scores, selected=np.array(selected))
    colors = np.random.default_rng(0).integers(32, 256, size=(len(selected), 3), dtype=np.uint8)
    overlay = np.asarray(image).copy()
    for idx, color in zip(selected, colors):
        small = torch.from_numpy(masks_256[idx].astype(np.float32))[None, None]
        mask = F.interpolate(small, size=(height, width), mode="nearest")[0, 0].numpy().astype(bool)
        overlay[mask] = np.uint8(0.55 * overlay[mask] + 0.45 * color)
    Image.fromarray(overlay).save(args.output / "sam2_overlay.png")
    record["outputs"]["sam2"] = {
        "raw": "sam2_masks.npz", "preview": "sam2_overlay.png",
        "method": "uniform point prompts; best IoU mask per point; IoU deduplication",
        "preprocessing": "PIL bilinear resize to 1024x1024; ImageNet mean/std; point coordinates scaled independently to 1024",
        "count": len(selected), "raw_count": len(masks_256), "shape": list(masks_256.shape),
        "seconds": round(time.monotonic() - start, 2),
    }
    (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
