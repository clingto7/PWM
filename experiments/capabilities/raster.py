"""Independent headless mesh and heightfield-mesh depth/ID probe."""
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
    import numpy as np
    import pybullet as p

    started = time.perf_counter()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    client = p.connect(p.DIRECT)
    try:
        n = 9
        vertices = [[(x - 4) * 0.5, (y - 4) * 0.5, 0.05 * ((x - 4) ** 2) / 16] for y in range(n) for x in range(n)]
        indices = []
        for y in range(n - 1):
            for x in range(n - 1):
                a = y * n + x
                indices.extend([a, a + 1, a + n, a + 1, a + n + 1, a + n])
        ground_visual = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices, rgbaColor=[0.4, 0.4, 0.4, 1])
        ground = p.createMultiBody(0, baseVisualShapeIndex=ground_visual)
        box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3], rgbaColor=[1, 0, 0, 1])
        box = p.createMultiBody(0, baseVisualShapeIndex=box_visual, basePosition=[0, 0, 0.45])
        near, far = 0.1, 10.0
        view = p.computeViewMatrix([2, -3, 2], [0, 0, 0], [0, 0, 1])
        projection = p.computeProjectionMatrixFOV(60, 1, near, far)
        _, _, _, zbuffer, segmentation = p.getCameraImage(128, 128, viewMatrix=view, projectionMatrix=projection, renderer=p.ER_TINY_RENDERER, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        zbuffer = np.asarray(zbuffer).reshape(128, 128)
        depth = far * near / (far - (far - near) * zbuffer)
        ids = np.asarray(segmentation, dtype=np.int32).reshape(128, 128) & ((1 << 24) - 1)
        ids[np.asarray(segmentation).reshape(128, 128) < 0] = -1
        path = out / "depth_ids.npz"
        np.savez(path, depth_m=depth.astype(np.float32), instance_id=ids)
        values, counts = np.unique(ids, return_counts=True)
        result = {"command": [sys.executable, str(Path(__file__).resolve()), str(out.resolve())], "python": platform.python_version(), "pybullet_api_version": p.getAPIVersion(), "renderer": "PyBullet TinyRenderer DIRECT", "seconds": time.perf_counter() - started, "outputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}, "id_pixel_counts": {str(k): int(v) for k, v in zip(values, counts)}, "finite_depth": bool(np.isfinite(depth).all()), "foreground_depth_range_m": [float(depth[ids >= 0].min()), float(depth[ids >= 0].max())] if (ids >= 0).any() else None, "passed": ground in values and box in values and bool(np.isfinite(depth).all()) and bool((depth[ids >= 0] < far).all())}
        (out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            sys.exit(1)
    finally:
        p.disconnect(client)


if __name__ == "__main__":
    main()
