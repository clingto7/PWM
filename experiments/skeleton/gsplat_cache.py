"""Persistent gsplat CUDA extension cache and optional one-time warmup.

Set the cache before importing gsplat/torch extension code. Reusing the same
directory with the same Python, PyTorch, CUDA and gsplat build avoids a second
CUDA compilation; a changed build may compile again.
"""

import argparse
import json
import os
from pathlib import Path
import time


def configure(cache_dir):
    if cache_dir is None:
        return os.environ.get('TORCH_EXTENSIONS_DIR')
    path = Path(cache_dir).expanduser().resolve()
    existing = os.environ.get('TORCH_EXTENSIONS_DIR')
    if existing and Path(existing).expanduser().resolve() != path:
        raise ValueError('TORCH_EXTENSIONS_DIR conflicts with gsplat cache directory')
    path.mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_EXTENSIONS_DIR'] = str(path)
    return str(path)


def warm(cache_dir=None):
    cache = configure(cache_dir)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('gsplat cache warmup requires CUDA')
    from gsplat.rendering import rasterization
    device = 'cuda'
    started = time.perf_counter()
    with torch.inference_mode():
        rasterization(
            means=torch.tensor([[0., 0., 3.]], device=device),
            quats=torch.tensor([[1., 0., 0., 0.]], device=device),
            scales=torch.tensor([[.2, .2, .02]], device=device),
            opacities=torch.tensor([.9], device=device),
            colors=torch.tensor([[1., 0., 0., 0.]], device=device),
            viewmats=torch.eye(4, device=device)[None],
            Ks=torch.tensor([[[100., 0., 16.], [0., 100., 16.], [0., 0., 1.]]], device=device),
            width=32, height=32, render_mode='RGB+ED', packed=False,
            near_plane=.1, far_plane=100.)
    torch.cuda.synchronize()
    return {'cache_dir': cache, 'warmup_seconds': time.perf_counter()-started,
            'torch': torch.__version__, 'cuda': torch.version.cuda}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(warm(args.cache_dir), sort_keys=True))


if __name__ == '__main__':
    main()
