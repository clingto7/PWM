"""Download the three first-round model snapshots into the ignored models directory.

Uses only Python's standard library and curl. Each file is resumed, checked against
ModelScope's file manifest, then atomically moved into place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import quote
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent / "models"
MIN_FREE_BYTES = 100 * 1024**3
SNAPSHOTS = (
    (
        "sam21large",
        "facebook/sam2.1-hiera-large",
        ("README.md", "config.json", "configuration.json", "model.safetensors",
         "preprocessor_config.json", "processor_config.json",
         "video_preprocessor_config.json", "sam2.1_hiera_l.yaml"),
    ),
    (
        "da3nested",
        "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        ("README.md", "config.json", "model.safetensors"),
    ),
    (
        "hunyuan21",
        "Tencent-Hunyuan/Hunyuan3D-2.1",
        ("README.md", "LICENSE", "Notice.txt", "configuration.json",
         "hunyuan3d-dit-v2-1/config.yaml", "hunyuan3d-dit-v2-1/model.fp16.ckpt",
         "hunyuan3d-vae-v2-1/config.yaml", "hunyuan3d-vae-v2-1/model.fp16.ckpt"),
    ),
)

# Pin large weights to the versions accepted in the first-round experiments.
PINNED = {
    ("sam21large", "model.safetensors"): (897897416, "dc407dce21301fd94abb395c5099b4f2c455fdc8a8f261ac3d0ea6d4cd197230"),
    ("da3nested", "model.safetensors"): (6759558100, "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c"),
    ("hunyuan21", "hunyuan3d-dit-v2-1/model.fp16.ckpt"): (7366389768, "6b519fc7242f78e9b5f47ea4d55668fe3d944a2d27332f4ca68d29a6ff603f5e"),
    ("hunyuan21", "hunyuan3d-vae-v2-1/model.fp16.ckpt"): (655648152, "5cbe97f25e6e7abd4bccc80ab07524ec0c86d24118486a9ba49bb5dfb070288a"),
}


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(repo: str) -> dict[str, dict]:
    url = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
    with urlopen(url, timeout=30) as response:
        data = json.load(response)
    if data.get("Code") != 200:
        raise RuntimeError(f"ModelScope manifest failed for {repo}: {data.get('Message', data.get('Code'))}")
    return {entry["Path"]: entry for entry in data["Data"]["Files"] if entry.get("Type") == "blob"}


def download_file(name: str, repo: str, relative: str, info: dict) -> None:
    expected_size = int(info["Size"])
    expected_hash = info["Sha256"].lower()
    if not expected_hash or len(expected_hash) != 64:
        raise RuntimeError(f"Missing SHA256 for {repo}/{relative}")
    pinned = PINNED.get((name, relative))
    if pinned and pinned != (expected_size, expected_hash):
        raise RuntimeError(f"Pinned weight changed: {repo}/{relative}")

    target = ROOT / name / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == expected_size and sha256(target) == expected_hash:
            log(f"verified existing {name}/{relative}")
            return
        raise RuntimeError(f"Existing file failed verification: {target}")

    partial = target.with_name(target.name + ".part")
    if partial.exists() and partial.stat().st_size > expected_size:
        raise RuntimeError(f"Partial file exceeds manifest size: {partial}")
    url = f"https://modelscope.cn/models/{repo}/resolve/master/{quote(relative)}"
    for attempt in range(1, 21):
        remaining = expected_size - (partial.stat().st_size if partial.exists() else 0)
        if shutil.disk_usage(ROOT).free < max(MIN_FREE_BYTES, remaining + 10 * 1024**3):
            raise RuntimeError("Shared filesystem has insufficient free space")
        if remaining:
            log(f"downloading {name}/{relative}: attempt {attempt}, {remaining} bytes remaining")
            result = subprocess.run(
                ["curl", "-q", "--fail", "--location", "--silent", "--show-error",
                 "--noproxy", "*", "--continue-at", "-", "--connect-timeout", "20",
                 "--speed-time", "60", "--speed-limit", "1024", "--output", str(partial), url],
                check=False,
            )
            if result.returncode != 0:
                log(f"curl exited {result.returncode}; retrying")
                time.sleep(min(60, attempt * 5))
                continue
        if partial.stat().st_size != expected_size:
            log(f"size mismatch for {name}/{relative}; retrying")
            time.sleep(min(60, attempt * 5))
            continue
        if sha256(partial) != expected_hash:
            raise RuntimeError(f"SHA256 mismatch for {name}/{relative}; kept {partial} for inspection")
        os.replace(partial, target)
        log(f"verified {name}/{relative}: {expected_size} bytes")
        return
    raise RuntimeError(f"Download exhausted retries: {name}/{relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="show selected files without downloading")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, repo, selected in SNAPSHOTS:
        files = manifest(repo)
        for relative in selected:
            if relative not in files:
                raise RuntimeError(f"ModelScope manifest lacks {repo}/{relative}")
            info = files[relative]
            total += int(info["Size"])
            if args.plan:
                log(f"plan {name}/{relative}: {info['Size']} bytes")
            else:
                download_file(name, repo, relative, info)
    log(f"complete: {len(SNAPSHOTS)} models, {total} bytes")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED: {exc}")
        sys.exit(1)
