"""Portable source-input bundle for a first-image run."""
import json
import hashlib
import shutil
import subprocess
from pathlib import Path

INPUT_NAMES = {'image': 'input.jpg', 'da3': 'prediction.npz',
               'sam_car': 'car_mask.png', 'hunyuan_mesh': 'car_shape.glb'}
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / 'first_image'
DEFAULT_PATHS = {'image': 'input.jpg', 'da3': 'da3nested_output/prediction.npz',
                 'sam_car': 'car_input/car_mask.png',
                 'hunyuan_mesh': 'hunyuan21_output/car_shape.glb'}


def sources(bundle):
    if bundle is None:
        return {name: DEFAULT_SOURCE / path for name, path in DEFAULT_PATHS.items()}
    bundle = Path(bundle).resolve()
    manifest = bundle / 'bundle.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
        entries = data['inputs']
        paths = {name: (bundle / entries[name]).resolve() for name in INPUT_NAMES}
    else:
        paths = {name: bundle / 'inputs' / filename for name, filename in INPUT_NAMES.items()}
    for name, path in paths.items():
        if not path.is_relative_to(bundle) or not path.is_file():
            raise ValueError(f'input {name} is missing or outside bundle: {path}')
    if manifest.exists():
        for name, path in paths.items():
            expected=data.get('source_sha256',{}).get(name)
            if expected is None or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'input {name} hash mismatch or missing bundle hash')
    return paths


def copy_inputs(paths, output):
    directory = output / 'inputs'
    directory.mkdir()
    entries = {}
    for name, filename in INPUT_NAMES.items():
        shutil.copyfile(paths[name], directory / filename)
        entries[name] = f'inputs/{filename}'
    return entries


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True).strip()
