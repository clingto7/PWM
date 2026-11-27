"""Versioned subject mesh stage; Hunyuan3D 2.1 or a verified saved GLB."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from stage_contracts import check as check_stage, digest, record

MODEL_ROOT = Path(__file__).resolve().parents[1]/'first_image/models/hunyuan21'


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def geometry(path):
    import math
    import struct
    with path.open('rb') as stream:
        header = stream.read(20)
        if len(header) != 20 or header[:4] != b'glTF' or struct.unpack_from('<I', header, 4)[0] != 2:
            raise ValueError('invalid GLB header')
        total = struct.unpack_from('<I', header, 8)[0]
        chunk_size = struct.unpack_from('<I', header, 12)[0]
        if total != path.stat().st_size or header[16:20] != b'JSON':
            raise ValueError('invalid GLB length or JSON chunk')
        document = json.loads(stream.read(chunk_size))
        binary_header = stream.read(8)
        if len(binary_header) != 8 or binary_header[4:] != b'BIN\0':
            raise ValueError('invalid GLB binary chunk')
        binary = stream.read(struct.unpack_from('<I', binary_header)[0])
    accessors = document.get('accessors', [])
    vertices = faces = 0
    bounds = []
    for mesh in document.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            if primitive.get('mode', 4) != 4:
                continue
            position = accessors[primitive['attributes']['POSITION']]
            indices = accessors[primitive['indices']]
            vertices += position['count']
            faces += indices['count']//3
            bounds.append((position['min'], position['max']))
    valid = bool(vertices >= 4 and faces >= 4 and bounds and
                 all(all(math.isfinite(x) for x in axis) for pair in bounds for axis in pair))
    if not valid:
        raise ValueError('invalid GLB geometry: missing or nonfinite triangles')
    minimum = [min(pair[0][i] for pair in bounds) for i in range(3)]
    maximum = [max(pair[1][i] for pair in bounds) for i in range(3)]
    if any(maximum[i] <= minimum[i] for i in range(3)):
        raise ValueError('invalid GLB geometry: degenerate bounds')
    try:
        import trimesh
    except ImportError:
        # Lightweight local validation when the A800's pinned trimesh is absent.
        import numpy as np
        volumes = []
        watertight = True
        for mesh in document['meshes']:
            for primitive in mesh['primitives']:
                if primitive.get('mode', 4) != 4:
                    continue
                positions = accessors[primitive['attributes']['POSITION']]
                indices = accessors[primitive['indices']]
                pview = document['bufferViews'][positions['bufferView']]
                iview = document['bufferViews'][indices['bufferView']]
                if positions['componentType'] != 5126 or indices['componentType'] not in (5123, 5125):
                    raise ValueError('unsupported GLB mesh accessor types')
                dtype = '<u2' if indices['componentType'] == 5123 else '<u4'
                points = np.frombuffer(binary, dtype='<f4', count=positions['count']*3,
                    offset=pview.get('byteOffset',0)+positions.get('byteOffset',0)).reshape(-1,3)
                triangles = np.frombuffer(binary, dtype=dtype, count=indices['count'],
                    offset=iview.get('byteOffset',0)+indices.get('byteOffset',0)).reshape(-1,3)
                if not np.isfinite(points).all() or triangles.max(initial=0) >= len(points):
                    raise ValueError('invalid GLB mesh vertices or indices')
                edges = np.sort(np.concatenate((triangles[:,[0,1]], triangles[:,[1,2]],
                    triangles[:,[2,0]])), axis=1)
                _, counts = np.unique(edges, axis=0, return_counts=True)
                watertight &= bool(np.all(counts == 2))
                volumes.append(abs(float(np.einsum('ij,ij->i', points[triangles[:,0]],
                    np.cross(points[triangles[:,1]],points[triangles[:,2]])).sum()/6)))
        volume = sum(volumes)
    else:
        loaded = trimesh.load(path, force='scene')
        components = list(loaded.geometry.values())
        watertight = bool(components and all(component.is_watertight for component in components))
        volume = sum(abs(float(component.volume)) for component in components)
    if not watertight or not math.isfinite(volume) or volume <= 0:
        raise ValueError('invalid GLB mesh: requires watertight positive finite volume')
    return {'valid': True, 'vertices': vertices, 'faces': faces,
            'bounds': [minimum, maximum], 'watertight': watertight, 'volume_model_units3': volume}



def prepare_rgba(image, mask, output):
    """Preserve the exact selected subject pixels in a padded square local crop."""
    import math
    from PIL import Image
    rgb = Image.open(image).convert('RGB')
    alpha = Image.open(mask).convert('L')
    if alpha.size != rgb.size:
        raise ValueError('selected mask must match original image')
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError('selected subject mask is empty')
    left, top, right, bottom = bbox
    side = max(1, math.ceil(max(right-left, bottom-top)*1.15))
    cx, cy = (left+right)/2, (top+bottom)/2
    x0, y0 = math.floor(cx-side/2), math.floor(cy-side/2)
    crop = (x0, y0, x0+side, y0+side)
    rgb_crop = Image.new('RGB', (side, side))
    alpha_crop = Image.new('L', (side, side))
    rgb_crop.paste(rgb.crop(crop), (0, 0))
    alpha_crop.paste(alpha.crop(crop), (0, 0))
    rgba = rgb_crop.convert('RGBA')
    rgba.putalpha(alpha_crop)
    path = output/'masked_subject.png'
    rgba.save(path)
    return rgba, {'selected_mask_bbox_xyxy': list(bbox), 'crop_xyxy': list(crop),
                  'padding_fraction': .15, 'source_image_size_xy': list(rgb.size)}


def hunyuan21(image, mask, output, weights):
    if not (weights/'hunyuan3d-dit-v2-1/model.fp16.ckpt').is_file():
        raise ValueError('Hunyuan3D 2.1 local weights required; no download is permitted')
    import torch
    from PIL import Image
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    rgba = Image.open(output/'masked_subject.png').convert('RGBA')
    load_start = time.perf_counter()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(str(weights))
    load_seconds = time.perf_counter()-load_start
    torch.manual_seed(42)
    inference_start = time.perf_counter()
    mesh = pipeline(image=rgba, num_inference_steps=30, guidance_scale=5.0,
                    octree_resolution=256)[0]
    inference_seconds = time.perf_counter()-inference_start
    mesh.export(output/'car_shape.glb')
    return {'load_seconds': load_seconds, 'inference_seconds': inference_seconds,
            'weight_sha256': digest(weights/'hunyuan3d-dit-v2-1/model.fp16.ckpt')}


def build(image, mask, output, saved_mesh=None, provider='hunyuan3d21', weights=MODEL_ROOT):
    start = time.perf_counter()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory is not empty')
    (output/'inputs').mkdir(parents=True)
    write(output/'stage_status.json', {'stage': 'subject_mesh', 'status': 'running'})
    try:
        shutil.copyfile(image, output/'inputs/input.jpg')
        shutil.copyfile(mask, output/'inputs/car_mask.png')
        from PIL import Image
        import numpy as np
        original = Image.open(image)
        selected = Image.open(mask).convert('L')
        if selected.size != original.size or not np.asarray(selected).any():
            raise ValueError('selected mask must match original image and contain foreground')
        _, crop = prepare_rgba(output/'inputs/input.jpg', output/'inputs/car_mask.png', output)
        if saved_mesh:
            saved_mesh = Path(saved_mesh)
            previous = saved_mesh.parent/'subject_mesh.json'
            if previous.is_file():
                prior = json.loads(previous.read_text())
                expected = prior.get('input_hashes', {})
                if (expected.get('original_image_sha256') != digest(output/'inputs/input.jpg') or
                        expected.get('selected_mask_sha256') != digest(output/'inputs/car_mask.png')):
                    raise ValueError('saved mesh image or selected mask hash mismatch')
            shutil.copyfile(saved_mesh, output/'car_shape.glb')
            provenance = {'saved_mesh_sha256': digest(Path(saved_mesh))}
            implementation = 'saved Hunyuan3D 2.1 mesh'
        elif provider == 'hunyuan3d21':
            provenance = hunyuan21(image, mask, output, weights)
            implementation = 'Tencent Hunyuan3D 2.1 shape'
        else:
            raise ValueError(f'unknown mesh provider: {provider}')
        provenance['crop'] = crop
        provenance['rgba_sha256'] = digest(output/'masked_subject.png')
        validity = geometry(output/'car_shape.glb')
        artifact = {'schema': 1, 'provider': provider, 'implementation': implementation,
                    'mesh': 'car_shape.glb', 'mesh_sha256': digest(output/'car_shape.glb'),
                    'units': 'normalized_model_units', 'orientation': 'model_native_unaligned',
                    'coordinate_frame': 'Hunyuan3D local frame; scene fitting establishes world pose and metric scale',
                    'validity': validity,
                    'input_hashes': {'original_image_sha256': digest(output/'inputs/input.jpg'),
                                     'selected_mask_sha256': digest(output/'inputs/car_mask.png')},
                    'timing_s': {**{key: value for key, value in provenance.items() if key.endswith('_seconds')},
                                 'total': time.perf_counter()-start},
                    'provenance': provenance}
        write(output/'subject_mesh.json', artifact)
        record(output, 'subject_mesh', ['inputs/input.jpg', 'inputs/car_mask.png'],
               ['car_shape.glb', 'subject_mesh.json', 'masked_subject.png'], implementation, '1',
               {'provider': provider, 'units': artifact['units'], 'orientation': artifact['orientation'],
                'coordinate_frame': artifact['coordinate_frame']}, time.perf_counter()-start)
        write(output/'stage_status.json', {'stage': 'subject_mesh', 'status': 'complete'})
        return artifact
    except Exception as exc:
        write(output/'stage_status.json', {'stage': 'subject_mesh', 'status': 'failed',
              'error_type': type(exc).__name__, 'reason': str(exc),
              'inputs': {'image': str(image), 'mask': str(mask)}})
        raise


def verify(root):
    stage = check_stage(root, 'subject_mesh')
    data = json.loads((root/'subject_mesh.json').read_text())
    if data.get('schema') != 1 or not data['validity']['valid'] or data['mesh_sha256'] != digest(root/data['mesh']):
        raise ValueError('invalid subject mesh artifact')
    return stage


def main():
    if len(sys.argv) == 3 and sys.argv[1] == 'check':
        print(json.dumps(verify(Path(sys.argv[2]))))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('mask', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--saved-mesh', type=Path)
    parser.add_argument('--provider', default='hunyuan3d21')
    parser.add_argument('--weights', type=Path, default=MODEL_ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.image, args.mask, args.output, args.saved_mesh, args.provider, args.weights)))


if __name__ == '__main__':
    main()
