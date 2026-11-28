"""CUDA triangle renderer for immutable scene artifacts and a saved trajectory.

The frame contract matches raster_replay: uint16 instance IDs and optical
depth in millimetres. Geometry is uploaded once and the controlled mesh alone
is transformed each frame. CUDA tensors remain available until readback.
"""

import math
import time
from pathlib import Path


def geometry_arrays(root, geometry):
    import numpy as np
    import trimesh

    vertex_parts = [np.asarray(geometry.vertices, dtype=np.float32)]
    triangle_parts = [np.asarray(geometry.indices, dtype=np.int32).reshape(-1, 3)]
    entity_parts = [np.full(len(triangle_parts[0]), 1, dtype=np.int32)]
    offset = len(vertex_parts[0])
    for component, (vertices, indices) in zip(geometry.components, geometry.obstacle_meshes):
        if not indices:
            continue
        part = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
        vertex_parts.append(part)
        triangle_parts.append(faces + offset)
        entity_parts.append(np.full(len(faces), component['instance_id'], dtype=np.int32))
        offset += len(part)
    visual = trimesh.load(Path(root)/'fitted_car.obj', force='mesh', process=False)
    car_vertices = np.asarray(visual.vertices, dtype=np.float32)
    car_faces = np.asarray(visual.faces, dtype=np.int32)
    if not len(car_vertices) or not len(car_faces):
        raise ValueError('empty controlled-body visual mesh')
    triangle_parts.append(car_faces + offset)
    entity_parts.append(np.full(len(car_faces), 2, dtype=np.int32))
    return (np.concatenate(vertex_parts), car_vertices,
            np.concatenate(triangle_parts), np.concatenate(entity_parts))


def render_saved(root, output, params, yaw_deg=0):
    import json
    import numpy as np
    import torch
    import nvdiffrast.torch as dr
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from artifacts import SceneArtifacts

    if not torch.cuda.is_available():
        raise RuntimeError('gpu_raster requires CUDA')
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'gpu raster output is not empty: {output}')
    geometry = SceneArtifacts(root).load()
    static, car, triangles, entities = geometry_arrays(root, geometry)
    rows = json.loads((root/'trajectory.json').read_text())
    frames = int(float(params['seconds']) * int(params['video_fps']))
    width, height = 504, 280
    near, far = .1, max(100.0, float(params['radius_m'])*4)
    camera = geometry.scene['camera']
    forward = np.asarray(camera['forward'], dtype=np.float32)
    up = np.asarray(camera['up'], dtype=np.float32)
    if yaw_deg:
        forward = Rotation.from_rotvec(up.astype(float)*math.radians(yaw_deg)).apply(forward).astype(np.float32)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    basis = torch.as_tensor(np.stack((right, up, forward)), device='cuda')
    intrinsic = camera['intrinsics']
    fx, fy, cx, cy = (float(intrinsic[0][0]), float(intrinsic[1][1]),
                      float(intrinsic[0][2]), float(intrinsic[1][2]))
    static_gpu = torch.as_tensor(static, device='cuda')
    car_gpu = torch.as_tensor(car, device='cuda')
    faces_gpu = torch.as_tensor(triangles, device='cuda')
    lookup = torch.as_tensor(np.r_[0, entities], device='cuda')
    output.mkdir(parents=True, exist_ok=True)
    ids_dir, depth_dir = output/'instance_id', output/'depth'
    ids_dir.mkdir(); depth_dir.mkdir()
    context = dr.RasterizeCudaContext()
    frame_times, readback_times, png_times = [], [], []

    def clip_positions(world):
        view = world @ basis.T
        x, y, z = view.unbind(-1)
        return torch.stack((2*fx/width*x + (2*cx/width-1)*z,
                            2*fy/height*y + (1-2*cy/height)*z,
                            (far+near)/(far-near)*z - 2*far*near/(far-near), z), dim=-1)

    with torch.inference_mode():
        for frame in range(frames):
            row = rows[frame*8]
            torch.cuda.synchronize(); started = time.perf_counter()
            q = torch.as_tensor(row[3:7], dtype=torch.float32, device='cuda')
            qv = q[:3].expand_as(car_gpu)
            t = 2*torch.cross(qv, car_gpu, dim=-1)
            car_world = car_gpu + q[3]*t + torch.cross(qv, t, dim=-1)
            car_world += torch.as_tensor(row[:3], dtype=torch.float32, device='cuda')
            clip = clip_positions(torch.cat((static_gpu, car_world)))[None]
            raster, _ = dr.rasterize(context, clip, faces_gpu, [height, width], grad_db=False)
            raster = raster[0].flip(0)
            triangle = raster[..., 3].to(torch.long)
            ids = lookup[triangle]
            z_ndc = raster[..., 2]
            depth = torch.where(triangle > 0,
                2*far*near/(far+near-z_ndc*(far-near)), 0.0)
            torch.cuda.synchronize(); frame_times.append(time.perf_counter()-started)
            started = time.perf_counter()
            ids_cpu = ids.cpu().numpy().astype(np.uint16)
            depth_cpu = (depth.cpu().numpy()*1000).clip(0, 65535).astype(np.uint16)
            readback_times.append(time.perf_counter()-started)
            started = time.perf_counter()
            name = f'{frame:04d}.png'
            Image.fromarray(ids_cpu).save(ids_dir/name)
            Image.fromarray(depth_cpu).save(depth_dir/name)
            png_times.append(time.perf_counter()-started)
    return {'frame_times_s': frame_times, 'readback_times_s': readback_times,
            'png_write_times_s': png_times, 'triangles': len(triangles),
            'gpu': torch.cuda.get_device_name(),
            'depth_method': 'nvdiffrast optical z depth; zero on uncovered pixels',
            'entity_id_method': 'triangle-to-entity lookup; zero on uncovered pixels'}
