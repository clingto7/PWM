"""Fit the Hunyuan vehicle to the fixed DA3 camera and SAM silhouette."""
import math
import numpy as np
import cv2
import trimesh
from scipy.optimize import differential_evolution

# BMW Group MINI Cooper S Convertible, 03/2021 technical data (metres).
REAL_SIZE_M = [1.727, 3.876, 1.415]  # width, length, height
REAL_SIZE_SOURCE = 'https://www.press.bmwgroup.com/global/article/attachment/T0325437EN/471396'


def fit(mesh_path, mask, depth, k, right, forward, normal, ground_z, anchor_xy, anchor_depth):
    loaded = trimesh.load(mesh_path, force='scene')
    meshes = []
    for node in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph.get(node)
        meshes.append(loaded.geometry[geometry_name].copy().apply_transform(transform))
    if not meshes:
        raise ValueError('Hunyuan GLB contains no mesh')
    mesh = trimesh.util.concatenate(meshes)
    raw = np.asarray(mesh.vertices, dtype=np.float64)
    # Hunyuan shape output is x=width, y=height, z=length. Centre it;
    # local y is forward and local z is up in the ground-aligned frame.
    local = raw[:, [0, 2, 1]]
    local -= (local.max(axis=0) + local.min(axis=0)) / 2
    local[:, 2] -= local[:, 2].min()
    span = local.max(axis=0) - local.min(axis=0)
    if np.any(span < .01):
        raise ValueError('degenerate vehicle mesh')
    sample = local[np.linspace(0, len(local)-1, min(3500, len(local)), dtype=int)]
    h, w = mask.shape
    target = cv2.resize(mask.astype('uint8'), (w//2, h//2), interpolation=cv2.INTER_NEAREST)
    fx, fy, cx, cy = k[0,0], k[1,1], k[0,2], k[1,2]
    def projected(points, scale, yaw, x, y):
        co, si = math.cos(yaw), math.sin(yaw)
        xyz = points * scale
        wx = xyz[:,0]*co-xyz[:,1]*si+x
        wy = xyz[:,0]*si+xyz[:,1]*co+y
        wz = xyz[:,2]+ground_z
        optical = wx[:,None]*right + wy[:,None]*forward + wz[:,None]*normal
        valid = optical[:,1] > .1
        if not valid.all():
            return None
        u = fx*optical[:,0]/optical[:,1]+cx
        v = -fy*optical[:,2]/optical[:,1]+cy
        return np.column_stack((u,v)), optical[:,1]
    def silhouette(points):
        hull = cv2.convexHull(np.rint(points/2).astype(np.int32))
        image = np.zeros_like(target)
        cv2.fillConvexPoly(image, hull, 1)
        intersection = np.count_nonzero(image & target)
        union = np.count_nonzero(image | target)
        return intersection/union if union else 0
    observed_depth = float(np.median(depth[mask & np.isfinite(depth) & (depth > 0)]))
    # Image evidence determines every fitted variable; real dimensions are absent.
    bounds = [(.55, 2.7), (-math.pi, math.pi), (anchor_xy[0]-2, anchor_xy[0]+2), (max(.5, anchor_xy[1]-2), anchor_xy[1]+4)]
    def objective(v):
        scale, yaw, x, y = v
        projection = projected(sample, scale, yaw, x, y)
        if projection is None:
            return 10
        pixels, optical_depth = projection
        iou = silhouette(pixels)
        depth_error = abs(float(np.median(optical_depth))-observed_depth)/max(observed_depth, 1)
        contact_error = abs(float(np.min(optical_depth))-anchor_depth)/max(anchor_depth, 1)
        return 1-iou + .12*depth_error + .03*contact_error
    result = differential_evolution(objective, bounds, seed=42, maxiter=35, popsize=6, polish=True, workers=1)
    scale, yaw, x, y = map(float, result.x)
    points = local * scale
    # Export local mesh in body coordinates; base pose puts wheel/contact plane on ground.
    points[:,2] -= (points[:,2].max()+points[:,2].min())/2
    # Swapping Y and Z reflects handedness; reverse winding for outward normals.
    fitted = trimesh.Trimesh(vertices=points, faces=mesh.faces[:, ::-1], process=False)
    size = (span*scale).tolist()
    ratios = [size[i]/REAL_SIZE_M[i] for i in range(3)]
    return fitted, {'scale':scale, 'yaw_rad':yaw, 'position_m':[x,y,ground_z+size[2]/2],
                    'size_m':size, 'size_check':{'reference_model':'MINI Cooper S Convertible (2021)',
                    'reference_size_m':REAL_SIZE_M, 'source':REAL_SIZE_SOURCE,
                    'fit_to_reference_ratio':ratios, 'reasonable':all(.7 <= r <= 1.3 for r in ratios)},
                    'fit_objective':float(result.fun), 'observed_car_depth_m':observed_depth,
                    'ground_contact_error_m':0.0}
