"""Fit the Hunyuan vehicle to the fixed DA3 camera and SAM silhouette."""
import math
import numpy as np
import cv2
import trimesh
from scipy.optimize import differential_evolution

# BMW Group MINI Cooper S Convertible, 03/2021 technical data (metres).
REAL_SIZE_M = [1.727, 3.876, 1.415]  # width, length, height
REAL_SIZE_SOURCE = 'https://www.press.bmwgroup.com/global/article/attachment/T0325437EN/471396'


def fit(mesh_path, mask, depth, k, right, forward, normal, ground_z, anchor_xy, anchor_depth, diagnostic_dir=None):
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
    def evidence(v):
        scale, yaw, x, y = v
        projection = projected(sample, scale, yaw, x, y)
        if projection is None:
            return None
        pixels, optical_depth = projection
        iou = silhouette(pixels)
        ix = np.rint(pixels[:, 0]).astype(int)
        iy = np.rint(pixels[:, 1]).astype(int)
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        ix, iy, optical_depth = ix[inside], iy[inside], optical_depth[inside]
        observed = depth[iy, ix]
        valid = mask[iy, ix] & np.isfinite(observed) & (observed > 0)
        depth_error = (float(np.median(np.abs(optical_depth[valid]-observed[valid])))
                       if np.any(valid) else float('inf'))
        contact_error = abs(float(np.min(optical_depth))-anchor_depth) if len(optical_depth) else float('inf')
        return {'silhouette_iou': iou, 'depth_median_absolute_error_m': depth_error,
                'support_contact_depth_error_m': contact_error,
                'depth_sample_count': int(np.count_nonzero(valid))}
    def objective(v):
        metrics = evidence(v)
        if metrics is None or not math.isfinite(metrics['depth_median_absolute_error_m']):
            return 10
        return (1-metrics['silhouette_iou'] + .12*metrics['depth_median_absolute_error_m']/max(observed_depth,1)
                + .03*metrics['support_contact_depth_error_m']/max(anchor_depth,1))
    initial = [1., 0., anchor_xy[0], anchor_xy[1]]
    before = evidence(initial)
    result = differential_evolution(objective, bounds, seed=42, maxiter=35, popsize=6, polish=True, workers=1)
    scale, yaw, x, y = map(float, result.x)
    points = local * scale
    # Export local mesh in body coordinates; base pose puts wheel/contact plane on ground.
    points[:,2] -= (points[:,2].max()+points[:,2].min())/2
    # Swapping Y and Z reflects handedness; reverse winding for outward normals.
    fitted = trimesh.Trimesh(vertices=points, faces=mesh.faces[:, ::-1], process=False)
    size = (span*scale).tolist()
    ratios = [size[i]/REAL_SIZE_M[i] for i in range(3)]
    after = evidence(result.x)
    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(exist_ok=True)
        for name, pose in (('before', initial), ('after', result.x)):
            candidate = projected(sample, *pose)
            visual = np.zeros((h, w, 3), dtype=np.uint8)
            visual[mask, 1] = 200
            if candidate is not None:
                hull_pixels = cv2.convexHull(np.rint(candidate[0]).astype(np.int32))
                outline = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(outline, hull_pixels, 1)
                visual[outline.astype(bool), 2] = 200
            cv2.imwrite(str(diagnostic_dir/f'{name}.png'), visual)
    # No class classifier is part of the observation contract. Segmentation confidence
    # cannot establish a semantic size class, so the frozen prior stays ineligible.
    alternatives = [float(result.fun)]
    for factor in (.8, 1.2):
        trial = result.x.copy(); trial[0] *= factor
        alternatives.append(float(objective(trial)))
    uncertainty = {'method': 'objective profile at +/-20 percent scale',
                   'relative_scale_probe': [.8, 1., 1.2], 'objective': [alternatives[1], alternatives[0], alternatives[2]]}
    return fitted, {'scale':scale, 'yaw_rad':yaw, 'position_m':[x,y,ground_z+size[2]/2],
                    'size_m':size, 'size_check':{'reference_model':'MINI Cooper S Convertible (2021)',
                    'reference_size_m':REAL_SIZE_M, 'source':REAL_SIZE_SOURCE,
                    'fit_to_reference_ratio':ratios, 'reasonable':all(.7 <= r <= 1.3 for r in ratios)},
                    'fit_objective':float(result.fun), 'observed_car_depth_m':observed_depth,
                    'ground_contact_error_m':0.0,
                    'diagnostics': {'before': before, 'after': after},
                    'scale_uncertainty': uncertainty,
                    'no_prior': {'scale': scale, 'objective': float(result.fun)},
                    'weak_category_prior': {'eligible': False, 'classification_evidence_threshold': .9,
                                            'reason': 'observation has no semantic class evidence', 'result': None}}
