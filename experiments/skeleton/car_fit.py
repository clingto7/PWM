"""Fit the Hunyuan vehicle to the fixed DA3 camera and SAM silhouette."""
import math
import time
import numpy as np


def visible_fit_score(iou, depth_mae, contact_gap, observed_depth, anchor_depth):
    if depth_mae is None or contact_gap is None:
        return float('inf')
    return (1-iou + .12*depth_mae/max(observed_depth, 1)
            + .03*abs(contact_gap)/max(anchor_depth, 1))


def support_gap(bottom, pose, ground_z, support):
    """Signed bottom-surface gap to reconstructed support cells under the pose."""
    values = support_clearance(bottom, pose, ground_z, support)
    return None if values is None else values[0]


def support_clearance(bottom, pose, ground_z, support):
    """Median and minimum vertex clearance over evidenced support cells."""
    grid, valid_grid, origin, cell = support
    scale, yaw, x, y = pose
    co, si = math.cos(yaw), math.sin(yaw)
    bx = (bottom[:, 0]*co-bottom[:, 1]*si)*scale+x
    by = (bottom[:, 0]*si+bottom[:, 1]*co)*scale+y
    ix = np.rint((bx-origin)/cell).astype(int)
    iy = np.rint(by/cell).astype(int)
    inside = (ix >= 0) & (ix < grid.shape[1]) & (iy >= 0) & (iy < grid.shape[0])
    ix, iy = ix[inside], iy[inside]
    observed = valid_grid[iy, ix]
    if not np.any(observed):
        return None
    gaps = ground_z+bottom[inside][observed, 2]*scale-grid[iy[observed], ix[observed]]
    return float(np.median(gaps)), float(np.min(gaps))


def raster_depth(vertices, faces, k, right, forward, normal, scale, yaw, x, y, ground_z, shape):
    """CPU triangle z-buffer in the original camera, with perspective-correct depth."""
    h, w = shape
    co, si = math.cos(yaw), math.sin(yaw)
    xyz = vertices * scale
    wx = xyz[:, 0]*co - xyz[:, 1]*si + x
    wy = xyz[:, 0]*si + xyz[:, 1]*co + y
    wz = xyz[:, 2] + ground_z
    optical = wx[:, None]*right + wy[:, None]*forward + wz[:, None]*normal
    z = optical[:, 1]
    u = k[0, 0]*optical[:, 0]/np.maximum(z, .001)+k[0, 2]
    v = -k[1, 1]*optical[:, 2]/np.maximum(z, .001)+k[1, 2]
    buffer = np.full((h, w), np.inf)
    for tri in faces:
        if np.any(z[tri] <= .1):
            continue
        px, py, dz = u[tri], v[tri], z[tri]
        x0, x1 = max(0, int(math.floor(px.min()))), min(w-1, int(math.ceil(px.max())))
        y0, y1 = max(0, int(math.floor(py.min()))), min(h-1, int(math.ceil(py.max())))
        if x0 > x1 or y0 > y1:
            continue
        denominator = ((py[1]-py[2])*(px[0]-px[2]) +
                       (px[2]-px[1])*(py[0]-py[2]))
        if abs(denominator) < 1e-8:
            continue
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
        xx = xx + .5; yy = yy + .5
        a = ((py[1]-py[2])*(xx-px[2]) + (px[2]-px[1])*(yy-py[2]))/denominator
        b = ((py[2]-py[0])*(xx-px[2]) + (px[0]-px[2])*(yy-py[2]))/denominator
        c = 1-a-b
        inside = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
        metric = 1/(a/dz[0] + b/dz[1] + c/dz[2])
        patch = buffer[y0:y1+1, x0:x1+1]
        np.minimum(patch, np.where(inside, metric, np.inf), out=patch)
    return buffer


def fit(mesh_path, mask, depth, k, right, forward, normal, ground_z, anchor_xy, anchor_depth, diagnostic_dir=None, support=None):
    import cv2
    import trimesh
    from scipy.optimize import differential_evolution
    started = time.perf_counter()
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
    bottom = local[local[:, 2] <= np.quantile(local[:, 2], .02)]
    def contact_error(v):
        if support is None:
            return float(np.min(bottom[:, 2])*v[0])
        return support_gap(bottom, v, ground_z, support)
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
        gap = contact_error(v)
        contact_penalty = abs(gap) if gap is not None else float('inf')
        return {'silhouette_iou': iou, 'depth_median_absolute_error_m': depth_error,
                'support_contact_error_m': contact_penalty,
                'depth_sample_count': int(np.count_nonzero(valid))}
    def objective(v):
        metrics = evidence(v)
        if metrics is None or not math.isfinite(metrics['depth_median_absolute_error_m']):
            return 10
        return (1-metrics['silhouette_iou'] + .12*metrics['depth_median_absolute_error_m']/max(observed_depth,1)
                + .03*metrics['support_contact_error_m']/max(anchor_depth,1))
    initial = [1., 0., anchor_xy[0], anchor_xy[1]]
    before = evidence(initial) or {'valid_projection': False}
    search_started = time.perf_counter()
    result = differential_evolution(objective, bounds, seed=42, maxiter=35, popsize=6, polish=True, workers=1)
    # A vertex depth sample can prefer the rear-facing side of a mesh. Include
    # the original silhouette/median-depth optimum and rank both headings with
    # a visible-surface z-buffer against the actual image evidence.
    def silhouette_objective(v):
        projection = projected(sample, *v)
        if projection is None:
            return 10
        pixels, optical_depth = projection
        depth_error = abs(float(np.median(optical_depth))-observed_depth)/max(observed_depth, 1)
        anchor_error = abs(float(np.min(optical_depth))-anchor_depth)/max(anchor_depth, 1)
        return 1-silhouette(pixels)+.12*depth_error+.03*anchor_error
    silhouette_result = differential_evolution(silhouette_objective, bounds,
        seed=42, maxiter=35, popsize=6, polish=True, workers=1)
    search_seconds = time.perf_counter()-search_started
    render_started = time.perf_counter()
    candidates = []
    for source, base in (('sampled_depth', result.x), ('silhouette_depth', silhouette_result.x)):
        for flip in (False, True):
            pose = base.copy()
            if flip:
                pose[1] = (pose[1]+2*math.pi) % (2*math.pi)-math.pi
            rendered = raster_depth(local, mesh.faces, k, right, forward, normal,
                                    *pose, ground_z, mask.shape)
            visible = np.isfinite(rendered)
            overlap = mask & visible & np.isfinite(depth) & (depth > 0)
            iou = np.count_nonzero(mask & visible)/max(1, np.count_nonzero(mask | visible))
            mae = float(np.mean(np.abs(rendered[overlap]-depth[overlap]))) if np.any(overlap) else None
            score = visible_fit_score(iou, mae, contact_error(pose), observed_depth, anchor_depth)
            candidates.append({'source': source, 'flipped': flip, 'score': score,
                               'visible_iou': float(iou), 'visible_depth_mae_m': mae,
                               'pose': pose, 'rendered': rendered})
    winner = min(candidates, key=lambda item: item['score'])
    render_seconds = time.perf_counter()-render_started
    result.x = winner['pose']
    result.fun = float(objective(result.x))
    scale, yaw, x, y = map(float, result.x)
    points = local * scale
    # Export local mesh in body coordinates; base pose puts wheel/contact plane on ground.
    points[:,2] -= (points[:,2].max()+points[:,2].min())/2
    # Swapping Y and Z reflects handedness; reverse winding for outward normals.
    fitted = trimesh.Trimesh(vertices=points, faces=mesh.faces[:, ::-1], process=False)
    size = (span*scale).tolist()
    after = evidence(result.x) or {'valid_projection': False}
    def rendered_metrics(rendered, pose, metrics):
        visible = np.isfinite(rendered)
        overlap = mask & visible & np.isfinite(depth) & (depth > 0)
        metrics['visible_depth_pixel_count'] = int(overlap.sum())
        metrics['visible_depth_mae_m'] = (float(np.mean(np.abs(rendered[overlap]-depth[overlap])))
                                          if np.any(overlap) else None)
        metrics['rendered_silhouette_iou'] = float(np.count_nonzero(mask & visible) /
            max(1, np.count_nonzero(mask | visible)))
        metrics['support_contact_gap_m'] = contact_error(pose)
    before_rendered = raster_depth(local, mesh.faces, k, right, forward, normal,
                                   *initial, ground_z, mask.shape)
    rendered_metrics(before_rendered, initial, before)
    rendered_metrics(winner['rendered'], result.x, after)
    for pose, metrics in ((initial, before), (result.x, after)):
        clearance = support_clearance(bottom, pose, ground_z, support) if support is not None else None
        metrics['support_penetration_m'] = max(0.0, -clearance[1]) if clearance is not None else None
    # Keep the established .80 first-image silhouette gate and the passive
    # trial's initial contact limits; pixel depth has no historical error gate.
    from gates import THRESHOLDS
    gap = after['support_contact_gap_m']
    trust_checks = {'silhouette': after['rendered_silhouette_iou'] >= THRESHOLDS['initial_mask_iou_min'],
                    'pixel_depth': after['visible_depth_pixel_count'] > 0 and
                                   after['visible_depth_mae_m'] is not None and
                                   math.isfinite(after['visible_depth_mae_m']),
                    'support_contact': gap is not None and -.10 <= gap <= .15 and
                                       after['support_penetration_m'] is not None,
                    'finite_score': math.isfinite(winner['score'])}
    trusted = all(trust_checks.values())
    visual_write_seconds = 0.0
    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(exist_ok=True)
        for name, pose, rendered, metrics in (('before', initial, before_rendered, before),
                                              ('after', result.x, winner['rendered'], after)):
            visual = np.zeros((h, w, 3), dtype=np.uint8)
            visual[mask, 1] = 200
            visual[np.isfinite(rendered), 2] = 200
            gap_label = metrics['support_contact_gap_m']
            cv2.putText(visual, f'gap {gap_label:.3f} m' if gap_label is not None else 'gap unknown',
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            write_started = time.perf_counter()
            if not cv2.imwrite(str(diagnostic_dir/f'{name}.png'), visual):
                raise OSError('failed to write silhouette diagnostic')
            visual_write_seconds += time.perf_counter()-write_started
            valid_depth = mask & np.isfinite(rendered) & np.isfinite(depth) & (depth > 0)
            residual = np.zeros((h,w), dtype=np.uint8)
            residual[valid_depth] = np.clip(np.abs(rendered[valid_depth]-depth[valid_depth])*50, 0, 255).astype(np.uint8)
            write_started = time.perf_counter()
            if not cv2.imwrite(str(diagnostic_dir/f'{name}_depth_error.png'), residual):
                raise OSError('failed to write depth diagnostic')
            visual_write_seconds += time.perf_counter()-write_started
    alternatives = [float(result.fun)]
    for factor in (.8, 1.2):
        trial = result.x.copy(); trial[0] *= factor
        alternatives.append(float(objective(trial)))
    uncertainty = {'method': 'objective profile at +/-20 percent scale',
                   'relative_scale_probe': [.8, 1., 1.2], 'objective': [alternatives[1], alternatives[0], alternatives[2]]}
    timings = {'subject_scale_pose_search': search_seconds,
               'candidate_rendering': render_seconds,
               'visual_writing': visual_write_seconds,
               'fit_total': time.perf_counter()-started}
    return fitted, {'scale':scale, 'yaw_rad':yaw, 'position_m':[x,y,ground_z+size[2]/2],
                    'size_m':size, 'fit_trusted':trusted, 'trust_checks':trust_checks,
                    'timings_s':timings,
                    'fit_objective':float(result.fun), 'observed_car_depth_m':observed_depth,
                    'ground_contact_error_m':after['support_contact_gap_m'],
                    'diagnostics': {'before': before, 'after': after},
                    'visible_candidate_ranking': [{key: value for key, value in candidate.items() if key not in ('pose','rendered')}
                                                  for candidate in candidates],
                    'scale_uncertainty': uncertainty}
