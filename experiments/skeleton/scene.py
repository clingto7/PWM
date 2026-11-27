"""Reconstruct scene artifacts from the four bundled source inputs.

The module writes scene.json, heightmap.npz, obstacles.npz, the fitted car OBJ,
and its collision hull. Its return value carries prepared geometry for the
solver and renderer adapters; the saved files remain the portable interface.
"""
import json
import math
import time
from dataclasses import dataclass
import numpy as np
from PIL import Image
from car_fit import fit
from scene_support import classify_support, select_support_plane, static_components, regularize_support_heights, contact_support_fit


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


@dataclass(frozen=True)
class Reconstruction:
    artifacts: object
    ground_time: float
    scene_time: float


def reconstruct(inputs, args, out, start):
    d = np.load(inputs['da3'])
    depth = d['depth'][0]
    k = d['intrinsics'][0]
    h, w = depth.shape
    if (h, w) != (280, 504):
        raise ValueError('expected DA3 504x280 metric depth')
    car = np.asarray(Image.open(inputs['sam_car']).convert('L').resize((w, h), Image.Resampling.NEAREST)) > 127
    ys, xs = np.indices(depth.shape)
    # Camera coordinates: x right, y away, z up; DA3 optical y points down.
    xx = (xs - k[0, 2]) / k[0, 0] * depth
    yy = depth
    zz = -(ys - k[1, 2]) / k[1, 1] * depth
    eligible = ~car & np.isfinite(depth) & (depth > 2) & (depth < args.radius)
    coeff, road, residual = select_support_plane(
        xx, yy, zz, eligible, car, args.ground_inlier)
    a, b, c = map(float, coeff)
    pitch = math.degrees(math.atan(b))
    # At the camera x=y=z=0, the fitted plane is c below the camera.
    height = abs(c) / math.sqrt(1 + a*a + b*b)
    normal = np.array([-a,-b,1.0]); normal /= np.linalg.norm(normal)
    forward = np.array([0.0,1.0,0.0]); forward -= np.dot(forward,normal)*normal; forward /= np.linalg.norm(forward)
    right = np.cross(forward,normal)
    def world(v):
        return [float(np.dot(v,right)),float(np.dot(v,forward)),float(np.dot(v,normal))]
    flat_z = float(c * normal[2])
    camera_forward = world(np.array([0.,1.,0.]))
    camera_up = world(np.array([0.,0.,1.]))
    ground_time = time.perf_counter() - start
    grid_start = time.perf_counter()
    n = int(math.ceil(2 * args.radius / args.cell)) + 1
    vertices = []
    indices = []
    valid = []
    half_angle = math.radians(args.angle / 2)
    # Keep a plane as an interpolation reference; only evidenced cells traverse.
    origin_x = -(n-1)/2*args.cell
    grid_z = np.full((n,n),flat_z,dtype=np.float64)
    provenance = np.full((n,n),2,dtype=np.uint8)  # 1 depth, 2 ray/plane, 3 car fill
    depth_road = road & (depth <= args.near_depth)
    camera_points = np.stack([xx[depth_road],yy[depth_road],zz[depth_road]],axis=1)
    wx = camera_points @ right
    wy = camera_points @ forward
    wz = camera_points @ normal
    gi = np.rint((wx-origin_x)/args.cell).astype(int)
    gj = np.rint(wy/args.cell).astype(int)
    in_grid = (gi>=0)&(gi<n)&(gj>=0)&(gj<n)
    sums = np.zeros((n,n),dtype=np.float64)
    counts = np.zeros((n,n),dtype=np.int32)
    np.add.at(sums,(gj[in_grid],gi[in_grid]),wz[in_grid])
    np.add.at(counts,(gj[in_grid],gi[in_grid]),1)
    measured = counts>0
    grid_z[measured] = sums[measured]/counts[measured]
    provenance[measured] = 1
    valid_grid = np.zeros((n,n), dtype=bool)
    car_grid = np.zeros((n,n), dtype=bool)
    for j in range(n):
        y = j * args.cell
        for i in range(n):
            x = origin_x + i * args.cell
            inside = y >= args.cell and x*x + y*y <= args.radius**2 and abs(math.atan2(x, y)) <= half_angle
            valid.append(inside)
            valid_grid[j,i] = inside
            # Project the ground grid back to the original image. Its car mask
            # marks an inferred footprint, never a depth source.
            camera_point = x*right + y*forward + flat_z*normal
            if camera_point[1] > 0:
                u = int(round(k[0,0]*camera_point[0]/camera_point[1]+k[0,2]))
                v = int(round(-k[1,1]*camera_point[2]/camera_point[1]+k[1,2]))
                if 0<=u<w and 0<=v<h and car[v,u]:
                    grid_z[j,i]=flat_z
                    provenance[j,i]=3
                    car_grid[j,i]=True
            vertices.append([x,y,float(grid_z[j,i])])
    # Support is measured where back-projected depth agrees with the robust
    # fitted layer. A small subject occlusion may be interpolated; the rest is
    # explicitly unknown and cannot be traversed.
    sector_grid = valid_grid.copy()
    support_codes, traversable, grid_z = classify_support(
        measured & valid_grid, car_grid, valid_grid, grid_z, flat_z)
    grid_z = regularize_support_heights(grid_z, support_codes>0)
    provenance = support_codes
    valid_grid = traversable
    for j in range(n):
        for i in range(n):
            vertices[j*n+i][2] = float(grid_z[j,i])
    # The road-only heightmap cannot see roadside surfaces. Back-project all
    # non-car DA3 pixels in the trusted range and vote in world-grid cells.
    surface = (~car) & np.isfinite(depth) & (depth > 2) & (depth <= args.radius)
    points = np.stack([xx[surface], yy[surface], zz[surface]], axis=1)
    ox, oy, oz = points @ right, points @ forward, points @ normal
    oi = np.rint((ox-origin_x)/args.cell).astype(int)
    oj = np.rint(oy/args.cell).astype(int)
    in_bounds = (oi>=0)&(oi<n)&(oj>=0)&(oj<n)
    oi,oj,oz = oi[in_bounds],oj[in_bounds],oz[in_bounds]
    selected = sector_grid[oj,oi] & ~car_grid[oj,oi]
    oi,oj,oz = oi[selected],oj[selected],oz[selected]
    votes = np.zeros((n,n),dtype=np.int32)
    tops = np.full((n,n),flat_z,dtype=np.float64)
    surface_provenance=np.full((n,n),2,dtype=np.uint8)  # 1 observed, 2 plane, 3 car fill
    np.maximum.at(tops,(oj,oi),oz)
    surface_provenance[oj,oi]=1
    surface_provenance[car_grid]=3
    tops[car_grid]=flat_z
    raised = oz-flat_z > args.obstacle_height
    np.add.at(votes,(oj[raised],oi[raised]),1)
    candidates = sector_grid & ~car_grid & (tops-flat_z > args.obstacle_height) & (votes >= args.obstacle_min_pixels)
    component_ids, components = static_components(
        candidates, votes, args.obstacle_min_cells, args.obstacle_min_pixels, traversable)
    for component in components:
        cells=np.argwhere(component_ids==component['instance_id'])
        component['area_m2']=component['cells']*args.cell**2
        component['max_height_above_plane_m']=float(max(tops[cy,cx]-flat_z for cy,cx in cells))
        component['collision_enabled']=not args.obstacles_render_only
        component['confidence']='reliable'
    uncertain_candidate_cells=int(np.count_nonzero(candidates & (component_ids==0)))
    road_points=np.stack([xx[road],yy[road],zz[road]],axis=1)
    ri=np.rint(((road_points @ right)-origin_x)/args.cell).astype(int)
    rj=np.rint((road_points @ forward)/args.cell).astype(int)
    road_cells=np.zeros((n,n),dtype=bool)
    rb=(ri>=0)&(ri<n)&(rj>=0)&(rj<n)
    road_cells[rj[rb],ri[rb]]=True
    road_cells &= sector_grid & ~car_grid
    road_count=int(road_cells.sum())
    road_false=int(np.count_nonzero(road_cells & (component_ids>0)))
    false_ratio=road_false/road_count if road_count else None
    np.savez_compressed(out/'obstacles.npz',instance_id=component_ids,candidate=candidates.astype(np.uint8),road_cells=road_cells.astype(np.uint8),car_footprint=car_grid.astype(np.uint8))
    for j in range(n-1):
        for i in range(n-1):
            q = j*n+i
            if valid_grid[j:j+2,i:i+2].all():
                indices.extend([q,q+1,q+n,q+1,q+n+1,q+n])
    if not indices:
        raise ValueError('empty scene sector')
    np.savez_compressed(out/'heightmap.npz', z_m=grid_z.astype(np.float32), ground_z_m=grid_z.astype(np.float32), surface_z_m=tops.astype(np.float32), ground_z_exact_m=grid_z, surface_z_exact_m=tops, valid=valid_grid.astype(np.uint8), provenance=provenance, surface_provenance=surface_provenance, surface_votes=votes, sector=sector_grid.astype(np.uint8), support_evidence=support_codes, traversable=traversable.astype(np.uint8))
    # The visible car is excluded from both mesh and collision generation; the fitted
    # plane spans its footprint, including pixels hidden by the original car.
    ys_car, xs_car = np.nonzero(car)
    if not len(xs_car):
        raise ValueError('empty SAM car mask')
    x0, x1 = int(xs_car.min()), int(xs_car.max())
    y0, y1 = int(ys_car.min()), int(ys_car.max())
    # The mask's lowest road-contact pixels supply a ground ray. Intersect
    # that ray with the fitted camera-space plane instead of using car-body
    # depth, which belongs to surfaces above the road.
    lower = xs_car[ys_car >= y1-2]
    u_contact = float(np.median(lower))
    ray_x = (u_contact-k[0,2])/k[0,0]
    ray_z = -(y1-k[1,2])/k[1,1]
    denominator = ray_z-a*ray_x-b
    if abs(denominator) < 1e-6:
        raise ValueError('car contact ray is parallel to fitted ground')
    car_depth = c/denominator
    if not 0 < car_depth <= args.radius:
        raise ValueError('car ground contact lies outside reconstruction sector')
    anchor_x,anchor_y,_ = world(np.array([ray_x*car_depth,car_depth,ray_z*car_depth]))
    contact_fit=contact_support_fit(xx, yy, zz, road,
        (ray_x*car_depth,car_depth,ray_z*car_depth))
    projected_width = (x1-x0+1)/k[0,0]*car_depth
    projected_height = (y1-y0+1)/k[1,1]*car_depth
    def projected_box_bbox(center,width,length,box_h):
        projected=[]
        for dx in (-width/2,width/2):
            for dy in (-length/2,length/2):
                for dz in (-box_h/2,box_h/2):
                    wx,wy,wz=center[0]+dx,center[1]+dy,center[2]+dz
                    px,py,pz=wx*right+wy*forward+wz*normal
                    if py <= 0:
                        return None
                    projected.append((k[0,0]*px/py+k[0,2],-k[1,1]*pz/py+k[1,2]))
        return [min(p[0] for p in projected),min(p[1] for p in projected),max(p[0] for p in projected),max(p[1] for p in projected)]
    target=[x0,y0,x1+1,y1+1]
    def bbox_iou(candidate):
        ix=max(0,min(candidate[2],target[2])-max(candidate[0],target[0]))
        iy=max(0,min(candidate[3],target[3])-max(candidate[1],target[1]))
        intersection=ix*iy
        area=(candidate[2]-candidate[0])*(candidate[3]-candidate[1])
        target_area=(target[2]-target[0])*(target[3]-target[1])
        return intersection/(area+target_area-intersection)
    best=None
    for width_scale in (.7,.85,1.,1.15):
        for length_scale in (.5,.75,1.,1.25,1.5,2.):
            for height_scale in (.6,.8,1.,1.2,1.4):
                width=projected_width*width_scale
                length=projected_width*length_scale
                box_h=projected_height*height_scale
                # The near face touches the mask's lower-edge ground ray.
                center=[anchor_x,anchor_y+length/2,flat_z+box_h/2]
                bbox=projected_box_bbox(center,width,length,box_h)
                if bbox is None:
                    continue
                score=bbox_iou(bbox)
                if best is None or score>best[0]:
                    best=(score,width,length,box_h,center,bbox)
    if best is None:
        raise ValueError('unable to project box into original camera')
    category_evidence = None
    if getattr(args, 'bundle', None) and (args.bundle/'classification.json').is_file():
        from stage_contracts import check as check_stage
        import shutil
        check_stage(args.bundle, 'classification')
        category_evidence = json.loads((args.bundle/'classification.json').read_text())
        shutil.copyfile(args.bundle/'classification.json', out/'classification.json')
    fitted_mesh, mesh_fit = fit(inputs['hunyuan_mesh'], car, depth, k, right, forward, normal,
                                flat_z, (anchor_x,anchor_y), car_depth, out/'fit_visuals',
                                category_evidence, (grid_z, valid_grid, origin_x, args.cell),
                                getattr(args, 'use_weak_prior', False))
    write_json(out/'fit_diagnostics.json', mesh_fit)
    fitted_mesh.export(out/'fitted_car.obj')
    hull = fitted_mesh.convex_hull
    hull.export(out/'car_collision_hull.obj')
    width,length,box_h = mesh_fit['size_m']
    car_pos = mesh_fit['position_m']
    yaw = mesh_fit['yaw_rad']
    car_orientation = [0.0, 0.0, math.sin(yaw/2), math.cos(yaw/2)]
    mesh_volume = float(fitted_mesh.volume)
    if not math.isfinite(mesh_volume) or mesh_volume <= 0:
        raise ValueError('Hunyuan vehicle mesh has invalid enclosed volume')
    mass = args.density * mesh_volume
    obstacle_meshes=[]
    for component in components:
        mesh_vertices=[]; mesh_indices=[]
        for cy,cx in np.argwhere(component_ids==component['instance_id']):
            x=origin_x+cx*args.cell; y=cy*args.cell
            bottom=flat_z; top=float(tops[cy,cx])
            half=args.cell/2
            q=len(mesh_vertices)
            mesh_vertices.extend([[x+dx*half,y+dy*half,z] for z in (bottom,top) for dy in (-1,1) for dx in (-1,1)])
            mesh_indices.extend(q+t for t in (0,2,1,1,2,3,4,5,6,5,7,6,0,1,4,1,5,4,2,6,3,3,6,7,0,4,2,2,4,6,1,3,5,3,7,5))
        obstacle_meshes.append((mesh_vertices,mesh_indices))
    scene = {'camera': {'intrinsics': k.tolist(), 'position_m': [0,0,0], 'forward': camera_forward, 'up': camera_up, 'fixed': True}, 'ground': {'plane_z_ax_by_c': [a,b,c], 'flat_world_z_m': flat_z, 'height_m': height, 'pitch_deg': pitch, 'rms_m': residual, 'contact_fit':contact_fit, 'sample_count': int(road.sum()), 'cell_m': args.cell, 'grid_size': n, 'heightmap_method': {'near': 'metric DA3 support inlier back-projection, cell mean with isolated-spike median filter', 'far': 'unknown outside measured support', 'car': 'local interpolation only where neighboring support is measured', 'near_depth_m': args.near_depth, 'ground_inlier_m': args.ground_inlier, 'provenance_codes': {'unknown':0,'visible':1,'interpolated':2}, 'cell_counts': {'near_depth':int(np.count_nonzero(provenance==1)), 'interpolated':int(np.count_nonzero(provenance==2)), 'unknown':int(np.count_nonzero(provenance==0))}}}, 'sector': {'radius_m': args.radius, 'angle_deg': args.angle, 'invisible_wall': True}, 'car': {'image_bbox_xyxy': [x0,y0,x1,y1], 'excluded_image_pixels': int(car.sum()), 'footprint_filled_with_plane': True, 'ground_contact_ray': {'pixel_uv': [u_contact,y1], 'plane_intersection_depth_m': car_depth, 'world_xy_m': [anchor_x,anchor_y]}, 'mesh_fit': mesh_fit, 'initial_orientation_xyzw': car_orientation, 'collision_shape': 'convex hull of fitted Hunyuan mesh', 'box_fit': {'method': 'ground-edge anchor and projected bounding-box search', 'reference_bbox_xyxy': target, 'predicted_bbox_xyxy': best[5], 'predicted_bbox_iou': best[0], 'width_from_bbox_at_ground_depth_m': projected_width, 'height_from_bbox_at_ground_depth_m': projected_height}, 'depth_m': car_depth, 'size_m': [width,length,box_h], 'volume_m3': mesh_volume, 'density_kg_m3': args.density, 'mass_kg': mass, 'initial_position_m': car_pos}, 'mesh': {'vertices': len(vertices), 'triangles': len(indices)//3}}
    scene['geometry_diagnostics'] = {'original_view_ground_rms_m':residual, 'original_view_support_rms_m':float(np.sqrt(np.mean((zz[road]-(a*xx[road]+b*yy[road]+c))**2))), 'ground_inlier_pixels':int(road.sum()), 'measured_height_range_m':[float(np.min(grid_z[measured])) if measured.any() else None,float(np.max(grid_z[measured])) if measured.any() else None], 'static_candidate_count':len(components), 'uncertain_candidate_cells':uncertain_candidate_cells, 'static_collider_count':sum(c['collision_enabled'] for c in components)}
    scene['support_evidence'] = {'file':'heightmap.npz', 'codes':{'unknown':0,'visible':1,'interpolated':2}, 'visible_cells':int(np.count_nonzero(support_codes==1)), 'interpolated_cells':int(np.count_nonzero(support_codes==2)), 'unknown_cells':int(np.count_nonzero(support_codes==0)), 'traversable_cells':int(traversable.sum()), 'boundary':'edges of traversable cells'}
    scene['entities'] = [{'id': 1, 'role': 'support', 'motion': 'fixed', 'collider': 'heightmap.npz'},
                         {'id': 2, 'role': 'controlled_subject', 'motion': 'dynamic',
                          'collider': 'car_collision_hull.obj', 'visual_mesh': 'fitted_car.obj',
                          'volume_m3': mesh_volume, 'density_kg_m3': args.density, 'mass_kg': mass}] + [
                         {'id': item['instance_id'], 'role': 'static_entity', 'motion': 'fixed',
                          'collision_enabled': item['collision_enabled'], 'collider': 'obstacles.npz'}
                         for item in components]
    scene['obstacles']={'components':components,'component_count':len(components),'detection':{'height_above_plane_m':args.obstacle_height,'min_pixels_per_cell':args.obstacle_min_pixels,'min_cells_per_component':args.obstacle_min_cells,'connectivity':4,'source':'DA3 metric depth outside car mask, including non-road pixels'},'render_only':args.obstacles_render_only,'road_cells':road_count,'road_false_positive_cells':road_false,'road_false_positive_ratio':false_ratio,'car_footprint_obstacle_cells':int(np.count_nonzero(car_grid & (component_ids>0))),'instance_id_file':'obstacles.npz'}
    write_json(out/'scene.json', scene)
    scene_time = time.perf_counter()-grid_start
    from artifacts import SceneArtifacts
    return Reconstruction(SceneArtifacts(out),ground_time,scene_time)
