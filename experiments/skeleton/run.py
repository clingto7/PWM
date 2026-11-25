"""Build the first-image scene and three center-of-mass impulse runs."""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
import pybullet as p
from car_fit import fit

ROOT = Path(__file__).resolve().parents[1] / 'first_image'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def build(args):
    start = time.perf_counter()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'output directory is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    inputs = {'image': ROOT / 'input.jpg', 'da3': ROOT / 'da3nested_output/prediction.npz', 'sam_car': ROOT / 'car_input/car_mask.png', 'hunyuan_mesh': ROOT / 'hunyuan21_output/car_shape.glb'}
    hashes = {name: digest(path) for name, path in inputs.items()}
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
    # Lower central image region is the observable near road. Exclude every car pixel.
    road = (ys > int(h * .68)) & (xs > int(w * .2)) & (xs < int(w * .8)) & ~car & (depth > 2) & (depth < args.radius)
    if road.sum() < 100:
        raise ValueError('insufficient trusted ground pixels')
    design = np.stack([xx[road], yy[road], np.ones(road.sum())], axis=1)
    coeff, _, _, _ = np.linalg.lstsq(design, zz[road], rcond=None)
    a, b, c = map(float, coeff)
    residual = float(np.sqrt(np.mean((design @ coeff - zz[road]) ** 2)))
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
    # Default to an explicit ray/plane intersection. In ground-aligned world
    # coordinates the intersection height is constant; near observed road pixels
    # replace those cells with their metric DA3 back-projections.
    origin_x = -(n-1)/2*args.cell
    grid_z = np.full((n,n),flat_z,dtype=np.float64)
    provenance = np.full((n,n),2,dtype=np.uint8)  # 1 depth, 2 ray/plane, 3 car fill
    depth_road = road & (depth <= args.near_depth) & (np.abs(zz-(a*xx+b*yy+c)) <= args.ground_inlier)
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
    # The road-only heightmap cannot see roadside surfaces. Back-project all
    # non-car DA3 pixels in the trusted range and vote in world-grid cells.
    surface = (~car) & np.isfinite(depth) & (depth > 2) & (depth <= args.radius)
    points = np.stack([xx[surface], yy[surface], zz[surface]], axis=1)
    ox, oy, oz = points @ right, points @ forward, points @ normal
    oi = np.rint((ox-origin_x)/args.cell).astype(int)
    oj = np.rint(oy/args.cell).astype(int)
    in_bounds = (oi>=0)&(oi<n)&(oj>=0)&(oj<n)
    oi,oj,oz = oi[in_bounds],oj[in_bounds],oz[in_bounds]
    selected = valid_grid[oj,oi] & ~car_grid[oj,oi]
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
    candidates = valid_grid & ~car_grid & (tops-flat_z > args.obstacle_height) & (votes >= args.obstacle_min_pixels)
    component_ids = np.zeros((n,n),dtype=np.uint16)
    visited = np.zeros((n,n),dtype=bool)
    components = []
    for j in range(n):
        for i in range(n):
            if not candidates[j,i] or visited[j,i]:
                continue
            cells=[]; stack=[(j,i)]; visited[j,i]=True
            while stack:
                cy,cx=stack.pop(); cells.append((cy,cx))
                for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                    if 0<=ny<n and 0<=nx<n and candidates[ny,nx] and not visited[ny,nx]:
                        visited[ny,nx]=True;stack.append((ny,nx))
            if len(cells) < args.obstacle_min_cells:
                continue
            instance_id=len(components)+3
            if instance_id>=65535:
                raise ValueError('too many obstacle components for uint16 instance IDs')
            for cy,cx in cells: component_ids[cy,cx]=instance_id
            components.append({'instance_id':instance_id,'cells':len(cells),'area_m2':len(cells)*args.cell**2,'max_height_above_plane_m':float(max(tops[cy,cx]-flat_z for cy,cx in cells)),'collision_enabled':not args.obstacles_render_only})
    road_points=np.stack([xx[road],yy[road],zz[road]],axis=1)
    ri=np.rint(((road_points @ right)-origin_x)/args.cell).astype(int)
    rj=np.rint((road_points @ forward)/args.cell).astype(int)
    road_cells=np.zeros((n,n),dtype=bool)
    rb=(ri>=0)&(ri<n)&(rj>=0)&(rj<n)
    road_cells[rj[rb],ri[rb]]=True
    road_cells &= valid_grid & ~car_grid
    road_count=int(road_cells.sum())
    road_false=int(np.count_nonzero(road_cells & (component_ids>0)))
    false_ratio=road_false/road_count if road_count else None
    np.savez_compressed(out/'obstacles.npz',instance_id=component_ids,candidate=candidates.astype(np.uint8),road_cells=road_cells.astype(np.uint8),car_footprint=car_grid.astype(np.uint8))
    for j in range(n-1):
        for i in range(n-1):
            q = j*n+i
            if valid[q] and valid[q+1] and valid[q+n] and valid[q+n+1]:
                indices.extend([q,q+1,q+n,q+1,q+n+1,q+n])
    if not indices:
        raise ValueError('empty scene sector')
    np.savez_compressed(out/'heightmap.npz', z_m=grid_z.astype(np.float32), ground_z_m=grid_z.astype(np.float32), surface_z_m=tops.astype(np.float32), valid=valid_grid.astype(np.uint8), provenance=provenance, surface_provenance=surface_provenance, surface_votes=votes)
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
    fitted_mesh, mesh_fit = fit(inputs['hunyuan_mesh'], car, depth, k, right, forward, normal, flat_z, (anchor_x,anchor_y), car_depth)
    fitted_mesh.export(out/'fitted_car.obj')
    hull = fitted_mesh.convex_hull
    hull.export(out/'car_collision_hull.obj')
    width,length,box_h = mesh_fit['size_m']
    car_pos = mesh_fit['position_m']
    yaw = mesh_fit['yaw_rad']
    car_orientation = p.getQuaternionFromEuler([0,0,yaw])
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
    scene = {'camera': {'intrinsics': k.tolist(), 'position_m': [0,0,0], 'forward': camera_forward, 'up': camera_up, 'fixed': True}, 'ground': {'plane_z_ax_by_c': [a,b,c], 'flat_world_z_m': flat_z, 'height_m': height, 'pitch_deg': pitch, 'rms_m': residual, 'sample_count': int(road.sum()), 'cell_m': args.cell, 'grid_size': n, 'heightmap_method': {'near': 'metric DA3 road pixel back-projection, cell mean', 'far': 'camera ray / fitted plane intersection', 'car': 'SAM mask footprint replaced by fitted plane', 'near_depth_m': args.near_depth, 'ground_inlier_m': args.ground_inlier, 'provenance_codes': {'near_depth':1,'ray_plane':2,'car_plane_fill':3}, 'cell_counts': {'near_depth':int(np.count_nonzero(provenance==1)), 'ray_plane':int(np.count_nonzero(provenance==2)), 'car_plane_fill':int(np.count_nonzero(provenance==3))}}}, 'sector': {'radius_m': args.radius, 'angle_deg': args.angle, 'invisible_wall': True}, 'car': {'image_bbox_xyxy': [x0,y0,x1,y1], 'excluded_image_pixels': int(car.sum()), 'footprint_filled_with_plane': True, 'ground_contact_ray': {'pixel_uv': [u_contact,y1], 'plane_intersection_depth_m': car_depth, 'world_xy_m': [anchor_x,anchor_y]}, 'mesh_fit': mesh_fit, 'initial_orientation_xyzw': car_orientation, 'collision_shape': 'convex hull of fitted Hunyuan mesh', 'box_fit': {'method': 'ground-edge anchor and projected bounding-box search', 'reference_bbox_xyxy': target, 'predicted_bbox_xyxy': best[5], 'predicted_bbox_iou': best[0], 'width_from_bbox_at_ground_depth_m': projected_width, 'height_from_bbox_at_ground_depth_m': projected_height}, 'depth_m': car_depth, 'size_m': [width,length,box_h], 'volume_m3': mesh_volume, 'density_kg_m3': args.density, 'mass_kg': mass, 'initial_position_m': car_pos}, 'mesh': {'vertices': len(vertices), 'triangles': len(indices)//3}}
    scene['obstacles']={'components':components,'component_count':len(components),'detection':{'height_above_plane_m':args.obstacle_height,'min_pixels_per_cell':args.obstacle_min_pixels,'min_cells_per_component':args.obstacle_min_cells,'connectivity':4,'source':'DA3 metric depth outside car mask, including non-road pixels'},'render_only':args.obstacles_render_only,'road_cells':road_count,'road_false_positive_cells':road_false,'road_false_positive_ratio':false_ratio,'car_footprint_obstacle_cells':int(np.count_nonzero(car_grid & (component_ids>0))),'instance_id_file':'obstacles.npz'}
    write_json(out/'scene.json', scene)
    scene_time = time.perf_counter()-grid_start
    sim_start = time.perf_counter()
    client = p.connect(p.DIRECT)
    try:
        def create_world():
            p.setGravity(0,0,-9.81)
            p.setTimeStep(1/240)
            p.setPhysicsEngineParameter(fixedTimeStep=1/240, numSolverIterations=100, deterministicOverlappingPairs=1)
            ground_collision = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=indices, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
            ground_visual = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices, rgbaColor=[.5,.5,.5,1])
            ground = p.createMultiBody(0, ground_collision, ground_visual)
            p.changeDynamics(ground,-1,lateralFriction=args.friction,restitution=0)
            obstacle_bodies={}
            for component,(mesh_vertices,mesh_indices) in zip(components,obstacle_meshes):
                collision=-1 if args.obstacles_render_only else p.createCollisionShape(p.GEOM_MESH,vertices=mesh_vertices,indices=mesh_indices,flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
                visual=p.createVisualShape(p.GEOM_MESH,vertices=mesh_vertices,indices=mesh_indices,rgbaColor=[.1,.5,.1,1])
                obstacle_body=p.createMultiBody(0,collision,visual)
                if collision != -1:
                    p.changeDynamics(obstacle_body,-1,lateralFriction=0,restitution=0)
                obstacle_bodies[obstacle_body]=component['instance_id']
            walls = []
            # Radial arc and two sector edges are collision-only static segments.
            segments = []
            for t in np.linspace(-half_angle, half_angle, 65)[:-1]:
                t2 = t + 2*half_angle/64
                segments.append(((args.radius*math.sin(t),args.radius*math.cos(t)),(args.radius*math.sin(t2),args.radius*math.cos(t2))))
            segments += [((0,0),(args.radius*math.sin(t),args.radius*math.cos(t))) for t in (-half_angle,half_angle)]
            for (u,v) in segments:
                dx,dy = v[0]-u[0],v[1]-u[1]
                mid = [(u[0]+v[0])/2,(u[1]+v[1])/2,flat_z+.5]
                shape = p.createCollisionShape(p.GEOM_BOX,halfExtents=[math.hypot(dx,dy)/2,.04,.5])
                walls.append(p.createMultiBody(0,shape,basePosition=mid,baseOrientation=p.getQuaternionFromEuler([0,0,math.atan2(dy,dx)])))
            shape = p.createCollisionShape(p.GEOM_MESH, vertices=hull.vertices.tolist())
            visual = p.createVisualShape(p.GEOM_MESH, fileName=str(out/'fitted_car.obj'), rgbaColor=[1,0,0,1])
            body = p.createMultiBody(mass,shape,visual,basePosition=car_pos,baseOrientation=car_orientation)
            p.changeDynamics(body,-1,lateralFriction=args.friction,restitution=0,linearDamping=0,angularDamping=0)
            return ground, body, obstacle_bodies
        ground, body, obstacle_bodies = create_world()
        inertia_diagonal=list(p.getDynamicsInfo(body,-1)[2])
        shape_counts={instance_id:len(p.getCollisionShapeData(obstacle_body,-1)) for obstacle_body,instance_id in obstacle_bodies.items()}
        for component in components:
            component['collision_shape_count']=shape_counts[component['instance_id']]
            if bool(component['collision_shape_count'])==args.obstacles_render_only:
                raise RuntimeError('obstacle collision shape disagrees with fallback mode')
        write_json(out/'scene.json',scene)
        targets=[]
        for cy,cx in np.argwhere(component_ids>0):
            x,y=origin_x+cx*args.cell,cy*args.cell
            distance=math.hypot(x-car_pos[0],y-car_pos[1])
            if distance>max(width,length) and tops[cy,cx]>car_pos[2]:
                targets.append((distance,x,y,int(component_ids[cy,cx])))
        obstacle_target=min(targets) if targets else None
        if obstacle_target:
            distance,x,y,target_id=obstacle_target
            direction=np.array([x-car_pos[0],y-car_pos[1],0.])/distance
            obstacle_speed=min(15.,max(args.speed,math.sqrt(2*args.friction*9.81*distance)*1.4))
        else:
            direction=np.array([0.,1.,0.]);obstacle_speed=args.speed;target_id=None
        impulses={'coast':[0.,args.speed,0.], 'ballistic':[0.,args.speed,args.ballistic_up_speed], 'obstacle':(direction*obstacle_speed).tolist()}
        specifications={name:{'delta_velocity_m_s':velocity,'magnitude_n_s':mass*math.sqrt(sum(v*v for v in velocity)), 'direction':(np.asarray(velocity)/math.sqrt(sum(v*v for v in velocity))).tolist(),'application_point':'center_of_mass','application_point_world_m':car_pos,'target_obstacle_id':target_id if name=='obstacle' else None} for name,velocity in impulses.items()}
        def trajectory(velocity):
            pos,quat=p.getBasePositionAndOrientation(body)
            vel,omega=p.getBaseVelocity(body)
            rows=[[ *pos,*quat,*vel,*omega ]]
            penetration=0.0
            obstacle_contacts={}
            p.resetBaseVelocity(body,linearVelocity=velocity)
            for step in range(int(args.seconds*240)-1):
                p.stepSimulation()
                pos,quat=p.getBasePositionAndOrientation(body)
                vel,omega=p.getBaseVelocity(body)
                rows.append([*pos,*quat,*vel,*omega])
                contacts=p.getContactPoints(bodyA=body)
                penetration=max(penetration,max([0]+[-float(q[8]) for q in contacts]))
                for q in contacts:
                    if q[2] in obstacle_bodies:
                        instance_id=obstacle_bodies[q[2]]
                        obstacle_contacts[instance_id]=obstacle_contacts.get(instance_id,0)+1
            return rows,penetration,obstacle_contacts
        trajectories={};penetrations={};replays={};hits={}
        for name,velocity in impulses.items():
            if name!='coast':
                p.disconnect(client);client=p.connect(p.DIRECT)
                ground,body,obstacle_bodies=create_world()
            case_rows,penetrations[name],hits[name]=trajectory(velocity)
            p.disconnect(client);client=p.connect(p.DIRECT)
            ground,body,obstacle_bodies=create_world()
            replay,_,_=trajectory(velocity)
            trajectories[name]=case_rows;replays[name]=replay
            prefix='' if name=='coast' else name+'/'
            if name!='coast': (out/name).mkdir()
            write_json(out/(prefix+'trajectory.json'),case_rows)
            write_json(out/(prefix+'replay_trajectory.json'),replay)
        rows=trajectories['coast'];penetration=penetrations['coast'];replay_identical=rows==replays['coast']
        write_json(out/'impulses.json',specifications)
        sim_time=time.perf_counter()-sim_start
        render_start=time.perf_counter()
        near,far=.1,max(100,args.radius*4)
        fx,fy,cx0,cy0=float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2])
        projection=[2*fx/w,0,0,0,0,2*fy/h,0,0,1-2*cx0/w,2*cy0/h-1,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]
        view=p.computeViewMatrix([0,0,0],camera_forward,camera_up)
        depth_dir=out/'depth'; ids_dir=out/'instance_id'; depth_dir.mkdir();ids_dir.mkdir()
        depth_video=out/'depth.mp4'; id_video=out/'instance_id.mp4'
        def encoder(path):
            return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','gray','-s',f'{w}x{h}','-r','30','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
        enc_depth,enc_id=encoder(depth_video),encoder(id_video)
        try:
            for frame in range(int(args.seconds*30)):
                row=rows[min(len(rows)-1,frame*8)]
                p.resetBasePositionAndOrientation(body,row[:3],row[3:7])
                _,_,_,zbuf,seg=p.getCameraImage(w,h,viewMatrix=view,projectionMatrix=projection,renderer=p.ER_TINY_RENDERER,flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
                zbuf=np.asarray(zbuf).reshape(h,w)
                metric=far*near/(far-(far-near)*zbuf)
                raw=np.asarray(seg,dtype=np.int32).reshape(h,w)
                ids=np.zeros((h,w),dtype=np.uint16)
                ids[(raw & 0xffffff) == ground]=1
                ids[(raw & 0xffffff) == body]=2
                for obstacle_body,instance_id in obstacle_bodies.items():
                    ids[(raw & 0xffffff) == obstacle_body]=instance_id
                ids[raw<0]=0
                # Zero is invalid/no entity; foreground retains metric mm.
                millimeters=np.where(ids>0,np.clip(metric*1000,1,65535),0).astype(np.uint16)
                Image.fromarray(millimeters).save(depth_dir/f'{frame:04d}.png')
                Image.fromarray(ids).save(ids_dir/f'{frame:04d}.png')
                if frame == 0:
                    first_ids=ids.copy()
                enc_depth.stdin.write(np.where(ids>0,np.clip(metric/args.radius*255,0,255),0).astype(np.uint8).tobytes())
                enc_id.stdin.write(np.clip(ids*53,0,255).astype(np.uint8).tobytes())
        finally:
            enc_depth.stdin.close();enc_id.stdin.close()
            depth_status = enc_depth.wait()
            id_status = enc_id.wait()
            if depth_status != 0 or id_status != 0:
                raise RuntimeError('ffmpeg failed')
        # Store pixel runs from the actual first ID render and resized SAM mask.
        def runs(mask):
            result=[]
            for row in range(h):
                active=np.flatnonzero(mask[row])
                if active.size:
                    cuts=np.where(np.diff(active)>1)[0]
                    starts=np.r_[0,cuts+1]; ends=np.r_[cuts,active.size-1]
                    result.extend([row,int(active[i]),int(active[j])+1] for i,j in zip(starts,ends))
            return result
        reference_runs=runs(car)
        rendered_runs=runs(first_ids==2)
        overlap=int(np.count_nonzero(car & (first_ids==2)))
        union=int(np.count_nonzero(car | (first_ids==2)))
        py,px=np.nonzero(first_ids==2)
        actual_bbox=[int(px.min()),int(py.min()),int(px.max())+1,int(py.max())+1] if len(px) else None
        projection_evidence={'reference_runs_y_x0_x1':reference_runs,'rendered_runs_y_x0_x1':rendered_runs,'reference_bbox_xyxy':target,'rendered_bbox_xyxy':actual_bbox,'intersection_pixels':overlap,'union_pixels':union,'mask_iou':overlap/union if union else 0,'frame':'instance_id/0000.png'}
        write_json(out/'initial_projection.json',projection_evidence)
        render_time=time.perf_counter()-render_start
    finally:
        p.disconnect(client)
    def in_camera(row):
        rotation=np.asarray(p.getMatrixFromQuaternion(row[3:7])).reshape(3,3)
        for dx in (-width/2,width/2):
            for dy in (-length/2,length/2):
                for dz in (-box_h/2,box_h/2):
                    point=np.asarray(row[:3])+rotation@np.asarray([dx,dy,dz])
                    px,py,pz=point[0]*right+point[1]*forward+point[2]*normal
                    if py<=0 or not (0<=k[0,0]*px/py+k[0,2]<w and 0<=-k[1,1]*pz/py+k[1,2]<h):
                        return False
        return True
    ballistic_visible=all(in_camera(row) for row in trajectories['ballistic'])
    outputs={str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*')) if path.is_file()}
    record={'schema':2,'inputs':{name:{'path':str(path.resolve()),'sha256':hashes[name]} for name,path in inputs.items()},'outputs':outputs,'parameters':{'radius_m':args.radius,'angle_deg':args.angle,'cell_m':args.cell,'friction':args.friction,'density_kg_m3':args.density,'derived_mass_kg':mass,'near_depth_m':args.near_depth,'ground_inlier_m':args.ground_inlier,'speed_m_s':args.speed,'seconds':args.seconds,'physics_hz':240,'obstacle_height_m':args.obstacle_height,'obstacle_min_pixels':args.obstacle_min_pixels,'obstacle_min_cells':args.obstacle_min_cells,'obstacles_render_only':args.obstacles_render_only,'video_fps':30},'timings_s':{'ground_fit':ground_time,'scene_build':scene_time,'simulation_and_replay':sim_time,'render_and_video':render_time,'total':time.perf_counter()-start},'versions':{'python':platform.python_version(),'numpy':version('numpy'),'pillow':version('pillow'),'pybullet':version('pybullet'),'pybullet_api':p.getAPIVersion(),'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]},'measurements':{'max_contact_penetration_m':penetration,'replay_trajectory_identical':replay_identical,'final_position_m':rows[-1][:3],'final_velocity_m_s':rows[-1][7:10],'initial_projection_bbox_xyxy':actual_bbox,'initial_projection_mask_iou':projection_evidence['mask_iou'],'mesh_fit':mesh_fit,'road_false_positive_ratio':false_ratio,'road_false_positive_cells':road_false,'road_cells':road_count,'obstacle_component_sizes_cells':[component['cells'] for component in components],'obstacle_component_count':len(components)}}
    record['stages'] = {
        'ground_fit_scene': {'inputs': hashes, 'outputs': {name:outputs[name] for name in ('scene.json','heightmap.npz','fitted_car.obj','car_collision_hull.obj','obstacles.npz')}, 'parameters': {'radius_m':args.radius,'angle_deg':args.angle,'cell_m':args.cell,'near_depth_m':args.near_depth,'ground_inlier_m':args.ground_inlier,'obstacle_height_m':args.obstacle_height,'obstacle_min_pixels':args.obstacle_min_pixels,'obstacle_min_cells':args.obstacle_min_cells,'obstacles_render_only':args.obstacles_render_only}, 'seconds':ground_time+scene_time},
        'simulation_replay': {'inputs': {'scene.json':outputs['scene.json'],'heightmap.npz':outputs['heightmap.npz'],'fitted_car.obj':outputs['fitted_car.obj'],'car_collision_hull.obj':outputs['car_collision_hull.obj'],'obstacles.npz':outputs['obstacles.npz']}, 'outputs': {name:outputs[name] for name in ('trajectory.json','replay_trajectory.json')}, 'parameters': {'density_kg_m3':args.density,'derived_mass_kg':mass,'friction':args.friction,'speed_m_s':args.speed,'physics_hz':240,'obstacles_render_only':args.obstacles_render_only}, 'seconds':sim_time},
        'proxy_render_video': {'inputs': {'scene.json':outputs['scene.json'],'heightmap.npz':outputs['heightmap.npz'],'trajectory.json':outputs['trajectory.json'],'obstacles.npz':outputs['obstacles.npz'],'fitted_car.obj':outputs['fitted_car.obj']}, 'outputs': {name:hashval for name,hashval in outputs.items() if name.startswith(('depth/','instance_id/')) or name in ('depth.mp4','instance_id.mp4','initial_projection.json')}, 'parameters': {'fps':30,'width':w,'height':h,'camera':'original fixed DA3'}, 'seconds':render_time}}
    record['impulse_cases']={name:{'trajectory':('' if name=='coast' else name+'/')+'trajectory.json','replay':('' if name=='coast' else name+'/')+'replay_trajectory.json','impulse':specifications[name],'inertia_diagonal_kg_m2':inertia_diagonal,'max_contact_penetration_m':penetrations[name],'obstacle_contact_samples_by_id':hits[name]} for name in impulses}
    record['parameters']['ballistic_up_speed_m_s']=args.ballistic_up_speed
    record['measurements']['ballistic_in_camera']=ballistic_visible
    record['stages']['simulation_replay']['outputs'].update({name:outputs[name] for name in outputs if name.startswith(('ballistic/','obstacle/')) or name=='impulses.json'})
    write_json(out/'run.json',record)
    print(json.dumps({'run_dir':str(out),'frames':int(args.seconds*30),'measurements':record['measurements']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--radius',type=float,default=27)
    parser.add_argument('--angle',type=float,default=95)
    parser.add_argument('--cell',type=float,default=.5)
    parser.add_argument('--friction',type=float,default=.6)
    parser.add_argument('--density',type=float,default=500)
    parser.add_argument('--speed',type=float,default=3)
    parser.add_argument('--ballistic-up-speed',type=float,default=2)
    parser.add_argument('--near-depth',type=float,default=18)
    parser.add_argument('--ground-inlier',type=float,default=.15)
    parser.add_argument('--obstacle-height',type=float,default=.35)
    parser.add_argument('--obstacle-min-pixels',type=int,default=3)
    parser.add_argument('--obstacle-min-cells',type=int,default=2)
    parser.add_argument('--obstacles-render-only',action='store_true')
    parser.add_argument('--seconds',type=float,default=5)
    args=parser.parse_args()
    values=(args.radius,args.angle,args.cell,args.near_depth,args.ground_inlier,args.obstacle_height,args.friction,args.density,args.speed,args.ballistic_up_speed,args.seconds)
    if not (all(math.isfinite(value) for value in values) and 0 < args.radius <= 100 and 0 < args.angle < 180 and 0 < args.cell <= args.radius/2 and 0 < args.near_depth <= args.radius and 0 < args.ground_inlier <= 2 and 0 < args.obstacle_height <= 10 and 1 <= args.obstacle_min_pixels <= 1000 and 1 <= args.obstacle_min_cells <= 1000 and 0 <= args.friction <= 2 and 0 < args.density <= 10000 and 0 < args.speed <= 100 and 0 < args.seconds <= 60 and int(args.seconds*240) >= 1 and int(args.seconds*30) >= 1):
        parser.error('invalid parameter range: require positive radius/cell/density/speed/duration/inlier; angle (0,180), near-depth <= radius, friction [0,2], and at least one physics and video frame')
    build(args)
