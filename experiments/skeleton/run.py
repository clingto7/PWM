"""Issue #8: build the local first-image scene and one sliding impulse run."""
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
    inputs = {'image': ROOT / 'input.jpg', 'da3': ROOT / 'da3nested_output/prediction.npz', 'sam_car': ROOT / 'car_input/car_mask.png'}
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
    for j in range(n):
        y = j * args.cell
        for i in range(n):
            x = origin_x + i * args.cell
            inside = y >= args.cell and x*x + y*y <= args.radius**2 and abs(math.atan2(x, y)) <= half_angle
            valid.append(inside)
            # Project the ground grid back to the original image. Its car mask
            # marks an inferred footprint, never a depth source.
            camera_point = x*right + y*forward + flat_z*normal
            if camera_point[1] > 0:
                u = int(round(k[0,0]*camera_point[0]/camera_point[1]+k[0,2]))
                v = int(round(-k[1,1]*camera_point[2]/camera_point[1]+k[1,2]))
                if 0<=u<w and 0<=v<h and car[v,u]:
                    grid_z[j,i]=flat_z
                    provenance[j,i]=3
            vertices.append([x,y,float(grid_z[j,i])])
    for j in range(n-1):
        for i in range(n-1):
            q = j*n+i
            if valid[q] and valid[q+1] and valid[q+n] and valid[q+n+1]:
                indices.extend([q,q+1,q+n,q+1,q+n+1,q+n])
    if not indices:
        raise ValueError('empty scene sector')
    np.savez_compressed(out/'heightmap.npz', z_m=grid_z.astype(np.float32), valid=np.asarray(valid,dtype=np.uint8).reshape(n,n), provenance=provenance)
    # The visible car is excluded from both mesh and collision generation; the fitted
    # plane spans its footprint, including pixels hidden by the original car.
    ys_car, xs_car = np.nonzero(car)
    if not len(xs_car):
        raise ValueError('empty SAM car mask')
    x0, x1 = int(xs_car.min()), int(xs_car.max())
    y0, y1 = int(ys_car.min()), int(ys_car.max())
    car_depth = float(np.median(depth[car & np.isfinite(depth)]))
    car_cam_x = ((x0+x1)/2-k[0,2])/k[0,0]*car_depth
    cx,cy,_ = world(np.array([car_cam_x,car_depth,a*car_cam_x+b*car_depth+c]))
    width = max(.2, (x1-x0)/k[0,0]*car_depth)
    length = max(.2, (y1-y0)/k[1,1]*car_depth)
    box_h = min(width, length) * .5
    mass = args.density * width * length * box_h
    base_z = flat_z
    car_pos = [cx, cy, base_z+box_h/2]
    scene = {'camera': {'intrinsics': k.tolist(), 'position_m': [0,0,0], 'forward': camera_forward, 'up': camera_up, 'fixed': True}, 'ground': {'plane_z_ax_by_c': [a,b,c], 'flat_world_z_m': flat_z, 'height_m': height, 'pitch_deg': pitch, 'rms_m': residual, 'sample_count': int(road.sum()), 'cell_m': args.cell, 'grid_size': n, 'heightmap_method': {'near': 'metric DA3 road pixel back-projection, cell mean', 'far': 'camera ray / fitted plane intersection', 'car': 'SAM mask footprint replaced by fitted plane', 'near_depth_m': args.near_depth, 'ground_inlier_m': args.ground_inlier, 'provenance_codes': {'near_depth':1,'ray_plane':2,'car_plane_fill':3}, 'cell_counts': {'near_depth':int(np.count_nonzero(provenance==1)), 'ray_plane':int(np.count_nonzero(provenance==2)), 'car_plane_fill':int(np.count_nonzero(provenance==3))}}}, 'sector': {'radius_m': args.radius, 'angle_deg': args.angle, 'invisible_wall': True}, 'car': {'image_bbox_xyxy': [x0,y0,x1,y1], 'excluded_image_pixels': int(car.sum()), 'footprint_filled_with_plane': True, 'depth_m': car_depth, 'size_m': [width,length,box_h], 'volume_m3': width*length*box_h, 'density_kg_m3': args.density, 'mass_kg': mass, 'initial_position_m': car_pos}, 'mesh': {'vertices': len(vertices), 'triangles': len(indices)//3}}
    write_json(out/'scene.json', scene)
    scene_time = time.perf_counter()-grid_start
    sim_start = time.perf_counter()
    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0,0,-9.81)
        p.setTimeStep(1/240)
        p.setPhysicsEngineParameter(fixedTimeStep=1/240, numSolverIterations=100, deterministicOverlappingPairs=1)
        ground_collision = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=indices, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        ground_visual = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices, rgbaColor=[.5,.5,.5,1])
        ground = p.createMultiBody(0, ground_collision, ground_visual)
        p.changeDynamics(ground,-1,lateralFriction=args.friction,restitution=0)
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
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[width/2,length/2,box_h/2])
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[width/2,length/2,box_h/2],rgbaColor=[1,0,0,1])
        body = p.createMultiBody(mass,shape,visual,basePosition=car_pos)
        p.changeDynamics(body,-1,lateralFriction=args.friction,restitution=0,linearDamping=0,angularDamping=0)
        p.resetBaseVelocity(body,linearVelocity=[0,args.speed,0])
        def trajectory():
            rows=[]
            penetration=0.0
            for step in range(int(args.seconds*240)):
                p.stepSimulation()
                pos,quat=p.getBasePositionAndOrientation(body)
                vel,omega=p.getBaseVelocity(body)
                rows.append([*pos,*quat,*vel,*omega])
                contacts=p.getContactPoints(bodyA=body)
                penetration=max(penetration,max([0]+[-float(q[8]) for q in contacts]))
            return rows,penetration
        rows,penetration=trajectory()
        p.resetBasePositionAndOrientation(body,car_pos,[0,0,0,1]); p.resetBaseVelocity(body,linearVelocity=[0,args.speed,0],angularVelocity=[0,0,0])
        replay,_=trajectory()
        replay_identical=rows==replay
        write_json(out/'trajectory.json',rows)
        write_json(out/'replay_trajectory.json',replay)
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
                ids=np.zeros((h,w),dtype=np.uint8)
                ids[(raw & 0xffffff) == ground]=1
                ids[(raw & 0xffffff) == body]=2
                ids[raw<0]=0
                millimeters=np.clip(metric*1000,0,65535).astype(np.uint16)
                Image.fromarray(millimeters).save(depth_dir/f'{frame:04d}.png')
                Image.fromarray(ids).save(ids_dir/f'{frame:04d}.png')
                enc_depth.stdin.write(np.clip(metric/args.radius*255,0,255).astype(np.uint8).tobytes())
                enc_id.stdin.write((ids*110).tobytes())
        finally:
            enc_depth.stdin.close();enc_id.stdin.close()
            depth_status = enc_depth.wait()
            id_status = enc_id.wait()
            if depth_status != 0 or id_status != 0:
                raise RuntimeError('ffmpeg failed')
        render_time=time.perf_counter()-render_start
    finally:
        p.disconnect(client)
    outputs={str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*')) if path.is_file()}
    record={'schema':1,'inputs':{name:{'path':str(path.resolve()),'sha256':hashes[name]} for name,path in inputs.items()},'outputs':outputs,'parameters':{'radius_m':args.radius,'angle_deg':args.angle,'cell_m':args.cell,'friction':args.friction,'density_kg_m3':args.density,'derived_mass_kg':mass,'near_depth_m':args.near_depth,'ground_inlier_m':args.ground_inlier,'speed_m_s':args.speed,'seconds':args.seconds,'physics_hz':240,'video_fps':30},'timings_s':{'ground_fit':ground_time,'scene_build':scene_time,'simulation_and_replay':sim_time,'render_and_video':render_time,'total':time.perf_counter()-start},'versions':{'python':platform.python_version(),'numpy':version('numpy'),'pillow':version('pillow'),'pybullet':version('pybullet'),'pybullet_api':p.getAPIVersion(),'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]},'measurements':{'max_contact_penetration_m':penetration,'replay_trajectory_identical':replay_identical,'final_position_m':rows[-1][:3],'final_velocity_m_s':rows[-1][7:10]}}
    record['stages'] = {
        'ground_fit_scene': {'inputs': hashes, 'outputs': {name:outputs[name] for name in ('scene.json','heightmap.npz')}, 'parameters': {'radius_m':args.radius,'angle_deg':args.angle,'cell_m':args.cell,'near_depth_m':args.near_depth,'ground_inlier_m':args.ground_inlier}, 'seconds':ground_time+scene_time},
        'simulation_replay': {'inputs': {'scene.json':outputs['scene.json'],'heightmap.npz':outputs['heightmap.npz']}, 'outputs': {name:outputs[name] for name in ('trajectory.json','replay_trajectory.json')}, 'parameters': {'density_kg_m3':args.density,'derived_mass_kg':mass,'friction':args.friction,'speed_m_s':args.speed,'physics_hz':240}, 'seconds':sim_time},
        'proxy_render_video': {'inputs': {'scene.json':outputs['scene.json'],'heightmap.npz':outputs['heightmap.npz'],'trajectory.json':outputs['trajectory.json']}, 'outputs': {name:hashval for name,hashval in outputs.items() if name.startswith(('depth/','instance_id/')) or name in ('depth.mp4','instance_id.mp4')}, 'parameters': {'fps':30,'width':w,'height':h,'camera':'original fixed DA3'}, 'seconds':render_time}}
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
    parser.add_argument('--near-depth',type=float,default=18)
    parser.add_argument('--ground-inlier',type=float,default=.15)
    parser.add_argument('--seconds',type=float,default=3)
    args=parser.parse_args()
    values=(args.radius,args.angle,args.cell,args.near_depth,args.ground_inlier,args.friction,args.density,args.speed,args.seconds)
    if not (all(math.isfinite(value) for value in values) and 0 < args.radius <= 100 and 0 < args.angle < 180 and 0 < args.cell <= args.radius/2 and 0 < args.near_depth <= args.radius and 0 < args.ground_inlier <= 2 and 0 <= args.friction <= 2 and 0 < args.density <= 10000 and 0 < args.speed <= 100 and 0 < args.seconds <= 60 and int(args.seconds*240) >= 1 and int(args.seconds*30) >= 1):
        parser.error('invalid parameter range: require positive radius/cell/density/speed/duration/inlier; angle (0,180), near-depth <= radius, friction [0,2], and at least one physics and video frame')
    build(args)
