"""Callable first-image physics and raster stages over saved scene artifacts."""
import json
import math
import subprocess
import time

import numpy as np
import pybullet as p

from artifacts import SceneArtifacts
from physics import PyBulletSolver
from raster import RasterRenderer


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def solve_saved_scene(out, args, scene, component_ids, tops):
    geometry = SceneArtifacts(out).load()
    components = scene['obstacles']['components']
    car_pos = scene['car']['initial_position_m']
    mass = scene['car']['mass_kg']
    width, length, _ = scene['car']['size_m']
    origin_x = -(component_ids.shape[0]-1)/2*args.cell
    sim_start = time.perf_counter()
    client = p.connect(p.DIRECT)
    try:
        solver=PyBulletSolver(p)
        ground, body, obstacle_bodies = solver.create_world(geometry,args,out)
        inertia_diagonal=list(p.getDynamicsInfo(body,-1)[2])
        shape_counts={instance_id:len(p.getCollisionShapeData(obstacle_body,-1)) for obstacle_body,instance_id in obstacle_bodies.items()}
        for component in components:
            component['collision_shape_count']=shape_counts[component['instance_id']]
            if bool(component['collision_shape_count'])==args.obstacles_render_only:
                raise RuntimeError('obstacle collision shape disagrees with fallback mode')
        scene['contact_friction']={'ground_lateral_friction':args.friction,'car_lateral_friction':1.0,'effective_ground_car_friction':args.friction,'obstacle_lateral_friction':0.0}
        scene['physics_sleep']={'car_activation_state':'enable_sleeping','activation_state_value':p.ACTIVATION_STATE_ENABLE_SLEEPING}
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
            half_extent=abs(direction[0])*width/2+abs(direction[1])*length/2+args.cell/2
            travel_to_contact=max(0.,distance-half_extent)
            obstacle_speed=min(15.,max(args.obstacle_min_speed,math.sqrt(2*args.friction*9.81*travel_to_contact+1.0)))
        else:
            direction=np.array([0.,1.,0.]);obstacle_speed=args.speed;target_id=None
        impulses={'coast':[0.,args.speed,0.], 'ballistic':[0.,args.speed,args.ballistic_up_speed], 'obstacle':(direction*obstacle_speed).tolist()}
        specifications={name:{'delta_velocity_m_s':velocity,'magnitude_n_s':mass*math.sqrt(sum(v*v for v in velocity)), 'direction':(np.asarray(velocity)/math.sqrt(sum(v*v for v in velocity))).tolist(),'application_point':'center_of_mass','application_point_world_m':car_pos,'target_obstacle_id':target_id if name=='obstacle' else None} for name,velocity in impulses.items()}
        trajectories={};penetrations={};replays={};hits={}
        for name,velocity in impulses.items():
            if name!='coast':
                p.disconnect(client);client=p.connect(p.DIRECT)
                ground,body,obstacle_bodies=solver.create_world(geometry,args,out)
            case_rows,penetrations[name],hits[name]=solver.solve(body,obstacle_bodies,velocity,args.seconds)
            p.disconnect(client);client=p.connect(p.DIRECT)
            ground,body,obstacle_bodies=solver.create_world(geometry,args,out)
            replay,_,_=solver.solve(body,obstacle_bodies,velocity,args.seconds)
            trajectories[name]=case_rows;replays[name]=replay
            prefix='' if name=='coast' else name+'/'
            if name!='coast': (out/name).mkdir()
            write_json(out/(prefix+'trajectory.json'),case_rows)
            write_json(out/(prefix+'replay_trajectory.json'),replay)
        rows=trajectories['coast'];penetration=penetrations['coast'];replay_identical=rows==replays['coast']
        write_json(out/'impulses.json',specifications)
        sim_time=time.perf_counter()-sim_start
        write_json(out/'contacts.json', {'schema': 1, 'cases': {
            name: {'max_contact_penetration_m': penetrations[name],
                   'obstacle_contact_samples_by_id': hits[name]}
            for name in impulses}})
        return {'inertia_diagonal': inertia_diagonal, 'trajectories': trajectories,
                'penetrations': penetrations, 'replays': replays, 'hits': hits,
                'specifications': specifications, 'impulses': impulses,
                'sim_time': time.perf_counter()-sim_start}
    finally:
        p.disconnect(client)


def render_saved_scene(out, args, scene, car, target, rows):
    geometry = SceneArtifacts(out).load()
    k = np.asarray(scene['camera']['intrinsics'])
    camera_forward = scene['camera']['forward']
    camera_up = scene['camera']['up']
    h, w = 280, 504
    render_start = time.perf_counter()
    client = p.connect(p.DIRECT)
    try:
        solver = PyBulletSolver(p)
        ground, body, obstacle_bodies = solver.create_world(geometry, args, out)
        near,far=.1,max(100,args.radius*4)
        fx,fy,cx0,cy0=float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2])
        projection=[2*fx/w,0,0,0,0,2*fy/h,0,0,1-2*cx0/w,2*cy0/h-1,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]
        view=p.computeViewMatrix([0,0,0],camera_forward,camera_up)
        depth_dir=out/'depth'; ids_dir=out/'instance_id'; depth_dir.mkdir();ids_dir.mkdir()
        depth_video=out/'depth.mp4'; id_video=out/'instance_id.mp4'
        def encoder(path):
            return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','gray','-s',f'{w}x{h}','-r','30','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
        enc_depth,enc_id=encoder(depth_video),encoder(id_video)
        renderer=RasterRenderer(p,w,h,near,far,view,projection,ground,body,obstacle_bodies,args.radius,args.seconds)
        try:
            first_ids,raster_frame_times=renderer.render(rows,depth_dir,ids_dir,(enc_depth,enc_id))
        finally:
            enc_depth.stdin.close();enc_id.stdin.close()
            if enc_depth.wait() != 0 or enc_id.wait() != 0:
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
        return {'raster_frame_times': raster_frame_times,
                'projection_evidence': projection_evidence,
                'actual_bbox': actual_bbox,
                'render_time': time.perf_counter()-render_start}
    finally:
        p.disconnect(client)
