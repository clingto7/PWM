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
from bundle import sources, copy_inputs, revision
from raster import RasterRenderer
from raster_replay import render_saved
from physics import PyBulletSolver
from artifacts import SceneArtifacts
from scene import reconstruct



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
    source_paths = sources(args.bundle)
    bundled = copy_inputs(source_paths, out)
    inputs = {name: out / relative for name, relative in bundled.items()}
    hashes = {name: digest(path) for name, path in inputs.items()}
    write_json(out/'bundle.json', {'schema': 1, 'inputs': bundled, 'source_sha256': hashes, 'revision': revision(), 'arguments': {key: value for key, value in vars(args).items() if key not in ('output', 'bundle')}})
    reconstruction=reconstruct(inputs,args,out,start)
    k=reconstruction.k
    w=reconstruction.w
    h=reconstruction.h
    car=reconstruction.car
    right=reconstruction.right
    forward=reconstruction.forward
    normal=reconstruction.normal
    camera_forward=reconstruction.camera_forward
    camera_up=reconstruction.camera_up
    flat_z=reconstruction.flat_z
    half_angle=reconstruction.half_angle
    vertices=reconstruction.vertices
    indices=reconstruction.indices
    components=reconstruction.components
    obstacle_meshes=reconstruction.obstacle_meshes
    hull=reconstruction.hull
    car_pos=reconstruction.car_pos
    car_orientation=reconstruction.car_orientation
    mass=reconstruction.mass
    width=reconstruction.width
    length=reconstruction.length
    box_h=reconstruction.box_h
    scene=reconstruction.scene
    mesh_fit=reconstruction.mesh_fit
    target=reconstruction.target
    residual=reconstruction.residual
    ground_time=reconstruction.ground_time
    scene_time=reconstruction.scene_time
    component_ids=reconstruction.component_ids
    origin_x=reconstruction.origin_x
    tops=reconstruction.tops
    a=reconstruction.a
    b=reconstruction.b
    c=reconstruction.c
    height=reconstruction.height
    pitch=reconstruction.pitch
    road=reconstruction.road
    road_count=reconstruction.road_count
    road_false=reconstruction.road_false
    false_ratio=reconstruction.false_ratio
    fitted_mesh=reconstruction.fitted_mesh
    scene_artifacts=SceneArtifacts(out)
    scene_artifacts.files()
    sim_start = time.perf_counter()
    client = p.connect(p.DIRECT)
    try:
        solver=PyBulletSolver(p)
        ground, body, obstacle_bodies = solver.create_world(reconstruction,args,out)
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
                ground,body,obstacle_bodies=solver.create_world(reconstruction,args,out)
            case_rows,penetrations[name],hits[name]=solver.solve(body,obstacle_bodies,velocity,args.seconds)
            p.disconnect(client);client=p.connect(p.DIRECT)
            ground,body,obstacle_bodies=solver.create_world(reconstruction,args,out)
            replay,_,_=solver.solve(body,obstacle_bodies,velocity,args.seconds)
            trajectories[name]=case_rows;replays[name]=replay
            prefix='' if name=='coast' else name+'/'
            if name!='coast': (out/name).mkdir()
            write_json(out/(prefix+'trajectory.json'),case_rows)
            write_json(out/(prefix+'replay_trajectory.json'),replay)
        rows=trajectories['coast'];penetration=penetrations['coast'];replay_identical=rows==replays['coast']
        write_json(out/'impulses.json',specifications)
        sim_time=time.perf_counter()-sim_start
        p.disconnect(client);client=p.connect(p.DIRECT)
        ground,body,obstacle_bodies=solver.create_world(SceneArtifacts(out).load(),args,out)
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
    finally:
        p.disconnect(client)
    replay_frame_times=render_saved(out,out/'raster_replay',{'radius_m':args.radius,'friction':args.friction,'obstacles_render_only':args.obstacles_render_only,'seconds':args.seconds})
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
    record={'schema':2,'inputs':{name:{'path':bundled[name],'sha256':hashes[name]} for name,path in inputs.items()},'outputs':outputs,'parameters':{'radius_m':args.radius,'angle_deg':args.angle,'cell_m':args.cell,'friction':args.friction,'car_lateral_friction':1.0,'effective_ground_car_friction':args.friction,'density_kg_m3':args.density,'derived_mass_kg':mass,'near_depth_m':args.near_depth,'ground_inlier_m':args.ground_inlier,'speed_m_s':args.speed,'obstacle_min_speed_m_s':args.obstacle_min_speed,'car_sleep_activation_state':'enable_sleeping','seconds':args.seconds,'physics_hz':240,'obstacle_height_m':args.obstacle_height,'obstacle_min_pixels':args.obstacle_min_pixels,'obstacle_min_cells':args.obstacle_min_cells,'obstacles_render_only':args.obstacles_render_only,'video_fps':30},'timings_s':{'ground_fit':ground_time,'scene_build':scene_time,'simulation_and_replay':sim_time,'render_and_video':render_time,'total':time.perf_counter()-start},'versions':{'python':platform.python_version(),'numpy':version('numpy'),'pillow':version('pillow'),'pybullet':version('pybullet'),'pybullet_api':p.getAPIVersion(),'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]},'measurements':{'max_contact_penetration_m':penetration,'replay_trajectory_identical':replay_identical,'final_position_m':rows[-1][:3],'final_velocity_m_s':rows[-1][7:10],'initial_projection_bbox_xyxy':actual_bbox,'initial_projection_mask_iou':projection_evidence['mask_iou'],'mesh_fit':mesh_fit,'road_false_positive_ratio':false_ratio,'road_false_positive_cells':road_false,'road_cells':road_count,'obstacle_component_sizes_cells':[component['cells'] for component in components],'obstacle_component_count':len(components)}}
    record['stages'] = {
        'ground_fit_scene': {'inputs': hashes, 'outputs': {name:outputs[name] for name in ('scene.json','heightmap.npz','fitted_car.obj','car_collision_hull.obj','obstacles.npz')}, 'parameters': {'radius_m':args.radius,'angle_deg':args.angle,'cell_m':args.cell,'near_depth_m':args.near_depth,'ground_inlier_m':args.ground_inlier,'obstacle_height_m':args.obstacle_height,'obstacle_min_pixels':args.obstacle_min_pixels,'obstacle_min_cells':args.obstacle_min_cells,'obstacles_render_only':args.obstacles_render_only}, 'seconds':ground_time+scene_time},
        'simulation_replay': {'inputs': {'scene.json':outputs['scene.json'],'heightmap.npz':outputs['heightmap.npz'],'fitted_car.obj':outputs['fitted_car.obj'],'car_collision_hull.obj':outputs['car_collision_hull.obj'],'obstacles.npz':outputs['obstacles.npz']}, 'outputs': {name:outputs[name] for name in ('trajectory.json','replay_trajectory.json')}, 'parameters': {'density_kg_m3':args.density,'derived_mass_kg':mass,'friction':args.friction,'speed_m_s':args.speed,'obstacle_min_speed_m_s':args.obstacle_min_speed,'car_sleep_activation_state':'enable_sleeping','physics_hz':240,'obstacles_render_only':args.obstacles_render_only}, 'seconds':sim_time},
        'proxy_render_video': {'inputs': {'scene.json':outputs['scene.json'],'heightmap.npz':outputs['heightmap.npz'],'trajectory.json':outputs['trajectory.json'],'obstacles.npz':outputs['obstacles.npz'],'fitted_car.obj':outputs['fitted_car.obj']}, 'outputs': {name:hashval for name,hashval in outputs.items() if name.startswith(('depth/','instance_id/')) or name in ('depth.mp4','instance_id.mp4','initial_projection.json')}, 'parameters': {'fps':30,'width':w,'height':h,'camera':'original fixed DA3'}, 'seconds':render_time}}
    record['impulse_cases']={name:{'trajectory':('' if name=='coast' else name+'/')+'trajectory.json','replay':('' if name=='coast' else name+'/')+'replay_trajectory.json','impulse':specifications[name],'inertia_diagonal_kg_m2':inertia_diagonal,'max_contact_penetration_m':penetrations[name],'obstacle_contact_samples_by_id':hits[name]} for name in impulses}
    record['parameters']['ballistic_up_speed_m_s']=args.ballistic_up_speed
    record['measurements']['ballistic_in_camera']=ballistic_visible
    record['stages']['simulation_replay']['outputs'].update({name:outputs[name] for name in outputs if name.startswith(('ballistic/','obstacle/')) or name=='impulses.json'})
    record['scene_artifacts']=list(SceneArtifacts.NAMES)
    record['adapters']={'physics':'PyBulletSolver','main_renderer':'RasterRenderer','comparison_renderer':'gsplat' if args.render_gaussians else None}
    record['required_renderers']=['raster'] + (['gaussian'] if args.render_gaussians else [])
    record['raster_frame_times_s']=raster_frame_times
    record['raster_replay']={'trajectory':'trajectory.json','camera':'scene.json','depth_dir':'raster_replay/depth','instance_id_dir':'raster_replay/instance_id','frame_times_s':replay_frame_times}
    record['stages']['raster_replay']={'inputs':{name:outputs[name] for name in ('scene.json','heightmap.npz','obstacles.npz','fitted_car.obj','trajectory.json')},'outputs':{name:value for name,value in outputs.items() if name.startswith('raster_replay/')},'parameters':{'fps':30,'width':w,'height':h,'camera':'original fixed DA3'},'seconds':sum(replay_frame_times)}
    if args.render_gaussians:
        record['required_renderers']=['raster','gaussian']
    write_json(out/'run.json',record)
    if args.render_gaussians:
        subprocess.run([sys.executable,str(Path(__file__).with_name('gaussian.py')),str(out)],check=True)
    print(json.dumps({'run_dir':str(out),'frames':int(args.seconds*30),'measurements':record['measurements']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--bundle',type=Path,help='run bundle containing bundle.json and inputs/')
    parser.add_argument('--radius',type=float,default=27)
    parser.add_argument('--angle',type=float,default=95)
    parser.add_argument('--cell',type=float,default=.5)
    parser.add_argument('--friction',type=float,default=.6)
    parser.add_argument('--density',type=float,default=500)
    parser.add_argument('--speed',type=float,default=3)
    parser.add_argument('--obstacle-min-speed',type=float,default=6)
    parser.add_argument('--ballistic-up-speed',type=float,default=2)
    parser.add_argument('--near-depth',type=float,default=18)
    parser.add_argument('--ground-inlier',type=float,default=.15)
    parser.add_argument('--obstacle-height',type=float,default=.35)
    parser.add_argument('--obstacle-min-pixels',type=int,default=3)
    parser.add_argument('--obstacle-min-cells',type=int,default=2)
    parser.add_argument('--obstacles-render-only',action='store_true')
    parser.add_argument('--seconds',type=float,default=5)
    parser.add_argument('--render-gaussians',action='store_true',help='also render geometric gsplat frames (requires CUDA)')
    args=parser.parse_args()
    values=(args.radius,args.angle,args.cell,args.near_depth,args.ground_inlier,args.obstacle_height,args.friction,args.density,args.speed,args.obstacle_min_speed,args.ballistic_up_speed,args.seconds)
    if not (all(math.isfinite(value) for value in values) and 0 < args.radius <= 100 and 0 < args.angle < 180 and 0 < args.cell <= args.radius/2 and 0 < args.near_depth <= args.radius and 0 < args.ground_inlier <= 2 and 0 < args.obstacle_height <= 10 and 1 <= args.obstacle_min_pixels <= 1000 and 1 <= args.obstacle_min_cells <= 1000 and 0 <= args.friction <= 2 and 0 < args.density <= 10000 and 0 < args.speed <= 100 and 0 < args.obstacle_min_speed <= 15 and 0 < args.seconds <= 60 and int(args.seconds*240) >= 1 and int(args.seconds*30) >= 1):
        parser.error('invalid parameter range: require positive radius/cell/density/speed/duration/inlier; angle (0,180), near-depth <= radius, friction [0,2], and at least one physics and video frame')
    build(args)
