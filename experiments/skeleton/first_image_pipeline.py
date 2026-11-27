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
from car_fit import fit
from bundle import sources, copy_inputs, revision
from raster import RasterRenderer
from raster_replay import render_saved
from physics import api_version
from artifacts import SceneArtifacts
from scene import reconstruct
from first_image_stages import render_saved_scene
from physics import solve_saved_scene
from stage_contracts import record as record_stage, validate_resume, check as check_stage



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
    write_json(out/'stage_status.json', {'stage':'observation','status':'running'})
    source_paths = sources(args.bundle)
    bundled = copy_inputs(source_paths, out)
    inputs = {name: out / relative for name, relative in bundled.items()}
    hashes = {name: digest(path) for name, path in inputs.items()}
    observation_time = time.perf_counter() - start
    write_json(out/'bundle.json', {'schema': 1, 'inputs': bundled, 'source_sha256': hashes, 'revision': revision(), 'arguments': {key: value for key, value in vars(args).items() if key not in ('output', 'bundle', 'resume_scene')}})
    write_json(out/'stage_status.json', {'stage':'scene','status':'running'})
    if args.resume_scene:
        import shutil
        from types import SimpleNamespace
        saved = args.resume_scene.resolve()
        saved_scene = validate_resume(saved, {
            'radius_m': args.radius, 'angle_deg': args.angle, 'cell_m': args.cell,
            'near_depth_m': args.near_depth, 'ground_inlier_m': args.ground_inlier,
            'obstacle_height_m': args.obstacle_height,
            'obstacle_min_pixels': args.obstacle_min_pixels,
            'obstacle_min_cells': args.obstacle_min_cells,
            'obstacles_render_only': args.obstacles_render_only})
        if saved_scene['inputs'] != {f'inputs/{name}': digest(inputs[key]) for key, name in {'image':'input.jpg', 'da3':'prediction.npz', 'sam_car':'car_mask.png', 'hunyuan_mesh':'car_shape.glb'}.items()}:
            raise ValueError('saved scene inputs differ from current bundle')
        for name in SceneArtifacts.NAMES:
            shutil.copyfile(saved/name, out/name)
        for name in ('classification.json', 'fit_diagnostics.json', 'fit_visuals/before.png', 'fit_visuals/after.png'):
            if (saved/name).is_file():
                (out/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(saved/name, out/name)
        reconstruction = SimpleNamespace(artifacts=SceneArtifacts(out), ground_time=0.0, scene_time=0.0)
    else:
        reconstruction=reconstruct(inputs,args,out,start)
    scene_artifacts=reconstruction.artifacts
    scene_artifacts.files()
    scene=json.loads((out/'scene.json').read_text())
    k=np.asarray(scene['camera']['intrinsics'])
    h,w=280,504
    car=np.asarray(Image.open(inputs['sam_car']).convert('L').resize((w,h),Image.Resampling.NEAREST))>127
    a,b,c=scene['ground']['plane_z_ax_by_c']
    normal=np.array([-a,-b,1.0]);normal/=np.linalg.norm(normal)
    forward=np.array([0.,1.,0.]);forward-=np.dot(forward,normal)*normal;forward/=np.linalg.norm(forward)
    right=np.cross(forward,normal)
    camera_forward=scene['camera']['forward'];camera_up=scene['camera']['up']
    components=scene['obstacles']['components']
    car_pos=scene['car']['initial_position_m'];mass=scene['car']['mass_kg']
    width,length,box_h=scene['car']['size_m']
    mesh_fit=scene['car']['mesh_fit'];target=scene['car']['box_fit']['reference_bbox_xyxy']
    ground_time=reconstruction.ground_time;scene_time=reconstruction.scene_time
    with np.load(out/'obstacles.npz') as data: component_ids=data['instance_id']
    with np.load(out/'heightmap.npz') as data:
        tops=data['surface_z_exact_m'] if 'surface_z_exact_m' in data else data['surface_z_m']
    road_count=scene['obstacles']['road_cells']
    road_false=scene['obstacles']['road_false_positive_cells']
    false_ratio=scene['obstacles']['road_false_positive_ratio']
    write_json(out/'stage_status.json', {'stage':'physics','status':'running'})
    physics_result = solve_saved_scene(out, args, scene, component_ids, tops)
    inertia_diagonal = physics_result['inertia_diagonal']
    trajectories = physics_result['trajectories']
    penetrations = physics_result['penetrations']
    hits = physics_result['hits']
    specifications = physics_result['specifications']
    impulses = physics_result['impulses']
    sim_time = physics_result['sim_time']
    rows = trajectories['coast']
    penetration = penetrations['coast']
    replay_identical = rows == physics_result['replays']['coast']
    write_json(out/'stage_status.json', {'stage':'raster','status':'running'})
    render_result = render_saved_scene(out, args, scene, car, target, rows)
    raster_frame_times = render_result['raster_frame_times']
    projection_evidence = render_result['projection_evidence']
    actual_bbox = render_result['actual_bbox']
    render_time = render_result['render_time']
    replay_frame_times=render_saved(out,out/'raster_replay',{'radius_m':args.radius,'friction':args.friction,'obstacles_render_only':args.obstacles_render_only,'seconds':args.seconds})
    def in_camera(row):
        qx,qy,qz,qw=row[3:7]
        rotation=np.asarray([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                             [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                             [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
        for dx in (-width/2,width/2):
            for dy in (-length/2,length/2):
                for dz in (-box_h/2,box_h/2):
                    point=np.asarray(row[:3])+rotation@np.asarray([dx,dy,dz])
                    px,py,pz=point[0]*right+point[1]*forward+point[2]*normal
                    if py<=0 or not (0<=k[0,0]*px/py+k[0,2]<w and 0<=-k[1,1]*pz/py+k[1,2]<h):
                        return False
        return True
    ballistic_visible=all(in_camera(row) for row in trajectories['ballistic'])
    outputs={str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*')) if path.is_file() and path.name != 'stage_status.json'}
    record={'schema':2,'inputs':{name:{'path':bundled[name],'sha256':hashes[name]} for name,path in inputs.items()},'outputs':outputs,'parameters':{'radius_m':args.radius,'angle_deg':args.angle,'cell_m':args.cell,'friction':args.friction,'car_lateral_friction':1.0,'effective_ground_car_friction':args.friction,'density_kg_m3':args.density,'derived_mass_kg':mass,'near_depth_m':args.near_depth,'ground_inlier_m':args.ground_inlier,'speed_m_s':args.speed,'obstacle_min_speed_m_s':args.obstacle_min_speed,'car_sleep_activation_state':'enable_sleeping','seconds':args.seconds,'physics_hz':240,'obstacle_height_m':args.obstacle_height,'obstacle_min_pixels':args.obstacle_min_pixels,'obstacle_min_cells':args.obstacle_min_cells,'obstacles_render_only':args.obstacles_render_only,'video_fps':30},'timings_s':{'ground_fit':ground_time,'scene_build':scene_time,'simulation_and_replay':sim_time,'render_and_video':render_time,'total':time.perf_counter()-start},'versions':{'python':platform.python_version(),'numpy':version('numpy'),'pillow':version('pillow'),'pybullet':version('pybullet'),'pybullet_api':api_version(),'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]},'measurements':{'max_contact_penetration_m':penetration,'replay_trajectory_identical':replay_identical,'final_position_m':rows[-1][:3],'final_velocity_m_s':rows[-1][7:10],'initial_projection_bbox_xyxy':actual_bbox,'initial_projection_mask_iou':projection_evidence['mask_iou'],'mesh_fit':mesh_fit,'road_false_positive_ratio':false_ratio,'road_false_positive_cells':road_false,'road_cells':road_count,'obstacle_component_sizes_cells':[component['cells'] for component in components],'obstacle_component_count':len(components)}}
    record['stages'] = {
        'ground_fit_scene': {'inputs': hashes, 'outputs': {name:outputs[name] for name in ('scene.json','heightmap.npz','fitted_car.obj','car_collision_hull.obj','obstacles.npz')}, 'parameters': {'radius_m':args.radius,'angle_deg':args.angle,'cell_m':args.cell,'near_depth_m':args.near_depth,'ground_inlier_m':args.ground_inlier,'obstacle_height_m':args.obstacle_height,'obstacle_min_pixels':args.obstacle_min_pixels,'obstacle_min_cells':args.obstacle_min_cells,'obstacles_render_only':args.obstacles_render_only}, 'seconds':ground_time+scene_time},
        'simulation_replay': {'inputs': {'scene.json':outputs['scene.json'],'heightmap.npz':outputs['heightmap.npz'],'fitted_car.obj':outputs['fitted_car.obj'],'car_collision_hull.obj':outputs['car_collision_hull.obj'],'obstacles.npz':outputs['obstacles.npz']}, 'outputs': {name:outputs[name] for name in ('trajectory.json','replay_trajectory.json')}, 'parameters': {'density_kg_m3':args.density,'derived_mass_kg':mass,'friction':args.friction,'speed_m_s':args.speed,'obstacle_min_speed_m_s':args.obstacle_min_speed,'car_sleep_activation_state':'enable_sleeping','physics_hz':240,'obstacles_render_only':args.obstacles_render_only}, 'seconds':sim_time},
        'proxy_render_video': {'inputs': {'scene.json':outputs['scene.json'],'heightmap.npz':outputs['heightmap.npz'],'trajectory.json':outputs['trajectory.json'],'obstacles.npz':outputs['obstacles.npz'],'fitted_car.obj':outputs['fitted_car.obj']}, 'outputs': {name:hashval for name,hashval in outputs.items() if name.startswith(('depth/','instance_id/')) or name in ('depth.mp4','instance_id.mp4','initial_projection.json')}, 'parameters': {'fps':30,'width':w,'height':h,'camera':'original fixed DA3'}, 'seconds':render_time}}
    record['impulse_cases']={name:{'trajectory':('' if name=='coast' else name+'/')+'trajectory.json','replay':('' if name=='coast' else name+'/')+'replay_trajectory.json','impulse':specifications[name],'inertia_diagonal_kg_m2':inertia_diagonal,'max_contact_penetration_m':penetrations[name],'obstacle_contact_samples_by_id':hits[name]} for name in impulses}
    record['parameters']['ballistic_up_speed_m_s']=args.ballistic_up_speed
    record['measurements']['ballistic_in_camera']=ballistic_visible
    record['stages']['simulation_replay']['outputs'].update({name:outputs[name] for name in outputs if name.startswith(('ballistic/','obstacle/')) or name in ('impulses.json','contacts.json')})
    record['scene_artifacts']=list(SceneArtifacts.NAMES)
    record['adapters']={'physics':'PyBulletSolver','main_renderer':'RasterRenderer','comparison_renderer':'gsplat' if args.render_gaussians else None}
    record['required_renderers']=['raster'] + (['gaussian'] if args.render_gaussians else [])
    record['raster_frame_times_s']=raster_frame_times
    record['raster_replay']={'trajectory':'trajectory.json','camera':'scene.json','depth_dir':'raster_replay/depth','instance_id_dir':'raster_replay/instance_id','frame_times_s':replay_frame_times}
    record['stages']['raster_replay']={'inputs':{name:outputs[name] for name in ('scene.json','heightmap.npz','obstacles.npz','fitted_car.obj','trajectory.json')},'outputs':{name:value for name,value in outputs.items() if name.startswith('raster_replay/')},'parameters':{'fps':30,'width':w,'height':h,'camera':'original fixed DA3'},'seconds':sum(replay_frame_times)}
    if args.render_gaussians:
        record['required_renderers']=['raster','gaussian']
    write_json(out/'stage_status.json', {'stage':'contracts','status':'running'})
    # Persist independent, hash-checked boundaries after all legacy outputs exist.
    input_files = list(bundled.values())
    parameters = record['parameters']
    observation_implementation = 'saved DA3 Nested + SAM 2.1 Tiny'
    observation_parameters = {'depth_scale': 'metric_m', 'camera': 'intrinsics'}
    if args.bundle and (args.bundle/'stages/observation.json').is_file():
        source_observation = check_stage(args.bundle, 'observation')
        source_depth = check_stage(args.bundle, 'depth')
        source_segmentation = check_stage(args.bundle, 'segmentation')
        observation_implementation = source_observation['implementation']
        observation_parameters = {
            **observation_parameters,
            'depth_implementation': source_depth['implementation'],
            'segmentation_implementation': source_segmentation['implementation'],
            'depth_parameters': source_depth['parameters'],
            'segmentation_parameters': source_segmentation['parameters'],
            'selection_sha256': digest(args.bundle/'selection.json')}
    record_stage(out, 'observation', input_files[:3], input_files[:3],
                 observation_implementation, '1', observation_parameters, observation_time)
    mesh_implementation = 'saved Hunyuan3D 2.1'
    mesh_parameters = {}
    mesh_seconds = 0.0
    if args.bundle and (args.bundle/'mesh/subject_mesh.json').is_file():
        from subject_mesh import verify as verify_mesh
        verify_mesh(args.bundle/'mesh')
        source_mesh = json.loads((args.bundle/'mesh/subject_mesh.json').read_text())
        mesh_implementation = source_mesh['implementation']
        mesh_parameters = {key: source_mesh[key] for key in
                           ('provider', 'units', 'orientation', 'coordinate_frame', 'validity', 'input_hashes')}
        mesh_seconds = source_mesh['timing_s']['total']
    record_stage(out, 'subject_mesh', [bundled['hunyuan_mesh']], [bundled['hunyuan_mesh']],
                 mesh_implementation, '1', mesh_parameters, mesh_seconds)
    fit_files = [name for name in ('classification.json', 'fit_diagnostics.json', 'fit_visuals/before.png',
                 'fit_visuals/after.png') if (out/name).is_file()]
    record_stage(out, 'scene', input_files, [*SceneArtifacts.NAMES, *fit_files],
                 'first_image.reconstruct', '1', record['stages']['ground_fit_scene']['parameters'],
                 ground_time + scene_time)
    record_stage(out, 'actions', ['scene.json'], ['impulses.json'],
                 'first_image.impulses', '1', {'speed_m_s': args.speed}, 0.0)
    trajectory_files = ['trajectory.json', 'replay_trajectory.json',
                        'ballistic/trajectory.json', 'ballistic/replay_trajectory.json',
                        'obstacle/trajectory.json', 'obstacle/replay_trajectory.json',
                        'contacts.json']
    record_stage(out, 'physics', [*SceneArtifacts.NAMES, 'impulses.json'],
                 [*trajectory_files, 'fit_trial.json'],
                 'PyBulletSolver', str(api_version()),
                 record['stages']['simulation_replay']['parameters'], sim_time)
    render_files = [name for name in outputs if name.startswith(('depth/', 'instance_id/'))
                    or name in ('depth.mp4', 'instance_id.mp4', 'initial_projection.json')]
    record_stage(out, 'raster', [*SceneArtifacts.NAMES, 'trajectory.json'], render_files,
                 'RasterRenderer', '1', record['stages']['proxy_render_video']['parameters'], render_time)
    record['stage_contract_schema'] = 1
    record['stage_contracts'] = {stage: f'stages/{stage}.json' for stage in
        ('observation', 'subject_mesh', 'scene', 'actions', 'physics', 'raster')}
    record['timings_s']['total'] = time.perf_counter() - start
    write_json(out/'run.json',record)
    if args.render_gaussians:
        write_json(out/'stage_status.json', {'stage':'gaussian','status':'running'})
        subprocess.run([sys.executable,str(Path(__file__).with_name('gaussian.py')),str(out)],check=True)
    write_json(out/'stage_status.json', {'stage':'complete','status':'complete','total_seconds':time.perf_counter()-start})
    print(json.dumps({'run_dir':str(out),'frames':int(args.seconds*30),'measurements':record['measurements']},indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--bundle',type=Path,help='run bundle containing bundle.json and inputs/')
    parser.add_argument('--resume-scene',type=Path,help='reuse validated saved scene artifacts')
    parser.add_argument('--use-weak-prior', action='store_true',
                        help='select a frozen model-specific prior fit when classifier evidence qualifies')
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
    args=parser.parse_args(argv)
    values=(args.radius,args.angle,args.cell,args.near_depth,args.ground_inlier,args.obstacle_height,args.friction,args.density,args.speed,args.obstacle_min_speed,args.ballistic_up_speed,args.seconds)
    if not (all(math.isfinite(value) for value in values) and 0 < args.radius <= 100 and 0 < args.angle < 180 and 0 < args.cell <= args.radius/2 and 0 < args.near_depth <= args.radius and 0 < args.ground_inlier <= 2 and 0 < args.obstacle_height <= 10 and 1 <= args.obstacle_min_pixels <= 1000 and 1 <= args.obstacle_min_cells <= 1000 and 0 <= args.friction <= 2 and 0 < args.density <= 10000 and 0 < args.speed <= 100 and 0 < args.obstacle_min_speed <= 15 and 0 < args.seconds <= 60 and int(args.seconds*240) >= 1 and int(args.seconds*30) >= 1):
        parser.error('invalid parameter range: require positive radius/cell/density/speed/duration/inlier; angle (0,180), near-depth <= radius, friction [0,2], and at least one physics and video frame')
    try:
        build(args)
    except Exception as exc:
        out = args.output.resolve()
        if out.is_dir():
            try:
                status = json.loads((out/'stage_status.json').read_text())
            except (OSError, ValueError):
                status = {'stage':'setup'}
            write_json(out/'stage_status.json', {'stage':status['stage'], 'status':'failed',
                'error_type':type(exc).__name__, 'reason':str(exc)})
        raise


def main_with_arguments(output, bundle, seconds, use_weak_prior=False):
    main([str(output), '--bundle', str(bundle), '--seconds', str(seconds)] +
         (['--use-weak-prior'] if use_weak_prior else []))


if __name__ == '__main__':
    main()
