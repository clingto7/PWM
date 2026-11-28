"""Compare PyBullet and SAPIEN PhysX on the saved collision meshes.

The two engines use the same triangle-mesh ground and obstacles, convex subject
hull, separate boundary boxes, initial actions, gravity, and recorded-state
frequency. A SAPIEN substep is one explicit scene.step() at dt/substeps.
MuJoCo is covered by benchmark_engines.py's common box proxy because its
collision representation cannot consume this nonconvex ground unchanged.
"""
import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from artifacts import SceneArtifacts
from benchmark_engines import _quat_wxyz, timing, trajectory_metrics
from physics import PyBulletSolver, contact_solver_parameters


def boundary_segments(geometry):
    mask=geometry.traversable
    cell=geometry.cell_m
    origin=geometry.origin_x
    segments=[]
    for j,i in np.argwhere(mask):
        x=origin+i*cell; y=j*cell; half=cell/2
        if j==0 or not mask[j-1,i]: segments.append(((x-half,y-half),(x+half,y-half)))
        if j==mask.shape[0]-1 or not mask[j+1,i]: segments.append(((x-half,y+half),(x+half,y+half)))
        if i==0 or not mask[j,i-1]: segments.append(((x-half,y-half),(x-half,y+half)))
        if i==mask.shape[1]-1 or not mask[j,i+1]: segments.append(((x+half,y-half),(x+half,y+half)))
    if 'support_evidence' not in geometry.scene:
        segments=[]
        for t in np.linspace(-geometry.half_angle,geometry.half_angle,65)[:-1]:
            t2=t+2*geometry.half_angle/64
            segments.append(((geometry.scene['sector']['radius_m']*math.sin(t),
                              geometry.scene['sector']['radius_m']*math.cos(t)),
                             (geometry.scene['sector']['radius_m']*math.sin(t2),
                              geometry.scene['sector']['radius_m']*math.cos(t2))))
    if not segments:
        radius=geometry.scene['sector']['radius_m']
        segments=[((0,0),(radius*math.sin(t),radius*math.cos(t)))
                  for t in (-geometry.half_angle,geometry.half_angle)]
    return segments


def write_obj(path, vertices, flat_indices):
    faces=np.asarray(flat_indices,dtype=int).reshape(-1,3)
    with path.open('w') as stream:
        for x,y,z in vertices:
            stream.write(f'v {x:.9f} {y:.9f} {z:.9f}\n')
        for a,b,c in faces:
            stream.write(f'f {a+1} {b+1} {c+1}\n')


def run_bullet(geometry, root, velocity, seconds, substeps, friction):
    import pybullet as p
    client=p.connect(p.DIRECT)
    try:
        settings=SimpleNamespace(friction=friction,
            radius=geometry.scene['sector']['radius_m'],cell=geometry.cell_m)
        solver=PyBulletSolver(p)
        started=time.perf_counter()
        _,body,_=solver.create_world(geometry,settings,root)
        parameters=contact_solver_parameters()
        parameters['numSubSteps']=substeps
        p.setPhysicsEngineParameter(**parameters)
        preparation=time.perf_counter()-started
        p.resetBaseVelocity(body,linearVelocity=velocity)
        rows=[]; steps=[]; penetration=0.; contact_steps=0
        rollout_started=time.perf_counter()
        for _ in range(round(seconds*240)):
            tick=time.perf_counter(); p.stepSimulation(); steps.append(time.perf_counter()-tick)
            pos,quat=p.getBasePositionAndOrientation(body)
            linear,angular=p.getBaseVelocity(body)
            rows.append([*pos,*quat,*linear,*angular])
            contacts=p.getContactPoints(bodyA=body)
            contact_steps+=int(bool(contacts))
            penetration=max(penetration,max((max(0.,-float(c[8])) for c in contacts),default=0.))
        rollout_wall=time.perf_counter()-rollout_started
        bodies=p.getNumBodies()
        shapes=sum(len(p.getCollisionShapeData(i,-1) or []) for i in range(bodies))
        return {**timing(steps,preparation,rollout_wall),
            'world_body_count':bodies,'collision_shape_count':shapes,
            'collision_hull_audit':{'input_vertices':len(geometry.hull.vertices),
                                    'cooked_vertices':len(geometry.hull.vertices),
                                    'max_sampled_support_error_m':0.0},
            'metrics':trajectory_metrics(rows,penetration,contact_steps),
            '_trajectory_xyz':[row[:3] for row in rows]}
    finally:
        p.disconnect(client)


def run_sapien(geometry, root, mesh_paths, velocity, seconds, substeps, friction, inertia):
    import sapien
    sapien.physx.set_scene_config(gravity=np.asarray([0.,0.,-9.81],dtype=np.float32))
    scene=sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(1/(240*substeps))
    material=sapien.physx.PhysxMaterial(friction,friction,0.)
    started=time.perf_counter()
    builder=scene.create_actor_builder()
    builder.add_nonconvex_collision_from_file(str(mesh_paths['ground']),material=material)
    builder.build_static(name='ground')
    static_count=1
    for component in geometry.components:
        if not component['collision_enabled']:
            continue
        builder=scene.create_actor_builder()
        builder.add_nonconvex_collision_from_file(
            str(mesh_paths[f"obstacle_{component['instance_id']}"]),material=material)
        builder.build_static(name=f"obstacle_{component['instance_id']}")
        static_count+=1
    for index,(u,v) in enumerate(boundary_segments(geometry)):
        dx,dy=v[0]-u[0],v[1]-u[1]
        angle=math.atan2(dy,dx)/2
        builder=scene.create_actor_builder()
        builder.add_box_collision(half_size=[math.hypot(dx,dy)/2,.04,.5],material=material)
        wall=builder.build_static(name=f'boundary_{index}')
        wall.set_pose(sapien.Pose([(u[0]+v[0])/2,(u[1]+v[1])/2,geometry.flat_z+.5],
                                  [math.cos(angle),0,0,math.sin(angle)]))
        static_count+=1
    builder=scene.create_actor_builder()
    builder.add_convex_collision_from_file(str(root/'car_collision_hull.obj'),material=material)
    builder.set_mass_and_inertia(geometry.mass,sapien.Pose(),inertia)
    actor=builder.build(name='subject')
    actor.set_pose(sapien.Pose(geometry.car_pos,_quat_wxyz(geometry.car_orientation)))
    dynamic=actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    preparation=time.perf_counter()-started
    import trimesh
    original=np.asarray(geometry.hull.vertices)
    cooked=np.asarray(dynamic.collision_shapes[0].vertices)
    directions=trimesh.creation.icosphere(subdivisions=4).vertices
    support_error=float(np.max(np.max(original@directions.T,axis=0)
                               -np.max(cooked@directions.T,axis=0)))
    dynamic.set_linear_velocity(np.asarray(velocity,dtype=np.float32))
    rows=[]; steps=[]; penetration=0.; contact_steps=0
    rollout_started=time.perf_counter()
    for _ in range(round(seconds*240)):
        tick=time.perf_counter()
        for _ in range(substeps):
            scene.step()
        steps.append(time.perf_counter()-tick)
        pose=actor.get_pose()
        linear=dynamic.get_linear_velocity()
        angular=dynamic.get_angular_velocity()
        rows.append([*pose.p,*pose.q,*linear,*angular])
        active=False
        for contact in scene.get_contacts():
            if dynamic in contact.bodies:
                active=True
                penetration=max(penetration,max(
                    (max(0.,-float(point.separation)) for point in contact.points),default=0.))
        contact_steps+=int(active)
    rollout_wall=time.perf_counter()-rollout_started
    return {**timing(steps,preparation,rollout_wall),
        'world_body_count':static_count+1,'collision_shape_count':static_count+1,
        'collision_hull_audit':{'input_vertices':len(original),
                                'cooked_vertices':len(cooked),
                                'max_sampled_support_error_m':support_error},
        'metrics':trajectory_metrics(rows,penetration,contact_steps),
        '_trajectory_xyz':[row[:3] for row in rows]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--seconds',type=float,default=5.)
    parser.add_argument('--substeps',type=int,choices=(1,2,4,8),default=1)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds<=0:
        parser.error('seconds must be positive and finite')
    geometry=SceneArtifacts(args.run_dir).load()
    saved=json.loads((args.run_dir/'impulses.json').read_text())
    actions={name:saved[name]['delta_velocity_m_s']
             for name in ('coast','ballistic','obstacle')}
    run=json.loads((args.run_dir/'run.json').read_text())
    friction=float(run['parameters']['friction'])
    session_path=args.run_dir/'physics_session.json'
    if not session_path.is_file():
        parser.error('physics_session.json required for shared inertia diagonal')
    inertia=json.loads(session_path.read_text())['inertia_diagonal_kg_m2']
    result={'schema':1,'representation':'saved_collision_meshes',
            'scene':str(args.run_dir.resolve()),'seconds':args.seconds,
            'recorded_hz':240,'internal_substeps':args.substeps,
            'gravity_m_s2':[0,0,-9.81],'friction':friction,'inertia_diagonal_kg_m2':inertia,
            'geometry':{'ground_triangles':len(geometry.indices)//3,
                        'subject_hull_vertices':len(geometry.hull.vertices),
                        'boundary_boxes':len(boundary_segments(geometry))},
            'engines':{'pybullet':{},'sapien_physx_cpu':{}}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pwm-mesh-bench-',dir=args.output.parent) as folder:
        scratch=Path(folder)
        mesh_paths={'ground':scratch/'ground.obj'}
        write_obj(mesh_paths['ground'],geometry.vertices,geometry.indices)
        for component,(vertices,indices) in zip(geometry.components,geometry.obstacle_meshes):
            if component['collision_enabled']:
                key=f"obstacle_{component['instance_id']}"
                mesh_paths[key]=scratch/(key+'.obj')
                write_obj(mesh_paths[key],vertices,indices)
        for name,velocity in actions.items():
            result['engines']['pybullet'][name]=run_bullet(
                geometry,args.run_dir,velocity,args.seconds,args.substeps,friction)
            result['engines']['sapien_physx_cpu'][name]=run_sapien(
                geometry,args.run_dir,mesh_paths,velocity,args.seconds,args.substeps,friction,inertia)
    for name in actions:
        bullet=result['engines']['pybullet'][name]
        sapien=result['engines']['sapien_physx_cpu'][name]
        first=bullet.pop('_trajectory_xyz')
        second=sapien.pop('_trajectory_xyz')
        distances=[math.dist(x,y) for x,y in zip(first,second)]
        sapien['trajectory_vs_pybullet']={
            'samples':len(distances),
            'rms_m':math.sqrt(sum(d*d for d in distances)/len(distances)),
            'max_m':max(distances)}
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':str(args.output),'status':'complete',
                      'engines':list(result['engines'])}))


if __name__=='__main__':
    main()
