"""Reproducible PyBullet timing and quality sweep over saved scene artifacts.

Usage (from experiments/skeleton):
  python benchmark_physics.py RUN_DIR --action coast --output result.json
  python benchmark_physics.py RUN_DIR --sweep
No image reconstruction or production-default changes are performed.
"""
import argparse
import itertools
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from artifacts import SceneArtifacts


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def boundary_segments(geometry, radius):
    import numpy as np
    mask, cell, origin = geometry.traversable, geometry.cell_m, geometry.origin_x
    segments = []
    for j, i in np.argwhere(mask):
        x, y, h = origin + i * cell, j * cell, cell / 2
        if j == 0 or not mask[j-1, i]: segments.append(((x-h,y-h),(x+h,y-h)))
        if j == mask.shape[0]-1 or not mask[j+1, i]: segments.append(((x-h,y+h),(x+h,y+h)))
        if i == 0 or not mask[j, i-1]: segments.append(((x-h,y-h),(x-h,y+h)))
        if i == mask.shape[1]-1 or not mask[j, i+1]: segments.append(((x+h,y-h),(x+h,y+h)))
    if 'support_evidence' not in geometry.scene:
        segments = []
        for t in np.linspace(-geometry.half_angle, geometry.half_angle, 65)[:-1]:
            t2=t+2*geometry.half_angle/64
            segments.append(((radius*math.sin(t),radius*math.cos(t)),
                             (radius*math.sin(t2),radius*math.cos(t2))))
    if not segments:
        segments=[((0,0),(radius*math.sin(t),radius*math.cos(t)))
                  for t in (-geometry.half_angle, geometry.half_angle)]
    return segments


def merge_collinear_segments(segments, tolerance=1e-9):
    """Merge contiguous axis-aligned segments; preserve exact union endpoints."""
    groups={}
    for a,b in segments:
        if abs(a[1]-b[1]) <= tolerance:
            key=('h', round(a[1]/tolerance),); lo,hi=sorted((a[0],b[0]))
        elif abs(a[0]-b[0]) <= tolerance:
            key=('v', round(a[0]/tolerance),); lo,hi=sorted((a[1],b[1]))
        else:
            key=('other', a, b); lo,hi=0,1
        groups.setdefault(key,[]).append((lo,hi,a,b))
    merged=[]
    for (axis,*coord), parts in groups.items():
        if axis=='other':
            merged.extend((p[2],p[3]) for p in parts); continue
        parts.sort()
        runs=[]
        lo,hi=parts[0][:2]
        for start,end,*_ in parts[1:]:
            if start <= hi+tolerance: hi=max(hi,end)
            else: runs.append((lo,hi)); lo,hi=start,end
        runs.append((lo,hi)); fixed=coord[0]*tolerance
        merged.extend(((lo,fixed),(hi,fixed)) if axis=='h' else ((fixed,lo),(fixed,hi)) for lo,hi in runs)
    return merged


def boundaries_equivalent(original, merged, tolerance=1e-8):
    def covered(point, segments):
        x,y=point
        for (ax,ay),(bx,by) in segments:
            cross=abs((x-ax)*(by-ay)-(y-ay)*(bx-ax))
            if cross <= tolerance and min(ax,bx)-tolerance <= x <= max(ax,bx)+tolerance and min(ay,by)-tolerance <= y <= max(ay,by)+tolerance:
                return True
        return False
    return all(covered(p, merged) for s in original for p in s) and all(covered(p, original) for s in merged for p in s)


def create_world(p, geometry, settings, root, hz, substeps, iterations, merged_walls, geometry_mode='exact'):
    from physics import contact_solver_parameters
    p.setGravity(0,0,-9.81); p.setTimeStep(1/hz)
    params=contact_solver_parameters()
    params.update(fixedTimeStep=1/hz, numSubSteps=substeps, numSolverIterations=iterations)
    p.setPhysicsEngineParameter(**params)
    g=geometry
    plane_ground=geometry_mode in ('plane_ground','all_boxes')
    box_car=geometry_mode in ('box_car','all_boxes')
    box_obstacles=geometry_mode in ('box_obstacles','all_boxes')
    if plane_ground:
        ground_shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=[100,100,.05])
        ground=p.createMultiBody(0,ground_shape,basePosition=[0,0,g.flat_z-.05])
    else:
        ground_shape=p.createCollisionShape(p.GEOM_MESH,vertices=g.vertices,indices=g.indices,flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        ground=p.createMultiBody(0,ground_shape)
    p.changeDynamics(ground,-1,lateralFriction=settings.friction,restitution=0)
    obstacles={}
    for component,(vertices,indices) in zip(g.components,g.obstacle_meshes):
        if not component['collision_enabled']:
            shape=-1; position=[0,0,0]
        elif box_obstacles:
            import numpy as np
            points=np.asarray(vertices,dtype=float); lower=points.min(axis=0); upper=points.max(axis=0)
            shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=((upper-lower)/2).tolist())
            position=((upper+lower)/2).tolist()
        else:
            shape=p.createCollisionShape(p.GEOM_MESH,vertices=vertices,indices=indices,flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
            position=[0,0,0]
        body=p.createMultiBody(0,shape,basePosition=position)
        if shape!=-1: p.changeDynamics(body,-1,lateralFriction=settings.friction,restitution=0)
        obstacles[body]=component['instance_id']
    segs=boundary_segments(g,settings.radius)
    wall_segments=[] if geometry_mode in ('no_walls','all_boxes') else (merge_collinear_segments(segs) if merged_walls else segs)
    for u,v in wall_segments:
        dx,dy=v[0]-u[0],v[1]-u[1]
        shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=[math.hypot(dx,dy)/2,.04,.5])
        p.createMultiBody(0,shape,basePosition=[(u[0]+v[0])/2,(u[1]+v[1])/2,g.flat_z+.5],baseOrientation=p.getQuaternionFromEuler([0,0,math.atan2(dy,dx)]))
    hull_shape=(p.createCollisionShape(p.GEOM_BOX,halfExtents=[x/2 for x in g.scene['car']['size_m']])
                if box_car else p.createCollisionShape(p.GEOM_MESH,vertices=g.hull.vertices.tolist()))
    body=p.createMultiBody(g.mass,hull_shape,basePosition=g.car_pos,baseOrientation=g.car_orientation)
    p.changeDynamics(body,-1,lateralFriction=1.,restitution=0,linearDamping=0,angularDamping=0,
                     activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
    return ground,body,obstacles,len(segs),len(wall_segments)


def run_case(root, geometry, settings, action_vectors, seconds, cfg):
    import pybullet as p
    client=p.connect(p.DIRECT)
    try:
        action_results={}
        build_times=[]
        for action,velocity in action_vectors.items():
            p.resetSimulation()
            started=time.perf_counter()
            ground,body,obstacles,wall_count,body_count=create_world(p,geometry,settings,root,cfg['hz'],cfg['substeps'],cfg['iterations'],cfg['merged_walls'],cfg['geometry_mode'])
            build_times.append(time.perf_counter()-started)
            initial_position=list(p.getBasePositionAndOrientation(body)[0])
            p.resetBaseVelocity(body,linearVelocity=velocity)
            timings=[]; positions=[]; speeds=[]; penetration=0.; contact_samples=0; obstacle_contacts=0
            rollout_started=time.perf_counter()
            for _ in range(max(1,round(seconds*cfg['hz']))):
                tick=time.perf_counter(); p.stepSimulation(); timings.append(time.perf_counter()-tick)
                pos,_=p.getBasePositionAndOrientation(body); linear,_=p.getBaseVelocity(body)
                positions.append(pos); speeds.append(math.sqrt(sum(v*v for v in linear)))
                if cfg['contact_queries']:
                    contacts=p.getContactPoints(bodyA=body); contact_samples+=len(contacts)>0
                    penetration=max(penetration,max((max(0.,-float(c[8])) for c in contacts),default=0.))
                    obstacle_contacts+=sum(c[2] in obstacles for c in contacts)
            wall=time.perf_counter()-rollout_started
            action_results[action]={'rollout_wall_seconds':wall,'steps':len(timings),
              'step_time_seconds':{'p50':statistics.median(timings),'p95':percentile(timings,.95)},
              'initial_position_m':initial_position,'final_position_m':list(positions[-1]),'horizontal_displacement_m':math.hypot(positions[-1][0]-initial_position[0],positions[-1][1]-initial_position[1]),
              'final_speed_m_s':speeds[-1],'max_penetration_m':penetration if cfg['contact_queries'] else None,
              'contact_steps':contact_samples if cfg['contact_queries'] else None,'obstacle_contact_points':obstacle_contacts if cfg['contact_queries'] else None}
        segs=boundary_segments(geometry,settings.radius); merged=merge_collinear_segments(segs)
        return {**cfg,'world_build_seconds':statistics.mean(build_times),
          'boundary_segment_count':wall_count,'boundary_collision_body_count':body_count,
          'merged_boundary_collision_body_count':len(merged),
          'boundary_merge_exact':boundaries_equivalent(segs,merged),'actions':action_results}
    finally: p.disconnect(client)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('run_directory',type=Path)
    ap.add_argument('--action',choices=('coast','ballistic','obstacle'),default='coast')
    ap.add_argument('--hz',type=int,default=240); ap.add_argument('--substeps',type=int,default=8)
    ap.add_argument('--iterations',type=int,default=100); ap.add_argument('--contact-queries',action='store_true',default=True)
    ap.add_argument('--no-contact-queries',action='store_false',dest='contact_queries')
    ap.add_argument('--merge-walls',action='store_true',default=False); ap.add_argument('--keep-walls',action='store_false',dest='merge_walls')
    ap.add_argument('--geometry-mode',choices=('exact','no_walls','plane_ground','box_car','box_obstacles','all_boxes'),default='exact')
    ap.add_argument('--seconds',type=float,default=5.); ap.add_argument('--all-actions',action='store_true',help='run coast, ballistic, and obstacle in each configuration')
    ap.add_argument('--substep-sweep',action='store_true',help='run 60/120/240 Hz by 1/2/4/8 substeps')
    ap.add_argument('--sweep',action='store_true',help='full parameter grid, beyond the primary substep sweep')
    ap.add_argument('--output',type=Path); args=ap.parse_args(argv)
    root=args.run_directory; geometry=SceneArtifacts(root).load()
    run=json.loads((root/'run.json').read_text()); impulses=json.loads((root/'impulses.json').read_text())
    friction=run['parameters']['friction']; radius=run['parameters']['radius_m']
    settings=SimpleNamespace(friction=friction,radius=radius)
    action_vectors={name:spec['delta_velocity_m_s'] for name,spec in impulses.items()
                    if name in (('coast','ballistic','obstacle') if args.all_actions else (args.action,))}
    if args.sweep:
        configs=[dict(hz=h,substeps=s,iterations=i,contact_queries=c,merged_walls=m,geometry_mode=args.geometry_mode)
          for h,s,i,c,m in itertools.product((60,120,240),(1,2,4,8),(10,20,50,100),(False,True),(False,True))]
    elif args.substep_sweep:
        configs=[dict(hz=h,substeps=s,iterations=args.iterations,contact_queries=args.contact_queries,merged_walls=args.merge_walls,geometry_mode=args.geometry_mode)
          for h,s in itertools.product((60,120,240),(1,2,4,8))]
    else: configs=[dict(hz=args.hz,substeps=args.substeps,iterations=args.iterations,contact_queries=args.contact_queries,merged_walls=args.merge_walls,geometry_mode=args.geometry_mode)]
    result={'schema':1,'run_directory':str(root.resolve()),'action':args.action,'all_actions':args.all_actions,'simulated_seconds':args.seconds,
      'actions':list(action_vectors),'cases':[run_case(root,geometry,settings,action_vectors,args.seconds,c) for c in configs]}
    encoded=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.output: args.output.write_text(encoded)
    else: print(encoded,end='')

if __name__=='__main__': main()
