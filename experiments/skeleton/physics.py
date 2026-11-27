"""Physics trajectory seam: a prepared world and impulse produce saved states.

The adapter owns collision-world construction, stepping, and contact
measurements. A replacement can use the same reconstructed scene files.
"""
import math
import numpy as np
import json
import time
from artifacts import SceneArtifacts
from physics_actions import action_records


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def api_version():
    import pybullet
    return pybullet.getAPIVersion()


class PyBulletSolver:
    def __init__(self, bullet, hz=240):
        self.p=bullet
        self.hz=hz

    def solve(self, body, obstacle_bodies, velocity, seconds):
        p=self.p
        pos,quat=p.getBasePositionAndOrientation(body)
        vel,omega=p.getBaseVelocity(body)
        rows=[[*pos,*quat,*vel,*omega]]
        penetration=0.0
        obstacle_contacts={}
        p.resetBaseVelocity(body,linearVelocity=velocity)
        for _ in range(int(seconds*self.hz)-1):
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

    def create_world(self, geometry, args, out):
        p=self.p
        vertices=geometry.vertices; indices=geometry.indices
        components=geometry.components; obstacle_meshes=geometry.obstacle_meshes
        half_angle=geometry.half_angle; flat_z=geometry.flat_z
        hull=geometry.hull; mass=geometry.mass
        expected_mass=float(geometry.scene['car']['density_kg_m3'])*float(geometry.scene['car']['volume_m3'])
        if not math.isfinite(expected_mass) or expected_mass <= 0 or not math.isclose(mass, expected_mass, rel_tol=1e-9):
            raise ValueError('dynamic body mass disagrees with density times volume')
        car_pos=geometry.car_pos; car_orientation=geometry.car_orientation
        p.setGravity(0,0,-9.81)
        p.setTimeStep(1/240)
        p.setPhysicsEngineParameter(fixedTimeStep=1/240, numSolverIterations=100, deterministicOverlappingPairs=1,
                                    useSplitImpulse=1, splitImpulsePenetrationThreshold=0.0)
        ground_collision = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=indices, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        ground_visual = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices, rgbaColor=[.5,.5,.5,1])
        ground = p.createMultiBody(0, ground_collision, ground_visual)
        p.changeDynamics(ground,-1,lateralFriction=args.friction,restitution=0)
        obstacle_bodies={}
        for component,(mesh_vertices,mesh_indices) in zip(components,obstacle_meshes):
            collision=-1 if not component['collision_enabled'] else p.createCollisionShape(p.GEOM_MESH,vertices=mesh_vertices,indices=mesh_indices,flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
            visual=p.createVisualShape(p.GEOM_MESH,vertices=mesh_vertices,indices=mesh_indices,rgbaColor=[.1,.5,.1,1])
            obstacle_body=p.createMultiBody(0,collision,visual)
            if collision != -1:
                p.changeDynamics(obstacle_body,-1,lateralFriction=args.friction,restitution=0)
            obstacle_bodies[obstacle_body]=component['instance_id']
        walls = []
        # Every exposed traversable-cell edge is an invisible collision wall.
        segments = []
        mask=geometry.traversable
        cell=geometry.cell_m
        origin=geometry.origin_x
        for j,i in np.argwhere(mask):
            x=origin+i*cell; y=j*cell; half=cell/2
            if j==0 or not mask[j-1,i]: segments.append(((x-half,y-half),(x+half,y-half)))
            if j==mask.shape[0]-1 or not mask[j+1,i]: segments.append(((x-half,y+half),(x+half,y+half)))
            if i==0 or not mask[j,i-1]: segments.append(((x-half,y-half),(x-half,y+half)))
            if i==mask.shape[1]-1 or not mask[j,i+1]: segments.append(((x+half,y-half),(x+half,y+half)))
        # Retain the sector edge for compatibility with legacy saved scenes.
        if 'support_evidence' not in geometry.scene:
            segments = []
        for t in np.linspace(-half_angle, half_angle, 65)[:-1] if not segments else ():
            t2 = t + 2*half_angle/64
            segments.append(((args.radius*math.sin(t),args.radius*math.cos(t)),(args.radius*math.sin(t2),args.radius*math.cos(t2))))
        if not segments:
            segments += [((0,0),(args.radius*math.sin(t),args.radius*math.cos(t))) for t in (-half_angle,half_angle)]
        for (u,v) in segments:
            dx,dy = v[0]-u[0],v[1]-u[1]
            mid = [(u[0]+v[0])/2,(u[1]+v[1])/2,flat_z+.5]
            shape = p.createCollisionShape(p.GEOM_BOX,halfExtents=[math.hypot(dx,dy)/2,.04,.5])
            walls.append(p.createMultiBody(0,shape,basePosition=mid,baseOrientation=p.getQuaternionFromEuler([0,0,math.atan2(dy,dx)])))
        shape = p.createCollisionShape(p.GEOM_MESH, vertices=hull.vertices.tolist())
        visual = p.createVisualShape(p.GEOM_MESH, fileName=str(out/'fitted_car.obj'), rgbaColor=[1,0,0,1])
        body = p.createMultiBody(mass,shape,visual,basePosition=car_pos,baseOrientation=car_orientation)
        # Bullet multiplies the two bodies' friction values at contact.
        # Keep the effective ground/car coefficient equal to the nominal μ.
        p.changeDynamics(body,-1,lateralFriction=1.0,restitution=0,linearDamping=0,angularDamping=0,
                         activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
        return ground, body, obstacle_bodies

def solve_saved_scene(out, args, scene, component_ids, tops, actions=None):
    geometry = SceneArtifacts(out).load()
    components = scene['obstacles']['components']
    car_pos = scene['car']['initial_position_m']
    mass = scene['car']['mass_kg']
    width, length, _ = scene['car']['size_m']
    origin_x = -(component_ids.shape[0]-1)/2*args.cell
    sim_start = time.perf_counter()
    import pybullet as p
    client = p.connect(p.DIRECT)
    try:
        solver=PyBulletSolver(p)
        ground, body, obstacle_bodies = solver.create_world(geometry,args,out)
        inertia_diagonal=list(p.getDynamicsInfo(body,-1)[2])
        shape_counts={instance_id:len(p.getCollisionShapeData(obstacle_body,-1)) for obstacle_body,instance_id in obstacle_bodies.items()}
        for component in components:
            component['collision_shape_count']=shape_counts[component['instance_id']]
            if bool(component['collision_shape_count'])!=component['collision_enabled']:
                raise RuntimeError('obstacle collision shape disagrees with fallback mode')
        scene['contact_friction']={'ground_lateral_friction':args.friction,'car_lateral_friction':1.0,'effective_ground_car_friction':args.friction,'obstacle_lateral_friction':args.friction,'restitution':0.0}
        scene['physics_sleep']={'car_activation_state':'enable_sleeping','activation_state_value':p.ACTIVATION_STATE_ENABLE_SLEEPING}
        scene['physics_solver']={'split_impulse':True,'split_impulse_penetration_threshold_m':0.0}
        write_json(out/'scene.json',scene)
        targets=[]
        solid_ids={c['instance_id'] for c in components if c['collision_enabled']}
        for cy,cx in np.argwhere(np.isin(component_ids,list(solid_ids))):
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
        impulses = actions if actions is not None else {'coast':[0.,args.speed,0.], 'ballistic':[0.,args.speed,args.ballistic_up_speed], 'obstacle':(direction*obstacle_speed).tolist()}
        specifications = action_records(impulses, mass, car_pos)
        if actions is None:
            specifications['obstacle']['target_obstacle_id'] = target_id
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
            if name!='coast': (out/name).mkdir(exist_ok=True)
            write_json(out/(prefix+'trajectory.json'),case_rows)
            write_json(out/(prefix+'replay_trajectory.json'),replay)
        if any(len(rows) != int(args.seconds * solver.hz) for rows in trajectories.values()):
            raise RuntimeError('physics trajectory has unexpected frame count')
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
    except Exception as exc:
        write_json(out/'physics_failure.json', {'stage':'physics','reason':str(exc),'type':type(exc).__name__})
        raise
    finally:
        p.disconnect(client)


class BulletRasterSession:
    """Renderer-facing frame source; Bullet handles never leave this adapter."""
    def __init__(self, artifacts, args, width, height, near, far, projection):
        self.artifacts=artifacts;self.args=args
        self.width=width;self.height=height;self.near=near;self.far=far
        self.projection=projection

    def __enter__(self):
        import pybullet as p
        self.p=p
        self.client=p.connect(p.DIRECT)
        try:
            geometry=self.artifacts.load()
            self.ground,self.body,self.obstacles=PyBulletSolver(p).create_world(
                geometry,self.args,self.artifacts.root)
            camera=geometry.scene['camera']
            self.camera_forward=camera['forward']
            self.camera_up=camera['up']
            self.view=p.computeViewMatrix([0,0,0],camera['forward'],camera['up'])
            return self
        except BaseException:
            p.disconnect(self.client)
            raise

    def __exit__(self, *_):
        self.p.disconnect(self.client)

    def set_camera_yaw(self, yaw_deg):
        """Rotate the saved camera about its up axis for a nearby geometry view."""
        forward=np.asarray(self.camera_forward,dtype=float)
        up=np.asarray(self.camera_up,dtype=float)
        axis=up/np.linalg.norm(up)
        angle=math.radians(yaw_deg)
        direction=(forward*math.cos(angle)+np.cross(axis,forward)*math.sin(angle)
                   +axis*np.dot(axis,forward)*(1-math.cos(angle)))
        self.view=self.p.computeViewMatrix([0,0,0],direction.tolist(),up.tolist())

    def frame(self, row):
        p=self.p
        p.resetBasePositionAndOrientation(self.body,row[:3],row[3:7])
        _,_,_,zbuf,seg=p.getCameraImage(self.width,self.height,
            viewMatrix=self.view,projectionMatrix=self.projection,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        raw=np.asarray(seg,dtype=np.int32).reshape(self.height,self.width)
        ids=np.zeros((self.height,self.width),dtype=np.uint16)
        ids[(raw & 0xffffff)==self.ground]=1
        ids[(raw & 0xffffff)==self.body]=2
        for handle,instance_id in self.obstacles.items():
            ids[(raw & 0xffffff)==handle]=instance_id
        ids[raw<0]=0
        zbuf=np.asarray(zbuf).reshape(self.height,self.width)
        metric=self.far*self.near/(self.far-(self.far-self.near)*zbuf)
        return metric,ids
