"""Physics trajectory seam: a prepared world and impulse produce saved states.

The adapter owns collision-world construction, stepping, and contact
measurements. A replacement can use the same reconstructed scene files.
"""
import math
import numpy as np


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
        car_pos=geometry.car_pos; car_orientation=geometry.car_orientation
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
        # Bullet multiplies the two bodies' friction values at contact.
        # Keep the effective ground/car coefficient equal to the nominal μ.
        p.changeDynamics(body,-1,lateralFriction=1.0,restitution=0,linearDamping=0,angularDamping=0,
                         activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
        return ground, body, obstacle_bodies
