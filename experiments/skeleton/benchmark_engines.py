"""Canonical box-proxy benchmark over a saved SceneArtifacts run.

The benchmark deliberately reduces production meshes to one flat ground box,
the fitted car's oriented box, and static obstacle AABBs derived from saved
obstacle mesh vertices. It is not an exact-mesh production comparison.

Run from experiments/skeleton, for example:
  python benchmark_engines.py /path/to/run --seconds 5
Without --action, the three saved impulses are used.
Optional engines are imported lazily; no dependencies are installed here.
"""
import argparse
import importlib
import json
import math
import time
from pathlib import Path

ENGINES = {"pybullet": "pybullet", "mujoco": "mujoco", "sapien": "sapien"}
DT = 1 / 240
GRAVITY = (0., 0., -9.81)


def parse_actions(values):
    actions = {}
    for item in values:
        try:
            name, raw = item.split("=", 1)
            vector = [float(v) for v in raw.split(",")]
        except (ValueError, AttributeError):
            raise ValueError(f"invalid action {item!r}; expected name=vx,vy,vz")
        if not name or name in actions or len(vector) != 3 or not all(map(math.isfinite, vector)):
            raise ValueError(f"invalid or duplicate action {item!r}")
        actions[name] = vector
    if not actions:
        raise ValueError("at least one --action name=vx,vy,vz is required")
    return actions


def box_proxy(scene, obstacle_meshes):
    """Return engine-independent OBB/AABB proxy data from loaded saved geometry."""
    car = scene["car"]
    size = [float(x) for x in car["size_m"]]
    if len(size) != 3 or min(size) <= 0 or not all(map(math.isfinite, size)):
        raise ValueError("saved car size_m must contain three positive finite dimensions")
    obstacles = []
    components = scene["obstacles"]["components"]
    if len(components) != len(obstacle_meshes):
        raise ValueError("obstacle component/mesh count mismatch")
    for component, (vertices, _faces) in zip(components, obstacle_meshes):
        if not component.get("collision_enabled", False):
            continue
        if not vertices:
            raise ValueError(f"colliding obstacle {component['instance_id']} has no vertices")
        points = [[float(v) for v in row] for row in vertices]
        if any(len(p) != 3 or not all(map(math.isfinite, p)) for p in points):
            raise ValueError(f"obstacle {component['instance_id']} has invalid vertices")
        lo = [min(p[k] for p in points) for k in range(3)]
        hi = [max(p[k] for p in points) for k in range(3)]
        extents = [hi[k]-lo[k] for k in range(3)]
        if min(extents) <= 0:
            raise ValueError(f"obstacle {component['instance_id']} has degenerate AABB")
        obstacles.append({"instance_id": int(component["instance_id"]),
                          "center": [(lo[k]+hi[k])/2 for k in range(3)],
                          "size": extents})
    return {"ground_z": float(scene["ground"]["flat_world_z_m"]),
            "car": {"size": size, "mass": float(car["mass_kg"]),
                    "position": [float(x) for x in car["initial_position_m"]],
                    "orientation_xyzw": [float(x) for x in car["initial_orientation_xyzw"]]},
            "obstacles": obstacles}


def trajectory_metrics(rows, penetration, contact_steps, reference=None):
    if not rows:
        raise ValueError("trajectory must contain at least one state")
    positions = [list(map(float, row[:3])) for row in rows]
    final = positions[-1]
    start = positions[max(0, len(positions)-121)]
    rest_drift = math.sqrt(sum((a-b)**2 for a,b in zip(final,start)))
    result = {"max_penetration_m": float(penetration),
              "rest_drift_last_0_5s_m": rest_drift,
              "contact_steps": int(contact_steps), "final_position_m": final}
    if reference is not None:
        n = min(len(positions), len(reference))
        ds = [math.sqrt(sum((positions[i][k]-reference[i][k])**2 for k in range(3))) for i in range(n)]
        result["trajectory_rms_difference_m"] = math.sqrt(sum(d*d for d in ds)/n)
        result["trajectory_max_difference_m"] = max(ds)
    return result


def timing(times, preparation, rollout_wall):
    import statistics
    return {"world_preparation_s": preparation, "rollout_wall_s": rollout_wall,
            "step_only_s": sum(times),
            "p50_step_ms": statistics.median(times)*1000,
            "p95_step_ms": sorted(times)[max(0, math.ceil(.95*len(times))-1)]*1000}


def _quat_wxyz(q):
    return [q[3], q[0], q[1], q[2]]


def _simulate(engine, proxy, velocity, seconds, friction):
    """Build and step a world; engine adapters receive identical proxy data."""
    nsteps = max(1, round(seconds/DT))
    rows=[]; step_times=[]; penetration=0.; contacts=0
    if engine == "pybullet":
        p=importlib.import_module("pybullet"); cid=p.connect(p.DIRECT)
        try:
            prep0 = time.perf_counter()
            p.setGravity(*GRAVITY); p.setTimeStep(DT)
            p.setPhysicsEngineParameter(numSubSteps=1, numSolverIterations=100)
            ground=p.createCollisionShape(p.GEOM_BOX,halfExtents=[100,100,.05])
            p.createMultiBody(0,ground,basePosition=[0,0,proxy["ground_z"]-.05])
            for ob in proxy["obstacles"]:
                sh=p.createCollisionShape(p.GEOM_BOX,halfExtents=[x/2 for x in ob["size"]])
                p.createMultiBody(0,sh,basePosition=ob["center"])
            c=proxy["car"]
            sh=p.createCollisionShape(p.GEOM_BOX,halfExtents=[x/2 for x in c["size"]])
            body=p.createMultiBody(c["mass"],sh,basePosition=c["position"],baseOrientation=c["orientation_xyzw"])
            for bid in range(p.getNumBodies()):
                p.changeDynamics(bid,-1,lateralFriction=1.0 if bid==body else friction,
                                 restitution=0,linearDamping=0,angularDamping=0)
            preparation=time.perf_counter()-prep0
            p.resetBaseVelocity(body,linearVelocity=velocity)
            rollout0=time.perf_counter()
            for _ in range(nsteps):
                t=time.perf_counter(); p.stepSimulation(); step_times.append(time.perf_counter()-t)
                pos,q=p.getBasePositionAndOrientation(body); v,w=p.getBaseVelocity(body)
                rows.append([*pos,*q,*v,*w]); cp=p.getContactPoints(bodyA=body)
                penetration=max(penetration,max([0.]+[-float(x[8]) for x in cp])); contacts+=bool(cp)
            rollout_wall=time.perf_counter()-rollout0
            world_bodies=p.getNumBodies(); collision_shapes=world_bodies
        finally: p.disconnect(cid)
    elif engine == "mujoco":
        mj=importlib.import_module("mujoco")
        prep0=time.perf_counter()
        from xml.etree.ElementTree import Element, SubElement, tostring
        root=Element("mujoco"); SubElement(root,"option",timestep=str(DT),gravity=" ".join(map(str,GRAVITY)), iterations="100")
        SubElement(SubElement(root,"worldbody"),"geom",name="ground",type="box",size=f"100 100 .05",pos=f"0 0 {proxy['ground_z']-.05}",friction=f"{friction} 0.005 0.0001")
        for i,ob in enumerate(proxy["obstacles"]):
            SubElement(root.find("worldbody"),"geom",name=f"obstacle{i}",type="box",size=" ".join(str(x/2) for x in ob["size"]),pos=" ".join(map(str,ob["center"])),friction=f"{friction} 0.005 0.0001")
        c=proxy["car"]; body=SubElement(root.find("worldbody"),"body",name="car",pos=" ".join(map(str,c["position"])),quat=" ".join(map(str,_quat_wxyz(c["orientation_xyzw"]))))
        SubElement(body,"freejoint"); SubElement(body,"geom",name="car_box",type="box",size=" ".join(str(x/2) for x in c["size"]),mass=str(c["mass"]),friction=f"{friction} 0.005 0.0001")
        model=mj.MjModel.from_xml_string(tostring(root,encoding="unicode")); data=mj.MjData(model)
        data.qvel[:3]=velocity; preparation=time.perf_counter()-prep0
        rollout0=time.perf_counter()
        for _ in range(nsteps):
            t=time.perf_counter(); mj.mj_step(model,data); step_times.append(time.perf_counter()-t)
            rows.append([*data.qpos[:3],*data.qpos[3:7],*data.qvel[:6]])
            contacts+=bool(data.ncon)
            if data.ncon: penetration=max(penetration,max(0.,max(-float(data.contact[i].dist) for i in range(data.ncon))))
        rollout_wall=time.perf_counter()-rollout0
        world_bodies=model.nbody-1; collision_shapes=model.ngeom
    elif engine == "sapien":
        s=importlib.import_module("sapien")
        import numpy as np
        prep0=time.perf_counter()
        s.physx.set_scene_config(gravity=np.asarray(GRAVITY,dtype=np.float32))
        scene=s.Scene([s.physx.PhysxCpuSystem()]); scene.set_timestep(DT)
        mat=s.physx.PhysxMaterial(friction,friction,0.)
        def add_box(builder,size): builder.add_box_collision(half_size=[x/2 for x in size],material=mat)
        ground=scene.create_actor_builder(); add_box(ground,[200,200,.1]); ground_actor=ground.build_static(name="ground"); ground_actor.set_pose(s.Pose([0,0,proxy["ground_z"]-.05]))
        for ob in proxy["obstacles"]:
            b=scene.create_actor_builder(); add_box(b,ob["size"]); static_actor=b.build_static(name=f"obstacle_{ob['instance_id']}"); static_actor.set_pose(s.Pose(ob["center"]))
        c=proxy["car"]; b=scene.create_actor_builder(); add_box(b,c["size"])
        sx,sy,sz=c["size"]; mass=c["mass"]
        inertia=[mass*(sy*sy+sz*sz)/12,mass*(sx*sx+sz*sz)/12,mass*(sx*sx+sy*sy)/12]
        b.set_mass_and_inertia(mass,s.Pose(),inertia)
        actor=b.build(name="car")
        actor.set_pose(s.Pose(c["position"],_quat_wxyz(c["orientation_xyzw"])))
        dynamic=actor.find_component_by_type(s.physx.PhysxRigidDynamicComponent)
        preparation=time.perf_counter()-prep0
        dynamic.set_linear_velocity(np.asarray(velocity,dtype=np.float32))
        rollout0=time.perf_counter()
        for _ in range(nsteps):
            t=time.perf_counter(); scene.step(); step_times.append(time.perf_counter()-t)
            pose=actor.get_pose(); v=dynamic.get_linear_velocity(); w=dynamic.get_angular_velocity()
            rows.append([*pose.p,*pose.q,*v,*w])
            in_contact=False
            for pair in scene.get_contacts():
                if dynamic in pair.bodies:
                    in_contact=True
                    penetration=max(penetration,max([0.]+[-float(pt.separation) for pt in pair.points]))
            contacts+=int(in_contact)
        rollout_wall=time.perf_counter()-rollout0
        world_bodies=1+len(proxy["obstacles"])+1; collision_shapes=world_bodies
    else: raise ValueError(engine)
    return {**timing(step_times,preparation,rollout_wall),"world_body_count":world_bodies,
            "collision_shape_count":collision_shapes,
            "metrics":trajectory_metrics(rows,penetration,contacts),
            "_trajectory_xyz":[row[:3] for row in rows]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir",type=Path); ap.add_argument("--action",action="append",default=[])
    ap.add_argument("--seconds",type=float,default=5.); ap.add_argument("--friction",type=float)
    ap.add_argument("--engines",nargs="+",choices=list(ENGINES),default=list(ENGINES)); ap.add_argument("--output",type=Path)
    a=ap.parse_args()
    if a.seconds<=0 or not math.isfinite(a.seconds) or (a.friction is not None and (a.friction<0 or not math.isfinite(a.friction))): ap.error("seconds must be positive and friction nonnegative")
    try:
        from artifacts import SceneArtifacts
        g=SceneArtifacts(a.run_dir).load(); proxy=box_proxy(g.scene,g.obstacle_meshes)
        saved_run=json.loads((a.run_dir/'run.json').read_text())
        friction=float(saved_run['parameters']['friction']) if a.friction is None else a.friction
        if a.action:
            actions=parse_actions(a.action)
        else:
            saved=json.loads((a.run_dir/'impulses.json').read_text())
            actions={name:spec['delta_velocity_m_s'] for name,spec in saved.items()
                     if name in ('coast','ballistic','obstacle')}
            if set(actions)!= {'coast','ballistic','obstacle'}:
                raise ValueError('saved impulses must contain coast, ballistic, and obstacle')
    except Exception as exc: ap.error(str(exc))
    result={"schema":1,"benchmark":"canonical_box_proxy","warning":"Approximate box proxy; results do not represent production mesh collisions.",
        "scene":str(a.run_dir.resolve()),"settings":{"gravity_m_s2":GRAVITY,"timestep_s":DT,"friction":friction,"seconds":a.seconds,"actions_delta_velocity_m_s":actions},"proxy":proxy,"engines":{}}
    for name in a.engines:
        try: importlib.import_module(ENGINES[name])
        except ImportError:
            result["engines"][name]={"status":"unavailable","message":f"Missing optional dependency {ENGINES[name]!r}; install it in the benchmark environment to enable {name}."}; continue
        result["engines"][name]={"status":"available","actions":{}}
        for action,velocity in actions.items():
            try: result["engines"][name]["actions"][action]=_simulate(name,proxy,velocity,a.seconds,friction)
            except Exception as exc: result["engines"][name]["actions"][action]={"status":"error","message":str(exc)}
    bullet=result["engines"].get("pybullet",{}).get("actions",{})
    references={action:entry.get("_trajectory_xyz") for action,entry in bullet.items()}
    for name,engine_result in result["engines"].items():
        for action,entry in engine_result.get("actions",{}).items():
            trace=entry.pop("_trajectory_xyz",None)
            reference=references.get(action) if name!="pybullet" else None
            if trace is not None and reference is not None:
                n=min(len(trace),len(reference))
                distances=[math.dist(trace[i],reference[i]) for i in range(n)]
                entry["trajectory_vs_pybullet"]={"samples":n,
                    "rms_m":math.sqrt(sum(d*d for d in distances)/n),
                    "max_m":max(distances)}
    encoded=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if a.output: a.output.write_text(encoded)
    print(encoded,end="")


if __name__ == "__main__": main()
