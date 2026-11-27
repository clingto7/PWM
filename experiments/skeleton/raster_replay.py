"""Render saved scene and trajectory with the fixed raster camera."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from artifacts import SceneArtifacts
from physics import BulletRasterSession
from raster import RasterRenderer


def render_saved(root, output, params=None, yaw_deg=0):
    root=Path(root).resolve(); output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'replay output is not empty: {output}')
    output.mkdir(parents=True,exist_ok=True)
    scene=SceneArtifacts(root).load()
    rows=json.loads((root/'trajectory.json').read_text())
    if params is None:
        params=json.loads((root/'run.json').read_text())['parameters']
    args=SimpleNamespace(radius=params['radius_m'],friction=params['friction'],
        obstacles_render_only=params['obstacles_render_only'])
    camera=scene.scene['camera'];k=camera['intrinsics']
    width,height=504,280
    near,far=.1,max(100,args.radius*4)
    fx,fy,cx,cy=k[0][0],k[1][1],k[0][2],k[1][2]
    projection=[2*fx/width,0,0,0,0,2*fy/height,0,0,
                1-2*cx/width,2*cy/height-1,(far+near)/(near-far),-1,
                0,0,2*far*near/(near-far),0]
    with BulletRasterSession(scene,args,width,height,near,far,projection) as frame_source:
        if yaw_deg:
            import math
            import numpy as np
            camera_forward=np.asarray(camera['forward'],dtype=float)
            camera_up=np.asarray(camera['up'],dtype=float)
            axis=camera_up/np.linalg.norm(camera_up)
            angle=math.radians(yaw_deg)
            direction=camera_forward*math.cos(angle)+np.cross(axis,camera_forward)*math.sin(angle)+axis*np.dot(axis,camera_forward)*(1-math.cos(angle))
            frame_source.view=frame_source.p.computeViewMatrix([0,0,0],direction.tolist(),camera_up.tolist())
        depth=output/'depth';ids=output/'instance_id';depth.mkdir();ids.mkdir()
        renderer=RasterRenderer(frame_source,width,height,args.radius,params['seconds'])
        _,times=renderer.render(rows,depth,ids)
        return times


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    render_saved(args.run_dir,args.output)
