"""Callable first-image physics and raster stages over saved scene artifacts."""
import json
import math
import subprocess
import time

import numpy as np

from artifacts import SceneArtifacts
from physics import BulletRasterSession
from raster import RasterRenderer


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')



def render_saved_scene(out, args, scene, car, target, rows):
    k = np.asarray(scene['camera']['intrinsics'])
    camera_forward = scene['camera']['forward']
    camera_up = scene['camera']['up']
    h, w = 280, 504
    render_start = time.perf_counter()
    near,far=.1,max(100,args.radius*4)
    fx,fy,cx0,cy0=float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2])
    projection=[2*fx/w,0,0,0,0,2*fy/h,0,0,1-2*cx0/w,2*cy0/h-1,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]
    with BulletRasterSession(SceneArtifacts(out),args,w,h,near,far,projection) as frame_source:
        depth_dir=out/'depth'; ids_dir=out/'instance_id'; depth_dir.mkdir();ids_dir.mkdir()
        depth_video=out/'depth.mp4'; id_video=out/'instance_id.mp4'
        def encoder(path):
            return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','gray','-s',f'{w}x{h}','-r','30','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
        enc_depth,enc_id=encoder(depth_video),encoder(id_video)
        renderer=RasterRenderer(frame_source,w,h,args.radius,args.seconds)
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
