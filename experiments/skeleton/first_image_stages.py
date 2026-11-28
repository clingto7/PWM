"""Callable first-image physics and raster stages over saved scene artifacts."""
import json
import math
import subprocess
import time

import numpy as np

from artifacts import SceneArtifacts
from physics import BulletRasterSession
from raster import RasterRenderer
from PIL import Image


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def projection_record(out, car, target, first_ids):
    h = first_ids.shape[0]
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
    evidence={'reference_runs_y_x0_x1':reference_runs,'rendered_runs_y_x0_x1':rendered_runs,
              'reference_bbox_xyxy':target,'rendered_bbox_xyxy':actual_bbox,
              'intersection_pixels':overlap,'union_pixels':union,
              'mask_iou':overlap/union if union else 0,'frame':'instance_id/0000.png'}
    write_json(out/'initial_projection.json',evidence)
    return evidence,actual_bbox


def encode_saved_pngs(out, radius, frames, width=504, height=280):
    def encoder(path):
        return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
            '-f','rawvideo','-pix_fmt','gray','-s',f'{width}x{height}','-r','30',
            '-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(path)],
            stdin=subprocess.PIPE)
    depth_encoder=encoder(out/'depth.mp4')
    id_encoder=encoder(out/'instance_id.mp4')
    try:
        for frame in range(frames):
            name=f'{frame:04d}.png'
            ids=np.asarray(Image.open(out/'instance_id'/name))
            millimeters=np.asarray(Image.open(out/'depth'/name))
            depth_encoder.stdin.write(np.where(ids>0,
                np.clip(millimeters/1000/radius*255,0,255),0).astype(np.uint8).tobytes())
            id_encoder.stdin.write(np.clip(ids*53,0,255).astype(np.uint8).tobytes())
    finally:
        depth_encoder.stdin.close();id_encoder.stdin.close()
        if depth_encoder.wait()!=0 or id_encoder.wait()!=0:
            raise RuntimeError('ffmpeg failed')


def render_saved_scene_gpu(out, args, scene, car, target, rows):
    from gpu_raster import render_saved
    started=time.perf_counter()
    params={'seconds':args.seconds,'video_fps':30,'radius_m':args.radius}
    details=render_saved(out,out,params)
    frames=int(args.seconds*30)
    encode_saved_pngs(out,args.radius,frames)
    first_ids=np.asarray(Image.open(out/'instance_id/0000.png'))
    evidence,actual_bbox=projection_record(out,car,target,first_ids)
    return {'raster_frame_times':details['frame_times_s'],
            'projection_evidence':evidence,'actual_bbox':actual_bbox,
            'render_time':time.perf_counter()-started}



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
        projection_evidence,actual_bbox=projection_record(out,car,target,first_ids)
        render_time=time.perf_counter()-render_start
        return {'raster_frame_times': raster_frame_times,
                'projection_evidence': projection_evidence,
                'actual_bbox': actual_bbox,
                'render_time': time.perf_counter()-render_start}
