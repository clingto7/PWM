"""Raster depth and instance-ID renderer adapter.

The renderer seam accepts saved trajectory states, a fixed camera, and scene
body IDs. Outputs are 16-bit PNGs. Preview writers receive 8-bit bytes only.
"""
import time
import numpy as np
from PIL import Image


class RasterRenderer:
    def __init__(self, bullet, width, height, near, far, view, projection,
                 ground, body, obstacles, radius, seconds):
        self.p=bullet; self.width=width; self.height=height
        self.near=near; self.far=far; self.view=view; self.projection=projection
        self.ground=ground; self.body=body; self.obstacles=obstacles
        self.radius=radius; self.seconds=seconds

    def render(self, trajectory_rows, depth_dir, ids_dir, encoders=None):
        p=self.p; w=self.width; h=self.height; far=self.far; near=self.near
        view=self.view; projection=self.projection; ground=self.ground; body=self.body
        obstacle_bodies=self.obstacles; radius=self.radius; seconds=self.seconds
        times=[]
        first_ids=None
        for frame in range(int(seconds*30)):
            row=trajectory_rows[min(len(trajectory_rows)-1,frame*8)]
            frame_start=time.perf_counter()
            p.resetBasePositionAndOrientation(body,row[:3],row[3:7])
            _,_,_,zbuf,seg=p.getCameraImage(w,h,viewMatrix=view,projectionMatrix=projection,renderer=p.ER_TINY_RENDERER,flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
            zbuf=np.asarray(zbuf).reshape(h,w)
            metric=far*near/(far-(far-near)*zbuf)
            raw=np.asarray(seg,dtype=np.int32).reshape(h,w)
            ids=np.zeros((h,w),dtype=np.uint16)
            ids[(raw & 0xffffff) == ground]=1
            ids[(raw & 0xffffff) == body]=2
            for obstacle_body,instance_id in obstacle_bodies.items():
                ids[(raw & 0xffffff) == obstacle_body]=instance_id
            ids[raw<0]=0
            millimeters=np.where(ids>0,np.clip(metric*1000,1,65535),0).astype(np.uint16)
            times.append(time.perf_counter()-frame_start)
            Image.fromarray(millimeters).save(depth_dir/f'{frame:04d}.png')
            Image.fromarray(ids).save(ids_dir/f'{frame:04d}.png')
            if frame == 0:
                first_ids=ids.copy()
            if encoders:
                encoders[0].stdin.write(np.where(ids>0,np.clip(metric/radius*255,0,255),0).astype(np.uint8).tobytes())
                encoders[1].stdin.write(np.clip(ids*53,0,255).astype(np.uint8).tobytes())
        return first_ids,times
