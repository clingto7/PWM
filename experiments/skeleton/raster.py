"""Raster depth and instance-ID renderer adapter.

The renderer seam accepts saved trajectory states, a fixed camera, and scene
body IDs. Outputs are 16-bit PNGs. Preview writers receive 8-bit bytes only.
"""
import time
import numpy as np
from PIL import Image


class RasterRenderer:
    def __init__(self, frame_source, width, height, radius, seconds):
        self.frame_source=frame_source
        self.width=width;self.height=height
        self.radius=radius;self.seconds=seconds

    def render(self, trajectory_rows, depth_dir, ids_dir, encoders=None):
        w=self.width;h=self.height;radius=self.radius;seconds=self.seconds
        times=[]
        first_ids=None
        for frame in range(int(seconds*30)):
            row=trajectory_rows[min(len(trajectory_rows)-1,frame*8)]
            frame_start=time.perf_counter()
            metric,ids=self.frame_source.frame(row)
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
