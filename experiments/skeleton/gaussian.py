"""Render fixed geometric Gaussians for an existing first-image run directory."""
import argparse
import hashlib
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
import trimesh
from gsplat.rendering import rasterization


def disk_quat(normal):
    """Local disk normal +Z to an outward mesh normal, in gsplat WXYZ order."""
    normal=np.asarray(normal,dtype=np.float64)
    normal/=np.linalg.norm(normal)
    vector=np.cross([0,0,1],normal)
    scalar=1+normal[2]
    if scalar < 1e-8:
        return [0,1,0,0]
    q=np.r_[vector,scalar];q/=np.linalg.norm(q)
    return q[[3,0,1,2]].tolist()


def render(root):
    if not torch.cuda.is_available():
        raise RuntimeError('geometric gsplat rendering requires CUDA')
    record=json.loads((root/'run.json').read_text())
    scene=json.loads((root/'scene.json').read_text())
    rows=json.loads((root/'trajectory.json').read_text())
    with np.load(root/'heightmap.npz') as data:
        height=data['ground_z_m']; valid=data['valid'].astype(bool); surface=data['surface_z_m']
    with np.load(root/'obstacles.npz') as data:
        obstacles=data['instance_id']
    cell=float(scene['ground']['cell_m']); n=height.shape[0]
    origin=-(n-1)*cell/2
    means=[]; scales=[]; quats=[]; ids=[]
    def add(center,scale,normal,instance_id):
        means.append(center);scales.append(scale);quats.append(disk_quat(normal));ids.append(instance_id)
    # One flat disk for every raster mesh quad; omit sector boundary quads
    # exactly as the raster geometry does. Centers average its four corners.
    for j in range(n-1):
        for i in range(n-1):
            if valid[j:j+2,i:i+2].all():
                add([origin+(i+.5)*cell,(j+.5)*cell,float(height[j:j+2,i:i+2].mean())],
                    [cell*.68,cell*.68,max(cell*.025,.005)],[0,0,1],1)
    # A top and four vertical faces per obstacle cell approximate the box mesh.
    for j,i in np.argwhere(obstacles>0):
        entity=int(obstacles[j,i]);x=origin+i*cell;y=j*cell
        bottom=float(scene['ground']['flat_world_z_m']);top=float(surface[j,i]);half=(top-bottom)/2
        add([x,y,top],[cell*.68,cell*.68,max(cell*.025,.005)],[0,0,1],entity)
        for dx,dy,normal in ((1,0,[1,0,0]),(-1,0,[-1,0,0]),(0,1,[0,1,0]),(0,-1,[0,-1,0])):
            add([x+dx*cell/2,y+dy*cell/2,bottom+half],
                [max(half*.68,.01),cell*.68,max(cell*.025,.005)] if dx else [cell*.68,max(half*.68,.01),max(cell*.025,.005)],normal,entity)
    # Reproducible area-weighted surface samples in the car's local body frame.
    mesh=trimesh.load(root/'fitted_car.obj',force='mesh',process=False)
    triangles=np.asarray(mesh.triangles,dtype=np.float64)
    area=np.asarray(mesh.area_faces,dtype=np.float64)
    if not len(triangles) or area.sum()<=0:
        raise ValueError('empty fitted car mesh')
    car_count=min(30000,max(2000,int(math.ceil(float(area.sum())/(cell/4)**2))))
    rng=np.random.default_rng(42)
    selected=rng.choice(len(area),size=car_count,p=area/area.sum())
    uv=rng.random((car_count,2));fold=uv.sum(axis=1)>1;uv[fold]=1-uv[fold]
    car_points=triangles[selected,0]+uv[:,0,None]*(triangles[selected,1]-triangles[selected,0])+uv[:,1,None]*(triangles[selected,2]-triangles[selected,0])
    normals=np.asarray(mesh.face_normals)[selected]
    car_quats=np.array([disk_quat(normal) for normal in normals],dtype=np.float32)
    car_scales=np.tile([math.sqrt(float(area.sum())/car_count)*1.25]*2+[max(cell*.012,.003)],(car_count,1)).astype(np.float32)
    entity_ids=np.asarray(ids+[2]*car_count,dtype=np.int32)
    counts={str(int(entity)):int(np.count_nonzero(entity_ids==entity)) for entity in np.unique(entity_ids)}
    device='cuda';to_gpu=lambda value:torch.as_tensor(np.asarray(value,dtype=np.float32),device=device)
    static_means=to_gpu(means);static_scales=to_gpu(scales);static_quats=to_gpu(quats)
    local_scales=to_gpu(car_scales)
    alpha_threshold=.2
    # All IDs contribute to occlusion in every pass. Use the four feature
    # channels confirmed by #7; pad the final batch with zeros.
    unique_ids=sorted(counts,key=int)
    feature_batches=[]
    for offset in range(0,len(unique_ids),4):
        batch=unique_ids[offset:offset+4]
        features=np.zeros((len(entity_ids),4),dtype=np.float32)
        for channel,entity in enumerate(batch):
            features[entity_ids==int(entity),channel]=1
        feature_batches.append((batch,to_gpu(features)))
    forward=np.asarray(scene['camera']['forward'],dtype=np.float64)
    up=np.asarray(scene['camera']['up'],dtype=np.float64)
    right=np.cross(forward,up)
    view=np.eye(4,dtype=np.float32);view[:3,:3]=np.stack([right,-up,forward])
    k=np.asarray(scene['camera']['intrinsics'],dtype=np.float32)
    view_gpu=to_gpu(view)[None];k_gpu=to_gpu(k)[None]
    width,height=504,280
    depth_dir=root/'gaussian_depth';ids_dir=root/'gaussian_instance_id';mix_dir=root/'gaussian_edge_mixing'
    for directory in (depth_dir,ids_dir,mix_dir): directory.mkdir(exist_ok=False)
    frame_times=[]
    for frame in range(int(record['parameters']['seconds']*record['parameters']['video_fps'])):
        row=rows[min(len(rows)-1,frame*8)]
        torch.cuda.synchronize();started=time.perf_counter()
        rotation=Rotation.from_quat(row[3:7])
        car_world=rotation.apply(car_points)+np.asarray(row[:3])
        car_world_quats=(rotation*Rotation.from_quat(car_quats[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
        positions=torch.cat([static_means,to_gpu(car_world)])
        all_scales=torch.cat([static_scales,local_scales])
        all_quats=torch.cat([static_quats,to_gpu(car_world_quats)])
        opacity=torch.full((len(entity_ids),),.98,device=device)
        scores=[];expected_depth=None;alpha_image=None
        with torch.no_grad():
            for batch,features in feature_batches:
                image,alpha,_=rasterization(positions,all_quats,all_scales,opacity,features,view_gpu,k_gpu,width,height,render_mode='RGB+ED',packed=False,near_plane=.1,far_plane=100)
                scores.append(image[0,:,:,:len(batch)])
                if expected_depth is None:
                    expected_depth=image[0,:,:,-1]
                    alpha_image=alpha[0,:,:,0]
        scores=torch.cat(scores,dim=-1)
        best,best_idx=scores.max(dim=-1)
        top_two=torch.topk(scores,k=min(2,len(unique_ids)),dim=-1).values
        mixed=(top_two[:,:,1]>=top_two[:,:,0]*.25) & (top_two[:,:,1]>.02) if len(unique_ids)>1 else torch.zeros_like(best,dtype=torch.bool)
        visible=(alpha_image>=alpha_threshold)&(best>.02)&torch.isfinite(expected_depth)&(expected_depth>0)
        id_lookup=torch.as_tensor([int(i) for i in unique_ids],device=device,dtype=torch.int32)
        rendered_ids=torch.where(visible,id_lookup[best_idx],0).cpu().numpy().astype(np.uint16)
        millimeters=torch.where(visible,(expected_depth*1000).clamp(1,65535),0).cpu().numpy().astype(np.uint16)
        mixed=(mixed&visible).to(torch.uint8).cpu().numpy()*255
        torch.cuda.synchronize();frame_times.append(time.perf_counter()-started)
        Image.fromarray(rendered_ids).save(ids_dir/f'{frame:04d}.png')
        Image.fromarray(millimeters).save(depth_dir/f'{frame:04d}.png')
        Image.fromarray(mixed).save(mix_dir/f'{frame:04d}.png')
    produced={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for directory in (depth_dir,ids_dir,mix_dir) for path in sorted(directory.glob('*.png'))}
    record['outputs'].update(produced)
    record['gaussian']={'image_size_wh':[width,height],'camera':'scene.json original fixed DA3','sampling':{'method':'one disk per valid ground quad; five box-face disks per obstacle cell; area-weighted car triangle barycentric disks','seed':42,'ground_xy_sigma_cell_fraction':.68,'obstacle_face_sigma_fraction':.68,'car_disk_sigma_m':float(car_scales[0,0]),'opacity':.98,'no_optimization':True},'entity_counts':counts,'frame_times_s':frame_times,'raster_frame_times_s':record['raster_frame_times_s'],'alpha_threshold':alpha_threshold,'feature_method':'chunked one-hot RGB+ED, argmax; all entities occlude each pass','depth_method':'gsplat expected optical z depth (ED); zero on uncovered pixels','fallback':'none; accepted #7 confirms gsplat 1.5.3 RGB+ED and four features','edge_mixing_rule':'second one-hot score >= 25% of highest and > 0.02'}
    record['versions']['gsplat']=version('gsplat');record['versions']['torch']=torch.__version__
    record['timings_s']['gaussian_render']=sum(frame_times)
    record['timings_s']['total']+=sum(frame_times)
    record['stages']['gaussian_render']={'inputs':{name:record['outputs'][name] for name in ('scene.json','heightmap.npz','obstacles.npz','fitted_car.obj','trajectory.json')},'outputs':produced,'parameters':record['gaussian']['sampling'],'seconds':sum(frame_times)}
    (root/'run.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_dir',type=Path)
    render(parser.parse_args().run_dir.resolve())
