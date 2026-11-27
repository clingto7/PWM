"""Original image to checked DA3/SAM 2.1 observation bundle.

Saved model outputs permit local replay; omit their source flags for inference with
local weights. The optional mesh is a downstream compatibility proxy for ticket 16.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from stage_contracts import check, digest, record

DEFAULT_MODELS = Path(__file__).resolve().parents[1]/'first_image/models'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def verify_saved_file(source, metadata, filename):
    expected = metadata.get('files', {}).get(filename, {}).get('sha256')
    if expected and digest(source/filename) != expected:
        raise ValueError(f'saved model output hash mismatch: {filename}')


def depth_live(image, destination, weights):
    if not weights or not (weights/'model.safetensors').is_file():
        raise ValueError('DA3 local weights required; no model download is permitted')
    import numpy as np
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(str(weights)).to('cuda').eval()
    prediction = model.inference([str(image)], infer_gs=False, process_res=504)
    arrays = {'depth': np.asarray(prediction.depth)}
    for name in ('conf', 'sky', 'intrinsics', 'extrinsics', 'processed_images'):
        value = getattr(prediction, name, None)
        if value is not None:
            arrays[name] = np.asarray(value)
    np.savez_compressed(destination/'prediction.npz', **arrays)
    return {'model': 'depth-anything/DA3NESTED-GIANT-LARGE-1.1',
            'is_metric': bool(prediction.is_metric), 'weight_sha256': digest(weights/'model.safetensors')}


def segmentation_live(image_path, destination, weights):
    if not weights or not (weights/'model.safetensors').is_file():
        raise ValueError('SAM 2.1 Large local weights required; no model download is permitted')
    import numpy as np
    import torch
    from PIL import Image
    from transformers import Sam2Model
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    rgb = np.asarray(image.resize((1024, 1024), Image.Resampling.BILINEAR), dtype=np.float32)/255
    rgb = (rgb-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
    pixels = torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].cuda()
    points = np.array([(x*width,y*height) for y in np.linspace(.04,.96,9)
                       for x in np.linspace(.04,.96,16)], dtype=np.float32)
    prompts = torch.from_numpy(points*np.array([1024/width,1024/height], dtype=np.float32)).cuda()[None,:,None]
    model = Sam2Model.from_pretrained(weights, local_files_only=True).cuda().eval()
    with torch.inference_mode():
        result = model(pixel_values=pixels, input_points=prompts, multimask_output=True)
    scores_all = result.iou_scores[0].float().cpu().numpy()
    best = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(len(points)), best]
    logits = result.pred_masks[0, torch.arange(len(points)), torch.from_numpy(best).cuda()]
    masks = (logits > 0).cpu().numpy()
    areas = masks.mean(axis=(1,2))
    eligible = np.where((scores > .8) & (areas > .0005) & (areas < .7))[0]
    selected = []
    for idx in eligible[np.argsort(scores[eligible])[::-1]]:
        mask = masks[idx]
        if all((mask & masks[prior]).sum() / max(1,(mask | masks[prior]).sum()) < .8 for prior in selected):
            selected.append(int(idx))
    np.savez_compressed(destination/'sam2_masks.npz', masks_256=masks, points_xy=points,
                        scores=scores, selected=np.array(selected, dtype=int))
    return {'model':'facebook/sam2.1-hiera-large', 'weight_sha256':digest(weights/'model.safetensors'),
            'method':'uniform 16x9 point prompts; best IoU mask; IoU deduplication'}


def select_subject(masks_path, image_path, output):
    import numpy as np
    from PIL import Image
    with np.load(masks_path) as data:
        masks, points, scores = data['masks_256'], data['points_xy'], data['scores']
        indices = data['selected'] if 'selected' in data else np.arange(len(masks))
    width, height = Image.open(image_path).size
    candidates = []
    for idx in indices:
        idx = int(idx)
        mask = masks[idx].astype(bool)
        ys,xs = np.nonzero(mask)
        if not len(xs):
            continue
        area = len(xs)/mask.size
        cx,cy = float(xs.mean()/mask.shape[1]),float(ys.mean()/mask.shape[0])
        # Foreground subject evidence: central, lower-half, coherent instance,
        # and neither a tiny fragment nor the entire background. No RGB/category.
        center = max(0., 1.-abs(cx-.5)*2.)
        lower = max(0., 1.-abs(cy-.72)/.72)
        extent = min(1., area/.025) * max(0., 1.-area/.55)
        quality = float(np.clip(scores[idx],0,1))
        score = .30*center + .25*lower + .25*extent + .20*quality
        candidates.append({'index':idx,'score':round(score,6),'model_iou':quality,
                           'area_fraction':area,'centroid_xy_fraction':[cx,cy],
                           'bbox_xyxy_256':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
                           'prompt_xy':points[idx].tolist()})
    candidates.sort(key=lambda value:value['score'], reverse=True)
    if not candidates or candidates[0]['score'] < .48 or (len(candidates)>1 and
            candidates[0]['score']-candidates[1]['score'] < .015):
        write(output/'selection.json', {'candidates':candidates,'selected_index':None,
                                        'reason':'no confident dominant subject'})
        raise ValueError('no confident dominant subject; inspect selection.json')
    index = candidates[0]['index']
    selected = Image.fromarray(np.uint8(masks[index])*255).resize((width,height),Image.Resampling.NEAREST)
    selected.save(output/'inputs/car_mask.png')
    write(output/'selection.json', {'candidates':candidates,'selected_index':index,
                                    'score_margin':candidates[0]['score']-(candidates[1]['score'] if len(candidates)>1 else 0),
                                    'method':'centrality + lower-image position + instance extent + model IoU'})


def build(args):
    import numpy as np
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('output directory is not empty')
    (out/'inputs').mkdir(parents=True)
    image = args.image.resolve()
    shutil.copyfile(image,out/'inputs/input.jpg')
    write(out/'stage_status.json', {'stage':'depth','status':'running'})
    start=time.perf_counter()
    if args.depth_source:
        source=args.depth_source.resolve()
        metadata=json.loads((source/'run.json').read_text())
        if metadata.get('input_sha256') and metadata['input_sha256'] != digest(image):
            raise ValueError('saved DA3 source belongs to a different image')
        verify_saved_file(source,metadata,'prediction.npz')
        shutil.copyfile(source/'prediction.npz',out/'inputs/prediction.npz')
        implementation='saved '+metadata.get('model','DA3')
    else:
        metadata=depth_live(image,out/'inputs',args.da3_weights)
        implementation=metadata['model']
    with np.load(out/'inputs/prediction.npz') as prediction:
        depth=prediction['depth']
        intrinsic=prediction['intrinsics'] if 'intrinsics' in prediction else None
        has_camera=(intrinsic is not None and intrinsic.shape==(1,3,3) and
                    np.isfinite(intrinsic).all() and intrinsic[0,0,0]>0 and intrinsic[0,1,1]>0)
        valid_depth=depth.ndim==3 and depth.shape[0]==1 and np.isfinite(depth).all() and (depth>0).any()
    capability={'scale':'metric_m' if metadata.get('is_metric') is True else 'relative',
                'camera_intrinsics':bool(has_camera),'positive_finite_depth':bool(valid_depth),
                'model':metadata.get('model'),'weight_sha256':metadata.get('weight_sha256')}
    write(out/'depth_capability.json',capability)
    if capability['scale']!='metric_m' or not has_camera or not valid_depth:
        raise ValueError('scene requires metric depth and camera intrinsics; see depth_capability.json')
    record(out,'depth',['inputs/input.jpg'],['inputs/prediction.npz','depth_capability.json'],
           implementation,'1',capability,time.perf_counter()-start)
    write(out/'stage_status.json', {'stage':'segmentation','status':'running'})
    start=time.perf_counter()
    if args.segmentation_source:
        source=args.segmentation_source.resolve()
        metadata=json.loads((source/'run.json').read_text())
        if metadata.get('input_sha256') and metadata['input_sha256'] != digest(image):
            raise ValueError('saved SAM source belongs to a different image')
        verify_saved_file(source,metadata,'sam2_masks.npz')
        shutil.copyfile(source/'sam2_masks.npz',out/'sam2_masks.npz')
        implementation='saved '+metadata.get('model','SAM 2.1')
    else:
        metadata=segmentation_live(image,out,args.sam_weights)
        implementation=metadata['model']
    select_subject(out/'sam2_masks.npz',image,out)
    record(out,'segmentation',['inputs/input.jpg'],['sam2_masks.npz','selection.json','inputs/car_mask.png'],
           implementation,'1',{'model':metadata.get('model'),'weight_sha256':metadata.get('weight_sha256'),
                                'method':metadata.get('method')},time.perf_counter()-start)
    if args.mesh:
        shutil.copyfile(args.mesh,out/'inputs/car_shape.glb')
    else:
        raise ValueError('ticket 16 downstream proxy requires --mesh; automatic mesh is ticket 17')
    write(out/'bundle.json', {'schema':1,'inputs':{'image':'inputs/input.jpg','da3':'inputs/prediction.npz',
        'sam_car':'inputs/car_mask.png','hunyuan_mesh':'inputs/car_shape.glb'},
        'source_sha256':{'image':digest(out/'inputs/input.jpg'),'da3':digest(out/'inputs/prediction.npz'),
        'sam_car':digest(out/'inputs/car_mask.png'),'hunyuan_mesh':digest(out/'inputs/car_shape.glb')}})
    record(out,'observation',['inputs/input.jpg'],['inputs/prediction.npz','inputs/car_mask.png',
        'inputs/car_shape.glb','selection.json','depth_capability.json','bundle.json'],
        'DA3 + SAM 2.1 Large + saved mesh proxy','1',
        {'depth_stage':'stages/depth.json','segmentation_stage':'stages/segmentation.json'},
        check(out,'depth')['seconds']+check(out,'segmentation')['seconds'])
    write(out/'stage_status.json', {'stage':'observation','status':'complete'})
    print(json.dumps({'bundle':str(out),'selected_index':json.loads((out/'selection.json').read_text())['selected_index']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path,nargs='?')
    parser.add_argument('output',type=Path)
    parser.add_argument('--depth-source',type=Path)
    parser.add_argument('--segmentation-source',type=Path)
    parser.add_argument('--da3-weights',type=Path,default=DEFAULT_MODELS/'da3nested')
    parser.add_argument('--sam-weights',type=Path,default=DEFAULT_MODELS/'sam21large')
    parser.add_argument('--mesh',type=Path)
    args=parser.parse_args()
    if args.image=='check':
        raise ValueError('use check ROOT STAGE')
    try:
        build(args)
    except Exception as exc:
        if args.output.is_dir():
            status=json.loads((args.output/'stage_status.json').read_text())
            write(args.output/'stage_status.json', {'stage':status['stage'],'status':'failed',
                'error_type':type(exc).__name__,'reason':str(exc)})
        raise


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='check':
        try:
            print(json.dumps(check(Path(sys.argv[2]),sys.argv[3])))
        except Exception as exc:
            sys.exit(str(exc))
    else:
        main()
