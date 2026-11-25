"""Read-only integrity and independent physical checks for an Issue #8 run."""
import argparse
import hashlib
import json
import math
from pathlib import Path


# First A800 run (6075606): penetration .00546 m, rest .00391 m,
# speed 3.74e-6 m/s, energy gain .000155 J/kg, fit RMS .00719 m.
# Round to the next simple bound above each observation; these are measured
# prototype gates, not claims of model accuracy or statistical tolerances.
BASELINE_INITIAL_MASK_IOU = 0.3749  # published 6075606 first-frame surrogate
THRESHOLDS = {'penetration_m': 0.01, 'resting_height_error_m': 0.01, 'final_speed_m_s': 0.001, 'energy_gain_j_kg': 0.001, 'ground_fit_rms_m': 0.01}

def check(root):
    record=json.loads((root/'run.json').read_text())
    scene=json.loads((root/'scene.json').read_text())
    trajectory=json.loads((root/'trajectory.json').read_text())
    replay=json.loads((root/'replay_trajectory.json').read_text())
    projection=json.loads((root/'initial_projection.json').read_text())
    integrity={}
    for name,expected in record['outputs'].items():
        path=(root/name).resolve()
        integrity[name]=path.is_relative_to(root) and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected
    for name,entry in record['inputs'].items():
        path=Path(entry['path'])
        integrity['input:'+name]=path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    for stage in record.get('stages',{}).values():
        for name,expected in stage['outputs'].items():
            integrity['stage_output:'+name]=record['outputs'].get(name)==expected
        for name,expected in stage['inputs'].items():
            source=record['inputs'].get(name,{}).get('sha256',record['outputs'].get(name))
            integrity['stage_input:'+name]=source==expected
    def pixels(runs):
        return {(y,x) for y,x0,x1 in runs for x in range(x0,x1)}
    reference=pixels(projection['reference_runs_y_x0_x1'])
    rendered=pixels(projection['rendered_runs_y_x0_x1'])
    def bbox(points):
        if not points:
            return None
        ys=[p[0] for p in points]; xs=[p[1] for p in points]
        return [min(xs),min(ys),max(xs)+1,max(ys)+1]
    union=len(reference|rendered)
    projection_iou=len(reference&rendered)/union if union else 0.0
    m=record['measurements']; params=record['parameters']; box=scene['car']; ground=scene['ground']
    if not trajectory or not replay:
        return {'passed': False, 'checks': {'hashes': False, 'trajectory_present': False}, 'integrity': integrity}
    final=trajectory[-1]
    surface=ground['flat_world_z_m']
    resting_error=abs(final[2]-box['size_m'][2]/2-surface)
    final_speed=math.sqrt(sum(v*v for v in final[7:10]))
    # Independent check: kinetic + gravitational energy after impulse.
    # Row zero is stationary before the impulse; exclude the intended energy jump.
    energies=[.5*sum(v*v for v in row[7:10])+9.81*row[2] for row in trajectory[1:]]
    max_gain=max(0,max(v-u for u,v in zip(energies,energies[1:])))
    metrics={'max_penetration_m':m['max_contact_penetration_m'],'final_resting_height_error_m':resting_error,'final_speed_m_s':final_speed,'max_step_specific_energy_gain_j_kg':max_gain,'replay_trajectory_identical':trajectory==replay,'ground_fit_rms_m':ground['rms_m'],'camera_height_m':ground['height_m'],'camera_pitch_deg':ground['pitch_deg'],'sector_radius_m':params['radius_m'],'sector_angle_deg':params['angle_deg'],'initial_projection_bbox_xyxy':bbox(rendered),'initial_projection_mask_iou':projection_iou}
    def gate(value, limit):
        return '未设' if limit is None else value <= limit
    limits=THRESHOLDS.copy()
    checks={'hashes':bool(integrity) and all(integrity.values()),
            'initial_stationary_pose':all(abs(a-b)<1e-9 for a,b in zip(trajectory[0][:3],box['initial_position_m'])) and all(abs(v)<1e-12 for v in trajectory[0][7:13]),
            'penetration':gate(metrics['max_penetration_m'],limits['penetration_m']),
            'resting_height':gate(resting_error,limits['resting_height_error_m']),
            'final_speed':gate(final_speed,limits['final_speed_m_s']),
            'no_energy_gain':gate(max_gain,limits['energy_gain_j_kg']),
            'ground_fit':gate(ground['rms_m'],limits['ground_fit_rms_m']),
            'replay':trajectory==replay,
            'initial_projection_improved': projection_iou > BASELINE_INITIAL_MASK_IOU,
            'initial_projection_record': projection['frame']=='instance_id/0000.png' and bbox(reference)==projection['reference_bbox_xyxy'] and bbox(rendered)==projection['rendered_bbox_xyxy']==m['initial_projection_bbox_xyxy'] and len(reference&rendered)==projection['intersection_pixels'] and union==projection['union_pixels'] and abs(projection_iou-projection['mask_iou'])<1e-12 and abs(projection_iou-m['initial_projection_mask_iou'])<1e-12,
            'frame_count':len(list((root/'depth').glob('*.png')))==int(params['seconds']*params['video_fps']) and len(list((root/'instance_id').glob('*.png')))==int(params['seconds']*params['video_fps'])}
    if 'hunyuan_mesh' in record['inputs']:
        fit=box['mesh_fit']
        size=fit['size_m']
        checks['mesh_record']=all(name in record['outputs'] for name in ('fitted_car.obj','car_collision_hull.obj')) and fit==m['mesh_fit'] and all(math.isfinite(v) and v>0 for v in size) and math.isfinite(fit['scale']) and fit['scale']>0 and box['collision_shape']=='convex hull of fitted Hunyuan mesh'
        checks['mesh_ground_contact']=abs(fit['position_m'][2]-size[2]/2-surface)<.01
        checks['size_reasonableness_record']=fit['size_check']['reasonable']==all(.7<=v<=1.3 for v in fit['size_check']['fit_to_reference_ratio'])
        from PIL import Image
        image=Image.open(root/'instance_id/0000.png')
        checks['rendered_silhouette_record']={(y,x) for y in range(image.height) for x in range(image.width) if image.getpixel((x,y))==2}==rendered
        metrics['mesh_size_m']=size
        metrics['mesh_fit_scale']=fit['scale']
        metrics['mesh_fit_yaw_rad']=fit['yaw_rad']
        metrics['mesh_size_reasonable']=fit['size_check']['reasonable']
    return {'passed':all(value is True for value in checks.values()),'checks':checks,'metrics':metrics,'limits':{**limits,'baseline_initial_mask_iou':BASELINE_INITIAL_MASK_IOU},'integrity':integrity}



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    result=check(parser.parse_args().run_dir.resolve())
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if result['passed'] else 1)
