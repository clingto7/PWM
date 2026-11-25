"""Read-only integrity and independent physical checks for an Issue #8 run."""
import argparse
import hashlib
import json
import math
from pathlib import Path


# Spec #6 gates remain unset until the first A800 measurement.
THRESHOLDS = {'penetration_m': None, 'resting_height_error_m': None, 'final_speed_m_s': None, 'energy_gain_j_kg': None, 'ground_fit_rms_m': None}

def check(root):
    record=json.loads((root/'run.json').read_text())
    scene=json.loads((root/'scene.json').read_text())
    trajectory=json.loads((root/'trajectory.json').read_text())
    replay=json.loads((root/'replay_trajectory.json').read_text())
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
    m=record['measurements']; params=record['parameters']; box=scene['car']; ground=scene['ground']
    if not trajectory or not replay:
        return {'passed': False, 'checks': {'hashes': False, 'trajectory_present': False}, 'integrity': integrity}
    final=trajectory[-1]
    surface=ground['flat_world_z_m']
    resting_error=abs(final[2]-box['size_m'][2]/2-surface)
    final_speed=math.sqrt(sum(v*v for v in final[7:10]))
    # Independent check: kinetic + gravitational energy after impulse.
    energies=[.5*sum(v*v for v in row[7:10])+9.81*row[2] for row in trajectory]
    max_gain=max(0,max(v-u for u,v in zip(energies,energies[1:])))
    metrics={'max_penetration_m':m['max_contact_penetration_m'],'final_resting_height_error_m':resting_error,'final_speed_m_s':final_speed,'max_step_specific_energy_gain_j_kg':max_gain,'replay_trajectory_identical':trajectory==replay,'ground_fit_rms_m':ground['rms_m'],'camera_height_m':ground['height_m'],'camera_pitch_deg':ground['pitch_deg'],'sector_radius_m':params['radius_m'],'sector_angle_deg':params['angle_deg']}
    def gate(value, limit):
        return '未设' if limit is None else value <= limit
    limits=THRESHOLDS.copy()
    checks={'hashes':bool(integrity) and all(integrity.values()),
            'penetration':gate(metrics['max_penetration_m'],limits['penetration_m']),
            'resting_height':gate(resting_error,limits['resting_height_error_m']),
            'final_speed':gate(final_speed,limits['final_speed_m_s']),
            'no_energy_gain':gate(max_gain,limits['energy_gain_j_kg']),
            'ground_fit':gate(ground['rms_m'],limits['ground_fit_rms_m']),
            'replay':trajectory==replay,
            'frame_count':len(list((root/'depth').glob('*.png')))==int(params['seconds']*params['video_fps']) and len(list((root/'instance_id').glob('*.png')))==int(params['seconds']*params['video_fps'])}
    return {'passed':all(value is True for value in checks.values()),'checks':checks,'metrics':metrics,'limits':limits,'integrity':integrity}



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    result=check(parser.parse_args().run_dir.resolve())
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if result['passed'] else 1)
