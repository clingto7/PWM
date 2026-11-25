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
THRESHOLDS = {'penetration_m': 0.01, 'resting_height_error_m': 0.01, 'final_speed_m_s': 0.001, 'energy_gain_j_kg': 0.001, 'ground_fit_rms_m': 0.01,
              'final_angular_speed_rad_s':0.001,'coast_distance_error_m':0.02,'flight_time_error_s':0.02,'range_error_m':0.05,'parabola_vertical_residual_m':0.01,'parabola_horizontal_residual_m':1e-6}

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
        stages=record.get('stages',{})
        required={'simulation_replay':('fitted_car.obj','car_collision_hull.obj'),
                  'proxy_render_video':('fitted_car.obj',)}
        checks['mesh_stage_provenance']=all(
            stages.get(stage,{}).get('inputs',{}).get(name)==record['outputs'].get(name)
            for stage,names in required.items() for name in names)
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
    if 'obstacles' in scene:
        stages=record.get('stages',{})
        checks['obstacle_stage_provenance']=all(
            stages.get(stage,{}).get('inputs',{}).get('obstacles.npz')==record['outputs'].get('obstacles.npz')
            for stage in ('simulation_replay','proxy_render_video'))
        import numpy as np
        from PIL import Image
        obstacles=scene['obstacles']
        with np.load(root/'obstacles.npz') as data:
            grid=data['instance_id']
            road=data['road_cells'].astype(bool)
            footprint=data['car_footprint'].astype(bool)
            candidate=data['candidate'].astype(bool)
        with np.load(root/'heightmap.npz') as data:
            surface=data['surface_z_m']
            valid=data['valid'].astype(bool)
            surface_votes=data['surface_votes']
            surface_provenance=data['surface_provenance']
        components=obstacles['components']
        expected_ids={c['instance_id'] for c in components}
        actual_ids=set(int(v) for v in np.unique(grid) if v)
        sizes=[int(np.count_nonzero(grid==c['instance_id'])) for c in components]
        connected=True
        for component in components:
            cells=set(map(tuple,np.argwhere(grid==component['instance_id'])))
            if not cells:
                connected=False;continue
            seen={next(iter(cells))}; pending=list(seen)
            while pending:
                y,x=pending.pop()
                for neighbor in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if neighbor in cells and neighbor not in seen:
                        seen.add(neighbor);pending.append(neighbor)
            connected &= seen==cells
        road_count=int(road.sum())
        false_count=int(np.count_nonzero(road & (grid>0)))
        ratio=false_count/road_count if road_count else None
        image_ids=set()
        for frame in sorted((root/'instance_id').glob('*.png')):
            image_ids.update(int(v) for v in np.unique(np.asarray(Image.open(frame))) if v)
        missing_ids=sorted(expected_ids-image_ids)
        detection=obstacles['detection']
        checks['obstacle_parameters']=detection['height_above_plane_m']==params['obstacle_height_m'] and detection['min_pixels_per_cell']==params['obstacle_min_pixels'] and detection['min_cells_per_component']==params['obstacle_min_cells'] and obstacles['render_only']==params['obstacles_render_only']
        checks['obstacle_components']=connected and actual_ids==expected_ids and [c['cells'] for c in components]==sizes and sizes==m['obstacle_component_sizes_cells'] and len(components)==m['obstacle_component_count'] and all(c['cells']>=params['obstacle_min_cells'] for c in components)
        checks['obstacle_heightmap']=bool(np.array_equal(candidate,valid & ~footprint & (surface-ground['flat_world_z_m']>params['obstacle_height_m']) & (surface_votes>=params['obstacle_min_pixels'])) and np.all(surface_provenance[footprint]==3))
        checks['obstacle_road_ratio']=road_count>0 and road_count==obstacles['road_cells']==m['road_cells'] and false_count==obstacles['road_false_positive_cells']==m['road_false_positive_cells'] and ratio==obstacles['road_false_positive_ratio']==m['road_false_positive_ratio']
        checks['car_footprint_clear']=not np.any(footprint & (grid>0)) and obstacles['car_footprint_obstacle_cells']==0
        checks['obstacle_render_ids']={1,2}.issubset(image_ids) and image_ids.issubset({1,2}|expected_ids) and not missing_ids
        checks['obstacle_collision_mode']=all(c['collision_enabled']==(not obstacles['render_only']) and (c['collision_shape_count']>0)==(not obstacles['render_only']) for c in components)
        metrics.update({'road_false_positive_ratio':ratio,'road_false_positive_cells':false_count,'road_cells':road_count,'obstacle_component_sizes_cells':sizes,'obstacle_rendered_ids':sorted(image_ids & expected_ids),'missing_obstacle_render_ids':missing_ids,'obstacle_collision_shape_counts':[c['collision_shape_count'] for c in components],'obstacles_render_only':obstacles['render_only']})
    cases_record=record.get('impulse_cases',{})
    schema=record.get('schema',1)
    if schema>=2:
        checks['three_case_record']=set(cases_record)=={'coast','ballistic','obstacle'} and 'impulses.json' in record['outputs'] and (root/'impulses.json').is_file() and all(
        all(key in case and case[key] in record['outputs'] and (root/case[key]).is_file() for key in ('trajectory','replay'))
        for case in cases_record.values())
    if schema>=2 and checks['three_case_record']:
        cases={}
        impulses=json.loads((root/'impulses.json').read_text())
        for name in ('coast','ballistic','obstacle'):
            case=record['impulse_cases'][name]
            rows=json.loads((root/case['trajectory']).read_text())
            again=json.loads((root/case['replay']).read_text())
            impulse=impulses[name]
            delta=impulse['delta_velocity_m_s']
            speed=math.sqrt(sum(v*v for v in delta))
            inertia=case.get('inertia_diagonal_kg_m2')
            def specific_energy(row):
                qx,qy,qz,qw=row[3:7]
                rotation=[[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                          [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                          [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]]
                local_omega=[sum(rotation[j][i]*row[10+j] for j in range(3)) for i in range(3)]
                rotational=sum(inertia[i]*local_omega[i]**2 for i in range(3))/(2*params['derived_mass_kg']) if inertia else 0
                return sum(v*v for v in row[7:10])/2+rotational+9.81*row[2]
            energy=[specific_energy(row) for row in rows[1:]]
            gain=max([0]+[b-a for a,b in zip(energy,energy[1:])])
            last=rows[-1]
            end_linear_speed=math.sqrt(sum(v*v for v in last[7:10]))
            end_angular_speed=math.sqrt(sum(v*v for v in last[10:13]))
            max_angular_speed=max(math.sqrt(sum(v*v for v in row[10:13])) for row in rows)
            penetration=case['max_contact_penetration_m']
            case_checks={'impulse_record':case['impulse']==impulse and impulse['application_point']=='center_of_mass' and impulse['application_point_world_m']==box['initial_position_m'] and math.isclose(impulse['magnitude_n_s'],params['derived_mass_kg']*speed,rel_tol=1e-9) and all(math.isclose(a,b/speed,abs_tol=1e-9) for a,b in zip(impulse['direction'],delta)),
                         'friction_record':scene.get('contact_friction')=={'ground_lateral_friction':params['friction'],'car_lateral_friction':1.0,'effective_ground_car_friction':params.get('effective_ground_car_friction'),'obstacle_lateral_friction':0.0} and params.get('effective_ground_car_friction')==params['friction'] and params.get('car_lateral_friction')==1.0,
                         'sleep_record':scene.get('physics_sleep')=={'car_activation_state':'enable_sleeping','activation_state_value':1} and params.get('car_sleep_activation_state')=='enable_sleeping',
                         'initial_stationary':rows[0][:3]==box['initial_position_m'] and all(abs(v)<1e-12 for v in rows[0][7:13]),
                         'penetration':penetration<=limits['penetration_m'],
                         'final_rest':end_linear_speed<=limits['final_speed_m_s'] and abs(last[2]-box['size_m'][2]/2-ground['flat_world_z_m'])<=limits['resting_height_error_m'],
                         'final_angular_rest':gate(end_angular_speed,limits['final_angular_speed_rad_s']),
                         'inertia_record':inertia is not None and len(inertia)==3 and all(v>0 for v in inertia),
                         'no_energy_gain':gain<=limits['energy_gain_j_kg'] and inertia is not None,
                         'replay':rows==again}
            case_metrics={'max_penetration_m':penetration,'final_speed_m_s':end_linear_speed,'final_angular_speed_rad_s':end_angular_speed,'max_angular_speed_rad_s':max_angular_speed,'max_step_specific_energy_gain_j_kg':gain}
            if name=='coast':
                effective_mu=params.get('effective_ground_car_friction',params['friction']**2)
                expected=speed**2/(2*effective_mu*9.81) if effective_mu else math.inf
                actual=math.hypot(last[0]-rows[0][0],last[1]-rows[0][1])
                case_metrics.update({'coast_distance_m':actual,'expected_distance_m':expected,'distance_error_m':actual-expected})
                case_checks['analytical_distance']=gate(abs(actual-expected),limits['coast_distance_error_m'])
            elif name=='ballistic':
                base=rows[0][2]
                airborne=[i for i,row in enumerate(rows) if row[2]>base+0.02]
                landing=next((i for i in range(airborne[-1]+1,len(rows)) if rows[i][2]<=base+0.02),None) if airborne else None
                expected_time=2*delta[2]/9.81
                actual_time=landing/240 if landing is not None else math.inf
                expected_range=math.hypot(*delta[:2])*expected_time
                actual_range=math.hypot(rows[landing][0]-rows[0][0],rows[landing][1]-rows[0][1]) if landing is not None else math.inf
                case_metrics.update({'flight_time_s':actual_time,'expected_flight_time_s':expected_time,'flight_time_error_s':actual_time-expected_time,'range_m':actual_range,'expected_range_m':expected_range,'range_error_m':actual_range-expected_range})
                residuals=[(abs(row[2]-(base+delta[2]*i/240-9.81*(i/240)**2/2)),abs(math.hypot(row[0]-rows[0][0],row[1]-rows[0][1])-math.hypot(*delta[:2])*i/240)) for i,row in enumerate(rows) if 0<i<(landing if landing is not None else len(rows))]
                case_metrics['max_prelanding_vertical_residual_m']=max((r[0] for r in residuals),default=math.inf)
                case_metrics['max_prelanding_horizontal_residual_m']=max((r[1] for r in residuals),default=math.inf)
                analytic_gates=[gate(abs(actual_time-expected_time),limits['flight_time_error_s']),gate(abs(actual_range-expected_range),limits['range_error_m']),gate(case_metrics['max_prelanding_vertical_residual_m'],limits['parabola_vertical_residual_m']),gate(case_metrics['max_prelanding_horizontal_residual_m'],limits['parabola_horizontal_residual_m'])]
                case_checks['analytical_flight']='未设' if '未设' in analytic_gates else all(analytic_gates)
                forward=scene['camera']['forward'];up=scene['camera']['up']
                right=[forward[1]*up[2]-forward[2]*up[1],forward[2]*up[0]-forward[0]*up[2],forward[0]*up[1]-forward[1]*up[0]]
                k=scene['camera']['intrinsics'];width_px=504;height_px=280
                def visible(row):
                    for dx in (-box['size_m'][0]/2,box['size_m'][0]/2):
                        for dy in (-box['size_m'][1]/2,box['size_m'][1]/2):
                            for dz in (-box['size_m'][2]/2,box['size_m'][2]/2):
                                qx,qy,qz,qw=row[3:7]
                                rotation=[[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                                          [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                                          [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]]
                                point=[row[i]+sum(rotation[i][j]*v for j,v in enumerate((dx,dy,dz))) for i in range(3)]
                                dot=lambda axis:sum(a*b for a,b in zip(axis,point))
                                depth=dot(forward)
                                if depth<=0:return False
                                u=k[0][0]*dot(right)/depth+k[0][2]
                                v=-k[1][1]*dot(up)/depth+k[1][2]
                                if not (0<=u<width_px and 0<=v<height_px):return False
                    return True
                case_checks['in_camera']=all(visible(row) for row in rows)
                case_checks['camera_record']=case_checks['in_camera']==m.get('ballistic_in_camera')
            else:
                applicable=not params['obstacles_render_only'] and impulse['target_obstacle_id'] is not None
                if applicable:
                    case_checks['solid_target']=impulse['target_obstacle_id'] in {c['instance_id'] for c in scene['obstacles']['components'] if c['collision_enabled']}
                    case_checks['obstacle_impact']=case.get('obstacle_contact_samples_by_id',{}).get(str(impulse['target_obstacle_id']),0)>0
                else:
                    case_checks['obstacle_impact']='不适用'
                    case_metrics['impact_applicability']='render-only or no detected solid obstacle'
                    if not params['obstacles_render_only'] and params['obstacle_height_m']==.35 and params['obstacle_min_cells']==2:
                        case_checks['solid_target']=False
            cases[name]={'checks':case_checks,'metrics':case_metrics,'passed':all(value is True or (isinstance(value,str) and value=='不适用') for value in case_checks.values())}
        checks['impulse_cases']=all(case['passed'] for case in cases.values())
        metrics['impulse_cases']=cases
    return {'passed':all(value is True for value in checks.values()),'checks':checks,'metrics':metrics,'limits':{**limits,'baseline_initial_mask_iou':BASELINE_INITIAL_MASK_IOU},'integrity':integrity}



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    result=check(parser.parse_args().run_dir.resolve())
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if result['passed'] else 1)
