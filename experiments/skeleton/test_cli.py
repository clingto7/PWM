"""External seam checks: the checker CLI reads and judges a run directory."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

CHECK=Path(__file__).with_name('check.py')
RUN=CHECK.with_name('run.py')


class CheckCliTest(unittest.TestCase):
    def fixture(self,root):
        root.mkdir(exist_ok=True)
        (root/'depth').mkdir();(root/'instance_id').mkdir()
        (root/'depth/0000.png').write_bytes(b'depth')
        (root/'instance_id/0000.png').write_bytes(b'id')
        scene={'car':{'size_m':[1,2,1],'initial_position_m':[0,0,.5]},'ground':{'plane_z_ax_by_c':[0,0,0],'flat_world_z_m':0,'rms_m':.008,'height_m':1.69,'pitch_deg':7.6}}
        trajectory=[[0,0,.5,0,0,0,1,0,0,0,0,0,0] for _ in range(8)]
        (root/'scene.json').write_text(json.dumps(scene));(root/'trajectory.json').write_text(json.dumps(trajectory));(root/'replay_trajectory.json').write_text(json.dumps(trajectory))
        projection={'reference_runs_y_x0_x1':[[1,1,3]],'rendered_runs_y_x0_x1':[[1,1,3]],'reference_bbox_xyxy':[1,1,3,2],'rendered_bbox_xyxy':[1,1,3,2],'intersection_pixels':2,'union_pixels':2,'mask_iou':1.0,'frame':'instance_id/0000.png'}
        (root/'initial_projection.json').write_text(json.dumps(projection))
        files=['initial_projection.json','scene.json','trajectory.json','replay_trajectory.json','depth/0000.png','instance_id/0000.png']
        rec={'outputs':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},'inputs':{},'parameters':{'seconds':1/30,'video_fps':30,'radius_m':27,'angle_deg':95},'measurements':{'max_contact_penetration_m':0,'replay_trajectory_identical':True,'initial_projection_bbox_xyxy':[1,1,3,2],'initial_projection_mask_iou':1.0}}
        rec['stages']={'scene':{'inputs':{},'outputs':{'scene.json':rec['outputs']['scene.json']}}}
        (root/'run.json').write_text(json.dumps(rec))
        return rec

    def invoke(self,root):
        return subprocess.run([sys.executable,str(CHECK),str(root)],capture_output=True,text=True)

    def obstacle_fixture(self,root,render_only=False):
        rec=self.fixture(root)
        grid=np.array([[3,3,0],[0,0,0],[0,0,0]],dtype=np.uint16)
        road=np.zeros((3,3),dtype=np.uint8);road[2,2]=1
        footprint=np.zeros((3,3),dtype=np.uint8);footprint[1,1]=1
        surface=np.zeros((3,3),dtype=np.float32);surface[0,:2]=.5
        votes=np.zeros((3,3),dtype=np.int32);votes[0,:2]=3
        provenance=np.full((3,3),2,dtype=np.uint8);provenance[0,:2]=1;provenance[1,1]=3
        np.savez_compressed(root/'heightmap.npz',surface_z_m=surface,valid=np.ones((3,3),dtype=np.uint8),surface_votes=votes,surface_provenance=provenance)
        np.savez_compressed(root/'obstacles.npz',instance_id=grid,candidate=(surface>.35).astype(np.uint8),road_cells=road,car_footprint=footprint)
        Image.fromarray(np.array([[3,3,0],[1,2,0]],dtype=np.uint16)).save(root/'instance_id/0000.png')
        scene=json.loads((root/'scene.json').read_text())
        scene['obstacles']={'components':[{'instance_id':3,'cells':2,'collision_enabled':not render_only,'collision_shape_count':0 if render_only else 1}], 'detection':{'height_above_plane_m':.35,'min_pixels_per_cell':3,'min_cells_per_component':2},'render_only':render_only,'road_cells':1,'road_false_positive_cells':0,'road_false_positive_ratio':0.0,'car_footprint_obstacle_cells':0}
        (root/'scene.json').write_text(json.dumps(scene))
        rec['parameters'].update({'obstacle_height_m':.35,'obstacle_min_pixels':3,'obstacle_min_cells':2,'obstacles_render_only':render_only})
        rec['measurements'].update({'road_false_positive_ratio':0.0,'road_false_positive_cells':0,'road_cells':1,'obstacle_component_sizes_cells':[2],'obstacle_component_count':1})
        for name in ('scene.json','heightmap.npz','obstacles.npz','instance_id/0000.png'):
            rec['outputs'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
        rec['stages']['scene']['outputs']['scene.json']=rec['outputs']['scene.json']
        (root/'run.json').write_text(json.dumps(rec))
        return rec

    def test_obstacle_record_and_render_only_fallback_at_checker_seam(self):
        for render_only in (False,True):
            with self.subTest(render_only=render_only),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);self.obstacle_fixture(root,render_only)
                before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
                result=self.invoke(root)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                report=json.loads(result.stdout)
                self.assertEqual(report['metrics']['obstacle_component_sizes_cells'],[2])
                self.assertEqual(report['metrics']['obstacle_rendered_ids'],[3])
                self.assertEqual(report['metrics']['road_false_positive_ratio'],0.0)
                self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_checker_rejects_car_footprint_obstacle_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=self.obstacle_fixture(root)
            grid=np.array([[3,3,0],[0,3,0],[0,0,0]],dtype=np.uint16)
            road=np.zeros((3,3),dtype=np.uint8);road[2,2]=1
            footprint=np.zeros((3,3),dtype=np.uint8);footprint[1,1]=1
            np.savez_compressed(root/'obstacles.npz',instance_id=grid,candidate=np.array([[1,1,0],[0,1,0],[0,0,0]],dtype=np.uint8),road_cells=road,car_footprint=footprint)
            rec['outputs']['obstacles.npz']=hashlib.sha256((root/'obstacles.npz').read_bytes()).hexdigest()
            (root/'run.json').write_text(json.dumps(rec))
            result=self.invoke(root)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(json.loads(result.stdout)['checks']['car_footprint_clear'])

    def test_checker_accounts_for_obstacle_ids_across_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=self.obstacle_fixture(root)
            Image.fromarray(np.array([[1,2,0]],dtype=np.uint16)).save(root/'instance_id/0000.png')
            Image.fromarray(np.array([[3,1,2]],dtype=np.uint16)).save(root/'instance_id/0001.png')
            (root/'depth/0001.png').write_bytes(b'depth')
            rec['parameters']['seconds']=2/30
            for name in ('instance_id/0000.png','instance_id/0001.png','depth/0001.png'):
                rec['outputs'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
            (root/'run.json').write_text(json.dumps(rec))
            report=json.loads(self.invoke(root).stdout)
            self.assertTrue(report['checks']['obstacle_render_ids'])
            self.assertEqual(report['metrics']['missing_obstacle_render_ids'],[])
            Image.fromarray(np.array([[1,2,0]],dtype=np.uint16)).save(root/'instance_id/0001.png')
            rec['outputs']['instance_id/0001.png']=hashlib.sha256((root/'instance_id/0001.png').read_bytes()).hexdigest()
            (root/'run.json').write_text(json.dumps(rec))
            report=json.loads(self.invoke(root).stdout)
            self.assertFalse(report['checks']['obstacle_render_ids'])
            self.assertEqual(report['metrics']['missing_obstacle_render_ids'],[3])

    def test_passes_good_run_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result=self.invoke(root)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            report=json.loads(result.stdout)
            self.assertTrue(report['checks']['penetration'])
            self.assertTrue(report['checks']['replay'])
            self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_rejects_tampered_trajectory_even_if_metrics_claim_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            (root/'trajectory.json').write_text('[]')
            result=self.invoke(root)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(json.loads(result.stdout)['checks']['hashes'])

    def test_rejects_replay_mismatch_despite_claimed_boolean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=self.fixture(root)
            replay=json.loads((root/'replay_trajectory.json').read_text())
            replay[0][0]=1
            (root/'replay_trajectory.json').write_text(json.dumps(replay))
            rec['outputs']['replay_trajectory.json']=hashlib.sha256((root/'replay_trajectory.json').read_bytes()).hexdigest()
            (root/'run.json').write_text(json.dumps(rec))
            result=self.invoke(root)
            self.assertFalse(json.loads(result.stdout)['checks']['replay'])

    def test_rejects_motion_before_first_rendered_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=self.fixture(root)
            for name in ('trajectory.json','replay_trajectory.json'):
                rows=json.loads((root/name).read_text())
                rows[0][7]=.1
                (root/name).write_text(json.dumps(rows))
                rec['outputs'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
            (root/'run.json').write_text(json.dumps(rec))
            result=self.invoke(root)
            self.assertFalse(json.loads(result.stdout)['checks']['initial_stationary_pose'])

    def test_rejects_forged_initial_projection_iou_with_valid_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=self.fixture(root)
            projection=json.loads((root/'initial_projection.json').read_text())
            projection['mask_iou']=.9
            (root/'initial_projection.json').write_text(json.dumps(projection))
            rec['outputs']['initial_projection.json']=hashlib.sha256((root/'initial_projection.json').read_bytes()).hexdigest()
            rec['measurements']['initial_projection_mask_iou']=.9
            (root/'run.json').write_text(json.dumps(rec))
            result=self.invoke(root)
            self.assertFalse(json.loads(result.stdout)['checks']['initial_projection_record'])

    def test_rejects_projection_worse_than_published_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=self.fixture(root)
            projection=json.loads((root/'initial_projection.json').read_text())
            projection['rendered_runs_y_x0_x1']=[[2,1,3]]
            projection['rendered_bbox_xyxy']=[1,2,3,3]
            projection['intersection_pixels']=0
            projection['union_pixels']=4
            projection['mask_iou']=0.0
            (root/'initial_projection.json').write_text(json.dumps(projection))
            rec['outputs']['initial_projection.json']=hashlib.sha256((root/'initial_projection.json').read_bytes()).hexdigest()
            rec['measurements']['initial_projection_bbox_xyxy']=[1,2,3,3]
            rec['measurements']['initial_projection_mask_iou']=0.0
            (root/'run.json').write_text(json.dumps(rec))
            result=self.invoke(root)
            self.assertFalse(json.loads(result.stdout)['checks']['initial_projection_improved'])

    def test_rejects_stage_hash_mapping_inconsistent_with_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=self.fixture(root)
            rec['stages']['scene']['outputs']['scene.json']='0'*64
            (root/'run.json').write_text(json.dumps(rec))
            result=self.invoke(root)
            self.assertFalse(json.loads(result.stdout)['checks']['hashes'])

    def test_rejects_resting_error_above_measured_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=self.fixture(root)
            for name in ('trajectory.json','replay_trajectory.json'):
                rows=json.loads((root/name).read_text())
                rows[-1][2]=.511
                (root/name).write_text(json.dumps(rows))
                rec['outputs'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
            (root/'run.json').write_text(json.dumps(rec))
            result=self.invoke(root)
            self.assertFalse(json.loads(result.stdout)['checks']['resting_height'])

    def test_rejects_penetration_with_intact_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=self.fixture(root)
            rec['measurements']['max_contact_penetration_m']=.1
            (root/'run.json').write_text(json.dumps(rec))
            result=self.invoke(root)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(json.loads(result.stdout)['checks']['penetration'])


@unittest.skipUnless(os.environ.get('PWM_RUN_INTEGRATION')=='1','requires the pinned PyBullet first-image environment')
class PipelineObstacleTest(unittest.TestCase):
    def test_height_filter_and_render_only_modes_through_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports=[]
            for name,options in (('default',[]),('high',['--obstacle-height','.9']),('filtered',['--obstacle-min-cells','1000']),('fallback',['--obstacles-render-only'])):
                root=Path(tmp)/name
                run=subprocess.run([sys.executable,str(RUN),str(root),'--seconds','.034',*options],capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stderr+run.stdout)
                checked=subprocess.run([sys.executable,str(CHECK),str(root)],capture_output=True,text=True)
                report=json.loads(checked.stdout)
                self.assertTrue(all(report['checks'][key] for key in ('obstacle_parameters','obstacle_components','obstacle_heightmap','obstacle_road_ratio','car_footprint_clear','obstacle_render_ids','obstacle_collision_mode')),checked.stdout)
                self.assertTrue(all((count==0)==report['metrics']['obstacles_render_only'] for count in report['metrics']['obstacle_collision_shape_counts']))
                reports.append(report)
            self.assertGreater(len(reports[0]['metrics']['obstacle_component_sizes_cells']),0)
            self.assertEqual(reports[1]['metrics']['obstacle_component_sizes_cells'],[])
            self.assertEqual(reports[2]['metrics']['obstacle_component_sizes_cells'],[])
            self.assertEqual(reports[0]['metrics']['obstacle_component_sizes_cells'],reports[3]['metrics']['obstacle_component_sizes_cells'])
            self.assertEqual(reports[3]['metrics']['obstacle_rendered_ids'],reports[0]['metrics']['obstacle_rendered_ids'])
            self.assertTrue(reports[3]['metrics']['obstacles_render_only'])


if __name__=='__main__': unittest.main()
