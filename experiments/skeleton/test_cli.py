"""External seam checks: the checker CLI reads and judges a run directory."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECK=Path(__file__).with_name('check.py')


class CheckCliTest(unittest.TestCase):
    def fixture(self,root):
        root.mkdir(exist_ok=True)
        (root/'depth').mkdir();(root/'instance_id').mkdir()
        (root/'depth/0000.png').write_bytes(b'depth')
        (root/'instance_id/0000.png').write_bytes(b'id')
        scene={'car':{'size_m':[1,2,1]},'ground':{'plane_z_ax_by_c':[0,0,0],'flat_world_z_m':0,'rms_m':.008,'height_m':1.69,'pitch_deg':7.6}}
        trajectory=[[0,0,.5,0,0,0,1,0,0,0,0,0,0] for _ in range(8)]
        (root/'scene.json').write_text(json.dumps(scene));(root/'trajectory.json').write_text(json.dumps(trajectory));(root/'replay_trajectory.json').write_text(json.dumps(trajectory))
        files=['scene.json','trajectory.json','replay_trajectory.json','depth/0000.png','instance_id/0000.png']
        rec={'outputs':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},'inputs':{},'parameters':{'seconds':1/30,'video_fps':30,'radius_m':27,'angle_deg':95},'measurements':{'max_contact_penetration_m':0,'replay_trajectory_identical':True}}
        rec['stages']={'scene':{'inputs':{},'outputs':{'scene.json':rec['outputs']['scene.json']}}}
        (root/'run.json').write_text(json.dumps(rec))
        return rec

    def invoke(self,root):
        return subprocess.run([sys.executable,str(CHECK),str(root)],capture_output=True,text=True)

    def test_passes_good_run_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result=self.invoke(root)
            self.assertNotEqual(result.returncode,0,result.stderr+result.stdout)
            report=json.loads(result.stdout)
            self.assertEqual(report['checks']['penetration'],'未设')
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

    def test_rejects_stage_hash_mapping_inconsistent_with_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=self.fixture(root)
            rec['stages']['scene']['outputs']['scene.json']='0'*64
            (root/'run.json').write_text(json.dumps(rec))
            result=self.invoke(root)
            self.assertFalse(json.loads(result.stdout)['checks']['hashes'])

    def test_rejects_penetration_with_intact_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=self.fixture(root)
            rec['measurements']['max_contact_penetration_m']=.1
            (root/'run.json').write_text(json.dumps(rec))
            result=self.invoke(root)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(json.loads(result.stdout)['checks']['penetration'],'未设')


if __name__=='__main__': unittest.main()
