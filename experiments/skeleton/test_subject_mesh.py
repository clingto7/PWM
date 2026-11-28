"""Saved mesh artifact and observation-to-mesh CLI seam."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = Path(__file__).with_name('subject_mesh.py')


class SubjectMeshCliTest(unittest.TestCase):
    def test_direct_mesh_input_isolates_distant_mask_component(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.asarray(Image.open(ROOT/'first_image/car_input/car_mask.png').convert('L')).copy()
            mask[700:702, 1320:1322] = 255
            mask[700:702, 1010:1012] = 255
            raw = root/'raw_mask.png'
            Image.fromarray(mask).save(raw)
            out = root/'mesh'
            result = subprocess.run([sys.executable, str(CLI), str(ROOT/'first_image/input.jpg'),
                str(raw), str(out), '--saved-mesh',
                str(ROOT/'first_image/hunyuan21_output/car_shape.glb')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            crop = json.loads((out/'subject_mesh.json').read_text())['provenance']['crop']
            self.assertEqual(crop['mask_cleanup']['removed_pixels'], 4)
            self.assertEqual(crop['mask_cleanup']['removed_components'], 1)
            self.assertGreaterEqual(crop['mask_cleanup']['retained_components'], 2)
            self.assertGreaterEqual(crop['mask_cleanup']['clean_bbox_xyxy'][2], 1012)
            self.assertLess(crop['crop_xyxy'][2]-crop['crop_xyxy'][0], 600)
            self.assertLess(Image.open(out/'masked_subject.png').getchannel('A').getbbox()[2], 600)

    def test_saved_mesh_produces_checked_artifact_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'mesh'
            command = [sys.executable, str(CLI), str(ROOT/'first_image/input.jpg'),
                       str(ROOT/'first_image/car_input/car_mask.png'), str(out),
                       '--saved-mesh', str(ROOT/'first_image/hunyuan21_output/car_shape.glb')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            artifact = json.loads((out/'subject_mesh.json').read_text())
            self.assertEqual(artifact['schema'], 1)
            self.assertEqual(artifact['validity']['valid'], True)
            self.assertEqual(artifact['units'], 'normalized_model_units')
            self.assertIn('orientation', artifact)
            self.assertIn('coordinate_frame', artifact)
            self.assertIn('original_image_sha256', artifact['input_hashes'])
            self.assertIn('selected_mask_sha256', artifact['input_hashes'])
            checked = subprocess.run([sys.executable, str(CLI), 'check', str(out)], capture_output=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_saved_mesh_records_exact_padded_subject_crop(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'mesh'
            result = subprocess.run([sys.executable, str(CLI), str(ROOT/'first_image/input.jpg'),
                str(ROOT/'first_image/car_input/car_mask.png'), str(out), '--saved-mesh',
                str(ROOT/'first_image/hunyuan21_output/car_shape.glb')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            artifact = json.loads((out/'subject_mesh.json').read_text())
            crop = artifact['provenance']['crop']
            self.assertEqual(crop['selected_mask_bbox_xyxy'], [631, 607, 1002, 911])
            self.assertLess(crop['crop_xyxy'][2]-crop['crop_xyxy'][0], 600)
            self.assertEqual(crop['crop_xyxy'][2]-crop['crop_xyxy'][0],
                             crop['crop_xyxy'][3]-crop['crop_xyxy'][1])
            rgba = Image.open(out/'masked_subject.png')
            self.assertEqual(rgba.mode, 'RGBA')
            self.assertEqual(rgba.size[0], crop['crop_xyxy'][2]-crop['crop_xyxy'][0])
            self.assertEqual(artifact['provenance']['rgba_sha256'], __import__('hashlib').sha256(
                (out/'masked_subject.png').read_bytes()).hexdigest())
            self.assertGreater(rgba.getchannel('A').getbbox()[2], 0)

    def test_header_valid_open_mesh_is_rejected(self):
        import struct
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vertices = struct.pack('<12f', 0,0,0, 1,0,0, 0,1,0, 0,0,1)
            indices = struct.pack('<12I', *(0,1,2)*4)
            document = {'asset': {'version':'2.0'}, 'buffers':[{'byteLength':96}],
                'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':48},
                               {'buffer':0,'byteOffset':48,'byteLength':48}],
                'accessors':[{'bufferView':0,'componentType':5126,'count':4,'type':'VEC3',
                              'min':[0,0,0],'max':[1,1,1]},
                             {'bufferView':1,'componentType':5125,'count':12,'type':'SCALAR'}],
                'meshes':[{'primitives':[{'attributes':{'POSITION':0},'indices':1}]}]}
            encoded = json.dumps(document).encode()
            encoded += b' ' * (-len(encoded)%4)
            body = struct.pack('<I4s',len(encoded),b'JSON')+encoded+struct.pack('<I4s',96,b'BIN\0')+vertices+indices
            bad = root/'open.glb'
            bad.write_bytes(struct.pack('<4sII',b'glTF',2,12+len(body))+body)
            out = root/'mesh'
            result = subprocess.run([sys.executable,str(CLI),str(ROOT/'first_image/input.jpg'),
                str(ROOT/'first_image/car_input/car_mask.png'),str(out),'--saved-mesh',str(bad)],
                capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('mesh',json.loads((out/'stage_status.json').read_text())['reason'].lower())

    def test_invalid_saved_mesh_preserves_failure_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root/'bad.glb'
            bad.write_bytes(b'bad mesh')
            out = root/'mesh'
            result = subprocess.run([sys.executable, str(CLI), str(ROOT/'first_image/input.jpg'),
                str(ROOT/'first_image/car_input/car_mask.png'), str(out), '--saved-mesh', str(bad)],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            status = json.loads((out/'stage_status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['stage'], 'subject_mesh')
            self.assertTrue((out/'inputs/input.jpg').is_file())
            self.assertTrue((out/'inputs/car_mask.png').is_file())


if __name__ == '__main__':
    unittest.main()
