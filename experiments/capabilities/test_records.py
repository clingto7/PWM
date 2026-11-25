"""Public seam: generated run directory, verified without importing probe internals."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CHECK = Path(__file__).with_name("check_record.py")


class RecordTest(unittest.TestCase):
    def test_physics_record_rejects_resting_error_and_penetration(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "trajectory.json").write_text("[]")
            import hashlib
            record = {"outputs": {"trajectory.json": hashlib.sha256(b"[]").hexdigest()}, "passed": True,
                      "resting_height_error_m": 0.02, "max_ground_penetration_m": 0.0,
                      "resting_height_tolerance_m": 0.01, "ground_penetration_tolerance_m": 0.01}
            command = [sys.executable, str(CHECK), str(out)]
            (out / "run.json").write_text(json.dumps(record))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            record["resting_height_error_m"] = 0.0
            record["max_ground_penetration_m"] = 0.02
            (out / "run.json").write_text(json.dumps(record))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            record["max_ground_penetration_m"] = 0.005
            (out / "run.json").write_text(json.dumps(record))
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_valid_record_and_tampered_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "trajectory.json").write_text("[]")
            import hashlib
            digest = hashlib.sha256(b"[]").hexdigest()
            (out / "run.json").write_text(json.dumps({"command": ["python", "physics.py", tmp], "python": "3.10", "seconds": 0.1, "outputs": {"trajectory.json": digest}, "passed": True}))
            cmd = [sys.executable, str(CHECK), str(out)]
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            (out / "trajectory.json").write_text("[0]")
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
