"""Public seam: generated run directory, verified without importing probe internals."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CHECK = Path(__file__).with_name("check_record.py")


class RecordTest(unittest.TestCase):
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
