"""Read-only integrity check for a capability probe run directory."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    record = json.loads((root / "run.json").read_text())
    checks = {}
    for name, expected in record["outputs"].items():
        path = (root / name).resolve()
        checks[name] = path.is_relative_to(root) and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    result = {"integrity": checks, "probe_passed": bool(record["passed"]), "passed": bool(checks) and all(checks.values()) and bool(record["passed"])}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
