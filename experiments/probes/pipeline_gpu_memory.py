"""Run a PWM command and sample its GPU memory by process and pipeline stage.

Usage: pipeline_gpu_memory.py OUTPUT_ROOT COMMAND [ARG ...]
The command must create OUTPUT_ROOT. Measurements are runtime diagnostics only.
"""

import argparse
import json
from pathlib import Path
import subprocess
import time


def stage(root):
    for name, path in (('scene_physics', root/'automatic/run/stage_status.json'),
                       ('observation', root/'automatic/observation/stage_status.json')):
        if path.is_file():
            try:
                return name + ':' + json.loads(path.read_text())['stage']
            except (OSError, ValueError, KeyError):
                pass
    return 'startup'


def process_tree(pid):
    pending = [pid]
    found = set()
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        try:
            children = Path(f'/proc/{current}/task/{current}/children').read_text().split()
            pending.extend(map(int, children))
        except OSError:
            pass
    return found


def used_by_pid(pid):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    pids = process_tree(pid)
    used = 0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) >= 2 and int(parts[0]) in pids:
            used += int(parts[1])
    return used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error('command required')
    result_path = args.output.parent/(args.output.name + '-gpu-memory.json')
    process = subprocess.Popen(args.command)
    peaks = {}
    samples = 0
    started = time.perf_counter()
    while process.poll() is None:
        name = stage(args.output)
        try:
            used = used_by_pid(process.pid)
        except (OSError, ValueError, subprocess.CalledProcessError):
            used = 0
        peaks[name] = max(peaks.get(name, 0), used)
        samples += 1
        time.sleep(.2)
    result = {'command_pid': process.pid, 'exit_code': process.returncode,
              'sample_interval_s': .2, 'samples': samples,
              'elapsed_s': time.perf_counter()-started,
              'peak_process_gpu_mib_by_stage': peaks,
              'peak_process_gpu_mib': max(peaks.values(), default=0)}
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
