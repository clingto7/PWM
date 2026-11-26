"""Reproduce a run from its portable input bundle and the pinned code revision."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from bundle import revision, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    bundle=args.bundle.resolve()
    output=args.output.resolve()
    if output.is_relative_to(bundle):
        parser.error('output must be outside the source bundle')
    manifest = json.loads((bundle / 'bundle.json').read_text())
    sources(bundle)
    if manifest['revision'] != revision():
        parser.error(f"code revision mismatch: bundle requires {manifest['revision']}")
    options = []
    for key, value in manifest['arguments'].items():
        if isinstance(value, bool):
            if value:
                options.append('--' + key.replace('_', '-'))
        else:
            options.extend(('--' + key.replace('_', '-'), str(value)))
    subprocess.run([sys.executable, str(Path(__file__).with_name('run.py')),
                    str(args.output), '--bundle', str(args.bundle), *options], check=True)


if __name__ == '__main__':
    main()
