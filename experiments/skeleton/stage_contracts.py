"""Versioned, portable file contracts between first-image pipeline stages."""
import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = 1


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _paths(root, names):
    result = {}
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f'missing or outside stage artifact: {name}')
        result[name] = digest(path)
    return result


def record(root, stage, inputs, outputs, implementation, version, parameters, seconds):
    root = Path(root).resolve()
    if not outputs:
        raise ValueError('stage requires an output')
    if seconds < 0:
        raise ValueError('negative stage duration')
    manifest = {'schema': SCHEMA, 'stage': stage,
                'implementation': implementation, 'version': version,
                'parameters': parameters, 'inputs': _paths(root, inputs),
                'outputs': _paths(root, outputs), 'seconds': seconds}
    target = root/'stages'/f'{stage}.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return manifest


def check(root, stage):
    root = Path(root).resolve()
    path = root/'stages'/f'{stage}.json'
    data = json.loads(path.read_text())
    if data.get('schema') != SCHEMA or data.get('stage') != stage:
        raise ValueError('unsupported stage contract')
    for field in ('inputs', 'outputs'):
        if not isinstance(data.get(field), dict) or (field == 'outputs' and not data[field]):
            raise ValueError(f'invalid {field}')
        actual = _paths(root, data[field])
        if actual != data[field]:
            raise ValueError(f'{stage} {field} hash mismatch')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('record')
    create.add_argument('root', type=Path)
    create.add_argument('stage')
    create.add_argument('--input', action='append', default=[])
    create.add_argument('--output', action='append', default=[])
    create.add_argument('--implementation', required=True)
    create.add_argument('--version', required=True)
    create.add_argument('--seconds', type=float, required=True)
    create.add_argument('--parameters', default='{}')
    verify = sub.add_parser('check')
    verify.add_argument('root', type=Path)
    verify.add_argument('stage')
    args = parser.parse_args()
    try:
        if args.command == 'record':
            value = record(args.root, args.stage, args.input, args.output,
                           args.implementation, args.version,
                           json.loads(args.parameters), args.seconds)
        else:
            value = check(args.root, args.stage)
        print(json.dumps(value, sort_keys=True))
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
