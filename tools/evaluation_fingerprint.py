#!/usr/bin/env python3
"""Conservative resume fingerprint; media/model stat inventory, source content hashes."""
from pathlib import Path
import argparse
import hashlib
import json


def fingerprint(mapping, code_root, assets):
    rows = json.loads(Path(mapping).read_text())
    inputs = set()
    for row in rows:
        for key in ('generated', 'ref_image', 'ref_video', 'ref_video_face_boxes', 'ref_video_facemask'):
            if row.get(key):
                inputs.add(Path(row[key]).resolve())
    for raw in assets:
        p = Path(raw)
        if p.is_dir():
            inputs.update(q.resolve() for q in p.rglob('*') if q.is_file())
        elif p.is_file():
            inputs.add(p.resolve())
    inventory = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(inputs)]
    root = Path(code_root)
    code = [(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
            for sub in ('tools', 'scripts', 'swapface_benchmark', 'vendor')
            for p in sorted((root/sub).rglob('*')) if p.is_file() and p.suffix in ('.py', '.sh')]
    payload = {'version': 3, 'inputs': inventory, 'code': code}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping', required=True)
    parser.add_argument('--code-root', required=True)
    parser.add_argument('assets', nargs='*')
    args = parser.parse_args()
    print(fingerprint(args.mapping, args.code_root, args.assets))
