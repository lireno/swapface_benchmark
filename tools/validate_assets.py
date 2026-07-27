#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, default=Path("assets"))
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()

    assets = args.assets_root.resolve()
    manifest_path = assets / "benchmark/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = []
    for case in manifest["cases"]:
        for key in ("ref_image", "origin_video", "face_boxes", "gan_swapped_video"):
            path = manifest_path.parent / case[key]
            if not path.is_file():
                missing.append(path.as_posix())

    model_manifest = json.loads((assets / "models/manifest.json").read_text(encoding="utf-8"))
    bad_hashes = []
    for model in model_manifest["models"]:
        path = assets / model["path"]
        if not path.is_file():
            missing.append(path.as_posix())
        elif not args.skip_sha256 and sha256(path) != model["sha256"]:
            bad_hashes.append(path.as_posix())

    if missing or bad_hashes:
        raise RuntimeError(
            f"asset validation failed: missing={len(missing)} bad_hashes={len(bad_hashes)}\n"
            + "\n".join((missing + bad_hashes)[:30])
        )
    print(
        json.dumps(
            {
                "benchmark_cases": len(manifest["cases"]),
                "model_files": len(model_manifest["models"]),
                "sha256_checked": not args.skip_sha256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
