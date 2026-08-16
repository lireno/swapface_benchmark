#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def ensure_link(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        if destination.is_symlink() and destination.resolve() == source:
            return
        raise FileExistsError(destination)
    destination.symlink_to(source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    source_dir = args.output_root / "source"
    target_dir = args.output_root / "target/sim"
    for index, item in enumerate(mapping, start=1):
        video_id = item["facebench_video_id"]
        ensure_link(Path(item["ref_video"]), source_dir / f"{video_id}.mp4")
        ensure_link(Path(item["ref_image"]), source_dir / f"{video_id}_ref_sim.png")
        ensure_link(Path(item["generated"]), target_dir / f"{video_id}_swapped.mp4")
        ensure_link(Path(item["ref_video_face_boxes"]), source_dir / f"{video_id}_boxes.json")
        if index % 20 == 0:
            print(f"[facebench-layout] {index}/{len(mapping)}", flush=True)

    print(
        json.dumps(
            {
                "cases": len(mapping),
                "source_dir": source_dir.resolve().as_posix(),
                "target_dir": (args.output_root / "target").resolve().as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
