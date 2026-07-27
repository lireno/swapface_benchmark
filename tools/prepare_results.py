#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_asset(manifest_path: Path, relative_path: str) -> Path:
    path = (manifest_path.parent / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def find_generated(results_dir: Path, case_id: str) -> Path:
    candidates = [
        results_dir / f"{case_id}.mp4",
        results_dir / "videos" / f"{case_id}.mp4",
    ]
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        matches = sorted(results_dir.glob(f"**/{case_id}*.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one result for {case_id}, found {len(matches)}")
    return matches[0].resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    payload = load_json(args.manifest)
    cases = payload["cases"]
    if args.limit > 0:
        cases = cases[: args.limit]
    mapping: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        face_boxes = resolve_asset(args.manifest, case["face_boxes"])
        row = {
            **case,
            "name": case_id,
            "facebench_video_id": f"{index:05d}",
            "ref_image": resolve_asset(args.manifest, case["ref_image"]).as_posix(),
            "ref_video": resolve_asset(args.manifest, case["origin_video"]).as_posix(),
            "ref_video_face_boxes": face_boxes.as_posix(),
            "ref_video_facemask": face_boxes.as_posix(),
            "ground_truth": resolve_asset(args.manifest, case["gan_swapped_video"]).as_posix(),
            "generated": find_generated(args.results_dir, case_id).as_posix(),
        }
        mapping.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mapping": args.output.as_posix(), "cases": len(mapping)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
