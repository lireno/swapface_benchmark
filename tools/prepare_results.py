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
    aliases = [case_id]
    if "-long-" in case_id:
        aliases.append(case_id.replace("-long-", "-", 1))
    candidates = [parent / f"{alias}.mp4" for parent in (results_dir, results_dir / "videos") for alias in aliases]
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        matches = sorted({path for alias in aliases for path in results_dir.glob(f"**/{alias}*.mp4")})
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one result for {case_id}, found {len(matches)}")
    return matches[0].resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--origin-dir", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--ref-dir", type=Path)
    parser.add_argument("--no-reference", action="store_true", help="skip reference images for attribute/quality-only evaluation")
    args = parser.parse_args()

    payload = load_json(args.manifest)
    cases = payload["cases"]
    if args.limit > 0:
        cases = cases[: args.limit]
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs in manifest")
    mapping: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        face_boxes = (
            next((path for path in (
                args.mask_dir / f"{case_id}.json",
                args.mask_dir / f"{case_id}.mp4",
                args.mask_dir / f"{case_id}_mask.mp4",
            ) if path.is_file()), None)
            if args.mask_dir else resolve_asset(args.manifest, case["face_boxes"])
        )
        if face_boxes is None:
            raise FileNotFoundError(f"mask/face boxes not found for {case_id} in {args.mask_dir}")
        origin_video = args.origin_dir / f"{case_id}.mp4" if args.origin_dir else resolve_asset(args.manifest, case["origin_video"])
        ref_image = None if args.no_reference else (args.ref_dir / f"{case_id}.jpg" if args.ref_dir else resolve_asset(args.manifest, case["ref_image"]))
        if not origin_video.is_file():
            raise FileNotFoundError(origin_video)
        if ref_image is not None and not ref_image.is_file():
            png = ref_image.with_suffix(".png")
            if not png.is_file():
                raise FileNotFoundError(ref_image)
            ref_image = png
        row = {
            **case,
            "name": case_id,
            "facebench_video_id": f"{index:05d}",
            "ref_image": ref_image.resolve().as_posix() if ref_image is not None else None,
            "ref_video": origin_video.resolve().as_posix(),
            "ref_video_face_boxes": face_boxes.resolve().as_posix(),
            "ref_video_facemask": face_boxes.resolve().as_posix(),
            "generated": find_generated(args.results_dir, case_id).as_posix(),
        }
        mapping.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mapping": args.output.as_posix(), "cases": len(mapping)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
