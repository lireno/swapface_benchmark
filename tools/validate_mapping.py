#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


def video_info(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(path.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video metadata: {path}")
    return {"path": path.resolve().as_posix(), "frames": frames, "fps": fps, "duration": frames / fps, "width": width, "height": height}


def mask_coverage(path: Path, origin: dict[str, Any]) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        return video_info(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid = [value for _, value in sorted(payload.items(), key=lambda item: int(item[0])) if isinstance(value, list) and len(value) >= 4]
    if not valid:
        raise RuntimeError(f"empty face boxes: {path}")
    return {"path": path.resolve().as_posix(), "type": "face_boxes", "frames": len(valid), "fps": origin["fps"], "duration": len(valid) / origin["fps"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0, help="validate first N generated frames; 0 validates all")
    args = parser.parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    cases, errors = [], []
    for item in mapping:
        case_id = item.get("case_id", item.get("name"))
        report: dict[str, Any] = {"case_id": case_id}
        try:
            generated = video_info(Path(item["generated"]))
            evaluated_frames = min(generated["frames"], args.max_frames) if args.max_frames > 0 else generated["frames"]
            generated["evaluated_frames"] = evaluated_frames
            generated["evaluated_duration"] = evaluated_frames / generated["fps"]
            report["generated"] = generated
        except Exception as error:
            errors.append({"case_id": case_id, "dependency": "generated", "error_code": "INVALID_GENERATED", "error": str(error)})
            cases.append(report)
            continue
        try:
            origin = video_info(Path(item["ref_video"]))
            report["origin"] = origin
            tolerance = 1.0 / origin["fps"]
            if origin["duration"] + tolerance < generated["evaluated_duration"]:
                errors.append({"case_id": case_id, "dependency": "origin", "error_code": "ORIGIN_TOO_SHORT", "required_duration": generated["evaluated_duration"], "available_duration": origin["duration"]})
        except Exception as error:
            errors.append({"case_id": case_id, "dependency": "origin", "error_code": "INVALID_ORIGIN", "error": str(error)})
            cases.append(report)
            continue
        try:
            mask = mask_coverage(Path(item["ref_video_facemask"]), origin)
            report["mask"] = mask
            tolerance = 1.0 / mask["fps"]
            if mask["duration"] + tolerance < generated["evaluated_duration"]:
                errors.append({"case_id": case_id, "dependency": "mask", "error_code": "MASK_TOO_SHORT", "required_duration": generated["evaluated_duration"], "available_duration": mask["duration"]})
        except Exception as error:
            errors.append({"case_id": case_id, "dependency": "mask", "error_code": "INVALID_MASK", "error": str(error)})
        cases.append(report)
    payload = {"case_count": len(cases), "error_count": len(errors), "max_frames": args.max_frames, "cases": cases, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.errors.write_text("\n".join(json.dumps(error, ensure_ascii=False) for error in errors) + ("\n" if errors else ""), encoding="utf-8")
    print(json.dumps({"case_count": len(cases), "error_count": len(errors)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
