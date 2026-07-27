#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def ensure_link(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        if destination.is_symlink() and destination.resolve() == source:
            return
        raise FileExistsError(destination)
    destination.symlink_to(source)


def resize_index(index: int, source_count: int, target_count: int) -> int:
    if source_count <= 1 or target_count <= 1:
        return 0
    return min(int(index * (source_count - 1) / (target_count - 1)), source_count - 1)


def build_mask_video(video_path: Path, boxes_path: Path, output_path: Path) -> None:
    capture = cv2.VideoCapture(video_path.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"failed to open {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    capture.release()
    if min(width, height, frame_count) <= 0:
        raise RuntimeError(f"invalid video metadata: {video_path}")

    payload = json.loads(boxes_path.read_text(encoding="utf-8"))
    boxes = [payload[key] for key in sorted(payload, key=int)]
    if not boxes:
        raise ValueError(f"empty face boxes: {boxes_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to create {output_path}")
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    try:
        for frame_index in range(frame_count):
            frame.fill(0)
            box = boxes[resize_index(frame_index, len(boxes), frame_count)]
            x1, y1, x2, y2 = (int(round(float(value))) for value in box[:4])
            x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width, x2))))
            y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height, y2))))
            if x2 > x1 and y2 > y1:
                frame[y1:y2, x1:x2] = 255
            writer.write(frame)
    finally:
        writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    source_dir = args.output_root / "source"
    target_dir = args.output_root / "target/sim"
    mask_cache = args.output_root / "masks"
    for index, item in enumerate(mapping, start=1):
        video_id = item["facebench_video_id"]
        ensure_link(Path(item["ref_video"]), source_dir / f"{video_id}.mp4")
        ensure_link(Path(item["ref_image"]), source_dir / f"{video_id}_ref_sim.png")
        ensure_link(Path(item["generated"]), target_dir / f"{video_id}_swapped.mp4")
        mask_path = mask_cache / f"{video_id}_mask.mp4"
        if not mask_path.is_file():
            build_mask_video(
                Path(item["ref_video"]),
                Path(item["ref_video_face_boxes"]),
                mask_path,
            )
        ensure_link(mask_path, source_dir / f"{video_id}_mask.mp4")
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
