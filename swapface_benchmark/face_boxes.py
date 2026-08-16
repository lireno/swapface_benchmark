from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TypeAlias


BBox: TypeAlias = tuple[float, float, float, float]


def load_ordered_face_boxes(path: str | Path) -> list[BBox]:
    """Load valid boxes ordered by their numeric JSON keys."""
    boxes_path = Path(path)
    payload = json.loads(boxes_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"face boxes must be a JSON object: {boxes_path}")

    try:
        ordered_items = sorted(payload.items(), key=lambda item: int(item[0]))
    except (TypeError, ValueError) as error:
        raise ValueError(f"face-box keys must be integers: {boxes_path}") from error

    boxes: list[BBox] = []
    for _, value in ordered_items:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            continue
        try:
            coordinates = tuple(float(component) for component in value[:4])
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(component) for component in coordinates):
            continue
        x1, y1, x2, y2 = coordinates
        if x1 == x2 or y1 == y2:
            continue
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))

    if not boxes:
        raise ValueError(f"no valid face boxes: {boxes_path}")
    return boxes


def frame_to_sequence_index(frame_index: int, frame_count: int, sequence_count: int) -> int:
    """Map a frame index to an item with the same relative timeline position."""
    if frame_count <= 1 or sequence_count <= 1:
        return 0
    clamped_index = min(max(int(frame_index), 0), frame_count - 1)
    return min(
        int(clamped_index * (sequence_count - 1) / (frame_count - 1)),
        sequence_count - 1,
    )


def face_box_for_frame(boxes: list[BBox], frame_index: int, frame_count: int) -> BBox:
    if not boxes:
        raise ValueError("face boxes are empty")
    return boxes[frame_to_sequence_index(frame_index, frame_count, len(boxes))]


def scale_bbox(
    bbox: BBox,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> BBox:
    """Scale an (x1, y1, x2, y2) box between image coordinate systems."""
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError(f"invalid image shapes: source={source_shape}, target={target_shape}")

    scale_x = target_width / source_width
    scale_y = target_height / source_height
    x1, y1, x2, y2 = bbox
    return x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
