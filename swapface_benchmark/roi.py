"""Explicit source-video ROI: xyxy, numeric-key order, no temporal interpolation."""
from pathlib import Path
import json
import math


def read_face_boxes(path):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"expected nonempty frame-keyed face boxes: {path}")
    keys = sorted(payload, key=int)
    if len({int(key) for key in keys}) != len(keys):
        raise ValueError(f"duplicate numeric frame keys: {path}")
    result = []
    for key in keys:
        value = payload[key]
        if not isinstance(value, list) or len(value) < 4:
            raise ValueError(f"invalid box at frame key {key}: {path}")
        box = tuple(float(v) for v in value[:4])
        if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"nonfinite or degenerate box at frame key {key}: {path}")
        result.append(box)
    return result


def scaled_box(box, source_shape, target_shape):
    sh, sw = source_shape
    th, tw = target_shape
    x1, y1, x2, y2 = (int(round(v * scale)) for v, scale in zip(box, (tw/sw, th/sh, tw/sw, th/sh)))
    x1, x2 = max(0, min(tw, x1)), max(0, min(tw, x2))
    y1, y2 = max(0, min(th, y1)), max(0, min(th, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("face box is empty after clipping to the frame")
    return x1, y1, x2, y2
