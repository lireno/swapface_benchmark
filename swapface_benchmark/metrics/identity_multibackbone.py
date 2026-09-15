#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from swapface_benchmark.metrics.identity_strict import (
    bbox_from_face_boxes,
    crop_face,
    detect_faces,
    expand_bbox,
    frame_count,
    load_face_boxes,
    read_frames,
    resolve_face_boxes_path,
    sample_eval_indices,
    time_aligned_index,
    video_fps,
)


DEFAULT_MODELS_ROOT = PROJECT_ROOT / "assets/models/identity"
DEFAULT_DETECTOR = DEFAULT_MODELS_ROOT / "antelope/scrfd_10g_bnkps.onnx"
DEFAULT_ID_ARC = DEFAULT_MODELS_ROOT / "arcface_w600k_r50.onnx"
DEFAULT_ID_INS = DEFAULT_MODELS_ROOT / "antelope/glintr100.onnx"
DEFAULT_ID_CUR = DEFAULT_MODELS_ROOT / "curricularface_ir101.onnx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "median": None, "count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "count": int(array.size),
    }


def video_shape(path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(path.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video dimensions: {path}")
    return height, width


class SharedFaceAligner:
    def __init__(self, detector_path: Path, device: int, det_size: int, det_thresh: float) -> None:
        import onnxruntime as ort
        from insightface.model_zoo.scrfd import SCRFD

        providers: list[Any] = ["CPUExecutionProvider"]
        if device >= 0:
            providers = [("CUDAExecutionProvider", {"device_id": device}), "CPUExecutionProvider"]
        session = ort.InferenceSession(detector_path.as_posix(), providers=providers)
        self.detector = SCRFD(model_file=detector_path.as_posix(), session=session)
        self.detector.prepare(device, input_size=(det_size, det_size), det_thresh=det_thresh)
        self.det_thresh = det_thresh
        if device >= 0 and "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError("CUDAExecutionProvider requested but SCRFD is not running on CUDA")
        print(f"[multi-id] detector providers: {session.get_providers()}", flush=True)

    def align(self, image_bgr: np.ndarray) -> np.ndarray | None:
        from insightface.utils import face_align

        if image_bgr.size == 0:
            return None
        bboxes, keypoints = detect_faces(self.detector, image_bgr, self.det_thresh, max_num=0)
        if bboxes is None or bboxes.shape[0] == 0 or keypoints is None:
            return None
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        return face_align.norm_crop(image_bgr, landmark=keypoints[int(np.argmax(areas))], image_size=112)


class OnnxFaceEncoder:
    def __init__(self, path: Path, device: int, batch_size: int) -> None:
        import onnxruntime as ort

        providers: list[Any] = ["CPUExecutionProvider"]
        if device >= 0:
            providers = [("CUDAExecutionProvider", {"device_id": device}), "CPUExecutionProvider"]
        self.session = ort.InferenceSession(path.as_posix(), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.batch_size = batch_size
        if device >= 0 and "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError(f"CUDAExecutionProvider requested but {path} is not running on CUDA")
        print(f"[multi-id] {path.name} providers: {self.session.get_providers()}", flush=True)

    @staticmethod
    def preprocess(images_bgr: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images_bgr]).astype(np.float32)
        batch = (batch - 127.5) / 127.5
        return np.transpose(batch, (0, 3, 1, 2)).copy()

    def embed(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        chunks = []
        for start in range(0, len(images_bgr), self.batch_size):
            inputs = self.preprocess(images_bgr[start : start + self.batch_size])
            outputs = self.session.run([self.output_name], {self.input_name: inputs})[0]
            chunks.append(np.asarray(outputs).reshape(inputs.shape[0], -1))
        return unit_rows(np.concatenate(chunks, axis=0))


def prepare_generated_frames(
    item: dict[str, Any], max_eval_frames: int, sample_frames: int, seed: int, frame_stride: int
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    generated_path = Path(item["generated"])
    input_video_path = Path(item["ref_video"])
    mask_path = Path(item["ref_video_facemask"])
    generated_count = frame_count(generated_path)
    input_count = frame_count(input_video_path)
    input_shape = video_shape(input_video_path)
    eval_count = generated_count
    positive_limits = [value for value in (sample_frames, max_eval_frames) if value > 0]
    sample_limit = min(positive_limits) if positive_limits else 0
    eval_indices = sample_eval_indices(
        eval_count, sample_limit, random_sampling=False, seed=seed, frame_stride=frame_stride
    )
    generated_fps = video_fps(generated_path)
    input_fps = video_fps(input_video_path)
    input_indices = [time_aligned_index(index, generated_fps, input_fps, input_count) for index in eval_indices]
    face_boxes_path = resolve_face_boxes_path(item, input_video_path, mask_path)
    face_boxes = load_face_boxes(face_boxes_path)
    if face_boxes_path is not None and (not face_boxes or max(input_indices, default=-1) >= len(face_boxes)):
        raise RuntimeError(
            f"face boxes are shorter than generated timeline: need index {max(input_indices, default=-1)}, "
            f"available={len(face_boxes)}"
        )
    from swapface_benchmark.metrics.identity_strict import calculate_bbox_from_mask
    masks = None
    if face_boxes_path is None:
        mask_fps, mask_count = video_fps(mask_path), frame_count(mask_path)
        mask_indices = [time_aligned_index(i, generated_fps, mask_fps, mask_count) for i in eval_indices]
        masks = read_frames(mask_path, mask_indices)
    frames = read_frames(generated_path, eval_indices)
    cropped = []
    for position, (frame, input_index) in enumerate(zip(frames, input_indices)):
        if masks is None:
            bbox = bbox_from_face_boxes(face_boxes, input_index, input_shape, frame.shape[:2])
        else:
            mask = cv2.resize(masks[position], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            bbox = calculate_bbox_from_mask(mask)
            if bbox is None:
                raise ValueError(f"empty ROI at frame {eval_indices[position]}")
        cropped.append(crop_face(frame, expand_bbox(bbox, frame.shape[:2])))
    ref_image = cv2.imread(item["ref_image"], cv2.IMREAD_COLOR)
    if ref_image is None:
        raise RuntimeError(f"failed to read ref image: {item['ref_image']}")
    return ref_image, cropped, eval_indices[: len(cropped)]


def evaluate_case(
    item: dict[str, Any], aligner: SharedFaceAligner, encoders: dict[str, OnnxFaceEncoder], args: argparse.Namespace
) -> dict[str, Any]:
    ref_image, frames, frame_indices = prepare_generated_frames(
        item, args.max_eval_frames, args.sample_frames, args.seed, args.frame_stride
    )
    aligned_ref = aligner.align(ref_image)
    if aligned_ref is None:
        raise RuntimeError("reference face detection failed")
    aligned_frames: list[np.ndarray] = []
    valid_indices: list[int] = []
    for frame_index, frame in zip(frame_indices, frames):
        aligned = aligner.align(frame)
        if aligned is not None:
            aligned_frames.append(aligned)
            valid_indices.append(frame_index)
    if not aligned_frames:
        raise RuntimeError("generated face detection failed for all sampled frames")

    metrics: dict[str, Any] = {}
    for name, encoder in encoders.items():
        embeddings = encoder.embed([aligned_ref, *aligned_frames])
        scores = embeddings[1:] @ embeddings[0]
        metrics[name] = float(scores.mean())
        metrics[f"{name}_scores"] = [float(value) for value in scores]
        metrics[f"{name}_variance"] = float(np.var(scores))
    return {
        "name": item["name"],
        "generated": item["generated"],
        "ref_image": item["ref_image"],
        "sampled_frame_count": len(frame_indices),
        "frame_stride": args.frame_stride,
        "valid_face_frames": len(aligned_frames),
        "valid_frame_indices": valid_indices,
        **metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--id-arc", type=Path, default=DEFAULT_ID_ARC)
    parser.add_argument("--id-ins", type=Path, default=DEFAULT_ID_INS)
    parser.add_argument("--id-cur", type=Path, default=DEFAULT_ID_CUR)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--det-thresh", type=float, default=0.5)
    parser.add_argument("--max-eval-frames", type=int, default=81)
    parser.add_argument("--sample-frames", type=int, default=81)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--metrics", default="id_arc,id_ins,id_cur", help="Comma-separated: id_arc,id_ins,id_cur")
    args = parser.parse_args()
    if args.frame_stride <= 0:
        raise ValueError("frame-stride must be positive")

    selected = {value.strip().replace("-", "_") for value in args.metrics.split(",") if value.strip()}
    available = {"id_arc": args.id_arc, "id_ins": args.id_ins, "id_cur": args.id_cur}
    unknown = selected - set(available)
    if unknown or not selected:
        raise ValueError(f"invalid identity metrics: {sorted(unknown) if unknown else 'empty selection'}")
    model_paths = {name: path for name, path in available.items() if name in selected}
    for path in [args.detector, *model_paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    mapping_payload = json.loads(args.mapping.read_text(encoding="utf-8"))
    items = mapping_payload["items"] if isinstance(mapping_payload, dict) else mapping_payload
    if args.start_index < 0 or args.end_index < 0 or (args.end_index and args.end_index < args.start_index):
        raise ValueError("invalid start/end index")
    items = items[args.start_index : args.end_index or None]
    if args.limit > 0:
        items = items[: args.limit]

    aligner = SharedFaceAligner(args.detector, args.device, args.det_size, args.det_thresh)
    encoders = {name: OnnxFaceEncoder(path, args.device, args.batch_size) for name, path in model_paths.items()}
    cases: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for index, item in enumerate(items, start=1):
        try:
            case = evaluate_case(item, aligner, encoders, args)
            cases[case["name"]] = case
            print(
                f"[{index:03d}/{len(items):03d}] {case['name']} "
                + " ".join(f"{name}={case[name]:.4f}" for name in model_paths),
                flush=True,
            )
        except Exception as error:
            failures.append({"name": item.get("name", f"case-{index}"), "error": str(error)})
            print(f"[{index:03d}/{len(items):03d}] failed: {error}", flush=True)

    sampled = sum(case["sampled_frame_count"] for case in cases.values())
    valid = sum(case["valid_face_frames"] for case in cases.values())
    metrics: dict[str, Any] = {}
    for name in model_paths:
        metrics[name] = stats([case[name] for case in cases.values()])
        metrics[f"{name}_variance"] = stats([case[f"{name}_variance"] for case in cases.values()])
    metrics["variance"] = metrics.get("id_arc_variance")
    metrics["face_detection_rate"] = float(valid / sampled) if sampled else None
    payload = {
        "protocol": {
            "reference": "DreamID-V Table 1 identity metric family",
            "alignment": "face_boxes crop, SCRFD largest face, 5-point 112x112 alignment",
            "aggregation": "frame cosine mean per video, then mean across videos",
            "variance": "population variance of frame-level ID-Arc cosine per video, then mean across videos",
            "sample_frames": args.sample_frames,
            "max_eval_frames": args.max_eval_frames,
            "frame_stride": args.frame_stride,
        },
        "models": {
            name: {"path": path.resolve().as_posix(), "sha256": sha256(path)} for name, path in model_paths.items()
        },
        "case_count": len(cases),
        "failure_count": len(failures),
        "metrics": metrics,
        "cases": cases,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output, flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
