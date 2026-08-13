#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "assets/models/identity/antelope"


def detect_faces(det_model: Any, image_bgr: np.ndarray, det_thresh: float, max_num: int) -> tuple[np.ndarray, Any]:
    try:
        return det_model.detect(image_bgr, threshold=det_thresh, max_num=max_num, metric="default")
    except TypeError as error:
        if "threshold" not in str(error):
            raise
    try:
        return det_model.detect(image_bgr, thresh=det_thresh, max_num=max_num, metric="default")
    except TypeError as error:
        if "thresh" not in str(error):
            raise
    return det_model.detect(image_bgr, max_num=max_num, metric="default")


def unit(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-8)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8))


def mean_std(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "count": 0}
    arr = np.asarray(values, dtype=np.float32)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "count": int(arr.size)}


def resize_frame_index(frame_idx: int, src_count: int, target_count: int) -> int:
    if src_count <= 1 or target_count <= 1:
        return 0
    if src_count >= target_count:
        index = int(frame_idx * (src_count - 1) / (target_count - 1))
        return min(max(index, 0), src_count - 1)
    if frame_idx < src_count:
        return frame_idx
    return src_count - 1


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(path.as_posix())
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if count <= 0:
        raise RuntimeError(f"could not read frame count: {path}")
    return count


def video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(path.as_posix())
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"could not read FPS: {path}")
    return fps


def time_aligned_index(frame_idx: int, target_fps: float, source_fps: float, source_count: int) -> int:
    index = int(round(frame_idx / target_fps * source_fps))
    if index >= source_count:
        raise RuntimeError(
            f"source is shorter than generated timeline: need frame {index}, available frames={source_count}"
        )
    return index


def read_frames(path: Path, indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path.as_posix())
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    cap.release()
    return frames


def sample_eval_indices(eval_frame_count: int, max_frames: int, random_sampling: bool, seed: int) -> list[int]:
    if max_frames <= 0 or eval_frame_count <= max_frames:
        return list(range(eval_frame_count))
    if random_sampling:
        np.random.seed(seed)
        return sorted(int(x) for x in np.random.choice(eval_frame_count, max_frames, replace=False))
    return list(range(max_frames))


def calculate_bbox_from_mask(mask_frame: np.ndarray) -> tuple[int, int, int, int] | None:
    if len(mask_frame.shape) == 3:
        mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    else:
        mask = mask_frame
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x, y, x + w, y + h


def load_face_boxes(path: Path | None) -> dict[int, tuple[float, float, float, float]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    boxes: dict[int, tuple[float, float, float, float]] = {}
    for index, (_, value) in enumerate(sorted(payload.items(), key=lambda item: int(item[0]))):
        if not isinstance(value, list) or len(value) < 4:
            continue
        boxes[index] = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    return boxes


def resolve_face_boxes_path(item: dict[str, Any], input_video_path: Path, mask_path: Path) -> Path | None:
    candidates: list[Path] = []
    for key in ("ref_video_face_boxes", "face_boxes"):
        if item.get(key):
            candidates.append(Path(item[key]))
    if mask_path.suffix.lower() == ".json":
        candidates.append(mask_path)
    for key in ("source_mask_path", "ref_video_facemask"):
        if item.get(key):
            candidates.append(Path(item[key]).parent / "face_boxes.json")
    candidates.append(input_video_path.parent / "face_boxes.json")

    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = candidate.as_posix()
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate.exists():
            return candidate
    return None


def scale_box_to_frame(
    box: tuple[float, float, float, float],
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    sx = dst_w / max(float(src_w), 1.0)
    sy = dst_h / max(float(src_h), 1.0)
    x_min, y_min, x_max, y_max = box
    return (
        int(round(x_min * sx)),
        int(round(y_min * sy)),
        int(round(x_max * sx)),
        int(round(y_max * sy)),
    )


def bbox_from_face_boxes(
    boxes: dict[int, tuple[float, float, float, float]],
    frame_idx: int,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    if frame_idx not in boxes:
        frame_idx = min(boxes.keys(), key=lambda key: abs(key - frame_idx))
    return scale_box_to_frame(boxes[frame_idx], src_shape, dst_shape)


def expand_bbox(
    bbox: tuple[int, int, int, int] | None,
    frame_shape: tuple[int, int],
    expand_ratio: tuple[float, float, float, float] = (0.5, 0.5, 0.75, 0.25),
) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    x_min, y_min, x_max, y_max = bbox
    width = abs(x_max - x_min)
    height = abs(y_max - y_min)
    x_min -= expand_ratio[0] * width
    x_max += expand_ratio[1] * width
    y_min -= expand_ratio[2] * height
    y_max += expand_ratio[3] * height
    return (
        int(max(x_min, 0)),
        int(max(y_min, 0)),
        int(min(frame_shape[1], x_max)),
        int(min(frame_shape[0], y_max)),
    )


def crop_face(frame: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    if bbox is None:
        return frame
    x_min, y_min, x_max, y_max = bbox
    if x_max <= x_min or y_max <= y_min:
        return frame
    return frame[y_min:y_max, x_min:x_max]


class StrictInsightFace:
    def __init__(self, models_dir: Path, ctx_id: int, det_size: int, det_thresh: float) -> None:
        import onnxruntime as ort
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        from insightface.model_zoo.scrfd import SCRFD

        det_path = models_dir / "scrfd_10g_bnkps.onnx"
        rec_path = models_dir / "glintr100.onnx"
        if not det_path.exists() or not rec_path.exists():
            raise FileNotFoundError(f"missing InsightFace models in {models_dir}")
        providers = ["CPUExecutionProvider"]
        if ctx_id >= 0:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        det_session = ort.InferenceSession(det_path.as_posix(), providers=providers)
        rec_session = ort.InferenceSession(rec_path.as_posix(), providers=providers)
        self.det_model = SCRFD(model_file=det_path.as_posix(), session=det_session)
        self.rec_model = ArcFaceONNX(model_file=rec_path.as_posix(), session=rec_session)
        self.det_thresh = det_thresh
        self.det_model.prepare(ctx_id, input_size=(det_size, det_size), det_thresh=det_thresh)
        self.rec_model.prepare(ctx_id)
        print(f"[idsim] det providers: {self.det_model.session.get_providers()}", flush=True)
        print(f"[idsim] rec providers: {self.rec_model.session.get_providers()}", flush=True)
        if ctx_id >= 0 and (
            "CUDAExecutionProvider" not in self.det_model.session.get_providers()
            or "CUDAExecutionProvider" not in self.rec_model.session.get_providers()
        ):
            raise RuntimeError("CUDAExecutionProvider requested but InsightFace models are running without CUDA")

    def recognize(self, image_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        if hasattr(self.rec_model, "get"):
            return np.asarray(self.rec_model.get(image_bgr, SimpleNamespace(kps=kps)))
        if hasattr(self.rec_model, "get_feat"):
            from insightface.utils import face_align

            aligned = face_align.norm_crop(image_bgr, landmark=kps)
            return np.asarray(self.rec_model.get_feat(aligned)).reshape(-1)
        raise AttributeError(f"unsupported recognizer API: {type(self.rec_model).__name__}")

    def embed(self, image_bgr: np.ndarray) -> np.ndarray | None:
        if image_bgr.size == 0:
            return None
        bboxes, kpss = detect_faces(self.det_model, image_bgr, self.det_thresh, max_num=0)
        if bboxes is None or bboxes.shape[0] == 0 or kpss is None:
            return None
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        best = int(np.argmax(areas))
        embedding = self.recognize(image_bgr, kpss[best])
        return unit(np.asarray(embedding))


def embed_frames(model: StrictInsightFace, frames: list[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    embeddings: list[np.ndarray] = []
    valid_positions: list[int] = []
    for pos, frame in enumerate(frames):
        emb = model.embed(frame)
        if emb is None:
            continue
        embeddings.append(emb)
        valid_positions.append(pos)
    return embeddings, valid_positions


def average_sim(ref_emb: np.ndarray, embeddings: list[np.ndarray]) -> tuple[float | None, list[float]]:
    scores = [cosine(ref_emb, emb) for emb in embeddings]
    if not scores:
        return None, []
    return float(np.mean(scores)), scores


def evaluate_case(
    model: StrictInsightFace,
    item: dict[str, Any],
    max_eval_frames: int,
    sample_frames: int,
    seed: int,
    random_sampling: bool,
    crop_mode: str,
) -> dict[str, Any]:
    generated_path = Path(item["generated"])
    ref_image_path = Path(item["ref_image"])
    input_video_path = Path(item["ref_video"])
    mask_video_path = Path(item["ref_video_facemask"])
    gt_video_path = Path(item["ground_truth"])
    mask_is_video = mask_video_path.suffix.lower() != ".json"
    face_boxes_path = resolve_face_boxes_path(item, input_video_path, mask_video_path) if crop_mode == "face-box" else None
    face_boxes = load_face_boxes(face_boxes_path) if crop_mode == "face-box" else {}

    gen_count = frame_count(generated_path)
    input_count = frame_count(input_video_path)
    mask_count = frame_count(mask_video_path) if crop_mode == "mask" and mask_is_video else input_count
    gt_count = frame_count(gt_video_path)
    eval_count = gen_count
    positive_limits = [value for value in (sample_frames, max_eval_frames) if value > 0]
    sample_limit = min(positive_limits) if positive_limits else 0
    eval_indices = sample_eval_indices(eval_count, sample_limit, random_sampling=random_sampling, seed=seed)
    generated_fps = video_fps(generated_path)
    input_fps = video_fps(input_video_path)
    gt_fps = video_fps(gt_video_path)
    input_indices = [time_aligned_index(idx, generated_fps, input_fps, input_count) for idx in eval_indices]
    gt_indices = [time_aligned_index(idx, generated_fps, gt_fps, gt_count) for idx in eval_indices]
    if crop_mode == "mask" and mask_is_video:
        mask_fps = video_fps(mask_video_path)
        mask_indices = [time_aligned_index(idx, generated_fps, mask_fps, mask_count) for idx in eval_indices]
    else:
        mask_indices = input_indices
    if crop_mode == "face-box" and (not face_boxes or max(input_indices, default=-1) >= len(face_boxes)):
        raise RuntimeError(
            f"face boxes are shorter than generated timeline: need index {max(input_indices, default=-1)}, "
            f"available={len(face_boxes)}"
        )

    ref_image = cv2.imread(ref_image_path.as_posix(), cv2.IMREAD_COLOR)
    if ref_image is None:
        raise RuntimeError(f"failed to read ref image: {ref_image_path}")
    ref_emb = model.embed(ref_image)
    if ref_emb is None:
        raise RuntimeError("ref image face detection failed")

    gen_frames = read_frames(generated_path, eval_indices)
    input_frames = read_frames(input_video_path, input_indices)
    mask_frames = read_frames(mask_video_path, mask_indices) if crop_mode == "mask" and mask_is_video else [None] * len(eval_indices)
    gt_frames = read_frames(gt_video_path, gt_indices)
    min_len = min(len(gen_frames), len(input_frames), len(mask_frames), len(gt_frames))
    gen_frames = gen_frames[:min_len]
    input_frames = input_frames[:min_len]
    mask_frames = mask_frames[:min_len]
    gt_frames = gt_frames[:min_len]
    eval_indices = eval_indices[:min_len]
    input_indices = input_indices[:min_len]
    mask_indices = mask_indices[:min_len]
    gt_indices = gt_indices[:min_len]

    cropped_gen: list[np.ndarray] = []
    cropped_input: list[np.ndarray] = []
    cropped_gt: list[np.ndarray] = []
    bbox_source = "full_frame_largest_face"
    for pos, (gen_frame, input_frame, mask_frame, gt_frame) in enumerate(
        zip(gen_frames, input_frames, mask_frames, gt_frames)
    ):
        if crop_mode == "full-frame":
            cropped_gen.append(gen_frame)
            cropped_input.append(input_frame)
            cropped_gt.append(gt_frame)
            continue

        input_shape = input_frame.shape[:2]
        base_bbox = None
        if crop_mode == "face-box":
            bbox_source = "face_boxes"
            base_bbox = bbox_from_face_boxes(face_boxes, input_indices[pos], input_shape, gen_frame.shape[:2])
        if input_frame.shape[:2] != gen_frame.shape[:2]:
            input_frame = cv2.resize(input_frame, (gen_frame.shape[1], gen_frame.shape[0]))
        if mask_frame is not None and mask_frame.shape[:2] != gen_frame.shape[:2]:
            mask_frame = cv2.resize(mask_frame, (gen_frame.shape[1], gen_frame.shape[0]))
        if gt_frame.shape[:2] != gen_frame.shape[:2]:
            gt_frame = cv2.resize(gt_frame, (gen_frame.shape[1], gen_frame.shape[0]))
        if crop_mode == "mask" and base_bbox is None and mask_frame is not None:
            bbox_source = "mask_video"
            base_bbox = calculate_bbox_from_mask(mask_frame)
        bbox = expand_bbox(base_bbox, gen_frame.shape[:2])
        cropped_gen.append(crop_face(gen_frame, bbox))
        cropped_input.append(crop_face(input_frame, bbox))
        cropped_gt.append(crop_face(gt_frame, bbox))

    gen_embs, gen_valid_positions = embed_frames(model, cropped_gen)
    input_embs, input_valid_positions = embed_frames(model, cropped_input)
    gt_embs, gt_valid_positions = embed_frames(model, cropped_gt)

    id_sim, id_scores = average_sim(ref_emb, gen_embs)
    input_ref, input_ref_scores = average_sim(ref_emb, input_embs)
    gt_ref, gt_ref_scores = average_sim(ref_emb, gt_embs)
    input_leak = None
    input_leak_scores: list[float] = []
    if input_embs and gen_embs:
        input_ref_emb = input_embs[0]
        input_leak, input_leak_scores = average_sim(input_ref_emb, gen_embs)

    return {
        "name": item["name"],
        "facebench_video_id": item["facebench_video_id"],
        "generated": generated_path.as_posix(),
        "ref_image": ref_image_path.as_posix(),
        "input_video": input_video_path.as_posix(),
        "ground_truth": gt_video_path.as_posix(),
        "crop_mode": crop_mode,
        "bbox_source": bbox_source,
        "face_boxes": face_boxes_path.as_posix() if face_boxes_path else None,
        "eval_frame_count": eval_count,
        "sampled_frame_count": min_len,
        "eval_frame_indices": eval_indices,
        "input_frame_indices": input_indices,
        "ground_truth_frame_indices": gt_indices,
        "id_sim": id_sim,
        "id_sim_scores": id_scores,
        "input_leak": input_leak,
        "input_leak_scores": input_leak_scores,
        "input_ref": input_ref,
        "input_ref_scores": input_ref_scores,
        "gt_ref": gt_ref,
        "gt_ref_scores": gt_ref_scores,
        "generated_valid_face_frames": len(gen_embs),
        "generated_valid_positions": gen_valid_positions,
        "input_valid_face_frames": len(input_embs),
        "input_valid_positions": input_valid_positions,
        "ground_truth_valid_face_frames": len(gt_embs),
        "ground_truth_valid_positions": gt_valid_positions,
    }


def summarize(label: str, cases: list[dict[str, Any]], failures: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    def metric(key: str) -> dict[str, float | int | None]:
        return mean_std([case[key] for case in cases if case.get(key) is not None])

    return {
        "crop_mode": args.crop_mode,
        "label": label,
        "backend": "insightface_scrfd_glintr100_strict",
        "alignment": (
            "full-frame SCRFD largest-face detection, 5-point landmark alignment, ArcFace glintr100; "
            "failed detections skipped"
            if args.crop_mode == "full-frame"
            else "cropped SCRFD 5-point landmark alignment, ArcFace glintr100; failed detections skipped"
        ),
        "max_eval_frames": args.max_eval_frames,
        "sample_frames": args.sample_frames,
        "random_sampling": args.random_sampling,
        "seed": args.seed,
        "case_count": len(cases),
        "failure_count": len(failures),
        "metrics": {
            "id_sim": metric("id_sim"),
            "input_leak": metric("input_leak"),
            "input_ref": metric("input_ref"),
            "gt_ref": metric("gt_ref"),
            "generated_valid_face_frames": mean_std([case["generated_valid_face_frames"] for case in cases]),
            "input_valid_face_frames": mean_std([case["input_valid_face_frames"] for case in cases]),
            "ground_truth_valid_face_frames": mean_std([case["ground_truth_valid_face_frames"] for case in cases]),
        },
        "cases": {case["name"]: case for case in cases},
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR.as_posix())
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--det-thresh", type=float, default=0.5)
    parser.add_argument("--max-eval-frames", type=int, default=81)
    parser.add_argument("--sample-frames", type=int, default=16)
    parser.add_argument("--crop-mode", choices=("full-frame", "face-box", "mask"), default="full-frame")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0)
    args = parser.parse_args()

    if args.device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        ctx_id = 0
    else:
        ctx_id = -1

    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    if args.start_index < 0 or args.end_index < 0:
        raise ValueError("start/end index must be non-negative")
    if args.end_index and args.end_index < args.start_index:
        raise ValueError("end-index must be >= start-index")
    mapping = mapping[args.start_index : args.end_index or None]
    if args.limit > 0:
        mapping = mapping[: args.limit]

    model = StrictInsightFace(Path(args.models_dir), ctx_id=ctx_id, det_size=args.det_size, det_thresh=args.det_thresh)
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, item in enumerate(mapping, start=1):
        try:
            case = evaluate_case(
                model,
                item,
                args.max_eval_frames,
                args.sample_frames,
                args.seed,
                args.random_sampling,
                args.crop_mode,
            )
            cases.append(case)
            print(
                f"[{idx:03d}/{len(mapping):03d}] {item['name']} "
                f"id={case['id_sim'] if case['id_sim'] is not None else 'NA'} "
                f"valid={case['generated_valid_face_frames']}/{case['sampled_frame_count']}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"name": item.get("name"), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{idx:03d}/{len(mapping):03d}] FAILED {item.get('name')}: {exc}", flush=True)

    output = summarize(args.label, cases, failures, args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0 if cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
