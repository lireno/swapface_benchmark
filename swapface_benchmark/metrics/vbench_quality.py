#!/usr/bin/env python3
"""VBench quality metrics adapted to the SwapFace mapping protocol.

The generated video is decoded once per case and shared by every requested
metric.  The formulas and preprocessing follow VBench's imaging quality,
subject consistency, and temporal flickering implementations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TVF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, (PROJECT_ROOT / "vendor/pyiqa").as_posix())

from pyiqa.archs.musiq_arch import MUSIQ  # noqa: E402

METRICS = {"imaging_quality", "subject_consistency", "temporal_flickering"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None, "count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "std": float(array.std()),
        "median": float(np.median(array)), "min": float(array.min()),
        "max": float(array.max()), "count": int(array.size),
    }


def parse_metrics(value: str) -> set[str]:
    values = {part.strip().replace("-", "_") for part in value.split(",") if part.strip()}
    unknown = values - METRICS
    if unknown:
        raise ValueError(f"unknown VBench metrics: {', '.join(sorted(unknown))}")
    return values


def load_dino(repo: Path, weights: Path, device: torch.device) -> torch.nn.Module:
    if not repo.is_dir():
        raise FileNotFoundError(f"DINO source directory not found: {repo}")
    if not weights.is_file():
        raise FileNotFoundError(f"DINO weights not found: {weights}")
    model = torch.hub.load(repo.as_posix(), "dino_vitb16", source="local", pretrained=False)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    return model.eval().to(device)


def dino_preprocess(frames: list[np.ndarray]) -> torch.Tensor:
    # VBench dino_transform(224): RGB, Resize(short side=224), ImageNet normalization.
    tensors = [torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) for frame in frames]
    batch = torch.stack(tensors).float() / 255.0
    batch = TVF.resize(batch, 224, antialias=False)
    return TVF.normalize(batch, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def musiq_preprocess(frames: list[np.ndarray]) -> torch.Tensor:
    tensors = [torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) for frame in frames]
    batch = torch.stack(tensors).float()
    _, _, height, width = batch.shape
    if max(height, width) > 512:
        scale = 512.0 / max(height, width)
        batch = TVF.resize(batch, [int(scale * height), int(scale * width)], antialias=False)
    return batch / 255.0


def evaluate_video(
    path: Path, selected: set[str], device: torch.device, batch_size: int,
    musiq: torch.nn.Module | None, dino: torch.nn.Module | None,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(path.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_count = 0
    pair_count = 0
    flicker_sum = 0.0
    previous_frame: np.ndarray | None = None
    first_feature: torch.Tensor | None = None
    previous_feature: torch.Tensor | None = None
    subject_sum = 0.0
    musiq_scores: list[float] = []
    pending: list[np.ndarray] = []

    def flush() -> None:
        nonlocal first_feature, previous_feature, subject_sum
        if not pending:
            return
        with torch.inference_mode():
            if musiq is not None:
                output = musiq(musiq_preprocess(pending).to(device))
                musiq_scores.extend(output.detach().float().reshape(-1).cpu().tolist())
            if dino is not None:
                features = F.normalize(dino(dino_preprocess(pending).to(device)), dim=-1, p=2).detach()
                for feature in features:
                    feature = feature.unsqueeze(0)
                    if first_feature is None:
                        first_feature = feature
                    else:
                        adjacent = max(0.0, F.cosine_similarity(previous_feature, feature).item())
                        first = max(0.0, F.cosine_similarity(first_feature, feature).item())
                        subject_sum += (adjacent + first) / 2.0
                    previous_feature = feature
        pending.clear()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_count and frame.shape != previous_frame.shape:
                raise RuntimeError(f"frame shape changed at frame {frame_count}")
            if "temporal_flickering" in selected and previous_frame is not None:
                flicker_sum += float(np.mean(cv2.absdiff(previous_frame.astype(np.float32), frame.astype(np.float32))))
                pair_count += 1
            previous_frame = frame
            pending.append(frame)
            frame_count += 1
            if len(pending) >= max(1, batch_size):
                flush()
        flush()
    finally:
        capture.release()
    if frame_count == 0:
        raise RuntimeError("no frames decoded")
    if declared > 0 and frame_count != declared:
        raise RuntimeError(f"decoded {frame_count} frames but container declares {declared}")
    result: dict[str, Any] = {"frame_count": frame_count, "adjacent_pair_count": max(0, frame_count - 1)}
    if "imaging_quality" in selected:
        result["raw_musiq_spaq"] = float(np.mean(musiq_scores))
        result["vbench_imaging_quality"] = result["raw_musiq_spaq"] / 100.0
    if "subject_consistency" in selected:
        if frame_count < 2:
            raise RuntimeError("subject consistency requires at least two frames")
        result["vbench_subject_consistency"] = subject_sum / (frame_count - 1)
    if "temporal_flickering" in selected:
        if pair_count == 0:
            raise RuntimeError("temporal flickering requires at least two frames")
        result["vbench_temporal_flickering"] = (255.0 - flicker_sum / pair_count) / 255.0
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", default=",".join(sorted(METRICS)))
    parser.add_argument("--musiq-model", type=Path, default=PROJECT_ROOT / "assets/models/vbench/musiq_spaq_ckpt-358bb6af.pth")
    parser.add_argument("--dino-repo", type=Path, default=PROJECT_ROOT / "vendor/dino")
    parser.add_argument("--dino-model", type=Path, default=PROJECT_ROOT / "assets/models/vbench/dino/dino_vitbase16_pretrain.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    selected = parse_metrics(args.metrics)
    if not selected:
        raise ValueError("no metrics selected")
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    musiq = None
    dino = None
    models: dict[str, Any] = {}
    if "imaging_quality" in selected:
        musiq = MUSIQ(pretrained_model_path=args.musiq_model.as_posix()).eval().to(device)
        models["musiq_spaq"] = {"path": args.musiq_model.resolve().as_posix(), "sha256": sha256(args.musiq_model)}
    if "subject_consistency" in selected:
        dino = load_dino(args.dino_repo, args.dino_model, device)
        models["dino_vitb16"] = {"path": args.dino_model.resolve().as_posix(), "sha256": sha256(args.dino_model), "source": args.dino_repo.resolve().as_posix()}
    cases, failures = [], []
    for index, item in enumerate(mapping, 1):
        try:
            result = evaluate_video(Path(item["generated"]), selected, device, args.batch_size, musiq, dino)
            cases.append({"case_id": item.get("case_id", item.get("name", str(index))), "video": item["generated"], **result})
        except Exception as error:
            failures.append({"case_id": item.get("case_id", item.get("name", str(index))), "video": item.get("generated"), "error": f"{type(error).__name__}: {error}"})
        if index % 10 == 0 or index == len(mapping):
            print(f"[vbench-quality] {index}/{len(mapping)} failures={len(failures)}", flush=True)
    metric_keys = {
        "imaging_quality": ("vbench_imaging_quality", "raw_musiq_spaq"),
        "subject_consistency": ("vbench_subject_consistency",),
        "temporal_flickering": ("vbench_temporal_flickering",),
    }
    metrics = {key: stats([case[key] for case in cases]) for name in selected for key in metric_keys[name]}
    payload = {
        "metric": "VBench quality metrics", "selected_metrics": sorted(selected),
        "protocol": {
            "imaging_quality": "VBench MUSIQ-SPAQ longer-side preprocessing; frame mean per video.",
            "subject_consistency": "VBench DINO ViT-B/16: mean of adjacent-frame and first-frame cosine similarities.",
            "temporal_flickering": "VBench adjacent-frame pixel MAE; intended for static videos and not motion compensated.",
            "aggregation": "Per-video macro mean in summary.",
        },
        "mapping": args.mapping.resolve().as_posix(), "models": models,
        "case_count": len(cases), "failure_count": len(failures), "metrics": metrics,
        "cases": cases, "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
