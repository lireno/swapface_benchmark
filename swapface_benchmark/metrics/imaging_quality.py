#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TVF


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "vendor/pyiqa"
sys.path.insert(0, RUNTIME_ROOT.as_posix())

from pyiqa.archs.musiq_arch import MUSIQ  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact VBench Imaging Quality (MUSIQ-SPAQ) on mapped videos.")
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "assets/models/vbench/musiq_spaq_ckpt-358bb6af.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-videos", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_video(path: Path) -> torch.Tensor:
    capture = cv2.VideoCapture(path.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded: {path}")
    return torch.stack(frames).float()


def vbench_resize(frames: torch.Tensor) -> torch.Tensor:
    _, _, height, width = frames.shape
    if max(height, width) > 512:
        scale = 512.0 / max(height, width)
        frames = TVF.resize(
            frames,
            [int(scale * height), int(scale * width)],
            antialias=False,
        )
    return frames / 255.0


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }


def main() -> None:
    args = parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    if not isinstance(mapping, list):
        raise ValueError("Expected mapping.json to contain a list")
    if args.max_videos > 0:
        mapping = mapping[: args.max_videos]
    if not mapping:
        raise ValueError("Mapping is empty")

    device = torch.device(args.device)
    model = MUSIQ(pretrained_model_path=args.model_path.as_posix()).eval().to(device)
    cases = []
    failures = []
    for item_index, item in enumerate(mapping):
        video_path = Path(item["generated"])
        try:
            frames = vbench_resize(read_video(video_path))
            scores = []
            with torch.inference_mode():
                for batch in frames.split(max(1, args.batch_size)):
                    output = model(batch.to(device))
                    scores.extend(output.detach().float().reshape(-1).cpu().tolist())
            raw_score = float(np.mean(scores))
            cases.append(
                {
                    "case_id": item.get("case_id", str(item_index)),
                    "video": video_path.as_posix(),
                    "frame_count": len(scores),
                    "raw_musiq_spaq": raw_score,
                    "vbench_imaging_quality": raw_score / 100.0,
                }
            )
        except Exception as error:
            failures.append({"case_id": item.get("case_id", str(item_index)), "video": str(video_path), "error": str(error)})
        if (item_index + 1) % 10 == 0:
            print(f"[vbench-iq] {item_index + 1}/{len(mapping)} failures={len(failures)}", flush=True)

    normalized = [case["vbench_imaging_quality"] for case in cases]
    raw = [case["raw_musiq_spaq"] for case in cases]
    output = {
        "metric": "VBench Imaging Quality",
        "definition": "MUSIQ-SPAQ frame scores averaged per video and across videos, divided by 100.",
        "official_preprocessing": "longer: resize only when max(H,W)>512 so the longest side is 512; antialias=False",
        "mapping": args.mapping.resolve().as_posix(),
        "model_path": args.model_path.resolve().as_posix(),
        "model_sha256": sha256(args.model_path),
        "case_count": len(cases),
        "failure_count": len(failures),
        "metrics": {
            "vbench_imaging_quality": stats(normalized),
            "raw_musiq_spaq": stats(raw),
        },
        "cases": cases,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["metrics"], indent=2), flush=True)
    if failures:
        raise RuntimeError(f"VBench Imaging Quality failed for {len(failures)} videos")


if __name__ == "__main__":
    main()
