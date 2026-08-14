#!/usr/bin/env python3
"""Atomically transcode registered result videos to browser-compatible H.264."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

import cv2


ROOT = Path(__file__).resolve().parents[1]
PRINT_LOCK = threading.Lock()


def ffmpeg_executable() -> str:
    configured = os.environ.get("FFMPEG_BIN")
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.as_posix()
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as error:
        raise RuntimeError("ffmpeg or imageio-ffmpeg is required") from error


def info(path: Path) -> tuple[int, float, str, bool]:
    capture = cv2.VideoCapture(path.as_posix())
    opened = capture.isOpened()
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((fourcc >> (8 * index)) & 0xff) for index in range(4))
    decoded, _ = capture.read()
    capture.release()
    return frames, fps, codec, opened and decoded


def transcode(path: Path, ffmpeg: str, crf: int) -> tuple[Path, str]:
    before = info(path)
    if before[2].lower() in {"h264", "avc1"} and before[3]:
        return path, "already-h264"
    temporary = path.with_suffix(".h264.tmp.mp4")
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", path.as_posix(), "-map", "0:v:0", "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                temporary.as_posix(),
            ],
            check=True,
        )
        after = info(temporary)
        tolerance = max(0.01, before[1] * 0.001)
        if not after[3] or after[2].lower() not in {"h264", "avc1"}:
            raise RuntimeError(f"H.264 validation failed: {after}")
        if after[0] != before[0] or abs(after[1] - before[1]) > tolerance:
            raise RuntimeError(f"timeline changed: before={before}, after={after}")
        temporary.replace(path)
        return path, f"{before[2]}->h264"
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-mode", choices=("short", "long"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()
    registry = json.loads((ROOT / "registries" / f"{args.benchmark_mode}.json").read_text(encoding="utf-8"))
    videos = sorted({
        Path(row["generated"]).resolve()
        for run in registry.get("runs", [])
        for row in json.loads(Path(run["mapping_path"]).read_text(encoding="utf-8"))
    })
    if not videos:
        raise RuntimeError(f"no registered {args.benchmark_mode} videos")
    ffmpeg = ffmpeg_executable()
    failures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(transcode, path, ffmpeg, args.crf): path for path in videos}
        for future in as_completed(futures):
            path = futures[future]
            try:
                _, action = future.result()
                completed += 1
                if completed % 25 == 0 or completed == len(videos):
                    with PRINT_LOCK:
                        print(f"[h264] {completed}/{len(videos)} latest={path.name} {action}", flush=True)
            except Exception as error:
                failures.append((path, error))
                with PRINT_LOCK:
                    print(f"[h264-error] {path}: {error}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(videos)} transcodes failed")
    print(f"[h264-complete] mode={args.benchmark_mode} videos={len(videos)} codec=h264 pix_fmt=yuv420p")


if __name__ == "__main__":
    main()
