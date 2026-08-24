#!/usr/bin/env python3
"""Run one metric command per GPU and merge its JSON shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stats(values: list[float], extended: bool = False, dtype: Any = np.float64) -> dict[str, float | int | None]:
    if not values:
        base: dict[str, float | int | None] = {"mean": None, "std": None, "count": 0}
        if extended:
            base.update({"median": None, "min": None, "max": None})
        return base
    array = np.asarray(values, dtype=dtype)
    result: dict[str, float | int | None] = {
        "mean": float(array.mean()), "std": float(array.std()), "count": int(array.size)
    }
    if extended:
        result.update({"median": float(np.median(array)), "min": float(array.min()), "max": float(array.max())})
    return result


def ordered_case_ids(mapping: Path) -> list[str]:
    payload = load(mapping)
    rows = payload.get("items", []) if isinstance(payload, dict) else payload
    return [str(row.get("case_id", row.get("name"))) for row in rows]


def merge_strict(shards: list[dict[str, Any]], order: list[str]) -> dict[str, Any]:
    output = dict(shards[0])
    cases = {key: value for shard in shards for key, value in shard.get("cases", {}).items()}
    output["cases"] = {key: cases[key] for key in order if key in cases}
    output["failures"] = [row for shard in shards for row in shard.get("failures", [])]
    values = list(output["cases"].values())
    metric_keys = ("id_sim", "input_leak", "input_ref", "gt_ref")
    output["metrics"] = {
        key: stats([float(row[key]) for row in values if row.get(key) is not None], dtype=np.float32)
        for key in metric_keys
    }
    for key in ("generated_valid_face_frames", "input_valid_face_frames", "ground_truth_valid_face_frames"):
        output["metrics"][key] = stats([float(row[key]) for row in values], dtype=np.float32)
    output["case_count"], output["failure_count"] = len(values), len(output["failures"])
    return output


def merge_multi(shards: list[dict[str, Any]], order: list[str]) -> dict[str, Any]:
    output = dict(shards[0])
    cases = {key: value for shard in shards for key, value in shard.get("cases", {}).items()}
    output["cases"] = {key: cases[key] for key in order if key in cases}
    output["failures"] = [row for shard in shards for row in shard.get("failures", [])]
    values = list(output["cases"].values())
    metrics: dict[str, Any] = {}
    for name in output.get("models", {}):
        metrics[name] = stats([float(row[name]) for row in values if row.get(name) is not None])
        metrics[f"{name}_variance"] = stats([float(row[f"{name}_variance"]) for row in values if row.get(f"{name}_variance") is not None])
    metrics["variance"] = metrics.get("id_arc_variance")
    sampled = sum(int(row["sampled_frame_count"]) for row in values)
    valid = sum(int(row["valid_face_frames"]) for row in values)
    metrics["face_detection_rate"] = float(valid / sampled) if sampled else None
    output.update({"metrics": metrics, "case_count": len(values), "failure_count": len(output["failures"])})
    return output


def merge_vbench(shards: list[dict[str, Any]], order: list[str]) -> dict[str, Any]:
    output = dict(shards[0])
    cases = {str(row["case_id"]): row for shard in shards for row in shard.get("cases", [])}
    output["cases"] = [cases[key] for key in order if key in cases]
    output["failures"] = [row for shard in shards for row in shard.get("failures", [])]
    keys = list(output.get("metrics", {}))
    output["metrics"] = {
        key: stats([float(row[key]) for row in output["cases"] if row.get(key) is not None], extended=True)
        for key in keys
    }
    output.update({"case_count": len(output["cases"]), "failure_count": len(output["failures"])})
    return output


MERGERS = {"strict": merge_strict, "multi": merge_multi, "vbench": merge_vbench}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=tuple(MERGERS), required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-list", required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("worker command is required after --")
    order = ordered_case_ids(args.mapping)
    gpus = [value.strip() for value in args.gpu_list.split(",") if value.strip()]
    if not gpus:
        raise ValueError("gpu list is empty")
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    total = len(order)
    for rank, gpu in enumerate(gpus):
        start, end = total * rank // len(gpus), total * (rank + 1) // len(gpus)
        shard = args.shard_dir / f"worker_{rank:02d}.json"
        shard.unlink(missing_ok=True)
        replacements = {"{output}": shard.as_posix(), "{start}": str(start), "{end}": str(end), "{rank}": str(rank)}
        worker = [replacements.get(value, value) for value in command]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        print(f"[metric-shard] kind={args.kind} rank={rank} physical_gpu={gpu} cases=[{start},{end})", flush=True)
        processes.append((rank, gpu, shard, subprocess.Popen(worker, env=env)))
    failed, missing = [], []
    for rank, gpu, shard, process in processes:
        code = process.wait()
        if code != 0:
            failed.append((rank, gpu, code))
        if not shard.is_file():
            missing.append((rank, gpu, code))
    if missing:
        print(f"[metric-shard-missing] {missing}", file=sys.stderr)
        return 1
    if failed:
        print(f"[metric-shard-error] {failed}", file=sys.stderr)
    payload = MERGERS[args.kind]([load(shard) for _, _, shard, _ in processes], order)
    write(args.output, payload)
    print(f"[metric-merge] kind={args.kind} cases={payload['case_count']} failures={payload['failure_count']} output={args.output}")
    return 0 if not failed and not payload["failure_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
