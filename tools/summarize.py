#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path | None) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path and path.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge any completed benchmark metric artifacts.")
    parser.add_argument("--identity-strict", type=Path)
    parser.add_argument("--identity-multibackbone", type=Path)
    parser.add_argument("--facebench", type=Path)
    parser.add_argument("--vbench-quality", type=Path)
    parser.add_argument("--imaging-quality", type=Path, help="Legacy standalone imaging-quality artifact")
    parser.add_argument("--input-report", type=Path)
    parser.add_argument("--selected", default="", help="Canonical comma-separated selected metrics")
    parser.add_argument("--benchmark-mode", choices=("short", "long"))
    parser.add_argument("--max-eval-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "identity_strict": load(args.identity_strict),
        "identity_multibackbone": load(args.identity_multibackbone),
        "facebench": load(args.facebench),
        "vbench_quality": load(args.vbench_quality),
        "imaging_quality": load(args.imaging_quality),
    }
    flat: dict[str, Any] = {}
    strict = sources["identity_strict"]
    if strict:
        flat.update({key: strict["metrics"][key] for key in ("id_sim", "input_leak") if key in strict.get("metrics", {})})
    multi = sources["identity_multibackbone"]
    if multi:
        flat.update({key: value for key, value in multi.get("metrics", {}).items() if key in {"id_arc", "id_ins", "id_cur", "face_detection_rate"}})
        if multi.get("metrics", {}).get("variance") is not None:
            flat["id_variance"] = multi["metrics"]["variance"]
    facebench = sources["facebench"]
    if facebench:
        flat.update(facebench.get("metrics", {}))
    quality = sources["vbench_quality"] or sources["imaging_quality"]
    if quality:
        flat.update(quality.get("metrics", {}))
    selected = {value for value in args.selected.split(",") if value}
    if selected:
        permitted = set()
        aliases = {
            "id_strict": {"id_sim"}, "input_leak": {"input_leak"},
            "id_arc": {"id_arc", "id_variance", "face_detection_rate"},
            "id_ins": {"id_ins", "face_detection_rate"}, "id_cur": {"id_cur", "face_detection_rate"},
            "face_similarity": {"face_similarity", "face_similarity_src"},
            "pose": {"pose_distance"},
            "gaze": {"gaze_l2_distance", "gaze_cosine_similarity"},
            "expression": {"exp_l2_distance"}, "lighting": {"gamma_l2_distance"},
            "imaging_quality": {"vbench_imaging_quality", "raw_musiq_spaq"},
            "subject_consistency": {"vbench_subject_consistency"},
            "temporal_flickering": {"vbench_temporal_flickering"},
        }
        for name in selected:
            permitted.update(aliases.get(name, set()))
        flat = {key: value for key, value in flat.items() if key in permitted}
    input_report = load(args.input_report)
    failures = {name: value.get("failure_count", 0) for name, value in sources.items() if value is not None}
    if input_report:
        failures["input_validation"] = input_report.get("error_count", 0)
    artifacts = {
        name: path.resolve().as_posix()
        for name, path in {
            "identity_strict": args.identity_strict,
            "identity_multibackbone": args.identity_multibackbone,
            "facebench": args.facebench,
            "vbench_quality": args.vbench_quality,
            "imaging_quality": args.imaging_quality,
            "input_report": args.input_report,
        }.items() if path and path.is_file()
    }
    grouped = {
        "identity": {key: value for key, value in flat.items() if key in {"id_sim", "id_arc", "id_ins", "id_cur", "id_variance", "input_leak", "face_detection_rate", "face_similarity", "face_similarity_src"}},
        "quality": {key: value for key, value in flat.items() if key in {"vbench_imaging_quality", "raw_musiq_spaq"}},
        "temporal": {key: value for key, value in flat.items() if key in {"vbench_subject_consistency", "vbench_temporal_flickering"}},
        "attribute_preservation": {key: value for key, value in flat.items() if key in {"pose_distance", "gaze_l2_distance", "gaze_cosine_similarity", "exp_l2_distance", "gamma_l2_distance"}},
    }
    summary = {
        "protocol": {
            "metric_protocol_version": (facebench or {}).get("metric_protocol_version", "legacy"),
            "gaze_cosine_definition": (facebench or {}).get("gaze_cosine_definition"),
            "benchmark_mode": args.benchmark_mode,
            "max_eval_frames": args.max_eval_frames,
            "frame_stride": args.frame_stride,
            "frame_indices": f"0,{args.frame_stride},{2 * args.frame_stride},...",
            "aggregation": "frame mean per case, then equal-weight mean across cases",
        },
        "case_count": input_report.get("case_count") if input_report else next((value.get("case_count") for value in sources.values() if value), None),
        "failure_count": failures,
        "metrics": {key: value for key, value in grouped.items() if value},
        "metrics_flat": flat,
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
