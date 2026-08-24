from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_metric_shards.py"
SPEC = importlib.util.spec_from_file_location("run_metric_shards", MODULE_PATH)
assert SPEC and SPEC.loader
shards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shards)


def test_merge_strict_preserves_manifest_order_and_recomputes_metrics() -> None:
    base = {
        "label": "x", "cases": {}, "failures": [], "metrics": {},
    }
    rows = {
        "a": {"id_sim": 0.2, "input_leak": 0.1, "input_ref": 0.0, "gt_ref": 0.8,
              "generated_valid_face_frames": 2, "input_valid_face_frames": 2, "ground_truth_valid_face_frames": 2},
        "b": {"id_sim": 0.6, "input_leak": 0.3, "input_ref": 0.2, "gt_ref": 0.6,
              "generated_valid_face_frames": 4, "input_valid_face_frames": 4, "ground_truth_valid_face_frames": 4},
    }
    merged = shards.merge_strict([{**base, "cases": {"b": rows["b"]}}, {**base, "cases": {"a": rows["a"]}}], ["a", "b"])
    assert list(merged["cases"]) == ["a", "b"]
    assert merged["metrics"]["id_sim"]["mean"] == pytest.approx(0.4)
    assert merged["case_count"] == 2


def test_merge_multi_recomputes_detection_rate() -> None:
    base = {"models": {"id_arc": {}}, "cases": {}, "failures": [], "metrics": {}}
    case = {"id_arc": 0.5, "id_arc_variance": 0.01, "sampled_frame_count": 10, "valid_face_frames": 8}
    merged = shards.merge_multi([{**base, "cases": {"a": case}}], ["a"])
    assert merged["metrics"]["face_detection_rate"] == 0.8
    assert merged["metrics"]["id_arc"]["count"] == 1


def test_merge_vbench_preserves_extended_statistics() -> None:
    base = {"metrics": {"vbench_imaging_quality": {}}, "cases": [], "failures": []}
    merged = shards.merge_vbench([
        {**base, "cases": [{"case_id": "b", "vbench_imaging_quality": 0.8}]},
        {**base, "cases": [{"case_id": "a", "vbench_imaging_quality": 0.4}]},
    ], ["a", "b"])
    assert [row["case_id"] for row in merged["cases"]] == ["a", "b"]
    assert merged["metrics"]["vbench_imaging_quality"]["mean"] == pytest.approx(0.6)
    assert merged["metrics"]["vbench_imaging_quality"]["min"] == 0.4
