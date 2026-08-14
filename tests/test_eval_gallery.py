from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "eval_gallery.py"
SPEC = importlib.util.spec_from_file_location("eval_gallery", MODULE_PATH)
assert SPEC and SPEC.loader
gallery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gallery)


def test_safe_id_normalizes_display_text() -> None:
    assert gallery.safe_id(" GTID id5 / 25 steps ") == "GTID-id5-25-steps"


def test_metric_number_accepts_scalar_and_mean() -> None:
    assert gallery.metric_number(0.5) == 0.5
    assert gallery.metric_number({"mean": 0.25, "std": 0.1}) == 0.25
    assert gallery.metric_number({"std": 0.1}) is None


def test_page_is_mode_specific() -> None:
    page = gallery.page_html("long")
    assert "SwapFace Long Benchmark" in page
    assert "swapface-gallery::long" in page
