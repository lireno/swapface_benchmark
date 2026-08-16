from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from swapface_benchmark.face_boxes import (
    face_box_for_frame,
    frame_to_sequence_index,
    load_ordered_face_boxes,
    scale_bbox,
)


class FaceBoxesTest(unittest.TestCase):
    def test_loads_boxes_by_numeric_key_not_json_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boxes.json"
            path.write_text(
                json.dumps(
                    {
                        "102": [30, 40, 80, 100],
                        "100": [10, 20, 60, 80],
                        "101": [20, 30, 70, 90],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                load_ordered_face_boxes(path),
                [
                    (10.0, 20.0, 60.0, 80.0),
                    (20.0, 30.0, 70.0, 90.0),
                    (30.0, 40.0, 80.0, 100.0),
                ],
            )

    def test_normalizes_coordinates_and_skips_invalid_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boxes.json"
            path.write_text(
                json.dumps(
                    {
                        "0": [30, 40, 10, 20],
                        "1": [1, 2, 1, 5],
                        "2": ["bad", 2, 3, 4],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(load_ordered_face_boxes(path), [(10.0, 20.0, 30.0, 40.0)])

    def test_maps_frame_and_box_timelines_by_relative_position(self) -> None:
        self.assertEqual(frame_to_sequence_index(0, 300, 150), 0)
        self.assertEqual(frame_to_sequence_index(149, 300, 150), 74)
        self.assertEqual(frame_to_sequence_index(299, 300, 150), 149)
        self.assertEqual(
            face_box_for_frame([(0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 2.0, 2.0)], 80, 81),
            (1.0, 1.0, 2.0, 2.0),
        )

    def test_scales_bbox_between_video_resolutions(self) -> None:
        actual = scale_bbox((600.0, 200.0, 900.0, 600.0), (1080, 1920), (720, 1280))
        expected = (400.0, 200.0 / 1.5, 600.0, 400.0)
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value)


if __name__ == "__main__":
    unittest.main()
