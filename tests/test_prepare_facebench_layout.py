from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PrepareFaceBenchLayoutTest(unittest.TestCase):
    def test_links_face_boxes_without_creating_mask_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            source_video = assets / "source.mp4"
            ref_image = assets / "ref.png"
            generated_video = assets / "generated.mp4"
            boxes = assets / "boxes.json"
            for path in (source_video, ref_image, generated_video):
                path.touch()
            boxes.write_text('{"100": [10, 20, 30, 40]}', encoding="utf-8")

            mapping = root / "mapping.json"
            mapping.write_text(
                json.dumps(
                    [
                        {
                            "facebench_video_id": "00001",
                            "ref_video": source_video.as_posix(),
                            "ref_image": ref_image.as_posix(),
                            "generated": generated_video.as_posix(),
                            "ref_video_face_boxes": boxes.as_posix(),
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "layout"

            subprocess.run(
                [
                    sys.executable,
                    (PROJECT_ROOT / "tools/prepare_facebench_layout.py").as_posix(),
                    "--mapping",
                    mapping.as_posix(),
                    "--output-root",
                    output.as_posix(),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual((output / "source/00001_boxes.json").resolve(), boxes.resolve())
            self.assertFalse((output / "source/00001_mask.mp4").exists())
            self.assertFalse((output / "masks").exists())


if __name__ == "__main__":
    unittest.main()
