#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-strict", type=Path, required=True)
    parser.add_argument("--identity-multibackbone", type=Path, required=True)
    parser.add_argument("--facebench", type=Path, required=True)
    parser.add_argument("--imaging-quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    strict = load(args.identity_strict)
    multi = load(args.identity_multibackbone)
    facebench = load(args.facebench)
    imaging = load(args.imaging_quality)
    summary = {
        "case_count": strict["case_count"],
        "failure_count": {
            "identity_strict": strict["failure_count"],
            "identity_multibackbone": multi["failure_count"],
            "imaging_quality": imaging["failure_count"],
        },
        "metrics": {
            "id_sim": strict["metrics"]["id_sim"],
            "input_leak": strict["metrics"]["input_leak"],
            "id_arc": multi["metrics"]["id_arc"],
            "id_ins": multi["metrics"]["id_ins"],
            "id_cur": multi["metrics"]["id_cur"],
            "id_variance": multi["metrics"]["variance"],
            "face_detection_rate": multi["metrics"]["face_detection_rate"],
            "vbench_imaging_quality": imaging["metrics"]["vbench_imaging_quality"],
            **facebench.get("metrics", {}),
        },
        "artifacts": {
            key: path.resolve().as_posix()
            for key, path in {
                "identity_strict": args.identity_strict,
                "identity_multibackbone": args.identity_multibackbone,
                "facebench": args.facebench,
                "imaging_quality": args.imaging_quality,
            }.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
