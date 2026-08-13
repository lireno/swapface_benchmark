#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -gt 0 ]] || { echo "usage: scripts/run_benchmark.sh RESULTS_DIR [OUTPUT_DIR]" >&2; exit 2; }
ARGS=("$1")
[[ $# -gt 1 ]] && ARGS+=(--output-dir "$2")
[[ -n "${ASSETS_ROOT:-}" ]] && ARGS+=(--assets-root "$ASSETS_ROOT")
ARGS+=(--gpu-list "${GPU_LIST:-0}" --sample-frames "${SAMPLE_FRAMES:-81}" --limit "${LIMIT:-0}")
bash "$ROOT/scripts/evaluate.sh" "${ARGS[@]}"
