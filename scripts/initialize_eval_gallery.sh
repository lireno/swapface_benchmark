#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${SWAPFACE_BENCHMARK_DATASET_ROOT:-/mnt/cpfs/users/lzk/dataset/swapface_benchmark}"
SHORT_MANIFEST="${SHORT_MANIFEST:-$DATASET_ROOT/short_200/benchmark/non_long_200/manifest.json}"
LONG_MANIFEST="${LONG_MANIFEST:-$DATASET_ROOT/long_200/benchmark/manifest.json}"

exec "${PYTHON_BIN:-python}" "$ROOT/tools/eval_gallery.py" init \
  --short-manifest "$SHORT_MANIFEST" \
  --long-manifest "$LONG_MANIFEST" "$@"
