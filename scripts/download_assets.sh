#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${1:-$ROOT/assets}"

modelscope download \
  --dataset lireno/swapface_benchmark \
  --local_dir "$DESTINATION" \
  --max-workers "${MAX_WORKERS:-8}"

python "$ROOT/tools/validate_assets.py" --assets-root "$DESTINATION"
