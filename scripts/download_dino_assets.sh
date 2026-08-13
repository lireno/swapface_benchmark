#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS_ROOT="${1:-$ROOT/assets}"
DINO_ROOT="$ASSETS_ROOT/models/vbench/dino"
WEIGHTS="$DINO_ROOT/dino_vitbase16_pretrain.pth"
mkdir -p "$DINO_ROOT"
[[ -f "$WEIGHTS" ]] || wget -O "$WEIGHTS" https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
printf 'DINO source is vendored at: %s\n' "$ROOT/vendor/dino"
sha256sum "$WEIGHTS"
