#!/usr/bin/env bash
# Usage: bash scripts/serve_long_method_picker.sh [port]
# Access remotely through SSH: ssh -L 8878:127.0.0.1:8878 user@host
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8878}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/cpfs/users/lyw/venvs/idvtrain/bin/python}"
[[ "$PORT" =~ ^[0-9]{1,5}$ ]] && (( 10#$PORT >= 1 && 10#$PORT <= 65535 )) || {
  echo "Invalid port: $PORT" >&2
  exit 2
}
[[ -f "$ROOT/web_reports/long_method_picker/index.html" ]] || {
  echo "Missing gallery. Run tools/build_long_method_picker.py first." >&2
  exit 1
}
echo "URL: http://localhost:$PORT/long_method_picker/"
echo "SSH tunnel: ssh -L $PORT:127.0.0.1:$PORT user@host"
exec "$PYTHON_BIN" "$ROOT/tools/serve_gallery.py" --port "$PORT" --directory "$ROOT/web_reports"
