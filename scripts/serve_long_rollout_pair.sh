#!/usr/bin/env bash
# Range-enabled server is required for reliable synchronized video seeking.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8879}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/cpfs/users/lyw/venvs/idvtrain/bin/python}"
[[ "$PORT" =~ ^[0-9]{1,5}$ ]] && (( 10#$PORT >= 1 && 10#$PORT <= 65535 )) || exit 2
[[ -f "$ROOT/web_reports/long_rollout_pair/index.html" ]] || {
  echo 'Build first: tools/build_long_rollout_pair.py' >&2; exit 1;
}
echo "URL: http://localhost:$PORT/long_rollout_pair/"
echo "SSH tunnel: ssh -L $PORT:127.0.0.1:$PORT user@host"
exec "$PYTHON_BIN" "$ROOT/tools/serve_gallery.py" --port "$PORT" --directory "$ROOT/web_reports"
