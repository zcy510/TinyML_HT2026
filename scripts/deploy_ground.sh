#!/usr/bin/env bash
# =============================================================================
# One-click deploy + start of the ground station on macOS / Linux
# (handy for testing at home on the MacBook; the lab PC uses deploy_ground.bat).
#
#   ./scripts/deploy_ground.sh [weights] [extra server args...]
#
#   ./scripts/deploy_ground.sh                    # yolov8s.pt on port 8000
#   ./scripts/deploy_ground.sh yolov8m.pt --port 9000
#   SKIP_TESTS=1 ./scripts/deploy_ground.sh       # skip the quick test suite
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV="$ROOT/.venv"
REQ="$ROOT/requirements-ground.txt"

say()  { printf '\n\033[1;34m[deploy-ground]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deploy-ground] WARNING:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[deploy-ground] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

WEIGHTS="yolov8s.pt"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then WEIGHTS="$1"; shift; fi
EXTRA=("$@")

command -v python3 >/dev/null || die "python3 not found (macOS: brew install python@3.12)"
python3 - <<'PY' || die "Python 3.9-3.13 required"
import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 13) else 1)
PY

if [ ! -x "$VENV/bin/python" ]; then
  say "1/4 Creating virtual environment in .venv"
  python3 -m venv "$VENV"
else
  say "1/4 Using existing .venv"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if cmp -s "$REQ" "$VENV/.installed-requirements-ground.txt"; then
  say "2/4 Python deps up to date"
else
  say "2/4 Installing Python deps (first time downloads PyTorch, a few minutes)"
  python -m pip install --upgrade pip
  python -m pip install -r "$REQ"
  cp "$REQ" "$VENV/.installed-requirements-ground.txt"
fi
python - <<'PY'
import torch
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"torch {torch.__version__} -> accelerator: {dev}")
PY

if [ "${SKIP_TESTS:-0}" != "1" ]; then
  say "3/4 Running quick tests"
  python -m pytest tests -q || warn "tests failed - starting the server anyway"
else
  say "3/4 Tests skipped"
fi

say "4/4 Starting ground station"
echo "Give the Raspberry Pi one of these addresses:"
if command -v ipconfig >/dev/null 2>&1 && [ "$(uname)" = "Darwin" ]; then
  for i in en0 en1 en2 en3 en4 en5 en6 en7 bridge100; do
    ip=$(ipconfig getifaddr "$i" 2>/dev/null || true)
    [ -n "$ip" ] && echo "  $i: http://$ip:8000"
  done
else
  for ip in $(hostname -I 2>/dev/null); do echo "  http://$ip:8000"; done
fi
echo "(Ctrl+C to stop)"
exec python -m ground_station.server --host 0.0.0.0 --port 8000 --weights "$WEIGHTS" \
  --log logs/server_log.csv ${EXTRA[@]+"${EXTRA[@]}"}
