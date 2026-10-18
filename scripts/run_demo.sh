#!/usr/bin/env bash
# =============================================================================
# Start the live demo dashboard (demo instance) on the Mac.
#
#   ./scripts/run_demo.sh          dashboard on port 8100, opens the browser
#   ./scripts/run_demo.sh --gs     also start the ground station on this Mac
#                                  (YOLOv8s on MPS), reporting to the dashboard
#   PORT=8200 ./scripts/run_demo.sh
#
# Then on the Raspberry Pi (see README, "Live demo"):
#   ./scripts/deploy_edge.sh <GS-IP> data/uav0000117_02622_v --realtime --loop 100 \
#       --mode adaptive --model models/yolov8n_int8_320.onnx --demo http://<Mac-IP>:8100
# The dashboard itself needs only the Python standard library.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8100}"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

if [ "${1:-}" = "--gs" ]; then
  shift
  echo "[run-demo] starting the ground station on this Mac (reports to the dashboard)"
  SKIP_TESTS=1 ./scripts/deploy_ground.sh yolov8s.pt --demo "http://127.0.0.1:$PORT" "$@" &
  GS_PID=$!
  trap 'kill $GS_PID 2>/dev/null || true' EXIT
fi

"$PY" -m demo.server --port "$PORT" --open     # Ctrl+C stops it (and the ground station)
