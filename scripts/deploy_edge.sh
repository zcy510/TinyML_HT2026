#!/usr/bin/env bash
# =============================================================================
# One-click deploy + run for the UAV side (Raspberry Pi 5; also works on Mac/Linux
# for testing).
#
#   ./scripts/deploy_edge.sh <ground-station> [source] [extra run_remote args...]
#
#   <ground-station>  192.168.31.50  |  192.168.31.50:8000  |  http://host:8000
#                     mock  = test the Pi alone: starts a fake ground station on
#                             the Pi itself (no model, returns no detections)
#                     local = edge mode: run the INT8 model on the Pi itself
#                             (default models/yolov8n_int8_320.tflite; --model X)
#   [source]         coco128 (default: downloads 128 labelled COCO images and
#                     also evaluates precision/recall) | picam | 0 (USB cam) |
#                     path/to/folder | video.mp4 |
#                     a VisDrone-VID sequence folder (.../sequences/<name>):
#                     labels are converted automatically and accuracy is scored
#   extra args        anything edge.run_remote accepts, e.g.
#                     --max-width 320 --quality 70 --max-frames 50
#                     --realtime [--fps 30]   replay video/sequence like a live
#                                             camera, dropping late frames
#
# Examples
#   ./scripts/deploy_edge.sh mock                      # Pi-only smoke test
#   ./scripts/deploy_edge.sh mock picam --max-frames 20
#   ./scripts/deploy_edge.sh 192.168.31.50
#   ./scripts/deploy_edge.sh 192.168.31.50 picam --max-frames 100
#   ./scripts/deploy_edge.sh 192.168.31.50 coco128 --max-width 320 --quality 70
#   ./scripts/deploy_edge.sh 192.168.31.50 \
#       data/VisDrone2019-VID-val/sequences/uav0000086_00000_v --realtime
#   ./scripts/deploy_edge.sh local \
#       data/VisDrone2019-VID-val/sequences/uav0000086_00000_v --realtime
#
# What it does: system packages (Pi only) -> .venv -> Python deps (only when
# requirements change) -> checks the ground station -> runs the remote pipeline
# -> prints latency / bandwidth / reliability (+ accuracy for coco128 and VisDrone-VID).
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV="$ROOT/.venv"
REQ="$ROOT/requirements-edge.txt"
COCO_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"

say()  { printf '\n\033[1;34m[deploy-edge]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deploy-edge] WARNING:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[deploy-edge] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ arguments
if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  sed -n '2,37p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi
SERVER="$1"; shift
SOURCE="coco128"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then SOURCE="$1"; shift; fi
EXTRA=("$@")

MOCK=0
MOCK_PORT=8765
LOCAL=0
if [ "$SERVER" = "local" ]; then
  LOCAL=1
elif [ "$SERVER" = "mock" ]; then
  MOCK=1
  SERVER="http://127.0.0.1:$MOCK_PORT"
fi

# normalise "192.168.31.50" -> "http://192.168.31.50:8000"
if [ "$LOCAL" = 0 ]; then
  [[ "$SERVER" == *://* ]] || SERVER="http://$SERVER"
  SERVER="${SERVER%/}"
  [[ "$SERVER" =~ :[0-9]+$ ]] || SERVER="$SERVER:8000"
fi

IS_PI=0
if [ -r /proc/device-tree/model ] && grep -qa "Raspberry Pi" /proc/device-tree/model; then
  IS_PI=1
fi

# ------------------------------------------------- 1. system packages (Pi only)
# On the Pi, numpy/OpenCV/picamera2 come from apt so they match each other
# (pip's OpenCV wheel pulls numpy 2.x, which breaks the apt-built picamera2).
if [ "$IS_PI" = 1 ]; then
  say "1/5 Raspberry Pi detected: checking system packages"
  PKGS=(python3-venv python3-pip python3-numpy python3-opencv python3-requests python3-pandas python3-picamera2)
  MISSING=()
  for p in "${PKGS[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p")
  done
  if [ ${#MISSING[@]} -gt 0 ]; then
    say "Installing: ${MISSING[*]} (needs sudo password)"
    sudo apt-get update
    sudo apt-get install -y "${MISSING[@]}" || warn "some apt packages failed; continuing"
  else
    echo "all system packages present"
  fi
else
  say "1/5 Not a Raspberry Pi: skipping apt packages"
fi

# ------------------------------------------------------- 2. virtual environment
command -v python3 >/dev/null || die "python3 not found"
if [ ! -x "$VENV/bin/python" ]; then
  say "2/5 Creating virtual environment in .venv"
  if [ "$IS_PI" = 1 ]; then
    python3 -m venv --system-site-packages "$VENV"   # see apt packages above
  else
    python3 -m venv "$VENV"
  fi
else
  say "2/5 Using existing .venv"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --------------------------------------------- 3. Python deps (only if changed)
if [ "$IS_PI" = 1 ]; then
  say "3/5 Python deps provided by apt on the Pi"
  python -c "import cv2, numpy, requests; print('cv2', cv2.__version__, '| numpy', numpy.__version__)" \
    || die "OpenCV/numpy/requests not importable - check the apt step above"
elif cmp -s "$REQ" "$VENV/.installed-requirements-edge.txt"; then
  say "3/5 Python deps up to date"
else
  say "3/5 Installing Python deps from requirements-edge.txt"
  python -m pip install --upgrade pip
  python -m pip install -r "$REQ"
  cp "$REQ" "$VENV/.installed-requirements-edge.txt"
fi

# LiteRT runtime for the on-board model (edge mode). numpy is pinned to the version
# already installed, so the apt-built OpenCV/picamera2 keep working on the Pi.
if [ "$LOCAL" = 1 ] && ! python -c "import ai_edge_litert" 2>/dev/null \
   && ! python -c "import tflite_runtime" 2>/dev/null; then
  say "Installing LiteRT runtime (ai-edge-litert) for local inference"
  NP_VER="$(python -c 'import numpy; print(numpy.__version__)')"
  python -m pip install ai-edge-litert "numpy==$NP_VER" \
    || python -m pip install tflite-runtime "numpy==$NP_VER" \
    || die "could not install a LiteRT/TFLite runtime for this Python version"
fi

# ------------------------------------------------------------ test data / source
LABELS=""
if [ "$SOURCE" = "coco128" ]; then
  if [ ! -d "$ROOT/data/coco128/images/train2017" ]; then
    say "Downloading COCO128 test images (~7 MB)"
    mkdir -p "$ROOT/data"
    if command -v curl >/dev/null; then
      curl -fL --retry 3 -o "$ROOT/data/coco128.zip" "$COCO_URL"
    else
      wget -O "$ROOT/data/coco128.zip" "$COCO_URL"
    fi
    python -m zipfile -e "$ROOT/data/coco128.zip" "$ROOT/data/"
    rm -f "$ROOT/data/coco128.zip"
  fi
  SOURCE="$ROOT/data/coco128/images/train2017"
  LABELS="$ROOT/data/coco128/labels/train2017"
fi

# VisDrone-VID sequence: .../sequences/<name> with .../annotations/<name>.txt
EVAL_ARGS=()
if [ -d "$SOURCE" ]; then
  SEQ_NAME="$(basename "$SOURCE")"
  VD_ANN="$(cd "$SOURCE/../.." && pwd)/annotations/$SEQ_NAME.txt"
  if [ -f "$VD_ANN" ]; then
    LABELS="$ROOT/data/visdrone_labels/$SEQ_NAME"
    if [ ! -d "$LABELS" ]; then
      say "VisDrone-VID sequence detected: converting labels to COCO classes"
      python -m tools.visdrone convert "$SOURCE" --out "$LABELS"
    fi
    EVAL_ARGS=(--classes 0,1,2,3,5,7)
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do
      [ "$a" = "--realtime" ] && EVAL_ARGS+=(--hold-last)
    done
  fi
fi

# ------------------------------------------- mock mode: fake GS on this device
if [ "$MOCK" = 1 ]; then
  say "MOCK MODE: starting a fake ground station on this device (no model)"
  python -m ground_station.mock_server --host 127.0.0.1 --port "$MOCK_PORT" &
  MOCK_PID=$!
  trap 'kill $MOCK_PID 2>/dev/null || true' EXIT
  python - "$SERVER" <<'PY' || die "mock ground station did not start"
import sys, time, requests
for _ in range(60):                      # wait up to ~15 s (OpenCV import is slow on the Pi)
    try:
        requests.get(sys.argv[1] + "/health", timeout=1); sys.exit(0)
    except Exception:
        time.sleep(0.25)
sys.exit(1)
PY
  LABELS=""   # no real detections -> accuracy would be meaningless
fi

# ------------------------------------------------------ 4. ground station check
if [ "$LOCAL" = 1 ]; then
  say "4/5 Local mode: no ground station needed"
  RUN_TARGET=(--mode local)
else
RUN_TARGET=(--server "$SERVER")
say "4/5 Checking ground station at $SERVER"
if ! python - "$SERVER" <<'PY'
import sys, requests
try:
    r = requests.get(sys.argv[1] + "/health", timeout=5)
    r.raise_for_status()
    info = r.json()
    print(f"ground station OK: model={info['model']} device={info['device']}")
except Exception as e:
    print(f"cannot reach ground station: {e}")
    sys.exit(1)
PY
then
  die "Ground station not reachable. Is the server running, on the same network,
       and is port 8000 allowed through the PC firewall?"
fi
fi

# ---------------------------------------------------------------- 5. run it
mkdir -p "$ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/remote_client_${STAMP}.csv"
DETS="$ROOT/logs/remote_dets_${STAMP}.json"

say "5/5 Running pipeline ($([ "$LOCAL" = 1 ] && echo local || echo remote) mode, source: $SOURCE)"
python -m edge.run_remote "${RUN_TARGET[@]}" --source "$SOURCE" \
  --log "$LOG" --save-json "$DETS" ${EXTRA[@]+"${EXTRA[@]}"}

say "Summary per setting (max width + JPEG quality)"
python -m tools.summarize_log "$LOG" || warn "summary needs pandas"

if [ -n "$LABELS" ]; then
  say "Detection accuracy vs ground-truth labels (IoU 0.5)"
  python -m tools.eval_detection --pred "$DETS" --images "$SOURCE" --labels "$LABELS" \
    ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}
fi

if [ "$MOCK" = 1 ]; then
  say "Mock run finished: capture, encoding, transfer and logging on the Pi all work."
  echo "Latency here is Pi-local (no Wi-Fi, no model); detections are empty by design."
fi
say "Done. Log: logs/$(basename "$LOG")   Detections: logs/$(basename "$DETS")"
echo "To keep using this environment in your shell:  source .venv/bin/activate"
