#!/usr/bin/env bash
# =============================================================================
# Compare remote-only, edge-only and adaptive processing under the SAME emulated
# link (plan objective 4). Run on the Raspberry Pi, with the ground station running.
#
#   ./scripts/run_experiment.sh <ground-station-IP> <sequence-dir> <trace.csv> [strategies]
#
#   strategies   comma list, default "remote,local,adaptive"
#   env vars     MODEL=models/yolov8n_int8_320.tflite  IFACE=wlan0  FPS=30
#                EXTRA="--no-rssi"   (extra args for edge.run_remote)
#
# Example
#   ./scripts/run_experiment.sh 192.168.31.182 \
#       data/VisDrone2019-VID-val/sequences/uav0000086_00000_v traces/fly_away_and_back.csv
#
# For each strategy: start tools.link_emulator (sudo, tc netem on traffic to the
# ground station only) -> run the pipeline in real-time mode for exactly the trace
# length (the sequence is looped) -> stop shaping -> score accuracy. Results go to
# logs/exp_<trace>_<time>/: one CSV log + detections per strategy, the applied trace,
# and summary.txt with the comparison table.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [ $# -lt 3 ]; then sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 1; fi
GS="$1"; SEQ="${2%/}"; TRACE="$3"; STRATS="${4:-remote,local,adaptive}"
MODEL="${MODEL:-models/yolov8n_int8_320.tflite}"; IFACE="${IFACE:-wlan0}"; FPS="${FPS:-30}"
EXTRA="${EXTRA:-}"
[[ "$GS" == *://* ]] || GS="http://$GS"; [[ "$GS" =~ :[0-9]+$ ]] || GS="$GS:8000"
GS_IP="$(echo "$GS" | sed -E 's#^[a-z]+://##; s#:[0-9]+$##')"

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "run ./scripts/deploy_edge.sh once first (creates .venv)"; exit 1; }
[ -d "$SEQ" ] || { echo "sequence not found: $SEQ"; exit 1; }
[ -f "$TRACE" ] || { echo "trace not found: $TRACE"; exit 1; }
[ -f "$MODEL" ] || { echo "local model not found: $MODEL"; exit 1; }
command -v tc >/dev/null || { echo "tc not found: sudo apt install iproute2"; exit 1; }

# trace length and how often the sequence must loop to cover it
DUR="$("$PY" -c "import sys; sys.path.insert(0,'.'); from tools.link_emulator import load_trace; print(load_trace('$TRACE')[-1].t_s)")"
NFR="$(ls "$SEQ" | grep -ciE '\.(jpg|jpeg|png)$')"
LOOPS="$("$PY" -c "import math; print(math.ceil($DUR * $FPS / $NFR) + 1)")"

# labels for scoring (VisDrone-VID layout)
SEQ_NAME="$(basename "$SEQ")"
ANN="$(cd "$SEQ/../.." && pwd)/annotations/$SEQ_NAME.txt"
LABELS=""
if [ -f "$ANN" ]; then
  LABELS="$ROOT/data/visdrone_labels/$SEQ_NAME"
  [ -d "$LABELS" ] || "$PY" -m tools.visdrone convert "$SEQ" --out "$LABELS" >/dev/null
fi

OUT="$ROOT/logs/exp_$(basename "$TRACE" .csv)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
cp "$TRACE" "$OUT/trace.csv"
echo "[exp] trace $(basename "$TRACE"): ${DUR}s | sequence $SEQ_NAME: $NFR frames, looped $LOOPS x | -> $OUT"
echo "[exp] sudo is needed for tc (link shaping):"
sudo -v

cleanup() { sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true; }
trap cleanup EXIT

curl -fsS -m 3 "$GS/health" >/dev/null || { echo "ground station not reachable at $GS"; exit 1; }

IFS=',' read -ra LIST <<< "$STRATS"
for S in "${LIST[@]}"; do
  echo; echo "================ $S ================"
  sudo "$PY" -m tools.link_emulator --trace "$TRACE" --ground "$GS_IP" --iface "$IFACE" \
       --log "$OUT/trace_applied_$S.csv" &
  EMU=$!
  sleep 1                                           # shaping in place before the first frame
  # shellcheck disable=SC2086
  "$PY" -m edge.run_remote --mode "$S" --server "$GS" --model "$MODEL" --source "$SEQ" \
       --realtime --fps "$FPS" --loop "$LOOPS" --max-seconds "$DUR" \
       --log "$OUT/$S.csv" --save-json "$OUT/${S}_dets.json" $EXTRA \
       | grep -vE '^\[uav\] +[0-9]' || true          # keep policy switches + summary, drop per-frame lines
  sudo kill -TERM "$EMU" 2>/dev/null || true
  wait "$EMU" 2>/dev/null || true
  cleanup
  sleep 2                                           # let the link settle between strategies
done

{
  echo "Trace: $(basename "$TRACE") (${DUR}s)   Sequence: $SEQ_NAME   Model: $(basename "$MODEL")"
  echo
  LOGS=(); for S in "${LIST[@]}"; do [ -f "$OUT/$S.csv" ] && LOGS+=("$OUT/$S.csv"); done
  "$PY" -m tools.summarize_log "${LOGS[@]}" --deadline-ms 500
  if [ -n "$LABELS" ]; then
    echo; echo "Accuracy over every frame of the run (dropped frames reuse the last result, IoU 0.5):"
    for S in "${LIST[@]}"; do
      R="$("$PY" -m tools.eval_detection --pred "$OUT/${S}_dets.json" --images "$SEQ" --labels "$LABELS" \
            --classes 0,1,2,3,5,7 --hold-last | awk '{printf "%s=%s  ", $1, $2}')"
      echo "  $S: $R"
    done
  fi
} | tee "$OUT/summary.txt"
echo; echo "[exp] done: $OUT/summary.txt"
