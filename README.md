# TinyML_HT2026: Adaptive Edge Processing for UAVs Based on Communication Link Quality

Course project (Tiny Machine Learning, Stockholm University, HT2026). Chengyi Zhou and Ting Chong Shuo Stanley.

**Research question:** Can a UAV (Raspberry Pi 5) adjust the complexity of its AI processing, such as image resolution or model depth, based on the quality of its link to a ground station?

## Repository layout

```
ground_station/      Remote high-complexity mode (runs on the Windows PC)
  detector.py          YOLOv8s wrapper (auto CUDA/CPU, thread-safe, timing)
  server.py            FastAPI server: /health, /ping, /detect; per-frame CSV log
  demo_push.py         optional status push to the live demo dashboard (--demo URL)
edge/                UAV side (Raspberry Pi 5)
  frame_source.py      folder / video / camera / picam frames; --realtime camera replay
  local_detector.py    edge mode: INT8 YOLOv8n (ONNX Runtime or LiteRT) on the Pi, same interface as remote
  link_monitor.py      background RTT / loss probes, remote-frame latency, RSSI
  policy.py            threshold policy (plan Table 1) with hysteresis and minimum dwell time
  adaptive.py          per-frame remote_full / remote_reduced / local selection + local fallback
  remote_client.py     JPEG encode + optional downscale, send, map boxes back, timing
  run_remote.py        CLI: any source -> server, CSV log + summary
  demo_link.py         live demo: link shaper controlled by the dashboard + frame reporter
demo/                Live demo instance (runs on the Mac, standard library only)
  server.py            dashboard server: receives UAV frames + GS status, link controls, trace playback
  tracker.py           IoU tracker (ids + trails) for the dashboard
  static/index.html    the dashboard (video + tracks, FPS / latency / policy, link emulator, charts)
tools/
  visdrone.py          VisDrone2019-VID annotations -> YOLO labels with COCO class ids
  mot.py               MOT17/MOT16/MOT20 pedestrian ground truth -> YOLO labels (COCO person)
  export_onnx_norm.py  YOLOv8 -> ONNX with 0..1 boxes (INT8-friendly output)
  make_calib_npy.py    calibration frames (.npy) for INT8 quantisation
  quantize_onnx.py     static INT8 quantisation with ONNX Runtime
  eval_detection.py    Precision / recall / F1 @ IoU vs YOLO-format labels (ignore regions, hold-last)
  summarize_log.py     Latency / bandwidth / reliability / drop-rate / mode-share table per setting
  link_emulator.py     replays a link trace with tc netem (only traffic to the ground station)
traces/                link-quality traces (simulated flights) for the emulator
tests/                 pytest (fake detector, no GPU needed); RUN_YOLO=1 for real model
scripts/               deploy scripts; run_experiment.sh = remote vs local vs adaptive on one trace
```

## Status

| Plan step | Status |
|---|---|
| Remote high-complexity detection pipeline (Oct 5-8) | Implemented |
| Lightweight TinyML edge pipeline (Oct 9-11) | Implemented (INT8 YOLOv8n, ONNX Runtime PTQ) |
| Link monitoring + threshold policy engine (Oct 12-15) | Implemented (RTT, loss, RSSI, remote latency; hysteresis) |
| Integration, experiments, analysis (Oct 16-29) | Adaptive mode + trace-driven experiment script ready; experiments next |
| Live demo | Dashboard on the Mac: live video + tracks, metrics, policy state, link emulator (keyboard / traces) |

## Quick start: one-click deploy

Each script creates `.venv`, installs dependencies (only again when the requirements file changes), and then starts the code.

| Machine | Command | What it does |
|---|---|---|
| Lab PC (Windows) | double-click `scripts\deploy_ground.bat` | venv, CUDA PyTorch if NVIDIA GPU, deps, quick tests, firewall rule, starts server on :8000 |
| Mac / Linux (home testing) | `./scripts/deploy_ground.sh` | same, uses MPS on Apple Silicon |
| Raspberry Pi 5 | `./scripts/deploy_edge.sh <PC-IP>` | apt packages, venv, downloads COCO128, checks server, runs pipeline, prints latency/bandwidth/reliability + precision/recall |

```bash
./scripts/deploy_edge.sh 192.168.31.50                                    # COCO128, full resolution
./scripts/deploy_edge.sh 192.168.31.50 coco128 --max-width 320 --quality 70   # reduced-resolution mode
./scripts/deploy_edge.sh 192.168.31.50 picam --max-frames 100             # Pi camera
./scripts/deploy_edge.sh 192.168.31.50 data/VisDrone2019-VID-val/sequences/uav0000086_00000_v --realtime   # drone video
```

Home test with only the Mac and the Pi: run `./scripts/deploy_ground.sh` on the Mac, then `./scripts/deploy_edge.sh <Mac-IP>` on the Pi.

## 1. Ground station (Windows PC), manual steps

```bat
cd TinyML_HT2026
scripts\setup_ground_windows.bat      :: venv, (CUDA) torch, deps, runs tests
scripts\run_ground_station.bat        :: starts on 0.0.0.0:8000 with yolov8s.pt
scripts\run_ground_station.bat yolov8m.pt   :: optional bigger model
```

`yolov8s.pt` downloads automatically the first time it is used. Allow Python through Windows Firewall on port 8000 (or run
`netsh advfirewall firewall add rule name="UAV GS" dir=in action=allow protocol=TCP localport=8000` as admin).

Quick check from any machine: open `http://<PC-IP>:8000/health` in a browser, or

```bash
curl -X POST "http://<PC-IP>:8000/detect" -H "Content-Type: image/jpeg" --data-binary @test.jpg
```

Response:

```json
{"frame_id": null, "model": "yolov8s.pt", "device": "cuda:0", "image_size": [640, 480],
 "detections": [{"class_id": 0, "label": "person", "confidence": 0.87, "box": [x1, y1, x2, y2]}],
 "timing": {"decode_ms": 2.1, "inference_ms": 11.4, "server_ms": 14.0}}
```

## 2. UAV / Raspberry Pi 5 client

```bash
pip install -r requirements-edge.txt          # + sudo apt install python3-picamera2 for the Pi camera
python -m edge.run_remote --server http://<PC-IP>:8000 --source data/images --save-json logs/remote_dets.json
python -m edge.run_remote --server http://<PC-IP>:8000 --source picam --max-frames 100
python -m edge.run_remote --server http://<PC-IP>:8000 --source data/images --max-width 320 --quality 70   # reduced-resolution remote mode
```

Each frame logs `encode_ms, network_ms, server_ms, inference_ms, e2e_ms, upload_bytes, rtt_ms, within_deadline` to `logs/remote_client_log.csv`.

## Video input and real-time replay (main experiments)

We have no Pi camera, so drone footage is replayed from disk. A COCO128 folder is a set of
unrelated photos; a drone sees a continuous video at a fixed frame rate. Two things matter:

1. **Temporal continuity.** Use a video file or an ordered frame sequence. Both work as
   `--source`; a folder is read in file-name order, so a VisDrone-VID sequence folder *is* a video.
2. **Real-time behaviour (`--realtime`).** Without it every frame is processed however long it
   takes, so a slow link shows up as high latency but zero lost frames. With it the source is
   replayed like a live camera at `--fps` (default: the video's own rate, 30 for a folder): the
   pipeline always gets the frame the camera is showing *now*, and frames that pass while it is
   busy are dropped, as on the real UAV. Extra log columns: `src_index`, `dropped` and `age_ms`
   (capture -> result available, always >= `e2e_ms`). The summary adds `drop_rate` and `processed_fps`.

Recommended dataset: **VisDrone2019-VID-val** (real UAV video, every frame annotated). Download it
from <https://github.com/VisDrone/VisDrone-Dataset> and unzip it into `data/`:

```bash
python -m tools.visdrone list data/VisDrone2019-VID-val                      # sequences, length, size
python -m tools.visdrone convert data/VisDrone2019-VID-val/sequences/uav0000086_00000_v   # -> data/visdrone_labels/...
python -m tools.visdrone convert <seq_dir> --video data/uav0000086.mp4        # optional: also an mp4

python -m edge.run_remote --server http://<PC-IP>:8000 --realtime --fps 30 \
    --source data/VisDrone2019-VID-val/sequences/uav0000086_00000_v --save-json logs/dets.json
python -m tools.eval_detection --pred logs/dets.json \
    --images data/VisDrone2019-VID-val/sequences/uav0000086_00000_v \
    --labels data/visdrone_labels/uav0000086_00000_v --classes 0,1,2,3,5,7 --hold-last
```

`deploy_edge.sh` does the conversion and evaluation automatically when the source is a VisDrone-VID
sequence folder.

Notes on scoring:
- **Class mapping VisDrone -> COCO:** pedestrian and people -> person, bicycle, car, van -> car,
  truck, bus, motor -> motorcycle. Tricycles, "others" and VisDrone's ignored regions become
  *ignore regions*: they are never required, and a prediction there is not a false positive.
  `--classes 0,1,2,3,5,7` scores only the COCO classes that exist in VisDrone.
- VisDrone has many tiny objects. `convert --min-size 8` turns objects under 8 px into ignore regions.
- `--hold-last` scores every frame of the sequence: frames dropped in real-time mode reuse the last
  processed result. That is what the UAV knows at each moment, so the score reflects both drops
  and misses. (It is slightly optimistic: in practice a result only arrives `age_ms` after capture.)
  Without `--hold-last`, only processed frames are scored.
- A plain mp4 without labels can still be used for latency, bandwidth and drop rate. To score a
  converted mp4, use `--labels ... --img-size 1344x756` instead of `--images`; video frames are
  named `0000001, 0000002, ...` like VisDrone frames.

## Second dataset: MOT17 (ground-level video, larger objects)

VisDrone is hard for the 320 px edge model: most objects are only a few pixels after resizing
(local recall 0.06, 0.13 when objects < 32 px are ignored). MOT17 is pedestrian video at street
level: real 30 fps (some 25/14 fps), every frame annotated, people tens to hundreds of pixels
tall. Pick a sparse sequence, e.g. **MOT17-09** (1080p, 30 fps, ~10 people per frame, static
camera) or MOT17-11 (~10.5 per frame, moving camera); MOT17-04 (~45 per frame) is crowded.

```bash
cd data && curl -L -C - -O https://motchallenge.net/data/MOT17.zip && unzip -q MOT17.zip && cd ..   # 5.5 GB
python -m tools.mot list data/MOT17/train                 # frames, fps, size, persons/frame, median height
./scripts/deploy_edge.sh local data/MOT17/train/MOT17-09-FRCNN --realtime
./scripts/run_experiment.sh <GS-IP> data/MOT17/train/MOT17-09-FRCNN traces/fly_away_and_back.csv
```

* MOT17 contains each video three times (`-DPM`, `-FRCNN`, `-SDP`: only the public detections
  differ); frames and ground truth are identical, use one of them.
* Labels are converted automatically (`data/mot_labels/<sequence>`) and scored as **person
  only** (`--classes 0`). As in the official MOTChallenge evaluation, only "considered"
  pedestrians count; persons on vehicles, static persons, distractors and reflections are
  ignore regions; vehicles and occluders too (not annotated consistently).
  `python -m tools.mot convert <img1> --min-visibility 0.25` also ignores heavily occluded people.
* Real-time replay uses the sequence's own frame rate from `seqinfo.ini` (25 fps for MOT17-13).
* Not a UAV view: it isolates the link / processing trade-off from the tiny-object problem
  of aerial video; the VisDrone runs remain the aerial reference.

## Edge mode: on-board INT8 model

`models/yolov8n_int8_320.onnx`: YOLOv8n (COCO, 320x320 input), static INT8 post-training
quantisation with ONNX Runtime (weights INT8 per channel, activations UINT8 calibrated on 100
VisDrone frames from a sequence that is *not* used for evaluation). 3.6 MB vs 12.7 MB float;
on 21 frames 86% of its detections match the float model at IoU 0.5, ~3x faster on CPU.

```bash
python -m tools.export_onnx_norm --weights yolov8n.pt --imgsz 320    # boxes in 0..1 (see file)
python -m tools.make_calib_npy data/uav0000117_02622_v --imgsz 320 --n 100
python -m tools.quantize_onnx --model yolov8n_norm_320.onnx --calib calib_320.npy
```

All convolutions run in INT8; only the detection head's decode arithmetic stays float.
(An Edge Impulse BYOM INT8 TFLite of the same model was unusable: its single INT8 output
tensor forced boxes and class scores into one scale, so either all scores or all boxes were
lost. `.tflite` models are still supported by `edge/local_detector.py`.)

Run it on the Pi (no ground station needed; `deploy_edge.sh` installs `onnxruntime`):

```bash
./scripts/deploy_edge.sh local data/VisDrone2019-VID-val/sequences/uav0000086_00000_v --realtime
python -m edge.run_remote --mode local --source <seq> --realtime
```

The log uses the same columns as remote mode (`network_ms` and `upload_bytes` are 0;
`inference_ms` is the model run time, `encode_ms` the letterbox step), so `summarize_log`
and `eval_detection` compare edge-only and remote runs directly.

## Adaptive mode: policy engine

`--mode adaptive` chooses, for every frame, between the ground station and the on-board model:

| Link (any of the "poor" conditions is enough) | Mode | What is sent |
|---|---|---|
| good: RTT < 100 ms, loss < 5 %, RSSI > -60 dBm | `remote_full` | full frame, JPEG 90, YOLOv8s on the GS |
| moderate: in between | `remote_reduced` | 640 px wide, JPEG 70 |
| poor: RTT > 250 ms or loss > 10 % or RSSI < -75 dBm | `local` | nothing; INT8 YOLOv8n on the Pi |

* **Measurements** (`edge/link_monitor.py`): a background thread probes `/ping` every 0.5 s
  (also while running locally, so recovery is noticed). RTT is an EWMA, a failed probe counts
  as a timeout. Loss = failed probes + failed remote frames in the last 5 s. RSSI comes from
  `/proc/net/wireless` (or from the link emulator, see below).
* **Latency rule**: RTT and RSSI do not see a bandwidth bottleneck, so if the median remote
  end-to-end time in the last 5 s exceeds the deadline (500 ms), the policy steps down a level.
* **Stability**: a worse link must last 1 s (`--down-hold`), a better one 3 s (`--up-hold`);
  at least 2 s between switches (`--min-dwell`); recovery goes up one level at a time.
* **Inference size per mode**: the ground station runs YOLO at `--full-imgsz` (default 1280)
  for full frames and `--reduced-imgsz` (default 640) for 640 px frames, sent with each request,
  so it never upscales a reduced frame and needs no restart. Fixed remote mode: `--imgsz 1280`.
  (MOT17-09: recall 0.71 at 640 vs 0.74 at 1280 for full frames; inference 62 -> 96 ms on the Mac.)
* **Fallback**: a remote request that fails or times out (timeout = deadline) is processed
  locally straight away, so every frame still gets a result (`--no-fallback` to disable).
* All thresholds are flags (`--rtt-good --rtt-poor --loss-good --loss-poor --rssi-good
  --rssi-poor`, `--no-rssi`, `--policy-modes remote_full,local` for a 2-level policy).

```bash
./scripts/deploy_edge.sh <GS-IP> data/VisDrone2019-VID-val/sequences/uav0000086_00000_v --realtime --mode adaptive
```

Extra log columns (all modes): `mode`, `fallback`, `switch` (reason, on the frame where the
mode changed), `link_rtt_ms`, `link_loss`, `link_e2e_ms`, `rssi_dbm`. `summarize_log` adds
the share of frames per mode, the number of switches and fallbacks, and the upload Mbit/s.

## Experiments: remote vs local vs adaptive on an emulated link

`tools/link_emulator.py` replays a trace (`traces/*.csv`: time, RSSI, delay, jitter, loss,
bandwidth) with `tc netem` on the Pi, shaping **only traffic to the ground station** (SSH keeps
working). netem cannot change the real Wi-Fi signal, so the trace RSSI is written to
`/tmp/uav_link_rssi`, which the link monitor uses instead while it is fresh.
`scripts/run_experiment.sh` runs all three strategies on the same sequence and trace and writes
`logs/exp_<trace>_<time>/summary.txt`:

```bash
./scripts/run_experiment.sh <GS-IP> data/VisDrone2019-VID-val/sequences/uav0000086_00000_v \
    traces/fly_away_and_back.csv            # needs sudo for tc; ~3 x 84 s
```

| Trace | What it tests |
|---|---|
| `fly_away_and_back.csv` (84 s) | good -> moderate -> poor -> moderate -> good, all metrics change together |
| `flaky_border.csv` (40 s) | RTT crossing the good/moderate border every 2 s: hysteresis must prevent flapping |
| `bandwidth_drop.csv` (65 s) | RSSI and RTT stay good, upload bandwidth collapses: only the latency rule reacts |

The sequence is looped (`--loop`) and the run stopped after the trace length (`--max-seconds`);
`eval_detection` scores looped runs per repetition. Note: shaping acts on the Pi's outgoing
packets, so `delay_ms` is added once per round trip and `rate_mbit` limits the upload.

## Live demo: dashboard with an adjustable link

Three processes: the **UAV** (Raspberry Pi, `edge.run_remote`), the **ground station**
(Mac or lab PC, `ground_station.server`) and the **demo instance** (`demo.server`, on the Mac).

```
 UAV (Pi) ──── frames, /ping (shaped) ────▶ ground station (YOLOv8s)
    │  ▲                                          │
    │  └── link settings (delay/loss/rate/RSSI) ──┐│ status (model, device, inference ms)
    └── every frame: mode, timings, link, policy, │▼
        detections + 640 px preview ─────▶ demo instance ◀── browser: http://localhost:8100
```

```bash
# 1. Mac: dashboard (opens the browser; prints the --demo URL to use)
./scripts/run_demo.sh                    # or: python -m demo.server --open
./scripts/run_demo.sh --gs               # ... and the ground station on the same Mac

# 2. ground station elsewhere (lab PC):  scripts\run_ground_station.bat yolov8s.pt http://<Mac-IP>:8100
#    or any machine:  python -m ground_station.server --demo http://<Mac-IP>:8100

# 3. Pi: adaptive pipeline, reporting to the dashboard
./scripts/deploy_edge.sh <GS-IP> data/uav0000117_02622_v --realtime --loop 100 \
    --mode adaptive --model models/yolov8n_int8_320.onnx --demo http://<Mac-IP>:8100
```

**What the dashboard shows:** the camera frame with tracked boxes (ids and trails from a
small IoU tracker on the demo side; the pipeline itself is detection-only), the current mode
and where inference ran (ground station or on board), processed vs camera FPS, end-to-end
latency, result age, inference and network time, uplink Mbit/s, drop rate, the policy's live
decision (what the link classifies as, the reason, and the hold / dwell timer before a switch),
mode share, switches and fallbacks, and 60 s charts of measured RTT (with the policy
thresholds), latency (with the deadline) and FPS, coloured by mode.

**Changing the link:** type a value, or select a metric (`1`–`5` / `←` `→`) and press `↑` `↓`
(`Shift` = ×5): extra delay, jitter, loss, uplink bandwidth, RSSI. Presets `G` good, `M`
moderate, `P` poor, `X` ground station down, `R` reset; `T` plays a trace from `traces/`
(the same flights as the netem experiments) in real time. The UAV fetches the settings
within ~0.25 s and applies them to its *real* traffic to the ground station
(`edge/demo_link.py`): delay is added to every `/ping` and `/detect`, a lost request waits
for the timeout, the upload takes `bytes / rate`, and the RSSI replaces the Wi-Fi reading.
The policy engine is never told the settings; it reacts to what the link monitor measures,
like with `tools.link_emulator` - but without root and on any OS. The demo traffic itself
is not shaped. With `--no-demo-shaper` the UAV only reports (e.g. when shaping with netem).

Notes: the dashboard needs only the Python standard library; allow incoming connections for
Python on the Mac (port 8100). Preview size/rate: `--demo-width 640 --demo-fps 15`. With a
bandwidth limit, pings stay fast while frames time out, so the policy goes to on-board via
loss and may try `remote_reduced` again when probes look good: that is the latency-rule case
from `bandwidth_drop.csv`, visible live on the dashboard.

## 3. Evaluation

```bash
python -m tools.summarize_log logs/remote_client_log.csv --deadline-ms 500
python -m tools.eval_detection --pred logs/remote_dets.json --images <images_dir> --labels <labels_dir>
```

To test with a labelled set, `coco128` from Ultralytics works: `images/train2017` plus YOLO-format `labels/train2017`.

## Tests

```bash
python -m pytest tests -q            # fast, fake detector
set RUN_YOLO=1 && python -m pytest tests -q   # Windows: also runs the real YOLOv8s smoke test
```
