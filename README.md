# TinyML_HT2026: Adaptive Edge Processing for UAVs Based on Communication Link Quality

Course project (Tiny Machine Learning, Stockholm University, HT2026). Chengyi Zhou and Ting Chong Shuo Stanley.

**Research question:** Can a UAV (Raspberry Pi 5) adjust the complexity of its AI processing, such as image resolution or model depth, based on the quality of its link to a ground station?

## Repository layout

```
ground_station/      Remote high-complexity mode (runs on the Windows PC)
  detector.py          YOLOv8s wrapper (auto CUDA/CPU, thread-safe, timing)
  server.py            FastAPI server: /health, /ping, /detect; per-frame CSV log
edge/                UAV side (Raspberry Pi 5)
  frame_source.py      folder / video / camera / picam frames; --realtime camera replay
  local_detector.py    edge mode: INT8 YOLOv8n LiteRT/TFLite on the Pi (same interface as remote)
  remote_client.py     JPEG encode + optional downscale, send, map boxes back, timing
  run_remote.py        CLI: any source -> server, CSV log + summary
tools/
  visdrone.py          VisDrone2019-VID annotations -> YOLO labels with COCO class ids
  make_calib_npy.py    representative .npy for INT8 quantisation (Edge Impulse BYOM)
  eval_detection.py    Precision / recall / F1 @ IoU vs YOLO-format labels (ignore regions, hold-last)
  summarize_log.py     Latency / bandwidth / reliability / drop-rate table per setting
tests/                 pytest (fake detector, no GPU needed); RUN_YOLO=1 for real model
scripts/               Windows setup + start scripts for the ground station
```

## Status

| Plan step | Status |
|---|---|
| Remote high-complexity detection pipeline (Oct 5-8) | Implemented |
| Lightweight TinyML edge pipeline (Oct 9-11) | Implemented (INT8 YOLOv8n via Edge Impulse) |
| Link monitoring + threshold policy engine (Oct 12-15) | Planned (`/ping` RTT already in place) |
| Integration, experiments, analysis (Oct 16-29) | Planned |

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

## Edge mode: on-board INT8 model

`models/yolov8n_int8_320.tflite`: YOLOv8n (COCO, 320x320 input), exported to ONNX and quantised
to INT8 with Edge Impulse BYOM:

```bash
yolo export model=yolov8n.pt format=onnx imgsz=320 opset=12 simplify=True
python -m tools.make_calib_npy data/uav0000117_02622_v --imgsz 320 --n 100   # calibration frames
# Edge Impulse: Upload your model (yolov8n.onnx + calib_320.npy); input image, pixels 0..1,
# resize "Fit longest axis"; output Object detection, "YOLOv11 (coordinates in absolute values)",
# 80 COCO labels, score threshold 0.25. Download the LiteRT/TFLite int8 model.
```

Run it on the Pi (no ground station needed; `deploy_edge.sh` installs `ai-edge-litert`):

```bash
./scripts/deploy_edge.sh local data/VisDrone2019-VID-val/sequences/uav0000086_00000_v --realtime
python -m edge.run_remote --mode local --model models/yolov8n_int8_320.tflite --source <seq> --realtime
```

The log uses the same columns as remote mode (`network_ms` and `upload_bytes` are 0;
`inference_ms` is the model invoke time, `encode_ms` the letterbox/quantise step), so
`summarize_log` and `eval_detection` compare edge-only and remote runs directly.

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
