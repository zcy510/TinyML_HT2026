# TinyML_HT2026: Adaptive Edge Processing for UAVs Based on Communication Link Quality

Course project (Tiny Machine Learning, Stockholm University, HT2026). Chengyi Zhou and Ting Chong Shuo Stanley.

**Research question:** Can a UAV (Raspberry Pi 5) adjust the complexity of its AI processing, such as image resolution or model depth, based on the quality of its link to a ground station?

## Repository layout

```
ground_station/      Remote high-complexity mode (runs on the Windows PC)
  detector.py          YOLOv8s wrapper (auto CUDA/CPU, thread-safe, timing)
  server.py            FastAPI server: /health, /ping, /detect; per-frame CSV log
edge/                UAV side (Raspberry Pi 5)
  remote_client.py     JPEG encode + optional downscale, send, map boxes back, timing
  run_remote.py        CLI: folder / video / camera / picam -> server, CSV log + summary
tools/
  eval_detection.py    Precision / recall / F1 @ IoU vs YOLO-format labels
  summarize_log.py     Latency / bandwidth / reliability table per resolution
tests/                 pytest (fake detector, no GPU needed); RUN_YOLO=1 for real model
scripts/               Windows setup + start scripts for the ground station
```

## Status

| Plan step | Status |
|---|---|
| Remote high-complexity detection pipeline (Oct 5-8) | Implemented |
| Lightweight TinyML edge pipeline (Oct 9-11) | Next |
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
