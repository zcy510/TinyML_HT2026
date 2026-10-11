"""Ground-station HTTP server for remote high-complexity object detection.

The UAV (Raspberry Pi 5) sends a JPEG/PNG frame, the server runs YOLOv8s (or
any other Ultralytics model) and returns the detections plus timing data.

Endpoints
---------
GET  /health   -> model / device info (used by the client at start-up)
GET  /ping     -> tiny response, used by the UAV to measure RTT for the policy engine
POST /detect   -> run detection. Body: raw image bytes (Content-Type: image/jpeg)
                  or multipart form-data with field "file".
                  Optional query params: conf, imgsz, frame_id

Run (Windows, from the repo root):
    python -m ground_station.server --host 0.0.0.0 --port 8000 --weights yolov8s.pt
"""
from __future__ import annotations

import argparse
import csv
import os
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

LOG_FIELDS = [
    "timestamp", "frame_id", "client", "bytes_received", "width", "height",
    "num_detections", "decode_ms", "inference_ms", "server_ms", "model", "device",
]


class CsvLogger:
    """Appends one row per processed frame (used later for evaluation)."""

    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            if not os.path.exists(path):
                with open(path, "w", newline="") as f:
                    csv.DictWriter(f, LOG_FIELDS).writeheader()

    def log(self, row: dict) -> None:
        if not self.path:
            return
        with self._lock, open(self.path, "a", newline="") as f:
            csv.DictWriter(f, LOG_FIELDS).writerow(row)


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


def create_app(detector, log_path: Optional[str] = None) -> FastAPI:
    """Build the FastAPI app. `detector` needs .detect(img, conf=, imgsz=),
    .model_name and .device (a fake detector is injected in the unit tests)."""
    app = FastAPI(title="UAV Ground Station - Remote Detection")
    logger = CsvLogger(log_path)
    stats = {"frames": 0, "started": time.time()}

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": detector.model_name,
            "device": detector.device,
            "frames_processed": stats["frames"],
            "uptime_s": round(time.time() - stats["started"], 1),
        }

    @app.get("/ping")
    def ping():
        return {"t": time.time()}

    @app.post("/detect")
    async def detect(
        request: Request,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        frame_id: Optional[str] = None,
    ):
        t_start = time.perf_counter()
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                raise HTTPException(400, "multipart request needs a 'file' field")
            data = await upload.read()
        else:
            data = await request.body()
        if not data:
            raise HTTPException(400, "empty body")

        t_dec = time.perf_counter()
        try:
            img = decode_image(data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        decode_ms = (time.perf_counter() - t_dec) * 1000.0

        result = detector.detect(img, conf=conf, imgsz=imgsz)
        server_ms = (time.perf_counter() - t_start) * 1000.0
        stats["frames"] += 1

        h, w = img.shape[:2]
        logger.log({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame_id": frame_id or "",
            "client": request.client.host if request.client else "",
            "bytes_received": len(data),
            "width": w, "height": h,
            "num_detections": len(result.detections),
            "decode_ms": round(decode_ms, 2),
            "inference_ms": round(result.inference_ms, 2),
            "server_ms": round(server_ms, 2),
            "model": result.model_name, "device": result.device,
        })
        return JSONResponse({
            "frame_id": frame_id,
            "model": result.model_name,
            "device": result.device,
            "image_size": [w, h],
            "detections": [d.to_dict() for d in result.detections],
            "timing": {
                "decode_ms": round(decode_ms, 2),
                "inference_ms": round(result.inference_ms, 2),
                "server_ms": round(server_ms, 2),
            },
        })

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Ground-station remote detection server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--weights", default="yolov8s.pt",
                   help="Ultralytics weights, e.g. yolov8s.pt, yolov8m.pt, yolo11s.pt")
    p.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--half", action="store_true", help="FP16 inference on CUDA")
    p.add_argument("--log", default="logs/server_log.csv")
    args = p.parse_args()

    import uvicorn

    from ground_station.detector import YoloDetector

    print(f"[ground-station] loading {args.weights} ...")
    det = YoloDetector(args.weights, args.device, args.imgsz, args.conf, half=args.half)
    det.warmup()
    print(f"[ground-station] model ready on {det.device}; listening on {args.host}:{args.port}")
    uvicorn.run(create_app(det, args.log), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
