"""Run the remote high-complexity pipeline from the UAV (Raspberry Pi 5) side.

Examples
--------
# images in a folder (e.g. COCO128), every frame processed (offline evaluation)
python -m edge.run_remote --server http://192.168.1.50:8000 --source data/images

# a video, or a VisDrone-VID frame sequence, replayed like a live 30 fps camera:
# frames that go by while one is being processed are dropped (see edge/frame_source.py)
python -m edge.run_remote --server http://192.168.1.50:8000 --source data/flight.mp4 --realtime
python -m edge.run_remote --server http://192.168.1.50:8000 \
    --source data/VisDrone2019-VID-val/sequences/uav0000086_00000_v --realtime --fps 30

# USB camera / Pi camera, 50 frames, reduced resolution (moderate-link mode)
python -m edge.run_remote --server http://192.168.1.50:8000 --source 0 --max-frames 50 --max-width 320
python -m edge.run_remote --server http://192.168.1.50:8000 --source picam

# edge mode: on-board INT8 YOLOv8n, no ground station needed
python -m edge.run_remote --mode local --model models/yolov8n_int8_320.tflite \
    --source data/VisDrone2019-VID-val/sequences/uav0000086_00000_v --realtime

Writes one CSV row per frame (default logs/remote_client_log.csv) and prints a
summary: latency percentiles, bytes/frame, and reliability (= share of frames
with a valid result within --deadline-ms); in --realtime mode also the share of
camera frames dropped, the processed frame rate and the age of each result
(capture -> result available).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from edge.frame_source import iter_frames
from edge.remote_client import RemoteDetector, draw_detections

LOG_FIELDS = [
    "timestamp", "frame_id", "setting", "ok", "error", "orig_w", "orig_h", "sent_w", "sent_h",
    "jpeg_quality", "upload_bytes", "num_detections", "encode_ms", "network_ms",
    "server_ms", "inference_ms", "e2e_ms", "rtt_ms", "within_deadline",
    "src_index", "dropped", "age_ms",
]


def open_log(path: str):
    """Append to an existing CSV log, or start a new one. If the existing file was
    written with different columns (older version), start a new file next to it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        with open(path, newline="") as f:
            header = f.readline().strip().split(",")
        if header != LOG_FIELDS:
            root, ext = os.path.splitext(path)
            path = f"{root}_{datetime.now():%Y%m%d_%H%M%S}{ext}"
            print(f"[uav] existing log has older columns, writing to {path} instead")
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, LOG_FIELDS)
    if new_file:
        writer.writeheader()
    return f, writer


def summarize(rows, deadline_ms: float, wall_s: float = 0.0) -> dict:
    if not rows:
        return {}
    ok = [r for r in rows if r["ok"]]
    e2e = np.array([r["e2e_ms"] for r in ok]) if ok else np.array([np.nan])
    age = np.array([r["age_ms"] for r in ok]) if ok else np.array([np.nan])
    dropped = sum(r["dropped"] for r in rows)
    out = {
        "frames": len(rows),
        "success_rate": len(ok) / len(rows),
        "reliability_within_deadline": sum(r["within_deadline"] for r in rows) / len(rows),
        "deadline_ms": deadline_ms,
        "e2e_ms_mean": float(np.nanmean(e2e)),
        "e2e_ms_p50": float(np.nanpercentile(e2e, 50)),
        "e2e_ms_p95": float(np.nanpercentile(e2e, 95)),
        "age_ms_mean": float(np.nanmean(age)),
        "age_ms_p95": float(np.nanpercentile(age, 95)),
        "inference_ms_mean": float(np.mean([r["inference_ms"] for r in ok])) if ok else None,
        "network_ms_mean": float(np.mean([r["network_ms"] for r in ok])) if ok else None,
        "upload_kb_per_frame": float(np.mean([r["upload_bytes"] for r in rows]) / 1024),
        "detections_per_frame": float(np.mean([r["num_detections"] for r in ok])) if ok else None,
        "frames_dropped": dropped,
        "drop_rate": dropped / (dropped + len(rows)),
    }
    if wall_s > 0:
        out["processed_fps"] = len(rows) / wall_s
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="UAV -> ground station remote detection")
    ap.add_argument("--mode", choices=["remote", "local"], default="remote",
                    help="remote = ground station (needs --server); local = on-board INT8 "
                         "model (needs --model)")
    ap.add_argument("--server", default=None, help="e.g. http://192.168.1.50:8000")
    ap.add_argument("--model", default="models/yolov8n_int8_320.tflite",
                    help="LiteRT/TFLite model for --mode local")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for --mode local")
    ap.add_argument("--source", required=True, help="folder | image | video | camera index | picam")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--max-width", type=int, default=None,
                    help="downscale before sending (e.g. 320 for reduced-resolution mode)")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality 1-100")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout in seconds")
    ap.add_argument("--deadline-ms", type=float, default=500.0)
    ap.add_argument("--ping-every", type=int, default=10, help="measure RTT every N frames (0=off)")
    ap.add_argument("--realtime", action="store_true",
                    help="replay a video / frame folder like a live camera: drop the frames "
                         "that go by while one is being processed")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="with --realtime: camera frame rate (0 = video's own rate, 30 for a "
                         "folder); without: throttle to this rate (0 = as fast as possible)")
    ap.add_argument("--log", default="logs/remote_client_log.csv")
    ap.add_argument("--save-json", default=None, help="write all detections to this JSON file")
    ap.add_argument("--save-vis", default=None, help="folder for annotated images")
    args = ap.parse_args()

    if args.mode == "local":
        from edge.local_detector import LocalDetector

        client = LocalDetector(args.model, conf=args.conf or 0.25, num_threads=args.threads)
        info = client.health()
        print(f"[uav] local mode: model={info['model']} ({info['device']}, "
              f"input {client.imgsz}x{client.imgsz})")
    else:
        if not args.server:
            ap.error("--server is required in remote mode")
        client = RemoteDetector(args.server, args.timeout)
        try:
            info = client.health()
            print(f"[uav] connected: model={info['model']} device={info['device']}")
        except Exception as e:
            sys.exit(f"[uav] ground station not reachable at {args.server}: {e}")

    if args.save_vis:
        os.makedirs(args.save_vis, exist_ok=True)
    logf, writer = open_log(args.log)

    rows, all_dets, rtt = [], {}, None
    period = 1.0 / args.fps if args.fps > 0 and not args.realtime else 0.0
    if args.mode == "local":                                    # e.g. "local_yolov8n_int8_320"
        setting = "local_" + os.path.splitext(os.path.basename(args.model))[0]
    else:                                                       # e.g. "wfull_q90", "w320_q70"
        setting = f"w{args.max_width or 'full'}_q{args.quality}"
    if args.realtime:
        setting += "_rt"
        fps = args.fps or "video's own (30 for a folder)"
        print(f"[uav] real-time replay at {fps} fps: frames missed while busy are dropped")
    t_start = time.perf_counter()
    for i, fr in enumerate(iter_frames(args.source, args.max_frames, args.realtime, args.fps)):
        t_loop = time.perf_counter()
        frame, fid = fr.image, fr.frame_id
        if args.mode == "remote" and args.ping_every and i % args.ping_every == 0:
            rtt = client.ping()
        res = client.detect(frame, fid, args.max_width, args.quality, args.conf)
        age_ms = (time.perf_counter() - fr.capture_t) * 1000.0
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame_id": fid, "setting": setting, "ok": res.ok, "error": res.error,
            "orig_w": frame.shape[1], "orig_h": frame.shape[0],
            "sent_w": res.sent_size[0], "sent_h": res.sent_size[1],
            "jpeg_quality": args.quality, "upload_bytes": res.upload_bytes,
            "num_detections": len(res.detections),
            "encode_ms": round(res.encode_ms, 2), "network_ms": round(res.network_ms, 2),
            "server_ms": round(res.server_ms, 2), "inference_ms": round(res.inference_ms, 2),
            "e2e_ms": round(res.e2e_ms, 2), "rtt_ms": round(rtt, 2) if rtt else "",
            "within_deadline": bool(res.ok and res.e2e_ms <= args.deadline_ms),
            "src_index": fr.index, "dropped": fr.dropped, "age_ms": round(age_ms, 2),
        }
        writer.writerow(row)
        rows.append(row)
        all_dets[fid] = res.detections
        status = f"{len(res.detections):2d} objs" if res.ok else f"FAIL ({res.error})"
        drop = f"  drop={fr.dropped:3d}" if args.realtime else ""
        print(f"[uav] {fid:>20s} {status}  e2e={res.e2e_ms:7.1f} ms  "
              f"infer={res.inference_ms:6.1f} ms  up={res.upload_bytes/1024:6.1f} KB{drop}")
        if args.save_vis and res.ok:
            cv2.imwrite(os.path.join(args.save_vis, f"{fid}.jpg"), draw_detections(frame, res.detections))
        if period:
            time.sleep(max(0.0, period - (time.perf_counter() - t_loop)))
    wall_s = time.perf_counter() - t_start
    logf.close()

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(all_dets, f)
    print("\n[uav] summary")
    for k, v in summarize(rows, args.deadline_ms, wall_s).items():
        print(f"  {k:30s} {v:.3f}" if isinstance(v, float) else f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
