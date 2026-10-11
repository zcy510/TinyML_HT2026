"""Run the remote high-complexity pipeline from the UAV (Raspberry Pi 5) side.

Examples
--------
# images in a folder (e.g. a VisDrone / COCO subset), server on the school PC
python -m edge.run_remote --server http://192.168.1.50:8000 --source data/images

# USB camera / Pi camera, 50 frames, reduced resolution (moderate-link mode)
python -m edge.run_remote --server http://192.168.1.50:8000 --source 0 --max-frames 50 --max-width 320
python -m edge.run_remote --server http://192.168.1.50:8000 --source picam

Writes one CSV row per frame (default logs/remote_client_log.csv) and prints a
summary: latency percentiles, bytes/frame, and reliability (= share of frames
with a valid result within --deadline-ms).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

from edge.remote_client import RemoteDetector, draw_detections

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOG_FIELDS = [
    "timestamp", "frame_id", "setting", "ok", "error", "orig_w", "orig_h", "sent_w", "sent_h",
    "jpeg_quality", "upload_bytes", "num_detections", "encode_ms", "network_ms",
    "server_ms", "inference_ms", "e2e_ms", "rtt_ms", "within_deadline",
]


def iter_frames(source: str, max_frames: int) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield (frame_id, BGR frame) from a folder, a single image, a video,
    a camera index or the Raspberry Pi camera ('picam')."""
    n = 0
    p = Path(source)
    if p.is_dir():
        for f in sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_EXT):
            if max_frames and n >= max_frames:
                return
            img = cv2.imread(str(f))
            if img is not None:
                n += 1
                yield f.stem, img
        return
    if p.is_file() and p.suffix.lower() in IMG_EXT:
        yield p.stem, cv2.imread(str(p))
        return
    if source == "picam":
        from picamera2 import Picamera2  # only on the Raspberry Pi

        cam = Picamera2()
        cam.configure(cam.create_still_configuration(main={"size": (640, 480), "format": "RGB888"}))
        cam.start()
        try:
            while not max_frames or n < max_frames:
                frame = cam.capture_array()  # RGB888 in picamera2 is BGR-ordered for OpenCV
                n += 1
                yield f"picam_{n:05d}", frame
        finally:
            cam.stop()
        return
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"cannot open source: {source}")
    try:
        while not max_frames or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            yield f"frame_{n:05d}", frame
    finally:
        cap.release()


def summarize(rows, deadline_ms: float) -> dict:
    if not rows:
        return {}
    ok = [r for r in rows if r["ok"]]
    e2e = np.array([r["e2e_ms"] for r in ok]) if ok else np.array([np.nan])
    return {
        "frames": len(rows),
        "success_rate": len(ok) / len(rows),
        "reliability_within_deadline": sum(r["within_deadline"] for r in rows) / len(rows),
        "deadline_ms": deadline_ms,
        "e2e_ms_mean": float(np.nanmean(e2e)),
        "e2e_ms_p50": float(np.nanpercentile(e2e, 50)),
        "e2e_ms_p95": float(np.nanpercentile(e2e, 95)),
        "inference_ms_mean": float(np.mean([r["inference_ms"] for r in ok])) if ok else None,
        "network_ms_mean": float(np.mean([r["network_ms"] for r in ok])) if ok else None,
        "upload_kb_per_frame": float(np.mean([r["upload_bytes"] for r in rows]) / 1024),
        "detections_per_frame": float(np.mean([r["num_detections"] for r in ok])) if ok else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="UAV -> ground station remote detection")
    ap.add_argument("--server", required=True, help="e.g. http://192.168.1.50:8000")
    ap.add_argument("--source", required=True, help="folder | image | video | camera index | picam")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--max-width", type=int, default=None,
                    help="downscale before sending (e.g. 320 for reduced-resolution mode)")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality 1-100")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout in seconds")
    ap.add_argument("--deadline-ms", type=float, default=500.0)
    ap.add_argument("--ping-every", type=int, default=10, help="measure RTT every N frames (0=off)")
    ap.add_argument("--fps", type=float, default=0.0, help="throttle to this frame rate (0=as fast as possible)")
    ap.add_argument("--log", default="logs/remote_client_log.csv")
    ap.add_argument("--save-json", default=None, help="write all detections to this JSON file")
    ap.add_argument("--save-vis", default=None, help="folder for annotated images")
    args = ap.parse_args()

    client = RemoteDetector(args.server, args.timeout)
    try:
        info = client.health()
        print(f"[uav] connected: model={info['model']} device={info['device']}")
    except Exception as e:
        sys.exit(f"[uav] ground station not reachable at {args.server}: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    if args.save_vis:
        os.makedirs(args.save_vis, exist_ok=True)
    new_file = not os.path.exists(args.log)
    logf = open(args.log, "a", newline="")
    writer = csv.DictWriter(logf, LOG_FIELDS)
    if new_file:
        writer.writeheader()

    rows, all_dets, rtt = [], {}, None
    period = 1.0 / args.fps if args.fps > 0 else 0.0
    setting = f"w{args.max_width or 'full'}_q{args.quality}"   # e.g. "wfull_q90", "w320_q70"
    for i, (fid, frame) in enumerate(iter_frames(args.source, args.max_frames)):
        t_loop = time.perf_counter()
        if args.ping_every and i % args.ping_every == 0:
            rtt = client.ping()
        res = client.detect(frame, fid, args.max_width, args.quality, args.conf)
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
        }
        writer.writerow(row)
        rows.append(row)
        all_dets[fid] = res.detections
        status = f"{len(res.detections):2d} objs" if res.ok else f"FAIL ({res.error})"
        print(f"[uav] {fid:>20s} {status}  e2e={res.e2e_ms:7.1f} ms  "
              f"infer={res.inference_ms:6.1f} ms  up={res.upload_bytes/1024:6.1f} KB")
        if args.save_vis and res.ok:
            cv2.imwrite(os.path.join(args.save_vis, f"{fid}.jpg"), draw_detections(frame, res.detections))
        if period:
            time.sleep(max(0.0, period - (time.perf_counter() - t_loop)))
    logf.close()

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(all_dets, f)
    print("\n[uav] summary")
    for k, v in summarize(rows, args.deadline_ms).items():
        print(f"  {k:30s} {v:.3f}" if isinstance(v, float) else f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
