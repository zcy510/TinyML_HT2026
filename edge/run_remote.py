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

# adaptive: the policy engine picks remote full / remote reduced / local per frame
# from the measured link (RTT, loss, RSSI, remote latency), see edge/policy.py
python -m edge.run_remote --mode adaptive --server http://192.168.1.50:8000 \
    --source data/VisDrone2019-VID-val/sequences/uav0000086_00000_v --realtime --loop 4

# live demo: stream every frame to the demo dashboard (demo/server.py on the Mac) and let
# the dashboard emulate the link to the ground station (delay / loss / bandwidth / RSSI)
python -m edge.run_remote --mode adaptive --server http://192.168.1.50:8000 \
    --demo http://192.168.1.20:8100 --source data/uav0000117_02622_v --realtime --loop 100

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
    "mode", "fallback", "switch", "link_rtt_ms", "link_loss", "link_e2e_ms", "rssi_dbm",
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
    modes = [r.get("mode") for r in rows if r.get("mode")]
    if len(set(modes)) > 1 or any(r.get("fallback") for r in rows):
        for m in sorted(set(modes)):
            out[f"share_{m}"] = modes.count(m) / len(rows)
        out["mode_switches"] = sum(1 for r in rows if r.get("switch"))
        out["fallbacks"] = sum(1 for r in rows if r.get("fallback"))
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="UAV pipeline: remote / local / adaptive detection")
    ap.add_argument("--mode", choices=["remote", "local", "adaptive"], default="remote",
                    help="remote = ground station (needs --server); local = on-board INT8 model "
                         "(--model); adaptive = policy engine switches between them (needs both)")
    ap.add_argument("--server", default=None, help="e.g. http://192.168.1.50:8000")
    ap.add_argument("--model", default="models/yolov8n_int8_320.tflite",
                    help="INT8 model for local / adaptive mode (.onnx via onnxruntime, or .tflite)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for the local model")
    ap.add_argument("--source", required=True, help="folder | image | video | camera index | picam")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop after this many seconds (e.g. the length of a link trace)")
    ap.add_argument("--loop", type=int, default=1,
                    help="play a folder/video source N times (for runs longer than one sequence)")
    ap.add_argument("--max-width", type=int, default=None,
                    help="downscale before sending (e.g. 320 for reduced-resolution mode)")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality 1-100")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="remote mode: YOLO inference size on the ground station for every frame "
                         "(e.g. 1280); default = the server's --imgsz")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=2.0,
                    help="HTTP timeout in seconds (adaptive: at most the deadline)")
    ap.add_argument("--deadline-ms", type=float, default=500.0)
    ap.add_argument("--ping-every", type=int, default=10, help=argparse.SUPPRESS)  # replaced by monitor
    ap.add_argument("--realtime", action="store_true",
                    help="replay a video / frame folder like a live camera: drop the frames "
                         "that go by while one is being processed")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="with --realtime: camera frame rate (0 = video's own rate, 30 for a "
                         "folder); without: throttle to this rate (0 = as fast as possible)")
    ap.add_argument("--log", default="logs/remote_client_log.csv")
    ap.add_argument("--save-json", default=None, help="write all detections to this JSON file")
    ap.add_argument("--save-vis", default=None, help="folder for annotated images")
    g = ap.add_argument_group("link monitor / policy (adaptive mode; monitor also logs in fixed modes)")
    g.add_argument("--probe-interval", type=float, default=0.5, help="seconds between /ping probes")
    g.add_argument("--iface", default="wlan0", help="Wi-Fi interface for RSSI")
    g.add_argument("--no-rssi", action="store_true", help="ignore RSSI in the policy")
    g.add_argument("--policy-modes", default="remote_full,remote_reduced,local",
                   help="modes the policy may use, best first (e.g. remote_full,local)")
    g.add_argument("--reduced-width", type=int, default=640)
    g.add_argument("--reduced-quality", type=int, default=70)
    g.add_argument("--full-imgsz", type=int, default=1280,
                   help="ground-station inference size for remote_full frames (0 = server default)")
    g.add_argument("--reduced-imgsz", type=int, default=640,
                   help="ground-station inference size for remote_reduced frames (0 = server default)")
    g.add_argument("--rtt-good", type=float, default=100.0)
    g.add_argument("--rtt-poor", type=float, default=250.0)
    g.add_argument("--loss-good", type=float, default=0.05)
    g.add_argument("--loss-poor", type=float, default=0.10)
    g.add_argument("--rssi-good", type=float, default=-60.0)
    g.add_argument("--rssi-poor", type=float, default=-75.0)
    g.add_argument("--down-hold", type=float, default=1.0, help="s a worse link must persist")
    g.add_argument("--up-hold", type=float, default=3.0, help="s a better link must persist")
    g.add_argument("--min-dwell", type=float, default=2.0, help="min s between mode switches")
    g.add_argument("--no-fallback", action="store_true",
                   help="adaptive: do not re-run a failed remote frame locally")
    d = ap.add_argument_group("live demo (demo/server.py)")
    d.add_argument("--demo", default=None, metavar="URL",
                   help="demo instance, e.g. http://192.168.1.20:8100: frames, detections and "
                        "metrics are streamed there, and its link controls shape the GS link")
    d.add_argument("--demo-width", type=int, default=640, help="width of the preview sent to the demo")
    d.add_argument("--demo-fps", type=float, default=15.0, help="max preview images per second")
    d.add_argument("--no-demo-shaper", action="store_true",
                   help="only report to the demo; ignore its link controls (e.g. with tc netem)")
    return ap


def build_client(args, ap):
    """Returns (detector, link monitor or None, setting name, demo reporter or None)."""
    shaper = reporter = None
    if args.demo:
        from edge.demo_link import DemoReporter, LinkShaper

        shaper = None if args.no_demo_shaper else LinkShaper()
        reporter = DemoReporter(args.demo, shaper, args.demo_width, args.demo_fps)
    monitor = None
    if args.server:
        from edge.link_monitor import LinkMonitor

        monitor = LinkMonitor(args.server, interval_s=args.probe_interval,
                              timeout_s=min(1.0, args.deadline_ms / 1000.0), iface=args.iface)
        if shaper:
            from edge.demo_link import attach_shaper

            attach_shaper(monitor, shaper)
    local = None
    if args.mode in ("local", "adaptive"):
        from edge.local_detector import LocalDetector

        local = LocalDetector(args.model, conf=args.conf or 0.25, num_threads=args.threads)
        info = local.health()
        print(f"[uav] local model: {info['model']} ({info['device']}, input {local.imgsz}x{local.imgsz})")
    remote = gs_info = None
    if args.mode in ("remote", "adaptive"):
        if not args.server:
            ap.error(f"--server is required in {args.mode} mode")
        timeout = min(args.timeout, args.deadline_ms / 1000.0) if args.mode == "adaptive" else args.timeout
        remote = RemoteDetector(args.server, timeout,
                                imgsz=args.imgsz if args.mode == "remote" else None)
        if shaper:
            from edge.demo_link import ShapedRemote

            remote = ShapedRemote(remote, shaper)
        try:
            info = remote.health()
            gs_info = info
            print(f"[uav] ground station: model={info['model']} device={info['device']}")
        except Exception as e:
            if args.mode == "remote":
                sys.exit(f"[uav] ground station not reachable at {args.server}: {e}")
            print(f"[uav] ground station not reachable now ({type(e).__name__}); adaptive starts locally")
    cfg = None
    if args.mode == "adaptive":
        from edge.policy import PolicyConfig

        cfg = PolicyConfig(rtt_good_ms=args.rtt_good, rtt_poor_ms=args.rtt_poor,
                           loss_good=args.loss_good, loss_poor=args.loss_poor,
                           rssi_good_dbm=args.rssi_good, rssi_poor_dbm=args.rssi_poor,
                           deadline_ms=args.deadline_ms, down_hold_s=args.down_hold,
                           up_hold_s=args.up_hold, min_dwell_s=args.min_dwell,
                           use_rssi=not args.no_rssi)
    if reporter:      # before the monitor's warm-up probes, so they already see the demo's link
        import socket
        from dataclasses import asdict

        reporter.start({
            "host": socket.gethostname(), "mode": args.mode, "source": args.source,
            "realtime": args.realtime, "camera_fps": (args.fps or 30.0) if args.realtime else None,
            "deadline_ms": args.deadline_ms, "server": args.server,
            "local_model": os.path.basename(args.model) if local else None,
            "local_device": local.health()["device"] if local else None,
            "gs_model": gs_info.get("model") if gs_info else None,
            "gs_device": gs_info.get("device") if gs_info else None,
            "policy": asdict(cfg) if cfg else None,
            "policy_modes": [m.strip() for m in args.policy_modes.split(",")],
            "reduced_width": args.reduced_width, "reduced_quality": args.reduced_quality,
            "full_quality": args.quality, "shaper": shaper is not None,
            "full_imgsz": args.full_imgsz, "reduced_imgsz": args.reduced_imgsz,
        })
    if monitor:
        monitor.start()
        print(f"[uav] link monitor: {monitor.snapshot().short()}")

    if args.mode == "local":
        return local, monitor, "local_" + os.path.splitext(os.path.basename(args.model))[0], reporter
    if args.mode == "remote":
        name = f"w{args.max_width or 'full'}_q{args.quality}" + (f"_i{args.imgsz}" if args.imgsz else "")
        return remote, monitor, name, reporter

    from edge.adaptive import AdaptiveDetector
    from edge.policy import Policy

    modes = [m.strip() for m in args.policy_modes.split(",")]
    policy = Policy(cfg, initial=modes[0], modes=modes)
    first, why = policy.classify(monitor.snapshot())     # start in the mode the link supports
    policy.level = first
    print(f"[policy] start in {policy.mode} ({why})")
    client = AdaptiveDetector(remote, local, monitor, policy, args.reduced_width,
                              args.reduced_quality, args.quality, fallback=not args.no_fallback,
                              full_imgsz=args.full_imgsz or None, reduced_imgsz=args.reduced_imgsz or None)
    return client, monitor, "adaptive", reporter


def demo_meta(row: dict, res, info: dict, client) -> dict:
    """One frame for the demo dashboard (see edge/demo_link.py)."""
    num = lambda v: None if v in ("", None) else float(v)
    meta = {k: row[k] for k in ("frame_id", "src_index", "dropped", "orig_w", "orig_h", "sent_w",
                                "sent_h", "ok", "error", "encode_ms", "network_ms", "server_ms",
                                "inference_ms", "e2e_ms", "age_ms", "upload_bytes", "mode",
                                "fallback", "switch")}
    meta["t"] = time.time()
    meta["source"] = "edge" if info["mode"] == "local" or info["fallback"] else "ground_station"
    meta["detections"] = res.detections
    meta["link"] = {"rtt_ms": num(info["link_rtt_ms"]), "loss": num(info["link_loss"]),
                    "frame_e2e_ms": num(info["link_e2e_ms"]), "rssi_dbm": num(info["rssi_dbm"])}
    policy = getattr(client, "policy", None)
    meta["policy"] = policy.status(time.monotonic()) if policy is not None else None
    return meta


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    client, monitor, setting, reporter = build_client(args, ap)

    if args.save_vis:
        os.makedirs(args.save_vis, exist_ok=True)
    logf, writer = open_log(args.log)

    rows, all_dets = [], {}
    period = 1.0 / args.fps if args.fps > 0 and not args.realtime else 0.0
    if args.realtime:
        setting += "_rt"
        fps = args.fps or "video's own (30 for a folder)"
        print(f"[uav] real-time replay at {fps} fps: frames missed while busy are dropped")
    t_start = time.perf_counter()
    try:
        for fr in iter_frames(args.source, args.max_frames, args.realtime, args.fps, args.loop):
            t_loop = time.perf_counter()
            if args.max_seconds and t_loop - t_start >= args.max_seconds:
                break
            frame, fid = fr.image, fr.frame_id
            res = client.detect(frame, fid, args.max_width, args.quality, args.conf)
            age_ms = (time.perf_counter() - fr.capture_t) * 1000.0
            info = getattr(client, "row_info", None)
            if info is None:                           # fixed mode: log the link state anyway
                st = monitor.snapshot() if monitor else None
                if monitor and args.mode == "remote":
                    monitor.report_frame(res.ok, res.e2e_ms)
                info = {"mode": args.mode, "fallback": False, "switch": "",
                        "link_rtt_ms": round(st.rtt_ms, 1) if st and st.rtt_ms is not None else "",
                        "link_loss": round(st.loss, 3) if st else "",
                        "link_e2e_ms": round(st.frame_e2e_ms, 1) if st and st.frame_e2e_ms is not None else "",
                        "rssi_dbm": st.rssi_dbm if st and st.rssi_dbm is not None else ""}
            row = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "frame_id": fid, "setting": setting, "ok": res.ok, "error": res.error,
                "orig_w": frame.shape[1], "orig_h": frame.shape[0],
                "sent_w": res.sent_size[0], "sent_h": res.sent_size[1],
                "jpeg_quality": args.quality, "upload_bytes": res.upload_bytes,
                "num_detections": len(res.detections),
                "encode_ms": round(res.encode_ms, 2), "network_ms": round(res.network_ms, 2),
                "server_ms": round(res.server_ms, 2), "inference_ms": round(res.inference_ms, 2),
                "e2e_ms": round(res.e2e_ms, 2), "rtt_ms": info["link_rtt_ms"],
                "within_deadline": bool(res.ok and res.e2e_ms <= args.deadline_ms),
                "src_index": fr.index, "dropped": fr.dropped, "age_ms": round(age_ms, 2),
                **info,
            }
            writer.writerow(row)
            rows.append(row)
            if reporter:
                reporter.submit(frame, demo_meta(row, res, info, client))
            all_dets[fid] = res.detections
            status = f"{len(res.detections):2d} objs" if res.ok else f"FAIL ({res.error})"
            drop = f"  drop={fr.dropped:3d}" if args.realtime else ""
            mode = f"  [{info['mode']}{'+fb' if info['fallback'] else ''}]" if args.mode == "adaptive" else ""
            print(f"[uav] {fid:>20s} {status}  e2e={res.e2e_ms:7.1f} ms  "
                  f"infer={res.inference_ms:6.1f} ms  up={res.upload_bytes/1024:6.1f} KB{drop}{mode}")
            if args.save_vis and res.ok:
                cv2.imwrite(os.path.join(args.save_vis, f"{fid}.jpg"), draw_detections(frame, res.detections))
            if period:
                time.sleep(max(0.0, period - (time.perf_counter() - t_loop)))
    except KeyboardInterrupt:
        print("\n[uav] interrupted")
    finally:
        wall_s = time.perf_counter() - t_start
        if monitor:
            monitor.stop()
        logf.close()
        if reporter:
            reporter.stop(summarize(rows, args.deadline_ms, wall_s))

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(all_dets, f)
    print("\n[uav] summary")
    for k, v in summarize(rows, args.deadline_ms, wall_s).items():
        print(f"  {k:30s} {v:.3f}" if isinstance(v, float) else f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
