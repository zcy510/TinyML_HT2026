"""Demo instance: live dashboard for the adaptive UAV pipeline (runs on the Mac).

    python -m demo.server                      # http://localhost:8100 , listens on all interfaces
    python -m demo.server --port 8100 --open   # also opens the browser

Then start the other two parts with --demo pointing here:

    ground station:  python -m ground_station.server --demo http://<Mac-IP>:8100
    UAV (Pi):        python -m edge.run_remote --mode adaptive --server http://<GS-IP>:8000 \
                         --demo http://<Mac-IP>:8100 --source <sequence> --realtime --loop 100

What it does
* receives every processed frame from the UAV (mode, where inference ran, timings, link
  state measured by the UAV, policy decision, detections) plus a downscaled preview, runs a
  small IoU tracker for ids and trails, and pushes everything to the browser (Server-Sent
  Events) where it is rendered live;
* receives status pushes from the ground station (model, device, inference time, frames);
* holds the *emulated link* settings (extra delay, jitter, loss, upload bandwidth, RSSI).
  The dashboard changes them (number fields, arrow keys, presets, or a trace from
  traces/*.csv played back in real time); the UAV picks them up within ~0.25 s and applies
  them to its real traffic to the ground station (edge/demo_link.py). The policy engine
  then reacts to what it *measures*.

Standard library only (no extra packages), so it runs on the Mac, the lab PC or the Pi.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from demo.tracker import IoUTracker

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
TRACES = ROOT.parent / "traces"

LINK_KEYS = ("delay_ms", "jitter_ms", "loss_pct", "rate_mbit", "rssi_dbm")
PRESETS = {
    "good":     {"delay_ms": 0,   "jitter_ms": 0,  "loss_pct": 0,   "rate_mbit": 0, "rssi_dbm": -50},
    "moderate": {"delay_ms": 150, "jitter_ms": 15, "loss_pct": 2,   "rate_mbit": 8, "rssi_dbm": -68},
    "poor":     {"delay_ms": 350, "jitter_ms": 40, "loss_pct": 15,  "rate_mbit": 1, "rssi_dbm": -80},
    "gs_down":  {"delay_ms": 0,   "jitter_ms": 0,  "loss_pct": 100, "rate_mbit": 0, "rssi_dbm": -90},
    "real":     {"delay_ms": 0,   "jitter_ms": 0,  "loss_pct": 0,   "rate_mbit": 0, "rssi_dbm": None},
}
EDGE_TIMEOUT_S = 3.0
GS_TIMEOUT_S = 3.0


def load_trace(path: Path) -> List[dict]:
    """traces/*.csv (same format as tools.link_emulator) -> list of {t_s, <link keys>}."""
    rows = [ln.split("#")[0].strip() for ln in path.read_text().splitlines()]
    rows = [r for r in rows if r]
    header = [h.strip() for h in rows[0].split(",")]
    steps = []
    for r in rows[1:]:
        d = dict(zip(header, (float(v) for v in r.split(","))))
        steps.append({"t_s": d["t_s"], "delay_ms": d.get("delay_ms", 0.0),
                      "jitter_ms": d.get("jitter_ms", 0.0), "loss_pct": d.get("loss_pct", 0.0),
                      "rate_mbit": d.get("rate_mbit", 0.0), "rssi_dbm": d.get("rssi_dbm")})
    steps.sort(key=lambda s: s["t_s"])
    return steps


def clean_link(d: dict, base: dict) -> dict:
    out = dict(base)
    for k in LINK_KEYS:
        if k not in d:
            continue
        v = d[k]
        if k == "rssi_dbm":
            out[k] = None if v in (None, "") else max(-120.0, min(0.0, float(v)))
        else:
            v = max(0.0, float(v or 0))
            out[k] = min(v, 100.0) if k == "loss_pct" else min(v, 10000.0)
    return out


class DemoState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.link = dict(PRESETS["real"])
        self.link_by = "start"
        self.hello: Optional[dict] = None
        self.edge_seen = 0.0
        self.edge_summary: Optional[dict] = None
        self.gs: Optional[dict] = None
        self.gs_seen = 0.0
        self.trace: Optional[dict] = None
        self.clients: List[queue.Queue] = []
        self.tracker = IoUTracker()
        self.last_frames: Optional[dict] = None
        self.frames_total = 0

    # ------------------------------------------------------------------ clients
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def broadcast(self, kind: str, data: dict) -> str:
        msg = json.dumps({"type": kind, **data}, separators=(",", ":"), default=str)
        with self.lock:
            clients = list(self.clients)
        for q in clients:
            try:
                q.put_nowait(msg)
            except queue.Full:                      # slow browser: drop its oldest message
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                except queue.Empty:
                    pass
        return msg

    # -------------------------------------------------------------------- state
    def snapshot(self) -> dict:
        now = time.time()
        with self.lock:
            tr = None
            if self.trace:
                tr = {"name": self.trace["name"], "elapsed_s": now - self.trace["t0"],
                      "duration_s": self.trace["duration"], "step": self.trace["i"]}
            return {
                "link": self.link, "link_by": self.link_by, "presets": PRESETS,
                "traces": sorted(p.name for p in TRACES.glob("*.csv")) if TRACES.exists() else [],
                "trace": tr, "hello": self.hello, "edge_summary": self.edge_summary,
                "edge_online": now - self.edge_seen < EDGE_TIMEOUT_S,
                "edge_age_s": now - self.edge_seen if self.edge_seen else None,
                "gs": self.gs, "gs_online": now - self.gs_seen < GS_TIMEOUT_S,
                "gs_age_s": now - self.gs_seen if self.gs_seen else None,
                "frames_total": self.frames_total,
            }

    def set_link(self, d: dict, by: str) -> dict:
        with self.lock:
            self.link = clean_link(d, self.link)
            self.link_by = by
            if by != "trace" and self.trace:
                self.trace["stop"] = True
            link = dict(self.link)
        self.broadcast("link", {"link": link, "by": by})
        return link

    # ------------------------------------------------------------------- traces
    def start_trace(self, name: str, speed: float = 1.0) -> dict:
        path = (TRACES / os.path.basename(name))
        steps = load_trace(path)
        with self.lock:
            if self.trace:
                self.trace["stop"] = True
            tr = {"name": path.name, "steps": steps, "t0": time.time(), "speed": max(0.1, speed),
                  "duration": steps[-1]["t_s"] / max(0.1, speed), "i": -1, "stop": False}
            self.trace = tr
        threading.Thread(target=self._play, args=(tr,), daemon=True, name="trace").start()
        return {"name": tr["name"], "duration_s": tr["duration"], "steps": len(steps)}

    def stop_trace(self) -> None:
        with self.lock:
            if self.trace:
                self.trace["stop"] = True

    def _play(self, tr: dict) -> None:
        steps = tr["steps"]
        while not tr["stop"]:
            el = (time.time() - tr["t0"]) * tr["speed"]
            i = max(k for k, s in enumerate(steps) if s["t_s"] <= el) if el >= 0 else 0
            if i != tr["i"]:
                tr["i"] = i
                self.set_link(steps[i], "trace")
                self.broadcast("trace", {"name": tr["name"], "step": i, "t_s": steps[i]["t_s"],
                                         "duration_s": tr["duration"]})
            if el >= steps[-1]["t_s"]:
                break
            time.sleep(0.1)
        with self.lock:
            if self.trace is tr:
                self.trace = None
        self.broadcast("trace", {"name": tr["name"], "finished": True,
                                 "stopped": bool(tr["stop"])})

    # --------------------------------------------------------------- from edge
    def on_hello(self, d: dict) -> dict:
        with self.lock:
            self.hello = d
            self.edge_seen = time.time()
            self.edge_summary = None
            self.tracker.reset()
            link = dict(self.link)
        print(f"[demo] UAV connected: {d.get('host')} mode={d.get('mode')} "
              f"gs={d.get('gs_model')} local={d.get('local_model')}", flush=True)
        self.broadcast("hello", {"hello": d})
        return {"link": link}

    def on_frames(self, d: dict) -> dict:
        frames = d.get("frames") or []
        with self.lock:
            self.edge_seen = time.time()
            self.frames_total += len(frames)
            tracks = []
            for f in frames:
                tracks = self.tracker.update(f.get("detections") or [], float(f.get("t", time.time())))
            link = dict(self.link)
        light = []
        for k, f in enumerate(frames):              # only the newest frame keeps its boxes
            g = {key: v for key, v in f.items() if key != "detections"}
            g["num_detections"] = len(f.get("detections") or [])
            light.append(g)
        msg = {"frames": light, "tracks": tracks, "link": link}
        if d.get("jpeg"):
            msg["jpeg"], msg["jpeg_w"] = d["jpeg"], d.get("jpeg_w")
            self.last_frames = self.broadcast("frames", msg)   # for browsers that connect later
        else:
            self.broadcast("frames", msg)
        return {"link": link}

    def on_poll(self) -> dict:
        with self.lock:
            self.edge_seen = time.time()
            return {"link": dict(self.link)}

    def on_bye(self, d: dict) -> dict:
        with self.lock:
            self.edge_summary = d.get("summary")
            self.edge_seen = 0.0
        print("[demo] UAV run finished", flush=True)
        self.broadcast("bye", {"summary": self.edge_summary})
        return {"ok": True}

    def on_gs(self, d: dict) -> dict:
        with self.lock:
            self.gs = d
            self.gs_seen = time.time()
        self.broadcast("gs", {"gs": d})
        return {"ok": True}


def make_handler(state: DemoState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def log_message(self, *a):
            pass

        def _json(self, obj, code: int = 200) -> None:
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if not n:
                return {}
            return json.loads(self.rfile.read(n) or b"{}")

        def _static(self, name: str, ctype: str) -> None:
            p = STATIC / name
            if not p.is_file():
                return self._json({"detail": "not found"}, 404)
            body = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.close_connection = True
            q = state.subscribe()
            try:
                first = [json.dumps({"type": "state", **state.snapshot()}, default=str)]
                if state.last_frames:
                    first.append(state.last_frames)
                for m in first:
                    self.wfile.write(f"data: {m}\n\n".encode())
                self.wfile.flush()
                last_status = 0.0
                while True:
                    try:
                        m = q.get(timeout=0.5)
                        self.wfile.write(f"data: {m}\n\n".encode())
                    except queue.Empty:
                        pass
                    if time.time() - last_status > 1.0:      # online flags, trace progress
                        last_status = time.time()
                        s = state.snapshot()
                        st = {k: s[k] for k in ("edge_online", "edge_age_s", "gs_online", "gs_age_s",
                                                "trace", "frames_total")}
                        self.wfile.write(f"data: {json.dumps({'type': 'status', **st}, default=str)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                state.unsubscribe(q)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._static("index.html", "text/html; charset=utf-8")
            if path == "/events":
                return self._events()
            if path == "/api/link":                  # polled by the UAV
                return self._json(state.on_poll())
            if path == "/api/state":
                return self._json(state.snapshot())
            if path == "/health":
                return self._json({"status": "ok", "service": "demo"})
            return self._json({"detail": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                d = self._body()
            except (ValueError, json.JSONDecodeError):
                return self._json({"detail": "bad json"}, 400)
            try:
                if path == "/api/frames":
                    return self._json(state.on_frames(d))
                if path == "/api/hello":
                    return self._json(state.on_hello(d))
                if path == "/api/bye":
                    return self._json(state.on_bye(d))
                if path == "/api/gs":
                    return self._json(state.on_gs(d))
                if path == "/api/link":              # from the dashboard
                    if d.get("preset") in PRESETS:
                        d = PRESETS[d["preset"]]
                    return self._json({"link": state.set_link(d, "user")})
                if path == "/api/trace":
                    if d.get("stop"):
                        state.stop_trace()
                        return self._json({"ok": True})
                    return self._json(state.start_trace(d.get("name", ""), float(d.get("speed", 1.0))))
            except (OSError, ValueError, KeyError) as e:
                return self._json({"detail": f"{type(e).__name__}: {e}"}, 400)
            return self._json({"detail": "not found"}, 404)

    return Handler


def local_ips() -> List[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def main() -> None:
    ap = argparse.ArgumentParser(description="Live demo dashboard for the adaptive UAV pipeline")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--open", action="store_true", help="open the dashboard in the browser")
    args = ap.parse_args()
    state = DemoState()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    srv.daemon_threads = True
    print(f"[demo] dashboard: http://localhost:{args.port}", flush=True)
    for ip in local_ips():
        print(f"[demo] for the UAV / ground station:  --demo http://{ip}:{args.port}", flush=True)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] bye")


if __name__ == "__main__":
    main()
