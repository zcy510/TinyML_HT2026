"""Mock ground station: same HTTP API as ground_station.server, but no model.

Lets the UAV side (Raspberry Pi) be tested on its own: capture, JPEG encoding,
HTTP transfer, logging and summaries all run for real; only the detector is
missing (it returns no detections). Uses only the Python standard library
(+ OpenCV if available, to report the image size), so it runs on the Pi.

    python -m ground_station.mock_server                    # 127.0.0.1:8000
    python -m ground_station.mock_server --delay-ms 40      # pretend inference takes 40 ms
    python -m ground_station.mock_server --host 0.0.0.0     # reachable from other machines
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

try:
    import cv2
    import numpy as np
except ImportError:  # still works, just without image size
    cv2 = None


def make_handler(delay_ms: float):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive, like uvicorn
        disable_nagle_algorithm = True  # else headers/body split adds ~40 ms per reply

        def log_message(self, *args):  # quiet
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                self._send({"status": "ok", "model": "mock (no model)", "device": "none"})
            elif path == "/ping":
                self._send({"t": time.time()})
            else:
                self._send({"detail": "not found"}, 404)

        def do_POST(self):
            t0 = time.perf_counter()
            url = urlparse(self.path)
            if url.path != "/detect":
                self._send({"detail": "not found"}, 404)
                return
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not data:
                self._send({"detail": "empty body"}, 400)
                return
            size = [0, 0]
            if cv2 is not None:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    self._send({"detail": "could not decode image"}, 400)
                    return
                size = [img.shape[1], img.shape[0]]
            decode_ms = (time.perf_counter() - t0) * 1000.0
            if delay_ms:
                time.sleep(delay_ms / 1000.0)
            frame_id = parse_qs(url.query).get("frame_id", [None])[0]
            self._send({
                "frame_id": frame_id, "model": "mock (no model)", "device": "none",
                "image_size": size, "detections": [],
                "timing": {"decode_ms": round(decode_ms, 2), "inference_ms": float(delay_ms),
                           "server_ms": round((time.perf_counter() - t0) * 1000.0, 2)},
            })

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Mock ground station (no model)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--delay-ms", type=float, default=0.0, help="fake inference time")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(args.delay_ms))
    print(f"[mock-gs] listening on {args.host}:{args.port} (fake inference {args.delay_ms} ms)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
