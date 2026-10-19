"""Live-demo hooks on the UAV side: a link shaper controlled from the demo dashboard,
and a reporter that streams every processed frame to the demo instance.

    python -m edge.run_remote --mode adaptive --server http://<GS>:8000 \
        --demo http://<Mac>:8100 --source <sequence> --realtime --loop 100

* **LinkShaper** - application-level link emulation. The demo dashboard sets an extra
  round-trip delay, jitter, packet loss, an upload bandwidth limit and an RSSI value;
  the shaper applies them to *real* traffic to the ground station: every /ping probe
  of the link monitor and every /detect request. The policy engine is not told the
  settings - it only sees what the link monitor measures, exactly as with
  tools.link_emulator (tc netem), but this works without root and on any OS.
  - delay / jitter: added to every request (once per round trip)
  - loss: a lost request is never answered, the client waits until its timeout
  - rate: upload time = bytes * 8 / rate is added to every /detect request
  - rssi: replaces the Wi-Fi RSSI the link monitor reads (None = real value)
* **DemoReporter** - background thread; the pipeline only hands over references, so it
  never waits for the demo. Per processed frame it sends the metadata (mode, timings,
  link state, policy decision, detections); a downscaled JPEG preview is attached to the
  newest frame at most `max_fps` times per second. Each reply carries the current link
  settings, and the thread also polls them when there are no frames, so a change on the
  dashboard reaches the shaper within ~0.25 s. The demo channel itself is not shaped.
"""
from __future__ import annotations

import base64
import json
import random
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np
import requests

from edge.remote_client import RemoteResult


# ---------------------------------------------------------------------------- shaper
@dataclass
class LinkSettings:
    delay_ms: float = 0.0       # extra round-trip delay
    jitter_ms: float = 0.0      # +- uniform
    loss_pct: float = 0.0       # share of requests that are never answered
    rate_mbit: float = 0.0      # upload bandwidth limit, 0 = not limited
    rssi_dbm: Optional[float] = None   # None = use the real Wi-Fi RSSI

    @classmethod
    def from_dict(cls, d: dict) -> "LinkSettings":
        s = cls()
        for k in ("delay_ms", "jitter_ms", "loss_pct", "rate_mbit"):
            if d.get(k) is not None:
                setattr(s, k, max(0.0, float(d[k])))
        s.loss_pct = min(s.loss_pct, 100.0)
        s.rssi_dbm = None if d.get("rssi_dbm") in (None, "") else float(d["rssi_dbm"])
        return s


class LinkShaper:
    def __init__(self, settings: Optional[LinkSettings] = None, seed: Optional[int] = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._s = settings or LinkSettings()
        self._lock = threading.Lock()
        self._rng = random.Random(seed)
        self._sleep = sleep

    @property
    def settings(self) -> LinkSettings:
        with self._lock:
            return LinkSettings(**asdict(self._s))

    def set(self, s: LinkSettings) -> None:
        with self._lock:
            self._s = s

    # -- helpers ---------------------------------------------------------------
    def _lost(self, s: LinkSettings) -> bool:
        return s.loss_pct > 0 and self._rng.random() * 100.0 < s.loss_pct

    def _delay_ms(self, s: LinkSettings) -> float:
        j = self._rng.uniform(-s.jitter_ms, s.jitter_ms) if s.jitter_ms > 0 else 0.0
        return max(0.0, s.delay_ms + j)

    @staticmethod
    def upload_ms(s: LinkSettings, nbytes: int) -> float:
        return nbytes * 8.0 / (s.rate_mbit * 1e6) * 1000.0 if s.rate_mbit > 0 else 0.0

    # -- hooks -----------------------------------------------------------------
    def wrap_probe(self, probe: Callable[[], Optional[float]], timeout_s: float):
        """Shaped version of LinkMonitor's /ping probe (returns RTT ms or None)."""
        def shaped() -> Optional[float]:
            s = self.settings
            if self._lost(s):
                self._sleep(timeout_s)
                return None
            t0 = time.perf_counter()
            rtt = probe()
            if rtt is None:
                return None
            extra = self._delay_ms(s)
            if rtt + extra > timeout_s * 1000.0:
                self._sleep(max(0.0, timeout_s - (time.perf_counter() - t0)))
                return None
            self._sleep(extra / 1000.0)
            return rtt + extra
        return shaped

    def wrap_rssi(self, rssi: Callable[[], Optional[float]]):
        def shaped() -> Optional[float]:
            v = self.settings.rssi_dbm
            return v if v is not None else rssi()
        return shaped


class ShapedRemote:
    """RemoteDetector seen through the LinkShaper (same interface)."""

    def __init__(self, remote, shaper: LinkShaper) -> None:
        self.remote, self.shaper = remote, shaper
        self.timeout_s = remote.timeout_s

    def health(self) -> dict:
        return self.remote.health()

    def ping(self, n: int = 3):
        return self.remote.ping(n)

    def detect(self, frame_bgr: np.ndarray, frame_id: str = "", max_width: Optional[int] = None,
               jpeg_quality: int = 90, conf: Optional[float] = None,
               imgsz: Optional[int] = None) -> RemoteResult:
        s = self.shaper.settings
        t0 = time.perf_counter()
        timeout_ms = self.timeout_s * 1000.0
        if self.shaper._lost(s):
            data, size, scale = self.remote.encode(frame_bgr, max_width, jpeg_quality)
            self.shaper._sleep(max(0.0, self.timeout_s - (time.perf_counter() - t0)))
            return RemoteResult(False, frame_id, e2e_ms=(time.perf_counter() - t0) * 1000.0,
                                upload_bytes=len(data), sent_size=size, scale=scale,
                                error="Timeout (simulated loss)")
        kw = {"imgsz": imgsz} if imgsz else {}
        res = self.remote.detect(frame_bgr, frame_id, max_width, jpeg_quality, conf, **kw)
        if not res.ok:
            return res
        extra = self.shaper._delay_ms(s) + self.shaper.upload_ms(s, res.upload_bytes)
        if res.e2e_ms + extra > timeout_ms:      # the reply would have come after the timeout
            self.shaper._sleep(max(0.0, self.timeout_s - (time.perf_counter() - t0)))
            return RemoteResult(False, frame_id, e2e_ms=(time.perf_counter() - t0) * 1000.0,
                                encode_ms=res.encode_ms, upload_bytes=res.upload_bytes,
                                sent_size=res.sent_size, scale=res.scale,
                                error="Timeout (simulated link)")
        self.shaper._sleep(extra / 1000.0)
        res.e2e_ms += extra
        res.network_ms += extra
        return res


def attach_shaper(monitor, shaper: LinkShaper) -> None:
    """Route a LinkMonitor's probes and RSSI reads through the shaper."""
    monitor._probe = shaper.wrap_probe(monitor._probe, monitor.timeout_s)
    monitor._rssi = shaper.wrap_rssi(monitor._rssi)


# -------------------------------------------------------------------------- reporter
class DemoReporter:
    def __init__(self, url: str, shaper: Optional[LinkShaper] = None, preview_width: int = 640,
                 max_fps: float = 15.0, jpeg_quality: int = 70, timeout_s: float = 1.0,
                 poll_s: float = 0.25) -> None:
        self.url = url.rstrip("/")
        self.shaper = shaper
        self.preview_width, self.jpeg_quality = preview_width, jpeg_quality
        self.min_img_period = 1.0 / max_fps if max_fps > 0 else 0.0
        self.timeout_s, self.poll_s = timeout_s, poll_s
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._pending: List[dict] = []
        self._latest_img: Optional[np.ndarray] = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_img_t = 0.0
        self.online = False
        self.sent_frames = 0

    # -- network ---------------------------------------------------------------
    def _apply(self, body: dict) -> None:
        if self.shaper is not None and isinstance(body.get("link"), dict):
            self.shaper.set(LinkSettings.from_dict(body["link"]))

    def _call(self, method: str, path: str, payload: Optional[dict] = None) -> Optional[dict]:
        try:
            data = None if payload is None else json.dumps(
                payload, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            r = self._session.request(method, f"{self.url}{path}", data=data, timeout=self.timeout_s,
                                      headers={"Content-Type": "application/json"})
            r.raise_for_status()
            body = r.json()
            if not self.online:
                print(f"[demo] connected to demo instance at {self.url}")
            self.online = True
            self._apply(body)
            return body
        except (requests.RequestException, ValueError) as e:
            if self.online:
                print(f"[demo] lost demo instance ({type(e).__name__}); retrying in the background")
            self.online = False
            return None

    def start(self, hello: dict) -> "DemoReporter":
        if self._call("POST", "/api/hello", hello) is None:
            print(f"[demo] demo instance not reachable at {self.url} - will keep trying")
        self._thread = threading.Thread(target=self._run, daemon=True, name="demo-reporter")
        self._thread.start()
        return self

    def stop(self, summary: Optional[dict] = None) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._call("POST", "/api/bye", {"summary": summary or {}})

    # -- pipeline side (must never block) --------------------------------------
    def submit(self, frame: np.ndarray, meta: dict) -> None:
        with self._lock:
            self._pending.append(meta)
            if len(self._pending) > 300:             # demo away: keep only the recent past
                del self._pending[:-300]
            self._latest_img = frame
        self._wake.set()

    # -- background thread -------------------------------------------------------
    def _encode(self, img: np.ndarray) -> Optional[str]:
        h, w = img.shape[:2]
        if self.preview_width and w > self.preview_width:
            img = cv2.resize(img, (self.preview_width, int(round(h * self.preview_width / w))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.poll_s)
            self._wake.clear()
            with self._lock:
                batch, img = self._pending, self._latest_img
                self._pending, self._latest_img = [], None
            if not batch:
                self._call("GET", "/api/link")
                continue
            payload = {"frames": batch}
            now = time.monotonic()
            if img is not None and now - self._last_img_t >= self.min_img_period:
                payload["jpeg"] = self._encode(img)
                payload["jpeg_w"] = min(img.shape[1], self.preview_width or img.shape[1])
                self._last_img_t = now
            if self._call("POST", "/api/frames", payload) is not None:
                self.sent_frames += len(batch)
            if self.min_img_period:                  # do not spin faster than the preview rate
                self._stop.wait(max(0.0, self.min_img_period - (time.monotonic() - now)) * 0.5)
