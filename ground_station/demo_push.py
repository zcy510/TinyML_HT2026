"""Ground station -> demo instance status push (python -m ground_station.server --demo URL).

A background thread posts the newest per-frame result (model, device, inference time,
detections, which client sent it) and a heartbeat about twice a second to the demo
dashboard's /api/gs. The request handler only stores a dict, so detection latency is
not affected when the demo is slow or offline.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Optional

import requests


class GSReporter:
    def __init__(self, url: str, model: str, device: str, period_s: float = 0.5,
                 timeout_s: float = 1.0) -> None:
        self.url = url.rstrip("/") + "/api/gs"
        self.model, self.device = model, device
        self.period_s, self.timeout_s = period_s, timeout_s
        self._lock = threading.Lock()
        self._last: Optional[dict] = None
        self._times: Deque[float] = deque()          # arrival times, for frames/s
        self._infer: Deque[float] = deque(maxlen=50)
        self.frames = 0
        self.started = time.time()
        self._stop = threading.Event()
        self._online = None
        self._session = requests.Session()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gs-demo")

    def start(self) -> "GSReporter":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def frame(self, row: dict, detections: list) -> None:
        now = time.time()
        with self._lock:
            self.frames += 1
            self._times.append(now)
            self._infer.append(float(row.get("inference_ms", 0.0)))
            self._last = {**row, "detections": detections, "t": now}

    def _payload(self) -> dict:
        now = time.time()
        with self._lock:
            while self._times and now - self._times[0] > 2.0:
                self._times.popleft()
            inf = sorted(self._infer)
            return {"model": self.model, "device": self.device, "frames": self.frames,
                    "uptime_s": round(now - self.started, 1),
                    "fps": round(len(self._times) / 2.0, 2),
                    "inference_ms_p50": inf[len(inf) // 2] if inf else None,
                    "last": self._last, "t": now}

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._session.post(self.url, json=self._payload(), timeout=self.timeout_s).raise_for_status()
                if self._online is not True:
                    print(f"[ground-station] reporting to demo instance {self.url}", flush=True)
                self._online = True
            except requests.RequestException as e:
                if self._online is not False:
                    print(f"[ground-station] demo instance not reachable ({type(e).__name__}), "
                          "retrying in the background", flush=True)
                self._online = False
            self._stop.wait(self.period_s)
