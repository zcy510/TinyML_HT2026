"""Adaptive detector: picks remote or local processing per frame from the link quality.

Same `detect()` interface as RemoteDetector / LocalDetector, so the pipeline in
run_remote.py stays the same. After every call, `row_info` holds the fields that go
into the log (mode, link metrics, switch reason, fallback).

Per frame:
    1. read the link state (LinkMonitor, updated in the background)
    2. Policy.update -> mode (with hysteresis)
    3. remote_full    -> ground station, full frame
       remote_reduced -> ground station, frame downscaled to `reduced_width`, JPEG `reduced_quality`
       local          -> on-board INT8 model
    4. if a remote request fails or times out (timeout = deadline), the same frame is
       processed locally right away ("fallback"), so the UAV still gets a result.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from edge.link_monitor import LinkMonitor
from edge.policy import Policy
from edge.remote_client import RemoteResult


class AdaptiveDetector:
    def __init__(self, remote, local, monitor: LinkMonitor, policy: Policy,
                 reduced_width: int = 640, reduced_quality: int = 70, full_quality: int = 90,
                 fallback: bool = True, verbose: bool = True,
                 full_imgsz: Optional[int] = None, reduced_imgsz: Optional[int] = None) -> None:
        self.remote, self.local, self.monitor, self.policy = remote, local, monitor, policy
        self.full_imgsz, self.reduced_imgsz = full_imgsz, reduced_imgsz
        self.reduced_width, self.reduced_quality, self.full_quality = reduced_width, reduced_quality, full_quality
        self.fallback, self.verbose = fallback, verbose
        self.row_info: dict = {}
        self.mode_frames = {m: 0 for m in policy.modes}
        self.fallbacks = 0

    def health(self) -> dict:
        return {"model": f"adaptive ({self.policy.mode} first)", "device": "remote + local"}

    def ping(self, n: int = 3) -> Optional[float]:
        return self.monitor.snapshot().rtt_ms

    def detect(self, frame_bgr: np.ndarray, frame_id: str = "", max_width: Optional[int] = None,
               jpeg_quality: int = 90, conf: Optional[float] = None) -> RemoteResult:
        state = self.monitor.snapshot()
        mode, switched, reason = self.policy.update(state, time.monotonic())
        if switched:
            self.monitor.clear_frames()
            if self.verbose:
                print(f"[policy] -> {mode:15s} ({reason}; {state.short()})")
        self.mode_frames[mode] += 1
        fallback = False
        if mode == "local":
            res = self.local.detect(frame_bgr, frame_id, conf=conf)
        else:
            w, q, sz = ((None, self.full_quality, self.full_imgsz) if mode == "remote_full"
                        else (self.reduced_width, self.reduced_quality, self.reduced_imgsz))
            res = self.remote.detect(frame_bgr, frame_id, w, q, conf, **({"imgsz": sz} if sz else {}))
            self.monitor.report_frame(res.ok, res.e2e_ms)
            if not res.ok and self.fallback:
                loc = self.local.detect(frame_bgr, frame_id, conf=conf)
                loc.e2e_ms += res.e2e_ms                     # time lost on the failed attempt counts
                loc.upload_bytes, loc.error = res.upload_bytes, f"fallback after {res.error}"
                res, fallback = loc, True
                self.fallbacks += 1
        self.row_info = {
            "mode": mode, "fallback": fallback, "switch": reason if switched else "",
            "link_rtt_ms": round(state.rtt_ms, 1) if state.rtt_ms is not None else "",
            "link_loss": round(state.loss, 3),
            "link_e2e_ms": round(state.frame_e2e_ms, 1) if state.frame_e2e_ms is not None else "",
            "rssi_dbm": state.rssi_dbm if state.rssi_dbm is not None else "",
        }
        return res
