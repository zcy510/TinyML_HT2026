"""UAV-side (Raspberry Pi 5) client for the remote high-complexity mode.

Encodes a frame as JPEG, optionally downscales it (used later by the policy
engine for the "moderate link" reduced-resolution mode), sends it to the ground
station and measures end-to-end latency, upload size and network time.

Only needs: numpy, opencv-python(-headless), requests.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests


@dataclass
class RemoteResult:
    ok: bool
    frame_id: str
    detections: List[dict] = field(default_factory=list)
    e2e_ms: float = 0.0          # encode + upload + server + download
    encode_ms: float = 0.0
    network_ms: float = 0.0      # e2e - encode - server_ms (transfer + queueing)
    server_ms: float = 0.0
    inference_ms: float = 0.0
    upload_bytes: int = 0
    sent_size: Tuple[int, int] = (0, 0)   # (w, h) actually transmitted
    scale: float = 1.0                    # sent / original, to map boxes back
    error: str = ""


class RemoteDetector:
    def __init__(self, server_url: str, timeout_s: float = 2.0) -> None:
        self.url = server_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()   # keep-alive: avoids a TCP handshake per frame

    # ------------------------------------------------------------------ link
    def health(self) -> dict:
        r = self.session.get(f"{self.url}/health", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def ping(self, n: int = 3) -> Optional[float]:
        """Application-level RTT in ms (median of n), None if unreachable."""
        rtts = []
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                self.session.get(f"{self.url}/ping", timeout=self.timeout_s).raise_for_status()
                rtts.append((time.perf_counter() - t0) * 1000.0)
            except requests.RequestException:
                pass
        return float(np.median(rtts)) if rtts else None

    # ------------------------------------------------------------- detection
    @staticmethod
    def encode(frame_bgr: np.ndarray, max_width: Optional[int], jpeg_quality: int):
        h, w = frame_bgr.shape[:2]
        scale = 1.0
        if max_width and w > max_width:
            scale = max_width / float(w)
            frame_bgr = cv2.resize(frame_bgr, (max_width, int(round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buf.tobytes(), (frame_bgr.shape[1], frame_bgr.shape[0]), scale

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_id: str = "",
        max_width: Optional[int] = None,
        jpeg_quality: int = 90,
        conf: Optional[float] = None,
    ) -> RemoteResult:
        t0 = time.perf_counter()
        data, sent_size, scale = self.encode(frame_bgr, max_width, jpeg_quality)
        encode_ms = (time.perf_counter() - t0) * 1000.0

        params = {"frame_id": frame_id}
        if conf is not None:
            params["conf"] = conf
        try:
            r = self.session.post(
                f"{self.url}/detect", data=data, params=params,
                headers={"Content-Type": "image/jpeg"}, timeout=self.timeout_s,
            )
            r.raise_for_status()
            body = r.json()
        except (requests.RequestException, ValueError) as e:
            return RemoteResult(False, frame_id, e2e_ms=(time.perf_counter() - t0) * 1000.0,
                                encode_ms=encode_ms, upload_bytes=len(data),
                                sent_size=sent_size, scale=scale, error=type(e).__name__)
        e2e_ms = (time.perf_counter() - t0) * 1000.0

        # Map boxes back to the original frame resolution
        dets = body["detections"]
        if scale != 1.0:
            for d in dets:
                d["box"] = [round(v / scale, 1) for v in d["box"]]
        timing = body.get("timing", {})
        server_ms = float(timing.get("server_ms", 0.0))
        return RemoteResult(
            ok=True, frame_id=frame_id, detections=dets, e2e_ms=e2e_ms,
            encode_ms=encode_ms, network_ms=max(e2e_ms - encode_ms - server_ms, 0.0),
            server_ms=server_ms, inference_ms=float(timing.get("inference_ms", 0.0)),
            upload_bytes=len(data), sent_size=sent_size, scale=scale,
        )


def draw_detections(frame_bgr: np.ndarray, detections: List[dict]) -> np.ndarray:
    out = frame_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f'{d["label"]} {d["confidence"]:.2f}', (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    return out
