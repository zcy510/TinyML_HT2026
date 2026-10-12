"""Communication-link monitoring on the UAV side (plan section 4.3).

A background thread probes the ground station's /ping endpoint every `interval_s`
seconds, so the link is measured even while the UAV is processing locally (that is
how the policy notices when the link has recovered). The monitor also takes the
outcome of every remote detection request (`report_frame`), which captures what a
small ping cannot: upload bandwidth and ground-station load.

Metrics in a :class:`LinkState` snapshot
* rtt_ms        EWMA of the probe round-trip time (a failed probe counts as `timeout_s`)
* loss          share of failed probes + failed remote frames in the last `window_s`
* frame_e2e_ms  median end-to-end latency of the remote frames in the last `window_s`
* rssi_dbm      Wi-Fi signal level: from `rssi_file` if a link emulator keeps it fresh,
                otherwise from /proc/net/wireless for `iface` (None if unavailable)
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple

import requests

DEFAULT_RSSI_FILE = "/tmp/uav_link_rssi"


@dataclass
class LinkState:
    rtt_ms: Optional[float]
    loss: float
    frame_e2e_ms: Optional[float]
    rssi_dbm: Optional[float]
    probes: int                       # probes in the window (0 = no information yet)

    def short(self) -> str:
        f = lambda v, fmt: "-" if v is None else format(v, fmt)
        return (f"rtt={f(self.rtt_ms, '.0f')}ms loss={self.loss:.0%} "
                f"e2e={f(self.frame_e2e_ms, '.0f')}ms rssi={f(self.rssi_dbm, '.0f')}dBm")


def read_rssi(iface: str = "wlan0", override_file: Optional[str] = DEFAULT_RSSI_FILE,
              max_age_s: float = 5.0, proc_path: str = "/proc/net/wireless") -> Optional[float]:
    """RSSI in dBm. An override file (written by tools.link_emulator) wins while it is
    fresh; otherwise /proc/net/wireless, where the line looks like
    ``wlan0: 0000   70.  -40.  -256        0      0      0      0      0        0``."""
    if override_file and os.path.exists(override_file):
        try:
            if time.time() - os.path.getmtime(override_file) <= max_age_s:
                with open(override_file) as f:
                    return float(f.read().strip())
        except (OSError, ValueError):
            pass
    try:
        with open(proc_path) as f:
            for line in f:
                if line.strip().startswith(f"{iface}:"):
                    level = float(line.split()[3].rstrip("."))
                    return level if level < 0 else level - 256   # some drivers report unsigned
    except (OSError, ValueError, IndexError):
        pass
    return None


class LinkMonitor:
    def __init__(self, server_url: Optional[str], interval_s: float = 0.5, timeout_s: float = 1.0,
                 alpha: float = 0.3, window_s: float = 5.0, iface: str = "wlan0",
                 rssi_file: Optional[str] = DEFAULT_RSSI_FILE,
                 probe: Optional[Callable[[], Optional[float]]] = None,
                 rssi: Optional[Callable[[], Optional[float]]] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.url = server_url.rstrip("/") if server_url else None
        self.interval_s, self.timeout_s, self.alpha, self.window_s = interval_s, timeout_s, alpha, window_s
        self.clock = clock
        self._session = requests.Session()        # own session: requests.Session is not thread-safe
        self._probe = probe or self._http_probe
        self._rssi = rssi or (lambda: read_rssi(iface, rssi_file))
        self._lock = threading.Lock()
        self._rtt: Optional[float] = None
        self._probes: Deque[Tuple[float, bool]] = deque()          # (t, ok)
        self._frames: Deque[Tuple[float, bool, float]] = deque()   # (t, ok, e2e_ms)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ----------------------------------------------------------------- probing
    def _http_probe(self) -> Optional[float]:
        if not self.url:
            return None
        t0 = time.perf_counter()
        try:
            self._session.get(f"{self.url}/ping", timeout=self.timeout_s).raise_for_status()
            return (time.perf_counter() - t0) * 1000.0
        except requests.RequestException:
            return None

    def probe_once(self) -> Optional[float]:
        rtt = self._probe()
        sample = rtt if rtt is not None else self.timeout_s * 1000.0
        with self._lock:
            self._rtt = sample if self._rtt is None else self.alpha * sample + (1 - self.alpha) * self._rtt
            self._probes.append((self.clock(), rtt is not None))
        return rtt

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            self.probe_once()
            self._stop.wait(max(0.0, self.interval_s - (time.monotonic() - t0)))

    def start(self, warmup: int = 3) -> "LinkMonitor":
        for _ in range(warmup):                   # first estimate before the first frame
            self.probe_once()
        self._thread = threading.Thread(target=self._run, daemon=True, name="link-monitor")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.timeout_s + 1)

    # ---------------------------------------------------------- frame outcomes
    def report_frame(self, ok: bool, e2e_ms: float) -> None:
        with self._lock:
            self._frames.append((self.clock(), ok, e2e_ms))

    def clear_frames(self) -> None:
        """Forget remote-frame statistics (call after a mode switch: they belong to the old mode)."""
        with self._lock:
            self._frames.clear()

    # ----------------------------------------------------------------- reading
    def snapshot(self) -> LinkState:
        now = self.clock()
        with self._lock:
            while self._probes and now - self._probes[0][0] > self.window_s:
                self._probes.popleft()
            while self._frames and now - self._frames[0][0] > self.window_s:
                self._frames.popleft()
            n_fail = sum(not ok for _, ok in self._probes) + sum(not ok for _, ok, _ in self._frames)
            n = len(self._probes) + len(self._frames)
            e2e = sorted(e for _, ok, e in self._frames if ok)
            state = LinkState(
                rtt_ms=self._rtt,
                loss=n_fail / n if n else 0.0,
                frame_e2e_ms=e2e[len(e2e) // 2] if e2e else None,
                rssi_dbm=None,
                probes=len(self._probes),
            )
        state.rssi_dbm = self._rssi()
        return state
