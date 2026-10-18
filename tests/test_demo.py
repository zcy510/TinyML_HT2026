"""Tests for the live demo: link shaper, demo reporter <-> demo server, tracker, policy status."""
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from demo.server import TRACES, DemoState, load_trace, make_handler
from demo.tracker import IoUTracker
from edge.demo_link import DemoReporter, LinkSettings, LinkShaper, ShapedRemote, attach_shaper
from edge.link_monitor import LinkMonitor, LinkState
from edge.policy import Policy
from edge.remote_client import RemoteResult


class Sleeps(list):
    def __call__(self, s):
        self.append(s)


def shaper_with(**kw):
    sl = Sleeps()
    return LinkShaper(LinkSettings(**kw), seed=1, sleep=sl), sl


# ------------------------------------------------------------------ shaper
def test_probe_delay_loss_and_timeout():
    sh, sl = shaper_with(delay_ms=120)
    probe = sh.wrap_probe(lambda: 5.0, timeout_s=0.5)
    assert probe() == pytest.approx(125.0) and sl[-1] == pytest.approx(0.12)
    sh.set(LinkSettings(delay_ms=600))                 # reply would come after the timeout
    assert probe() is None
    sh.set(LinkSettings(loss_pct=100))
    assert probe() is None and sl[-1] == 0.5           # lost: waits the whole timeout
    sh.set(LinkSettings())
    assert probe() == 5.0


def test_rssi_override_and_monitor_hook():
    sh, _ = shaper_with()
    m = LinkMonitor(None, timeout_s=0.5, probe=lambda: 10.0, rssi=lambda: -40.0)
    attach_shaper(m, sh)
    m.probe_once()
    assert m.snapshot().rssi_dbm == -40.0              # None = real value
    sh.set(LinkSettings(rssi_dbm=-82, delay_ms=0))
    assert m.snapshot().rssi_dbm == -82.0


class FakeRemote:
    timeout_s = 0.5

    def encode(self, frame, w, q):
        return b"x" * 1000, (64, 48), 1.0

    def detect(self, frame, fid="", w=None, q=90, conf=None):
        return RemoteResult(True, fid, [{"box": [0, 0, 1, 1]}], e2e_ms=20.0, network_ms=5.0,
                            upload_bytes=50_000, sent_size=(64, 48))

    def health(self):
        return {"model": "fake", "device": "cpu"}


def test_shaped_remote_delay_rate_loss():
    frame = np.zeros((48, 64, 3), np.uint8)
    sh, sl = shaper_with(delay_ms=100, rate_mbit=4)    # 50 kB at 4 Mbit/s = 100 ms
    r = ShapedRemote(FakeRemote(), sh).detect(frame, "f")
    assert r.ok and r.e2e_ms == pytest.approx(220.0) and r.network_ms == pytest.approx(205.0)
    sh.set(LinkSettings(rate_mbit=0.5))                # 800 ms upload > 500 ms timeout
    r = ShapedRemote(FakeRemote(), sh).detect(frame, "f")
    assert not r.ok and "Timeout" in r.error and r.upload_bytes == 50_000
    sh.set(LinkSettings(loss_pct=100))
    r = ShapedRemote(FakeRemote(), sh).detect(frame, "f")
    assert not r.ok and "loss" in r.error


def test_link_settings_from_dict_clamps():
    s = LinkSettings.from_dict({"delay_ms": -5, "loss_pct": 250, "rssi_dbm": "", "rate_mbit": "2"})
    assert s.delay_ms == 0 and s.loss_pct == 100 and s.rssi_dbm is None and s.rate_mbit == 2


# ----------------------------------------------------------------- tracker
def test_tracker_keeps_ids_and_trails():
    t = IoUTracker()
    a = t.update([{"class_id": 2, "label": "car", "box": [0, 0, 10, 10]}], 0.0)
    b = t.update([{"class_id": 2, "label": "car", "box": [4, 0, 14, 10]},       # moved
                  {"class_id": 0, "label": "person", "box": [50, 50, 55, 60]}], 0.1)
    assert a[0]["id"] == b[0]["id"] and b[1]["id"] != b[0]["id"] and len(b[0]["trail"]) == 2
    c = t.update([{"class_id": 2, "label": "car", "box": [100, 100, 110, 110]}], 2.0)
    assert c[0]["id"] not in (a[0]["id"], b[1]["id"])                          # old tracks expired


# ------------------------------------------------------------------ policy
def test_policy_status_shows_pending_switch():
    p = Policy()
    st = lambda rtt: LinkState(rtt_ms=rtt, loss=0.0, frame_e2e_ms=None, rssi_dbm=None, probes=5)
    p.update(st(20), 0.0)
    p.update(st(400), 3.0)
    s = p.status(3.5)
    assert s["mode"] == "remote_full" and s["target"] == "local" and s["pending"] == "local"
    assert s["pending_s"] == pytest.approx(0.5) and s["hold_s"] == 1.0
    p.update(st(400), 4.1)
    assert p.status(4.1)["mode"] == "local" and p.status(4.1)["pending"] is None


# ------------------------------------------------------- reporter <-> server
@pytest.fixture
def demo_server():
    state = DemoState()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield state, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=2).read())


def wait_for(cond, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_reporter_streams_frames_and_receives_link(demo_server):
    state, url = demo_server
    post(url + "/api/link", {"delay_ms": 200, "loss_pct": 5})
    sh = LinkShaper()
    rep = DemoReporter(url, sh, preview_width=32, max_fps=50, poll_s=0.05).start({"host": "test", "mode": "adaptive"})
    assert sh.settings.delay_ms == 200 and state.hello["host"] == "test"   # applied at hello
    frame = np.full((48, 64, 3), 120, np.uint8)
    for i in range(3):
        rep.submit(frame, {"t": time.time(), "frame_id": str(i), "mode": "local",
                           "detections": [{"class_id": 2, "label": "car", "box": [1, 1, 20, 20]}]})
    assert wait_for(lambda: state.frames_total == 3)
    assert state.last_frames and '"jpeg"' in state.last_frames
    post(url + "/api/link", {"preset": "poor"})                            # dashboard change
    assert wait_for(lambda: sh.settings.loss_pct == 15)
    rep.stop({"frames": 3})
    assert state.edge_summary == {"frames": 3}


def test_trace_playback_and_manual_override(demo_server):
    state, url = demo_server
    for p in TRACES.glob("*.csv"):
        steps = load_trace(p)
        assert steps[0]["t_s"] == 0 and all("delay_ms" in s for s in steps)
    r = post(url + "/api/trace", {"name": "fly_away_and_back.csv", "speed": 1})
    assert r["steps"] == 8
    assert wait_for(lambda: state.link_by == "trace" and state.link["rssi_dbm"] == -45)
    post(url + "/api/link", {"delay_ms": 10})                             # manual change stops it
    assert wait_for(lambda: state.trace is None)
    assert state.link["delay_ms"] == 10
