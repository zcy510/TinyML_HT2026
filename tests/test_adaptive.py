"""Tests for link monitoring, the policy engine, the adaptive detector and the link emulator."""
import os
import socket
import threading
import time

import numpy as np
import pandas as pd
import pytest

from edge.adaptive import AdaptiveDetector
from edge.link_monitor import LinkMonitor, LinkState, read_rssi
from edge.policy import Policy, PolicyConfig
from edge.remote_client import RemoteDetector, RemoteResult


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def state(rtt=20.0, loss=0.0, e2e=None, rssi=None):
    return LinkState(rtt_ms=rtt, loss=loss, frame_e2e_ms=e2e, rssi_dbm=rssi, probes=5)


# ------------------------------------------------------------------ monitor
def test_read_rssi_proc_and_override(tmp_path):
    proc = tmp_path / "wireless"
    proc.write_text("Inter-| sta-|   Quality        |   Discarded packets\n"
                    " face | tus | link level noise |  nwid  crypt   frag  retry   misc\n"
                    " wlan0: 0000   54.  -56.  -256        0      0      0      0     12        0\n")
    assert read_rssi("wlan0", None, proc_path=str(proc)) == -56.0
    assert read_rssi("eth0", None, proc_path=str(proc)) is None
    ov = tmp_path / "rssi"
    ov.write_text("-81\n")
    assert read_rssi("wlan0", str(ov), proc_path=str(proc)) == -81.0          # emulator wins
    os.utime(ov, (time.time() - 60, time.time() - 60))
    assert read_rssi("wlan0", str(ov), proc_path=str(proc)) == -56.0          # stale -> real


def test_monitor_ewma_loss_and_window():
    clk, rtts = Clock(), iter([10.0, 10.0, None, 30.0])
    m = LinkMonitor(None, timeout_s=0.5, alpha=0.5, window_s=5.0, probe=lambda: next(rtts),
                    rssi=lambda: -60.0, clock=clk)
    for _ in range(4):
        m.probe_once()
        clk.t += 1
    s = m.snapshot()
    # EWMA: 10 -> 10 -> (500+10)/2=255 -> (30+255)/2=142.5 ; timeout counts as 500 ms
    assert s.rtt_ms == pytest.approx(142.5) and s.loss == pytest.approx(0.25) and s.rssi_dbm == -60
    m.report_frame(True, 300.0)
    m.report_frame(True, 100.0)
    m.report_frame(False, 500.0)
    s = m.snapshot()
    assert s.frame_e2e_ms == 300.0 and s.loss == pytest.approx(2 / 7)
    clk.t += 10                                                  # everything leaves the window
    s = m.snapshot()
    assert s.loss == 0 and s.frame_e2e_ms is None and s.probes == 0


# ------------------------------------------------------------------- policy
def test_classify_follows_plan_table():
    p = Policy()
    assert p.modes[p.classify(state(rtt=40))[0]] == "remote_full"
    assert p.modes[p.classify(state(rtt=150))[0]] == "remote_reduced"
    assert p.modes[p.classify(state(rtt=40, loss=0.07))[0]] == "remote_reduced"
    assert p.modes[p.classify(state(rtt=40, rssi=-70))[0]] == "remote_reduced"
    assert p.modes[p.classify(state(rtt=300))[0]] == "local"
    assert p.modes[p.classify(state(rtt=40, loss=0.2))[0]] == "local"
    assert p.modes[p.classify(state(rtt=40, rssi=-80))[0]] == "local"
    assert p.modes[p.classify(state(rtt=None))[0]] == "local"
    no_rssi = Policy(PolicyConfig(use_rssi=False))
    assert no_rssi.modes[no_rssi.classify(state(rtt=40, rssi=-80))[0]] == "remote_full"


def test_latency_rule_steps_down_when_remote_misses_deadline():
    p = Policy()
    lvl, why = p.classify(state(rtt=30, e2e=800))               # good RTT, but frames too slow
    assert p.modes[lvl] == "remote_reduced" and "deadline" in why
    p.level = 1
    assert p.modes[p.classify(state(rtt=30, e2e=800))[0]] == "local"
    p.level = 2                                                   # local: no remote frames -> rule off
    assert p.modes[p.classify(state(rtt=30, e2e=800))[0]] == "remote_full"


def test_hysteresis_fast_down_slow_up_one_level():
    p = Policy(PolicyConfig(down_hold_s=1.0, up_hold_s=3.0, min_dwell_s=2.0))
    t = 0.0
    p.update(state(rtt=20), t)
    for t in (2.0, 2.5):                                          # poor for 0.5 s: hold
        assert p.update(state(rtt=400), t)[0] == "remote_full"
    mode, switched, _ = p.update(state(rtt=400), 3.0)            # poor for 1 s: down
    assert (mode, switched) == ("local", True)
    for t in (3.5, 4.0, 5.0, 6.0):                                # good, but < 3 s
        assert p.update(state(rtt=20), t)[0] == "local"
    assert p.update(state(rtt=20), 6.5)[:2] == ("remote_reduced", True)   # one level up
    assert p.update(state(rtt=20), 8.0)[0] == "remote_reduced"
    assert p.update(state(rtt=20), 9.5)[:2] == ("remote_full", True)
    assert p.switches == 3


def test_flapping_link_does_not_flap_policy():
    p = Policy()
    switches = []
    for i in range(40):                                           # 2 s good / 2 s moderate, 20 s
        t = i * 0.5
        rtt = 60 if (int(t) // 2) % 2 == 0 else 140
        switches.append(p.update(state(rtt=rtt), t)[1])
    assert sum(switches) <= 1 and p.mode == "remote_reduced"


def test_dwell_time_blocks_quick_second_switch():
    p = Policy(PolicyConfig(down_hold_s=0.0, min_dwell_s=2.0))
    p.update(state(rtt=20), 0.0)
    assert p.update(state(rtt=150), 2.0)[:2] == ("remote_reduced", True)
    assert p.update(state(rtt=400), 2.5)[:2] == ("remote_reduced", False)   # dwell
    assert p.update(state(rtt=400), 4.0)[:2] == ("local", True)


def test_two_mode_policy():
    p = Policy(modes=["remote_full", "local"])
    assert p.modes[p.classify(state(rtt=150))[0]] == "remote_full"           # moderate -> best remote
    assert p.modes[p.classify(state(rtt=400))[0]] == "local"


# --------------------------------------------------------- adaptive detector
class FakeRemote:
    def __init__(self):
        self.ok, self.calls, self.imgsz = True, [], []

    def detect(self, frame, fid="", max_width=None, jpeg_quality=90, conf=None, imgsz=None):
        self.calls.append((max_width, jpeg_quality))
        self.imgsz.append(imgsz)
        if not self.ok:
            return RemoteResult(False, fid, e2e_ms=500.0, upload_bytes=1000, error="ReadTimeout")
        return RemoteResult(True, fid, [{"class_id": 2, "label": "car", "confidence": 0.9,
                                         "box": [0, 0, 1, 1]}], e2e_ms=80.0, upload_bytes=1000)


class FakeLocal:
    def __init__(self):
        self.calls = 0

    def detect(self, frame, fid="", max_width=None, jpeg_quality=0, conf=None):
        self.calls += 1
        return RemoteResult(True, fid, [], e2e_ms=30.0, inference_ms=25.0)


def test_adaptive_routes_by_mode_and_falls_back():
    clk = Clock()
    rtt = {"v": 20.0}
    mon = LinkMonitor(None, probe=lambda: rtt["v"], rssi=lambda: None, clock=clk, alpha=1.0)
    mon.probe_once()
    pol = Policy(PolicyConfig(down_hold_s=0, up_hold_s=0, min_dwell_s=0))
    remote, local = FakeRemote(), FakeLocal()
    ad = AdaptiveDetector(remote, local, mon, pol, reduced_width=640, reduced_quality=70, verbose=False,
                          full_imgsz=1280, reduced_imgsz=640)
    img = np.zeros((10, 10, 3), np.uint8)

    ad.detect(img, "1")
    assert ad.row_info["mode"] == "remote_full" and remote.calls[-1] == (None, 90)
    rtt["v"] = 150.0; mon.probe_once()
    ad.detect(img, "2")
    assert ad.row_info["mode"] == "remote_reduced" and remote.calls[-1] == (640, 70)
    assert ad.row_info["switch"].startswith("moderate")
    remote.ok = False                                               # request fails -> local fallback
    res = ad.detect(img, "3")
    assert res.ok and ad.row_info["fallback"] and local.calls == 1 and res.e2e_ms == 530.0
    rtt["v"] = 400.0; mon.probe_once()
    ad.detect(img, "4")
    assert ad.row_info["mode"] == "local" and local.calls == 2 and len(remote.calls) == 3
    assert remote.imgsz == [1280, 640, 640]          # inference size follows the mode


def test_remote_detector_sends_imgsz(monkeypatch):
    sent = []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"detections": [], "timing": {}}

    rd = RemoteDetector("http://gs:8000", imgsz=1280)
    monkeypatch.setattr(rd.session, "post", lambda url, **kw: sent.append(kw["params"]) or Resp())
    img = np.zeros((48, 64, 3), np.uint8)
    rd.detect(img, "a")
    rd.detect(img, "b", imgsz=640)
    RemoteDetector("http://gs:8000").session                    # default: no imgsz sent
    rd2 = RemoteDetector("http://gs:8000")
    monkeypatch.setattr(rd2.session, "post", lambda url, **kw: sent.append(kw["params"]) or Resp())
    rd2.detect(img, "c")
    assert sent[0]["imgsz"] == 1280 and sent[1]["imgsz"] == 640 and "imgsz" not in sent[2]


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_adaptive_over_http_goes_local_when_ground_station_dies():
    import uvicorn
    from ground_station.server import create_app
    from tests.test_remote_pipeline import FakeDetector

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(FakeDetector(), None), host="127.0.0.1",
                                           port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    url = f"http://127.0.0.1:{port}"
    mon = LinkMonitor(url, interval_s=0.05, timeout_s=0.2, rssi=lambda: None).start()
    pol = Policy(PolicyConfig(down_hold_s=0.2, min_dwell_s=0.2))
    local = FakeLocal()
    ad = AdaptiveDetector(RemoteDetector(url, 0.3), local, mon, pol, verbose=False)
    img = np.random.randint(0, 255, (120, 160, 3), np.uint8)
    try:
        r = ad.detect(img, "a")
        assert ad.row_info["mode"] == "remote_full" and r.ok and r.detections   # remote result
        server.should_exit = True
        th.join(timeout=5)
        modes, fb = [], 0
        t_end = time.time() + 3
        while time.time() < t_end and (not modes or modes[-1] != "local"):
            r = ad.detect(img, "b")
            modes.append(ad.row_info["mode"])
            fb += ad.row_info["fallback"]
            assert r.ok                                            # every frame still has a result
            time.sleep(0.05)
        assert modes[-1] == "local" and fb >= 1
    finally:
        mon.stop()
        server.should_exit = True


# ------------------------------------------------------- loops and emulator
def test_loop_ids_and_eval_per_repetition(tmp_path):
    from edge.frame_source import iter_frames
    from tests.test_video_visdrone import make_sequence
    from tools.eval_detection import evaluate
    from tools.visdrone import EVAL_CLASSES, convert_sequence

    seq = make_sequence(tmp_path, n=3)
    ids = [f.frame_id for f in iter_frames(str(seq), loop=2)]
    assert ids == ["0000001", "0000002", "0000003", "0000001#1", "0000002#1", "0000003#1"]
    convert_sequence(seq, tmp_path / "labels")
    car = {"class_id": 2, "label": "car", "confidence": 0.9, "box": [10, 10, 30, 30]}
    pred = {"0000001": [car], "0000002": [car], "0000003": [car], "0000001#1": [car]}
    r = evaluate(pred, seq, tmp_path / "labels", classes=set(EVAL_CLASSES), hold_last=True)
    # repetition 0: 3 frames; repetition 1 cut short after frame 1 -> 1 frame
    assert r["repetitions"] == 2 and r["frames"] == 4 and r["tp"] == 4 and r["fn"] == 4


def test_link_emulator_trace_and_dry_run(tmp_path):
    from tools.link_emulator import Shaper, load_trace, netem_args, run

    tr = tmp_path / "t.csv"
    tr.write_text("# c\nt_s,rssi_dbm,delay_ms,jitter_ms,loss_pct,rate_mbit\n"
                  "0,-50,0,0,0,0\n0.05,-80,300,20,10,1   # poor\n0.1,-50,0,0,0,0\n")
    steps = load_trace(str(tr))
    assert len(steps) == 3 and steps[1].rssi_dbm == -80
    assert netem_args(steps[1]) == "delay 300ms 20ms distribution normal loss 10% rate 1mbit"
    sh = Shaper("wlan0", "10.0.0.2", dry_run=True)
    rssi, log = tmp_path / "rssi", tmp_path / "applied.csv"
    run(steps, sh, 1.0, str(log), str(rssi), refresh_s=0.01)
    assert any("match ip dst 10.0.0.2/32 flowid 1:3" in c for c in sh.commands)
    assert sh.commands[-1] == "tc qdisc del dev wlan0 root"           # shaping removed
    assert not rssi.exists()                                            # override removed
    df = pd.read_csv(log)
    assert list(df["rssi_dbm"]) == [-50, -80, -50]
    for f in ("traces/fly_away_and_back.csv", "traces/flaky_border.csv", "traces/bandwidth_drop.csv"):
        assert load_trace(os.path.join(os.path.dirname(__file__), "..", f))


def test_summarize_log_reports_modes_and_switches():
    from tools.summarize_log import summarize

    rows = []
    for i, (m, sw, fb) in enumerate([("remote_full", "", False), ("remote_full", "", False),
                                     ("remote_reduced", "moderate", False), ("local", "poor", True)]):
        rows.append({"timestamp": f"2026-10-12T10:00:0{i}.000", "setting": "adaptive_rt", "ok": True,
                     "e2e_ms": 100, "inference_ms": 10, "network_ms": 5, "upload_bytes": 1000,
                     "num_detections": 3, "sent_w": 640, "sent_h": 360, "dropped": 1, "age_ms": 120,
                     "mode": m, "switch": sw, "fallback": fb})
    out = summarize(pd.DataFrame(rows), 500).iloc[0]
    assert out["%remote_full"] == 0.5 and out["%local"] == 0.25
    assert out["switches"] == 2 and out["fallbacks"] == 1
