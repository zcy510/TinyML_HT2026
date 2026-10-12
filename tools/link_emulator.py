"""Trace-driven link emulation on the Raspberry Pi (plan section 4.3, "emulated environment").

Replays a link-quality trace (a simulated flight) with Linux `tc netem` so that the
remote-only, edge-only and adaptive strategies can be compared under *identical*,
repeatable conditions. Only traffic to the ground station is shaped (prio qdisc + u32
filter on its IP), so SSH and everything else on the Pi stay normal.

The trace also carries an RSSI value. netem cannot change the real Wi-Fi signal, so the
emulator writes the trace RSSI to /tmp/uav_link_rssi, which edge.link_monitor reads
instead of the real RSSI while the file is fresh. RSSI, delay, loss and bandwidth
therefore stay consistent with each other.

Trace CSV (step changes; each row holds until the next row's t_s; last row = end):
    t_s,rssi_dbm,delay_ms,jitter_ms,loss_pct,rate_mbit
    0,-50,0,0,0,0              # rate 0 = not limited
    20,-68,120,10,2,6
    ...

    sudo .venv/bin/python -m tools.link_emulator --trace traces/fly_away_and_back.csv \
        --ground 192.168.31.182 --iface wlan0
    python -m tools.link_emulator --trace traces/fly_away_and_back.csv --ground 1.2.3.4 --dry-run

Shaping applies to the Pi's *outgoing* packets towards the ground station (frame uploads
and ping requests): delay_ms is therefore added once per round trip, and rate_mbit
limits the upload. Every applied step is logged to --log (default logs/link_trace_applied.csv)
with wall-clock timestamps, for aligning with the pipeline log.
"""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List

RSSI_FILE = "/tmp/uav_link_rssi"


@dataclass
class Step:
    t_s: float
    rssi_dbm: float
    delay_ms: float
    jitter_ms: float
    loss_pct: float
    rate_mbit: float


def load_trace(path: str) -> List[Step]:
    steps = []
    with open(path) as f:
        rows = [r for r in csv.reader(line for line in f if line.strip() and not line.startswith("#"))]
    header = [h.strip() for h in rows[0]]
    for r in rows[1:]:
        d = dict(zip(header, (float(v.split("#")[0]) for v in r)))
        steps.append(Step(d["t_s"], d["rssi_dbm"], d["delay_ms"], d.get("jitter_ms", 0.0),
                          d["loss_pct"], d.get("rate_mbit", 0.0)))
    steps.sort(key=lambda s: s.t_s)
    if not steps or steps[0].t_s != 0:
        raise ValueError("trace must start at t_s = 0")
    return steps


def netem_args(s: Step) -> str:
    a = f"delay {s.delay_ms:g}ms"
    if s.jitter_ms > 0:
        a += f" {s.jitter_ms:g}ms distribution normal"
    if s.loss_pct > 0:
        a += f" loss {s.loss_pct:g}%"
    if s.rate_mbit > 0:
        a += f" rate {s.rate_mbit:g}mbit"
    return a


class Shaper:
    def __init__(self, iface: str, ground_ip: str, dry_run: bool = False) -> None:
        self.iface, self.ip, self.dry = iface, ground_ip, dry_run
        self.commands: List[str] = []

    def _tc(self, cmd: str, check: bool = True) -> None:
        full = f"tc {cmd}"
        self.commands.append(full)
        if self.dry:
            print(f"  $ {full}")
            return
        r = subprocess.run(shlex.split(full), capture_output=True, text=True)
        if r.returncode and check:
            raise RuntimeError(f"{full}\n{r.stderr.strip()}")

    def setup(self, first: Step) -> None:
        self._tc(f"qdisc del dev {self.iface} root", check=False)       # start clean
        # 3-band prio; priomap sends all normal traffic to band 2 (1:2), untouched
        self._tc(f"qdisc add dev {self.iface} root handle 1: prio bands 3 "
                 "priomap 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1")
        self._tc(f"qdisc add dev {self.iface} parent 1:3 handle 30: netem {netem_args(first)}")
        self._tc(f"filter add dev {self.iface} protocol ip parent 1:0 prio 1 u32 "
                 f"match ip dst {self.ip}/32 flowid 1:3")

    def apply(self, s: Step) -> None:
        self._tc(f"qdisc change dev {self.iface} parent 1:3 handle 30: netem {netem_args(s)}")

    def cleanup(self) -> None:
        self._tc(f"qdisc del dev {self.iface} root", check=False)


def write_rssi(v: float, path: str = RSSI_FILE) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"{v:g}\n")
    os.replace(tmp, path)                     # atomic: the reader never sees half a file
    try:
        os.chmod(path, 0o666)
    except OSError:
        pass


def run(steps: List[Step], shaper: Shaper, time_scale: float = 1.0, log_path: str = "",
        rssi_file: str = RSSI_FILE, refresh_s: float = 1.0) -> None:
    end_t = steps[-1].t_s * time_scale
    logf = None
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        logf = open(log_path, "w", newline="")
        w = csv.writer(logf)
        w.writerow(["timestamp", "trace_t_s", "rssi_dbm", "delay_ms", "jitter_ms", "loss_pct", "rate_mbit"])
    shaper.setup(steps[0])
    t0 = time.monotonic()
    i = -1
    try:
        while True:
            now = time.monotonic() - t0
            while i + 1 < len(steps) and steps[i + 1].t_s * time_scale <= now:
                i += 1
                s = steps[i]
                if i > 0:
                    shaper.apply(s)
                print(f"[link] t={now:6.1f}s  rssi {s.rssi_dbm:5.0f} dBm  {netem_args(s)}", flush=True)
                if logf:
                    csv.writer(logf).writerow([datetime.now().isoformat(timespec="milliseconds"),
                                               s.t_s, s.rssi_dbm, s.delay_ms, s.jitter_ms,
                                               s.loss_pct, s.rate_mbit])
                    logf.flush()
            write_rssi(steps[i].rssi_dbm, rssi_file)        # keep it fresh for the monitor
            if now >= end_t:
                break
            time.sleep(min(refresh_s, max(0.01, (steps[i + 1].t_s * time_scale - now)
                                          if i + 1 < len(steps) else refresh_s)))
    finally:
        shaper.cleanup()
        if logf:
            logf.close()
        try:
            os.remove(rssi_file)
        except OSError:
            pass
        print("[link] trace finished, shaping removed", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a link-quality trace with tc netem")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--ground", required=True, help="ground-station IP (only its traffic is shaped)")
    ap.add_argument("--iface", default="wlan0")
    ap.add_argument("--time-scale", type=float, default=1.0, help="0.5 = play the trace twice as fast")
    ap.add_argument("--log", default="logs/link_trace_applied.csv")
    ap.add_argument("--rssi-file", default=RSSI_FILE)
    ap.add_argument("--dry-run", action="store_true", help="print tc commands, do not run them")
    args = ap.parse_args()

    if not args.dry_run and os.geteuid() != 0:
        sys.exit("tc needs root: run with sudo (e.g. sudo .venv/bin/python -m tools.link_emulator ...)")
    ground = args.ground.split("://")[-1].split(":")[0].split("/")[0]
    steps = load_trace(args.trace)
    print(f"[link] {len(steps)} steps, {steps[-1].t_s * args.time_scale:.0f} s, shaping "
          f"{args.iface} -> {ground}{' (dry run)' if args.dry_run else ''}")
    shaper = Shaper(args.iface, ground, args.dry_run)

    def _stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)
    try:
        run(steps, shaper, args.time_scale, args.log, args.rssi_file)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
