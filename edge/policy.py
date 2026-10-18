"""Threshold-based policy engine (plan section 4.4, Table 1) with hysteresis.

Modes, best link first:
    remote_full     remote high-complexity processing (full-resolution frames)
    remote_reduced  reduced-resolution remote processing (e.g. 640 px, JPEG 70)
    local           local lightweight edge processing (INT8 model on the Pi)

Link classification (defaults = plan Table 1):
    poor      RTT > 250 ms  or  loss > 10 %  or  RSSI < -75 dBm   -> local
    good      RTT < 100 ms and loss <  5 % and (RSSI > -60 dBm or unknown) -> remote_full
    moderate  anything in between                                  -> remote_reduced
Latency rule: RTT and RSSI do not show a bandwidth bottleneck, so while in a remote
mode, if the median end-to-end time of recent remote frames exceeds the deadline, the
target is at least one level worse than the current mode.

Against oscillation near a threshold:
    * a worse target must persist `down_hold_s` (default 1 s), a better one `up_hold_s`
      (default 3 s) before the mode changes - degrading fast, recovering cautiously;
    * after any switch the mode is kept at least `min_dwell_s` (default 2 s);
    * an upgrade only goes one level at a time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from edge.link_monitor import LinkState

MODES = ["remote_full", "remote_reduced", "local"]


@dataclass
class PolicyConfig:
    rtt_good_ms: float = 100.0
    rtt_poor_ms: float = 250.0
    loss_good: float = 0.05
    loss_poor: float = 0.10
    rssi_good_dbm: float = -60.0
    rssi_poor_dbm: float = -75.0
    deadline_ms: float = 500.0
    down_hold_s: float = 1.0
    up_hold_s: float = 3.0
    min_dwell_s: float = 2.0
    use_rssi: bool = True


class Policy:
    def __init__(self, cfg: Optional[PolicyConfig] = None, initial: str = "remote_full",
                 modes: Optional[List[str]] = None) -> None:
        self.cfg = cfg or PolicyConfig()
        self.modes = modes or MODES                        # e.g. ["remote_full", "local"]
        self.level = self.modes.index(initial)
        self.last_switch_t: Optional[float] = None
        self._pending: Optional[Tuple[int, float]] = None  # (target level, since)
        self.switches = 0
        self.last_target: int = self.level                 # last classification (for the demo)
        self.last_reason: str = ""

    @property
    def mode(self) -> str:
        return self.modes[self.level]

    def _to_level(self, mode: str) -> int:
        """Map a mode of the full 3-level scheme onto this policy's (maybe 2-level) list."""
        if mode in self.modes:
            return self.modes.index(mode)
        return len(self.modes) - 1 if mode == "local" else 0

    def classify(self, s: LinkState) -> Tuple[int, str]:
        c = self.cfg
        rssi = s.rssi_dbm if c.use_rssi else None
        why = []
        if s.rtt_ms is None or s.rtt_ms > c.rtt_poor_ms:
            why.append(f"rtt {'-' if s.rtt_ms is None else f'{s.rtt_ms:.0f}'}>{c.rtt_poor_ms:.0f}")
        if s.loss > c.loss_poor:
            why.append(f"loss {s.loss:.0%}>{c.loss_poor:.0%}")
        if rssi is not None and rssi < c.rssi_poor_dbm:
            why.append(f"rssi {rssi:.0f}<{c.rssi_poor_dbm:.0f}")
        if why:
            target, reason = self._to_level("local"), "poor: " + ", ".join(why)
        elif (s.rtt_ms < c.rtt_good_ms and s.loss < c.loss_good
              and (rssi is None or rssi > c.rssi_good_dbm)):
            target, reason = self._to_level("remote_full"), "good link"
        else:
            target = self._to_level("remote_reduced")
            reason = f"moderate: rtt {s.rtt_ms:.0f}ms loss {s.loss:.0%}" + (
                f" rssi {rssi:.0f}" if rssi is not None else "")
        cur_mode = self.mode
        if (cur_mode != "local" and s.frame_e2e_ms is not None and s.frame_e2e_ms > c.deadline_ms
                and target <= self.level):
            target = min(self.level + 1, len(self.modes) - 1)
            reason = f"remote e2e {s.frame_e2e_ms:.0f}ms > deadline {c.deadline_ms:.0f}ms"
        return target, reason

    def update(self, s: LinkState, now: float) -> Tuple[str, bool, str]:
        """Returns (mode, switched, reason)."""
        target, reason = self.classify(s)
        self.last_target, self.last_reason = target, reason
        if self.last_switch_t is None:
            self.last_switch_t = now
        if target == self.level:
            self._pending = None
            return self.mode, False, reason
        worse = target > self.level
        if self._pending is None or (self._pending[0] > self.level) != worse:
            self._pending = (target, now)              # new direction: start the hold timer
        else:
            self._pending = (target, self._pending[1]) # same direction: keep timer, latest target
        hold = self.cfg.down_hold_s if worse else self.cfg.up_hold_s
        if now - self._pending[1] >= hold and now - self.last_switch_t >= self.cfg.min_dwell_s:
            since = self._pending[1]
            self.level = target if worse else self.level - 1     # recover one level at a time
            self.last_switch_t = now
            # still better than the new level: keep the timer, the next step only waits for the dwell
            self._pending = (target, since) if not worse and target < self.level else None
            self.switches += 1
            return self.mode, True, reason
        return self.mode, False, reason

    def status(self, now: float) -> dict:
        """What the policy is thinking right now (shown live by the demo dashboard)."""
        st = {"mode": self.mode, "target": self.modes[self.last_target], "reason": self.last_reason,
              "pending": None, "pending_s": 0.0, "hold_s": 0.0,
              "dwell_left_s": 0.0 if self.last_switch_t is None
              else max(0.0, self.cfg.min_dwell_s - (now - self.last_switch_t))}
        if self._pending is not None and self._pending[0] != self.level:
            worse = self._pending[0] > self.level
            st["pending"] = self.modes[self._pending[0] if worse else self.level - 1]
            st["pending_s"] = now - self._pending[1]
            st["hold_s"] = self.cfg.down_hold_s if worse else self.cfg.up_hold_s
        return st
