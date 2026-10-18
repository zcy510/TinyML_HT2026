"""Small IoU tracker for the demo dashboard: gives detections stable ids and trails.

Only for visualisation. The UAV pipeline itself is detection-only; frames arrive at the
processed rate (dropped frames in between), so matching is greedy on IoU with a
centre-distance fallback for fast motion, per class.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _center(b) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


@dataclass
class Track:
    id: int
    label: str
    class_id: int
    box: List[float]
    confidence: float
    last_t: float
    hits: int = 1
    trail: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=30))


class IoUTracker:
    def __init__(self, iou_thr: float = 0.2, max_age_s: float = 1.0, dist_factor: float = 0.75) -> None:
        self.iou_thr, self.max_age_s, self.dist_factor = iou_thr, max_age_s, dist_factor
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    def update(self, detections: List[dict], t: float) -> List[dict]:
        for tid in [k for k, tr in self.tracks.items() if t - tr.last_t > self.max_age_s]:
            del self.tracks[tid]
        dets = [d for d in detections if len(d.get("box", [])) == 4]
        pairs = []
        for di, d in enumerate(dets):
            for tid, tr in self.tracks.items():
                if tr.class_id != d.get("class_id"):
                    continue
                v = iou(tr.box, d["box"])
                if v >= self.iou_thr:
                    pairs.append((v, di, tid))
                else:                                    # fast motion / low fps: centre distance
                    (cx, cy), (dx, dy) = _center(tr.box), _center(d["box"])
                    size = max(tr.box[2] - tr.box[0], tr.box[3] - tr.box[1], 1.0)
                    dist = ((cx - dx) ** 2 + (cy - dy) ** 2) ** 0.5
                    if dist < self.dist_factor * size:
                        pairs.append((self.iou_thr * (1 - dist / (self.dist_factor * size)) * 0.99, di, tid))
        pairs.sort(reverse=True)
        used_d, used_t, assign = set(), set(), {}
        for _, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            assign[di] = tid
        out = []
        for di, d in enumerate(dets):
            if di in assign:
                tr = self.tracks[assign[di]]
                tr.box, tr.confidence, tr.last_t = list(d["box"]), d.get("confidence", 0.0), t
                tr.hits += 1
            else:
                tr = Track(self.next_id, d.get("label", "?"), d.get("class_id", -1), list(d["box"]),
                           d.get("confidence", 0.0), t)
                self.tracks[tr.id] = tr
                self.next_id += 1
            tr.trail.append(_center(tr.box))
            out.append({"id": tr.id, "label": tr.label, "class_id": tr.class_id,
                        "confidence": tr.confidence, "box": tr.box, "hits": tr.hits,
                        "trail": [[round(x, 1), round(y, 1)] for x, y in tr.trail]})
        return out
