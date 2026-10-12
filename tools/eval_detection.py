"""Detection-performance evaluation (precision / recall / F1 at an IoU threshold).

Compares the detections saved by `edge.run_remote --save-json` (or later by the
edge pipeline) against YOLO-format ground-truth labels
(`labels/<image>.txt`, rows: class cx cy w h, normalised).

python -m tools.eval_detection --pred logs/remote_dets.json \
    --images datasets/coco128/images/train2017 --labels datasets/coco128/labels/train2017

Video / VisDrone-VID (labels from ``tools.visdrone convert``):

python -m tools.eval_detection --pred logs/remote_dets.json \
    --images data/VisDrone2019-VID-val/sequences/uav0000086_00000_v \
    --labels data/visdrone_labels/uav0000086_00000_v --classes 0,1,2,3,5,7 --hold-last

Label rows with class -1 are ignore regions: an unmatched prediction whose centre
lies inside one is not a false positive. ``--classes`` scores only those class ids
(predictions of other classes are dropped). ``--hold-last`` scores *every* frame of
the sequence, using the most recent processed frame's detections for frames that
were dropped in --realtime mode: this is what the UAV actually knows at each
moment, so it penalises both missed and dropped frames. Without it, only the
processed frames are scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iou(a: List[float], b: List[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_yolo_labels(path: Path, w: int, h: int) -> List[dict]:
    gts = []
    if not path.exists():
        return gts
    for line in path.read_text().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
        gts.append({"class_id": c, "box": [(cx - bw / 2) * w, (cy - bh / 2) * h,
                                           (cx + bw / 2) * w, (cy + bh / 2) * h]})
    return gts


IGNORE = -1


def _inside(box: List[float], region: List[float]) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def match_frame(preds: List[dict], gts: List[dict], iou_thr: float):
    """Greedy matching by confidence, class-aware. Returns (tp, fp, fn).
    Ground truths with class_id -1 are ignore regions (see module doc)."""
    ignores = [g["box"] for g in gts if g["class_id"] == IGNORE]
    gts = [g for g in gts if g["class_id"] != IGNORE]
    used = set()
    tp = fp = 0
    for p in sorted(preds, key=lambda d: -d["confidence"]):
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in used or g["class_id"] != p["class_id"]:
                continue
            v = iou(p["box"], g["box"])
            if v > best:
                best, best_j = v, j
        if best >= iou_thr:
            used.add(best_j)
            tp += 1
        elif not any(_inside(p["box"], r) for r in ignores):
            fp += 1
    return tp, fp, len(gts) - tp


def _frame_list(images: Optional[Path], labels: Path) -> List[Tuple[str, Optional[Path]]]:
    if images is not None:
        return [(p.stem, p) for p in sorted(images.iterdir()) if p.suffix.lower() in IMG_EXT]
    return [(p.stem, None) for p in sorted(labels.glob("*.txt"))]


def evaluate(pred: Dict[str, List[dict]], images: Optional[Path], labels: Path,
             iou_thr: float = 0.5, min_conf: float = 0.0, classes: Optional[Set[int]] = None,
             hold_last: bool = False, img_size: Optional[Tuple[int, int]] = None) -> dict:
    """Looped runs (--loop: ids '0000001#2') are scored per repetition and summed."""
    if any("#" in k for k in pred):
        loops: Dict[int, Dict[str, List[dict]]] = {}
        for k, v in pred.items():
            stem, _, rep = k.partition("#")
            loops.setdefault(int(rep or 0), {})[stem] = v
        reps = [p for _, p in sorted(loops.items())]
        # the last repetition of a looped run usually stops part-way (--max-seconds):
        # score it only up to its last processed frame
        parts = [_evaluate_once(p, images, labels, iou_thr, min_conf, classes, hold_last, img_size,
                                truncate=(i == len(reps) - 1)) for i, p in enumerate(reps)]
        tot = {k: sum(r[k] for r in parts) for k in ("frames", "frames_held", "tp", "fp", "fn")}
        P = tot["tp"] / (tot["tp"] + tot["fp"]) if tot["tp"] + tot["fp"] else 0.0
        R = tot["tp"] / (tot["tp"] + tot["fn"]) if tot["tp"] + tot["fn"] else 0.0
        return {**tot, "precision": P, "recall": R, "f1": 2 * P * R / (P + R) if P + R else 0.0,
                "iou_threshold": iou_thr, "repetitions": len(parts)}
    return _evaluate_once(pred, images, labels, iou_thr, min_conf, classes, hold_last, img_size)


def _evaluate_once(pred, images, labels, iou_thr, min_conf, classes, hold_last, img_size,
                   truncate: bool = False) -> dict:
    TP = FP = FN = 0
    frames = held = 0
    last: Optional[List[dict]] = None
    size = img_size
    frame_list = _frame_list(images, labels)
    if truncate:
        last = max((i for i, (stem, _) in enumerate(frame_list) if stem in pred), default=-1)
        frame_list = frame_list[: last + 1]
    for stem, img_path in frame_list:
        dets = pred.get(stem)
        if dets is None:
            if not hold_last or last is None:
                continue
            dets, held = last, held + 1       # frame was dropped: reuse the last result
        else:
            last = dets
        if img_size is None:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            size = (img.shape[1], img.shape[0])
        w, h = size
        gts = load_yolo_labels(labels / f"{stem}.txt", w, h)
        preds = [d for d in dets if d["confidence"] >= min_conf]
        if classes is not None:
            preds = [d for d in preds if d["class_id"] in classes]
            gts = [g for g in gts if g["class_id"] in classes or g["class_id"] == IGNORE]
        tp, fp, fn = match_frame(preds, gts, iou_thr)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        frames += 1
    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"frames": frames, "frames_held": held, "tp": TP, "fp": FP, "fn": FN,
            "precision": precision, "recall": recall, "f1": f1, "iou_threshold": iou_thr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--images", default=None,
                    help="image folder (gives frame list + sizes); omit for a video and use --img-size")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--img-size", default=None, help="WxH of the frames, e.g. 1344x756")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--classes", default=None, help="comma-separated class ids to score")
    ap.add_argument("--hold-last", action="store_true",
                    help="score every frame; dropped frames reuse the last processed result")
    args = ap.parse_args()
    if args.images is None and args.img_size is None:
        ap.error("give --images or --img-size")
    with open(args.pred) as f:
        pred = json.load(f)
    img_size = tuple(int(v) for v in args.img_size.lower().split("x")) if args.img_size else None
    classes = {int(c) for c in args.classes.split(",")} if args.classes else None
    res = evaluate(pred, Path(args.images) if args.images else None, Path(args.labels),
                   args.iou, args.min_conf, classes, args.hold_last, img_size)
    for k, v in res.items():
        print(f"{k:15s} {v:.4f}" if isinstance(v, float) else f"{k:15s} {v}")


if __name__ == "__main__":
    main()
