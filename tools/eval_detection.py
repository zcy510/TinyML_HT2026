"""Detection-performance evaluation (precision / recall / F1 at an IoU threshold).

Compares the detections saved by `edge.run_remote --save-json` (or later by the
edge pipeline) against YOLO-format ground-truth labels
(`labels/<image>.txt`, rows: class cx cy w h, normalised).

python -m tools.eval_detection --pred logs/remote_dets.json \
    --images datasets/coco128/images/train2017 --labels datasets/coco128/labels/train2017
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2


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


def match_frame(preds: List[dict], gts: List[dict], iou_thr: float):
    """Greedy matching by confidence, class-aware. Returns (tp, fp, fn)."""
    used = set()
    tp = 0
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
    return tp, len(preds) - tp, len(gts) - tp


def evaluate(pred: Dict[str, List[dict]], images: Path, labels: Path,
             iou_thr: float = 0.5, min_conf: float = 0.0) -> dict:
    TP = FP = FN = 0
    for img_path in sorted(images.iterdir()):
        if img_path.stem not in pred:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_yolo_labels(labels / f"{img_path.stem}.txt", w, h)
        preds = [d for d in pred[img_path.stem] if d["confidence"] >= min_conf]
        tp, fp, fn = match_frame(preds, gts, iou_thr)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": TP, "fp": FP, "fn": FN, "precision": precision, "recall": recall,
            "f1": f1, "iou_threshold": iou_thr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-conf", type=float, default=0.0)
    args = ap.parse_args()
    with open(args.pred) as f:
        pred = json.load(f)
    res = evaluate(pred, Path(args.images), Path(args.labels), args.iou, args.min_conf)
    for k, v in res.items():
        print(f"{k:15s} {v:.4f}" if isinstance(v, float) else f"{k:15s} {v}")


if __name__ == "__main__":
    main()
