"""VisDrone2019-VID support: turn the per-sequence annotations into YOLO-format
labels with COCO class ids, so the COCO-trained YOLO models (ground station now,
TinyML edge model later) can be scored with ``tools.eval_detection``.

Dataset layout (download VisDrone2019-VID-val from
https://github.com/VisDrone/VisDrone-Dataset and unzip into data/):

    data/VisDrone2019-VID-val/
        sequences/uav0000086_00000_v/0000001.jpg, 0000002.jpg, ...
        annotations/uav0000086_00000_v.txt

Annotation rows: frame,target_id,left,top,width,height,score,category,truncation,occlusion
(frame is 1-based and matches the image name; score 0 = "ignore this box").

Output: one ``<frame>.txt`` per frame, rows ``class cx cy w h`` (normalised), where
class is a COCO id, or -1 for an *ignore region* (VisDrone "ignored regions" and
"others", score-0 boxes, unmapped classes such as tricycles, and boxes smaller than
--min-size). ``tools.eval_detection`` does not count a prediction that falls inside
an ignore region as a false positive, and never expects one to be found there.

    python -m tools.visdrone convert data/VisDrone2019-VID-val/sequences/uav0000086_00000_v
    python -m tools.visdrone convert <seq_dir> --video data/uav0000086.mp4   # also make an mp4
    python -m tools.visdrone list data/VisDrone2019-VID-val                  # sequences + sizes
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2

VISDRONE_NAMES = {
    0: "ignored-region", 1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others",
}
# VisDrone category -> COCO class id (None = no COCO equivalent -> ignore region).
# "van" has no COCO class; YOLO usually calls vans "car", so they are scored as car.
VISDRONE_TO_COCO: Dict[int, Optional[int]] = {
    1: 0,    # pedestrian -> person
    2: 0,    # people (sitting, riding...) -> person
    3: 1,    # bicycle
    4: 2,    # car
    5: 2,    # van -> car
    6: 7,    # truck
    7: None, 8: None,   # (awning-)tricycle: not in COCO
    9: 5,    # bus
    10: 3,   # motor -> motorcycle
}
# The COCO classes that exist in VisDrone; pass to eval_detection --classes so that
# predictions of other COCO classes (dog, traffic light, ...) are not counted as FP.
EVAL_CLASSES = sorted({c for c in VISDRONE_TO_COCO.values() if c is not None})
IGNORE = -1


def find_annotation(seq_dir: Path) -> Path:
    ann = seq_dir.parent.parent / "annotations" / f"{seq_dir.name}.txt"
    if not ann.exists():
        raise FileNotFoundError(f"no annotation file for {seq_dir.name}: expected {ann}")
    return ann


def parse_annotations(ann: Path) -> Dict[int, List[List[float]]]:
    """frame (1-based) -> list of [left, top, w, h, score, category]."""
    out: Dict[int, List[List[float]]] = defaultdict(list)
    for line in ann.read_text().splitlines():
        parts = line.strip().rstrip(",").split(",")
        if len(parts) < 8:
            continue
        f, _tid, x, y, w, h, score, cat = (float(v) for v in parts[:8])
        out[int(f)].append([x, y, w, h, score, int(cat)])
    return out


def to_yolo_rows(boxes: List[List[float]], img_w: int, img_h: int,
                 min_size: float = 0.0) -> List[str]:
    rows = []
    for x, y, w, h, score, cat in boxes:
        if w <= 0 or h <= 0:
            continue
        cls = VISDRONE_TO_COCO.get(int(cat))
        if cls is None or score == 0 or min(w, h) < min_size:
            cls = IGNORE
        rows.append(f"{cls} {(x + w / 2) / img_w:.6f} {(y + h / 2) / img_h:.6f} "
                    f"{w / img_w:.6f} {h / img_h:.6f}")
    return rows


def convert_sequence(seq_dir: Path, out_dir: Path, min_size: float = 0.0,
                     video: Optional[Path] = None, fps: float = 30.0) -> dict:
    from edge.frame_source import list_images

    frames = list_images(seq_dir)
    if not frames:
        raise FileNotFoundError(f"no images in {seq_dir}")
    first = cv2.imread(str(frames[0]))
    img_h, img_w = first.shape[:2]
    anns = parse_annotations(find_annotation(seq_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_obj = n_ign = 0
    for f in frames:
        rows = to_yolo_rows(anns.get(int(f.stem), []), img_w, img_h, min_size)
        n_ign += sum(r.startswith(f"{IGNORE} ") for r in rows)
        n_obj += sum(not r.startswith(f"{IGNORE} ") for r in rows)
        (out_dir / f"{f.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
    if video:
        video.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img_w, img_h))
        for f in frames:
            img = cv2.imread(str(f))
            vw.write(img if img.shape[:2] == (img_h, img_w) else cv2.resize(img, (img_w, img_h)))
        vw.release()
    return {"sequence": seq_dir.name, "frames": len(frames), "size": f"{img_w}x{img_h}",
            "objects": n_obj, "ignore_regions": n_ign, "labels": str(out_dir),
            "video": str(video) if video else ""}


def main() -> None:
    ap = argparse.ArgumentParser(description="VisDrone2019-VID -> YOLO labels (COCO ids)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="convert one sequence")
    c.add_argument("seq_dir", help=".../sequences/<sequence name>")
    c.add_argument("--out", default=None,
                   help="label folder (default: data/visdrone_labels/<sequence name>)")
    c.add_argument("--min-size", type=float, default=0.0,
                   help="treat objects smaller than this many pixels (shorter side) as ignore "
                        "regions; VisDrone has many tiny objects no 640-px model can see")
    c.add_argument("--video", default=None, help="also write the sequence as an mp4")
    c.add_argument("--fps", type=float, default=30.0, help="frame rate of the mp4")
    ls = sub.add_parser("list", help="list sequences of a VisDrone-VID split")
    ls.add_argument("root", help="e.g. data/VisDrone2019-VID-val")
    args = ap.parse_args()

    if args.cmd == "list":
        from edge.frame_source import list_images

        for seq in sorted(p for p in (Path(args.root) / "sequences").iterdir() if p.is_dir()):
            imgs = list_images(seq)
            img = cv2.imread(str(imgs[0])) if imgs else None
            size = f"{img.shape[1]}x{img.shape[0]}" if img is not None else "?"
            print(f"{seq.name:28s} {len(imgs):5d} frames  {size}")
        return
    seq = Path(args.seq_dir).resolve()
    out = Path(args.out) if args.out else Path("data/visdrone_labels") / seq.name
    res = convert_sequence(seq, out, args.min_size, Path(args.video) if args.video else None,
                           args.fps)
    for k, v in res.items():
        print(f"{k:15s} {v}")
    print(f"evaluate with:  --classes {','.join(map(str, EVAL_CLASSES))}")


if __name__ == "__main__":
    main()
