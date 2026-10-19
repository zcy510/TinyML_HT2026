"""MOT17 / MOT16 / MOT20 support: pedestrian video (ground level, large objects) as an
alternative to VisDrone. Converts MOTChallenge ground truth into the YOLO-format labels
with COCO class ids that tools.eval_detection reads.

Layout (unzip MOT17.zip into data/):

    data/MOT17/train/MOT17-09-FRCNN/img1/000001.jpg, 000002.jpg, ...
                                    gt/gt.txt
                                    seqinfo.ini      (frameRate, imWidth, imHeight)

MOT17 ships every sequence three times (-DPM, -FRCNN, -SDP: only the public detections
differ); frames and ground truth are identical, so use any one, e.g. -FRCNN.

gt.txt rows: frame, id, left, top, width, height, considered (0/1), class, visibility
Classes: 1 pedestrian, 2 person on vehicle, 3 car, 4 bicycle, 5 motorbike,
6 non-motorized vehicle, 7 static person, 8 distractor, 9-11 occluders, 12 reflection.

Like the official MOTChallenge evaluation, only *considered* pedestrians (class 1, flag 1)
are objects to find. Persons on vehicles, static persons, distractors and reflections are
ignore regions (a person detection there is neither a hit nor a false positive), and so
are pedestrians below --min-visibility / --min-size. Vehicles and occluders are not
annotated consistently, so they are ignore regions too: score with --classes 0 (person).

    python -m tools.mot list data/MOT17/train                 # sequences: frames, fps, size, persons/frame
    python -m tools.mot convert data/MOT17/train/MOT17-09-FRCNN/img1
    python -m tools.mot convert <img1> --min-visibility 0.25 --min-size 24
"""
from __future__ import annotations

import argparse
import configparser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PERSON = 0
IGNORE = -1
EVAL_CLASSES = [PERSON]
PEDESTRIAN = 1
IGNORE_PERSON_CLASSES = {2, 7, 8, 12}          # person on vehicle, static, distractor, reflection


def seq_dir_of(img_dir: Path) -> Path:
    """Accept .../MOT17-09-FRCNN/img1 or .../MOT17-09-FRCNN."""
    return img_dir.parent if img_dir.name == "img1" else img_dir


def find_gt(img_dir: Path) -> Path:
    gt = seq_dir_of(img_dir) / "gt" / "gt.txt"
    if not gt.exists():
        raise FileNotFoundError(f"no ground truth at {gt} (test sequences have none: use train/)")
    return gt


def read_seqinfo(img_dir: Path) -> Dict[str, str]:
    ini = seq_dir_of(img_dir) / "seqinfo.ini"
    if not ini.exists():
        return {}
    cp = configparser.ConfigParser()
    cp.read(ini)
    return dict(cp["Sequence"]) if "Sequence" in cp else {}


def seq_fps(img_dir: Path) -> Optional[float]:
    try:
        return float(read_seqinfo(img_dir).get("framerate", ""))
    except ValueError:
        return None


def parse_gt(gt: Path) -> Dict[int, List[Tuple[float, float, float, float, int, int, float]]]:
    """frame -> [(left, top, w, h, considered, class, visibility)]"""
    out: Dict[int, list] = defaultdict(list)
    for line in gt.read_text().splitlines():
        p = line.strip().split(",")
        if len(p) < 7:
            continue
        frame = int(float(p[0]))
        x, y, w, h = (float(v) for v in p[2:6])
        considered = int(float(p[6]))
        cls = int(float(p[7])) if len(p) > 7 else PEDESTRIAN      # MOT15-style rows: no class
        vis = float(p[8]) if len(p) > 8 else 1.0
        out[frame].append((x, y, w, h, considered, cls, vis))
    return out


def to_yolo_rows(boxes, img_w: int, img_h: int, min_visibility: float = 0.0,
                 min_size: float = 0.0) -> List[str]:
    rows = []
    for x, y, w, h, considered, cls, vis in boxes:
        # clip to the image (MOT boxes may extend beyond it)
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(img_w), x + w), min(float(img_h), y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        if cls == PEDESTRIAN and considered:
            label = PERSON if (vis >= min_visibility and min(x2 - x1, y2 - y1) >= min_size) else IGNORE
        else:
            label = IGNORE            # other persons, vehicles, occluders, unconsidered boxes
        rows.append(f"{label} {(x1 + x2) / 2 / img_w:.6f} {(y1 + y2) / 2 / img_h:.6f} "
                    f"{(x2 - x1) / img_w:.6f} {(y2 - y1) / img_h:.6f}")
    return rows


def image_size(img_dir: Path, first_image: Path) -> Tuple[int, int]:
    info = read_seqinfo(img_dir)
    try:
        return int(info["imwidth"]), int(info["imheight"])
    except (KeyError, ValueError):
        import cv2

        img = cv2.imread(str(first_image))
        return img.shape[1], img.shape[0]


def convert_sequence(img_dir: Path, out_dir: Path, min_visibility: float = 0.0,
                     min_size: float = 0.0) -> dict:
    from edge.frame_source import list_images

    img_dir = img_dir if img_dir.name == "img1" or not (img_dir / "img1").is_dir() else img_dir / "img1"
    frames = list_images(img_dir)
    if not frames:
        raise FileNotFoundError(f"no images in {img_dir}")
    w, h = image_size(img_dir, frames[0])
    gt = parse_gt(find_gt(img_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_obj = n_ign = 0
    heights = []
    for f in frames:
        rows = to_yolo_rows(gt.get(int(f.stem), []), w, h, min_visibility, min_size)
        for r in rows:
            if r.startswith(f"{IGNORE} "):
                n_ign += 1
            else:
                n_obj += 1
                heights.append(float(r.split()[4]) * h)
        (out_dir / f"{f.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
    heights.sort()
    return {"sequence": seq_dir_of(img_dir).name, "frames": len(frames), "size": f"{w}x{h}",
            "fps": seq_fps(img_dir) or "?", "persons": n_obj, "ignore_regions": n_ign,
            "persons_per_frame": round(n_obj / len(frames), 1),
            "median_person_height_px": round(heights[len(heights) // 2]) if heights else 0,
            "labels": str(out_dir)}


def list_sequences(root: Path) -> List[dict]:
    """One row per unique sequence (MOT17's -DPM/-FRCNN/-SDP copies collapsed)."""
    rows, seen = [], set()
    for seq in sorted(p for p in root.iterdir() if (p / "img1").is_dir()):
        base = seq.name.rsplit("-", 1)[0] if seq.name.count("-") >= 2 else seq.name
        if base in seen:
            continue
        seen.add(base)
        info = read_seqinfo(seq / "img1")
        gt = seq / "gt" / "gt.txt"
        ppf, med_h = "-", "-"
        if gt.exists():
            boxes = parse_gt(gt)
            ped = [b for v in boxes.values() for b in v if b[5] == PEDESTRIAN and b[4]]
            n = int(info.get("seqlength", len(boxes) or 1))
            ppf = f"{len(ped) / n:.1f}"
            hs = sorted(b[3] for b in ped)
            med_h = f"{hs[len(hs) // 2]:.0f}" if hs else "-"
        rows.append({"dir": seq.name, "frames": info.get("seqlength", "?"),
                     "fps": info.get("framerate", "?"),
                     "size": f"{info.get('imwidth', '?')}x{info.get('imheight', '?')}",
                     "persons/frame": ppf, "median_h_px": med_h})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="MOTChallenge ground truth -> YOLO labels (COCO person)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="convert one sequence")
    c.add_argument("img_dir", help=".../MOT17-09-FRCNN/img1 (or the sequence folder)")
    c.add_argument("--out", default=None, help="default: data/mot_labels/<sequence>")
    c.add_argument("--min-visibility", type=float, default=0.0,
                   help="pedestrians less visible than this (0-1) become ignore regions")
    c.add_argument("--min-size", type=float, default=0.0,
                   help="pedestrians smaller than this (px, shorter side) become ignore regions")
    ls = sub.add_parser("list", help="list the sequences of a split")
    ls.add_argument("root", help="e.g. data/MOT17/train")
    args = ap.parse_args()

    if args.cmd == "list":
        rows = list_sequences(Path(args.root))
        if not rows:
            raise SystemExit(f"no sequences (folders with img1/) in {args.root}")
        keys = list(rows[0])
        print("  ".join(f"{k:>14s}" if k != "dir" else f"{k:22s}" for k in keys))
        for r in rows:
            print("  ".join(f"{str(r[k]):>14s}" if k != "dir" else f"{r[k]:22s}" for k in keys))
        print("(MOT17: -DPM/-FRCNN/-SDP are the same video; one shown per sequence)")
        return
    img_dir = Path(args.img_dir)
    out = Path(args.out) if args.out else Path("data/mot_labels") / seq_dir_of(img_dir).name
    res = convert_sequence(img_dir, out, args.min_visibility, args.min_size)
    for k, v in res.items():
        print(f"{k:25s} {v}")
    print("evaluate with:  --classes 0")


if __name__ == "__main__":
    main()
