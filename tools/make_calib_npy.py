"""Build a representative-data .npy file for INT8 quantisation (e.g. Edge Impulse BYOM).

Images are letterboxed to the model input size exactly like the pipeline does
(edge.local_detector.letterbox), converted to RGB and scaled to 0..1.

    python -m tools.make_calib_npy ../datasets/coco128/images/train2017 --imgsz 320 --n 100
    python -m tools.make_calib_npy data/VisDrone2019-VID-val/sequences/uav0000117_02622_v --n 100

Default layout is NCHW (N, 3, H, W) = the input of an ONNX model exported by Ultralytics;
use --nhwc for a TensorFlow/LiteRT model. Use frames that are NOT in your evaluation
sequence, so calibration does not leak into the test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    gain = min(size / h, size / w)
    nw, nh = int(round(w * gain)), int(round(h * gain))
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="+", help="one or more image folders")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--n", type=int, default=100, help="number of images (spread evenly)")
    ap.add_argument("--nhwc", action="store_true", help="(N,H,W,3) instead of (N,3,H,W)")
    ap.add_argument("--out", default="calib_{imgsz}.npy")
    args = ap.parse_args()

    files = sorted(f for d in args.folder for f in Path(d).iterdir() if f.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit("no images found")
    pick = [files[int(i)] for i in np.linspace(0, len(files) - 1, min(args.n, len(files)))]
    batch = []
    for f in pick:
        img = cv2.imread(str(f))
        if img is None:
            continue
        x = cv2.cvtColor(letterbox(img, args.imgsz), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch.append(x if args.nhwc else x.transpose(2, 0, 1))
    arr = np.stack(batch)
    out = args.out.format(imgsz=args.imgsz)
    np.save(out, arr)
    print(f"saved {out}: shape {arr.shape}, dtype {arr.dtype}, range {arr.min():.2f}..{arr.max():.2f}, "
          f"{arr.nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
