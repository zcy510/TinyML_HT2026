"""Export YOLOv8 to ONNX with box coordinates normalised to 0..1 (for INT8 quantisation).

Why: the YOLOv8 head has ONE output tensor (1, 84, N) holding both box coordinates
(0..imgsz pixels) and class scores (0..1). Full-integer quantisation gives that tensor
ONE scale; with pixel boxes it is about 320/255 = 1.6, so every score below ~0.8
rounds to 0 and the model "detects nothing". Dividing the boxes by imgsz inside the
graph makes the whole tensor 0..1 (scale ~ 1/255), which keeps the scores. This is the
same trick Ultralytics uses in its own TFLite export.

    python -m tools.export_onnx_norm --weights yolov8n.pt --imgsz 320
    # -> yolov8n_norm_320.onnx; in Edge Impulse choose "YOLOv11 (coordinates scaled 0..1)"

Needs ultralytics + torch + onnx (the ground-station venv has them).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


class NormBoxes(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, imgsz: int) -> None:
        super().__init__()
        self.model = model
        self.imgsz = float(imgsz)

    def forward(self, x):
        y = self.model(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        return torch.cat([y[:, :4] / self.imgsz, y[:, 4:]], dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--check-image", default=None, help="optional image to sanity-check scores")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights).model.fuse().eval().float()
    head = model.model[-1]
    head.export, head.format, head.dynamic = True, "onnx", False   # plain (1, 84, N) output
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = NormBoxes(model, args.imgsz).eval()

    x = torch.zeros(1, 3, args.imgsz, args.imgsz)
    if args.check_image:
        import cv2

        from tools.make_calib_npy import letterbox

        img = cv2.cvtColor(letterbox(cv2.imread(args.check_image), args.imgsz), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        y = wrapped(x)
    print(f"output shape {tuple(y.shape)}; box range {y[0, :4].min():.3f}..{y[0, :4].max():.3f}; "
          f"max class score {y[0, 4:].max():.3f}")
    if y[0, :4].max() > 1.5:
        raise SystemExit("boxes are not normalised - unexpected model output")

    out = args.out or f"{Path(args.weights).stem}_norm_{args.imgsz}.onnx"
    kw = dict(opset_version=args.opset, input_names=["images"], output_names=["output0"])
    try:
        torch.onnx.export(wrapped, torch.zeros(1, 3, args.imgsz, args.imgsz), out, dynamo=False, **kw)
    except TypeError:                                   # older torch: no dynamo argument
        torch.onnx.export(wrapped, torch.zeros(1, 3, args.imgsz, args.imgsz), out, **kw)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
