"""Static INT8 post-training quantisation of the YOLOv8 ONNX model with ONNX Runtime.

    python -m tools.quantize_onnx --model yolov8n_norm_320.onnx --calib calib_320.npy
    # -> models/yolov8n_int8_320.onnx   (run on the Pi with: pip install onnxruntime)

* Weights: INT8, per output channel. Activations: UINT8, per tensor, ranges calibrated on
  the representative frames in calib_320.npy (tools.make_calib_npy).
* QDQ format: the graph keeps float input/output, every Conv runs as an INT8 kernel.
* The detection head's *decode* part (DFL softmax, anchor/stride arithmetic, final
  concat of boxes + class scores) stays in float. It is a few cheap element-wise ops,
  and quantising it is exactly what broke the Edge Impulse INT8 model (boxes and
  scores forced into one INT8 scale). All convolutions, including the head's cv2/cv3
  branches, are quantised.
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process


class NpyReader(CalibrationDataReader):
    def __init__(self, path: str, input_name: str, limit: int = 0) -> None:
        data = np.load(path).astype(np.float32)
        if data.ndim == 4 and data.shape[-1] == 3:          # NHWC -> NCHW
            data = data.transpose(0, 3, 1, 2)
        self.data = data[:limit] if limit else data
        self.name, self.i = input_name, 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        self.i += 1
        return {self.name: self.data[self.i - 1: self.i]}


def decode_nodes(model: onnx.ModelProto) -> list:
    """Nodes of the Detect head (last module) outside its conv branches cv2/cv3."""
    head = max((n.name.split("/")[-2] for n in model.graph.node if "/model." in n.name
                and n.name.split("/")[-2].startswith("model.")),
               key=lambda s: int(s.split(".")[1]))
    return [n.name for n in model.graph.node
            if f"/{head}/" in n.name and "/cv2" not in n.name and "/cv3" not in n.name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n_norm_320.onnx")
    ap.add_argument("--calib", default="calib_320.npy")
    ap.add_argument("--out", default="models/yolov8n_int8_320.onnx")
    ap.add_argument("--method", choices=["minmax", "percentile", "entropy"], default="minmax")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N calibration frames")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        src = args.model
        opset = max(o.version for o in onnx.load(src).opset_import if o.domain in ("", "ai.onnx"))
        if opset < 13:                       # per-channel DequantizeLinear needs opset >= 13
            from onnx import version_converter

            src = os.path.join(tmp, "opset13.onnx")
            onnx.save(version_converter.convert_version(onnx.load(args.model), 13), src)
            print(f"upgraded opset {opset} -> 13")
        prep = os.path.join(tmp, "prep.onnx")
        quant_pre_process(src, prep, skip_symbolic_shape=False)
        m = onnx.load(prep)
        exclude = decode_nodes(m)
        print(f"{len(m.graph.node)} nodes; keeping {len(exclude)} head-decode nodes in float")
        reader = NpyReader(args.calib, m.graph.input[0].name, args.limit)
        print(f"calibrating on {len(reader.data)} frames ({args.method}) ...")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        quantize_static(
            prep, args.out, reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method={"minmax": CalibrationMethod.MinMax,
                              "percentile": CalibrationMethod.Percentile,
                              "entropy": CalibrationMethod.Entropy}[args.method],
            nodes_to_exclude=exclude,
        )
    print(f"saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, "
          f"float model {os.path.getsize(args.model) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
