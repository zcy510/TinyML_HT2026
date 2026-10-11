"""Lightweight on-board detector (edge mode): an INT8-quantised YOLOv8n LiteRT/TFLite
model running on the Raspberry Pi 5 CPU.

Same interface and result type as :class:`edge.remote_client.RemoteDetector`, so
the pipeline, logging, real-time replay and evaluation work unchanged and the
policy engine can switch between the two per frame. Classes are COCO, like the
ground-station model, so the accuracy of both modes is directly comparable.

Runtime (any one of): ai-edge-litert (pip install ai-edge-litert),
tflite-runtime, or full tensorflow.

    det = LocalDetector("models/yolov8n_int8_320.tflite")
    res = det.detect(frame_bgr, "0000001")      # -> RemoteResult, network fields = 0
"""
from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from edge.remote_client import RemoteResult

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def _load_interpreter(path: str, num_threads: int):
    errors = []
    for mod, attr in (("ai_edge_litert.interpreter", "Interpreter"),
                      ("tflite_runtime.interpreter", "Interpreter"),
                      ("tensorflow.lite", "Interpreter")):
        try:
            module = __import__(mod, fromlist=[attr])
            return getattr(module, attr)(model_path=path, num_threads=num_threads)
        except ImportError as e:
            errors.append(f"{mod}: {e}")
    raise ImportError("no LiteRT/TFLite runtime found; on the Pi run: pip install ai-edge-litert\n"
                      + "\n".join(errors))


def letterbox(img: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize keeping aspect ratio and pad to size x size (like Ultralytics)."""
    h, w = img.shape[:2]
    gain = min(size / h, size / w)
    nw, nh = int(round(w * gain)), int(round(h * gain))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, gain, (left, top)


class LocalDetector:
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45,
                 num_threads: Optional[int] = None, max_det: int = 300) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model not found: {model_path}")
        self.model_name = os.path.basename(model_path)
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.interp = _load_interpreter(model_path, num_threads or os.cpu_count() or 4)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        shape = list(self.inp["shape"])                       # NHWC or NCHW
        self.nchw = shape[1] == 3
        self.imgsz = int(shape[2] if self.nchw else shape[1])
        self.in_scale, self.in_zero = self.inp.get("quantization", (0.0, 0))
        self.out_scale, self.out_zero = self.out.get("quantization", (0.0, 0))
        self.detect(np.zeros((self.imgsz, self.imgsz, 3), np.uint8))   # warm-up

    def health(self) -> dict:      # same shape as RemoteDetector.health()
        return {"model": self.model_name, "device": f"local cpu, {self.inp['dtype'].__name__} input"}

    def ping(self, n: int = 3) -> Optional[float]:   # no link involved
        return None

    # --------------------------------------------------------------- inference
    def _preprocess(self, frame_bgr: np.ndarray):
        img, gain, pad = letterbox(frame_bgr, self.imgsz)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.nchw:
            x = x.transpose(2, 0, 1)
        x = x[None]
        dtype = self.inp["dtype"]
        if dtype in (np.int8, np.uint8):                      # quantised input
            x = np.round(x / self.in_scale + self.in_zero)
            info = np.iinfo(dtype)
            x = np.clip(x, info.min, info.max).astype(dtype)
        else:
            x = x.astype(dtype)
        return x, gain, pad

    def _postprocess(self, y: np.ndarray, gain: float, pad: Tuple[int, int],
                     orig: Tuple[int, int]) -> List[dict]:
        if self.out["dtype"] in (np.int8, np.uint8):
            y = (y.astype(np.float32) - self.out_zero) * self.out_scale
        y = y[0]
        ch = 4 + len(COCO_NAMES)                              # 84 = box + class scores
        if y.shape[0] == ch and y.shape[1] != ch:             # (84, N) -> (N, 84)
            y = y.T
        elif y.shape[1] != ch and y.shape[0] < y.shape[1]:    # other class count
            y = y.T
        boxes, scores = y[:, :4], y[:, 4:]
        if boxes.max() <= 2.0:                                # normalised xywh -> pixels
            boxes = boxes * self.imgsz
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
        if not len(conf):
            return []
        xywh = boxes.copy()
        xywh[:, 0] -= xywh[:, 2] / 2                          # centre -> top-left
        xywh[:, 1] -= xywh[:, 3] / 2
        # class-aware NMS: shift each class far apart so boxes of different classes never overlap
        shifted = xywh.copy()
        shifted[:, :2] += cls[:, None] * 4096.0
        idx = cv2.dnn.NMSBoxes(shifted.tolist(), conf.tolist(), self.conf, self.iou)
        idx = np.array(idx).reshape(-1)[: self.max_det]
        w0, h0 = orig
        dets = []
        for i in idx:
            x, y_, w, h = xywh[i]
            x1 = np.clip((x - pad[0]) / gain, 0, w0)
            y1 = np.clip((y_ - pad[1]) / gain, 0, h0)
            x2 = np.clip((x + w - pad[0]) / gain, 0, w0)
            y2 = np.clip((y_ + h - pad[1]) / gain, 0, h0)
            c = int(cls[i])
            dets.append({"class_id": c, "label": COCO_NAMES[c] if c < len(COCO_NAMES) else str(c),
                         "confidence": round(float(conf[i]), 4),
                         "box": [round(float(v), 1) for v in (x1, y1, x2, y2)]})
        return dets

    def detect(self, frame_bgr: np.ndarray, frame_id: str = "", max_width: Optional[int] = None,
               jpeg_quality: int = 0, conf: Optional[float] = None) -> RemoteResult:
        """max_width / jpeg_quality are accepted for interface compatibility and ignored:
        the model input size (e.g. 320) sets the resolution."""
        t0 = time.perf_counter()
        x, gain, pad = self._preprocess(frame_bgr)
        t1 = time.perf_counter()
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])
        t2 = time.perf_counter()
        old = self.conf
        if conf is not None:
            self.conf = conf
        try:
            dets = self._postprocess(y, gain, pad, (frame_bgr.shape[1], frame_bgr.shape[0]))
        finally:
            self.conf = old
        t3 = time.perf_counter()
        return RemoteResult(
            ok=True, frame_id=frame_id, detections=dets, e2e_ms=(t3 - t0) * 1000.0,
            encode_ms=(t1 - t0) * 1000.0,              # here: letterbox + quantise input
            inference_ms=(t2 - t1) * 1000.0, server_ms=(t3 - t1) * 1000.0,  # invoke + NMS
            sent_size=(self.imgsz, self.imgsz), scale=gain,
        )
