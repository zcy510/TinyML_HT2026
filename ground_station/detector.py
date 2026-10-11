"""High-complexity object detector that runs on the ground station.

Wraps an Ultralytics YOLO model (default: YOLOv8s) and returns detections as
plain Python dicts together with timing information, so the server can send
them back to the UAV (Raspberry Pi 5) as JSON.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    cls_id: int
    label: str
    confidence: float
    # Bounding box in pixel coordinates of the image that was sent: x1, y1, x2, y2
    box: List[float]

    def to_dict(self) -> dict:
        return {
            "class_id": self.cls_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "box": [round(v, 1) for v in self.box],
        }


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0          # model forward pass incl. pre/post-processing
    model_name: str = ""
    device: str = ""


def pick_device(requested: Optional[str] = None) -> str:
    """Return 'cuda:0' if a GPU is available (school Windows PC), else 'cpu'."""
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # torch missing or broken -> let ultralytics decide
        pass
    return "cpu"


class YoloDetector:
    """Thread-safe wrapper around an Ultralytics YOLO model."""

    def __init__(
        self,
        weights: str = "yolov8s.pt",
        device: Optional[str] = None,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        half: bool = False,
    ) -> None:
        from ultralytics import YOLO  # imported lazily so tests can run without it

        self.weights = weights
        self.device = pick_device(device)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half and self.device.startswith("cuda")
        self.model = YOLO(weights)
        self.names = self.model.names
        self._lock = threading.Lock()  # YOLO models are not safe for concurrent calls

    @property
    def model_name(self) -> str:
        return self.weights.replace("\\", "/").split("/")[-1]

    def warmup(self, runs: int = 3) -> None:
        """Run a few dummy inferences so the first real request is not slow."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(runs):
            self.detect(dummy)

    def detect(
        self,
        image_bgr: np.ndarray,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> DetectionResult:
        with self._lock:
            t0 = time.perf_counter()
            results = self.model.predict(
                image_bgr,
                imgsz=imgsz or self.imgsz,
                conf=self.conf if conf is None else conf,
                iou=self.iou,
                device=self.device,
                half=self.half,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - t0) * 1000.0

        dets: List[Detection] = []
        r = results[0]
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for box, c, k in zip(xyxy, confs, clss):
                dets.append(Detection(int(k), str(self.names[int(k)]), float(c), box.tolist()))
        return DetectionResult(dets, inference_ms, self.model_name, self.device)
