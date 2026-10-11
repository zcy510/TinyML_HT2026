"""Unit / integration tests for the remote detection pipeline.

These use a fake detector, so they run fast and need no model weights or GPU:
    python -m pytest tests -q
Set RUN_YOLO=1 to additionally run a real YOLOv8s smoke test.
"""
import os
import socket
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from edge.remote_client import RemoteDetector
from ground_station.detector import Detection, DetectionResult
from ground_station.server import create_app
from tools.eval_detection import iou, match_frame


class FakeDetector:
    model_name = "fake.pt"
    device = "cpu"

    def __init__(self):
        self.last_shape = None

    def detect(self, img, conf=None, imgsz=None):
        self.last_shape = img.shape
        h, w = img.shape[:2]
        # one box covering the central quarter of the image
        return DetectionResult([Detection(0, "person", 0.9, [w / 4, h / 4, 3 * w / 4, 3 * h / 4])],
                               5.0, self.model_name, self.device)


def jpeg(w=640, h=480):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return img, cv2.imencode(".jpg", img)[1].tobytes()


@pytest.fixture
def app_client(tmp_path):
    det = FakeDetector()
    app = create_app(det, str(tmp_path / "log.csv"))
    return TestClient(app), det, tmp_path / "log.csv"


def test_health_and_ping(app_client):
    c, _, _ = app_client
    assert c.get("/health").json()["model"] == "fake.pt"
    assert "t" in c.get("/ping").json()


def test_detect_raw_body(app_client):
    c, det, log = app_client
    _, data = jpeg()
    r = c.post("/detect?frame_id=f1", content=data, headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200
    body = r.json()
    assert body["image_size"] == [640, 480]
    assert body["detections"][0]["label"] == "person"
    assert body["timing"]["inference_ms"] == 5.0
    assert det.last_shape == (480, 640, 3)
    assert log.read_text().count("\n") == 2  # header + 1 row


def test_detect_multipart(app_client):
    c, _, _ = app_client
    _, data = jpeg(320, 240)
    r = c.post("/detect", files={"file": ("x.jpg", data, "image/jpeg")})
    assert r.status_code == 200 and r.json()["image_size"] == [320, 240]


def test_bad_image_and_empty_body(app_client):
    c, _, _ = app_client
    assert c.post("/detect", content=b"not an image",
                  headers={"Content-Type": "image/jpeg"}).status_code == 400
    assert c.post("/detect", content=b"", headers={"Content-Type": "image/jpeg"}).status_code == 400


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def live_server(tmp_path):
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(FakeDetector(), None),
                                           host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    th.join(timeout=5)


def test_client_end_to_end_with_downscaling(live_server):
    client = RemoteDetector(live_server)
    assert client.ping() is not None
    img, _ = jpeg(640, 480)
    res = client.detect(img, "f1", max_width=320, jpeg_quality=80)
    assert res.ok and res.sent_size == (320, 240) and res.scale == 0.5
    # box was computed on 320x240 and must be mapped back to 640x480
    assert res.detections[0]["box"] == [160.0, 120.0, 480.0, 360.0]
    assert res.e2e_ms >= res.server_ms >= 0 and res.upload_bytes > 0


def test_client_handles_unreachable_server():
    client = RemoteDetector(f"http://127.0.0.1:{_free_port()}", timeout_s=0.5)
    assert client.ping(n=1) is None
    img, _ = jpeg(64, 48)
    res = client.detect(img, "x")
    assert not res.ok and res.error


def test_iou_and_matching():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    preds = [{"class_id": 0, "confidence": 0.9, "box": [0, 0, 10, 10]},
             {"class_id": 1, "confidence": 0.8, "box": [0, 0, 10, 10]}]
    gts = [{"class_id": 0, "box": [1, 1, 10, 10]}, {"class_id": 2, "box": [50, 50, 60, 60]}]
    assert match_frame(preds, gts, 0.5) == (1, 1, 1)


@pytest.mark.skipif(os.environ.get("RUN_YOLO") != "1", reason="set RUN_YOLO=1 for the real-model test")
def test_real_yolo_smoke():
    from ultralytics.utils import ASSETS

    from ground_station.detector import YoloDetector

    det = YoloDetector("yolov8s.pt")
    img = cv2.imread(str(ASSETS / "bus.jpg"))
    res = det.detect(img)
    labels = {d.label for d in res.detections}
    assert "bus" in labels and "person" in labels
