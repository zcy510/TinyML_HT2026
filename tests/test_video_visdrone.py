"""Tests for real-time video replay (frame dropping) and VisDrone-VID support."""
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from edge.frame_source import iter_frames
from tools.eval_detection import evaluate, match_frame
from tools.visdrone import EVAL_CLASSES, convert_sequence


def make_sequence(root: Path, n: int = 40, w: int = 64, h: int = 48, name: str = "uav_test_v"):
    """Fake VisDrone-VID layout: <root>/sequences/<name>/0000001.jpg ... + annotations/<name>.txt"""
    seq = root / "sequences" / name
    seq.mkdir(parents=True)
    for i in range(1, n + 1):
        img = np.full((h, w, 3), i * 5 % 255, dtype=np.uint8)
        cv2.imwrite(str(seq / f"{i:07d}.jpg"), img)
    (root / "annotations").mkdir()
    lines = []
    for i in range(1, n + 1):
        lines += [f"{i},1,10,10,20,20,1,4,0,0",     # car
                  f"{i},2,40,5,10,30,1,1,0,0",      # pedestrian
                  f"{i},3,0,30,10,10,1,7,0,0",      # tricycle -> ignore
                  f"{i},4,50,40,4,4,0,0,0,0"]       # ignored region
    (root / "annotations" / f"{name}.txt").write_text("\n".join(lines) + "\n")
    return seq


def test_offline_folder_yields_every_frame(tmp_path):
    seq = make_sequence(tmp_path, n=10)
    frames = list(iter_frames(str(seq)))
    assert [f.frame_id for f in frames] == [f"{i:07d}" for i in range(1, 11)]
    assert all(f.dropped == 0 for f in frames)


def test_realtime_folder_drops_frames_when_pipeline_is_slow(tmp_path):
    seq = make_sequence(tmp_path, n=40)
    got = []
    for fr in iter_frames(str(seq), realtime=True, fps=100):   # 10 ms per frame
        got.append(fr)
        time.sleep(0.045)                                      # "processing" takes ~45 ms
    idx = [f.index for f in got]
    assert idx == sorted(set(idx)) and idx[0] == 0
    assert len(got) < 20                                       # ~40 / 4.5 frames kept
    assert sum(f.dropped for f in got) + len(got) <= 40
    assert np.mean([f.dropped for f in got[1:]]) >= 2.5        # ~3.5 dropped per frame
    for a, b in zip(got, got[1:]):
        assert b.index - a.index - 1 == b.dropped


def test_realtime_waits_when_pipeline_is_fast(tmp_path):
    seq = make_sequence(tmp_path, n=6)
    t0 = time.perf_counter()
    frames = list(iter_frames(str(seq), realtime=True, fps=50))   # 20 ms per frame
    assert [f.index for f in frames] == list(range(6))           # nothing dropped
    assert time.perf_counter() - t0 >= 5 * 0.02 * 0.9            # but paced at camera speed


def test_video_offline_and_realtime(tmp_path):
    seq = make_sequence(tmp_path, n=30)
    video = tmp_path / "seq.mp4"
    convert_sequence(seq, tmp_path / "labels", video=video, fps=100)
    if not video.exists() or not cv2.VideoCapture(str(video)).isOpened():
        pytest.skip("no mp4 writer/reader in this OpenCV build")
    off = list(iter_frames(str(video)))
    assert len(off) == 30 and off[0].frame_id == "0000001"   # same names as VisDrone frames
    rt = []
    for fr in iter_frames(str(video), realtime=True):        # native 100 fps from the file
        rt.append(fr)
        time.sleep(0.035)
    assert len(rt) < 15 and sum(f.dropped for f in rt) > 10


def test_visdrone_conversion_maps_classes_and_ignores(tmp_path):
    seq = make_sequence(tmp_path, n=3)
    res = convert_sequence(seq, tmp_path / "labels")
    assert res["frames"] == 3 and res["objects"] == 6 and res["ignore_regions"] == 6
    rows = (tmp_path / "labels" / "0000001.txt").read_text().split("\n")
    classes = [int(r.split()[0]) for r in rows if r]
    assert classes == [2, 0, -1, -1]                  # car, person, tricycle, ignored region
    cx, cy, w, h = map(float, rows[0].split()[1:])
    assert (cx, cy, w, h) == pytest.approx((20 / 64, 20 / 48, 20 / 64, 20 / 48))
    assert EVAL_CLASSES == [0, 1, 2, 3, 5, 7]


def test_ignore_region_prediction_is_not_false_positive():
    gts = [{"class_id": 2, "box": [0, 0, 10, 10]}, {"class_id": -1, "box": [50, 50, 70, 70]}]
    preds = [{"class_id": 2, "confidence": 0.9, "box": [0, 0, 10, 10]},
             {"class_id": 0, "confidence": 0.8, "box": [55, 55, 65, 65]},   # in ignore region
             {"class_id": 0, "confidence": 0.7, "box": [20, 20, 30, 30]}]   # real FP
    assert match_frame(preds, gts, 0.5) == (1, 1, 0)


def test_eval_hold_last_scores_dropped_frames(tmp_path):
    seq = make_sequence(tmp_path, n=4)
    convert_sequence(seq, tmp_path / "labels")
    car = {"class_id": 2, "label": "car", "confidence": 0.9, "box": [10, 10, 30, 30]}
    dog = {"class_id": 16, "label": "dog", "confidence": 0.9, "box": [0, 0, 5, 5]}
    pred = {"0000001": [car, dog], "0000003": []}          # frames 2 and 4 were dropped
    kw = dict(classes=set(EVAL_CLASSES))
    only = evaluate(pred, seq, tmp_path / "labels", **kw)
    assert (only["frames"], only["tp"], only["fp"], only["fn"]) == (2, 1, 0, 3)  # dog not FP
    held = evaluate(pred, seq, tmp_path / "labels", hold_last=True, **kw)
    assert (held["frames"], held["frames_held"], held["tp"], held["fn"]) == (4, 2, 2, 6)
    by_size = evaluate(pred, None, tmp_path / "labels", img_size=(64, 48), **kw)
    assert by_size["tp"] == only["tp"] and by_size["fn"] == only["fn"]
