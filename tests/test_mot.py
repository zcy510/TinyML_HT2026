"""Tests for MOTChallenge (MOT17) support: label conversion, frame rate, evaluation."""
from pathlib import Path

import cv2
import numpy as np

from edge.frame_source import folder_fps, iter_frames
from tools.eval_detection import evaluate
from tools.mot import EVAL_CLASSES, convert_sequence, list_sequences


def make_mot_seq(root: Path, name: str = "MOT17-09-FRCNN", n: int = 4, w: int = 192, h: int = 108,
                 fps: int = 25) -> Path:
    seq = root / name
    (seq / "img1").mkdir(parents=True)
    (seq / "gt").mkdir()
    for i in range(1, n + 1):
        cv2.imwrite(str(seq / "img1" / f"{i:06d}.jpg"), np.full((h, w, 3), 100, np.uint8))
    (seq / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate={fps}\nseqLength={n}\n"
        f"imWidth={w}\nimHeight={h}\nimExt=.jpg\n")
    rows = []
    for f in range(1, n + 1):
        rows += [f"{f},1,10,10,20,60,1,1,0.9",        # pedestrian, considered      -> person
                 f"{f},2,60,10,20,60,1,1,0.1",        # pedestrian, 10 % visible    -> person / ignore
                 f"{f},3,100,10,20,60,0,1,1.0",       # pedestrian, not considered  -> ignore
                 f"{f},4,130,10,20,60,1,7,1.0",       # static person               -> ignore
                 f"{f},5,150,50,40,30,1,3,1.0",       # car                         -> ignore
                 f"{f},6,180,80,40,60,1,1,1.0"]       # pedestrian leaving the image -> clipped
    (seq / "gt" / "gt.txt").write_text("\n".join(rows) + "\n")
    return seq


def test_convert_maps_pedestrians_and_ignores(tmp_path):
    seq = make_mot_seq(tmp_path)
    res = convert_sequence(seq / "img1", tmp_path / "labels")
    assert res["frames"] == 4 and res["size"] == "192x108" and res["fps"] == 25.0
    assert res["persons"] == 3 * 4 and res["ignore_regions"] == 3 * 4
    rows = (tmp_path / "labels" / "000001.txt").read_text().split("\n")
    assert [int(r.split()[0]) for r in rows if r] == [0, 0, -1, -1, -1, 0]
    cx, cy, bw, bh = map(float, rows[5].split()[1:])               # clipped at the image border
    assert (cx + bw / 2) <= 1.0 + 1e-6 and (cy + bh / 2) <= 1.0 + 1e-6
    # the 10 %-visible pedestrian becomes an ignore region with --min-visibility
    res = convert_sequence(seq, tmp_path / "labels_vis", min_visibility=0.25)   # sequence folder works too
    assert res["persons"] == 2 * 4
    assert (tmp_path / "labels_vis" / "000001.txt").read_text().split("\n")[1].startswith("-1 ")


def test_seqinfo_frame_rate_and_frame_ids(tmp_path):
    seq = make_mot_seq(tmp_path, fps=25)
    assert folder_fps(seq / "img1") == 25.0
    assert folder_fps(tmp_path) == 30.0                              # no seqinfo -> default
    assert [f.frame_id for f in iter_frames(str(seq / "img1"))] == ["000001", "000002", "000003", "000004"]


def test_list_collapses_mot17_detector_copies(tmp_path):
    for det in ("DPM", "FRCNN", "SDP"):
        make_mot_seq(tmp_path, f"MOT17-09-{det}")
    make_mot_seq(tmp_path, "MOT17-02-FRCNN", fps=30)
    rows = list_sequences(tmp_path)
    assert [r["dir"] for r in rows] == ["MOT17-02-FRCNN", "MOT17-09-DPM"]
    assert rows[1]["persons/frame"] == "3.0" and rows[1]["fps"] == "25"


def test_eval_on_mot_labels(tmp_path):
    seq = make_mot_seq(tmp_path, n=2)
    convert_sequence(seq / "img1", tmp_path / "labels")
    person = {"class_id": 0, "label": "person", "confidence": 0.9, "box": [10, 10, 30, 70]}
    static = {"class_id": 0, "label": "person", "confidence": 0.8, "box": [130, 10, 150, 70]}
    car = {"class_id": 2, "label": "car", "confidence": 0.9, "box": [150, 50, 190, 80]}
    pred = {"000001": [person, static, car], "000002": [person]}
    r = evaluate(pred, seq / "img1", tmp_path / "labels", classes=set(EVAL_CLASSES))
    # static person = ignore region (no FP), car filtered by class; 3 pedestrians per frame
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 4)
