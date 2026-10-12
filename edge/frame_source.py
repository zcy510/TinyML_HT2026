"""Frame sources for the UAV pipeline: image folders, image sequences, video files,
USB cameras and the Raspberry Pi camera.

Two ways of reading a recorded source (video file or ordered frame folder such
as a VisDrone-VID sequence):

* offline (default): every frame is processed, one after the other, however long
  processing takes. Good for accuracy evaluation, but not how a real camera behaves.
* real-time (``realtime=True``): the source is replayed like a live camera running
  at ``fps``. A wall clock starts at the first frame; whenever the pipeline asks for
  the next frame it gets the frame the camera would be showing *now*, and every
  frame that went by while the previous one was being processed is dropped. If the
  pipeline is faster than the camera it waits for the next frame instead.
  This is what a UAV sees: a slow link or a slow model means fewer frames
  processed (drops) and older results, not a growing backlog.

Every yielded :class:`Frame` carries ``capture_t`` (``time.perf_counter()`` at the
moment the frame was "captured"), so the caller can log the age of each result.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SEQ_FPS = 30.0   # assumed camera rate for frame folders (VisDrone-VID is ~25-30 fps)


@dataclass
class Frame:
    frame_id: str          # used as key for labels / detections
    image: np.ndarray      # BGR
    index: int             # 0-based index in the source (camera: running count)
    dropped: int           # source frames skipped since the previous yielded frame
    capture_t: float       # perf_counter() timestamp of the (simulated) capture


def list_images(folder: Path) -> List[Path]:
    return sorted(x for x in folder.iterdir() if x.suffix.lower() in IMG_EXT)


class _RealtimeClock:
    """Maps wall-clock time to a frame index of a camera running at ``fps``."""

    def __init__(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0 in real-time mode")
        self.period = 1.0 / fps
        self.t0: Optional[float] = None
        self.last = -1

    def next_index(self) -> int:
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
        idx = int((now - self.t0) / self.period)
        if idx <= self.last:                       # pipeline is ahead of the camera:
            idx = self.last + 1                    # wait for the next frame
            wait = self.t0 + idx * self.period - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        return idx

    def capture_time(self, idx: int) -> float:
        return self.t0 + idx * self.period


def _iter_folder(files: List[Path], max_frames: int, realtime: bool,
                 fps: float, ids: Optional[List[str]] = None) -> Iterator[Frame]:
    ids = ids or [f.stem for f in files]
    n = 0
    if not realtime:
        for i, f in enumerate(files):
            if max_frames and n >= max_frames:
                return
            img = cv2.imread(str(f))
            if img is not None:
                n += 1
                yield Frame(ids[i], img, i, 0, time.perf_counter())
        return
    clock, last = _RealtimeClock(fps), -1
    while not max_frames or n < max_frames:
        idx = clock.next_index()
        if idx >= len(files):
            return
        clock.last = idx
        img = cv2.imread(str(files[idx]))      # only the frames we keep are decoded
        if img is None:
            continue
        n += 1
        yield Frame(ids[idx], img, idx, idx - last - 1, clock.capture_time(idx))
        last = idx


def _iter_video(path: str, max_frames: int, realtime: bool, fps: float) -> Iterator[Frame]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open source: {path}")
    if not fps:
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_SEQ_FPS
    clock = _RealtimeClock(fps) if realtime else None
    pos, last, n = -1, -1, 0             # pos = index of the frame last decoded
    try:
        while not max_frames or n < max_frames:
            target = clock.next_index() if clock else pos + 1
            while pos < target - 1:          # skip dropped frames without decoding
                if not cap.grab():
                    return
                pos += 1
            ok, img = cap.read()
            if not ok:
                return
            pos += 1
            if clock:
                clock.last = pos
            n += 1
            # 1-based, 7 digits: same naming as VisDrone-VID frames (0000001.jpg)
            yield Frame(f"{pos + 1:07d}", img, pos, pos - last - 1,
                        clock.capture_time(pos) if clock else time.perf_counter())
            last = pos
    finally:
        cap.release()


def _iter_camera(source: str, max_frames: int) -> Iterator[Frame]:
    n = 0
    if source == "picam":
        from picamera2 import Picamera2  # only on the Raspberry Pi

        cam = Picamera2()
        cam.configure(cam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
        cam.start()
        try:
            while not max_frames or n < max_frames:
                img = cam.capture_array()  # RGB888 in picamera2 is BGR-ordered for OpenCV
                yield Frame(f"picam_{n + 1:05d}", img, n, 0, time.perf_counter())
                n += 1
        finally:
            cam.stop()
        return
    cap = cv2.VideoCapture(int(source))
    if not cap.isOpened():
        sys.exit(f"cannot open camera: {source}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always hand out the newest frame
    try:
        while not max_frames or n < max_frames:
            ok, img = cap.read()
            if not ok:
                return
            yield Frame(f"cam_{n + 1:05d}", img, n, 0, time.perf_counter())
            n += 1
    finally:
        cap.release()


def loop_id(frame_id: str, k: int) -> str:
    """Frame id in the k-th repetition of a looped source: '0000001', '0000001#1', ..."""
    return frame_id if k == 0 else f"{frame_id}#{k}"


def iter_frames(source: str, max_frames: int = 0, realtime: bool = False,
                fps: float = 0.0, loop: int = 1) -> Iterator[Frame]:
    """Yield :class:`Frame` objects from a folder (sorted by name = an image
    sequence), a single image, a video file, a camera index or ``'picam'``.

    ``realtime`` / ``fps`` apply to folders and video files (see module doc);
    ``fps=0`` means: the video's own frame rate, or 30 fps for a folder.
    Live cameras are real-time by nature and ignore both.
    ``loop`` plays a folder or video N times (longer experiments); frame ids of the k-th
    repetition get a ``#k`` suffix (see loop_id), which tools.eval_detection understands.
    """
    p = Path(source)
    if p.is_dir():
        files = list_images(p)
        loop = max(1, loop)
        yield from _iter_folder(files * loop, max_frames, realtime, fps or DEFAULT_SEQ_FPS,
                                [loop_id(f.stem, k) for k in range(loop) for f in files])
    elif p.is_file() and p.suffix.lower() in IMG_EXT:
        img = cv2.imread(str(p))
        if img is None:
            sys.exit(f"cannot read image: {source}")
        yield Frame(p.stem, img, 0, 0, time.perf_counter())
    elif source == "picam" or source.isdigit():
        yield from _iter_camera(source, max_frames)
    else:
        n = 0
        for k in range(max(1, loop)):
            for fr in _iter_video(source, max_frames - n if max_frames else 0, realtime, fps):
                fr.frame_id = loop_id(fr.frame_id, k)
                n += 1
                yield fr
