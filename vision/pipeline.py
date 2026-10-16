"""Frame-by-frame segmentation, detection, tracking and overlay pipeline.

Stages (each one is timed per frame):
    1. gray + Gaussian blur   reduce sensor noise
    2. adaptive threshold     local-contrast mask (shapes/edges, robust to lighting)
    3. background model       MOG2 learns the static scene -> mask of what moves
    4. combine + morphology   merge the two masks, close gaps, remove specks
    5. contours + filtering   extract shapes, keep those inside pixel bounds, classify by size/shape
    6. tracking               centroid tracker assigns persistent IDs
    7. draw                   boxes, IDs, live per-class counts and FPS onto the frame

Classes are shape heuristics, not a trained classifier: a blob is "person-like" if it is tall
and person-sized, "large blob" if its bounding box is bigger than that (e.g. merged people, vehicles),
else "small object". Thin streaks (e.g. fluttering tape) are rejected, and an object is only reported
once it has been tracked for `track_min_hits` frames, which removes one-frame flicker.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .tracker import CentroidTracker

CLASSES = ("person-like", "large blob", "small object")
COLORS_BGR = {"person-like": (214, 120, 42), "large blob": (52, 104, 235), "small object": (122, 175, 27)}
PANEL_BGR = {"person-like": (255, 190, 120), "large blob": (130, 170, 255), "small object": (150, 230, 150)}  # lighter, for the dark panel
STAGES = ("blur", "threshold", "background", "morphology", "contours", "tracking", "draw")


@dataclass
class Config:
    blur_kernel: int = 5            # Gaussian kernel size (odd)
    adaptive_block: int = 21        # neighbourhood size for adaptive threshold (odd)
    adaptive_c: int = 6             # constant subtracted from the local mean
    bg_history: int = 200           # frames the background model remembers
    bg_var_threshold: float = 25.0  # how different a pixel must be to count as foreground
    morph_kernel: int = 5           # structuring element size for open/close
    min_area: int = 150             # smallest contour kept (px^2)
    max_person_box: int = 6000      # bounding-box area (px^2) above which a blob is "large blob"
    person_min_aspect: float = 1.1  # height/width needed to count as person-like
    max_aspect: float = 2.0         # width/height above this is a thin streak, not a target
    min_side: int = 8               # shortest bounding-box side kept (px)
    track_max_distance: float = 60.0
    track_max_missing: int = 8      # frames a hidden object (e.g. behind a pole) is remembered
    track_min_hits: int = 5         # frames an object must persist before it is reported
    warmup_frames: int = 20         # initial frames ignored while the background model learns the scene


@dataclass
class Detection:
    track_id: int
    label: str
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    centroid: Tuple[float, float]
    area: float


@dataclass
class FrameResult:
    detections: List[Detection]
    mask: np.ndarray                  # final binary segmentation mask
    annotated: np.ndarray             # frame with overlays
    timings_ms: Dict[str, float]      # per-stage time + "total"
    counts: Dict[str, int] = field(default_factory=dict)


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


class Pipeline:
    def __init__(self, config: Config = None):
        self.cfg = config or Config()
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.bg_history, varThreshold=self.cfg.bg_var_threshold, detectShadows=True)
        self.tracker = CentroidTracker(self.cfg.track_max_distance, self.cfg.track_max_missing, self.cfg.track_min_hits)
        k = self.cfg.morph_kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.fps_ema = 0.0
        self.frame_index = 0

    def keep_shape(self, w: int, h: int) -> bool:
        """Pixel-bound filter: reject slivers and thin streaks (e.g. fluttering tape)."""
        return min(w, h) >= self.cfg.min_side and w / h <= self.cfg.max_aspect

    def classify(self, w: int, h: int) -> str:
        if w * h > self.cfg.max_person_box:
            return "large blob"
        if h / max(w, 1) >= self.cfg.person_min_aspect:
            return "person-like"
        return "small object"

    def process(self, frame: np.ndarray) -> FrameResult:
        cfg, t = self.cfg, {}
        clock = time.perf_counter
        start = mark = clock()

        def lap(stage: str) -> None:
            nonlocal mark
            now = clock()
            t[stage] = (now - mark) * 1000
            mark = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (_odd(cfg.blur_kernel),) * 2, 0)
        lap("blur")

        adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                         _odd(cfg.adaptive_block), cfg.adaptive_c)
        lap("threshold")

        fg = self.bg.apply(blur)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)  # drop the shadow label (127)
        lap("background")

        # Fill the moving region with its own contrast edges, then clean up.
        region = cv2.dilate(fg, self.kernel, iterations=2)
        mask = cv2.bitwise_or(fg, cv2.bitwise_and(adaptive, region))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        lap("morphology")

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < cfg.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if not self.keep_shape(w, h):
                continue
            blobs.append(((x, y, w, h), (x + w / 2, y + h / 2), area))
        if self.frame_index < cfg.warmup_frames:  # background model not ready: everything looks like motion
            blobs = []
        self.frame_index += 1
        lap("contours")

        ids = self.tracker.update([b[1] for b in blobs])
        detections = [Detection(tid, self.classify(b[0][2], b[0][3]), b[0], b[1], b[2])
                      for tid, b in zip(ids, blobs) if self.tracker.is_confirmed(tid)]
        lap("tracking")

        counts = {name: 0 for name in CLASSES}
        for d in detections:
            counts[d.label] += 1
        annotated = self._draw(frame.copy(), detections, counts)
        lap("draw")

        t["total"] = (clock() - start) * 1000
        self.fps_ema = 1000 / t["total"] if self.fps_ema == 0 else 0.9 * self.fps_ema + 0.1 * (1000 / t["total"])
        return FrameResult(detections, mask, annotated, t, counts)

    def _draw(self, img: np.ndarray, detections: List[Detection], counts: Dict[str, int]) -> np.ndarray:
        for d in detections:
            x, y, w, h = d.bbox
            color = COLORS_BGR[d.label]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"#{d.track_id}", (x, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        lines = [(f"{name}: {counts[name]}", PANEL_BGR[name]) for name in CLASSES]
        lines += [(f"unique objects: {self.tracker.total_ids_created}", (255, 255, 255)),
                  (f"FPS: {self.fps_ema:.0f}", (255, 255, 255))]
        panel_h = 8 + 20 * len(lines)
        overlay = img.copy()
        cv2.rectangle(overlay, (6, 6), (200, 6 + panel_h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        for i, (text, color) in enumerate(lines):
            cv2.putText(img, text, (14, 26 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return img
