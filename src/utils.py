"""Cross-cutting utilities: logging, geometry and time formatting.

These helpers carry no domain knowledge so they can be reused by any module
(detector, tracker, behaviour rules, annotator).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

# A bounding box is (x1, y1, x2, y2) in pixel coordinates.
BBox = Tuple[float, float, float, float]


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the package logger.

    Idempotent: repeated calls will not add duplicate handlers.
    """
    logger = logging.getLogger("operator_monitor")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def inflate_box(box: BBox, margin: float) -> BBox:
    """Expand a box outward by ``margin`` fraction of its width/height."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * margin, h * margin
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def _intersection_area(a: BBox, b: BBox) -> float:
    """Area of overlap between two boxes (0.0 if disjoint)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes in [0, 1]."""
    inter = _intersection_area(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer`` (in [0, 1]).

    Better than IoU for a small phone box vs. a large person box: a phone fully
    inside a person yields ~1.0 here but a low IoU because of the size gap.
    """
    inner_area = _area(inner)
    if inner_area <= 0:
        return 0.0
    return _intersection_area(inner, outer) / inner_area


def format_timestamp(frame_index: int, fps: float) -> str:
    """Convert a frame index to an ``HH:MM:SS.mmm`` timestamp string."""
    seconds = frame_index / fps if fps > 0 else 0.0
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def clamp_box_to_frame(box: BBox, width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp and round a box so it fits inside the frame (for cropping)."""
    x1 = int(np.clip(box[0], 0, width - 1))
    y1 = int(np.clip(box[1], 0, height - 1))
    x2 = int(np.clip(box[2], 0, width))
    y2 = int(np.clip(box[3], 0, height))
    return x1, y1, x2, y2
