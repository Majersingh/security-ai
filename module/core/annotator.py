"""Frame annotation: draws boxes and labels onto frames.

Colour scheme (per spec): person=green, phone=blue, violation=red. This module
knows nothing about *why* a track is a violation; it just renders the decision
handed to it by the :class:`~behavior.BehaviorEngine`.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from behavior import FrameResult
from config import Config
from utils import BBox

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Annotator:
    """Draws detection/tracking/violation overlays on a frame."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def annotate(
        self,
        frame: np.ndarray,
        persons: sv.Detections,
        phones: sv.Detections,
        result: FrameResult,
    ) -> np.ndarray:
        out = frame.copy()

        # Phones (blue) first so person boxes sit on top.
        for i in range(len(phones)):
            conf = float(phones.confidence[i]) if phones.confidence is not None else 0.0
            self._draw_box(
                out, tuple(phones.xyxy[i]), self._config.color_phone, f"Phone {conf:.2f}"
            )

        # Persons: red if violating, else green.
        for i in range(len(persons)):
            track_id = int(persons.tracker_id[i]) if persons.tracker_id is not None else -1
            box: BBox = tuple(persons.xyxy[i])
            if track_id in result.violating_track_ids:
                label = f"ID {track_id} | {result.labels.get(track_id, 'Using Mobile Phone')}"
                self._draw_box(out, box, self._config.color_violation, label)
            else:
                self._draw_box(out, box, self._config.color_person, f"Person ID {track_id}")

        return out

    def _draw_box(self, frame: np.ndarray, box: BBox, color, label: str) -> None:
        x1, y1, x2, y2 = (int(v) for v in box)
        thickness = self._config.box_thickness
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        (tw, th), baseline = cv2.getTextSize(label, _FONT, self._config.font_scale, 1)
        top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2),
            _FONT,
            self._config.font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
