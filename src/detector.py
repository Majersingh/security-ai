"""Perception stage: YOLOv11 detection + built-in tracking.

Wraps the Ultralytics model. Two entry points:

* :meth:`detect`  -- pure detection on a single image (no identities). Handy for
  tests or non-video use.
* :meth:`track`   -- detection *and* tracking in one forward pass, using
  Ultralytics' built-in ByteTrack. This is the maintained tracking path;
  ``supervision.ByteTrack`` is deprecated for removal in supervision 0.30, and
  fusing detect+track avoids a second forward pass.

Either way only the classes we care about (person, cell phone) are returned.
"""

from __future__ import annotations

import logging

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import Config

logger = logging.getLogger("operator_monitor")


class Detector:
    """YOLOv11-based person / phone detector with optional built-in tracking."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._keep_classes = sorted({config.person_class_id, config.phone_class_id})
        logger.info("Loading YOLO model from '%s' ...", config.model_path)
        # Ultralytics downloads the weight automatically if it is not present.
        self._model = YOLO(str(config.model_path))
        logger.info("Model loaded (device=%s).", config.device)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Detection only (no tracking ids)."""
        results = self._model.predict(
            source=frame,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            imgsz=self._config.inference_imgsz,
            device=self._config.device,
            classes=self._keep_classes,
            verbose=False,
        )[0]
        return self._filter(sv.Detections.from_ultralytics(results))

    def track(self, frame: np.ndarray) -> sv.Detections:
        """Detection + tracking. Persons carry a persistent ``tracker_id``.

        ``persist=True`` keeps tracker state across sequential calls, so IDs stay
        stable frame to frame. Must be called on frames in temporal order.
        """
        results = self._model.track(
            source=frame,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            imgsz=self._config.inference_imgsz,
            device=self._config.device,
            classes=self._keep_classes,
            tracker=self._config.tracker_config,
            persist=self._config.persist_tracks,
            verbose=False,
        )[0]
        return self._filter(sv.Detections.from_ultralytics(results))

    def _filter(self, detections: sv.Detections) -> sv.Detections:
        """Defensive class filter in case the model returns other classes."""
        if detections.class_id is not None and len(detections):
            mask = np.isin(detections.class_id, self._keep_classes)
            detections = detections[mask]
        return detections
