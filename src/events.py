"""Event modelling and persistence: CSV log + violation snapshots.

Kept separate from behaviour logic so the rules only *decide* that something
happened, while this module is solely responsible for recording it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd

from config import Config
from utils import BBox, clamp_box_to_frame

logger = logging.getLogger("operator_monitor")


@dataclass
class Event:
    """A single confirmed violation, ready to be written to the CSV."""

    timestamp: str
    frame_number: int
    person_id: int
    event: str
    confidence: float


class EventLog:
    """Accumulates :class:`Event` objects and flushes them to ``events.csv``."""

    # Column order/labels as required by the specification.
    _COLUMNS = {
        "timestamp": "Timestamp",
        "frame_number": "Frame Number",
        "person_id": "Person ID",
        "event": "Event",
        "confidence": "Confidence",
    }

    def __init__(self, config: Config) -> None:
        self._config = config
        self._events: List[Event] = []

    def add(self, event: Event) -> None:
        self._events.append(event)
        logger.info(
            "EVENT | frame=%d | id=%d | %s | conf=%.2f",
            event.frame_number,
            event.person_id,
            event.event,
            event.confidence,
        )

    def __len__(self) -> int:
        return len(self._events)

    @property
    def events(self) -> List[Event]:
        """Read-only view of accumulated events (for streaming/inspection)."""
        return list(self._events)

    def save(self) -> None:
        """Write all accumulated events to the configured CSV path."""
        df = pd.DataFrame([asdict(e) for e in self._events])
        if df.empty:
            df = pd.DataFrame(columns=list(self._COLUMNS.keys()))
        df = df.rename(columns=self._COLUMNS)[list(self._COLUMNS.values())]
        df.to_csv(self._config.events_csv, index=False)
        logger.info("Wrote %d event(s) to '%s'.", len(self._events), self._config.events_csv)


class SnapshotManager:
    """Saves JPEG crops of the frame when a violation occurs."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def save(
        self,
        frame: np.ndarray,
        frame_number: int,
        prefix: str = "phone",
        crop_box: BBox | None = None,
    ) -> Path:
        """Save a snapshot named e.g. ``phone_000123.jpg``.

        If ``crop_box`` is given the snapshot is cropped to that region with a
        small pad; otherwise the full annotated frame is saved.
        """
        filename = f"{prefix}_{frame_number:06d}.jpg"
        path = self._config.snapshots_dir / filename

        image = frame
        if crop_box is not None:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = clamp_box_to_frame(crop_box, w, h)
            if x2 > x1 and y2 > y1:
                image = frame[y1:y2, x1:x2]

        cv2.imwrite(str(path), image)
        logger.debug("Saved snapshot '%s'.", path)
        return path
