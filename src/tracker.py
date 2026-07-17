"""Track/entity routing stage.

The actual tracking algorithm (ByteTrack) runs inside the detector's built-in
tracker (see :meth:`detector.Detector.track`). This module owns the *identity
policy*: it turns the raw tracked detections into the two clean streams the
behaviour layer needs.

* persons  -- only those with a valid, persistent ``tracker_id``
* phones   -- tracker ids stripped (we re-associate phones to a person each
              frame, so a phone identity is not useful)

Keeping this separate means the behaviour rules never worry about malformed ids,
and future track-lifecycle logic (smoothing, re-id, dwell time) has a home.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import supervision as sv

from config import Config

logger = logging.getLogger("operator_monitor")


class Tracker:
    """Splits tracked detections into validated persons and phones."""

    def __init__(self, config: Config) -> None:
        self._config = config
        logger.info("Tracker routing ready (tracker=%s).", config.tracker_config)

    def route(
        self, detections: sv.Detections
    ) -> Tuple[sv.Detections, sv.Detections]:
        """Return ``(persons_with_ids, phones)``."""
        if len(detections) == 0:
            return detections, detections

        is_person = detections.class_id == self._config.person_class_id

        persons = detections[is_person]
        # Keep only persons the tracker has actually assigned an id to.
        if persons.tracker_id is not None and len(persons):
            valid = np.array([tid is not None and tid >= 0 for tid in persons.tracker_id])
            persons = persons[valid]

        phones = detections[~is_person]
        phones.tracker_id = None  # phones are re-associated per frame, not tracked
        return persons, phones
