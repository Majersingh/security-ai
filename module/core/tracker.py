"""Track/entity routing stage.

The actual tracking algorithm (ByteTrack) runs per-feed in
:class:`streaming.FrameProcessor`, fed by the shared model's detections. This
module owns the *identity policy*: it turns the raw tracked detections into the
two clean streams the behaviour layer needs.

* persons  -- only those with a valid, persistent ``tracker_id``
* objects  -- every other class, keyed by class id, with tracker ids stripped
              (we re-associate a phone or a helmet to a person each frame, so
              an identity for the object itself is not useful)

Keyed by class id rather than split into named streams, so adding a PPE item is
a config entry plus a rule — this module does not change.

Keeping this separate means the behaviour rules never worry about malformed ids,
and future track-lifecycle logic (smoothing, re-id, dwell time) has a home.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import supervision as sv

from config import Config

logger = logging.getLogger("operator_monitor")


class Tracker:
    """Splits tracked detections into validated persons and per-class objects."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._object_ids = sorted(config.object_class_names())
        logger.info("Tracker routing ready (object classes: %s).", self._object_ids)

    def route(
        self, detections: sv.Detections
    ) -> Tuple[sv.Detections, Dict[int, sv.Detections]]:
        """Return ``(persons_with_ids, {class_id: detections})``.

        Every configured object class gets an entry, empty or not, so rules can
        index without guarding — a rule asking for helmets on a frame with none
        should see "no helmets", not a missing key.
        """
        empty = sv.Detections.empty()
        objects: Dict[int, sv.Detections] = {cid: empty for cid in self._object_ids}
        if len(detections) == 0:
            return detections, objects

        is_person = detections.class_id == self._config.person_class_id

        persons = detections[is_person]
        # Keep only persons the tracker has actually assigned an id to.
        if persons.tracker_id is not None and len(persons):
            valid = np.array([tid is not None and tid >= 0 for tid in persons.tracker_id])
            persons = persons[valid]

        for cid in self._object_ids:
            found = detections[detections.class_id == cid]
            # Re-associated to a person each frame, so an identity is not useful.
            found.tracker_id = None
            objects[cid] = found
        return persons, objects
