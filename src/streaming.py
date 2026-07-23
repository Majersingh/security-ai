"""Per-frame processing core for the live-stream pipeline.

:class:`FrameProcessor` wraps detector / tracker / behaviour engine / annotator
and processes ONE frame at a time, returning either an annotated frame or plain
detection data plus any events that fired. Tracking state persists across calls,
so it drives a live stream frame by frame. It is used by :mod:`feeds`, which owns
one processor (and thus one model) per feed.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import List, Optional, Tuple

import numpy as np

import supervision as sv

from annotator import Annotator
from behavior import BehaviorEngine, build_rules
from config import Config
from events import Event, EventLog, SnapshotManager
from detector import Detector
from tracker import Tracker
from utils import BBox  # noqa: F401  (re-exported for type users)

logger = logging.getLogger("operator_monitor")

Point = Tuple[int, int]


class FrameProcessor:
    """Stateful, single-frame pipeline (one model + tracker per instance)."""

    def __init__(
        self,
        config: Config,
        fps: float,
        processing_fps: float | None = None,
        zone_polygon: Optional[List[Point]] = None,
        line_start: Optional[Point] = None,
        line_end: Optional[Point] = None,
        batched: bool = False,
    ) -> None:
        # zone/line coordinates are *normalized* (0..1 fractions of width/height)
        # so they are independent of the frame resolution. They are scaled to
        # pixels lazily on the first frame, once the real frame size is known.
        self._config = config
        config.ensure_output_dirs()
        # Batched mode: a shared model does detection elsewhere and hands us the
        # detections; this feed owns only its identity tracker (supervision
        # ByteTrack). Non-batched: this feed owns a full Detector (model + tracker).
        self._batched = batched
        if batched:
            self._detector = None
            self._bytetrack = sv.ByteTrack(
                frame_rate=max(1, int(round(processing_fps or fps or 30)))
            )
        else:
            self._detector = Detector(config)
            self._bytetrack = None
        self._tracker = Tracker(config)
        self._event_log = EventLog(config)
        self._snapshots = SnapshotManager(config)
        self._annotator = Annotator(config)
        self._fps = fps
        self._processing_fps = processing_fps
        self._norm_zone = zone_polygon
        self._norm_line_start = line_start
        self._norm_line_end = line_end
        self._engine: Optional[BehaviorEngine] = None  # built on first frame
        # Geometry drawn on a running feed lands here (a single atomic reference
        # assignment from another thread), applied at the start of the next frame.
        self._pending_geom: Optional[tuple] = None

    def set_geometry(self, zone_polygon, line_start, line_end) -> None:
        """Update the detection zone/line on a *running* processor (normalized
        0..1 coords). The behaviour engine is rebuilt on the next frame."""
        self._pending_geom = (zone_polygon, line_start, line_end)

    def _build_engine(self, width: int, height: int) -> None:
        def poly_px(poly):
            if not poly:
                return None
            return [(int(round(fx * width)), int(round(fy * height))) for fx, fy in poly]

        def pt_px(pt):
            return (int(round(pt[0] * width)), int(round(pt[1] * height))) if pt else None

        rules = build_rules(
            self._config,
            zone_polygon=poly_px(self._norm_zone),
            line_start=pt_px(self._norm_line_start),
            line_end=pt_px(self._norm_line_end),
        )
        self._engine = BehaviorEngine(
            config=self._config,
            rules=rules,
            event_log=self._event_log,
            snapshots=self._snapshots,
            fps=self._fps,
            processing_fps=self._processing_fps,
        )

    def _run(self, frame: np.ndarray, frame_index: int, detections=None):
        """Shared core: detect -> track -> rules. Returns raw results.

        In batched mode ``detections`` are supplied by the shared model; otherwise
        this processor's own detector produces them.
        """
        pending = self._pending_geom
        if pending is not None:                       # geometry changed at runtime
            self._pending_geom = None
            self._norm_zone, self._norm_line_start, self._norm_line_end = pending
            self._engine = None                       # force rebuild below
        if self._engine is None:
            h, w = frame.shape[:2]
            self._build_engine(w, h)
        before = len(self._event_log)
        if self._batched:
            # Detection came from the shared batched model; track identities here.
            tracked = self._bytetrack.update_with_detections(detections)
            persons, phones = self._tracker.route(tracked)
        else:
            det = self._detector.track(frame)   # own model: fused detect + track
            persons, phones = self._tracker.route(det)
        result = self._engine.process(persons, phones, frame_index, frame)
        new_events = self._event_log.events[before:]
        return persons, phones, result, new_events

    def process(self, frame: np.ndarray, frame_index: int, detections=None) -> Tuple[np.ndarray, List[Event]]:
        """Run the pipeline and return an ANNOTATED frame (used for stream feeds)."""
        persons, phones, result, new_events = self._run(frame, frame_index, detections)
        annotated = self._annotator.annotate(frame, persons, phones, result)
        self._engine.draw_overlays(annotated)  # zone/line overlays
        return annotated, new_events

    def process_json(self, frame: np.ndarray, frame_index: int, detections=None):
        """Run the pipeline and return DETECTIONS as plain data (no image).

        Returns (boxes, new_events, width, height).
        """
        h, w = frame.shape[:2]
        persons, phones, result, new_events = self._run(frame, frame_index, detections)
        boxes: List[dict] = []
        for i in range(len(persons)):
            tid = int(persons.tracker_id[i]) if persons.tracker_id is not None else -1
            x1, y1, x2, y2 = (float(v) for v in persons.xyxy[i])
            boxes.append({
                "cls": "person", "tid": tid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "violation": tid in result.violating_track_ids,
                "label": result.labels.get(tid, ""),
            })
        for j in range(len(phones)):
            x1, y1, x2, y2 = (float(v) for v in phones.xyxy[j])
            conf = float(phones.confidence[j]) if phones.confidence is not None else 0.0
            boxes.append({"cls": "phone", "conf": conf,
                          "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return boxes, new_events, w, h

    @property
    def event_log(self) -> EventLog:
        return self._event_log

    @property
    def event_dicts(self) -> List[dict]:
        return [asdict(e) for e in self._event_log.events]

    def finalize(self) -> None:
        """Flush the events CSV. Safe to call once at the end of a session."""
        self._event_log.save()
