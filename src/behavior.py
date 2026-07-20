"""Behaviour analysis: pluggable rules + debounced episode detection.

Design
------
* ``BehaviorRule`` is an abstract base. Each behaviour (phone usage today;
  absence, gaze, sleeping in later phases) is a self-contained subclass.
* ``BehaviorEngine`` runs every registered rule per frame and maintains a
  per-(track, rule) state machine. Raw per-frame hits are debounced into
  *episodes*: a violation must persist for ``violation_start_seconds`` before
  an :class:`Event` is emitted, and must be absent for ``violation_end_seconds``
  before the episode is considered over. This prevents duplicate/noisy events.

To add a Phase-2 behaviour: implement a new ``BehaviorRule`` and register it in
``main.py``. No other file needs to change.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import supervision as sv

from config import Config
from events import Event, EventLog, SnapshotManager
from utils import BBox, containment, format_timestamp, inflate_box, iou

logger = logging.getLogger("operator_monitor")

Point = Tuple[int, int]


@dataclass
class Observation:
    """A per-frame signal from a rule that a track is *sustaining* a violation.

    Sustained observations go through the debounce state machine (one event per
    episode). Use this for state-like behaviours (phone usage, being inside a
    zone).
    """

    track_id: int
    rule_name: str
    confidence: float
    person_box: BBox
    # Region to snapshot / highlight (defaults to the person box).
    focus_box: BBox


@dataclass
class InstantEvent:
    """A one-off event that fired on this exact frame (e.g. a line crossing).

    Instant events bypass the debounce machine and are logged immediately, since
    they represent a momentary transition rather than a sustained state.
    """

    track_id: int
    label: str
    confidence: float
    focus_box: BBox


class BehaviorRule(ABC):
    """Base class for all behaviour detectors."""

    #: Machine key used in state maps and event labels lookup.
    name: str = "rule"

    def __init__(self, config: Config) -> None:
        self.config = config

    @property
    def label(self) -> str:
        """Human-readable label pulled from config."""
        return self.config.event_labels.get(self.name, self.name)

    @abstractmethod
    def evaluate(
        self, persons: sv.Detections, phones: sv.Detections
    ) -> List[Observation]:
        """Return a *sustained* observation for every currently-violating track.

        Return an empty list for rules that only produce instant events.
        """
        raise NotImplementedError

    def instant_events(
        self, persons: sv.Detections, phones: sv.Detections
    ) -> List[InstantEvent]:
        """Return one-off events that fired on this frame (default: none)."""
        return []

    def draw(self, frame: np.ndarray) -> None:
        """Optionally draw this rule's geometry (zone/line) onto the frame."""
        return None


class PhoneUsageRule(BehaviorRule):
    """Flags a person as using a phone when a phone box is on/near them."""

    name = "phone_usage"

    def evaluate(
        self, persons: sv.Detections, phones: sv.Detections
    ) -> List[Observation]:
        observations: List[Observation] = []
        if len(persons) == 0 or len(phones) == 0:
            return observations

        for i in range(len(persons)):
            track_id = persons.tracker_id[i] if persons.tracker_id is not None else -1
            if track_id is None or track_id < 0:
                continue
            person_box: BBox = tuple(persons.xyxy[i])
            inflated = inflate_box(person_box, self.config.proximity_margin)

            best_phone: Optional[Tuple[BBox, float]] = None
            for j in range(len(phones)):
                phone_box: BBox = tuple(phones.xyxy[j])
                phone_conf = float(phones.confidence[j]) if phones.confidence is not None else 0.0
                # A phone is "with" a person if enough of it sits inside the
                # (inflated) person box, OR the raw boxes overlap meaningfully.
                if (
                    containment(phone_box, inflated) >= self.config.min_containment
                    or iou(phone_box, person_box) > 0.0
                ):
                    if best_phone is None or phone_conf > best_phone[1]:
                        best_phone = (phone_box, phone_conf)

            if best_phone is not None:
                phone_box, phone_conf = best_phone
                # Confidence of the violation = the phone's detection score.
                observations.append(
                    Observation(
                        track_id=int(track_id),
                        rule_name=self.name,
                        confidence=phone_conf,
                        person_box=person_box,
                        focus_box=person_box,
                    )
                )
        return observations


class ZoneIntrusionRule(BehaviorRule):
    """Flags a person whose bounding box OVERLAPS a user-defined polygon zone.

    Overlap-based (not a single anchor point), so *any* part of the person
    entering the zone counts. Sustained + debounced: the overlap must persist for
    ``violation_start_seconds`` before an event fires, so brief edge-clipping does
    not spam events.
    """

    name = "zone_intrusion"

    def __init__(self, config: Config, polygon: np.ndarray) -> None:
        super().__init__(config)
        self._polygon = np.asarray(polygon, dtype=np.int64)

    def _overlap_ratio(self, box: BBox) -> float:
        """Fraction of the person box area that lies inside the polygon."""
        x1, y1, x2, y2 = (int(v) for v in box)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return 0.0
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = (self._polygon - np.array([x1, y1])).astype(np.int32)
        cv2.fillPoly(mask, [shifted], 1)
        return float(mask.sum()) / float(w * h)

    def evaluate(
        self, persons: sv.Detections, phones: sv.Detections
    ) -> List[Observation]:
        observations: List[Observation] = []
        if len(persons) == 0 or persons.tracker_id is None:
            return observations
        for i in range(len(persons)):
            track_id = persons.tracker_id[i]
            if track_id is None or track_id < 0:
                continue
            box: BBox = tuple(persons.xyxy[i])
            if self._overlap_ratio(box) > self.config.zone_overlap_ratio:
                conf = float(persons.confidence[i]) if persons.confidence is not None else 1.0
                observations.append(Observation(int(track_id), self.name, conf, box, box))
        return observations

    def draw(self, frame: np.ndarray) -> None:
        pts = self._polygon.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
        x, y = int(self._polygon[0][0]), int(self._polygon[0][1])
        cv2.putText(frame, "ZONE", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 165, 255), 2, cv2.LINE_AA)


class LineCrossingRule(BehaviorRule):
    """Tripwire: fires when a person's bounding box touches/crosses the line.

    Fires immediately on the transition from *not touching* to *touching* (so it
    catches a quick pass-through), and only once per touch — it will not re-fire
    while the box stays on the line, but will fire again on a fresh touch. This
    matches "if anyone crosses the line, detect it".
    """

    name = "line_crossing"

    def __init__(self, config: Config, start: Point, end: Point) -> None:
        super().__init__(config)
        self._start = (int(start[0]), int(start[1]))
        self._end = (int(end[0]), int(end[1]))
        self._touching: Dict[int, bool] = {}  # track_id -> was touching last frame
        self._hits = 0

    def _box_touches_line(self, box: BBox) -> bool:
        x1, y1, x2, y2 = (int(v) for v in box)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return False
        intersects, _, _ = cv2.clipLine((x1, y1, w, h), self._start, self._end)
        return bool(intersects)

    def evaluate(
        self, persons: sv.Detections, phones: sv.Detections
    ) -> List[Observation]:
        return []  # handled as instant (tripwire) events below

    def instant_events(
        self, persons: sv.Detections, phones: sv.Detections
    ) -> List[InstantEvent]:
        events: List[InstantEvent] = []
        if len(persons) == 0 or persons.tracker_id is None:
            return events
        seen = set()
        for i in range(len(persons)):
            track_id = persons.tracker_id[i]
            if track_id is None or track_id < 0:
                continue
            tid = int(track_id)
            seen.add(tid)
            box: BBox = tuple(persons.xyxy[i])
            touching = self._box_touches_line(box)
            was = self._touching.get(tid, False)
            if touching and not was:  # fresh touch -> fire once
                conf = float(persons.confidence[i]) if persons.confidence is not None else 1.0
                events.append(InstantEvent(tid, self.label, conf, box))
                self._hits += 1
            self._touching[tid] = touching
        # forget tracks no longer present so a returning person can re-fire
        for gone in [t for t in self._touching if t not in seen]:
            self._touching.pop(gone, None)
        return events

    def draw(self, frame: np.ndarray) -> None:
        cv2.line(frame, self._start, self._end, (255, 255, 0), 2)
        cv2.putText(frame, f"LINE  hits:{self._hits}", (self._start[0], max(0, self._start[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)


def build_rules(
    config: Config,
    zone_polygon: Optional[Sequence[Point]] = None,
    line_start: Optional[Point] = None,
    line_end: Optional[Point] = None,
) -> List[BehaviorRule]:
    """Assemble the active rule list. Zone/line rules are added only if defined.

    Falls back to the geometry in ``config`` when arguments are not given, so the
    CLI (config-driven) and the web UI (drawn coordinates) share this factory.
    """
    zone_polygon = zone_polygon if zone_polygon is not None else config.zone_polygon
    line_start = line_start if line_start is not None else config.line_start
    line_end = line_end if line_end is not None else config.line_end

    rules: List[BehaviorRule] = [PhoneUsageRule(config)]
    if zone_polygon and len(zone_polygon) >= 3:
        rules.append(ZoneIntrusionRule(config, np.array(zone_polygon, dtype=np.int64)))
        logger.info("ZoneIntrusionRule active (%d points).", len(zone_polygon))
    if line_start and line_end:
        rules.append(LineCrossingRule(config, line_start, line_end))
        logger.info("LineCrossingRule active (%s -> %s).", line_start, line_end)
    return rules


@dataclass
class _TrackState:
    """Debounce state for one (track, rule) pair."""

    active_frames: int = 0
    inactive_frames: int = 0
    episode_open: bool = False
    last_snapshot_frame: int = -10**9
    last_confidence: float = 0.0


@dataclass
class FrameResult:
    """What the engine produced for a frame, consumed by the annotator."""

    violating_track_ids: set = field(default_factory=set)
    # track_id -> label to render above the red box
    labels: Dict[int, str] = field(default_factory=dict)


class BehaviorEngine:
    """Runs rules, debounces into episodes, emits events and snapshots."""

    def __init__(
        self,
        config: Config,
        rules: List[BehaviorRule],
        event_log: EventLog,
        snapshots: SnapshotManager,
        fps: float,
        processing_fps: float | None = None,
    ) -> None:
        self._config = config
        self._rules = rules
        self._event_log = event_log
        self._snapshots = snapshots
        # Real video fps: used for event timestamps and the snapshot cooldown
        # (which compares real frame indices).
        self._fps = fps if fps > 0 else 30.0
        # Effective rate of frames actually processed (fps / frame_stride).
        # Start/end thresholds count *processed* frames, so they use this.
        proc_fps = processing_fps if processing_fps and processing_fps > 0 else self._fps

        self._start_frames = max(1, round(config.violation_start_seconds * proc_fps))
        self._end_frames = max(1, round(config.violation_end_seconds * proc_fps))
        self._snapshot_cooldown = max(1, round(config.snapshot_cooldown_seconds * self._fps))

        # (track_id, rule_name) -> _TrackState
        self._states: Dict[Tuple[int, str], _TrackState] = {}
        logger.info(
            "BehaviorEngine ready: start=%d frames, end=%d frames, snapshot_cooldown=%d frames.",
            self._start_frames,
            self._end_frames,
            self._snapshot_cooldown,
        )

    def process(
        self,
        persons: sv.Detections,
        phones: sv.Detections,
        frame_index: int,
        frame,
    ) -> FrameResult:
        """Evaluate all rules for one frame and update episode state."""
        result = FrameResult()

        for rule in self._rules:
            observations = {o.track_id: o for o in rule.evaluate(persons, phones)}
            self._update_rule_states(rule, observations, frame_index, frame, result)

            # Instant (momentary) events bypass the debounce machine.
            for inst in rule.instant_events(persons, phones):
                self._emit_instant(rule, inst, frame_index, frame, result)

        return result

    def draw_overlays(self, frame: np.ndarray) -> None:
        """Let each rule render its geometry (zones/lines) onto the frame."""
        for rule in self._rules:
            rule.draw(frame)

    def _update_rule_states(
        self,
        rule: BehaviorRule,
        observations: Dict[int, Observation],
        frame_index: int,
        frame,
        result: FrameResult,
    ) -> None:
        # 1) Tracks observed as violating this frame.
        for track_id, obs in observations.items():
            state = self._states.setdefault((track_id, rule.name), _TrackState())
            state.active_frames += 1
            state.inactive_frames = 0
            state.last_confidence = obs.confidence

            if not state.episode_open and state.active_frames >= self._start_frames:
                state.episode_open = True
                self._emit_event(rule, obs, frame_index)

            if state.episode_open:
                result.violating_track_ids.add(track_id)
                result.labels[track_id] = f"{rule.label} {obs.confidence:.2f}"
                self._maybe_snapshot(rule, obs, state, frame_index, frame)

        # 2) Tracks with open episodes not seen this frame -> maybe close.
        for (track_id, rule_name), state in list(self._states.items()):
            if rule_name != rule.name or track_id in observations:
                continue
            if state.episode_open:
                state.inactive_frames += 1
                if state.inactive_frames >= self._end_frames:
                    state.episode_open = False
                    state.active_frames = 0
                    logger.debug("Episode closed: id=%d rule=%s", track_id, rule_name)
            else:
                state.active_frames = 0

    def _emit_event(self, rule: BehaviorRule, obs: Observation, frame_index: int) -> None:
        self._event_log.add(
            Event(
                timestamp=format_timestamp(frame_index, self._fps),
                frame_number=frame_index,
                person_id=obs.track_id,
                event=rule.label,
                confidence=round(obs.confidence, 3),
            )
        )

    def _emit_instant(
        self,
        rule: BehaviorRule,
        inst: InstantEvent,
        frame_index: int,
        frame,
        result: FrameResult,
    ) -> None:
        """Log a one-off event immediately and flag it for this frame's overlay."""
        self._event_log.add(
            Event(
                timestamp=format_timestamp(frame_index, self._fps),
                frame_number=frame_index,
                person_id=inst.track_id,
                event=inst.label,
                confidence=round(inst.confidence, 3),
            )
        )
        result.violating_track_ids.add(inst.track_id)
        result.labels[inst.track_id] = inst.label
        # Crossings are discrete, so snapshot every one (no cooldown).
        self._snapshots.save(
            frame=frame,
            frame_number=frame_index,
            prefix=rule.name.split("_")[0],  # e.g. "line"
            crop_box=inflate_box(inst.focus_box, 0.05),
        )

    def _maybe_snapshot(
        self,
        rule: BehaviorRule,
        obs: Observation,
        state: _TrackState,
        frame_index: int,
        frame,
    ) -> None:
        if frame_index - state.last_snapshot_frame < self._snapshot_cooldown:
            return
        state.last_snapshot_frame = frame_index
        self._snapshots.save(
            frame=frame,
            frame_number=frame_index,
            prefix=rule.name.split("_")[0],  # e.g. "phone"
            crop_box=inflate_box(obs.focus_box, 0.05),
        )
