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
from typing import Dict, List, Optional, Tuple

import supervision as sv

from config import Config
from events import Event, EventLog, SnapshotManager
from utils import BBox, containment, format_timestamp, inflate_box, iou

logger = logging.getLogger("operator_monitor")


@dataclass
class Observation:
    """A per-frame signal from a rule that a track is violating."""

    track_id: int
    rule_name: str
    confidence: float
    person_box: BBox
    # Region to snapshot / highlight (defaults to the person box).
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
        """Return an observation for every *currently violating* track."""
        raise NotImplementedError


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
    ) -> None:
        self._config = config
        self._rules = rules
        self._event_log = event_log
        self._snapshots = snapshots
        self._fps = fps if fps > 0 else 30.0

        self._start_frames = max(1, round(config.violation_start_seconds * self._fps))
        self._end_frames = max(1, round(config.violation_end_seconds * self._fps))
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

        return result

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
