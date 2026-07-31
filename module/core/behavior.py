"""Behaviour analysis: pluggable rules + debounced episode detection.

Design
------
* ``BehaviorRule`` is an abstract base. Each behaviour (phone usage, zone
  intrusion, line crossing, crowd gathering and helmet compliance today; gaze,
  sleeping in later phases) is a self-contained subclass. Rules receive persons
  plus a ``{class_id: detections}`` table, so adding a PPE item needs a rule and
  a config entry, nothing in between.
* ``BehaviorEngine`` runs every registered rule per frame and maintains a
  per-(track, rule) state machine. Raw per-frame hits are debounced into
  *episodes*: a violation must persist for ``violation_start_seconds`` before
  an :class:`Event` is emitted, and must be absent for ``violation_end_seconds``
  before the episode is considered over. This prevents duplicate/noisy events.
  A rule whose behaviour runs on a different clock can override those two with
  its own ``start_seconds``/``end_seconds`` — a crowd needs seconds to mean
  anything, while a phone appearing does not.

To add a new behaviour: implement a new ``BehaviorRule`` and register it in
:func:`build_rules`. No other file needs to change.
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

#: Non-person detections for one frame, keyed by class id (see
#: ``Config.object_class_names``). Every configured class has an entry, empty or
#: not, so a rule can index it without guarding.
Objects = Dict[int, sv.Detections]


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

    #: Per-rule debounce, in seconds. ``None`` = use the engine's global
    #: ``violation_start_seconds`` / ``violation_end_seconds``. Override when a
    #: behaviour has a genuinely different timescale from phone usage (a crowd
    #: needs seconds to be meaningful; a phone appearing does not).
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None

    def __init__(self, config: Config) -> None:
        self.config = config

    @property
    def label(self) -> str:
        """Human-readable label pulled from config."""
        return self.config.event_labels.get(self.name, self.name)

    @abstractmethod
    def evaluate(
        self, persons: sv.Detections, objects: Objects
    ) -> List[Observation]:
        """Return a *sustained* observation for every currently-violating track.

        Return an empty list for rules that only produce instant events.
        """
        raise NotImplementedError

    def instant_events(
        self, persons: sv.Detections, objects: Objects
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
        self, persons: sv.Detections, objects: Objects
    ) -> List[Observation]:
        observations: List[Observation] = []
        phones = objects.get(self.config.phone_class_id, sv.Detections.empty())
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
        self, persons: sv.Detections, objects: Objects
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
        self, persons: sv.Detections, objects: Objects
    ) -> List[Observation]:
        return []  # handled as instant (tripwire) events below

    def instant_events(
        self, persons: sv.Detections, objects: Objects
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


class CrowdGatheringRule(BehaviorRule):
    """Flags people standing together in a group larger than ``crowd_max_persons``.

    Needs no extra model — it is single-linkage clustering over the person boxes
    the detector already produces: A and B are in the same group when their boxes
    are close, and a group is the transitive closure of that (so a line of people
    each near the next counts as one gathering, which is what a queue or a huddle
    actually looks like).

    "Close" is measured as the edge-to-edge gap relative to the people's own box
    height, not in pixels. A person near the camera is simply bigger, so a fixed
    pixel threshold would group distant strangers while missing an adjacent pair;
    scaling by height makes one setting work across the whole frame and across
    source resolutions.

    Every member of an over-size group gets its own observation, so each person
    is debounced, highlighted and logged individually — same as zone intrusion.
    """

    name = "crowd_gathering"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # 0 would make every lone person a "crowd"; the allowance is at least 1.
        self._allowed = max(1, int(config.crowd_max_persons))
        self._factor = max(0.0, float(config.crowd_proximity_factor))
        # A crowd is slower than a phone: use its own debounce (see Config).
        self.start_seconds = float(config.crowd_hold_seconds)
        self.end_seconds = float(config.crowd_clear_seconds)
        # (group box, size) from the last evaluated frame, for draw().
        self._last_groups: List[Tuple[BBox, int]] = []

    @staticmethod
    def _gap(a: BBox, b: BBox) -> float:
        """Edge-to-edge distance between two boxes; 0 when they overlap."""
        dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
        dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
        return float(np.hypot(dx, dy))

    def _cluster(self, boxes: List[BBox]) -> Dict[int, List[int]]:
        """Single-linkage grouping -> ``{root index: [member indices]}``."""
        parent = list(range(len(boxes)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]      # path halving
                i = parent[i]
            return i

        for a in range(len(boxes)):
            ha = boxes[a][3] - boxes[a][1]
            for b in range(a + 1, len(boxes)):
                hb = boxes[b][3] - boxes[b][1]
                near = self._factor * 0.5 * (ha + hb)
                if self._gap(boxes[a], boxes[b]) <= near:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb

        groups: Dict[int, List[int]] = {}
        for i in range(len(boxes)):
            groups.setdefault(find(i), []).append(i)
        return groups

    def evaluate(
        self, persons: sv.Detections, objects: Objects
    ) -> List[Observation]:
        self._last_groups = []
        observations: List[Observation] = []
        if len(persons) == 0 or persons.tracker_id is None:
            return observations

        # Only tracked persons can be debounced into episodes.
        idx = [
            i for i in range(len(persons))
            if persons.tracker_id[i] is not None and persons.tracker_id[i] >= 0
        ]
        # Cheap exit: too few people on screen to exceed the allowance at all.
        # This is the common case, and it skips the O(n^2) pairing entirely.
        if len(idx) <= self._allowed:
            return observations

        boxes: List[BBox] = [tuple(persons.xyxy[i]) for i in idx]
        for members in self._cluster(boxes).values():
            if len(members) <= self._allowed:
                continue
            group_box: BBox = (
                min(boxes[k][0] for k in members),
                min(boxes[k][1] for k in members),
                max(boxes[k][2] for k in members),
                max(boxes[k][3] for k in members),
            )
            if persons.confidence is not None:
                conf = float(np.mean([persons.confidence[idx[k]] for k in members]))
            else:
                conf = 1.0
            self._last_groups.append((group_box, len(members)))
            for k in members:
                # focus_box is the whole group: a snapshot of one person tells you
                # nothing about a gathering.
                observations.append(
                    Observation(
                        track_id=int(persons.tracker_id[idx[k]]),
                        rule_name=self.name,
                        confidence=conf,
                        person_box=boxes[k],
                        focus_box=group_box,
                    )
                )
        return observations

    def draw(self, frame: np.ndarray) -> None:
        for (x1, y1, x2, y2), size in self._last_groups:
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
            cv2.putText(frame, f"CROWD x{size}", (p1[0], max(0, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


class HelmetComplianceRule(BehaviorRule):
    """Flags a person who is not wearing a helmet. Two modes, picked by the model.

    REQUIRES A PPE-TRAINED MODEL (``config.ppe_enabled`` + ``helmet_class_id``).
    COCO has no helmet class, so on the stock model this rule is never built.

    **Position match** (``head_class_id`` set — preferred). The model detects every
    visible head and every helmet, and the rule compares their positions: a helmet
    overlapping the detected head box is compliance, a helmet anywhere else is not.
    This is the only way to distinguish a *worn* helmet from a *carried* one, since
    a helmet held at chest height still falls inside any region estimated from the
    person box.

    **Coarse fallback** (no ``head_class_id``). With no head detection to aim at,
    the rule can only ask whether a helmet is somewhere in the estimated head
    region — the top ``ppe_head_region`` of the person box. That reads a carried
    helmet as compliance, which is the main reason to train the head class.

    Either way the claim is the same, and it is an **absence**: "no helmet on this
    person's head". Every other rule fires on evidence that exists — phone usage
    needs a phone box — but here a detection failure and a real violation produce
    identical evidence. A worker facing away, occluded by machinery, cropped by the
    frame edge, or too far to resolve all look like a bare head.

    So a large part of this class is suppression, each check mapping to one of those
    failure modes (the ``ppe_*`` knobs in :class:`Config`), with ``ppe_hold_seconds``
    the strongest: the helmet must be missing *continuously* before anything fires.
    A person who fails a check is **skipped, not cleared** — the rule is saying
    "cannot judge", not "compliant".
    """

    name = "no_helmet"

    def __init__(self, config: Config, frame_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(config)
        self._helmet_id = int(config.helmet_class_id)
        self._head_id = (None if config.head_class_id is None
                         else int(config.head_class_id))
        self._positive = self._head_id is not None
        self._head_region = max(0.05, float(config.ppe_head_region))
        self._head_margin = max(0.0, float(config.ppe_head_margin))
        self._min_height = max(0, int(config.ppe_min_person_height))
        self._min_conf = float(config.ppe_min_person_conf)
        self._edge_margin = max(0, int(config.ppe_edge_margin))
        self.start_seconds = float(config.ppe_hold_seconds)
        self.end_seconds = float(config.ppe_clear_seconds)
        # Frame height is only needed for the top-edge check; without it that one
        # suppressor is skipped rather than guessed.
        self._frame_size = frame_size
        self._last_violations: List[BBox] = []

    def _head_box(self, person: BBox) -> BBox:
        """The region a helmet would occupy: top slice, extended upward.

        Detectors clip the person box at the scalp, but a helmet sits *above*
        that line — without the margin a correctly-worn helmet can fall entirely
        outside the region we search.
        """
        x1, y1, x2, y2 = person
        height = y2 - y1
        return (x1, y1 - height * self._head_margin, x2, y1 + height * self._head_region)

    def _cannot_judge(self, box: BBox, conf: float) -> bool:
        """True when the evidence is too weak to accuse this person.

        Applies in both modes: the claim is always that a helmet is MISSING, so a
        person the model could not have resolved a helmet on must not be accused.
        """
        x1, y1, x2, y2 = box
        if (y2 - y1) < self._min_height:
            return True                      # too far away to resolve a helmet
        if conf < self._min_conf:
            return True                      # probably not even a person
        if y1 <= self._edge_margin:
            return True                      # head cropped by the top frame edge
        return False

    def _in_head_region(self, box: BBox, head: BBox) -> bool:
        """Association test, same shape as phone usage: contained, or overlapping."""
        return (containment(box, head) >= self.config.min_containment
                or iou(box, head) > 0.0)

    def _best_in_head(
        self, dets: sv.Detections, head: BBox
    ) -> Optional[Tuple[BBox, float]]:
        """The strongest detection sitting in ``head``, as ``(box, confidence)``."""
        best: Optional[Tuple[BBox, float]] = None
        for j in range(len(dets)):
            box: BBox = tuple(dets.xyxy[j])
            if not self._in_head_region(box, head):
                continue
            conf = float(dets.confidence[j]) if dets.confidence is not None else 1.0
            if best is None or conf > best[1]:
                best = (box, conf)
        return best

    def evaluate(
        self, persons: sv.Detections, objects: Objects
    ) -> List[Observation]:
        self._last_violations = []
        observations: List[Observation] = []
        if len(persons) == 0 or persons.tracker_id is None:
            return observations
        empty = sv.Detections.empty()
        helmets = objects.get(self._helmet_id, empty)
        heads = empty if self._head_id is None else objects.get(self._head_id, empty)

        for i in range(len(persons)):
            track_id = persons.tracker_id[i]
            if track_id is None or track_id < 0:
                continue
            person_box: BBox = tuple(persons.xyxy[i])
            person_conf = (
                float(persons.confidence[i]) if persons.confidence is not None else 1.0
            )
            head_region = self._head_box(person_box)

            # The claim is always "no helmet on this person's head", so a person we
            # could not have seen a helmet on is skipped in both modes.
            if self._cannot_judge(person_box, person_conf):
                continue

            if self._positive:
                head_hit = self._best_in_head(heads, head_region)
                if head_hit is None:
                    continue          # head not visible -> nothing to judge
                head_box, head_conf = head_hit
                # Match the helmet against the DETECTED head box, not the estimated
                # region: the region spans the top third of the person, which on a
                # standing worker reaches the chest, so a helmet CARRIED in the hand
                # would sit inside it and read as "wearing". It cannot overlap the
                # head itself. This position match is the only way to tell a worn
                # helmet from a carried one.
                if self._best_in_head(helmets, head_box) is not None:
                    continue
                confidence = head_conf
            else:
                # No head class: fall back to the coarse question, against the
                # estimated region — see `ppe_head_region` for what that costs.
                if self._best_in_head(helmets, head_region) is not None:
                    continue
                # Confidence is the PERSON's score: there is no detection of the
                # violation itself to score, and 1.0 would overstate an inference
                # drawn from missing evidence.
                confidence = person_conf

            self._last_violations.append(head_region)
            observations.append(
                Observation(
                    track_id=int(track_id),
                    rule_name=self.name,
                    confidence=confidence,
                    person_box=person_box,
                    focus_box=person_box,
                )
            )
        return observations

    def draw(self, frame: np.ndarray) -> None:
        for (x1, y1, x2, y2) in self._last_violations:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 140, 255), 2)
            cv2.putText(frame, "NO HELMET", (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2, cv2.LINE_AA)


def build_rules(
    config: Config,
    zone_polygon: Optional[Sequence[Point]] = None,
    line_start: Optional[Point] = None,
    line_end: Optional[Point] = None,
    frame_size: Optional[Tuple[int, int]] = None,
) -> List[BehaviorRule]:
    """Assemble the active rule list. Zone/line rules are added only if defined.

    Falls back to the geometry in ``config`` when arguments are not given, so the
    CLI (config-driven) and the web UI (drawn coordinates) share this factory.

    ``frame_size`` is ``(width, height)`` in pixels, used by rules that reason
    about the frame edge. Optional: a rule that does not get it drops the checks
    that need it rather than guessing.
    """
    zone_polygon = zone_polygon if zone_polygon is not None else config.zone_polygon
    line_start = line_start if line_start is not None else config.line_start
    line_end = line_end if line_end is not None else config.line_end

    rules: List[BehaviorRule] = [PhoneUsageRule(config)]
    if config.ppe_enabled:
        rules.append(HelmetComplianceRule(config, frame_size=frame_size))
        if config.head_class_id is not None:
            logger.info(
                "HelmetComplianceRule active, matching helmets against DETECTED "
                "heads (head_class_id=%d, helmet_class_id=%d, hold=%.1fs).",
                config.head_class_id, config.helmet_class_id, config.ppe_hold_seconds,
            )
        else:
            logger.info(
                "HelmetComplianceRule active, falling back to the COARSE head "
                "region (helmet_class_id=%d, hold=%.1fs, min_person_height=%dpx). "
                "Train a head class to tell a worn helmet from a carried one.",
                config.helmet_class_id, config.ppe_hold_seconds,
                config.ppe_min_person_height,
            )
    if config.crowd_enabled:
        rules.append(CrowdGatheringRule(config))
        logger.info(
            "CrowdGatheringRule active (allowed=%d, proximity=%.2f x height, "
            "hold=%.1fs).",
            max(1, int(config.crowd_max_persons)),
            config.crowd_proximity_factor,
            config.crowd_hold_seconds,
        )
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

        # Rules may declare their own timescale (see BehaviorRule.start_seconds);
        # resolve it once here rather than per frame.
        self._rule_frames: Dict[str, Tuple[int, int]] = {}
        for rule in rules:
            if rule.start_seconds is None and rule.end_seconds is None:
                continue
            start = (self._start_frames if rule.start_seconds is None
                     else max(1, round(rule.start_seconds * proc_fps)))
            end = (self._end_frames if rule.end_seconds is None
                   else max(1, round(rule.end_seconds * proc_fps)))
            self._rule_frames[rule.name] = (start, end)
            logger.info("Rule '%s' debounce: start=%d frames, end=%d frames.",
                        rule.name, start, end)

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
        objects: Objects,
        frame_index: int,
        frame,
    ) -> FrameResult:
        """Evaluate all rules for one frame and update episode state."""
        result = FrameResult()

        for rule in self._rules:
            observations = {o.track_id: o for o in rule.evaluate(persons, objects)}
            self._update_rule_states(rule, observations, frame_index, frame, result)

            # Instant (momentary) events bypass the debounce machine.
            for inst in rule.instant_events(persons, objects):
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
        start_frames, end_frames = self._rule_frames.get(
            rule.name, (self._start_frames, self._end_frames)
        )

        # 1) Tracks observed as violating this frame.
        for track_id, obs in observations.items():
            state = self._states.setdefault((track_id, rule.name), _TrackState())
            state.active_frames += 1
            state.inactive_frames = 0
            state.last_confidence = obs.confidence

            if not state.episode_open and state.active_frames >= start_frames:
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
                if state.inactive_frames >= end_frames:
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
