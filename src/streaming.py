"""Streaming pipeline: a reusable per-frame processor plus two drivers.

* :class:`FrameProcessor` -- the shared core. Wraps detector / tracker /
  behaviour engine / annotator and processes ONE frame at a time, returning the
  annotated frame and any events that fired on it. Tracking state persists
  across calls, so it works equally well for sequential video frames or a live
  camera feed.
* :class:`StreamingScanner` -- drives the processor over an uploaded video file
  and *yields* results (used by the ``/ws/{job_id}`` endpoint).
* Live camera scanning uses :class:`FrameProcessor` directly: the server feeds
  it frames pushed from the browser (``/ws/live`` endpoint).

The heavy CV work is identical to ``main.VideoProcessor``; only the frame source
and output sink differ.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np

from annotator import Annotator
from behavior import BehaviorEngine, PhoneUsageRule
from config import Config
from events import Event, EventLog, SnapshotManager
from detector import Detector
from tracker import Tracker

logger = logging.getLogger("operator_monitor")


@dataclass
class ScanUpdate:
    """One processed frame handed back to the caller (server)."""

    frame_index: int
    total_frames: int
    fps: float
    annotated: np.ndarray          # BGR frame with boxes drawn
    new_events: List[Event]        # events that fired on THIS frame (usually 0)


class FrameProcessor:
    """Stateful, single-frame pipeline shared by video and live camera modes."""

    def __init__(self, config: Config, fps: float) -> None:
        self._config = config
        config.ensure_output_dirs()
        self._detector = Detector(config)
        self._tracker = Tracker(config)
        self._event_log = EventLog(config)
        self._snapshots = SnapshotManager(config)
        self._engine = BehaviorEngine(
            config=config,
            rules=[PhoneUsageRule(config)],
            event_log=self._event_log,
            snapshots=self._snapshots,
            fps=fps,
        )
        self._annotator = Annotator(config)

    def process(self, frame: np.ndarray, frame_index: int) -> Tuple[np.ndarray, List[Event]]:
        """Run the full pipeline on one frame; return (annotated, new_events)."""
        before = len(self._event_log)
        detections = self._detector.track(frame)
        persons, phones = self._tracker.route(detections)
        result = self._engine.process(persons, phones, frame_index, frame)
        annotated = self._annotator.annotate(frame, persons, phones, result)
        new_events = self._event_log.events[before:]
        return annotated, new_events

    @property
    def event_log(self) -> EventLog:
        return self._event_log

    @property
    def event_dicts(self) -> List[dict]:
        return [asdict(e) for e in self._event_log.events]

    def finalize(self) -> None:
        """Flush the events CSV. Safe to call once at the end of a session."""
        self._event_log.save()


class StreamingScanner:
    """Drives :class:`FrameProcessor` over an uploaded video file."""

    def __init__(self, config: Config, video_path: Path) -> None:
        self._config = config
        config.input_video = Path(video_path)

        if not config.input_video.exists():
            raise FileNotFoundError(f"Input video not found: {config.input_video}")
        self._capture = cv2.VideoCapture(str(config.input_video))
        if not self._capture.isOpened():
            raise IOError(f"Could not open video: {config.input_video}")

        self.fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self._proc = FrameProcessor(config, self.fps)

    def scan(self) -> Iterator[ScanUpdate]:
        """Generator over processed frames."""
        frame_index = 0
        try:
            while True:
                ok, frame = self._capture.read()
                if not ok:
                    break
                annotated, new_events = self._proc.process(frame, frame_index)
                yield ScanUpdate(
                    frame_index=frame_index,
                    total_frames=self.total_frames,
                    fps=self.fps,
                    annotated=annotated,
                    new_events=new_events,
                )
                frame_index += 1
        finally:
            self._capture.release()
            self._proc.finalize()
            logger.info("Streaming scan finished: %d event(s).", len(self._proc.event_log))

    @property
    def event_dicts(self) -> List[dict]:
        return self._proc.event_dicts

    def close(self) -> None:
        if self._capture.isOpened():
            self._capture.release()
