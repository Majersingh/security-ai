"""Streaming scanner: runs the pipeline and *yields* per-frame results.

This is the web-facing counterpart to :class:`main.VideoProcessor`. Instead of
writing an output file, it reuses the exact same detector / tracker / behaviour
engine / annotator and yields each annotated frame plus any events that fired on
that frame, so a server can push them to a browser over a WebSocket.

The heavy CV work is unchanged; only the *output sink* differs.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List

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


class StreamingScanner:
    """Processes a video frame-by-frame, yielding annotated frames + events."""

    def __init__(self, config: Config, video_path: Path) -> None:
        self._config = config
        config.input_video = Path(video_path)
        config.ensure_output_dirs()

        if not config.input_video.exists():
            raise FileNotFoundError(f"Input video not found: {config.input_video}")
        self._capture = cv2.VideoCapture(str(config.input_video))
        if not self._capture.isOpened():
            raise IOError(f"Could not open video: {config.input_video}")

        self.fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self._detector = Detector(config)
        self._tracker = Tracker(config)
        self._event_log = EventLog(config)
        self._snapshots = SnapshotManager(config)
        self._engine = BehaviorEngine(
            config=config,
            rules=[PhoneUsageRule(config)],
            event_log=self._event_log,
            snapshots=self._snapshots,
            fps=self.fps,
        )
        self._annotator = Annotator(config)

    def scan(self) -> Iterator[ScanUpdate]:
        """Generator over processed frames. Same pipeline as VideoProcessor."""
        frame_index = 0
        try:
            while True:
                ok, frame = self._capture.read()
                if not ok:
                    break

                before = len(self._event_log)
                detections = self._detector.track(frame)
                persons, phones = self._tracker.route(detections)
                result = self._engine.process(persons, phones, frame_index, frame)
                annotated = self._annotator.annotate(frame, persons, phones, result)

                new_events = self._event_log.events[before:]
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
            self._event_log.save()
            logger.info("Streaming scan finished: %d event(s).", len(self._event_log))

    @property
    def event_dicts(self) -> List[dict]:
        return [asdict(e) for e in self._event_log.events]

    def close(self) -> None:
        if self._capture.isOpened():
            self._capture.release()
