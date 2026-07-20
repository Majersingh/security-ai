"""Entry point: orchestrates the operator-monitoring pipeline.

    detector -> tracker -> behaviour engine -> annotator -> video/CSV/snapshots

Run with defaults:
    python src/main.py

Override any config value on the CLI, e.g.:
    python src/main.py --input input/operator.mp4 --device 0 --no-video
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from annotator import Annotator
from behavior import BehaviorEngine, build_rules
from config import Config
from detector import Detector
from events import EventLog, SnapshotManager
from tracker import Tracker
from utils import setup_logging


class VideoProcessor:
    """Runs the full pipeline over a single video file."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._logger = setup_logging(config.log_level)
        config.ensure_output_dirs()

        self._capture = self._open_video(config.input_video)
        self._fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self._stride = max(1, config.frame_stride)
        self._effective_fps = self._fps / self._stride

        self._detector = Detector(config)
        self._tracker = Tracker(config)
        self._event_log = EventLog(config)
        self._snapshots = SnapshotManager(config)
        self._engine = BehaviorEngine(
            config=config,
            rules=build_rules(config),  # phone + optional zone/line rules
            event_log=self._event_log,
            snapshots=self._snapshots,
            fps=self._fps,                    # real fps -> timestamps
            processing_fps=self._effective_fps,  # effective rate -> thresholds
        )
        self._annotator = Annotator(config)
        self._writer = self._open_writer() if config.write_output_video else None

        self._logger.info(
            "Video: %s | %dx%d @ %.2f fps | %d frames | stride=%d (effective %.2f fps)",
            config.input_video.name,
            self._width,
            self._height,
            self._fps,
            self._total,
            self._stride,
            self._effective_fps,
        )

    def _open_video(self, path: Path) -> cv2.VideoCapture:
        if not path.exists():
            raise FileNotFoundError(f"Input video not found: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        return cap

    def _open_writer(self) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Only processed frames are written, so use the effective fps to keep the
        # output video's wall-clock duration equal to the original.
        writer = cv2.VideoWriter(
            str(self._config.output_video), fourcc, self._effective_fps,
            (self._width, self._height),
        )
        if not writer.isOpened():
            raise IOError(f"Could not open output video for writing: {self._config.output_video}")
        return writer

    def run(self) -> None:
        frame_index = 0      # real frame position (read from the video)
        analyzed = 0         # frames actually run through the pipeline
        try:
            while True:
                ok, frame = self._capture.read()
                if not ok:
                    break

                # Frame sampling: skip frames not on the stride (but keep the
                # real frame_index advancing so timestamps stay accurate).
                if frame_index % self._stride != 0:
                    frame_index += 1
                    continue

                detections = self._detector.track(frame)
                persons, phones = self._tracker.route(detections)
                result = self._engine.process(persons, phones, frame_index, frame)
                annotated = self._annotator.annotate(frame, persons, phones, result)
                self._engine.draw_overlays(annotated)  # draw zone/line overlays
                analyzed += 1

                if self._writer is not None:
                    self._writer.write(annotated)

                if analyzed % 100 == 0 and self._total > 0:
                    self._logger.info(
                        "Analyzed %d frames | at %d/%d (%.1f%%)",
                        analyzed,
                        frame_index,
                        self._total,
                        100.0 * frame_index / self._total,
                    )
                frame_index += 1
        finally:
            self._cleanup()
            self._event_log.save()
            self._logger.info(
                "Done. Analyzed %d / %d frames (stride=%d), %d event(s).",
                analyzed,
                frame_index,
                self._stride,
                len(self._event_log),
            )

    def _cleanup(self) -> None:
        self._capture.release()
        if self._writer is not None:
            self._writer.release()


def build_config_from_args(argv: list[str]) -> Config:
    """Parse CLI args and fold overrides into a Config instance."""
    parser = argparse.ArgumentParser(description="CCTV Operator Monitoring System (PoC).")
    parser.add_argument("--input", type=Path, help="Path to input MP4.")
    parser.add_argument("--output", type=Path, help="Path to annotated output MP4.")
    parser.add_argument("--model", type=Path, help="Path to YOLOv11 weights.")
    parser.add_argument("--device", type=str, help="Inference device: cpu, 0, cuda:0, mps.")
    parser.add_argument("--conf", type=float, help="Detection confidence threshold.")
    parser.add_argument(
        "--start-seconds", type=float, help="Seconds a violation must persist before firing."
    )
    parser.add_argument(
        "--frame-stride", type=int,
        help="Process every Nth frame (1=all). Higher = faster, coarser timing.",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip writing the output video.")
    parser.add_argument("--log-level", type=str, help="DEBUG, INFO, WARNING, ERROR.")
    args = parser.parse_args(argv)

    config = Config()
    if args.input:
        config.input_video = args.input
    if args.output:
        config.output_video = args.output
    if args.model:
        config.model_path = args.model
    if args.device:
        config.device = args.device
    if args.conf is not None:
        config.confidence_threshold = args.conf
    if args.start_seconds is not None:
        config.violation_start_seconds = args.start_seconds
    if args.frame_stride is not None:
        config.frame_stride = max(1, args.frame_stride)
    if args.no_video:
        config.write_output_video = False
    if args.log_level:
        config.log_level = args.log_level
    return config


def main(argv: list[str] | None = None) -> int:
    config = build_config_from_args(argv if argv is not None else sys.argv[1:])
    logger = setup_logging(config.log_level)
    try:
        VideoProcessor(config).run()
    except (FileNotFoundError, IOError) as exc:
        logger.error("Fatal: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
