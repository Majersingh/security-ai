"""Central configuration for the CCTV Operator Monitoring System.

Every tunable value lives here. No other module hard-codes paths, thresholds,
colours or class ids. Values can be overridden at runtime via CLI flags in
``main.py`` (which mutate a :class:`Config` instance before the pipeline runs).

Thresholds are expressed in *seconds* rather than frames. The pipeline converts
them to a frame count using the video's real FPS, so behaviour is consistent
whether the source is 15, 25 or 30 fps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Project root = one level above this ``src`` directory.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# BGR colours (OpenCV convention).
Color = Tuple[int, int, int]


@dataclass
class Config:
    """Runtime configuration. Instantiate once and pass down the pipeline."""

    # ------------------------------------------------------------------ paths
    input_video: Path = ROOT_DIR / "input" / "operator.mp4"
    output_video: Path = ROOT_DIR / "output" / "annotated.mp4"
    events_csv: Path = ROOT_DIR / "output" / "events.csv"
    snapshots_dir: Path = ROOT_DIR / "output" / "snapshots"
    model_path: Path = ROOT_DIR / "models" / "yolo11n.pt"

    # ------------------------------------------------------------------ model
    # COCO class ids: 0 = person, 67 = cell phone. We ignore all other classes.
    person_class_id: int = 0
    phone_class_id: int = 67
    confidence_threshold: float = 0.25   # min detection confidence to keep
    iou_threshold: float = 0.5           # NMS IoU for the detector
    # "auto" picks GPU (cuda/mps) if available, else cpu. Or force: "cpu","0","mps".
    device: str = "auto"
    # High-res CCTV footage: a phone is tiny relative to the frame, so 640
    # downscaling loses it. 1280 detects it reliably (see README notes).
    inference_imgsz: int = 1280          # model input resolution

    # Process only every Nth frame (1 = every frame). E.g. on a 30 fps video,
    # frame_stride=30 analyses ~1 frame/second: much faster, coarser timing.
    # Time-based thresholds auto-adjust to the effective rate (fps / stride).
    frame_stride: int = 10

    # ---------------------------------------------------------------- tracker
    # Ultralytics built-in tracker config: "bytetrack.yaml" or "botsort.yaml".
    tracker_config: str = "bytetrack.yaml"
    persist_tracks: bool = True          # keep ids stable across frames

    # -------------------------------------------------------------- behaviour
    # A phone counts as "in use" when it is close to a person. We inflate the
    # person box by this fraction and check whether the phone overlaps it.
    proximity_margin: float = 0.15       # 15% of person box size, each side
    min_containment: float = 0.10        # >=10% of the phone box inside person

    # Debounced state machine (seconds -> frames at runtime).
    violation_start_seconds: float = 1.0   # sustained before an event fires
    violation_end_seconds: float = 1.5     # sustained absence before it ends
    snapshot_cooldown_seconds: float = 5.0 # min gap between snapshots per track

    # ------------------------------------------------------- zone / line rules
    # Optional geometry in native frame pixels. None = rule disabled.
    # zone_polygon: list of (x, y) points, e.g. [(400,200),(900,200),(900,700),(400,700)]
    # line_start / line_end: two (x, y) points defining a crossing line.
    # The web UI can supply these by letting the user draw on the video.
    zone_polygon: Optional[List[Tuple[int, int]]] = None
    line_start: Optional[Tuple[int, int]] = None
    line_end: Optional[Tuple[int, int]] = None
    # A person triggers the zone when their bounding box OVERLAPS the polygon by
    # at least this fraction of the box area. 0.0 = any overlap at all triggers.
    zone_overlap_ratio: float = 0.0

    # -------------------------------------------------------------- rendering
    color_person: Color = (0, 200, 0)      # green
    color_phone: Color = (255, 128, 0)     # blue-ish (BGR)
    color_violation: Color = (0, 0, 255)   # red
    box_thickness: int = 2
    font_scale: float = 0.6
    write_output_video: bool = True

    # --------------------------------------------------------------- runtime
    log_level: str = "INFO"

    # Human-readable event label, kept here so wording is not scattered around.
    event_labels: dict = field(
        default_factory=lambda: {
            "phone_usage": "Mobile Phone Usage",
            "zone_intrusion": "Zone Intrusion",
            "line_crossing": "Line Crossing",
        }
    )

    def ensure_output_dirs(self) -> None:
        """Create output directories if they do not exist."""
        self.output_video.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.events_csv.parent.mkdir(parents=True, exist_ok=True)
