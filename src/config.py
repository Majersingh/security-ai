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
from typing import Tuple

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
    device: str = "cpu"                  # "cpu", "0", "cuda:0", "mps"
    # High-res CCTV footage: a phone is tiny relative to the frame, so 640
    # downscaling loses it. 1280 detects it reliably (see README notes).
    inference_imgsz: int = 1280          # model input resolution

    # ---------------------------------------------------------------- tracker
    # Ultralytics built-in tracker config: "bytetrack.yaml" or "botsort.yaml".
    tracker_config: str = "bytetrack.yaml"
    persist_tracks: bool = True          # keep ids stable across frames

    # -------------------------------------------------------------- behaviour
    # A phone counts as "in use" when it is close to a person. We inflate the
    # person box by this fraction and check whether the phone overlaps it.
    proximity_margin: float = 0.15       # 15% of person box size, each side
    min_containment: float = 0.30        # >=30% of the phone box inside person

    # Debounced state machine (seconds -> frames at runtime).
    violation_start_seconds: float = 1.0   # sustained before an event fires
    violation_end_seconds: float = 1.5     # sustained absence before it ends
    snapshot_cooldown_seconds: float = 5.0 # min gap between snapshots per track

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
        default_factory=lambda: {"phone_usage": "Mobile Phone Usage"}
    )

    def ensure_output_dirs(self) -> None:
        """Create output directories if they do not exist."""
        self.output_video.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.events_csv.parent.mkdir(parents=True, exist_ok=True)
