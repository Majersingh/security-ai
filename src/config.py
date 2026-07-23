"""Central configuration for the CCTV Operator Monitoring System.

Every tunable value lives here. No other module hard-codes paths, thresholds,
colours or class ids. The web server constructs a :class:`Config` per feed and
may tweak a few fields (e.g. ``inference_imgsz``) before the pipeline runs.

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
    # Model input resolution — the single knob trading accuracy vs throughput
    # (also caps how many feeds fit on one GPU, since cost scales ~quadratically):
    #   640  -> ~4x faster, many more feeds, but MISSES tiny/distant phones
    #   1280 -> reliably detects small phones in high-res CCTV, but ~4x heavier
    # In batched mode this applies to the one shared model (all feeds use it).
    inference_imgsz: int = 1280           # raise to 1280 if phones are small/far

    # Process only every Nth frame (1 = every frame). E.g. on a 30 fps video,
    # frame_stride=30 analyses ~1 frame/second: much faster, coarser timing.
    # Time-based thresholds auto-adjust to the effective rate (fps / stride).
    frame_stride: int = 1

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

    # ------------------------------------------------- multi-feed (concurrent)
    max_feeds: int = 80
    # Non-batched path only: inference is serialized by a shared GPU gate this
    # many deep. Keep this SMALL (2-4) — a high value makes many feeds thrash the
    # GPU + GIL and *lowers* throughput. Ignored when batched_inference is on.
    max_concurrent_inferences: int = 4

    # Drop-when-behind: if a feed can't keep up with real time, skip stale frames
    # so latency stays bounded instead of growing forever (matters under load /
    # for live cameras). A feed more than this many seconds behind schedule drops
    # frames to catch up.
    drop_when_behind: bool = True
    stream_max_lag_seconds: float = 0.5

    # ---- Batched inference (throughput unlock for many feeds on one GPU) ----
    # When on, ALL feeds share ONE model and their frames are combined into a
    # single batched predict() call (far more GPU-efficient than one-at-a-time).
    # Tracking then runs per-feed via supervision.ByteTrack (the model is used
    # statelessly). When off, each feed gets its own model + the GPU gate above.
    batched_inference: bool = True
    batch_max_size: int = 16        # max frames combined into one GPU call
    batch_max_wait_ms: int = 12     # how long to wait to fill a batch

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
