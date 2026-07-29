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

from utils import physical_cores

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
    # This applies to the one shared model, so it affects every feed at once.
    inference_imgsz: int = 1280           # lower to 640 if throughput matters more

    # Process only every Nth frame (1 = every frame). E.g. on a 30 fps video,
    # frame_stride=30 analyses ~1 frame/second: much faster, coarser timing.
    # Time-based thresholds auto-adjust to the effective rate (fps / stride).
    frame_stride: int = 1

    # ---------------------------------------------------------------- tracker
    # Identity tracking is per-feed (supervision.ByteTrack in FrameProcessor),
    # because the one shared model does detection statelessly for all feeds.
    # Frames a track can be missing before its id is retired.
    track_buffer_frames: int = 30

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
    log_timing: bool = True  # per-frame TIMING lines (profiling; noisy, and ON here)

    # ------------------------------------------------- multi-feed (concurrent)
    max_feeds: int = 80

    # Drop-when-behind: if a feed can't keep up with real time, skip stale frames
    # so latency stays bounded instead of growing forever (matters under load /
    # for live cameras). A feed more than this many seconds behind schedule drops
    # frames to catch up.
    drop_when_behind: bool = True
    stream_max_lag_seconds: float = 0.5

    # Hardware-accelerated decode (NVDEC on NVIDIA). Software H.264/HEVC decode is
    # the largest CPU cost per feed and the GPU's decode engines are otherwise
    # idle, so this is what actually raises the feed ceiling. Falls back to
    # software automatically when no CUDA decoder is available.
    hw_decode: bool = True

    # libav decode threads PER FEED. 0 = let libav decide, which means roughly one
    # thread per core *per feed* — with many feeds that is hundreds of threads
    # fighting over the same cores. Parallelism here comes from running many feeds,
    # so each feed wants 1 decode thread. Raise only if running very few feeds.
    decode_threads: int = 1

    # ---- Inference service (exactly ONE process owns the GPU) ----
    # Every feed, in every worker process, sends its frames to a single inference
    # process that holds the one model and combines whatever is waiting into one
    # predict() call. One CUDA context, one set of weights, and batches that
    # actually fill: N per-worker batchers would each see 1/N of the frames and
    # give back most of the batching win.
    batch_max_size: int = 16        # max frames combined into one GPU call
    batch_max_wait_ms: int = 12     # how long to wait to fill a batch

    # Frames reach that process through a pool of fixed-size shared-memory slots;
    # pickling ~3 MB arrays through a queue at hundreds of fps would otherwise
    # dominate the cost. A frame too large for a slot falls back to the queue.
    # Shared memory used = infer_slots * slot_max_height * slot_max_width * 3.
    infer_slots: int = 64
    infer_slot_max_height: int = 1088   # 1080p + slack
    infer_slot_max_width: int = 1920

    # ---- Viewer stream (browser delivery) ----
    # Annotated frames sent to browsers are throttled + shrunk INDEPENDENTLY of
    # detection, so remote viewing (especially over a tunnel) stays smooth even
    # though the server processes far faster. Detection still runs on every
    # processed frame; frames that carry events are always forwarded.
    viewer_max_fps: float = 12.0
    viewer_max_width: int = 640
    viewer_jpeg_quality: int = 45

    # ---- Multiprocess workers (escape the single-process GIL) ----
    # 0 = run everything in the web process: simple, no shared memory, no
    # inference process — the right mode for a few feeds and for CPU-only hosts.
    # >0 = spawn N worker processes; the web process becomes a thin coordinator
    # that assigns feeds to workers and relays their frames/events. Workers own
    # decode + track/rules/annotate/encode — the CPU-bound work that is the real
    # bottleneck — and do NOT own the GPU; inference is centralized (above).
    # Sized to PHYSICAL cores: hyperthread siblings add nothing for this workload.
    num_workers: int = field(default_factory=lambda: max(1, min(physical_cores(), 8)))

    # Per-process intra-op thread caps, applied at each process's entry point.
    # With N worker processes on one box each process wants ~1 compute thread; the
    # library default is for every one of them to size its pool from the whole
    # machine, which oversubscribes the CPU N-fold and thrashes the scheduler.
    cv_threads: int = 1
    torch_threads: int = 1

    # The INFERENCE process is the exception — do NOT cap it to 1. Ultralytics
    # preprocesses every frame on the CPU *inside* that process (letterbox to
    # inference_imgsz, BGR->RGB, HWC->CHW, stack), for every frame of every batch.
    # At imgsz 1280 that is tens of ms per batch of 16, and it runs serialized
    # against the GPU: starve this process of threads and the GPU sits idle waiting
    # for preprocessing, showing low utilisation while feeds queue up behind it.
    # 0 = auto: the physical cores the workers aren't using (minimum 2).
    infer_threads: int = 0

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
