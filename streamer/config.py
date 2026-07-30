"""Streamer settings — env-driven, deliberately independent of the module's Config.

The streamer is its own deployable with its own dependencies, so it must not import
the detection module's `Config` (which pulls in paths, model settings and thresholds
it has no use for). These are the only knobs it actually reads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class StreamerConfig:
    """Runtime settings, all overridable by environment variable."""

    # Concurrency: each stream is one extra decode of that camera, so this is what
    # stops a video wall's "all" button from swamping the host.
    streamer_max_streams: int = 16
    streamer_max_width: int = 960        # downscale for the wire
    streamer_jpeg_quality: int = 55
    # Decode/encode threads for this process. It never runs a model, so a couple of
    # OpenCV threads is right — bounded because it shares the box with detection.
    streamer_cv_threads: int = 2

    # Decode behaviour, matching the detection module's defaults.
    hw_decode: bool = True               # NVDEC, with automatic software fallback
    decode_threads: int = 1              # per stream; parallelism comes from streams
    stream_max_lag_seconds: float = 0.5  # drop stale frames rather than drift behind

    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "StreamerConfig":
        d = cls()
        return cls(
            streamer_max_streams=_int("STREAMER_MAX_STREAMS", d.streamer_max_streams),
            streamer_max_width=_int("STREAMER_MAX_WIDTH", d.streamer_max_width),
            streamer_jpeg_quality=_int("STREAMER_JPEG_QUALITY",
                                       d.streamer_jpeg_quality),
            streamer_cv_threads=_int("STREAMER_CV_THREADS", d.streamer_cv_threads),
            hw_decode=_bool("STREAMER_HW_DECODE", d.hw_decode),
            decode_threads=_int("STREAMER_DECODE_THREADS", d.decode_threads),
            stream_max_lag_seconds=_float("STREAMER_MAX_LAG_SECONDS",
                                          d.stream_max_lag_seconds),
            log_level=os.environ.get("LOG_LEVEL", d.log_level),
        )
