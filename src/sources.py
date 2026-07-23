"""Stream frame source — never persists video to disk.

Uses PyAV (in-process libav) to decode a stream URL (RTSP / HLS / HTTP) or a
plain video file URL. Snapshots and ``events.csv`` are the only artefacts, and
those are written downstream by the pipeline — not here.

Decoding is **sequential and paced** (every frame, in order, at the source's
frame rate) so playback is smooth and frame numbers are contiguous. A source is
either:

* **live** (``rtsp://``/``rtmp://``/``.m3u8``/``.mpd``) — unbounded; reconnects
  with backoff on error/EOF; ``total_frames == 0``.
* **finite file** (a plain ``.mp4`` etc.) — plays once then ends; reports a real
  ``total_frames`` so progress and the timeline are correct.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Optional, Tuple

import numpy as np

import av

logger = logging.getLogger("operator_monitor")

Frame = Tuple[int, np.ndarray]  # (frame_index, BGR image)


class StreamURLSource:
    """Sequentially decode a stream/file URL with PyAV, paced to its fps."""

    _LIVE_SCHEMES = ("rtsp://", "rtmp://", "udp://", "srt://", "rtp://")
    _LIVE_EXTS = (".m3u8", ".m3u", ".mpd")

    def __init__(
        self, url: str, *, live: Optional[bool] = None, target_fps: Optional[float] = None,
        reconnect_backoff: float = 3.0, open_timeout: float = 15.0,
    ) -> None:
        self._url = url
        self.is_live = self._detect_live(url) if live is None else live
        self._target_fps = target_fps
        self._backoff = max(0.5, reconnect_backoff)
        self._open_timeout = open_timeout
        self._stop = False
        self._container = None
        self._stream = None

        self.fps = float(target_fps) if target_fps else 30.0
        self.width = 0
        self.height = 0
        self.total_frames = 0  # 0 = unknown/unbounded (live)
        # Timing hooks (ms), updated per frame — for profiling. last_decode_ms is
        # real decode/network time; last_wait_ms is the pacing sleep (expected).
        self.last_decode_ms = 0.0
        self.last_wait_ms = 0.0

    @classmethod
    def _detect_live(cls, url: str) -> bool:
        u = url.lower()
        if u.startswith(cls._LIVE_SCHEMES):
            return True
        path = u.split("?", 1)[0]
        return path.endswith(cls._LIVE_EXTS)

    def _open(self):
        options = {}
        if self._url.lower().startswith("rtsp"):
            options["rtsp_transport"] = "tcp"    # TCP is more reliable than UDP
            options["stimeout"] = "5000000"      # 5s socket timeout (microseconds)
        container = av.open(self._url, options=options, timeout=self._open_timeout)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        return container, stream

    def start(self) -> "StreamURLSource":
        """Open the source and read its metadata (raises on failure)."""
        self._container, self._stream = self._open()
        cc = self._stream.codec_context
        self.width = int(cc.width or 0)
        self.height = int(cc.height or 0)
        rate = self._stream.average_rate or self._stream.base_rate
        if not self._target_fps and rate:
            self.fps = float(rate)
        self.total_frames = 0 if self.is_live else int(self._stream.frames or 0)
        logger.info(
            "StreamURLSource open: %s (%s, %dx%d @ %.2f fps, total_frames=%d).",
            self._url, "live" if self.is_live else "file",
            self.width, self.height, self.fps, self.total_frames,
        )
        return self

    def frames(self) -> Iterator[Frame]:
        """Yield ``(index, BGR frame)`` sequentially, paced to the source fps.

        Live sources reconnect on error/EOF; finite files stop at EOF. Pacing
        keeps fast (e.g. HTTP-downloaded) sources playing at natural speed and
        smooths bursty HLS; if the consumer (inference) is slower than real time
        it simply runs slower — no debt accumulates.
        """
        idx = 0
        interval = 1.0 / self.fps if self.fps > 0 else 0.0
        next_t = time.monotonic()
        while not self._stop:
            try:
                decoder = self._container.decode(self._stream)
                while not self._stop:
                    t_dec = time.monotonic()           # time the real decode/network read
                    try:
                        frame = next(decoder)
                    except StopIteration:
                        break
                    self.last_decode_ms = (time.monotonic() - t_dec) * 1000.0
                    img = frame.to_ndarray(format="bgr24")
                    if self.width == 0 or self.height == 0:
                        self.height, self.width = img.shape[:2]
                    if interval:                       # pace to source fps
                        delay = next_t - time.monotonic()
                        self.last_wait_ms = max(0.0, delay) * 1000.0
                        if delay > 0:
                            time.sleep(delay)
                        next_t += interval
                        if next_t < time.monotonic():  # fell behind -> don't bank debt
                            next_t = time.monotonic()
                    yield idx, img
                    idx += 1
            except Exception as exc:  # noqa: BLE001 - decode/network hiccup
                logger.warning("Stream decode error (%s): %s", self._url, exc)

            if not (self.is_live and not self._stop):
                break                                  # finite file: played once
            logger.info("Stream ended; reconnecting in %.1fs: %s", self._backoff, self._url)
            time.sleep(self._backoff)
            try:
                if self._container:
                    self._container.close()
                self._container, self._stream = self._open()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reconnect failed (%s): %s", self._url, exc)

    def close(self) -> None:
        self._stop = True
        try:
            if self._container:
                self._container.close()
        except Exception:  # noqa: BLE001 - best-effort
            pass
