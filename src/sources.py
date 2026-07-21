"""Live stream frame source — never persists video to disk.

Uses PyAV (in-process libav) to decode a live stream URL. Snapshots and
``events.csv`` are the only artefacts, and those are written downstream by the
pipeline — not here.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Iterator, Optional, Tuple

import numpy as np

import av

logger = logging.getLogger("operator_monitor")

Frame = Tuple[int, np.ndarray]  # (frame_index, BGR image)


class StreamURLSource:
    """Decode a live stream URL (RTSP / HLS / HTTP) with PyAV.

    A background thread decodes continuously and keeps only the LATEST frame
    (*drop-to-latest*): a live camera runs in real time and cannot be slowed, so
    when inference lags we skip stale frames to keep latency bounded. On stream
    error or end-of-stream it reconnects with backoff (``reconnect_on_eof=False``
    disables this — used for finite test inputs so iteration terminates).

    Exposes ``fps``/``width``/``height``/``total_frames`` attributes and a
    ``frames()`` generator yielding ``(seq, bgr_frame)``. ``seq`` counts
    *decoded* frames (including dropped ones), so ``seq / fps`` approximates
    elapsed wall-clock time.
    """

    def __init__(
        self, url: str, *, target_fps: Optional[float] = None,
        reconnect_backoff: float = 3.0, reconnect_on_eof: bool = True,
        open_timeout: float = 15.0,
    ) -> None:
        self._url = url
        self._target_fps = target_fps
        self._backoff = max(0.5, reconnect_backoff)
        self._reconnect_on_eof = reconnect_on_eof
        self._open_timeout = open_timeout

        self._cond = threading.Condition()
        self._latest: Optional[np.ndarray] = None
        self._seq = 0
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

        self.fps = float(target_fps) if target_fps else 15.0
        self.width = 0
        self.height = 0
        self.total_frames = 0  # unknown / unbounded for a live stream

    def start(self) -> "StreamURLSource":
        """Open the stream (in a background thread) and block until the first
        frame arrives, so ``width``/``height``/``fps`` are populated. Raises on
        failure to connect within ``open_timeout``."""
        self._thread = threading.Thread(target=self._run, name="stream-decode", daemon=True)
        self._thread.start()
        if not self._ready.wait(self._open_timeout):
            self._stop.set()
            raise IOError(f"stream did not start within {self._open_timeout:.0f}s: {self._url}")
        if self.width == 0 and self._error:
            self._stop.set()
            raise IOError(self._error)
        logger.info(
            "StreamURLSource open: %s (%dx%d @ %.2f fps).",
            self._url, self.width, self.height, self.fps,
        )
        return self

    def _open_and_decode(self) -> None:
        options = {}
        if self._url.lower().startswith("rtsp"):
            options["rtsp_transport"] = "tcp"    # TCP is more reliable than UDP
            options["stimeout"] = "5000000"      # 5s socket timeout (microseconds)
        container = av.open(self._url, options=options, timeout=self._open_timeout)
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            self.width = int(stream.codec_context.width or 0)
            self.height = int(stream.codec_context.height or 0)
            rate = stream.average_rate or stream.base_rate
            if not self._target_fps and rate:
                self.fps = float(rate)
            for frame in container.decode(stream):
                if self._stop.is_set():
                    return
                img = frame.to_ndarray(format="bgr24")
                if self.width == 0 or self.height == 0:
                    self.height, self.width = img.shape[:2]
                with self._cond:
                    self._latest = img
                    self._seq += 1
                    self._cond.notify_all()
                self._ready.set()
        finally:
            container.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._open_and_decode()
                if not self._reconnect_on_eof:
                    break
                logger.info("Stream ended; reconnecting in %.1fs: %s", self._backoff, self._url)
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                logger.warning("Stream error (%s): %s", self._url, exc)
                self._ready.set()  # unblock start(); it checks width/error
                if not self._reconnect_on_eof:
                    break
            if self._stop.wait(self._backoff):  # interruptible sleep
                break
        with self._cond:                        # wake any waiting consumer
            self._cond.notify_all()

    def frames(self) -> Iterator[Frame]:
        last = 0
        while not self._stop.is_set():
            with self._cond:
                self._cond.wait_for(
                    lambda: self._seq != last or self._stop.is_set(), timeout=1.0
                )
                if self._stop.is_set():
                    break
                new = self._seq != last
                if new:
                    last = self._seq
                    frame = self._latest
            if new and frame is not None:
                yield last, frame
            elif not self._reconnect_on_eof and self._thread and not self._thread.is_alive():
                break  # finite input exhausted and consumer has caught up

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
