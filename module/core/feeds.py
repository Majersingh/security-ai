"""Concurrent multi-feed orchestration — detection only, no video.

This module does **not** serve video. Playback lives in a separate process
(``module/rawapp.py``) that decodes independently at the source frame rate, because
display fps here was necessarily capped by *detection* fps. Removing it takes the
annotate + JPEG-encode + base64 + viewer-throttle work out of the hot loop
entirely, along with per-feed viewer queues and the view on/off control path.

What remains for the UI is a single still frame for drawing zone/line geometry
(``/feeds/probe``), which needs no running feed at all.


Each :class:`Feed` owns its own :class:`FrameProcessor` — its tracker, rules,
event log and annotator — but **not** a model. Detection is centralized: one
model in one process serves every feed (see :mod:`inference`), and each feed
keeps its identities separate with its own ``supervision.ByteTrack``. So what
caps the feed count is CPU throughput for decode/track/annotate/encode, not VRAM.

All feeds in a process share **one** :class:`FeedManager`, which holds the
inference handle and a small thread pool for the post-detection CPU work. The
handle is either an in-process batcher or a client for the inference process;
:class:`FeedManager` doesn't care which.

A ``Feed`` is transport-agnostic: it drives a source and emits result dicts to an
async ``on_update`` callback. The web layer wires that callback to a WebSocket;
nothing here imports FastAPI.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable, List, Optional, Protocol, Set, Tuple

from config import Config
from streaming import FrameProcessor
from utils import format_timestamp

logger = logging.getLogger("operator_monitor")

# on_update(payload) -> awaitable. Returns None; may raise if the client is gone.
UpdateFn = Callable[[dict], Awaitable[None]]

_SENTINEL = object()


def _next(iterator):
    """Blocking ``next`` wrapper for run_in_executor (returns sentinel at end)."""
    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL


class FrameSource(Protocol):
    """What a feed needs from any source (in-memory upload, stream URL, …)."""

    fps: float
    width: int
    height: int
    total_frames: int

    def frames(self): ...      # -> Iterator[Tuple[int, np.ndarray]]
    def close(self) -> None: ...


@dataclass
class FeedInfo:
    """Serializable snapshot of a feed's state (for GET /feeds / the dashboard)."""

    feed_id: str
    name: str
    kind: str = "upload"          # upload | stream
    status: str = "pending"       # pending | running | done | stopped | error
    fps: float = 0.0
    width: int = 0
    height: int = 0
    total_frames: int = 0
    frame_index: int = 0
    progress: float = 0.0
    event_count: int = 0
    error: str = ""
    started_at: float = field(default_factory=time.time)


class Feed:
    """One running pipeline: a source + its own FrameProcessor (+ model)."""

    def __init__(
        self, manager: "FeedManager", feed_id: str, source: FrameSource,
        cfg: Config, zone, line_start, line_end, name: str, kind: str,
    ) -> None:
        self._manager = manager
        self.feed_id = feed_id
        self._source = source
        self._cfg = cfg
        self._geom = (zone, line_start, line_end)
        self._stride = max(1, cfg.frame_stride)
        # The source pull gets its OWN thread, because `frames()` does the
        # real-time pacing `sleep` inside it — on the shared default executor that
        # sleeping thread competes with latency-critical work (the shared-memory
        # submit), and with many feeds it can starve the pool outright. These
        # threads are asleep, not burning CPU, so one per feed is cheap.
        self._pull_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"pull-{feed_id[:8]}"
        )
        self._proc: Optional[FrameProcessor] = None
        self._stop = asyncio.Event()
        self._on_update: Optional[UpdateFn] = None
        self.info = FeedInfo(
            feed_id=feed_id, name=name, kind=kind,
            fps=round(source.fps, 2), width=source.width,
            height=source.height, total_frames=source.total_frames,
        )

    def stop(self) -> None:
        """Request the feed to stop after the current frame."""
        self._stop.set()

    def set_geometry(self, zone, line_start, line_end) -> bool:
        """Update the detection zone/line on this feed at runtime (normalized
        coords). Returns False if the pipeline is not built yet."""
        self._geom = (zone, line_start, line_end)   # in case the engine rebuilds
        if self._proc is None:
            return False
        self._proc.set_geometry(zone, line_start, line_end)
        return True

    async def _emit(self, payload: dict) -> bool:
        """Hand a payload to the driver (the coordinator relay, or an upload client).

        Returns False if the driver is gone, which stops the feed.
        """
        if self._on_update is not None:
            try:
                await self._on_update(payload)
            except Exception:  # noqa: BLE001 - driver (client socket) went away
                return False
        return True

    async def run(self, on_update: Optional[UpdateFn] = None) -> None:
        """Drive the source to completion, broadcasting result dicts.

        ``on_update`` is the direct driver (an upload client's socket); if it
        raises, the feed stops. Stream feeds pass no driver and run in the
        background, with viewers attached via :meth:`subscribe`.
        """
        loop = asyncio.get_event_loop()
        self._on_update = on_update
        zone, line_start, line_end = self._geom
        try:
            # Building rules/tracker/annotator touches disk -> off the event loop.
            self._proc = await loop.run_in_executor(
                None,
                lambda: FrameProcessor(
                    self._cfg, self._source.fps,
                    processing_fps=self._source.fps / self._stride,
                    zone_polygon=zone, line_start=line_start, line_end=line_end,
                ),
            )
            self.info.status = "running"
            await self._emit({
                "type": "meta", "feed_id": self.feed_id, "kind": self.info.kind,
                "fps": self.info.fps, "total_frames": self.info.total_frames,
                "width": self.info.width, "height": self.info.height,
            })

            gen = self._source.frames()
            decode_accum = 0.0          # real decode/network ms since last processed frame
            pace_accum = 0.0            # pacing-sleep ms since last processed frame
            pulled = 0                  # frames pulled (incl. skipped) since last processed
            last_proc_t = time.monotonic()
            last_emit_t = 0.0           # last progress/event payload sent
            while not self._stop.is_set():
                item = await loop.run_in_executor(self._pull_pool, _next, gen)
                decode_accum += getattr(self._source, "last_decode_ms", 0.0)
                pace_accum += getattr(self._source, "last_wait_ms", 0.0)
                # >1 when the source strided/dropped frames internally.
                pulled += int(getattr(self._source, "last_pulled", 1) or 1)
                if item is _SENTINEL:
                    break
                raw_idx, frame = item
                if raw_idx % self._stride:      # honour stride (raw index kept)
                    continue

                payload, (detect_ms, post_ms) = await self._manager.infer(
                    self._proc, frame, raw_idx
                )
                events = payload.get("events") or []
                payload.update({
                    "type": "frame", "feed_id": self.feed_id, "i": raw_idx,
                    "total": self.info.total_frames,
                    "timestamp": format_timestamp(raw_idx, self.info.fps),
                })
                self.info.frame_index = raw_idx
                self.info.progress = (
                    round(100.0 * (raw_idx + 1) / self.info.total_frames, 1)
                    if self.info.total_frames else 0.0
                )
                self.info.event_count = len(self._proc.event_log)
                payload["progress"] = self.info.progress

                # Events always reach the global events channel (all feeds, even
                # unviewed ones) so the dashboard alerts never miss anything.
                if events:
                    self._manager.publish_events(self.feed_id, self.info.name, events)

                # Emit on events, plus a slow heartbeat so the coordinator's
                # frame_index/progress stay live without a payload per frame (at 80
                # feeds that would be thousands of queue messages a second for
                # numbers a dashboard reads once a second anyway).
                t_now = time.monotonic()
                t_em = t_now
                alive = True
                if events or (t_now - last_emit_t) >= 1.0:
                    alive = await self._emit(payload)
                    last_emit_t = t_now
                emit_ms = (time.monotonic() - t_em) * 1000.0

                now = time.monotonic()
                gap_ms = (now - last_proc_t) * 1000.0
                eff_fps = 1000.0 / gap_ms if gap_ms > 0 else 0.0
                if getattr(self._cfg, "log_timing", False):
                    logger.info(
                        "TIMING %s f=%d | %d pulled | decode=%.0fms pace=%.0fms "
                        "detect=%.0fms post=%.0fms emit=%.0fms | gap=%.0fms (%.1f proc-fps)",
                        self.feed_id[:8], raw_idx, pulled, decode_accum, pace_accum,
                        detect_ms, post_ms, emit_ms, gap_ms, eff_fps,
                    )
                decode_accum = pace_accum = 0.0
                pulled = 0
                last_proc_t = now

                if not alive:                    # driver (upload client) gone
                    self._stop.set()
                    break

            await loop.run_in_executor(None, self._proc.finalize)  # flush events.csv
            self.info.status = "stopped" if self._stop.is_set() else "done"
            await self._emit({
                "type": "done", "feed_id": self.feed_id,
                "total_events": len(self._proc.event_dicts),
                "events": self._proc.event_dicts,
            })
        except Exception as exc:  # noqa: BLE001 - surface to caller, keep others alive
            self.info.status = "error"
            self.info.error = str(exc)
            logger.exception("Feed %s failed", self.feed_id)
            await self._emit({
                "type": "error", "feed_id": self.feed_id, "message": str(exc),
            })
        finally:
            self._source.close()
            self._pull_pool.shutdown(wait=False)
            self._manager.remove(self.feed_id)


class FeedManager:
    """Registry of active feeds + this process's handle on inference."""

    def __init__(self, cfg: Optional[Config] = None, inferencer=None) -> None:
        cfg = cfg or Config()
        self._cfg = cfg
        self.max_feeds = int(getattr(cfg, "max_feeds", 8))
        # Post-detection work (track/rules/annotate/JPEG-encode) is CPU-bound but
        # releases the GIL in OpenCV/NumPy, so a small thread pool is the right
        # shape. Kept modest on purpose: real parallelism comes from having
        # several worker processes, and each process is thread-capped.
        self._post_pool = ThreadPoolExecutor(
            max_workers=max(2, int(getattr(cfg, "cv_threads", 1)) * 4),
            thread_name_prefix="post",
        )
        self._feeds: dict[str, Feed] = {}
        self._event_subs: Set[asyncio.Queue] = set()   # global events (all feeds)

        # Inference handle: an InferenceClient (GPU lives in another process) or a
        # LocalInferencer (num_workers == 0). Both expose `await infer(frame)`.
        if inferencer is None:
            from inference import LocalInferencer

            inferencer = LocalInferencer(cfg).start()
        self._inferencer = inferencer
        logger.info("FeedManager ready: max_feeds=%d, inference=%s.",
                    self.max_feeds, type(inferencer).__name__)

    def create(
        self, source: FrameSource, cfg: Config, zone, line_start, line_end,
        name: str, kind: str = "upload",
        feed_id: Optional[str] = None,
    ) -> Feed:
        """Register a new feed. Raises RuntimeError if the feed limit is hit.

        ``feed_id`` lets a coordinator assign the id (used by the worker pool);
        otherwise one is generated.
        """
        if len(self._feeds) >= self.max_feeds:
            raise RuntimeError(f"feed limit reached ({self.max_feeds} concurrent)")
        feed_id = feed_id or uuid.uuid4().hex
        # Per-feed artefact dirs so concurrent feeds never clobber events.csv /
        # each other's snapshots. Base dir = wherever config points (output/).
        base = cfg.events_csv.parent / feed_id
        cfg.events_csv = base / "events.csv"
        cfg.snapshots_dir = base / "snapshots"
        feed = Feed(self, feed_id, source, cfg, zone, line_start, line_end,
                    name, kind)
        self._feeds[feed_id] = feed
        logger.info("Feed %s created (%s '%s'); %d active.", feed_id, kind, name, len(self._feeds))
        return feed

    def remove(self, feed_id: str) -> None:
        if self._feeds.pop(feed_id, None) is not None:
            logger.info("Feed %s removed; %d active.", feed_id, len(self._feeds))

    def get(self, feed_id: str) -> Optional[Feed]:
        return self._feeds.get(feed_id)

    def stop(self, feed_id: str) -> bool:
        feed = self._feeds.get(feed_id)
        if feed is None:
            return False
        feed.stop()
        return True

    def count(self) -> int:
        return len(self._feeds)

    def list(self) -> List[dict]:
        return [asdict(f.info) for f in self._feeds.values()]

    # ---- global events channel (all feeds, independent of video viewing) ----
    def subscribe_events(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._event_subs.add(q)
        return q

    def unsubscribe_events(self, q: asyncio.Queue) -> None:
        self._event_subs.discard(q)

    def publish_events(self, feed_id: str, name: str, events: list) -> None:
        """Fan violation events to global-events viewers (dashboard alerts)."""
        if not self._event_subs or not events:
            return
        msg = {"type": "events", "feed_id": feed_id, "name": name, "events": events}
        for q in list(self._event_subs):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    async def infer(
        self, proc: FrameProcessor, frame, frame_index: int,
    ) -> Tuple[dict, Tuple[float, float]]:
        """Process one frame; return ``(payload, (detect_ms, post_ms))``.

        Two stages, timed separately because they have different cures.
        ``detect_ms`` is the round-trip to the shared model (IPC + queue wait +
        preprocess + GPU) — cured by imgsz, batch fill, TensorRT. ``post_ms`` is
        this feed's own track/rules/snapshot work.

        The payload is always plain data (``{boxes, events, w, h}``). No image is
        produced here: video is a separate service, so nothing in this loop
        annotates or JPEG-encodes.
        """
        t0 = time.monotonic()
        detections = await self._inferencer.infer(frame)      # awaits the batch
        t1 = time.monotonic()
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            self._post_pool, self._infer_boxes, proc, frame, frame_index, detections
        )
        t2 = time.monotonic()
        return payload, ((t1 - t0) * 1000.0, (t2 - t1) * 1000.0)

    def shutdown(self) -> None:
        """Release the inference handle and the post-processing pool."""
        shut = getattr(self._inferencer, "shutdown", None)
        if callable(shut):
            shut()
        self._post_pool.shutdown(wait=False)

    @staticmethod
    def _infer_boxes(proc: FrameProcessor, frame, frame_index: int, detections=None) -> dict:
        boxes, events, w, h = proc.process_json(frame, frame_index, detections)
        return {"w": w, "h": h, "boxes": boxes, "events": [asdict(e) for e in events]}

