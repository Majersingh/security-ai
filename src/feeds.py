"""Concurrent multi-feed orchestration for a single GPU.

Each :class:`Feed` owns its own :class:`FrameProcessor` — and therefore its own
YOLO model instance. This is mandatory, not a choice: Ultralytics keeps tracker
state *on the model object* (``model.track(persist=True)``), so two feeds sharing
one model would corrupt each other's track IDs. VRAM is what caps the feed count
(``Config.max_feeds``).

All feeds share **one** :class:`FeedManager`, whose GPU gate (an
``asyncio.Semaphore``) serializes inference so N feeds don't thrash the single
GPU. Decode happens in parallel (cheap, libav); only the model forward pass is
gated.

A ``Feed`` is transport-agnostic: it drives a source and emits result dicts to an
async ``on_update`` callback. The web layer wires that callback to a WebSocket;
nothing here imports FastAPI.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable, List, Optional, Protocol, Set, Tuple

import cv2

from config import Config
from streaming import FrameProcessor
from utils import format_timestamp

logger = logging.getLogger("operator_monitor")

# on_update(payload) -> awaitable. Returns None; may raise if the client is gone.
UpdateFn = Callable[[dict], Awaitable[None]]

# Annotated stream frames are downscaled to this width before JPEG/base64
# (display quality only; detection still runs at full resolution).
_STREAM_MAX_WIDTH = 960
_JPEG_QUALITY = 70

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
        emit_image: bool = False,
    ) -> None:
        self._manager = manager
        self.feed_id = feed_id
        self._source = source
        self._cfg = cfg
        self._geom = (zone, line_start, line_end)
        self._stride = max(1, cfg.frame_stride)
        # Stream feeds have no local copy of the video in the browser, so we send
        # server-annotated JPEG frames; upload feeds send boxes-only JSON and the
        # browser draws them over its own local <video>.
        self._emit_image = emit_image
        # Cap how often we encode+send a frame to viewers (display rate), so the
        # browser stream stays light regardless of how fast detection runs.
        self._viewer_interval = 1.0 / max(1.0, float(getattr(cfg, "viewer_max_fps", 12.0)))
        self._proc: Optional[FrameProcessor] = None
        self._stop = asyncio.Event()
        self._on_update: Optional[UpdateFn] = None
        self._subscribers: Set[asyncio.Queue] = set()
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

    def subscribe(self) -> asyncio.Queue:
        """Register a viewer; returns a **latest-only** queue (maxsize=1).

        A slow viewer (e.g. over a bandwidth-limited tunnel) must never make the
        browser play an ever-growing backlog — it should always jump to the most
        recent frame. `_emit` drops the stale frame when a newer one arrives.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def _emit(self, payload: dict) -> bool:
        """Fan a payload out to all subscribers (dropping the oldest if a viewer
        lags) and to the direct driver. Returns False if the driver is gone."""
        for q in list(self._subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                try:  # viewer is slow -> drop its oldest to stay near-live
                    q.get_nowait()
                    q.put_nowait(payload)
                except Exception:  # noqa: BLE001
                    pass
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
            # Model load is heavy -> build the processor off the event loop.
            self._proc = await loop.run_in_executor(
                None,
                lambda: FrameProcessor(
                    self._cfg, self._source.fps,
                    processing_fps=self._source.fps / self._stride,
                    zone_polygon=zone, line_start=line_start, line_end=line_end,
                    batched=self._manager.batched,
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
            last_emit_t = 0.0           # last time we sent an annotated frame to viewers
            while not self._stop.is_set():
                item = await loop.run_in_executor(None, _next, gen)
                decode_accum += getattr(self._source, "last_decode_ms", 0.0)
                pace_accum += getattr(self._source, "last_wait_ms", 0.0)
                pulled += 1
                if item is _SENTINEL:
                    break
                raw_idx, frame = item
                if raw_idx % self._stride:      # honour stride (raw index kept)
                    continue

                # Detection runs on EVERY processed frame (for events/accuracy),
                # but we only encode+send an annotated frame to viewers at
                # viewer_max_fps — the display stream is decoupled from detection.
                t_now = time.monotonic()
                want_image = self._emit_image and (t_now - last_emit_t >= self._viewer_interval)

                t_inf = time.monotonic()
                if self._manager.batched:
                    payload = await self._manager.infer_batched(
                        self._proc, frame, raw_idx, annotated=want_image
                    )
                else:
                    payload = await self._manager.infer(
                        self._proc, frame, raw_idx, annotated=want_image
                    )
                infer_ms = (time.monotonic() - t_inf) * 1000.0
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

                # Send when: it's a display frame (has image), OR it carries events
                # (so violations always reach the panel), OR it's a boxes-only feed.
                t_em = time.monotonic()
                alive = True
                if want_image or events or not self._emit_image:
                    alive = await self._emit(payload)
                    if want_image:
                        last_emit_t = t_now
                emit_ms = (time.monotonic() - t_em) * 1000.0

                now = time.monotonic()
                gap_ms = (now - last_proc_t) * 1000.0
                eff_fps = 1000.0 / gap_ms if gap_ms > 0 else 0.0
                if getattr(self._cfg, "log_timing", False):
                    logger.info(
                        "TIMING %s f=%d | %d pulled | decode=%.0fms pace=%.0fms "
                        "infer=%.0fms emit=%.0fms | gap=%.0fms (%.1f proc-fps)",
                        self.feed_id[:8], raw_idx, pulled, decode_accum, pace_accum,
                        infer_ms, emit_ms, gap_ms, eff_fps,
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
            self._manager.remove(self.feed_id)


class FeedManager:
    """Registry of active feeds + the shared GPU gate for the whole process."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        cfg = cfg or Config()
        self.max_feeds = int(getattr(cfg, "max_feeds", 8))
        depth = max(1, int(getattr(cfg, "max_concurrent_inferences", 2)))
        self._gate = asyncio.Semaphore(depth)
        # A small pool for the post-detection work (track/rules/annotate/encode).
        # Sized a bit above the gate depth so batched feeds aren't thread-starved.
        self._infer_pool = ThreadPoolExecutor(
            max_workers=max(depth, 8), thread_name_prefix="infer"
        )
        self._feeds: dict[str, Feed] = {}

        # Batched inference: ONE shared model for all feeds (see batch.py).
        self.batched = bool(getattr(cfg, "batched_inference", False))
        self._batcher = None
        if self.batched:
            from batch import BatchInferencer  # local import: optional dependency path
            self._batcher = BatchInferencer(cfg).start()
        logger.info(
            "FeedManager ready: max_feeds=%d, mode=%s%s.",
            self.max_feeds,
            "batched" if self.batched else "per-feed-model",
            "" if self.batched else f", gpu_gate_depth={depth}",
        )

    def create(
        self, source: FrameSource, cfg: Config, zone, line_start, line_end,
        name: str, kind: str = "upload", emit_image: bool = False,
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
                    name, kind, emit_image=emit_image)
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

    async def infer(
        self, proc: FrameProcessor, frame, frame_index: int, annotated: bool = False,
    ) -> dict:
        """Run one frame through the model (serialized by the shared GPU gate)
        and return a partial result payload.

        ``annotated=False`` -> ``{boxes, events, w, h}`` (upload feeds; the browser
        draws the boxes). ``annotated=True`` -> ``{image, events, w, h}`` with a
        base64 JPEG of the annotated frame (stream feeds; no local video exists).
        """
        loop = asyncio.get_event_loop()
        fn = self._infer_annotated if annotated else self._infer_boxes
        async with self._gate:
            return await loop.run_in_executor(self._infer_pool, fn, proc, frame, frame_index)

    async def infer_batched(
        self, proc: FrameProcessor, frame, frame_index: int, annotated: bool = False,
    ) -> dict:
        """Batched path: the shared model detects (combined with other feeds in
        one GPU call); the per-feed track/rules/annotate runs in a worker thread."""
        detections = await self._batcher.infer(frame)     # awaits the batch (GPU)
        loop = asyncio.get_event_loop()
        fn = self._infer_annotated if annotated else self._infer_boxes
        return await loop.run_in_executor(self._infer_pool, fn, proc, frame, frame_index, detections)

    @staticmethod
    def _infer_boxes(proc: FrameProcessor, frame, frame_index: int, detections=None) -> dict:
        boxes, events, w, h = proc.process_json(frame, frame_index, detections)
        return {"w": w, "h": h, "boxes": boxes, "events": [asdict(e) for e in events]}

    @staticmethod
    def _infer_annotated(proc: FrameProcessor, frame, frame_index: int, detections=None) -> dict:
        annotated, events = proc.process(frame, frame_index, detections)
        h, w = annotated.shape[:2]
        cfg = proc._config
        max_w = int(getattr(cfg, "viewer_max_width", _STREAM_MAX_WIDTH))
        quality = int(getattr(cfg, "viewer_jpeg_quality", _JPEG_QUALITY))
        if w > max_w:  # shrink the wire payload; detection ran at full res
            annotated = cv2.resize(annotated, (max_w, int(h * (max_w / w))))
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
        image = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
        return {"w": w, "h": h, "image": image, "events": [asdict(e) for e in events]}
