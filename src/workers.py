"""Multiprocess worker pool — true parallelism across the GIL.

The web process becomes a thin **coordinator**: it assigns each feed to a worker
**process** and relays that worker's result frames/events to browsers. Each
worker runs the full per-feed pipeline (decode → detect → track → annotate →
encode) in its own process (own GIL) and shares the one GPU. This is what lets
many feeds run in parallel when the bottleneck is CPU-bound per-feed work.

Enabled when ``Config.num_workers > 0``. IPC is plain ``multiprocessing`` queues
(spawn context, required for CUDA): one control queue per worker (coordinator →
worker) and one shared result queue (workers → coordinator).
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import queue as pyqueue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger("operator_monitor")


# ----------------------------- worker process -----------------------------

def _worker_process(worker_id: int, cfg, ctrl_q: "mp.Queue", result_q: "mp.Queue") -> None:
    """Entry point for a worker process (spawned)."""
    from utils import setup_logging
    setup_logging(getattr(cfg, "log_level", "INFO"))
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    log = logging.getLogger("operator_monitor")
    log.info("Worker %d up (pid=%d).", worker_id, os.getpid())
    try:
        asyncio.run(_worker_loop(worker_id, cfg, ctrl_q, result_q))
    except Exception:  # noqa: BLE001
        log.exception("Worker %d crashed", worker_id)


async def _worker_loop(worker_id: int, cfg, ctrl_q: "mp.Queue", result_q: "mp.Queue") -> None:
    from feeds import FeedManager
    from sources import StreamURLSource

    loop = asyncio.get_event_loop()
    mgr = FeedManager(cfg)
    feeds: Dict[str, object] = {}
    max_lag = cfg.stream_max_lag_seconds if getattr(cfg, "drop_when_behind", False) else 0.0

    def make_on_update(feed_id: str):
        async def on_update(payload: dict) -> None:
            try:
                result_q.put_nowait((feed_id, payload))     # drop if coordinator is behind
            except pyqueue.Full:
                pass
        return on_update

    async def run_feed(feed, feed_id: str) -> None:
        try:
            await feed.run(make_on_update(feed_id))
        finally:
            feeds.pop(feed_id, None)

    async def handle_add(msg: dict) -> None:
        feed_id = msg["feed_id"]
        try:
            source = await loop.run_in_executor(
                None, lambda: StreamURLSource(msg["url"], max_lag_s=max_lag).start()
            )
        except Exception as exc:  # noqa: BLE001
            result_q.put((feed_id, {"type": "error", "feed_id": feed_id,
                                    "message": f"could not open stream: {exc}"}))
            return
        import copy
        feed_cfg = copy.copy(cfg)
        feed_cfg.write_output_video = False
        try:
            feed = mgr.create(source, feed_cfg, msg.get("zone"), msg.get("line_start"),
                              msg.get("line_end"), name=msg.get("name", msg["url"]),
                              kind="stream", emit_image=True, feed_id=feed_id)
        except RuntimeError as exc:
            source.close()
            result_q.put((feed_id, {"type": "error", "feed_id": feed_id, "message": str(exc)}))
            return
        feeds[feed_id] = feed
        asyncio.create_task(run_feed(feed, feed_id))

    while True:
        msg = await loop.run_in_executor(None, ctrl_q.get)   # block for next command
        cmd = msg.get("cmd")
        if cmd == "add":
            await handle_add(msg)
        elif cmd == "stop":
            f = feeds.get(msg["feed_id"])
            if f:
                f.stop()
        elif cmd == "geometry":
            f = feeds.get(msg["feed_id"])
            if f:
                f.set_geometry(msg.get("zone"), msg.get("line_start"), msg.get("line_end"))
        elif cmd == "view":
            f = feeds.get(msg["feed_id"])
            if f:
                f.set_viewing(msg.get("on", False))
        elif cmd == "shutdown":
            break


# ----------------------------- coordinator side -----------------------------

@dataclass
class _FeedRec:
    feed_id: str
    name: str
    worker_id: int
    kind: str = "stream"
    status: str = "pending"
    fps: float = 0.0
    width: int = 0
    height: int = 0
    total_frames: int = 0
    frame_index: int = 0
    progress: float = 0.0
    event_count: int = 0
    error: str = ""
    subscribers: Set[asyncio.Queue] = field(default_factory=set)


class WorkerPool:
    """Coordinator: owns worker processes, assigns feeds, relays results."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self.max_feeds = int(getattr(cfg, "max_feeds", 8))
        self._n = max(1, int(cfg.num_workers))
        self._ctx = mp.get_context("spawn")          # required for CUDA in children
        self._result_q: "mp.Queue" = self._ctx.Queue(maxsize=512)
        self._ctrl_qs: List["mp.Queue"] = []
        self._procs: List["mp.Process"] = []
        self._counts: List[int] = [0] * self._n      # feeds per worker (for balancing)
        self._feeds: Dict[str, _FeedRec] = {}
        self._event_subs: Set[asyncio.Queue] = set()   # global events (all feeds)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._drain_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> "WorkerPool":
        self._loop = asyncio.get_event_loop()
        for wid in range(self._n):
            ctrl_q = self._ctx.Queue()
            p = self._ctx.Process(
                target=_worker_process, args=(wid, self._cfg, ctrl_q, self._result_q),
                name=f"cctv-worker-{wid}", daemon=True,
            )
            p.start()
            self._ctrl_qs.append(ctrl_q)
            self._procs.append(p)
        self._drain_thread = threading.Thread(target=self._drain, name="result-drain", daemon=True)
        self._drain_thread.start()
        logger.info("WorkerPool started: %d worker process(es).", self._n)
        return self

    def shutdown(self) -> None:
        self._stop.set()
        for q in self._ctrl_qs:
            try:
                q.put({"cmd": "shutdown"})
            except Exception:  # noqa: BLE001
                pass
        for p in self._procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()

    # ---- feed lifecycle ----
    def add_stream(self, url: str, name: str, zone, line_start, line_end) -> Optional[str]:
        if len(self._feeds) >= self.max_feeds:
            return None
        wid = min(range(self._n), key=lambda i: self._counts[i])   # least-loaded worker
        feed_id = uuid.uuid4().hex
        self._counts[wid] += 1
        self._feeds[feed_id] = _FeedRec(feed_id=feed_id, name=name, worker_id=wid)
        self._ctrl_qs[wid].put({
            "cmd": "add", "feed_id": feed_id, "url": url, "name": name,
            "zone": zone, "line_start": line_start, "line_end": line_end,
        })
        logger.info("Feed %s -> worker %d (%d active).", feed_id, wid, len(self._feeds))
        return feed_id

    def stop(self, feed_id: str) -> bool:
        rec = self._feeds.get(feed_id)
        if rec is None:
            return False
        self._ctrl_qs[rec.worker_id].put({"cmd": "stop", "feed_id": feed_id})
        return True

    def set_geometry(self, feed_id: str, zone, line_start, line_end) -> bool:
        rec = self._feeds.get(feed_id)
        if rec is None:
            return False
        self._ctrl_qs[rec.worker_id].put({
            "cmd": "geometry", "feed_id": feed_id,
            "zone": zone, "line_start": line_start, "line_end": line_end,
        })
        return True

    # ---- viewer subscription (video) ----
    def subscribe(self, feed_id: str) -> Optional[asyncio.Queue]:
        rec = self._feeds.get(feed_id)
        if rec is None:
            return None
        was_empty = not rec.subscribers
        q: asyncio.Queue = asyncio.Queue(maxsize=1)     # latest-only
        rec.subscribers.add(q)
        if was_empty:                                   # first viewer -> start video
            self._ctrl_qs[rec.worker_id].put({"cmd": "view", "feed_id": feed_id, "on": True})
        return q

    def unsubscribe(self, feed_id: str, q: asyncio.Queue) -> None:
        rec = self._feeds.get(feed_id)
        if rec is not None:
            rec.subscribers.discard(q)
            if not rec.subscribers:                     # last viewer left -> stop video
                self._ctrl_qs[rec.worker_id].put({"cmd": "view", "feed_id": feed_id, "on": False})

    # ---- global events channel ----
    def subscribe_events(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._event_subs.add(q)
        return q

    def unsubscribe_events(self, q: asyncio.Queue) -> None:
        self._event_subs.discard(q)

    def get_info(self, feed_id: str) -> Optional[dict]:
        rec = self._feeds.get(feed_id)
        return _rec_info(rec) if rec else None

    def count(self) -> int:
        return len(self._feeds)

    def list(self) -> List[dict]:
        return [_rec_info(r) for r in self._feeds.values()]

    # ---- result relay (runs in a background thread) ----
    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                feed_id, payload = self._result_q.get(timeout=0.5)
            except pyqueue.Empty:
                continue
            except Exception:  # noqa: BLE001
                continue
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._dispatch, feed_id, payload)

    def _dispatch(self, feed_id: str, payload: dict) -> None:
        """On the event loop: update the record + fan the payload to viewers."""
        rec = self._feeds.get(feed_id)
        if rec is None:
            return
        ptype = payload.get("type")
        if ptype == "meta":
            rec.status = "running"
            rec.fps = payload.get("fps", rec.fps)
            rec.width = payload.get("width", rec.width)
            rec.height = payload.get("height", rec.height)
            rec.total_frames = payload.get("total_frames", rec.total_frames)
        elif ptype == "frame":
            rec.frame_index = payload.get("i", rec.frame_index)
            rec.progress = payload.get("progress", rec.progress) or rec.progress
            rec.event_count += len(payload.get("events", []))
        elif ptype in ("done", "error"):
            rec.status = "error" if ptype == "error" else "done"
            rec.error = payload.get("message", "")

        # Global events channel: forward any violations to all-feeds subscribers.
        evs = payload.get("events")
        if evs:
            emsg = {"type": "events", "feed_id": feed_id, "name": rec.name, "events": evs}
            for q in list(self._event_subs):
                try:
                    q.put_nowait(emsg)
                except asyncio.QueueFull:
                    pass

        # Video viewers: image frames, plus terminal messages so their socket
        # closes. (Event-only frames carry no image and aren't video-relevant.)
        if payload.get("image") or ptype in ("done", "error"):
            for q in list(rec.subscribers):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    try:
                        q.get_nowait()
                        q.put_nowait(payload)
                    except Exception:  # noqa: BLE001
                        pass

        if ptype in ("done", "error"):
            self._counts[rec.worker_id] = max(0, self._counts[rec.worker_id] - 1)
            self._feeds.pop(feed_id, None)


def _rec_info(rec: _FeedRec) -> dict:
    # Build manually — asdict() would deep-copy `subscribers` (asyncio.Queue
    # objects hold a _contextvars.Context that can't be copied).
    return {
        "feed_id": rec.feed_id, "name": rec.name, "kind": rec.kind,
        "status": rec.status, "fps": rec.fps, "width": rec.width,
        "height": rec.height, "total_frames": rec.total_frames,
        "frame_index": rec.frame_index, "progress": rec.progress,
        "event_count": rec.event_count, "worker_id": rec.worker_id, "error": rec.error,
    }
