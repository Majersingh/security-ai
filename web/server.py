"""FastAPI server for the multi-feed live-stream monitoring UI.

The system processes **live stream URLs** (RTSP / HLS / HTTP). Each stream is a
background :class:`feeds.Feed` that decodes with PyAV (NVDEC when available), gets
its detections from the one shared inference process, and broadcasts annotated
frames + events to any viewers.

Endpoints
---------
* ``GET  /feeds``                  -- snapshot of all active feeds.
* ``POST /feeds/stream {url}``     -- start a feed from a stream URL.
* ``POST /feeds/{id}/stop``        -- stop a feed.
* ``WS   /feeds/{id}/subscribe``   -- watch a feed's annotated frames + events.

Nothing is stored except event snapshots and per-feed ``events.csv``.
Only the transport lives here; all CV logic is reused from ``src/``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Make the CV package importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from contextlib import asynccontextmanager  # noqa: E402

from config import Config  # noqa: E402
from utils import limit_process_threads, resolve_device, setup_logging  # noqa: E402

# Cap this process's intra-op thread pools BEFORE the imports below drag in
# torch/OpenCV — the env vars they read are only consulted at import time.
limit_process_threads(Config().cv_threads, Config().torch_threads)

from feeds import FeedManager  # noqa: E402
from sources import StreamURLSource  # noqa: E402

logger = setup_logging("INFO")

# Ultralytics logs a per-inference "'half' is deprecated" warning that floods the
# console at scale; we intentionally use half=True on CUDA, so quiet it.
logging.getLogger("ultralytics").setLevel(logging.ERROR)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _log_hardware() -> None:
    """Log, at startup, which device inference will run on (GPU vs CPU)."""
    device = resolve_device(Config().device)  # what "auto" resolves to
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info(
                "GPU DETECTED: %s | CUDA %s | inference device=%s | FP16=ON",
                name, torch.version.cuda, device,
            )
        elif device == "mps":
            logger.info("Apple GPU (MPS) detected | inference device=mps")
        else:
            logger.warning(
                "NO GPU detected — inference will run on CPU (slow). "
                "Install a CUDA build of PyTorch and use a GPU host for real-time."
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not query GPU info (%s); resolved device=%s", exc, device)


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    logger.info("Server starting up…")
    _log_hardware()
    cfg = Config()
    # num_workers>0: coordinator mode — N worker processes for the CPU-bound work
    # plus one inference process that owns the GPU. 0: everything here, in-process
    # (simple, no shared memory; the right mode for a few feeds or a CPU-only box).
    if getattr(cfg, "num_workers", 0) and cfg.num_workers > 0:
        from workers import WorkerPool
        app.state.pool = WorkerPool(cfg).start()
        app.state.feeds = None
        logger.info(
            "Coordinator mode: %d worker process(es), cv_threads=%d, torch_threads=%d.",
            cfg.num_workers, cfg.cv_threads, cfg.torch_threads,
        )
    else:
        app.state.pool = None
        app.state.feeds = FeedManager(cfg)
        logger.info("In-process mode (num_workers=0).")
    yield
    if getattr(app.state, "pool", None) is not None:
        app.state.pool.shutdown()
    if getattr(app.state, "feeds", None) is not None:
        app.state.feeds.shutdown()
    logger.info("Server shutting down.")


app = FastAPI(title="CCTV Operator Monitoring - Live Streams", lifespan=lifespan)


@app.get("/feeds")
async def list_feeds() -> dict:
    """Snapshot of all active feeds (for the dashboard)."""
    mgr = app.state.pool or app.state.feeds
    return {"feeds": mgr.list(), "active": mgr.count(), "max_feeds": mgr.max_feeds}


@app.post("/feeds/{feed_id}/stop")
async def stop_feed(feed_id: str) -> dict:
    """Ask a running feed to stop."""
    pool = app.state.pool
    if pool is not None:
        return {"stopped": pool.stop(feed_id), "feed_id": feed_id}
    return {"stopped": app.state.feeds.stop(feed_id), "feed_id": feed_id}


def _parse_geometry(msg: dict):
    """Extract (zone_polygon, line_start, line_end) from a client message.

    Coordinates are *normalized* (0..1 fractions of width/height), so they are
    resolution-independent; the pipeline scales them to pixels. Missing/invalid
    parts return None so the corresponding rule is simply not registered.
    """
    if not isinstance(msg, dict):
        return None, None, None
    zone = msg.get("zone_polygon")
    if isinstance(zone, list) and len(zone) >= 3:
        zone = [(float(p[0]), float(p[1])) for p in zone]
    else:
        zone = None

    def _pt(v):
        return (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) and len(v) == 2 else None

    line_start = _pt(msg.get("line_start"))
    line_end = _pt(msg.get("line_end"))
    if not (line_start and line_end):
        line_start = line_end = None
    return zone, line_start, line_end


@app.post("/feeds/{feed_id}/geometry")
async def set_feed_geometry(feed_id: str, payload: dict) -> JSONResponse:
    """Set/clear a running feed's detection zone/line (normalized coords).

    Sent by the UI after the user draws on the live stream. Takes effect on the
    next frame; annotated frames then show the zone/line.
    """
    zone, line_start, line_end = _parse_geometry(payload)
    pool = app.state.pool
    if pool is not None:
        applied = pool.set_geometry(feed_id, zone, line_start, line_end)
        return JSONResponse({"applied": applied, "feed_id": feed_id})
    feed = app.state.feeds.get(feed_id)
    if feed is None:
        return JSONResponse({"error": "unknown or finished feed"}, status_code=404)
    applied = feed.set_geometry(zone, line_start, line_end)
    return JSONResponse({"applied": applied, "feed_id": feed_id})


@app.post("/feeds/stream")
async def add_stream(payload: dict) -> JSONResponse:
    """Start a background feed from a live stream URL (RTSP / HLS / HTTP).

    In coordinator mode the feed is assigned to a worker process; viewers watch
    via ``WS /feeds/{id}/subscribe``. Nothing is stored except event snapshots +
    ``events.csv``.
    """
    url = (payload.get("url") or "").strip() if isinstance(payload, dict) else ""
    name = (payload.get("name") if isinstance(payload, dict) else None) or url
    if not url:
        return JSONResponse({"error": "missing 'url'"}, status_code=400)
    zone, line_start, line_end = _parse_geometry(payload)

    pool = app.state.pool
    if pool is not None:
        # Optimistic: the worker opens the stream; failures surface on the
        # subscribe socket as an "error" payload.
        feed_id = pool.add_stream(url, name, zone, line_start, line_end)
        if feed_id is None:
            return JSONResponse({"error": f"feed limit reached ({pool.max_feeds})"},
                                status_code=429)
        return JSONResponse({"feed_id": feed_id, "name": name})

    # ---- in-process mode ----
    mgr: FeedManager = app.state.feeds
    loop = asyncio.get_event_loop()
    cfg = Config()
    cfg.write_output_video = False
    max_lag = cfg.stream_max_lag_seconds if cfg.drop_when_behind else 0.0
    try:
        source = await loop.run_in_executor(
            None,
            lambda: StreamURLSource(
                url, max_lag_s=max_lag, hw_decode=cfg.hw_decode,
                stride=cfg.frame_stride,
            ).start(),
        )
    except Exception as exc:  # noqa: BLE001 - bad URL / unreachable stream
        return JSONResponse({"error": f"could not open stream: {exc}"}, status_code=400)
    try:
        feed = mgr.create(source, cfg, zone, line_start, line_end,
                          name=name, kind="stream", emit_image=True)
    except RuntimeError as exc:  # feed limit reached
        source.close()
        return JSONResponse({"error": str(exc)}, status_code=429)
    asyncio.create_task(feed.run())
    return JSONResponse({
        "feed_id": feed.feed_id, "name": name,
        "width": source.width, "height": source.height, "fps": round(source.fps, 2),
    })


@app.websocket("/feeds/{feed_id}/subscribe")
async def subscribe_feed(websocket: WebSocket, feed_id: str) -> None:
    """Attach a viewer to a running feed and relay its result payloads."""
    await websocket.accept()
    pool = websocket.app.state.pool

    if pool is not None:
        info = pool.get_info(feed_id)
        if info is None:
            await websocket.send_json({"type": "error", "message": "unknown or finished feed"})
            await websocket.close()
            return
        queue = pool.subscribe(feed_id)
        try:
            await websocket.send_json({
                "type": "meta", "feed_id": feed_id, "kind": info.get("kind", "stream"),
                "fps": info.get("fps", 0.0), "total_frames": info.get("total_frames", 0),
                "width": info.get("width", 0), "height": info.get("height", 0),
            })
            while True:
                payload = await queue.get()
                await websocket.send_json(payload)
                if payload.get("type") in ("done", "error"):
                    break
        except WebSocketDisconnect:
            logger.info("Viewer left feed %s.", feed_id)
        except Exception:  # noqa: BLE001
            logger.exception("subscribe relay failed for feed %s", feed_id)
        finally:
            pool.unsubscribe(feed_id, queue)
            try:
                await websocket.close()
            except Exception:
                pass
        return

    # ---- in-process mode ----
    mgr: FeedManager = websocket.app.state.feeds
    feed = mgr.get(feed_id)
    if feed is None:
        await websocket.send_json({"type": "error", "message": "unknown or finished feed"})
        await websocket.close()
        return
    queue = feed.subscribe()
    try:
        await websocket.send_json({
            "type": "meta", "feed_id": feed_id, "kind": feed.info.kind,
            "fps": feed.info.fps, "total_frames": feed.info.total_frames,
            "width": feed.info.width, "height": feed.info.height,
        })
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
            if payload.get("type") in ("done", "error"):
                break
    except WebSocketDisconnect:
        logger.info("Viewer left feed %s.", feed_id)
    except Exception:  # noqa: BLE001
        logger.exception("subscribe relay failed for feed %s", feed_id)
    finally:
        feed.unsubscribe(queue)
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/events")
async def events_ws(websocket: WebSocket) -> None:
    """Global events stream: violations from ALL feeds (viewed or not), so the
    dashboard's alert panel never misses anything regardless of which videos are
    being watched."""
    await websocket.accept()
    hub = websocket.app.state.pool or websocket.app.state.feeds
    queue = hub.subscribe_events()
    try:
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.exception("events relay failed")
    finally:
        hub.unsubscribe_events(queue)
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/")
async def index() -> FileResponse:
    """Serve the dashboard with no-store so browsers never run a stale copy
    (we iterate on the UI a lot; caching kept biting)."""
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


# Serve remaining static assets at "/".
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
