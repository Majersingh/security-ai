"""FastAPI server for the multi-feed live-stream monitoring UI.

The system processes **live stream URLs** (RTSP / HLS / HTTP). Each stream is a
background :class:`feeds.Feed` that decodes with PyAV, runs detection through a
shared GPU gate, and broadcasts annotated frames + events to any viewers.

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
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from contextlib import asynccontextmanager  # noqa: E402

from config import Config  # noqa: E402
from feeds import FeedManager  # noqa: E402
from sources import StreamURLSource  # noqa: E402
from utils import resolve_device, setup_logging  # noqa: E402

logger = setup_logging("INFO")

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
    # One manager for the whole process: registry + shared batched model (or GPU
    # gate). Built here so the async bits bind to the running event loop.
    # Detection resolution etc. come from Config (see inference_imgsz there).
    app.state.feeds = FeedManager(Config())
    yield
    logger.info("Server shutting down.")


app = FastAPI(title="CCTV Operator Monitoring - Live Streams", lifespan=lifespan)


@app.get("/feeds")
async def list_feeds() -> dict:
    """Snapshot of all active feeds (for the dashboard)."""
    mgr: FeedManager = app.state.feeds
    return {"feeds": mgr.list(), "active": mgr.count(), "max_feeds": mgr.max_feeds}


@app.post("/feeds/{feed_id}/stop")
async def stop_feed(feed_id: str) -> dict:
    """Ask a running feed to stop."""
    mgr: FeedManager = app.state.feeds
    return {"stopped": mgr.stop(feed_id), "feed_id": feed_id}


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
    mgr: FeedManager = app.state.feeds
    feed = mgr.get(feed_id)
    if feed is None:
        return JSONResponse({"error": "unknown or finished feed"}, status_code=404)
    zone, line_start, line_end = _parse_geometry(payload)
    applied = feed.set_geometry(zone, line_start, line_end)
    return JSONResponse({"applied": applied, "feed_id": feed_id})


@app.post("/feeds/stream")
async def add_stream(payload: dict) -> JSONResponse:
    """Start a background feed from a live stream URL (RTSP / HLS / HTTP).

    The feed runs on the server with no client attached; viewers watch via
    ``WS /feeds/{id}/subscribe``. Nothing is stored except event snapshots +
    ``events.csv``.
    """
    mgr: FeedManager = app.state.feeds
    url = (payload.get("url") or "").strip() if isinstance(payload, dict) else ""
    name = (payload.get("name") if isinstance(payload, dict) else None) or url
    if not url:
        return JSONResponse({"error": "missing 'url'"}, status_code=400)

    loop = asyncio.get_event_loop()
    cfg = Config()
    cfg.write_output_video = False
    max_lag = cfg.stream_max_lag_seconds if cfg.drop_when_behind else 0.0
    try:
        source = await loop.run_in_executor(
            None, lambda: StreamURLSource(url, max_lag_s=max_lag).start()
        )
    except Exception as exc:  # noqa: BLE001 - bad URL / unreachable stream
        return JSONResponse({"error": f"could not open stream: {exc}"}, status_code=400)

    try:
        feed = mgr.create(source, cfg, None, None, None, name=name, kind="stream", emit_image=True)
    except RuntimeError as exc:  # feed limit reached
        source.close()
        return JSONResponse({"error": str(exc)}, status_code=429)

    asyncio.create_task(feed.run())  # background; viewers attach via subscribe
    return JSONResponse({
        "feed_id": feed.feed_id, "name": name,
        "width": source.width, "height": source.height, "fps": round(source.fps, 2),
    })


@app.websocket("/feeds/{feed_id}/subscribe")
async def subscribe_feed(websocket: WebSocket, feed_id: str) -> None:
    """Attach a viewer to a running feed and relay its result payloads."""
    await websocket.accept()
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


# Serve the single-page UI at "/".
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
