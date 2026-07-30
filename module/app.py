"""FastAPI server for the multi-feed live-stream monitoring UI.

The system processes **live stream URLs** (RTSP / HLS / HTTP). Each stream is a
background :class:`feeds.Feed` that decodes with PyAV (NVDEC when available), gets
its detections from the one shared inference process, and reports violations.

Endpoints
---------
* ``GET  /feeds``                  -- snapshot of all active feeds.
* ``POST /feeds/stream {url}``     -- start a feed from a stream URL.
* ``POST /feeds/{id}/stop``        -- stop a feed.
* ``POST /feeds/probe``            -- one still frame, for drawing zone/line.
* ``WS   /events``                 -- violations from every feed on this module.

This service serves NO video. Playback is a separate process (module/rawapp.py)
that decodes at the source frame rate; display here was capped by detection fps.

Nothing is stored except event snapshots and per-feed ``events.csv``.
Only the transport lives here; all CV logic is reused from ``src/``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Make the CV core importable (flat imports: `from config import Config`), and this
# directory too, so sibling modules like `reporting` resolve when the app is loaded
# as `module.app:app` — in that case only the repo root is on sys.path, not here.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "core"))   # shared: sources, procutil
sys.path.insert(0, str(ROOT / "core"))          # detection-only CV modules
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

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


async def _forward_events_to_central(app: "FastAPI") -> None:
    """Drain the global events channel and hand violations to the reporter.

    This is the whole integration seam. The channel already carries every event
    from every feed (worker processes included), keyed by ``feed_id``, so the
    reporter can be wired in here without touching the CV pipeline, the workers,
    or the inference process. Central knows cameras by *its* ``camera_id``, so we
    translate through the mapping recorded when the feed was created.
    """
    hub = app.state.pool or app.state.feeds
    reporter = app.state.reporter
    queue = hub.subscribe_events()
    try:
        while True:
            msg = await queue.get()
            events = msg.get("events") or []
            if not events:
                continue
            cam_id = app.state.camera_ids.get(msg.get("feed_id"))
            if cam_id:
                reporter.report_events(cam_id, events)
            # No camera_id means this feed was started directly on the module
            # (not placed by central) — it still logs locally, just isn't reported.
    except asyncio.CancelledError:
        raise
    finally:
        hub.unsubscribe_events(queue)


def _module_status(app: "FastAPI") -> dict:
    """Payload for the heartbeat: load + per-camera status, for central."""
    hub = getattr(app.state, "pool", None) or getattr(app.state, "feeds", None)
    if hub is None:
        return {"active_feeds": 0, "feeds": []}
    feeds = []
    for f in hub.list():
        cam_id = app.state.camera_ids.get(f.get("feed_id"))
        if cam_id:
            feeds.append({"camera_id": cam_id, "feed_id": f.get("feed_id"),
                          "status": f.get("status")})
    return {"active_feeds": hub.count(), "feeds": feeds}


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    logger.info("Server starting up…")
    _log_hardware()
    cfg = Config()
    # feed_id -> central's camera_id, for attributing events and status upstream.
    app.state.camera_ids = {}
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

    # Report to central if CENTRAL_URL is set; otherwise a no-op and the module
    # runs exactly as it did standalone.
    from reporting import CentralReporter

    app.state.reporter = CentralReporter(
        cfg, status_provider=lambda: _module_status(app)
    ).start()
    app.state.event_forwarder = (
        asyncio.create_task(_forward_events_to_central(app))
        if app.state.reporter.enabled else None
    )

    yield

    if getattr(app.state, "event_forwarder", None) is not None:
        app.state.event_forwarder.cancel()
    if getattr(app.state, "reporter", None) is not None:
        app.state.reporter.stop()
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
    stopped = (pool.stop(feed_id) if pool is not None
               else app.state.feeds.stop(feed_id))
    # Drop the central mapping so it doesn't leak as feeds come and go.
    app.state.camera_ids.pop(feed_id, None)
    return {"stopped": stopped, "feed_id": feed_id}


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
    next frame, so events start honouring it immediately.
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


PROBE_TIMEOUT_S = 25.0


def _grab_one_frame(url: str, cfg: Config):
    """Open a stream, take ONE frame, close. Blocking — call in an executor.

    Returns ``(width, height, fps, frame)``. Deliberately creates no ``Feed``:
    no registry entry, no ``max_feeds`` slot, no detection, no artefacts. This is
    what lets the UI show a still for zone/line drawing before the camera is
    actually added.
    """
    source = StreamURLSource(
        url, max_lag_s=0.0, hw_decode=cfg.hw_decode, stride=1,
        decode_threads=cfg.decode_threads,
    ).start()
    try:
        for _idx, frame in source.frames():
            return source.width, source.height, source.fps, frame
        return source.width, source.height, source.fps, None   # opened but no frames
    finally:
        source.close()


@app.post("/feeds/probe")
async def probe_stream(payload: dict) -> JSONResponse:
    """Return one still frame + the TRUE source dimensions, without adding a feed.

    The dashboard draws zone/line geometry on this preview. ``width``/``height`` are
    the real source size so the preview keeps the source aspect ratio — drawing on a
    distorted canvas would put the zone somewhere other than where it looked.
    The JPEG itself is downscaled for the wire; geometry is normalized 0..1 so the
    preview's display size is irrelevant to correctness.
    """
    import base64

    import cv2

    url = (payload.get("url") or "").strip() if isinstance(payload, dict) else ""
    if not url:
        return JSONResponse({"error": "missing 'url'"}, status_code=400)

    cfg = Config()
    loop = asyncio.get_event_loop()
    try:
        w, h, fps, frame = await asyncio.wait_for(
            loop.run_in_executor(None, _grab_one_frame, url, cfg),
            timeout=PROBE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": f"stream did not deliver a frame within {PROBE_TIMEOUT_S:.0f}s"},
            status_code=504,
        )
    except Exception as exc:  # noqa: BLE001 - bad URL, auth, unreachable, codec
        return JSONResponse({"error": f"could not open stream: {exc}"}, status_code=400)

    if frame is None:
        return JSONResponse({"error": "stream opened but produced no frames"},
                            status_code=502)

    fh, fw = frame.shape[:2]
    max_w = int(getattr(cfg, "probe_max_width", 960))
    shown = frame
    if fw > max_w:                     # shrink the wire payload only
        shown = cv2.resize(frame, (max_w, int(fh * (max_w / fw))))
    ok, buf = cv2.imencode(".jpg", shown,
                           [cv2.IMWRITE_JPEG_QUALITY,
                            int(getattr(cfg, "probe_jpeg_quality", 75))])
    if not ok:
        return JSONResponse({"error": "could not encode preview frame"},
                            status_code=500)
    logger.info("Probed %s -> %dx%d @ %.2f fps (no feed created).", url, fw, fh, fps)
    return JSONResponse({
        "width": fw, "height": fh, "fps": round(float(fps), 2),
        "image": base64.b64encode(buf.tobytes()).decode("ascii"),
    })


@app.post("/feeds/stream")
async def add_stream(payload: dict) -> JSONResponse:
    """Start a background feed from a live stream URL (RTSP / HLS / HTTP).

    In coordinator mode the feed is assigned to a worker process; viewers watch
    Violations are reported to central and written to ``events.csv``; snapshots are
    the only images kept. For live viewing, use the raw video service.
    """
    url = (payload.get("url") or "").strip() if isinstance(payload, dict) else ""
    name = (payload.get("name") if isinstance(payload, dict) else None) or url
    if not url:
        return JSONResponse({"error": "missing 'url'"}, status_code=400)
    zone, line_start, line_end = _parse_geometry(payload)
    # Central sends its own camera_id so events can be attributed back to it.
    # Absent when a feed is started directly against the module (standalone use).
    camera_id = payload.get("camera_id") if isinstance(payload, dict) else None

    pool = app.state.pool
    if pool is not None:
        # Optimistic: the worker opens the stream; failures surface on the
        # subscribe socket as an "error" payload.
        feed_id = pool.add_stream(url, name, zone, line_start, line_end)
        if feed_id is None:
            return JSONResponse({"error": f"feed limit reached ({pool.max_feeds})"},
                                status_code=429)
        if camera_id:
            app.state.camera_ids[feed_id] = camera_id
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
                decode_threads=cfg.decode_threads,
            ).start(),
        )
    except Exception as exc:  # noqa: BLE001 - bad URL / unreachable stream
        return JSONResponse({"error": f"could not open stream: {exc}"}, status_code=400)
    try:
        feed = mgr.create(source, cfg, zone, line_start, line_end,
                          name=name, kind="stream")
    except RuntimeError as exc:  # feed limit reached
        source.close()
        return JSONResponse({"error": str(exc)}, status_code=429)
    if camera_id:
        app.state.camera_ids[feed.feed_id] = camera_id
    asyncio.create_task(feed.run())
    return JSONResponse({
        "feed_id": feed.feed_id, "name": name,
        "width": source.width, "height": source.height, "fps": round(source.fps, 2),
    })


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
async def index() -> dict:
    """Status only. This service has no UI and serves no video.

    The dashboard lives in the central app; video playback is a separate process
    (``module/rawapp.py``). What used to be served here was a per-module video
    dashboard, which duplicated central's and could only ever show frames at
    detection rate.
    """
    mgr = app.state.pool or app.state.feeds
    return {
        "service": "detection",
        "active_feeds": mgr.count(),
        "max_feeds": mgr.max_feeds,
        "reporting_to": getattr(app.state.reporter, "central", "") or None,
        "endpoints": ["GET /feeds", "POST /feeds/stream", "POST /feeds/{id}/stop",
                      "POST /feeds/{id}/geometry", "POST /feeds/probe", "WS /events"],
    }
