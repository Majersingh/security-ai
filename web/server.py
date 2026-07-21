"""FastAPI server for the live streaming operator-monitoring UI.

Flow
----
1. Browser POSTs a video to ``/upload`` -> saved under ``uploads/`` -> job_id.
2. Browser opens WebSocket ``/ws/{job_id}``.
3. Server runs :class:`streaming.StreamingScanner`, and for every processed
   frame pushes a JSON message containing a base64 JPEG of the annotated frame,
   progress, and any events that just fired. The browser renders it live.

Only the transport lives here; all CV logic is reused from ``src/``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

# Make the CV package importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from config import Config  # noqa: E402
from streaming import FrameProcessor, StreamingScanner  # noqa: E402
from utils import format_timestamp, setup_logging  # noqa: E402

logger = setup_logging("INFO")

UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Frames are downscaled to this width before JPEG encoding, to keep the
# WebSocket payload small (display quality only; detection still runs full-res).
STREAM_MAX_WIDTH = 960
JPEG_QUALITY = 70

app = FastAPI(title="CCTV Operator Monitoring - Live Scan")

# Registry of uploaded jobs: job_id -> saved video path.
_JOBS: dict[str, Path] = {}


@app.post("/upload")
async def upload(file: UploadFile) -> JSONResponse:
    """Save an uploaded video and return a job id."""
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    job_id = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{job_id}{suffix}"

    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1 << 20):  # 1 MB chunks
            out.write(chunk)
            size += len(chunk)

    _JOBS[job_id] = dest
    logger.info("Uploaded %s (%.1f MB) -> job %s", file.filename, size / 1e6, job_id)
    return JSONResponse({"job_id": job_id, "filename": file.filename})


def _encode_frame(frame) -> str:
    """Downscale + JPEG-encode a BGR frame into a base64 data string."""
    h, w = frame.shape[:2]
    if w > STREAM_MAX_WIDTH:
        scale = STREAM_MAX_WIDTH / w
        frame = cv2.resize(frame, (STREAM_MAX_WIDTH, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_frame(data_url: str):
    """Decode a base64 (data-URL or bare) JPEG string into a BGR frame."""
    b64 = data_url.split(",", 1)[-1]  # strip "data:image/jpeg;base64," if present
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _parse_geometry(msg: dict):
    """Extract (zone_polygon, line_start, line_end) from a client message.

    Coordinates are *normalized* (0..1 fractions of the frame width/height), so
    they are resolution-independent; the pipeline scales them to pixels once it
    knows the real frame size. Missing/invalid parts return None so the
    corresponding rule is simply not registered.
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


_SENTINEL = object()


def _next(iterator):
    """Blocking ``next`` wrapper for run_in_executor (returns sentinel at end)."""
    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL


@app.websocket("/ws/{job_id}")
async def scan_ws(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()

    video_path = _JOBS.get(job_id)
    if video_path is None:
        await websocket.send_json({"type": "error", "message": "Unknown job id."})
        await websocket.close()
        return

    loop = asyncio.get_event_loop()
    try:
        # First message carries optional zone/line geometry (may be empty {}).
        cfg_msg = await websocket.receive_json()
        zone, line_start, line_end = _parse_geometry(cfg_msg)

        # Construct the scanner (loads model) off the event loop.
        scanner: StreamingScanner = await loop.run_in_executor(
            None,
            lambda: StreamingScanner(
                Config(), video_path,
                zone_polygon=zone, line_start=line_start, line_end=line_end,
            ),
        )
        await websocket.send_json(
            {
                "type": "meta",
                "fps": round(scanner.fps, 2),
                "total_frames": scanner.total_frames,
                "width": scanner.width,
                "height": scanner.height,
            }
        )

        generator = scanner.scan()
        while True:
            update = await loop.run_in_executor(None, _next, generator)
            if update is _SENTINEL:
                break

            events = [
                {**asdict(e)} for e in update.new_events
            ]
            await websocket.send_json(
                {
                    "type": "frame",
                    "i": update.frame_index,
                    "total": update.total_frames,
                    "progress": round(
                        100.0 * (update.frame_index + 1) / max(1, update.total_frames), 1
                    ),
                    "timestamp": format_timestamp(update.frame_index, update.fps),
                    "image": _encode_frame(update.annotated),
                    "events": events,
                }
            )

        await websocket.send_json(
            {"type": "done", "total_events": len(scanner.event_dicts), "events": scanner.event_dicts}
        )
    except WebSocketDisconnect:
        logger.info("Client disconnected from job %s; stopping scan.", job_id)
    except Exception as exc:  # noqa: BLE001 - surface any error to the client
        logger.exception("Scan failed for job %s", job_id)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/ws-live")
async def live_ws(websocket: WebSocket) -> None:
    """Live camera scanning: the browser pushes frames, we push back results.

    Protocol (JSON messages):
      client -> {"type":"init","fps":<n>}          once, first
      server -> {"type":"ready"}
      client -> {"type":"frame","image":"<dataURL>"}   (paced: one at a time)
      server -> {"type":"result","i":n,"image":..,"events":[..]}
      client -> {"type":"stop"}                     to end
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()
    processor: FrameProcessor | None = None
    frame_index = 0
    try:
        init = await websocket.receive_json()
        fps = float(init.get("fps", 6.0)) if isinstance(init, dict) else 6.0
        zone, line_start, line_end = _parse_geometry(init)
        # Live webcam: the person is large/close, so a smaller inference size is
        # plenty and much faster than the 1280 used for CCTV upload footage.
        live_cfg = Config()
        live_cfg.inference_imgsz = 640
        processor = await loop.run_in_executor(
            None,
            lambda: FrameProcessor(
                live_cfg, fps,
                zone_polygon=zone, line_start=line_start, line_end=line_end,
            ),
        )
        await websocket.send_json({"type": "ready"})

        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type") if isinstance(msg, dict) else None
            if mtype == "stop":
                break
            if mtype != "frame":
                continue

            frame = _decode_frame(msg["image"])
            if frame is None:
                continue

            boxes, new_events, w, h = await loop.run_in_executor(
                None, processor.process_json, frame, frame_index
            )
            # Return only lightweight JSON (boxes + events), no image. The browser
            # draws these over its own local video, so the network carries ~1 KB.
            await websocket.send_json(
                {
                    "type": "result",
                    "i": frame_index,
                    "w": w,
                    "h": h,
                    "boxes": boxes,
                    "events": [asdict(e) for e in new_events],
                }
            )
            frame_index += 1
    except WebSocketDisconnect:
        logger.info("Live client disconnected after %d frames.", frame_index)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Live scan failed")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if processor is not None:
            await loop.run_in_executor(None, processor.finalize)
        try:
            await websocket.close()
        except Exception:
            pass


# Serve the single-page UI at "/".
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
