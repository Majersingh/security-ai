"""Raw video service — a SEPARATE PROCESS from detection.

Runs on its own port and shares nothing with the analysis pipeline: no model, no
`FeedManager`, no worker processes, no inference process, no event loop in common.

    uvicorn module.rawapp:app --host 0.0.0.0 --port 8011

Why its own process rather than a couple of routes on the detection app: pushing
frames at the source rate means ~28 `send_json` calls per second *per tile* on
whichever event loop serves them. On the detection app that is the same loop
relaying analysed frames from the worker processes to browsers, so a wall of tiles
would add latency to the thing the product actually sells. Separating the process
means raw playback cannot degrade detection, and a crash or a runaway decode here
takes nothing else down.

The trade is one extra decode per watched camera. Decode is cheap next to a GPU
pass, and it only happens while a human has that tile open.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Same import shape as module/app.py: flat imports from the CV core, plus this
# directory so `rawstream` resolves when loaded as `module.rawapp:app`.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from config import Config  # noqa: E402
from utils import limit_process_threads, setup_logging  # noqa: E402

_cfg = Config()
# This process decodes and JPEG-encodes; it never runs the model. Give OpenCV a
# little room but stay bounded, since it shares the box with the detection
# processes and their own thread caps.
limit_process_threads(max(1, int(getattr(_cfg, "raw_cv_threads", 2))), 1)

logger = setup_logging(getattr(_cfg, "log_level", "INFO"))

from rawstream import active_streams, issue_ticket, resolve_ticket, stream_raw  # noqa: E402

app = FastAPI(title="CCTV Operator Monitoring — Raw Video")


@app.get("/health")
async def health() -> dict:
    """Liveness plus how many tiles are currently being decoded here."""
    return {
        "ok": True,
        "service": "rawvideo",
        "active_streams": active_streams(),
        "max_streams": int(getattr(_cfg, "raw_max_streams", 16)),
    }


@app.post("/stream/raw/ticket")
async def raw_ticket(payload: dict) -> JSONResponse:
    """Exchange a stream URL for a short-lived ticket.

    The browser connects with the ticket, never the URL — an
    ``rtsp://user:pass@host`` in a WebSocket query string would end up in browser
    history and access logs.
    """
    url = (payload.get("url") or "").strip() if isinstance(payload, dict) else ""
    if not url:
        return JSONResponse({"error": "missing 'url'"}, status_code=400)
    ticket, ttl = issue_ticket(url)
    return JSONResponse({"ticket": ticket, "expires_in": ttl})


@app.websocket("/stream/raw")
async def raw_stream_ws(websocket: WebSocket) -> None:
    """Play raw video at the source frame rate. No feed, no detection, no GPU."""
    await websocket.accept()
    q = websocket.query_params
    url = resolve_ticket(q.get("ticket", ""))
    if not url:
        await websocket.send_json({"type": "error",
                                   "message": "unknown or expired ticket"})
        await websocket.close()
        return
    try:
        await stream_raw(
            websocket, url, _cfg,
            target_fps=float(q.get("fps", 0) or 0),
            max_width=int(q.get("width", 0) or 0),
        )
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


logger.info("Raw video service ready (max_streams=%d, width<=%d, quality=%d).",
            int(getattr(_cfg, "raw_max_streams", 16)),
            int(getattr(_cfg, "raw_max_width", 960)),
            int(getattr(_cfg, "raw_jpeg_quality", 55)))
