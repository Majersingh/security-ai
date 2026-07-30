"""Streamer service — live video, standalone.

A third role alongside `central` (fleet brain) and the detection module. It plays
video and does nothing else: no model, no `FeedManager`, no worker processes, no
GPU. It **registers itself** with central, so a video-only host needs nothing but
this process — no detection service running purely to advertise it.

 uvicorn streamer.app:app --env-file streamer/.env --host 0.0.0.0 --port 8011

Why it is not part of the detection service: pushing frames at the source rate is
~28 `send_json` calls per second *per tile*. On the detection app that is the same
event loop relaying results from the worker processes, so a wall of tiles would add
latency to the thing the product actually sells. Separate processes mean video
cannot degrade detection, and a runaway decode here takes nothing else down.

Why it plays at source rate: the detection pipeline can only show a viewer frames it
ran detection on, so display fps was chained to detection fps (and `frame_stride`
discarded intermediate frames before they were even converted). Streaming decodes
independently — one extra decode per *watched* camera, which is the deliberate trade.

Env:
    CENTRAL_URL             where to register (blank = works, but central won't know)
    MODULE_TOKEN            shared secret, must match CENTRAL_TOKEN
    STREAMER_ID             stable id for this streamer (default: <host>-streamer)
    STREAMER_PUBLIC_URL     where BROWSERS reach this service
    HOST_ID                 groups this streamer with the detection module on the
                            same machine, so central prefers a co-located streamer
                            (default: hostname)
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Own directory plus the shared top-level `core` package (the stream decoder and the
# process helpers). Nothing from the detection module's tree is imported.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "core"))
sys.path.insert(0, str(HERE))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from config import StreamerConfig  # noqa: E402
from procutil import limit_process_threads, setup_logging  # noqa: E402

_cfg = StreamerConfig.from_env()
# This process decodes and JPEG-encodes; it never runs a model. A little room for
# OpenCV, but bounded — it shares the box with the detection processes.
limit_process_threads(max(1, _cfg.streamer_cv_threads), 1)

logger = setup_logging(_cfg.log_level)

from videostream import (  # noqa: E402
    active_streams, issue_ticket, resolve_ticket, stream_video,
)

HEARTBEAT_S = 10.0

CENTRAL = (os.environ.get("CENTRAL_URL") or "").rstrip("/")
TOKEN = os.environ.get("MODULE_TOKEN") or ""
HOST_ID = os.environ.get("HOST_ID") or socket.gethostname()
STREAMER_ID = os.environ.get("STREAMER_ID") or f"{HOST_ID}-streamer"
PUBLIC_URL = (os.environ.get("STREAMER_PUBLIC_URL") or "").rstrip("/")

app = FastAPI(title="CCTV Operator Monitoring — Streamer")


# --------------------------------------------------------------- registration

class _Registrar:
    """Announce this streamer to central and keep it marked online.

    It registers under its OWN id with ``role="streamer"``, so it is a first-class
    host in central's registry rather than a field on some detection module. That is
    what makes a video-only box possible, and it stops central ever treating a
    streamer as a placement target (``max_feeds`` is 0).
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self.registered = False
        self._last_complaint = 0.0

    @property
    def enabled(self) -> bool:
        return bool(CENTRAL)

    def start(self) -> "_Registrar":
        if not self.enabled:
            logger.info("CENTRAL_URL not set — streaming works, but central will not "
                        "know this streamer exists.")
            return self
        if not PUBLIC_URL:
            # Central requires a non-empty url and returns 400 without it, so this is
            # fatal for registration, not a warning to shrug at.
            logger.error(
                "STREAMER_PUBLIC_URL is not set, so central WILL REJECT registration "
                "and no camera will be playable. Set it to an address a browser can "
                "reach (e.g. http://<this-host>:8011) — and check you passed "
                "--env-file streamer/.env, not module/.env."
            )
        threading.Thread(target=self._loop, name="streamer-registrar",
                         daemon=True).start()
        logger.info("Registering with central %s as streamer '%s' (host=%s, public=%s).",
                    CENTRAL, STREAMER_ID, HOST_ID, PUBLIC_URL or "unset")
        return self

    def stop(self) -> None:
        self._stop.set()

    def _post(self, path: str, body: dict) -> None:
        req = urllib.request.Request(f"{CENTRAL}{path}",
                                     data=json.dumps(body).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        if TOKEN:
            req.add_header("X-Module-Token", TOKEN)
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()

    def _loop(self) -> None:
        last = 0.0
        while not self._stop.is_set():
            try:
                if not self.registered:
                    self._post("/api/modules/register", {
                        "id": STREAMER_ID,
                        "url": PUBLIC_URL,
                        "role": "streamer",
                        "host": HOST_ID,
                        "max_feeds": 0,       # never a placement target
                        "version": "streamer/1",
                    })
                    self.registered = True
                    logger.info("Registered with central as streamer '%s'.", STREAMER_ID)
                elif time.time() - last >= HEARTBEAT_S:
                    self._post(f"/api/modules/{STREAMER_ID}/heartbeat",
                               {"active_feeds": active_streams(), "feeds": []})
                    last = time.time()
            except Exception as exc:  # noqa: BLE001 - never let this kill the service
                was = self.registered
                self.registered = False        # central may have restarted/forgotten us
                # Visible on the FIRST failure and then rate-limited. Logging this at
                # debug meant a rejected registration looked like nothing happening at
                # all, which is the worst way for this to fail.
                now = time.time()
                if was or now - self._last_complaint > 30.0:
                    self._last_complaint = now
                    logger.warning(
                        "Streamer registration/heartbeat FAILED: %s — central will not "
                        "know this streamer exists. Check CENTRAL_URL, MODULE_TOKEN "
                        "and STREAMER_PUBLIC_URL.", exc,
                    )
            self._stop.wait(2.0)


registrar = _Registrar()


@app.on_event("startup")
async def _startup() -> None:
    registrar.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    registrar.stop()


# ------------------------------------------------------------------- endpoints

@app.get("/health")
async def health() -> dict:
    """Liveness plus how many tiles this host is currently decoding."""
    return {
        "ok": True,
        "service": "streamer",
        "id": STREAMER_ID,
        "host": HOST_ID,
        "registered": registrar.registered,
        "active_streams": active_streams(),
        "max_streams": _cfg.streamer_max_streams,
    }


@app.post("/stream/ticket")
async def stream_ticket(payload: dict) -> JSONResponse:
    """Exchange a stream URL for a short-lived ticket.

    Browsers connect with the ticket, never the URL — an
    ``rtsp://user:pass@host`` in a WebSocket query string would end up in browser
    history and access logs.
    """
    url = (payload.get("url") or "").strip() if isinstance(payload, dict) else ""
    if not url:
        return JSONResponse({"error": "missing 'url'"}, status_code=400)
    ticket, ttl = issue_ticket(url)
    return JSONResponse({"ticket": ticket, "expires_in": ttl})


@app.websocket("/stream")
async def stream_ws(websocket: WebSocket) -> None:
    """Play video at the source frame rate. No feed, no detection, no GPU."""
    await websocket.accept()
    q = websocket.query_params
    url = resolve_ticket(q.get("ticket", ""))
    if not url:
        await websocket.send_json({"type": "error",
                                   "message": "unknown or expired ticket"})
        await websocket.close()
        return
    try:
        await stream_video(
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


logger.info("Streamer ready (id=%s, max_streams=%d, width<=%d, quality=%d).",
            STREAMER_ID, _cfg.streamer_max_streams,
            _cfg.streamer_max_width,
            _cfg.streamer_jpeg_quality)
