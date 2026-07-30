"""Raw video streaming — completely independent of detection.

Why this exists: the analysis path can only show a viewer frames it ran detection
on, so display fps was capped by detection fps (and by `frame_stride`, which
discards intermediate frames before they are even converted to BGR). That makes
smooth video impossible without spending the whole GPU on a handful of cameras.

This path shares **nothing** with the pipeline: its own decode, its own pacing, no
`Feed`, no `FeedManager` slot, no model, no events, no `frame_stride`. It plays at
the source frame rate. Detection carries on untouched at whatever rate it can
manage, and the two cannot affect each other.

The cost is a second decode of that camera while someone is watching it, which is
the deliberate trade: decode is cheap next to a GPU pass, and it only happens for
cameras a human has actually opened.

Credentials: browsers connect with a short-lived **ticket**, never the RTSP URL.
An `rtsp://user:pass@host` URL in a WebSocket query string would land in browser
history, referrers and access logs. `issue_ticket()` keeps the URL server-side.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from typing import Dict, Optional, Tuple

import cv2

from sources import StreamURLSource

logger = logging.getLogger("operator_monitor")

TICKET_TTL_S = 120.0          # generous enough to survive a slow page load
_tickets: Dict[str, Tuple[str, float]] = {}     # ticket -> (url, expires_at)
_active = 0                                     # concurrent raw streams


def issue_ticket(url: str) -> Tuple[str, float]:
    """Store a URL server-side and return an opaque ticket for the browser."""
    _prune()
    ticket = uuid.uuid4().hex
    _tickets[ticket] = (url, time.time() + TICKET_TTL_S)
    return ticket, TICKET_TTL_S


def resolve_ticket(ticket: str) -> Optional[str]:
    """URL for a ticket, or None if unknown/expired. Tickets are reusable until
    they expire so a dropped socket can reconnect without a new round trip."""
    _prune()
    entry = _tickets.get(ticket or "")
    return entry[0] if entry else None


def _prune() -> None:
    now = time.time()
    for t in [t for t, (_u, exp) in _tickets.items() if exp <= now]:
        _tickets.pop(t, None)


def active_streams() -> int:
    return _active


async def stream_raw(websocket, url: str, cfg, target_fps: float = 0.0,
                     max_width: int = 0) -> None:
    """Decode `url` and push JPEG frames down an already-accepted WebSocket.

    Returns when the client goes away or the source ends. Every blocking step runs
    in a thread so this never stalls the event loop serving other feeds.
    """
    global _active

    limit = int(getattr(cfg, "raw_max_streams", 16))
    if _active >= limit:
        await websocket.send_json({
            "type": "error",
            "message": f"raw stream limit reached ({limit}) — close a tile first",
        })
        return

    loop = asyncio.get_event_loop()
    width_cap = int(max_width or getattr(cfg, "raw_max_width", 960))
    quality = int(getattr(cfg, "raw_jpeg_quality", 55))

    try:
        source = await loop.run_in_executor(
            None,
            lambda: StreamURLSource(
                url,
                # Raw playback wants EVERY frame: no stride, and drop-when-behind so
                # a slow viewer stays near live instead of drifting into slow motion.
                stride=1,
                max_lag_s=getattr(cfg, "stream_max_lag_seconds", 0.5),
                hw_decode=getattr(cfg, "hw_decode", True),
                decode_threads=getattr(cfg, "decode_threads", 1),
            ).start(),
        )
    except Exception as exc:  # noqa: BLE001 - bad URL, auth, unreachable
        await websocket.send_json({"type": "error",
                                   "message": f"could not open stream: {exc}"})
        return

    _active += 1
    sent = 0
    try:
        await websocket.send_json({
            "type": "meta", "width": source.width, "height": source.height,
            "fps": round(float(source.fps), 2),
            "decode": "nvdec" if source.hw_active else "cpu",
        })

        gen = source.frames()
        # Only skip frames if the client asked for fewer than the source provides.
        src_fps = float(source.fps) or 30.0
        want = float(target_fps) if target_fps and target_fps > 0 else src_fps
        keep_every = max(1, int(round(src_fps / max(1.0, min(want, src_fps)))))

        idx = 0
        while True:
            item = await loop.run_in_executor(None, _next_frame, gen)
            if item is None:
                break
            _raw_idx, frame = item
            idx += 1
            if keep_every > 1 and idx % keep_every:
                continue
            payload = await loop.run_in_executor(
                None, _encode, frame, width_cap, quality)
            if payload is None:
                continue
            await websocket.send_json({"type": "frame", "image": payload})
            sent += 1
    except Exception:  # noqa: BLE001 - client vanished, or decode blew up
        logger.debug("raw stream ended", exc_info=True)
    finally:
        _active -= 1
        await loop.run_in_executor(None, source.close)
        logger.info("Raw stream closed after %d frame(s): %s", sent, url)


def _next_frame(gen):
    try:
        return next(gen)
    except StopIteration:
        return None


def _encode(frame, width_cap: int, quality: int) -> Optional[str]:
    h, w = frame.shape[:2]
    if w > width_cap:
        frame = cv2.resize(frame, (width_cap, int(h * (width_cap / w))))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
