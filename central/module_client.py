"""Outbound calls from central to an analysis module.

Deliberately the ONLY place central dials a module. Right now that is plain HTTP
to the module's existing ``/feeds/*`` API, which works because every host is
publicly reachable. When a module ends up behind NAT at a client site, central
will no longer be able to initiate anything — at which point this class is what
gets swapped for "push the command down the WebSocket the module already opened
to us." Keeping every outbound call behind this one interface is what makes that
a contained change instead of a rewrite.

Uses stdlib ``urllib`` in a thread rather than an async HTTP client: these are
control-plane calls (a few per camera, not per frame), so the dependency isn't
worth it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger("central")

DEFAULT_TIMEOUT = 10.0


class ModuleError(RuntimeError):
    """A module rejected a command or was unreachable."""


def _request(url: str, method: str = "GET", body: Optional[dict] = None,
             timeout: float = DEFAULT_TIMEOUT) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:200]
        raise ModuleError(f"{method} {url} -> HTTP {exc.code}: {detail}") from exc
    except Exception as exc:  # noqa: BLE001 - unreachable, DNS, timeout, bad JSON
        raise ModuleError(f"{method} {url} -> {type(exc).__name__}: {exc}") from exc


class ModuleClient:
    """Talks to one module. `base_url` is what the module advertised at register."""

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")

    async def _call(self, path: str, method: str = "GET",
                    body: Optional[dict] = None) -> Any:
        # Blocking urllib -> off the event loop.
        return await asyncio.to_thread(_request, f"{self.base}{path}", method, body)

    async def start_feed(self, url: str, name: str,
                         geometry: Optional[dict] = None,
                         camera_id: Optional[str] = None) -> Dict[str, Any]:
        """Start a feed. ``camera_id`` is OUR id for the camera — the module keeps
        it so the events and statuses it reports back are attributable, since the
        module otherwise only knows its own ``feed_id``."""
        payload: Dict[str, Any] = {"url": url, "name": name}
        if camera_id:
            payload["camera_id"] = camera_id
        if geometry:
            payload.update(geometry)          # zone_polygon / line_start / line_end
        return await self._call("/feeds/stream", "POST", payload)

    async def stop_feed(self, feed_id: str) -> Dict[str, Any]:
        return await self._call(f"/feeds/{feed_id}/stop", "POST", {})

    async def set_geometry(self, feed_id: str, geometry: dict) -> Dict[str, Any]:
        return await self._call(f"/feeds/{feed_id}/geometry", "POST", geometry)

    async def feeds(self) -> Dict[str, Any]:
        return await self._call("/feeds")

    async def probe(self, url: str) -> Dict[str, Any]:
        """One still frame + true source dimensions, without creating a feed.

        Opening an RTSP stream can be slow, so this gets a longer timeout than the
        other control calls.
        """
        return await asyncio.to_thread(
            _request, f"{self.base}/feeds/probe", "POST", {"url": url}, 30.0
        )

    def video_url(self, feed_id: str) -> str:
        """WebSocket URL the BROWSER uses — video never passes through central."""
        ws = self.base.replace("https://", "wss://").replace("http://", "ws://")
        return f"{ws}/feeds/{feed_id}/subscribe"
