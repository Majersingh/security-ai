"""Central app — the fleet's brain. Owns everything except frames.

Responsibilities: the camera registry (users add RTSP links here), the module
registry and health, placement, the event store, and the dashboard.

It never decodes a frame, never loads a model, and needs no GPU — which is why
``central/requirements.txt`` shares nothing with ``module/requirements.txt``.
Video goes straight from a module's raw-video service to the browser; central only
hands out a short-lived URL. The detection module serves no video at all.

Scaling: deploy another module. It registers itself, central sees the new
headroom, and cameras start landing on it. Nothing here is edited.

Run:
    uvicorn central.app:app --host 0.0.0.0 --port 9000
Env:
    CENTRAL_DB          path to the SQLite file (default central/central.db)
    CENTRAL_TOKEN       shared secret modules must present to register
    CENTRAL_STRIDE      frame_stride the modules run, for capacity math (default 1)
    CENTRAL_STALE_AFTER seconds without a heartbeat before a module is offline (30)
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from central.db import Store
from central.module_client import ModuleClient, ModuleError
from central.placement import choose_module, fleet_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | central | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("central")

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

DB_PATH = Path(os.environ.get("CENTRAL_DB", HERE / "central.db"))
TOKEN = os.environ.get("CENTRAL_TOKEN") or ""
FRAME_STRIDE = int(os.environ.get("CENTRAL_STRIDE", "1"))
STALE_AFTER = float(os.environ.get("CENTRAL_STALE_AFTER", "30"))


def _check_token(supplied: Optional[str]) -> None:
    """Modules must present the shared secret. No token configured = open."""
    if TOKEN and supplied != TOKEN:
        raise HTTPException(status_code=401, detail="bad or missing module token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = Store(DB_PATH)
    logger.info("Central up. db=%s token=%s stride=%d",
                DB_PATH, "set" if TOKEN else "NOT SET (open)", FRAME_STRIDE)
    reaper = asyncio.create_task(_reap_dead_modules(app))
    yield
    reaper.cancel()
    app.state.store.close()


async def _reap_dead_modules(app: FastAPI) -> None:
    """Detach cameras from modules that stopped heartbeating, so they can be
    re-placed onto a healthy module rather than sitting silently dead."""
    while True:
        try:
            await asyncio.sleep(STALE_AFTER / 2)
            store: Store = app.state.store
            for mod in store.modules(STALE_AFTER):
                if mod["online"]:
                    continue
                orphaned = store.unplace_module_cameras(mod["id"])
                if orphaned:
                    logger.warning(
                        "Module %s offline (%.0fs) — released %d camera(s) for "
                        "re-placement. Track identities will restart.",
                        mod["id"], mod["seconds_since_seen"], len(orphaned),
                    )
            await _place_unplaced(app)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - never let the reaper die
            logger.exception("reaper iteration failed")


app = FastAPI(title="CCTV Operator Monitoring — Central", lifespan=lifespan)


# ------------------------------------------------------------- module contract

@app.post("/api/modules/register")
async def register_module(payload: dict,
                         x_module_token: Optional[str] = Header(None)) -> dict:
    """A module announces itself on startup. Idempotent — restart re-registers."""
    _check_token(x_module_token)
    for required in ("id", "url"):
        if not payload.get(required):
            raise HTTPException(400, f"missing '{required}'")
    app.state.store.upsert_module(payload)
    logger.info("Module registered: %s at %s (gpu=%s max_feeds=%s fps_budget=%s)",
                payload["id"], payload["url"], payload.get("gpu"),
                payload.get("max_feeds"), payload.get("fps_budget"))
    await _place_unplaced(app)              # new capacity -> use it immediately
    return {"ok": True, "module_id": payload["id"]}


@app.post("/api/modules/{module_id}/heartbeat")
async def heartbeat(module_id: str, payload: dict,
                    x_module_token: Optional[str] = Header(None)) -> dict:
    """Liveness + current load. Also carries per-feed status back to central."""
    _check_token(x_module_token)
    store: Store = app.state.store
    if store.module(module_id) is None:
        # Central restarted, or the module was reaped: ask it to register again.
        return JSONResponse({"ok": False, "reregister": True}, status_code=409)
    store.touch_module(module_id, int(payload.get("active_feeds", 0)))

    for feed in payload.get("feeds", []) or []:
        cam_id = feed.get("camera_id")
        status = feed.get("status")
        if cam_id and status:
            cam = store.camera(cam_id)
            if cam and cam["status"] != status:
                store.set_camera_status(cam_id, status)
    return {"ok": True}


@app.post("/api/events")
async def ingest_events(payload: dict,
                        x_module_token: Optional[str] = Header(None)) -> dict:
    """Modules POST violation batches here. This is the product's data path.

    Modules spool locally and retry, so this may receive duplicates after a
    network blip — better a duplicate alert than a lost one.
    """
    _check_token(x_module_token)
    cam_id = payload.get("camera_id")
    if not cam_id:
        raise HTTPException(400, "missing 'camera_id'")
    events = payload.get("events") or []
    n = app.state.store.add_events(cam_id, payload.get("module_id"), events)
    if n:
        logger.info("Ingested %d event(s) from camera %s", n, cam_id[:8])
    return {"ok": True, "stored": n}


# --------------------------------------------------------------- camera admin

@app.post("/api/cameras/probe")
async def probe_camera(payload: dict) -> JSONResponse:
    """Preview a stream before adding it: one still frame + true source size.

    Borrows any online module to open the stream briefly. No camera is registered
    and no feed is created, so this is safe to call repeatedly while the user is
    getting the URL right. The dashboard draws zone/line on the returned frame and
    then posts url+name+geometry together, so the camera starts already configured
    instead of running unconfigured until someone remembers to draw.
    """
    url = (payload.get("url") or "").strip() if isinstance(payload, dict) else ""
    if not url:
        raise HTTPException(400, "missing 'url'")

    store: Store = app.state.store
    online = [m for m in store.modules(STALE_AFTER) if m["online"]]
    if not online:
        return JSONResponse(
            {"error": "no module is online to open the stream — start a module first"},
            status_code=503,
        )

    # Probe on the module that placement would ACTUALLY choose, so "preview worked"
    # implies "this camera will work once added" — same host, same network, same
    # route to the camera. Probing an arbitrary module can fail (or succeed) for
    # reachability reasons that have nothing to do with where the feed will run,
    # which is misleading once modules sit on different networks.
    #
    # Source fps isn't known until we have probed, so assume 30 purely for ordering.
    likely = choose_module(online, store.cameras(), {"source_fps": 30.0}, FRAME_STRIDE)
    order = ([m for m in online if m["id"] == likely] +
             [m for m in online if m["id"] != likely])

    # Still fall through to the others: a sick or unreachable module shouldn't
    # block preview entirely. Geometry is normalized, so a frame from any module
    # stays valid wherever the camera is finally placed.
    last = ""
    for mod in order:
        try:
            res = await ModuleClient(mod["url"]).probe(url)
            res["probed_by"] = mod["id"]
            return JSONResponse(res)
        except ModuleError as exc:
            last = str(exc)
            logger.warning("Probe via %s failed: %s", mod["id"], exc)
    return JSONResponse({"error": f"could not preview stream: {last}"}, status_code=502)


@app.post("/api/cameras/{cam_id}/stream")
async def camera_stream_url(cam_id: str,
                            payload: Optional[dict] = None) -> JSONResponse:
    """WebSocket URL for live video of this camera — independent of detection.

    Smooth playback at the source frame rate, because it does not wait on the
    model. Costs one extra decode on the module while a viewer is watching, which
    is why it is requested per camera rather than opened for everything.

    The RTSP URL never reaches the browser: the module issues a short-lived ticket
    and we hand back only that.
    """
    store: Store = app.state.store
    cam = store.camera(cam_id)
    if cam is None:
        raise HTTPException(404, "unknown camera")

    # ANY streamer can play ANY camera it can reach: streaming shares nothing with
    # detection — own decode, no feed, no model — so the host serving video need not
    # be the one analysing the camera.
    #
    # Order matters. A streamer on the SAME machine as the detecting module provably
    # has a route to this camera (that machine is decoding it right now), so prefer
    # it. Others are still worth trying: they may sit on another network and fail, but
    # one of them may be the only streamer running at all.
    all_mods = store.modules(STALE_AFTER)
    streamers = [m for m in all_mods if m["online"] and m.get("role") == "streamer"]
    if not streamers:
        return JSONResponse(
            {"error": "no streamer is running — start one with "
                      "`uvicorn module.streamer:app --port 8011` and set "
                      "STREAMER_PUBLIC_URL"},
            status_code=503,
        )
    owner = next((m for m in all_mods if m["id"] == cam.get("module_id")), None)
    owner_host = (owner or {}).get("host")
    candidates = ([m for m in streamers if owner_host and m.get("host") == owner_host] +
                  [m for m in streamers if not (owner_host and m.get("host") == owner_host)])

    body = payload or {}
    fps = int(body.get("fps") or 0)
    width = int(body.get("width") or 0)
    last = ""
    for mod in candidates:
        try:
            client = ModuleClient(mod["url"])
            res = await client.stream_ticket(cam["url"])
            return JSONResponse({
                "ws_url": client.stream_ws_url(res["ticket"], fps, width),
                # Which host is decoding for this tile — not necessarily the one
                # detecting the camera.
                "served_by": mod["id"],
                "detected_by": cam.get("module_id"),
                "expires_in": res.get("expires_in"),
            })
        except (ModuleError, KeyError) as exc:
            last = str(exc)
            logger.warning("stream ticket via %s failed: %s", mod["id"], exc)
    return JSONResponse({"error": f"could not start video: {last}"}, status_code=502)


@app.post("/api/cameras")
async def add_camera(payload: dict) -> dict:
    """Register a camera and place it on a module with headroom."""
    url = (payload.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "missing 'url'")
    name = (payload.get("name") or url).strip()
    geometry = payload.get("geometry")
    # Optional: pin this camera to a specific module instead of auto-placing.
    pinned = (payload.get("module_id") or "").strip() or None
    if pinned:
        target = app.state.store.module(pinned)
        if target is None:
            raise HTTPException(400, f"unknown module '{pinned}'")
        if target.get("role") == "streamer":
            raise HTTPException(
                400, f"'{pinned}' is a streamer (video only) and cannot run detection")

    cam_id = app.state.store.add_camera(name, url, geometry, pinned_module=pinned)
    placed = await _try_place(app, cam_id)
    return {"camera_id": cam_id, "name": name, "placed": placed, "pinned_to": pinned}


@app.get("/api/cameras")
async def list_cameras() -> dict:
    """Every camera with its module and health.

    No video URL here: the detection module serves no video. Live playback is
    requested per camera via ``POST /api/cameras/{id}/raw``, which returns a
    WebSocket on the module host's separate raw-video service.
    """
    store: Store = app.state.store
    all_mods = store.modules(STALE_AFTER)
    mods = {m["id"]: m for m in all_mods}
    any_streamer = any(m["online"] and m.get("role") == "streamer"
                       for m in all_mods)
    out = []
    for cam in store.cameras():
        row = dict(cam)
        mod = mods.get(cam["module_id"]) if cam["module_id"] else None
        row["module_online"] = bool(mod and mod["online"])
        # Can this camera be played, and if not, WHY? The reason is surfaced in the
        # UI: "not playable" with no explanation sends people hunting through logs
        # for a missing RAW_PUBLIC_URL.
        # Playable if ANY online host runs the raw service — it does not have to be
        # the module detecting this camera, because raw playback is independent.
        if any_streamer:
            row["playable"], row["playable_reason"] = True, None
        elif mod is None:
            row["playable"], row["playable_reason"] = False, "not placed on a module yet"
        else:
            row["playable"] = False
            row["playable_reason"] = (
                "no streamer is running — start one with "
                "`uvicorn module.streamer:app --port 8011` and set "
                "STREAMER_PUBLIC_URL"
            )
        out.append(row)
    return {"cameras": out}


@app.delete("/api/cameras/{cam_id}")
async def delete_camera(cam_id: str) -> dict:
    """Stop the feed on its module (best effort), then forget the camera."""
    store: Store = app.state.store
    cam = store.camera(cam_id)
    if cam is None:
        raise HTTPException(404, "unknown camera")
    stopped = False
    if cam.get("module_id") and cam.get("feed_id"):
        mod = store.module(cam["module_id"])
        if mod:
            try:
                await ModuleClient(mod["url"]).stop_feed(cam["feed_id"])
                stopped = True
            except ModuleError as exc:
                logger.warning("stop_feed failed for %s: %s", cam_id[:8], exc)
    store.delete_camera(cam_id)
    return {"deleted": True, "stopped_on_module": stopped}


@app.post("/api/cameras/{cam_id}/geometry")
async def set_geometry(cam_id: str, payload: dict) -> dict:
    """Persist zone/line on the camera AND push it to the owning module."""
    store: Store = app.state.store
    cam = store.camera(cam_id)
    if cam is None:
        raise HTTPException(404, "unknown camera")
    store.set_camera_geometry(cam_id, payload or None)
    applied = False
    if cam.get("module_id") and cam.get("feed_id"):
        mod = store.module(cam["module_id"])
        if mod:
            try:
                res = await ModuleClient(mod["url"]).set_geometry(
                    cam["feed_id"], payload or {})
                applied = bool(res.get("applied"))
            except ModuleError as exc:
                logger.warning("set_geometry failed for %s: %s", cam_id[:8], exc)
    return {"saved": True, "applied_live": applied}


# ------------------------------------------------------------------ fleet/UI

@app.get("/api/fleet")
async def fleet() -> dict:
    """Capacity view: what each module is committed to, and what's left."""
    store: Store = app.state.store
    summary = fleet_summary(store.modules(STALE_AFTER), store.cameras(), FRAME_STRIDE)
    summary["frame_stride"] = FRAME_STRIDE
    summary["event_counts"] = store.event_counts()
    return summary


@app.get("/api/events")
async def list_events(limit: int = 100, camera_id: Optional[str] = None) -> dict:
    return {"events": app.state.store.recent_events(limit, camera_id)}


@app.get("/wall")
async def wall():
    """Video-wall page: pick cameras, see live tiles in a responsive grid.

    Note browsers cannot play RTSP directly — no <video> or MSE support for it — so
    tiles use the module's frame WebSocket, the same transport the dashboard uses.
    Each tile connects straight to its owning module; central only serves this page.
    """
    page = STATIC_DIR / "wall.html"
    if not page.exists():
        raise HTTPException(404, "wall page not built")
    return FileResponse(page, headers={"Cache-Control": "no-store"})


@app.get("/")
async def index():
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        return JSONResponse({"error": "dashboard not built yet",
                             "api": ["/api/fleet", "/api/cameras", "/api/events"]})
    return FileResponse(idx, headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- placement

async def _try_place(app: FastAPI, cam_id: str) -> Optional[str]:
    """Place one camera on the best module. Returns the module id, or None.

    None is a legitimate outcome: the fleet is full and the answer is to deploy
    another module, not to overcommit an existing one.
    """
    store: Store = app.state.store
    cam = store.camera(cam_id)
    if cam is None or cam.get("module_id"):
        return cam.get("module_id") if cam else None

    modules = store.modules(STALE_AFTER)
    pinned = cam.get("pinned_module")
    if pinned:
        # A pin is honoured strictly and NEVER silently overridden. Users pin a
        # camera to a module for reasons central can't see — usually that this host
        # is the one with a network route to it — so quietly relocating it would
        # break the camera in a way nobody could diagnose. If the pinned module is
        # down or full, the camera stays unplaced and says so.
        mod = next((m for m in modules if m["id"] == pinned), None)
        if mod is None or not mod["online"]:
            logger.warning("Camera %s (%s) is pinned to module '%s', which is %s — "
                           "leaving it unplaced rather than moving it.",
                           cam_id[:8], cam["name"], pinned,
                           "unknown" if mod is None else "offline")
            store.set_camera_status(cam_id, "waiting for pinned module")
            return None
        module_id = pinned
    else:
        module_id = choose_module(modules, store.cameras(), cam, FRAME_STRIDE)
        if module_id is None:
            logger.warning("No module has headroom for camera %s (%s) — deploy another.",
                           cam_id[:8], cam["name"])
            store.set_camera_status(cam_id, "unplaced")
            return None

    mod = store.module(module_id)
    try:
        res = await ModuleClient(mod["url"]).start_feed(
            cam["url"], cam["name"], cam.get("geometry"), camera_id=cam_id)
    except ModuleError as exc:
        logger.error("Module %s refused camera %s: %s", module_id, cam_id[:8], exc)
        store.set_camera_status(cam_id, "error")
        return None

    feed_id = res.get("feed_id")
    if not feed_id:
        store.set_camera_status(cam_id, "error")
        return None
    store.assign_camera(cam_id, module_id, feed_id, "running")
    logger.info("Camera %s (%s) -> module %s as feed %s",
                cam_id[:8], cam["name"], module_id, feed_id[:8])
    return module_id


async def _place_unplaced(app: FastAPI) -> None:
    """Try to place every camera that has no module (new, or orphaned by a death)."""
    store: Store = app.state.store
    for cam in store.cameras():
        if not cam.get("module_id"):
            await _try_place(app, cam["id"])


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
