"""Central app — the fleet's brain. Owns everything except frames.

Responsibilities: the camera registry (users add RTSP links here), the module
registry and health, placement, the event store, and the dashboard.

It never decodes a frame, never loads a model, and needs no GPU — which is why
``central/requirements.txt`` shares nothing with ``module/requirements.txt``.
Video goes straight from a module to the browser; central only hands out the URL.

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

    online = [m for m in app.state.store.modules(STALE_AFTER) if m["online"]]
    if not online:
        return JSONResponse(
            {"error": "no module is online to open the stream — start a module first"},
            status_code=503,
        )

    # Any module can probe; geometry is normalized so it stays valid wherever the
    # camera is eventually placed. Try each so one sick module doesn't block preview.
    last = ""
    for mod in online:
        try:
            res = await ModuleClient(mod["url"]).probe(url)
            res["probed_by"] = mod["id"]
            return JSONResponse(res)
        except ModuleError as exc:
            last = str(exc)
            logger.warning("Probe via %s failed: %s", mod["id"], exc)
    return JSONResponse({"error": f"could not preview stream: {last}"}, status_code=502)


@app.post("/api/cameras")
async def add_camera(payload: dict) -> dict:
    """Register a camera and place it on a module with headroom."""
    url = (payload.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "missing 'url'")
    name = (payload.get("name") or url).strip()
    geometry = payload.get("geometry")
    cam_id = app.state.store.add_camera(name, url, geometry)
    placed = await _try_place(app, cam_id)
    return {"camera_id": cam_id, "name": name, "placed": placed}


@app.get("/api/cameras")
async def list_cameras() -> dict:
    """Every camera, with its module and the direct video URL for the browser."""
    store: Store = app.state.store
    mods = {m["id"]: m for m in store.modules(STALE_AFTER)}
    out = []
    for cam in store.cameras():
        row = dict(cam)
        mod = mods.get(cam["module_id"]) if cam["module_id"] else None
        row["module_online"] = bool(mod and mod["online"])
        row["video_url"] = (
            ModuleClient(mod["url"]).video_url(cam["feed_id"])
            if mod and cam.get("feed_id") else None
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

    module_id = choose_module(store.modules(STALE_AFTER), store.cameras(),
                              cam, FRAME_STRIDE)
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
