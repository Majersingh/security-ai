"""Which module should own a new camera.

Two independent budgets have to fit, and ignoring either one produces a module
that accepts cameras it cannot actually serve:

* **feed slots** — the module's own ``max_feeds`` admission limit.
* **detection throughput** — measured aggregate fps on that module's GPU, divided
  by what each camera demands (``source_fps / frame_stride``). This is the budget
  that silently overcommits: a module will happily accept a 40th camera and then
  every feed on it falls behind together.

Placement is *sticky*: a camera stays on its module for its lifetime, because
ByteTrack identities and the violation debounce state machine are sequential per
feed. Rebalancing a running camera would reset both, so we only place on create
and on module loss.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("central")


def camera_demand_fps(camera: Dict[str, Any], frame_stride: int) -> float:
    """Detection fps a camera will ask of its module."""
    fps = float(camera.get("source_fps") or 30.0)
    return fps / max(1, int(frame_stride))


def module_load(module: Dict[str, Any], cameras: List[Dict[str, Any]],
                frame_stride: int) -> Dict[str, Any]:
    """Current commitment of one module, plus what it has left."""
    mine = [c for c in cameras if c.get("module_id") == module["id"]]
    used_fps = sum(camera_demand_fps(c, frame_stride) for c in mine)
    budget = float(module.get("fps_budget") or 0.0)
    return {
        "module_id": module["id"],
        "online": module.get("online", False),
        "cameras": len(mine),
        "slots_total": int(module.get("max_feeds") or 0),
        "slots_free": max(0, int(module.get("max_feeds") or 0) - len(mine)),
        "fps_budget": budget,
        "fps_used": round(used_fps, 1),
        "fps_free": round(budget - used_fps, 1) if budget else None,
    }


def choose_module(modules: List[Dict[str, Any]], cameras: List[Dict[str, Any]],
                  new_camera: Dict[str, Any], frame_stride: int) -> Optional[str]:
    """Pick the online module with the most headroom, or None if none fits.

    Returning None is a real answer — it means the fleet is full and the honest
    response is to tell the user to deploy another module, not to overload one.
    """
    need = camera_demand_fps(new_camera, frame_stride)
    candidates = []
    for m in modules:
        if not m.get("online"):
            continue
        if m.get("role") == "streamer":
            continue          # video-only host: serves tiles, never runs detection
        load = module_load(m, cameras, frame_stride)
        if load["slots_free"] <= 0:
            continue
        # Only enforce the fps budget when the module reported one.
        if load["fps_free"] is not None and load["fps_free"] < need:
            continue
        candidates.append(load)

    if not candidates:
        return None
    # Most free throughput first; fall back to most free slots when no budget known.
    candidates.sort(key=lambda l: (l["fps_free"] if l["fps_free"] is not None
                                   else float(l["slots_free"])), reverse=True)
    return candidates[0]["module_id"]


def fleet_summary(modules: List[Dict[str, Any]], cameras: List[Dict[str, Any]],
                  frame_stride: int) -> Dict[str, Any]:
    loads = [module_load(m, cameras, frame_stride) for m in modules
             if m.get("role") != "streamer"]
    online = [l for l in loads if l["online"]]
    return {
        "modules_total": len(loads),
        "modules_online": len(online),
        "cameras_total": len(cameras),
        "cameras_placed": sum(1 for c in cameras if c.get("module_id")),
        "cameras_unplaced": sum(1 for c in cameras if not c.get("module_id")),
        "fps_budget_total": round(sum(l["fps_budget"] for l in online), 1),
        "fps_used_total": round(sum(l["fps_used"] for l in online), 1),
        "modules": loads,
    }
