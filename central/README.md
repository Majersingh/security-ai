# Central app

The fleet's brain. Owns everything **except** frames.

- camera registry — the user adds RTSP links here, not on a module
- module registry + health — which analysis modules exist, their capacity, liveness
- placement — which module owns which camera
- event store — violations from every module, in one database
- dashboard — one pane of glass across the fleet

**It never touches a frame, never loads a model, and needs no GPU.** That is why
`central/requirements.txt` shares nothing with `module/requirements.txt`: this
deploys as a small container anywhere, while the module carries the ~3 GB CUDA
stack. Keep it that way — the moment central imports torch, it stops being
deployable on a cheap VM.

## Scaling model

Adding capacity = **deploy another module**. A module announces itself on startup;
central sees new headroom and starts assigning cameras to it. Nothing on central
is edited, no ports are coordinated, no camera lists are maintained by hand.

```
                    ┌──────────── CENTRAL (this) ────────────┐
                    │  registry · placement · events · UI    │
                    └────────────────────────────────────────┘
                       ▲ register/heartbeat/events   │ assign/stop/geometry
   ┌───────────────────┴──────────┬──────────────────┴─────────┐
 MODULE 1 (1 GPU, ~30 cams)   MODULE 2 (1 GPU)          MODULE 3 (1 GPU)
   │
   └──── raw video (separate process on the host) ──▶ browser  (watched tiles only)
```

## Why the module always initiates the connection

| Direction | Works across sites? |
|---|---|
| Module → central | ✅ outbound HTTPS/WSS — works through NAT and firewalls |
| Central → module | ❌ usually blocked when a module sits at a client site |

**As built:** the module initiates registration, heartbeats and event delivery, and
central calls back over REST for assignments. That works today because every host is
publicly reachable.

**The known limit:** the first time a module sits behind NAT at a client site,
central will not be able to dial it. The fix is to carry commands down a WebSocket
the *module* opens — `central/module_client.py` is deliberately the only place
central makes an outbound call, so that becomes a contained change rather than a
redesign. Not built yet; nothing depends on it while hosts are reachable.

## The contract

**Module → central**

| Endpoint | When | Carries |
|---|---|---|
| `POST /api/modules/register` | startup | id, public URL, **raw_url**, GPU, `max_feeds`, `fps_budget` |
| `POST /api/modules/{id}/heartbeat` | ~10 s | active feeds, per-camera status |
| `POST /api/events` | on violation | batched events, spooled and retried |

**Central → module** (REST, today): `POST /feeds/stream` — carrying central's own
`camera_id` so reported events are attributable — plus `/feeds/{id}/stop`,
`/feeds/{id}/geometry`, `/feeds/probe`, `GET /feeds`, and
`POST /stream/raw/ticket` on the host's separate raw-video service.

**Browser → the host's raw-video service, directly:** `WS /stream/raw?ticket=…`,
obtained via `POST /api/cameras/{id}/raw`. Frames never pass through central. The
detection service serves no video at all — display fps was otherwise capped by
detection fps — and the RTSP URL never reaches the browser, only a short-lived
ticket.

## Two things that will bite

**Event durability.** Modules spool events to local disk and drop nothing until
central acknowledges, so a blip delays alerts instead of losing them — losing a
violation is losing the product. The trade is at-least-once delivery: a retry after
a partial failure can duplicate. Verified by `tests/test_central.py` check 6.

**Cameras are sticky to a module.** ByteTrack IDs and the violation debounce state
machine are sequential per feed, so a camera cannot move mid-stream without losing
identity. Placement assigns once; on module failure central reassigns and tracking
restarts from scratch. Acceptable for CCTV, but state it rather than discover it.

## Video transport — decided

Hosts are publicly reachable, so the browser pulls video **directly** from each
host's raw-video service. Central hands out a short-lived ticket and nothing else.

If a host later sits behind NAT the browser cannot reach it either, and watched
tiles would have to be relayed through central (or WebRTC + TURN). That costs
central bandwidth per viewed tile, which is exactly what the direct path avoids —
so it is worth deferring until a deployment actually requires it.

## Status

Implemented: module registration and heartbeats, capacity-aware placement with
strict pinning, event ingestion with a durable module-side spool, stale-module
reaping, the dashboard, the video wall, camera preview with geometry drawing, and
raw playback via ticketed WebSockets.

Not done: authentication anywhere, reports/retention/audit, and any supervision of
the module processes. See `docs/ARCHITECTURE.md` §7.

Tests: `tests/test_central.py` (stub module), `tests/test_wired.py` (real module),
`tests/test_pinning.py`, `tests/test_preview.py`, `tests/test_central_light.py`.
