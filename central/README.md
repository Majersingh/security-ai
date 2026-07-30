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
   └──── annotated video ────▶ browser  (watched tiles only, never via central)
```

## Why the module always initiates the connection

| Direction | Works across sites? |
|---|---|
| Module → central | ✅ outbound HTTPS/WSS — works through NAT and firewalls |
| Central → module | ❌ usually blocked when a module sits at a client site |

So central must **never** need to dial a module. The module opens a persistent
WebSocket to central and keeps it open; central pushes assignments down that
existing socket. The live connection doubles as the heartbeat — socket closed
means module down. This is the single most important constraint in the design; a
REST-only "central calls module" contract works on one LAN and then fails the
first time a module is deployed off-site.

## The contract

**Module → central**

| Endpoint | When | Carries |
|---|---|---|
| `POST /api/modules/register` | startup | module id, public URL, GPU name, `max_feeds`, measured fps budget |
| `WS   /api/modules/{id}/link` | held open | heartbeat up, commands down |
| `POST /api/events` | on violation | batched events + snapshot references |

**Central → module** (over the WebSocket above, not a fresh connection)
`assign_feed`, `stop_feed`, `set_geometry`, `set_view`.

These map onto endpoints the module already has: `POST /feeds/stream`,
`POST /feeds/{id}/stop`, `POST /feeds/{id}/geometry`, `GET /feeds`.

**Browser → module, directly:** `WS /feeds/{id}/subscribe`. Central only hands the
browser the module URL. Frames never pass through central — that is what stops
central becoming the bandwidth bottleneck, and it costs nothing because encoding
already only happens for tiles someone is watching.

## Two things that will bite

**Event durability.** Modules must spool events to local disk and retry, not
fire-and-forget. If central is briefly unreachable and events are dropped, you
lose violations — which is the entire product. Build this into the module's
reporting client from the start.

**Cameras are sticky to a module.** ByteTrack IDs and the violation debounce state
machine are sequential per feed, so a camera cannot move mid-stream without losing
identity. Placement assigns once; on module failure central reassigns and tracking
restarts from scratch. Acceptable for CCTV, but state it rather than discover it.

## Open decision, blocking video transport

Are the GPU hosts, central, and the operators' browsers on **one network**, or at
**separate sites**?

- one network → browser pulls video directly from each module. Simplest, cheapest.
- separate sites → the module is behind NAT and the browser cannot reach it, so
  watched tiles must be relayed through central over the module's existing
  WebSocket (or WebRTC + TURN later).

Everything else in this design is identical either way.

## Status

Not implemented yet. Suggested order:

1. **Module-side reporting** (`module/reporting.py`): register, heartbeat link,
   event spool + forward. Testable against a stub central, no CV changes.
2. **Central skeleton**: registry + module registry + event ingestion + SQLite.
   Placement can start dumb (least-loaded).
3. **Move the dashboard here**, pointing video at module URLs.
4. **Deploy module #2** and verify a camera added in central lands on it.

Step 1 is useful even with a single module: events in a real database instead of
per-feed CSVs scattered across hosts.
