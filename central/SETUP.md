# Central app — setup

The fleet's brain: camera registry, module registry, placement, event store,
dashboard. It never decodes a frame and needs **no GPU**.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r central/requirements.txt        # fastapi, uvicorn, pydantic only
```

That is the whole dependency list — no torch, no CUDA, no OpenCV. Storage is
stdlib `sqlite3` and outbound HTTP is stdlib `urllib`. If a change ever pulls
torch in here, central stops being deployable on a small VM; keep it out.

## Configure

```bash
cp central/.env.example central/.env     # then edit
```

| Variable | Why it matters |
|---|---|
| `CENTRAL_DB` | SQLite file: camera registry *and* violation history. **Disposable — there are no migrations.** A schema change means deleting it; cameras are re-added and modules re-register. |
| `CENTRAL_TOKEN` | Shared secret modules must present. Empty = any host can register as a module. |
| `CENTRAL_STRIDE` | Must match `frame_stride` in `module/core/config.py`, or placement miscalculates capacity. |
| `CENTRAL_STALE_AFTER` | Seconds of silence before a module is declared dead and its cameras are re-placed. |

## Run

```bash
uvicorn central.app:app --env-file central/.env --host 0.0.0.0 --port 9000
```

Dashboard at `http://<host>:9000/`, video wall at `http://<host>:9000/wall`
(responsive: 3 tiles per row on a laptop, 5 on a 4K wall, or pick a count). Expect this on a clean start:

```
Central up. db=central/central.db token=set stride=1
```

`token=NOT SET (open)` means `CENTRAL_TOKEN` is empty.

## Verify

```bash
curl localhost:9000/api/fleet      # {"modules_total":0, "cameras_total":0, ...}
```

Zero modules is correct until one registers — central never dials out to find
them; modules make contact. Start a module (see `module/SETUP.md`) and it appears
within a few seconds.

Full contract test, no GPU needed:

```bash
python tests/test_central.py       # stub module: register, place, ingest, reap, spool
```

## Adding cameras

On the dashboard: paste the URL, then either **quick add** (straight in) or
**Preview** — which fetches one still frame so you can draw a tripwire line or an
intrusion zone *before* the camera starts, so it never runs unconfigured. A module
dropdown pins the camera to a specific host; leave it on Auto for most headroom.

Or by API:

```bash
curl -X POST localhost:9000/api/cameras \
  -H 'Content-Type: application/json' \
  -d '{"url":"rtsp://user:pass@10.0.1.50/stream1","name":"Floor 2 East"}'
```

The response includes `"placed"`. **`"placed": null` is a real answer, not an
error** — it means no online module had headroom, and the fix is to deploy another
module, not to retry. The camera stays registered and is placed automatically as
soon as capacity appears.

## Pinning

`POST /api/cameras {module_id: "gpu-host-2"}` forces a camera onto one module. A pin
is **never silently overridden**: if that module is offline the camera stays unplaced
with status `waiting for pinned module`, because people pin for reasons central
cannot see (usually network reachability) and relocating it would break the camera
with nothing explaining why.

## Scaling

Deploy another module. It registers itself, central sees the headroom, cameras
start landing on it. Nothing here is edited and central is not restarted.

## API

| Route | Purpose |
|---|---|
| `GET /api/fleet` | capacity view: per-module commitment and headroom |
| `GET/POST /api/cameras`, `DELETE /api/cameras/{id}` | camera registry |
| `POST /api/cameras/{id}/geometry` | zone/line, persisted and pushed live to the module |
| `POST /api/cameras/probe` | one still frame, to draw geometry before adding |
| `POST /api/cameras/{id}/stream` | short-lived WebSocket URL for live video |
| `GET /api/events?limit=&camera_id=` | violation history |
| `POST /api/modules/register`, `/api/modules/{id}/heartbeat`, `/api/events` | the module contract (token-checked) |

## Things to know

**Video never passes through central, and the detection module serves none.**
Playback is a separate deployable (`streamer/`) that decodes
independently at the source frame rate — display used to be capped by *detection*
fps, which made smooth video impossible. `POST /api/cameras/{id}/raw` returns a
short-lived WebSocket URL on a streamer; the browser connects to it directly.
Cameras show `playable: true` when **any** streamer is online — it need not be on the
host detecting them, which is what allows GPU-less video hosts.

**Cameras are sticky to a module.** Track identities and the violation debounce
state machine are sequential per feed, so a camera cannot be moved mid-stream
without resetting both. Central places once, and only re-places when a module dies
— at which point tracking for those cameras restarts.

**Event delivery is at-least-once.** Modules spool to disk and retry, so a network
blip delays alerts rather than losing them; the trade is that a retry after a
partial failure can duplicate. Losing a violation is worse than showing one twice.

**No auth on the camera-admin routes yet** — only the module contract checks a
token. Don't expose central to an untrusted network as-is.
