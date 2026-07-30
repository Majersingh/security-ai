# Streamer — setup

Live video playback. One deployable per video host, alongside `central/` (fleet
brain) and `module/` (detection).

It plays video and nothing else: **no model, no GPU, no torch.** It registers itself
with central, so a video-only box needs nothing else running.

## Why it is separate

Two reasons, both measured:

1. **Display fps was chained to detection fps.** The detection pipeline can only show
   a viewer frames it ran detection on, and `frame_stride` discarded the rest before
   they were even converted to BGR. Smooth video was impossible without spending the
   whole GPU on a handful of cameras. This service decodes independently and plays at
   the source rate — measured **28.0 fps from a 28.79 fps source**.
2. **Frame pushing is loud on an event loop.** ~28 `send_json` calls per second *per
   tile*. On the detection app that is the same loop relaying results from the worker
   processes, so a wall of tiles would add latency to the thing the product sells.

The cost is **one extra decode per watched camera**. Decode is cheap next to a GPU
pass, and it stops the moment the tile closes.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r streamer/requirements.txt
```

Five packages: `av`, `opencv-python-headless`, `numpy`, `fastapi`, `uvicorn`. No
torch, no ultralytics — ~150 MB of wheels instead of ~3 GB, and it runs on a box
with no GPU. `tests/test_streamer_light.py` enforces this by importing the service
with the CV stack blocked; if it fails, something crept in that belongs in `module/`.

## Configure

```bash
cp streamer/.env.example streamer/.env      # then edit
```

| Variable | Notes |
|---|---|
| `CENTRAL_URL` | Where to register. Blank = video still works, but central won't know it exists and cameras show `playable: false`. |
| `MODULE_TOKEN` | Must match `CENTRAL_TOKEN` on central. |
| `HOST_ID` | Identifies the **machine**. Set it to the same value as the detection module's `HOST_ID` when both run on one box — central then prefers this co-located streamer, which provably has a route to that host's cameras. |
| `STREAMER_PUBLIC_URL` | Where **browsers** reach this service. The browser connects here directly, so not `127.0.0.1` unless the browser is on this machine, and not a docker-internal hostname. |
| `STREAMER_ID` | Optional; defaults to `<HOST_ID>-streamer`. |

Tuning is all optional and env-driven — see `.env.example`. The one worth knowing is
`STREAMER_MAX_STREAMS` (16): each stream is an extra decode, so this is what stops a
video wall's "all" button from swamping the host.

## Run

```bash
uvicorn streamer.app:app --env-file streamer/.env --host 0.0.0.0 --port 8011
```

Healthy startup:

```
Streamer ready (id=gpu-host-1-streamer, max_streams=16, width<=960, quality=55).
Registering with central http://... as streamer 'gpu-host-1-streamer' (host=gpu-host-1, ...)
Registered with central as streamer 'gpu-host-1-streamer'.
```

Check it:

```bash
curl -s localhost:8011/health
# {"ok":true,"service":"streamer","id":"...","registered":true,"active_streams":0,...}
```

`registered: false` means central hasn't accepted it — check `CENTRAL_URL` and that
`MODULE_TOKEN` matches.

## A video-only host

This is the whole point of the split. On a box with no GPU:

```bash
pip install -r streamer/requirements.txt
CENTRAL_URL=https://central.yourorg.internal \
MODULE_TOKEN=<secret> \
HOST_ID=video-1 \
STREAMER_PUBLIC_URL=http://10.0.1.40:8011 \
uvicorn streamer.app:app --host 0.0.0.0 --port 8011
```

Central will use it for tiles of cameras detected on *other* hosts. Two things must
hold: this box needs a **network route to those cameras** (it opens its own RTSP
connection), and the cameras must tolerate **one more concurrent connection** — the
detecting host is already pulling one. Some IP cameras cap that.

Central never places cameras on a streamer (it registers with `max_feeds: 0` and
`role: "streamer"`), and pinning a camera to one is rejected outright.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /health` | liveness, registration state, active/max streams |
| `POST /stream/ticket` | central swaps a stream URL for a short-lived ticket |
| `WS /stream?ticket=…&fps=&width=` | the browser plays video here |

Browsers get a **ticket, never the URL** — an `rtsp://user:pass@host` in a WebSocket
query string would land in browser history and access logs.

## Notes

**No authentication.** `/stream/ticket` will mint a ticket for any URL handed to it,
so this service will decode whatever it is asked to. Don't expose it to an untrusted
network as-is.

**Shared code** lives in the top-level `core/` package (`sources.py` — the stream
decoder with NVDEC and drop-when-behind — and `procutil.py`). The streamer imports
nothing from `module/`.
