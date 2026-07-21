# Multi-Feed Streaming Architecture

Design for scaling the operator-monitoring system from **one uploaded video at a
time** to **many concurrent feeds** — streamed uploads today, live stream URLs
(RTSP/HLS) later — all on **a single powerful machine with one GPU**.

Status: **Stream-only** (production shape). The system now processes **live
stream URLs only** — the upload-video and live-camera (webcam) features were
removed, since real deployments only have camera stream links. Each stream is a
concurrent `Feed` on one GPU, viewable in the dashboard, with **runtime zone/line
drawing** (draw on a connected stream; geometry is applied live).

History: this grew through Phases 1–4 (in-memory upload scan → multi-feed + GPU
gate → dashboard → stream URLs). The upload/camera paths (`/ws-scan`, `/ws-live`,
`InMemoryVideoSource`) have since been deleted; that build history is kept below
for context, but the shipping surface is the `/feeds/*` API only.

Endpoints (current): `GET /feeds`, `POST /feeds/stream {url}`,
`POST /feeds/{id}/stop`, `POST /feeds/{id}/geometry {zone,line}` (runtime rule
update), `WS /feeds/{id}/subscribe` (annotated frames + events).

Two decisions were made during build and supersede the original ffmpeg sketch:
- **Decode backend = PyAV** (`av`, pip-installed in `.venv`). No system `ffmpeg`
  is installed; OpenCV can't decode a pipe/in-memory stream. PyAV decodes from
  RAM and (later) opens RTSP/HLS URLs directly.
- **Nothing is stored except snapshots + `events.csv`.** Uploaded bytes live in
  RAM only — never written to `uploads/` — and no annotated video is produced.

---

## 1. Goals & non-goals

**Goals**
- Process video **as it uploads** (start inference on the first frames while the
  rest of the file is still arriving) instead of the current upload-then-process.
- Run **multiple feeds at once** on one box, sharing one GPU safely.
- Make a **live stream URL** (RTSP/HLS/HTTP) a drop-in new source later, with no
  changes to the detection/tracking/behaviour code.
- **Robust on a single box**: survive a server restart and (for live feeds)
  network blips, without external infra (no Redis/queue/object-store).

**Non-goals (for now)**
- Horizontal scaling across multiple machines or GPUs.
- Distributed queues, object storage, autoscaling worker pools.
  (The `Feed` abstraction below leaves the door open to these, but we are not
  building them.)

---

## 2. Deployment target

- **One machine**, generous RAM / storage / a single capable GPU.
- Concurrency is bounded by **VRAM** (number of model instances) and **GPU
  compute** (inference throughput), not by CPU or connections.

---

## 3. The two constraints that drive the design

These come from the existing code and are non-negotiable:

1. **Tracker state lives on the model object.**
   `detector.py` uses `model.track(..., persist=True)` (`src/detector.py:67`).
   Ultralytics keeps ByteTrack/BoT-SORT state *on the model's predictor*, so a
   single `YOLO` instance **cannot** be shared across concurrent feeds — their
   track IDs would corrupt each other.
   → **Each feed owns its own model instance.** (Decision confirmed.)

2. **One GPU is a serial, shared resource.**
   Many feeds can *decode* frames in parallel (cheap, done by ffmpeg on CPU), but
   inference must be **serialized** so feeds don't thrash the GPU and blow up
   latency for each other.
   → **A single global GPU gate (semaphore)** that every feed awaits before it
   calls its model.

**Consequence:** VRAM sets the feed ceiling (one model per feed). On a beefy
GPU with the small `yolo11n.pt` weight this is cheap; `max_feeds` is the knob.

---

## 4. Core abstraction: everything is a `Feed`

Every input — uploaded file, RTSP camera, HLS stream, webcam — becomes a `Feed`
with three parts. Only the **source** and the **drop policy** differ; everything
downstream of a decoded frame is identical.

```
        ┌──────────────── FeedManager (one per process) ────────────────┐
        │  registry: feed_id -> Feed   (durable in SQLite)               │
        │  global GPU_GATE = asyncio.Semaphore(max_concurrent_inferences)│
        │  create / list / stop / reconnect                              │
        └────────────────────────────────────────────────────────────────┘
                   │  each Feed owns:
                   ▼
   ┌── FrameSource ──┐    ┌──── FrameProcessor ────┐    ┌──── Sink ────┐
   │ ffmpeg -i ...   │──▶ │ EXISTING code, one     │──▶ │ WS results   │
   │ (pipe OR url)   │    │ instance PER feed:     │    │ events CSV   │
   │ MJPEG out       │    │ model + tracker + rules│    │ snapshots    │
   └─────────────────┘    └────────────────────────┘    └──────────────┘
                                    │
                             await GPU_GATE      (shared across ALL feeds)
```

### 4.1 `FrameSource` (PyAV) — the universal decoder

`src/sources.py`. PyAV (in-process libav) decodes every source; nothing is
written to disk. Source type only changes what PyAV opens and the drop policy:

| Source                 | PyAV input                                 | Policy            | Status |
|------------------------|--------------------------------------------|-------------------|--------|
| Uploaded file          | `av.open(BytesIO(bytes))` — decode from RAM | **No drop**       | ✅ done |
| RTSP camera *(later)*  | `av.open("rtsp://…")`                       | **Drop to latest**| planned |
| HLS/HTTP *(later)*     | `av.open("https://…/index.m3u8")`           | **Drop to latest**| planned |
| Webcam *(existing)*    | browser pushes frames (unchanged)           | Drop to latest    | ✅ existing |

Implementation notes:
- **`InMemoryVideoSource`** buffers the whole upload in RAM first (a typical MP4
  keeps its `moov` index at the end, so the full file is needed before decode),
  then yields `(raw_frame_index, bgr_frame)`. The **raw** index preserves the
  original timeline so event timestamps and the snapshot cooldown (measured in
  real fps) stay correct; the caller applies `frame_stride`.
- **Backpressure**: because processing is one decode+infer step per frame on the
  event loop's executor, memory stays bounded — we never buffer decoded frames
  ahead of inference.
- **Live drop-to-latest** *(later)*: a camera can't be slowed, so the stream
  reader keeps only the newest decoded frame and discards stale ones to bound
  latency. Not needed for the in-memory upload path.

### 4.2 `FrameProcessor` (existing, unchanged) — the CV core

`src/streaming.py::FrameProcessor` already is a stateful, single-frame,
source-agnostic pipeline. We reuse it as-is, **one instance per feed** (which
gives each feed its own `Detector`/model → correct tracker isolation).

Use `process_json()` for the live preview: it returns boxes + events as ~1 KB of
JSON, no image. The browser draws boxes over its own local video/stream. Attach
an annotated JPEG (via the existing encode path) **only on frames that fired an
event**, for the evidence thumbnail. This is the main bandwidth/latency win.

### 4.3 `Feed` (new) — one running pipeline

Bundles a `FrameSource` + a `FrameProcessor` + state
(`feed_id`, `status`, geometry, recent events). Its loop:

```python
async for frame, idx in source:
    async with GPU_GATE:                       # shared gate: serialize inference
        boxes, events, w, h = await run_in_executor(proc.process_json, frame, idx)
    await sink.publish(feed_id, idx, w, h, boxes, events)
```

### 4.4 `FeedManager` (new) — the single registry + GPU gate

- Owns the registry `feed_id -> Feed` and the **global `GPU_GATE`**
  (`asyncio.Semaphore(max_concurrent_inferences)`, start at 1–2).
- `create_feed(source_spec, geometry)`, `list_feeds()`, `stop_feed(id)`.
- Enforces `max_feeds` (VRAM ceiling).

### 4.5 Durable registry (SQLite, new) — robustness without infra

- Persist feeds + events to SQLite; uploaded videos already live on disk under
  `uploads/`.
- **On restart**: uploaded-file feeds can re-run from the saved file; live-URL
  feeds **auto-reconnect** with backoff. This is the robustness story for a
  single box — no Redis or queue required.

### 4.6 Control API + delivery (rework `web/server.py`)

- `WS /feeds/upload` — stream a file up, get a `feed_id`, receive results on the
  same socket (replaces today's `POST /upload` + `WS /ws/{job_id}`).
- `POST /feeds` — register a live stream URL *(phase 5)*.
- `WS /feeds/{id}/live` — subscribe to a running feed's results.
- `GET /feeds` — dashboard state: all feeds, statuses, recent events.
- Keep the existing `/ws-live` webcam endpoint.

### 4.7 Multi-feed dashboard (rework `web/static/index.html`)

Grid of feed tiles; each draws boxes over its own video/stream using the
existing live-mode renderer, plus a shared events panel.

---

## 5. Config additions (extend `src/config.py::Config`)

```python
max_concurrent_inferences: int = 2       # GPU gate width — tune to the card
max_feeds: int = 8                       # VRAM ceiling (~ one model per feed)
live_drop_to_latest: bool = True         # per-feed; False for file sources
reconnect_backoff_seconds: float = 3.0   # live-stream resilience
```

Keep `inference_imgsz = 1280` for uploaded CCTV (phones are tiny — existing
gotcha), but use `640` for live/webcam sources where subjects are closer. Lower
imgsz directly increases how many feeds fit on the GPU.

---

## 6. Data flow — ephemeral upload scan (`WS /ws-scan`, implemented)

```
Browser                          Server (single process)
  │  WS /ws-scan
  │  {geometry}          ──────▶   _parse_geometry
  │  ◀── {type:ready} ──
  │  file.slice() chunks ──────▶   buffer in RAM (bytearray) — NOT written to disk
  │  "EOF" (text) ──────────────▶  InMemoryVideoSource(bytes)  ← PyAV, from RAM
  │  ◀── {type:meta, fps, total, w, h} ──
  │                                for raw_idx, frame in source.frames():
  │                                    if raw_idx % stride: skip
  │                                    proc.process_json(frame, raw_idx)   ← existing
  │  ◀── {type:frame, i, w, h, boxes, events, progress} ──  (lightweight JSON)
  │                                proc.finalize()  → flush events.csv
  │  ◀── {type:done, total_events, events} ──
```

Artefacts written: **event snapshots + `events.csv`** only. The video buffer is
discarded when the socket closes.

---

## 7. Build order (phased, each independently testable)

1. ✅ **`InMemoryVideoSource` + `WS /ws-scan`** (`src/sources.py`, `web/server.py`).
   Ephemeral upload scan, in-memory PyAV decode, wired into the UI.
2. ✅ **`Feed` + `FeedManager` + GPU gate** (`src/feeds.py`). Concurrent feeds on
   one GPU: model-per-feed (tracker isolation), shared `asyncio.Semaphore` gate,
   per-feed `output/<feed_id>/` dirs, `GET /feeds`, `POST /feeds/{id}/stop`.
   Verified: two concurrent uploads, distinct feed_ids, registry drains cleanly.
3. ✅ **Multi-feed dashboard** (`web/static/index.html`). Upload tab is a grid:
   each selected video opens its own `/ws-scan` feed tile (own WebSocket, local
   `<video>` scrubbed in lockstep, per-tile box overlay), with per-tile Stop
   (`POST /feeds/{id}/stop`), a shared aggregated events panel (Feed column), and
   an "active feeds" summary polled from `GET /feeds`. Live Camera tab retained.
   NOTE: zone/line drawing is camera-only for now — upload feeds start with no
   geometry (phone-usage detection is unaffected). Per-tile upload geometry is a
   possible follow-up.
4. ✅ **Live stream URL source** (`src/sources.py::StreamURLSource`). PyAV opens
   RTSP/HLS/HTTP URLs in a background thread; **drop-to-latest** (keep newest
   frame, skip stale) + **auto-reconnect with backoff**. Registered via
   `POST /feeds/stream {url}` as a background feed (no client driving it); viewers
   attach via `WS /feeds/{id}/subscribe`. Because streams have no local browser
   copy, stream feeds emit **server-annotated JPEG frames**; `Feed` gained a
   broadcast/subscribe model (`emit_image` flag; upload feeds stay boxes-only).
   Dashboard has a stream-URL input; stream tiles render the annotated `<img>`.
   Verified with a simulated stream (drop-to-latest confirmed; upload unaffected).

Live streams note: event timestamps/debounce use `seq/fps` (seq = decoded-frame
count), which ≈ wall-clock when the GPU keeps up with the stream; on a box that
can't, sustained-violation thresholds trigger slightly late. Fine for the target
GPU host; revisit with wall-clock timing if needed.

(Durable/SQLite restart recovery was dropped: uploads are ephemeral by design —
nothing to recover but live-stream reconnection, which lives in Phase 4.)

---

## 8. Trade-offs & risks

- **Model-per-feed uses VRAM per feed.** Correct and simple; VRAM is the cap.
  If feed count ever outgrows VRAM, revisit a shared-model + external-tracker
  design (more code; `supervision.ByteTrack` is deprecated).
- **ffmpeg is now an explicit dependency** — pin it in requirements/Docker.
- **GPU gate width is a latency/throughput dial** — too wide thrashes, too
  narrow starves feeds. Start at 1–2 and measure.
- **Exotic codecs** may make ffmpeg buffer before the first frame; MJPEG output
  bounds this, but test with real CCTV samples.
- **One ffmpeg process per active feed** — fine for a handful; `max_feeds` caps it.

---

## 9. Open questions

- Target concurrent feed count and GPU VRAM (sets `max_feeds`)?
- For live streams later: expected count and protocol mix (RTSP vs HLS)?
- Retention: how long to keep uploaded videos and event snapshots on disk?
