# Multi-Feed Streaming Architecture

Design for scaling the operator-monitoring system from **one uploaded video at a
time** to **many concurrent feeds** — streamed uploads today, live stream URLs
(RTSP/HLS) later — all on **a single powerful machine with one GPU**.

Status: **Stream-only** (production shape). The system now processes **live
stream URLs only** — the upload-video and live-camera (webcam) features were
removed, since real deployments only have camera stream links. Each stream is a
concurrent `Feed` on one GPU, viewable in the dashboard, with **runtime zone/line
drawing** (draw on a connected stream; geometry is applied live).

History: this grew through Phases 1–6 (in-memory upload scan → multi-feed →
dashboard → stream URLs → worker processes → centralized inference). The
upload/camera paths (`/ws-scan`, `/ws-live`, `InMemoryVideoSource`) have since been
deleted; that build history is kept below for context, but the shipping surface is
the `/feeds/*` API only. Where an early phase's reasoning has been superseded, the
section says so rather than being quietly left in place.

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

- **One machine**, generous RAM / storage / a single capable GPU (RTX 4090 class).
- Concurrency is bounded by **CPU** — per-feed decode, tracking, annotation and
  JPEG encode — long before VRAM or GPU compute. One `yolo11n.pt` weight is 5.6 MB
  and a single batched model serves every feed, so VRAM is not the ceiling.

> Revised. This section previously claimed the bound was VRAM "not CPU", which was
> true only under the retired one-model-per-feed design. Measured on the current
> pipeline, the CPU-side work dominates; see §3.

---

## 3. The two constraints that drive the design

1. **Identity tracking is per-feed; detection is not.**
   Ultralytics keeps tracker state *on the model's predictor*, so `model.track(…,
   persist=True)` cannot be shared across feeds. That once forced a model per
   feed. It no longer does: detection runs **statelessly** on one shared model,
   and each feed keeps its own `supervision.ByteTrack`
   (`src/streaming.py::FrameProcessor`) for identity. Track IDs stay isolated
   without duplicating weights.
   → **One model for all feeds; one tracker per feed.**

2. **One GPU is a serial, shared resource — so exactly one process owns it.**
   Frames from all feeds are combined into a single batched `predict()` call in a
   dedicated inference process (`src/inference.py`). That gives one CUDA context,
   one copy of the weights, and batches deep enough to be worth batching.
   → **A single inference process, not a semaphore around N models.**

   The retired design gated N per-feed models with an `asyncio.Semaphore`. When
   batching arrived it kept a batcher *per worker process*, so with N workers each
   batcher saw 1/N of the frames — N CUDA contexts and batches too small to help.

**Consequence:** the feed ceiling is CPU throughput for decode + post-processing.
The knobs that actually move it are `frame_stride`, `inference_imgsz`, `hw_decode`
(NVDEC) and `num_workers` — not `max_feeds`.

---

## 4. Core abstraction: everything is a `Feed`

Every input — uploaded file, RTSP camera, HLS stream, webcam — becomes a `Feed`
with three parts. Only the **source** and the **drop policy** differ; everything
downstream of a decoded frame is identical.

```
   coordinator (web process)          worker process 0..N-1        inference process
   ┌───────────────────────┐        ┌──────────────────────┐      ┌──────────────┐
   │ WorkerPool            │  ctrl  │ FeedManager          │      │ ONE model    │
   │  feed_id -> worker    │ ─────▶ │  Feed 0..k           │      │ ONE batch    │
   │  relays frames/events │        │   FrameSource (NVDEC)│      │ queue        │
   │  GET /feeds, WS subs  │ ◀───── │   FrameProcessor     │      │ FP16 on CUDA │
   └───────────────────────┘ result │    ByteTrack + rules │      └──────────────┘
                                    │    annotate + encode │         ▲        │
                                    └──────────────────────┘         │        │
                                       frames (shared-memory ring) ──┘        │
                                       detections ◀──────────────────────────-┘
```

Stages are split by what they contend on: decode and post-processing are
CPU-bound and scale with **process count**; detection is GPU-bound and is
centralized so batches actually fill. Each process caps its own OpenCV/torch
thread pools (`cv_threads`, `torch_threads`) so N processes don't each try to
claim the whole machine.

`num_workers = 0` collapses all of this into the web process with an in-process
batcher (`LocalInferencer`) and no shared memory — the simple path, and the only
sensible one on a CPU-only host.

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

### 4.2 `FrameProcessor` — the per-feed CV core

`src/streaming.py::FrameProcessor` is a stateful, single-frame, source-agnostic
pipeline, **one instance per feed**. It holds everything that must *not* be shared
between feeds — `supervision.ByteTrack` identities, the rule state machine, the
event log, the annotator — and is handed detections from the shared model. It owns
no model. `detections=None` is treated as "nothing detected this frame", so a
failed batch degrades to a quiet frame instead of killing the feed.

Use `process_json()` for the live preview: it returns boxes + events as ~1 KB of
JSON, no image. The browser draws boxes over its own local video/stream. Attach
an annotated JPEG (via the existing encode path) **only on frames that fired an
event**, for the evidence thumbnail. This is the main bandwidth/latency win.

### 4.3 `Feed` (new) — one running pipeline

Bundles a `FrameSource` + a `FrameProcessor` + state
(`feed_id`, `status`, geometry, recent events). Its loop:

```python
async for frame, idx in source:
    detections = await inferencer.infer(frame)          # shared model, batched
    payload = await run_in_executor(post_pool, proc.process_json, frame, idx,
                                    detections)         # this feed's own state
    await sink.publish(feed_id, idx, payload)
```

Annotation and JPEG encoding happen **only while a viewer is attached** (the
`view` control message), and then at no more than `viewer_max_fps`. Detection and
events run regardless, so an unwatched feed costs no encode. This is the single
biggest CPU saving in the pipeline.

### 4.4 `FeedManager` — the per-process registry + inference handle

- Owns the registry `feed_id -> Feed` for its process.
- Holds one inference handle: an `InferenceClient` (GPU in another process) or a
  `LocalInferencer` (`num_workers = 0`). Both expose `await infer(frame)`, so
  `FeedManager` does not know which it has.
- Owns the small post-processing thread pool (track/rules/annotate/encode). Kept
  modest deliberately: parallelism comes from process count, and each process is
  thread-capped.
- Enforces `max_feeds`.

### 4.4b Inference service (`src/inference.py`)

- `ModelRunner` — the one place the model is loaded and called. Pads every batch
  up to a power-of-two size and pre-warms each, so cuDNN autotunes a handful of
  fixed shapes instead of re-tuning per batch (~1s+ each, stalling every feed).
- `FrameRing` — pool of fixed-size shared-memory slots. A 1080p BGR frame is
  ~6 MB; pickling that through a queue at hundreds of fps would cost more than the
  inference it feeds. The **consumer** returns the slot after reading, which is
  what stops a producer from overwriting a frame still in flight. Detections are
  converted to plain arrays *before* slots are released, because Ultralytics keeps
  a reference to the source array in `Results.orig_img`.
- Fallback: a frame too large for a slot, or arriving when the pool is momentarily
  empty, is carried inline in the request message. Slower, never wrong.
- Responses are ~KB (boxes/conf/class), so they travel as ordinary queue messages.

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
# throughput knobs, roughly in order of impact
frame_stride: int = 1            # process every Nth frame — see the note below
inference_imgsz: int = 1280      # cost scales ~quadratically
hw_decode: bool = True           # NVDEC; falls back to software automatically
num_workers: int = physical_cores()   # capped at 8; 0 = everything in-process
cv_threads: int = 1              # per-process intra-op caps, so N procs don't
torch_threads: int = 1           #   each size their pools from the whole box

# inference service
batch_max_size: int = 16         # frames combined into one GPU call
batch_max_wait_ms: int = 12      # how long to wait to fill a batch
infer_slots: int = 64            # shared-memory frame slots
infer_slot_max_height: int = 1088
infer_slot_max_width: int = 1920

max_feeds: int = 80              # admission limit, NOT a VRAM ceiling
```

`inference_imgsz = 1280` is deliberate: on the sample CCTV clip, 1280 detects both
the person and the phone where 640/320 finds only the person. It is also ~4x the
cost of 640, so it and `frame_stride` are the two dials that set the feed ceiling.

**Aggregate detection demand = feeds × (source_fps / frame_stride).** With
`frame_stride = 1` at 25 fps, 80 feeds ask for ~2000 fps of detection, which is
several times what one 4090 can deliver at imgsz 1280. `max_feeds = 80` is
therefore an admission limit, not a throughput promise — raising `frame_stride` is
what makes a high feed count real. Monitoring rules (phone usage, zone, line) do
not need every frame.

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
2. ✅ **`Feed` + `FeedManager`** (`src/feeds.py`). Concurrent feeds, per-feed
   `output/<feed_id>/` dirs, `GET /feeds`, `POST /feeds/{id}/stop`.
   Verified: two concurrent uploads, distinct feed_ids, registry drains cleanly.
   *(Originally model-per-feed behind an `asyncio.Semaphore` GPU gate; both are
   retired — see phase 6.)*
3. ✅ **Multi-feed dashboard** (`web/static/index.html`). Upload tab is a grid:
   each selected video opens its own `/ws-scan` feed tile (own WebSocket, local
   `<video>` scrubbed in lockstep, per-tile box overlay), with per-tile Stop
   (`POST /feeds/{id}/stop`), a shared aggregated events panel (Feed column), and
   an "active feeds" summary polled from `GET /feeds`. Live Camera tab retained.
   NOTE: zone/line drawing is camera-only for now — upload feeds start with no
   geometry (phone-usage detection is unaffected). Per-tile upload geometry is a
   possible follow-up.
4. ✅ **Stream URL source** (`src/sources.py::StreamURLSource`). PyAV opens
   RTSP/HLS/HTTP URLs and decodes **sequentially, no drop, paced to the source
   fps** — smooth playback with contiguous frame numbers (the RTX 4090 keeps up
   at 640px). It auto-detects **live** (`rtsp://`/`rtmp://`/`.m3u8`/`.mpd` →
   unbounded, `total_frames=0`, auto-reconnect on EOF) vs **finite file** (plain
   `.mp4` etc. → plays once, real `total_frames`, correct progress/timeline).
   Registered via `POST /feeds/stream {url}` as a background feed; viewers attach
   via `WS /feeds/{id}/subscribe`. Because streams have no local browser copy,
   stream feeds emit **server-annotated JPEG frames**; `Feed` has a
   broadcast/subscribe model (`emit_image`). The dashboard renders each frame onto
   a `<canvas>` (double-buffered, no flicker), with per-tile line/zone drawing.

   (Earlier used background-thread drop-to-latest; switched to sequential+paced —
   drop-to-latest fought HLS's segment bursts and made playback choppy. If a live
   feed ever falls behind on a weaker GPU, latency grows rather than dropping —
   acceptable on the 4090 target; revisit with a bounded queue if needed.)

5. ✅ **Multiprocess workers** (`src/workers.py`). The web process becomes a
   coordinator; N worker processes run the per-feed pipeline and relay
   frames/events over `multiprocessing` queues. Selective encoding (`view`) means
   an unwatched feed costs no annotate/JPEG.

6. ✅ **One inference process + thread caps + NVDEC** (`src/inference.py`,
   `src/utils.py`, `src/sources.py`). Consolidated onto a single inference design
   and deleted the alternative:
   - `src/detector.py` and `src/batch.py` **removed**; `max_concurrent_inferences`
     and `batched_inference` removed from config. There is now one way to run the
     model.
   - Detection centralized in one process; frames travel over a shared-memory
     `FrameRing`. Previously each worker had its own batcher (N contexts, batches
     1/N as deep).
   - `cv_threads`/`torch_threads` capped per process; `num_workers` sized to
     *physical* cores. On a 4-logical/2-physical box this took the worst case from
     ~56 potential compute threads to ~2 workers × 1 thread.
   - NVDEC hardware decode with automatic software fallback, probed once per
     process.

   Verified on CPU-only hardware (2 physical cores, imgsz/stride lowered for the
   test only): shared-memory ring round-trips across processes with oversize and
   pool-exhaustion fallbacks; in-process mode produces boxes and a terminal
   `done`; coordinator mode with 1 worker runs a 964-frame file to completion and
   propagates `done`; coordinator mode with 2 workers + 1 inference process keeps
   two feeds at ~real time with distinct workers and no error payloads.
   **Not yet verified on GPU:** NVDEC actually engaging, and FP16 batch throughput.

Live streams note: event timestamps/debounce use `seq/fps` (seq = decoded-frame
count), which ≈ wall-clock when the GPU keeps up with the stream; on a box that
can't, sustained-violation thresholds trigger slightly late. Fine for the target
GPU host; revisit with wall-clock timing if needed.

(Durable/SQLite restart recovery was dropped: uploads are ephemeral by design —
nothing to recover but live-stream reconnection, which lives in Phase 4.)

---

## 8. Trade-offs & risks

- **`supervision.ByteTrack` is deprecated** (removed in supervision 0.30) and the
  shared-model design *depends* on it for per-feed identity. `requirements.txt`
  pins `supervision>=0.25,<0.30` for exactly this reason. This is the real debt of
  choosing one shared model: before supervision 0.30, either vendor a ByteTrack or
  move identity tracking in-house. Tracked as an open question below.
- **Shared memory costs RAM up front**: `infer_slots × slot_h × slot_w × 3` =
  ~400 MB at the defaults. Cheap on the target box; lower `infer_slots` if not.
- **A frame larger than a slot silently takes the slow path** (inline pickling). At
  4K, *every* frame would, and throughput would quietly drop. Raise
  `infer_slot_max_*` if 4K cameras arrive.
- **The inference process is a single point of failure.** If it dies, every feed
  stalls — its batches fail and each feed sees an exception per frame. There is no
  supervisor restarting it yet.
- **`num_workers > 1` only pays off if post-processing is the bottleneck.** With
  detection centralized, more workers add decode/annotate parallelism but also
  more IPC. Measure before raising it.
- **ffmpeg/libav is an explicit dependency** — pin it in requirements/Docker.
- **Exotic codecs** may make ffmpeg buffer before the first frame; MJPEG output
  bounds this, but test with real CCTV samples.
- **One ffmpeg process per active feed** — fine for a handful; `max_feeds` caps it.

---

## 9. Open questions

- **What replaces `supervision.ByteTrack` before supervision 0.30?** The one hard
  dependency of the shared-model design. Options: pin indefinitely, vendor
  ByteTrack, or write the identity layer in-house.
- **What `frame_stride` do the rules actually need?** This sets the real feed
  ceiling (§5) and nothing else moves it as cheaply. Needs a measurement on real
  footage: at what analysis fps do phone-usage/zone/line detections start being
  missed?
- Target concurrent feed count, and the measured detection fps of the deployment
  GPU at `inference_imgsz = 1280` (the two numbers that decide whether 80 feeds is
  reachable)?
- For live streams later: expected count and protocol mix (RTSP vs HLS)?
- Retention: how long to keep uploaded videos and event snapshots on disk?
