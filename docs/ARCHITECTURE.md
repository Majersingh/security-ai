# Architecture — CCTV operator monitoring

Detects operator behaviour (phone usage, zone intrusion, line crossing) across many
live camera streams, with runtime zone/line drawing, and scales by adding GPU hosts.

Two deployables:

- **`central/`** — the fleet's brain. Camera registry, module registry, placement,
  event store, dashboard. Never touches a frame, needs no GPU.
- **`module/`** — the analysis unit. Owns one GPU, decodes its assigned cameras,
  detects, streams annotated video to browsers, reports events to central.

Setup: `central/SETUP.md`, `module/SETUP.md`.

---

## 1. Scaling model

**Adding capacity = deploying another module.** A module announces itself on
startup; central sees the new headroom and starts placing cameras on it. Nothing on
central is edited, no ports coordinated, no camera lists maintained by hand.

```
                    ┌──────────── CENTRAL ────────────┐
                    │  registry · placement · events  │
                    │  dashboard   (no GPU, no torch)  │
                    └─────────────────────────────────┘
                       ▲ register/heartbeat/events   │ assign/stop/geometry
   ┌───────────────────┴──────────┬──────────────────┴─────────┐
 MODULE 1 (1 GPU, ~30 cams)   MODULE 2 (1 GPU)          MODULE 3 (1 GPU)
   │
   └──── annotated video ────▶ browser  (watched tiles only, never via central)
```

The scaling unit is a **GPU host**, not a camera and not a container. Adding
containers to a saturated box buys nothing — they contend for the same GPU and the
same cores. Adding a host brings a real GPU and real cores.

### Why not a container per camera

It was considered and rejected on measurements. A container per camera means a
model per camera, which forces batch-of-1 forever:

| | batch 1 (container/camera) | batch 4 (shared, measured) |
|---|---|---|
| per frame | 16 ms | 7 ms |
| aggregate | ~63 fps | **~143 fps** |

That is ~2.3x of the GPU thrown away, plus a CUDA context (~300–500 MB VRAM) and
~2–3 GB RSS per camera, plus a model load on every camera add. Thirty cameras would
exhaust a 16 GB card before doing any work. Containerise by **role**, not by camera.

---

## 2. Deployment target

- Per module: **one machine, one GPU** (RTX 4070 Ti SUPER class), generous cores.
- Central: any small VM or container. No GPU.
- Concurrency per module is bounded by **CPU** (decode, tracking, annotation, JPEG
  encode) and by **GPU detection throughput** — not by VRAM. One `yolo11n.pt` is
  5.6 MB and a single shared model serves every feed.

> Earlier versions of this document claimed VRAM was the ceiling. That was true only
> under the retired one-model-per-feed design. It isn't now.

---

## 3. The constraints that drive the design

**1. Identity tracking is per-feed; detection is not.**
Ultralytics keeps tracker state on the model's predictor, so `model.track(persist=True)`
cannot be shared. That once forced a model per feed. It no longer does: detection
runs **statelessly** on one shared model, and each feed keeps its own
`supervision.ByteTrack` (`module/core/streaming.py`). Track IDs stay isolated
without duplicating weights.
→ **One model for all feeds; one tracker per feed.**

**2. One GPU is a serial resource, so exactly one process owns it.**
Frames from all feeds are combined into a single batched `predict()` in a dedicated
inference process (`module/core/inference.py`): one CUDA context, one copy of the
weights, batches deep enough to be worth batching.
→ **A single inference process, not a semaphore around N models.**

**3. Cameras are sticky to a module.**
ByteTrack identities and the violation debounce state machine are sequential per
feed, so a camera cannot migrate mid-stream without resetting both. Placement
assigns once; central re-places only when a module dies, and tracking restarts for
those cameras.
→ **Sharded, not load-balanced.** No distributed state on the hot path.

**4. Central must never need to dial a module.**
Outbound HTTPS works from behind NAT; inbound usually does not. The module
initiates registration, heartbeats and event delivery. Today central *does* call
back to `/feeds/*` because hosts are publicly reachable; `central/module_client.py`
is the single seam to change when a module lands behind NAT.

---

## 4. Inside a module

```
   coordinator (web process)          worker process 0..N-1        inference process
   ┌───────────────────────┐        ┌──────────────────────┐      ┌──────────────┐
   │ WorkerPool            │  ctrl  │ FeedManager          │      │ ONE model    │
   │  feed_id -> worker    │ ─────▶ │  Feed 0..k           │      │ ONE batch    │
   │  relays frames/events │        │   FrameSource (NVDEC)│      │ queue        │
   │  /feeds/*, WS subs    │ ◀───── │   FrameProcessor     │      │ FP16 on CUDA │
   └───────────────────────┘ result │    ByteTrack + rules │      └──────────────┘
                                    │    annotate + encode │         ▲        │
                                    └──────────────────────┘         │        │
                                       frames (shared-memory ring) ──┘        │
                                       detections ◀──────────────────────────-┘
```

Stages split by what they contend on: decode and post-processing are CPU-bound and
scale with **process count**; detection is GPU-bound and centralized so batches
fill. Every process caps its own thread pools — with N processes on one box, each
sizing its pool from the machine's core count oversubscribes the CPU N-fold.

`num_workers = 0` collapses this into the web process with an in-process batcher
(`LocalInferencer`), no shared memory. The simple path, and the only sensible one
on a CPU-only host.

### 4.1 `FrameSource` (`sources.py`)
PyAV decodes RTSP/HLS/HTTP/file. Asks libav for **CUDA (NVDEC)** decode and falls
back to software automatically, probing once per process. One decode thread per
feed (`decode_threads`), because parallelism comes from feed count — left to itself
libav takes roughly one thread per core *per feed*.

Only frames that will be **used** are converted to BGR: `to_ndarray` is one of the
most expensive per-frame operations, and under a stride most decoded frames are
discarded. Decode itself cannot be skipped (H.264 frames reference each other).

Drop-when-behind: a feed more than `stream_max_lag_seconds` behind schedule skips
frames without resyncing, so the backlog shrinks until it is live again. Measured
with a 200 ms/frame consumer on a 25 fps source: 2.3 s of video per 12 s wall-clock
before the fix (5x slow motion), 11.4 s per 12 s after.

### 4.2 `FrameProcessor` (`streaming.py`)
Per-feed state that must **not** be shared: ByteTrack identities, the rule state
machine, the event log, the annotator. Owns no model; detections are handed to it.
`detections=None` means "nothing this frame", so a failed batch degrades to a quiet
frame rather than killing the feed.

### 4.3 `Feed` / `FeedManager` (`feeds.py`)
```python
async for frame, idx in source:
    detections = await inferencer.infer(frame)       # shared model, batched
    payload = await run_in_executor(post_pool, proc.process_json, frame, idx,
                                    detections)      # this feed's own state
```
`FeedManager` holds one inference handle — an `InferenceClient` (GPU elsewhere) or a
`LocalInferencer` — and does not know which. Annotation and JPEG encoding run
**only while a viewer is attached**, capped at `viewer_max_fps`; detection and
events run regardless. That is the single biggest CPU saving in the pipeline, and it
means video cost scales with *viewers*, not cameras.

Each feed's source pull gets its own thread: `frames()` sleeps there for real-time
pacing, and on a shared pool that sleeping thread starves latency-critical work.

### 4.4 Inference service (`inference.py`)
- **`ModelRunner`** — the only place the model is loaded and called. Pads batches to
  power-of-two sizes and pre-warms each, so cuDNN autotunes a handful of fixed
  shapes instead of re-tuning per batch (~1 s+ each, stalling every feed).
- **`FrameRing`** — pool of fixed-size shared-memory slots. A 1080p BGR frame is
  ~6 MB; pickling that at hundreds of fps would cost more than the inference it
  feeds. The *consumer* releases the slot after reading, which is what stops a
  producer overwriting a frame still in flight. Detections are converted to plain
  arrays **before** release, because ultralytics keeps a reference to the source
  array in `Results.orig_img`.
- **Fallbacks:** a frame too big for a slot, or arriving when the pool is empty, is
  carried inline. Slot acquisition times out in 50 ms — waiting longer would show up
  as pure inference latency for a frame that could have been sent immediately.
- **Opportunistic batching:** takes whatever already arrived; `batch_max_wait_ms`
  defaults to **0**. A fixed window is pure latency when batches can't fill, and at
  low feed counts it is entirely wasted — each feed has one frame in flight, so the
  frames being waited for belong to feeds blocked on this very batch. Measured with
  one feed at 12 ms: a flat ~10–12 ms added to every frame.
- **Warmup shapes:** what forces a re-autotune is the letterboxed tensor shape,
  which depends on source **aspect ratio** and imgsz, not resolution. At imgsz 1280
  every 16:9 source becomes 1280x736; 4:3 becomes 1280x960. `warmup_shapes` is
  deduped by preprocessed shape. This matters because the first frame of an unwarmed
  shape stalls the *shared* process — every feed, not just the new camera.

---

## 5. Capacity

Measured, RTX 4070 Ti SUPER, imgsz 1280, FP16:

| `BATCH n=` | `predict` | per frame | aggregate |
|---|---|---|---|
| 1 | 15–17 ms | 16 ms | ~63 fps |
| 2 | 20 ms | 10 ms | ~100 fps |
| 3 | 27 ms | 9 ms | ~110 fps |
| 4 | 28 ms | 7 ms | **~143 fps** |

```
demand = cameras × source_fps ÷ frame_stride        keep under ~150 fps per module
```

30 cameras at 30 fps needs `frame_stride ≈ 6`. At stride 1 the same 30 cameras
demand 900 fps — 6x the GPU — and every feed sits at `pace=0ms` dropping frames.
`max_feeds` (80) is an **admission limit, not a throughput promise**.

Two things capacity math must not forget:

- **Stride does not reduce decode cost.** Every frame is still decoded at source
  rate (~5–9 ms software, ~1–2 ms NVDEC). At 30 cameras that is ~7 of 36 cores; at
  80 cameras ~19. NVDEC is what raises this ceiling.
- **IPC is not the bottleneck.** With one feed alone, `detect=16ms` against
  `predict=15ms` — about 1 ms of transport. Don't optimise the plumbing.

`inference_imgsz = 1280` is deliberate: on real footage 1280 detects both the person
and the phone where 640/320 find only the person. It is also ~4x the cost of 640, so
it and `frame_stride` are the two dials that set the ceiling.

---

## 6. The central/module contract

**Module → central** (always module-initiated)

| Endpoint | When | Carries |
|---|---|---|
| `POST /api/modules/register` | startup | id, public URL, GPU, `max_feeds`, fps budget |
| `POST /api/modules/{id}/heartbeat` | ~10 s | active feeds, per-camera status |
| `POST /api/events` | on violation | batched events, spooled and retried |

**Central → module** — the module's existing `POST /feeds/stream`,
`/feeds/{id}/stop`, `/feeds/{id}/geometry`, `GET /feeds`. Central passes its own
`camera_id` on start so events come back attributable; the module only knows its
own `feed_id`.

**Browser → module, directly** — `WS /feeds/{id}/subscribe`. Central hands out the
URL and nothing more; frames never traverse central.

Events are wired in at the module's existing global events channel, so no CV code
knows central exists.

**Event delivery is at-least-once.** Modules spool to disk and drop nothing until
central acknowledges, so a blip delays alerts instead of losing them. A retry after
partial failure can duplicate — showing a violation twice beats losing one.

---

## 7. Trade-offs and risks

- **`supervision.ByteTrack` is deprecated** (removed in supervision 0.30) and the
  shared-model design depends on it for per-feed identity. `requirements.txt` pins
  `<0.30` for exactly this reason. Before that pin has to move: vendor ByteTrack or
  own the identity layer.
- **The inference process has no pipelining.** One thread does preprocess → GPU →
  respond in sequence, so while the GPU computes batch K nothing prepares K+1. GPU
  utilisation cannot reach 100% regardless of `infer_threads`. Fixing it properly
  means double-buffering, or TensorRT/ONNX so preprocessing moves onto the GPU.
  **This is the next real throughput ceiling.**
- **The inference process is a single point of failure** per module, with no
  supervisor. A worker crash is worse: it dies silently and its feeds simply stop.
- **Shared memory is allocated up front** — `infer_slots × h × w × 3` ≈ 400 MB at
  defaults. A frame larger than a slot silently takes the slow inline path; at 4K
  *every* frame would. Raise `infer_slot_max_*` if 4K cameras arrive.
- **`num_workers` is capped at 8** by default. On a 36-core box that leaves cores
  idle for the CPU-bound half; raise it and measure `post=`/`gap=`.
- **No authentication anywhere on the module**, and none on central's camera-admin
  routes. Neither is safe on an untrusted network as-is.
- **Central has no auth on the dashboard/API**, and `CENTRAL_TOKEN` empty means any
  host can register as a module.
- **Placement ignores the fps budget until modules report one.** `Config` has no
  `fps_budget` field yet, so modules report 0 and placement falls back to counting
  feed slots — which will overcommit a box. Add it with the measured ~150.

---

## 8. Open questions

- What replaces `supervision.ByteTrack` before 0.30?
- What `frame_stride` do the rules actually need? This sets the real capacity
  ceiling and nothing else moves it as cheaply. Needs a measurement on real
  footage: at what analysis fps do detections start being missed?
- Does NVDEC engage on the deployment host? Check the startup log for `NVDEC
  hardware decode active.` and confirm with `nvidia-smi dmon` (dec column). If not,
  that is the largest single CPU cost in the system.
- Video path when modules sit behind NAT at client sites: relay watched tiles
  through central, or WebRTC + TURN?
