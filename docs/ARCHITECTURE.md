# Architecture — CCTV operator monitoring

Detects operator behaviour (phone usage, zone intrusion, line crossing, crowd
gathering, and helmet compliance when a PPE model is loaded) across many
live camera streams, with zone/line geometry drawn per camera, and scales by adding
GPU hosts.

The system is split by **what each part contends on**, which is the single idea
behind every boundary below:

| Concern | Contends on | Where it lives | Scales by |
|---|---|---|---|
| Registry, placement, events, UI | nothing heavy | `central/` | one instance |
| Decode, tracking, rules, snapshots | CPU | `module/app.py` + workers | worker processes |
| Detection | GPU | one inference process | batch depth |
| Video playback | CPU, latency-sensitive | `streamer/` | streamer hosts (no GPU) |

Setup: `central/SETUP.md`, `module/SETUP.md`, `streamer/SETUP.md`.

---

## 1. Processes

Per GPU host:

```
central (elsewhere, one per fleet) ── no GPU, no torch
   │  assign / stop / geometry / stream ticket     ▲ register / heartbeat / events
   ▼                                              │
module/app.py  :8001   DETECTION — serves no video
   ├── N worker processes   decode → track → rules → snapshots
   └── 1 inference process  the only thing touching the GPU
streamer/app.py  :8011  VIDEO — no model, own decode, source frame rate
   └── browser connects here directly for tiles

Deployables: `central/` · `module/` · `streamer/`, plus a shared `core/` package
(the stream decoder and the process helpers — the only code two of them share).

The streamer **registers itself**, so it does not need the detection module running:
a video-only host installs `streamer/requirements.txt` (no torch, ~150 MB) and needs
no GPU. `HOST_ID` pairs a streamer with the detection module on the same machine, so
central prefers a co-located streamer — it provably reaches that host's cameras.

Five process types on a combined host. That is the current operational weak point —
see §7.

**Scaling = deploy another module host.** It registers itself, central sees the
headroom and places cameras on it. Nothing on central is edited.

### Why not a container per camera

Rejected on measurements: a container per camera means a model per camera, forcing
batch-of-1 forever.

| | batch 1 (container/camera) | batch 4 (shared, measured) |
|---|---|---|
| per frame | 16 ms | 7 ms |
| aggregate | ~63 fps | **~143 fps** |

~2.3x of the GPU thrown away, plus a CUDA context (~300–500 MB VRAM) and ~2–3 GB
RSS per camera, plus a model load per camera added. Containerise by **role**.

---

## 2. Deployment target

- Per module host: **one machine, one GPU** (RTX 4070 Ti SUPER class), many cores.
- Central: any small VM. No GPU, and `central/requirements.txt` shares nothing with
  the module's — enforced by `tests/test_central_light.py`, which imports central
  with torch/cv2/numpy blocked.
- Streamer hosts: **no GPU needed.** `streamer/requirements.txt` is five packages
  (av, opencv-headless, numpy, fastapi, uvicorn), enforced by
  `tests/test_streamer_light.py`. This is what lets video scale independently of
  detection — video load follows *viewers*, detection follows *cameras*.
- Concurrency per host is bounded by **CPU** (decode, tracking, snapshots, plus one
  extra decode per watched tile) and by **GPU detection throughput** — not VRAM. One
  `yolo26n.pt` is 5.5 MB and a single shared model serves every feed.

---

## 3. The constraints that drive the design

**1. Identity tracking is per-feed; detection is not.**
Ultralytics keeps tracker state on the model's predictor, so `model.track(persist=True)`
cannot be shared. Detection therefore runs **statelessly** on one shared model, and
each feed keeps its own `supervision.ByteTrack`.
→ **One model for all feeds; one tracker per feed.**

**2. One GPU is a serial resource, so exactly one process owns it.**
All feeds' frames are combined into a single batched `predict()` in a dedicated
process: one CUDA context, one copy of the weights, batches deep enough to matter.

**3. Display fps must not depend on detection fps.**
The pipeline can only show a viewer frames it detected on, so display was capped by
detection rate — and `frame_stride` discarded intermediate frames before they were
even converted to BGR. Smooth video was impossible without spending the whole GPU on
a handful of cameras.
→ **Playback is a separate deployable with its own decode** (`streamer/`), at the
source frame rate. The detection service produces no video at all; its only image is
the still frame for geometry drawing.

**4. Cameras are sticky to a module.**
ByteTrack identities and the violation debounce state machine are sequential per
feed, so a camera cannot migrate mid-stream without resetting both.
→ **Sharded, not load-balanced.** No distributed state on the hot path.

**5. Central must never need to dial a module.**
Outbound HTTPS works from behind NAT; inbound usually does not. The module
initiates registration, heartbeats and event delivery. Central *does* currently call
back to `/feeds/*`; `central/module_client.py` is the single seam to change when a
module lands behind NAT.

---

## 4. Inside a module

### 4.1 `FrameSource` (`core/sources.py`)
PyAV decodes RTSP/HLS/HTTP/file. Asks libav for **CUDA (NVDEC)** and falls back to
software automatically, probing once per process. **One decode thread per feed**
(`decode_threads`) — left alone libav takes roughly one thread per core *per feed*.

Only frames that will be used are converted to BGR: `to_ndarray` is among the most
expensive per-frame operations and under a stride most decoded frames are discarded.
Decode itself cannot be skipped (H.264 frames reference each other).

Drop-when-behind: a feed more than `stream_max_lag_seconds` behind schedule skips
frames *without resyncing*, so the backlog shrinks until it is live again. Measured
with a 200 ms/frame consumer on a 25 fps source: 2.3 s of video per 12 s wall-clock
before the fix (5x slow motion), 11.4 s per 12 s after.

### 4.2 `FrameProcessor` (`core/streaming.py`)
Per-feed state that must not be shared: ByteTrack identities, the rule state
machine, the event log. Owns no model and **renders nothing** — there is no
annotated-frame path. `detections=None` means "nothing this frame", so a failed
batch degrades to a quiet frame rather than killing the feed.

### 4.3 `Feed` / `FeedManager` (`core/feeds.py`)
```python
async for frame, idx in source:
    detections = await inferencer.infer(frame)       # shared model, batched
    payload = await run_in_executor(post_pool, proc.process_json, frame, idx,
                                    detections)      # this feed's own state
```
`FeedManager` holds one inference handle — an `InferenceClient` (GPU elsewhere) or a
`LocalInferencer` (`num_workers == 0`) — and does not know which.

Payloads are emitted on **events, plus a 1 Hz heartbeat** for progress. Emitting per
frame would put thousands of queue messages per second in front of numbers a
dashboard reads once a second.

Each feed's source pull gets its own thread: `frames()` sleeps there for real-time
pacing, and on a shared pool that sleeping thread starves latency-critical work.

### 4.4 Inference service (`core/inference.py`)
- **`ModelRunner`** — the only place the model is loaded and called. Pads batches to
  power-of-two sizes and pre-warms each, so cuDNN autotunes a handful of fixed shapes
  instead of re-tuning per batch (~1 s+ each, stalling every feed).
- **`FrameRing`** — fixed-size shared-memory slots. A 1080p BGR frame is ~6 MB;
  pickling that at hundreds of fps would cost more than the inference it feeds. The
  *consumer* releases the slot after reading, which stops a producer overwriting a
  frame still in flight. Detections are converted to plain arrays **before** release,
  because ultralytics keeps a reference to the source array in `Results.orig_img`.
- **Fallbacks:** a frame too big for a slot, or arriving when the pool is empty, goes
  inline. Slot acquisition times out in 50 ms — longer would surface as pure
  inference latency for a frame that could have been sent immediately.
- **Opportunistic batching:** takes whatever already arrived; `batch_max_wait_ms`
  defaults to **0**. A fixed window is pure latency when batches can't fill, and at
  low feed counts it is entirely wasted — each feed has one frame in flight, so the
  frames being waited for belong to feeds blocked on this very batch. Measured with
  one feed at 12 ms: a flat ~10–12 ms added to every frame.
- **Warmup shapes:** what forces a re-autotune is the letterboxed tensor shape, which
  depends on source **aspect ratio** and imgsz, not resolution. At imgsz 1280 every
  16:9 source becomes 1280x736; 4:3 becomes 1280x960. `warmup_shapes` is deduped by
  preprocessed shape. It matters because the first frame of an unwarmed shape stalls
  the *shared* process — every feed, not just the new camera.

### 4.5 Streamer (`streamer/app.py`, `streamer/videostream.py`)
Its own deployable, process and port. No model, no `FeedManager`, no worker
processes, no shared event loop, and — enforced by `tests/test_streamer_light.py` —
no torch. Plays at the source frame rate: measured **28.0 fps from a 28.79 fps
source**. It has its own env-driven `StreamerConfig` rather than the module's, and
registers itself with central under `role: "streamer"`, `max_feeds: 0`.

**Any streamer can serve any camera it can reach**, including one detected on a
different host — which is what makes a video-only host possible. Central prefers a
streamer sharing the camera's `HOST_ID`, then falls back to the rest.

Its own process rather than routes on the detection app because pushing frames at
source rate means ~28 `send_json` calls per second *per tile*; on the detection app
that is the same event loop relaying results from the worker processes.

- **Tickets, not URLs.** Browsers connect with a short-lived ticket. An
  `rtsp://user:pass@host` in a WebSocket query string would land in browser history
  and access logs.
- **`STREAMER_MAX_STREAMS`** caps concurrent decodes so a video wall's "all" button
  cannot swamp the box.
- Central never places cameras on a streamer, and pinning one is rejected — it has
  `max_feeds: 0` and is excluded from the capacity table so it cannot skew headroom.
- Cost: one extra decode per *watched* camera. That is the deliberate trade — decode
  is cheap next to a GPU pass, and it stops only when the tile closes.

### 4.6 Geometry (`/feeds/probe`)
The one image the detection service still produces. Opens the stream, takes **one**
frame, closes: no `Feed`, no `max_feeds` slot, no detection, no artefacts. Reports
the **true** source dimensions so the preview keeps the source aspect ratio — drawing
on a distorted canvas puts the zone somewhere other than where it looked.

Geometry is **normalized 0..1** throughout, so it is resolution- and
module-independent: a zone drawn on a probe frame from module A stays correct if
placement later puts the camera on module B.

---

## 5. Capacity

Measured, RTX 4070 Ti SUPER, imgsz 1280, FP16:

| `BATCH n=` | `predict` | per frame | aggregate |
|---|---|---|---|
| 1 | 15–17 ms | 16 ms | ~63 fps |
| 2 | 20 ms | 10 ms | ~100 fps |
| 4 | 28 ms | 7 ms | **~143 fps** |

```
demand = cameras × source_fps ÷ frame_stride        keep under ~150 fps per host
```

30 cameras at 30 fps needs `frame_stride ≈ 6`. At stride 1 the same 30 cameras demand
900 fps — 6x the GPU — and every feed sits at `pace=0ms` dropping frames. `max_feeds`
(80) is an **admission limit, not a throughput promise**; `fps_budget` is what makes
central enforce the real ceiling, and while it is 0 placement only counts slots.

Run `python tests/bench_gpu.py 1920 1080` on the host to measure it rather than
trusting the table. Use the **median** column and set `fps_budget` slightly under it:
the benchmark loop has no IPC, no queue wait and no competing decode.

Two things capacity math must not forget:

- **Stride does not reduce decode cost.** Every frame is still decoded at source rate
  (~5–9 ms software, ~1–2 ms NVDEC). At 30 cameras that is ~7 of 36 cores; at 80
  cameras ~19. NVDEC is what raises this ceiling.
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
| `POST /api/modules/register` | startup | id, public URL, **role** (`detection`/`streamer`), **host**, GPU, `max_feeds`, `fps_budget` |
| `POST /api/modules/{id}/heartbeat` | ~10 s | active feeds, per-camera status |
| `POST /api/events` | on violation | batched events, spooled and retried |

**Central → module** — `POST /feeds/stream` (with central's `camera_id` so events come
back attributable), `/feeds/{id}/stop`, `/feeds/{id}/geometry`, `/feeds/probe`,
`GET /feeds`; and `POST /stream/ticket` on a **streamer**.

**Browser → a streamer, directly** — `WS /stream?ticket=…`, obtained from
`POST /api/cameras/{id}/stream`. Central hands out the URL and nothing more; frames
never traverse central, and the RTSP URL never reaches the browser.

Events are wired in at the module's existing global events channel, so no CV code
knows central exists.

**Event delivery is at-least-once.** Modules spool to disk and drop nothing until
central acknowledges, so a blip delays alerts instead of losing them. A retry after
partial failure can duplicate — showing a violation twice beats losing one.

**Placement** is automatic (most headroom) unless a camera is **pinned** to a module.
A pin is honoured strictly and never silently overridden: if the pinned module is
offline the camera stays unplaced with status `waiting for pinned module`. People pin
for reasons central cannot see — usually that this host has the network route — so
relocating it would break the camera with nothing explaining why.

**Central's database is disposable.** There are deliberately no migrations. On
startup `Store._ensure_schema()` compares each table against `SCHEMA` (read from a
scratch in-memory DB, so there is no version to bump) and **rebuilds any that
drifted**, warning loudly. Cameras are re-added and hosts re-register. This stops
being acceptable once event history has to survive an upgrade.

---

## 7. Trade-offs and risks

- **Detection accuracy is unmeasured.** `yolo26n` (the smallest variant) on generic
  COCO class 67 "cell phone", no labelled test set, no precision/recall. Every
  measurement in this document is throughput. **This is the largest risk in the
  project** and none of the architecture work touches it.
- **Five process types per host with no supervision.** The inference process is a
  single point of failure per host; a worker crash is worse because it dies
  *silently* and its feeds simply stop. This is now the weakest part of the system.
- **The inference process has no pipelining.** One thread does preprocess → GPU →
  respond in sequence, so while the GPU computes batch K nothing prepares K+1. GPU
  utilisation cannot reach 100% regardless of `infer_threads`. Fixing it means
  double-buffering, or TensorRT/ONNX to move preprocessing onto the GPU.
- **Streaming decodes each watched camera a second time.** Fine for a few tiles;
  `STREAMER_MAX_STREAMS` (16) is what stops a wall's "all" from swamping the host —
  but it also means "all" silently shows only the first 16. A streamer host also
  needs its own network route to the cameras, and each adds one more concurrent
  connection to them, which some IP cameras cap.
- **`supervision.ByteTrack` is deprecated** (removed in 0.30) and per-feed identity
  depends on it. `requirements.txt` pins `<0.30` for exactly this reason.
- **Shared memory is allocated up front** — `infer_slots × h × w × 3` ≈ 400 MB. A
  frame larger than a slot silently takes the slow inline path; at 4K *every* frame
  would.
- **`num_workers` is capped at 8** by default. On a 36-core box that leaves cores
  idle for the CPU-bound half.
- **No authentication** on the module, the raw service, or central's camera-admin
  routes. `CENTRAL_TOKEN` empty means any host can register as a module. None of it
  is safe on an untrusted network.

---

## 8. Open questions

- **What is the detection accuracy?** Needs labelled real footage. Everything else is
  secondary to this number.
- **Does NVDEC engage on the host?** Check the module log for `NVDEC hardware decode
  active.` and confirm with `nvidia-smi dmon` (dec column). If not, it is the largest
  single CPU cost in the system.
- **Are the worker processes still earning their complexity?** Post-processing
  measures 0–5 ms per frame, and decode/encode release the GIL. Test: run
  `num_workers=0` with ~20 feeds on the GPU and compare `gap=` against coordinator
  mode. If it matches, a large amount of machinery can go.
- **Motion-gated inference.** Most control-room frames are identical to the previous
  one. Skipping detection on unchanged frames could cut inference volume several-fold
  — likely a bigger win than anything in §5.
- **What `frame_stride` do the rules actually need?** High stride coarsens the
  debounce state machine (`violation_start_seconds = 1.0` is ~3 analysed frames at
  stride 10) and may cost recall. Asserted but never measured. Rules with a longer
  clock of their own (`crowd_hold_seconds = 3.0`) tolerate stride better, which is
  a reason to prefer per-rule timescales over one global value.
- **Absence-based PPE is a different kind of claim.** Every other rule fires on
  evidence that exists; `HelmetComplianceRule` fires on evidence that is missing, so
  a detector failure and a real violation are indistinguishable at the rule level.
  The suppressors and the 5 s hold bound the damage but cannot remove it. A trained
  `head` class (`head_class_id`) tightens the question from "is a helmet somewhere in
  the top third of this person?" to "is a helmet on THIS head?", which is the only
  way to tell a worn helmet from a carried one — but the claim stays an absence.
  Whether the dataset carries that class is a labelling decision, not a code one.
- **One model or two for PPE?** A second, PPE-only model doubles GPU cost on a box
  already at ~143 fps; fine-tuning one model over person + phone + helmet keeps a
  single forward pass. Untested — no PPE weights exist in this repo yet.
- **Should a crowd be one event or N?** `CrowdGatheringRule` emits an observation per
  member, so a group of 5 logs 5 rows and 5 snapshots of the same group box per
  cooldown. That matches the `Event` schema (which is keyed by `Person ID`) and how
  zone intrusion already behaves, but a genuine group-level event would need a
  collective path through `BehaviorEngine` alongside `Observation`/`InstantEvent`.
- **What replaces `supervision.ByteTrack` before 0.30?**
