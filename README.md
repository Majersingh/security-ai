# AI-Powered CCTV Operator Monitoring System

Monitors **live camera streams** (RTSP / HLS / HTTP) of control-room operators
and detects **mobile phone usage on duty**, plus optional **zone-intrusion** and
**line-crossing** rules drawn on each stream. It detects and tracks people,
associates phones with tracked operators, and streams annotated frames + a live
events feed to a web dashboard. Many streams run concurrently on a single GPU.

Artefacts kept: per-feed violation snapshots + `events.csv` under
`output/<feed_id>/`. Video itself is **never stored**.

> **Behaviours implemented:** phone-usage, zone-intrusion, line-crossing.
> Sleeping / gaze / absence detection are intentionally **not** implemented yet
> (see [Future Improvements](#future-improvements)).

---

## Architecture

The pipeline is deliberately modular — one responsibility per file — so later
phases plug in without touching the core loop:

```
web/server.py   FastAPI: /feeds/* REST + WebSocket transport (stream-only)
sources.py      StreamURLSource: PyAV decode of RTSP/HLS/HTTP (NVDEC when available)
feeds.py        Feed + FeedManager: concurrent feeds, per-process registry
workers.py      WorkerPool: coordinator + N worker processes (the CPU-bound work)
inference.py    ONE process owns the GPU: shared model, batching, shared-mem ring
streaming.py    FrameProcessor: per-frame track -> rules -> annotate (no model)
tracker.py      Tracker:        routes tracked detections to persons / phones
behavior.py     BehaviorEngine + BehaviorRule + PhoneUsageRule / Zone / Line
                                business logic, debounced into episodes
annotator.py    Annotator:      draws green/blue/red boxes + labels
events.py       EventLog + SnapshotManager: CSV + JPEG persistence
config.py       Config:         every tunable value (typed dataclass)
utils.py        logging, geometry (IoU/containment), timestamp helpers
```

See `docs/ARCHITECTURE.md` for the multi-feed / single-GPU design in depth.

**Extensibility:** to add a new behaviour (e.g. sleeping), implement a new
`BehaviorRule` subclass and register it in `build_rules()` (`behavior.py`).
No other file changes. The debounce/episode logic, event logging and snapshots
are handled generically for every rule.

**Robustness choices beyond the brief:**
- Thresholds are configured in **seconds** and converted to frames using the
  video's real FPS (works across 15/25/30 fps sources).
- A per-track **state machine** debounces raw hits into episodes, so a 10-second
  phone use produces **one** event, not hundreds of duplicate CSV rows.
- Phone↔person association uses **containment** (fraction of the phone box
  inside an inflated person box) rather than raw IoU, which is more reliable for
  a small phone against a large person box.

---

## Installation

### 1. Virtual environment

```bash
cd security-ai
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Dependencies

```bash
pip install -r requirements.txt
```

This installs the full stack — Ultralytics (YOLOv11), Supervision (ByteTrack +
annotators), OpenCV, PyAV (stream decode), NumPy, Pandas, and the web server
(FastAPI + uvicorn) — from the single `requirements.txt`. The YOLOv11 weight
(`yolo11n.pt`) is downloaded automatically on first run and cached in `models/`.

> GPU strongly recommended for real-time multi-stream inference. Device is
> auto-detected (`Config.device = "auto"` → CUDA / MPS / CPU).

---

## Running

Start the server:

```bash
PYTHONPATH=src .venv/bin/python -m uvicorn --app-dir web server:app \
    --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`, paste a camera **stream URL** (RTSP / HLS / HTTP)
and click **Add Stream**. Each stream runs as its own feed; draw a **line** or
**zone** on a connected stream to add tripwire / intrusion rules. Detected
violations appear in the events panel and are saved as per-feed snapshots +
`events.csv` under `output/<feed_id>/`. Nothing else is stored.

REST/WS API: `GET /feeds`, `POST /feeds/stream {url}`, `POST /feeds/{id}/stop`,
`POST /feeds/{id}/geometry`, `WS /feeds/{id}/subscribe`.

---

## Expected Output

Per feed, under `output/<feed_id>/`:

```
output/<feed_id>/
  events.csv             # one row per confirmed violation episode
  snapshots/
    phone_000123.jpg     # JPEG evidence, named <prefix>_<frame>.jpg
```

Annotated frames are streamed to the dashboard live; **no video is stored**.

`events.csv` columns:

| Timestamp | Frame Number | Person ID | Event | Confidence |
|-----------|--------------|-----------|-------|------------|
| 00:00:04.120 | 103 | 1 | Mobile Phone Usage | 0.812 |

---

## Folder Structure

```
security-ai/
  output/           <feed_id>/{events.csv, snapshots/}   (per-feed artefacts)
  models/           yolo11n.pt            (auto-downloaded weights)
  src/              detector, tracker, behavior, annotator, events, config,
                    utils, streaming (FrameProcessor), sources, feeds
  web/              server.py (FastAPI), static/index.html (dashboard)
  docs/             ARCHITECTURE.md
  requirements.txt
  README.md         SETUP.md
```

---

## How it works

A stream URL is opened by `StreamURLSource` (PyAV) and processed frame by frame
through `FrameProcessor` (`Detector` → `Tracker` → `BehaviorEngine` →
`Annotator`). Each stream is a `Feed`; `FeedManager` runs many feeds concurrently
and **serializes GPU inference behind a shared semaphore** so one GPU is shared
cleanly. Live streams use **drop-to-latest** (skip stale frames to bound latency)
and **auto-reconnect**. The browser subscribes over a WebSocket and receives
annotated JPEG frames + events; drawn zone/line geometry is pushed back with
`POST /feeds/{id}/geometry` and applied on the next frame.

See `docs/ARCHITECTURE.md` for the full design.

## Zone & line detection (intrusion + crossing)

Two extra behaviours can be enabled by defining geometry:

- **Zone intrusion** — alert when a person's bounding box **overlaps** the
  polygon (any part enters, not just a single point). Sustained + debounced
  (one event per stay). Sensitivity via `zone_overlap_ratio` (0.0 = any overlap).
- **Line tripwire** — alert when a person's box **touches/crosses** the line.
  Fires immediately on the first touch and once per touch (re-fires on a fresh
  touch), so a quick pass-through is caught.

Both are implemented as `BehaviorRule` subclasses in `src/behavior.py` and are
registered automatically by `build_rules()` when geometry is provided — no
pipeline changes. Events land in the same `events.csv` (`Zone Intrusion`,
`Line Crossing (in)`/`(out)`), with snapshots.

### Define geometry by drawing on a live stream

On each connected stream tile, use the **Line / Zone / Clear** tools:

1. Click **Line** and click 2 points, and/or **Zone**, click ≥3 points, then
   **Finish**. **Clear** removes them.
2. The drawn (normalized) coordinates are sent via `POST /feeds/{id}/geometry`;
   the feed rebuilds its rules on the next frame and the annotated stream then
   shows the zone/line and fires intrusion / crossing events.

## Tuning knobs (where to change behaviour)

**When is a "Mobile Phone Usage" event logged?**
A single phone detection is *not* enough. A phone must stay near the person
**continuously for `violation_start_seconds` (default 1.0 s)**, then **one**
event fires for the whole episode. The episode ends after
`violation_end_seconds` (1.5 s) without the phone.

All knobs live in `config.py` (a typed dataclass); the web server tweaks a few
per feed (e.g. `inference_imgsz`).

| Knob (`config.py`) | Default | Effect |
|---|---|---|
| `violation_start_seconds` | 1.0 | How long the phone must be used before logging |
| `violation_end_seconds` | 1.5 | Gap of no-phone that ends an episode |
| `confidence_threshold` | 0.25 | Min detection score |
| `proximity_margin` | 0.15 | Person box inflation when testing "near" |
| `min_containment` | 0.30 | Fraction of phone inside person to count |
| `snapshot_cooldown_seconds` | 5.0 | Gap between snapshots in one episode |
| `inference_imgsz` | 1280 | Detection resolution (accuracy vs speed); cost scales ~quadratically |
| `frame_stride` | 1 | Process every Nth frame — with `inference_imgsz`, the main throughput dial |
| `max_feeds` | 80 | Admission limit on concurrent feeds (not a VRAM ceiling) |
| `num_workers` | physical cores (max 8) | Worker processes for decode/track/annotate; 0 = all in-process |
| `cv_threads` / `torch_threads` | 1 | Per-process thread caps, so N processes don't oversubscribe the CPU |
| `hw_decode` | True | NVDEC hardware decode, with automatic software fallback |
| `batch_max_size` / `batch_max_wait_ms` | 16 / 12 | Batching in the single inference process |

Time-based thresholds are expressed in **seconds** and converted to frames using
each stream's real FPS, so "1 second of phone use" means 1 real second across
15/25/30 fps sources.

## Tuning notes (accuracy)

- **Detection resolution matters on high-res footage.** In 2560×1440 CCTV a
  phone is a very small object. At the default YOLO `imgsz=640` the nano model
  detected **0** phones; at `imgsz=1280` it detected the phone in ~93% of
  sampled frames. `inference_imgsz` therefore defaults to `1280` (the server
  uses `640` for live streams, where subjects are closer). Raise it if phones
  are missed.
- **Model size.** `yolo11n` (nano) is the fastest and auto-downloaded default.
  For better small-object recall set `model_path` to `yolo11s`/`yolo11m` in
  `config.py` (Ultralytics downloads it automatically).
- **Confidence / thresholds.** `confidence_threshold`, `violation_start_seconds`
  and the proximity values in `config.py` trade sensitivity against false positives.
- **Speed.** A CUDA GPU is strongly recommended for real-time multi-stream
  inference; CPU is fine only for a single low-fps stream.

## Future Improvements

**Phase 2 — attention & presence**
- Head-pose estimation & gaze detection ("looking away from monitor")
- Sleeping detection (eye-closure / posture over time)
- Absence detection (no person at workstation for N seconds)
- Distraction scoring

**Phase 3 — identity & scale**
- Face recognition + operator attendance logging
- Multiple simultaneous operators / multi-camera
- Live RTSP ingestion instead of file playback
- REST API + web dashboard (timeline, live alerts, evidence review)
- Event store in a database (SQLite/Postgres) instead of flat CSV
- Alerting integrations (email/Slack/webhook)

The current rule/engine/event separation is designed so these slot in as new
`BehaviorRule` implementations and new output sinks without rewrites.
