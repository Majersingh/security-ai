# AI-Powered CCTV Operator Monitoring System

Monitors **live camera streams** (RTSP / HLS / HTTP) of control-room operators
and detects **mobile phone usage on duty**, plus optional **zone-intrusion** and
**line-crossing** rules drawn on each stream. It detects and tracks people,
associates phones with tracked operators, and reports violations to a central
dashboard with a live events feed and a video wall. Many streams run concurrently on
a single GPU, and capacity grows by adding GPU hosts.

Artefacts kept: per-feed violation snapshots + `events.csv` under
`module/output/<feed_id>/`, and — when central is running — every violation in
central's database. Video itself is **never stored**.

> **Behaviours implemented:** phone-usage, zone-intrusion, line-crossing.
> Sleeping / gaze / absence detection are intentionally **not** implemented yet
> (see [Future Improvements](#future-improvements)).

---

## Architecture

Two deployables. **`central/`** is the fleet's brain (registry, placement, event
store, dashboard) and needs no GPU. **`module/`** is the analysis unit: one GPU, N
cameras. Scale by deploying another module — it registers itself with central.

Inside a module the pipeline is one responsibility per file:

```
app.py          FastAPI: /feeds/* — detection control. Serves NO video.
rawapp.py       SEPARATE process: raw playback at source frame rate
rawstream.py    the raw decode/encode loop + ticket store (no model)
reporting.py    register + heartbeat + durable event spool -> central
core/
sources.py      StreamURLSource: PyAV decode of RTSP/HLS/HTTP (NVDEC when available)
feeds.py        Feed + FeedManager: concurrent feeds, per-process registry
workers.py      WorkerPool: coordinator + N worker processes (the CPU-bound work)
inference.py    ONE process owns the GPU: shared model, batching, shared-mem ring
streaming.py    FrameProcessor: per-frame track -> rules (no model, renders nothing)
tracker.py      Tracker:        routes tracked detections to persons / phones
behavior.py     BehaviorEngine + BehaviorRule + PhoneUsageRule / Zone / Line
                                business logic, debounced into episodes
events.py       EventLog + SnapshotManager: CSV + JPEG persistence
config.py       Config:         every tunable value (typed dataclass)
utils.py        logging, geometry (IoU/containment), timestamp helpers
```

See `docs/ARCHITECTURE.md` for the design, and `central/SETUP.md` /
`module/SETUP.md` to run each side.

**Extensibility:** to add a new behaviour (e.g. sleeping), implement a new
`BehaviorRule` subclass and register it in `build_rules()` (`module/core/behavior.py`).
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
pip install -r module/requirements.txt     # analysis module: the heavy CV stack
pip install -r central/requirements.txt    # central: fastapi + uvicorn only
```

The two share nothing on purpose — central has no torch, no CUDA, no OpenCV, so it
deploys as a small container anywhere. The module installs Ultralytics (YOLOv11),
Supervision (ByteTrack + annotators), OpenCV, PyAV, NumPy and Pandas. The YOLOv11
weight is downloaded on first run and cached in `module/models/`.

Install a **CUDA build of torch** for the module; the default wheel is CPU-only and
runs ~50x slower.

> GPU strongly recommended for real-time multi-stream inference. Device is
> auto-detected (`Config.device = "auto"` → CUDA / MPS / CPU).

---

## Running

Start an analysis module (standalone — no central needed):

```bash
uvicorn module.app:app    --host 0.0.0.0 --port 8001   # detection
uvicorn module.rawapp:app --host 0.0.0.0 --port 8011   # video (optional)
```

Or the full fleet — central plus one or more module hosts:

```bash
uvicorn central.app:app   --env-file central/.env --port 9000   # dashboard + registry
uvicorn module.app:app    --env-file module/.env  --port 8001   # detection
uvicorn module.rawapp:app --env-file module/.env  --port 8011   # video
```

With central, add cameras on **its** dashboard (port 9000) and it places them on a
module with headroom. Standalone, open the module directly (port 8001) and paste a
camera **stream URL** (RTSP / HLS / HTTP)
and click **Add Stream**. Each stream runs as its own feed; draw a **line** or
**zone** on a connected stream to add tripwire / intrusion rules. Detected
violations appear in the events panel and are saved as per-feed snapshots +
`events.csv` under `output/<feed_id>/`. Nothing else is stored.

Module API: `GET /feeds`, `POST /feeds/stream {url}`, `POST /feeds/{id}/stop`,
`POST /feeds/{id}/geometry`, `POST /feeds/probe`, `WS /events`.
Raw video service: `POST /stream/raw/ticket`, `WS /stream/raw?ticket=…`.

---

## Expected Output

Per feed, under `output/<feed_id>/`:

```
output/<feed_id>/
  events.csv             # one row per confirmed violation episode
  snapshots/
    phone_000123.jpg     # JPEG evidence, named <prefix>_<frame>.jpg
```

Live video is played by the separate raw-video service at the source frame rate;
**no video is stored**. Detection writes only snapshots and event rows.

`events.csv` columns:

| Timestamp | Frame Number | Person ID | Event | Confidence |
|-----------|--------------|-----------|-------|------------|
| 00:00:04.120 | 103 | 1 | Mobile Phone Usage | 0.812 |

---

## Folder Structure

```
security-ai/
  central/          app.py, db.py, module_client.py, placement.py, static/
                    SETUP.md, .env.example, requirements.txt   (no GPU, no torch)
  module/           app.py, reporting.py, static/
                    core/     the CV pipeline (config, feeds, inference, sources,
                              streaming, behavior, tracker, annotator, events, utils)
                    models/   yolo11n.pt        (auto-downloaded)
                    output/   <feed_id>/{events.csv, snapshots/}
                    input/    sample media
                    SETUP.md, .env.example, requirements.txt
  tests/            ring, e2e, central contract, wired module, batch window
  docs/             ARCHITECTURE.md
  README.md
```

`module/` is self-contained (models, input, output live inside it) because
`config.py` derives its root from its own location.

---

## How it works

A stream URL is opened by `StreamURLSource` (PyAV) and processed frame by frame
through `FrameProcessor` (`Tracker` → `BehaviorEngine`). Each stream is a `Feed`;
`FeedManager` runs many concurrently. Detection is **centralized in one process that
owns the GPU** and batches frames from every feed, so there is one CUDA context and
one copy of the weights; identity tracking stays per-feed. Live streams **drop stale
frames** when they fall behind real time, and **auto-reconnect**.

The detection service renders nothing. **Video is a separate process** (`rawapp.py`)
with its own decode at the source frame rate, because display fps was otherwise
capped by detection fps. The one image detection still produces is the still frame
from `POST /feeds/probe`, used to draw zone/line geometry before a camera starts.

See `docs/ARCHITECTURE.md` for the full design.

## Zone & line detection (intrusion + crossing)

Two extra behaviours can be enabled by defining geometry:

- **Zone intrusion** — alert when a person's bounding box **overlaps** the
  polygon (any part enters, not just a single point). Sustained + debounced
  (one event per stay). Sensitivity via `zone_overlap_ratio` (0.0 = any overlap).
- **Line tripwire** — alert when a person's box **touches/crosses** the line.
  Fires immediately on the first touch and once per touch (re-fires on a fresh
  touch), so a quick pass-through is caught.

Both are implemented as `BehaviorRule` subclasses in `module/core/behavior.py` and are
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
| `fps_budget` | 0 | Measured detection fps of this GPU; 0 makes central count slots and overcommit |
| `raw_max_streams` | 16 | Concurrent raw video decodes (one per watched tile) |
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

## Three things waiting for you on the GPU box:

  1. Check the startup log for NVDEC hardware decode active. vs CUDA decode 
  unavailable — biggest open unknown, one grep.
  2. python tests/bench_gpu.py 1920 1080 → set Config.fps_budget from the
  median column.
  3. Keep CENTRAL_STRIDE equal to frame_stride when you change either.

  And the two experiments worth doing before more optimisation — I've not
  recorded these anywhere, so they're worth a note in your tracker:

  - num_workers=0 with ~20 feeds on the GPU. If gap= matches coordinator mode,
  the worker processes aren't earning their complexity.
  - Motion gating on one camera — measure what fraction of frames actually need
  inference. Potentially a larger win than everything from this session.
