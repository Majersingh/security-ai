# AI-Powered CCTV Operator Monitoring System (PoC)

A proof-of-concept that analyses CCTV footage of a control-room operator and
detects **mobile phone usage while on duty**. It detects and tracks people,
associates phones with a tracked operator, and produces an annotated video, an
events CSV and per-violation snapshots.

> **Milestone 1 scope:** person detection, phone detection, person tracking,
> phone-usage detection, annotated video, events CSV, violation snapshots.
> Sleeping / gaze / absence detection are intentionally **not** implemented yet
> (see [Future Improvements](#future-improvements)).

---

## Architecture

The pipeline is deliberately modular — one responsibility per file — so later
phases plug in without touching the core loop:

```
main.py         VideoProcessor: opens video, drives the loop, writes outputs
detector.py     Detector:       YOLOv11 -> person + phone detections only
tracker.py      Tracker:        ByteTrack -> persistent person IDs
behavior.py     BehaviorEngine + BehaviorRule + PhoneUsageRule
                                business logic, debounced into episodes
annotator.py    Annotator:      draws green/blue/red boxes + labels
events.py       EventLog + SnapshotManager: CSV + JPEG persistence
config.py       Config:         every tunable value (typed dataclass)
utils.py        logging, geometry (IoU/containment), timestamp helpers
```

**Extensibility:** to add a Phase-2 behaviour (e.g. sleeping), implement a new
`BehaviorRule` subclass and register it in the `rules=[...]` list in `main.py`.
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

This installs Ultralytics (YOLOv11), Supervision (ByteTrack + annotators),
OpenCV, NumPy and Pandas. The YOLOv11 weight (`yolo11n.pt`) is downloaded
automatically on first run and cached in `models/`.

> GPU is optional. Default device is `cpu`; pass `--device 0` for CUDA or
> `--device mps` on Apple Silicon.

---

## Running


Place your MP4 at `input/operator.mp4`, then:

```bash
python src/main.py
```

Common overrides (no code edits needed):

```bash
python src/main.py --input input/operator.mp4 \
                   --device 0 \
                   --conf 0.4 \
                   --start-seconds 1.5
python src/main.py --no-video          # CSV + snapshots only (faster)
python src/main.py --log-level DEBUG
```

---

## Expected Output

```
output/
  annotated.mp4          # video with green (person) / blue (phone) / red (violation) boxes
  events.csv             # one row per confirmed violation episode
  snapshots/
    phone_000123.jpg     # JPEG evidence, named <prefix>_<frame>.jpg
```

`events.csv` columns:

| Timestamp | Frame Number | Person ID | Event | Confidence |
|-----------|--------------|-----------|-------|------------|
| 00:00:04.120 | 103 | 1 | Mobile Phone Usage | 0.812 |

---

## Folder Structure

```
security-ai/
  input/            operator.mp4          (source video)
  output/           annotated.mp4, events.csv, snapshots/
  models/           yolo11n.pt            (auto-downloaded weights)
  src/              main, detector, tracker, behavior, annotator, events, config, utils
  requirements.txt
  README.md
```

---

## Web UI (live streaming scan)

Two input modes, chosen with a toggle on the page:

1. **📁 Upload Video** — upload a file and watch it scanned frame-by-frame.
2. **📹 Live Camera** — scan your webcam feed in real time.

```bash
pip install -r requirements-web.txt
python -m uvicorn web.server:app --host 0.0.0.0 --port 8000 --app-dir .
```

Then open <http://localhost:8000>.

### How it works

Both modes reuse the **same** CV pipeline via `FrameProcessor` in
`src/streaming.py` (which wraps `Detector`/`Tracker`/`BehaviorEngine`/
`Annotator`). Only the frame *source* differs:

| Mode | Endpoint | Frame source |
|------|----------|--------------|
| Upload | `POST /upload` → WS `/ws/{job_id}` | server reads the uploaded file |
| Camera | WS `/ws-live` | browser captures frames and pushes them up |

For the camera, the browser grabs frames with `getUserMedia`, sends each one as
a JPEG over the WebSocket, the server runs the pipeline and returns the
annotated frame + any events. Sending is **paced** (one frame in flight at a
time), so it naturally throttles to the server's processing speed.

> ⚠️ **Camera access requires a secure context.** Browsers only allow
> `getUserMedia` on `https://` **or** `http://localhost`. Over a plain-HTTP LAN
> IP the camera button will be blocked — deploy behind HTTPS (e.g. a Cloudflare
> tunnel or a TLS reverse proxy) for the camera mode to work remotely. Upload
> mode works over plain HTTP.

> On CPU, inference is ~2–5 fps at 1440p, so the preview plays in slow motion —
> which reads naturally as "scanning". A GPU (`device` in `config.py`) makes it
> real-time.

```
web/
  server.py          FastAPI: /upload, /ws/{job_id}, /ws-live, serves the UI
  static/index.html  single-page front-end (mode toggle, live video, event log)
```

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

### Define geometry by drawing in the browser (recommended)

In the web UI, use the **Draw detection area** toolbar under the video:

1. Pick a video (or enable the camera) — the first frame appears.
2. Click **Line** and click 2 points, and/or **Zone**, click ≥3 points, then
   **Finish Zone**. **Clear** removes them.
3. Start the scan. The drawn coordinates are sent to the server, which runs the
   rules and draws the zone/line onto the output.

### Define geometry in config (fixed camera)

Coordinates are in native frame pixels:

```python
# src/config.py
zone_polygon = [(400, 200), (900, 200), (900, 700), (400, 700)]
line_start   = (0, 500)
line_end     = (1280, 500)
```

## Tuning knobs (where to change behaviour)

All values live in `src/config.py`; the common ones also have CLI flags.

**When is a "Mobile Phone Usage" event logged?**
A single phone detection is *not* enough. A phone must stay near the person
**continuously for `violation_start_seconds` (default 1.0 s)**, then **one**
event fires for the whole episode. The episode ends after
`violation_end_seconds` (1.5 s) without the phone.

| Knob (`config.py`) | Default | Effect | CLI |
|---|---|---|---|
| `violation_start_seconds` | 1.0 | How long the phone must be used before logging | `--start-seconds` |
| `violation_end_seconds` | 1.5 | Gap of no-phone that ends an episode | — |
| `confidence_threshold` | 0.25 | Min detection score | `--conf` |
| `proximity_margin` | 0.15 | Person box inflation when testing "near" | — |
| `min_containment` | 0.30 | Fraction of phone inside person to count | — |
| `snapshot_cooldown_seconds` | 5.0 | Gap between snapshots in one episode | — |
| `inference_imgsz` | 1280 | Detection resolution (accuracy vs speed) | — |
| `frame_stride` | 1 | Analyse every Nth frame (speed) | `--frame-stride` |

**Frame sampling (`frame_stride`).** To analyse fewer frames on a 30 fps video:

```bash
python src/main.py --frame-stride 30    # ~1 analysed frame per second (~30x faster)
python src/main.py --frame-stride 5     # every 5th frame (~5x faster)
```

Time-based thresholds **auto-adjust** to the effective rate (`fps / stride`), so
"1 second of phone use" still means 1 real second regardless of stride. Event
timestamps also stay accurate to the original video time. Trade-off: higher
stride = coarser timing and slightly less stable tracking IDs during fast
motion. Also honoured by the web UI's upload mode.

## Tuning notes (accuracy)

- **Detection resolution matters on high-res footage.** In 2560×1440 CCTV a
  phone is a very small object. At the default YOLO `imgsz=640` the nano model
  detected **0** phones; at `imgsz=1280` it detected the phone in ~93% of
  sampled frames. `inference_imgsz` therefore defaults to `1280`. Lower it for
  speed on low-res footage, raise it if phones are still missed.
- **Model size.** `yolo11n` (nano) is the fastest and auto-downloaded default.
  For better small-object recall swap to `yolo11s`/`yolo11m` via
  `--model models/yolo11m.pt` (Ultralytics downloads it automatically).
- **Confidence / thresholds.** `--conf`, `--start-seconds` and the proximity
  values in `config.py` trade sensitivity against false positives.
- **Speed.** CPU inference on 1440p is ~2–5 fps. Use `--device 0` (CUDA) or
  `--device mps` (Apple) for real-time-class throughput.

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
