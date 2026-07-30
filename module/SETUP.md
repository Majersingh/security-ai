# Analysis module — setup

The deployable unit: owns **one GPU**, decodes its assigned camera feeds, detects
behaviour, streams annotated video to browsers, and reports events to central.
Deploy one per GPU host; that is how the system scales.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r module/requirements.txt
```

**Install a CUDA build of torch**, not the default CPU wheel — ultralytics will
pull a CPU-only torch and the pipeline will silently run ~50x slower. Check with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

`ffmpeg`/libav comes bundled with PyAV; no system ffmpeg needed. The model weight
(`module/models/yolo11n.pt`) is downloaded by ultralytics on first use if absent.

## Configure

Two separate places, and the split matters:

**1. `module/.env` — how this module talks to central**

```bash
cp module/.env.example module/.env      # then edit
```

| Variable | Notes |
|---|---|
| `CENTRAL_URL` | Leave **empty** to run standalone (serves `/feeds/*` and its own dashboard, reports nothing). Supported mode. |
| `MODULE_TOKEN` | Must match `CENTRAL_TOKEN` on central. |
| `MODULE_ID` | Stable per host. Unset = random id, so a restart looks like a new module to central. |
| `MODULE_PUBLIC_URL` | Where **browsers** reach this box for video. Not `127.0.0.1`, not a docker-internal name. |

**2. `module/core/config.py` — pipeline tuning**

Not environment-driven; edit the file per deployment. The knobs that matter, in
order of impact:

| Setting | Effect |
|---|---|
| `frame_stride` | Frames analysed = `source_fps / stride`. **The main capacity dial.** Must match `CENTRAL_STRIDE`. |
| `inference_imgsz` | 1280 detects small/distant phones; 640 is ~4x cheaper and misses them. Verified on real footage. |
| `hw_decode` | NVDEC. Software decode is the largest per-feed CPU cost. |
| `num_workers` | Worker processes for decode/track/annotate. Defaults to physical cores, capped 8 — **raise the cap on a big box**. |
| `cv_threads` / `torch_threads` | 1 each. Do not raise: parallelism comes from feed count, not per-feed threads. |
| `infer_threads` | The inference process — deliberately **not** 1, because ultralytics preprocesses on CPU there. 0 = auto. |
| `warmup_shapes` | One entry per camera **aspect ratio** you deploy (resolution is irrelevant). |
| `max_feeds` | Admission limit, not a throughput promise. |

## Run

```bash
uvicorn module.app:app --env-file module/.env --host 0.0.0.0 --port 8001
```

Healthy startup looks like:

```
GPU DETECTED: NVIDIA GeForce RTX 4070 Ti SUPER | CUDA 12.x | inference device=cuda:0
Coordinator mode: 8 worker process(es), cv_threads=1, torch_threads=1
Frame ring: 64 slots x 6.3 MB = 401 MB shared.
Inference process up (batch<=16, wait=0ms, threads=28).
Inference ready (device=cuda:0, precision=fp16, imgsz=1280, warmed tensor shapes=['1280x736','1280x960'])
Reporting to central https://... as module 'gpu-host-1'
Registered with central as 'gpu-host-1'.
```

**Check three things in that output:**

1. `device=cuda:0` and `precision=fp16` — not `cpu`/`fp32`.
2. `NVDEC hardware decode active.` vs `CUDA decode unavailable (…); using software
   decode.` The second means every feed decodes on CPU, which is the usual cause of
   a pinned CPU with an idle GPU.
3. `Registered with central` — if absent, check `CENTRAL_URL` and the token.

Startup does **not** block on model warmup; the API is up while the inference
process warms in the background. The first feed added before warmup finishes will
show a one-off multi-second `detect=`.

## Scaling out

Same code, another GPU box, three env vars. Nothing on central changes:

```bash
CENTRAL_URL=https://central.yourorg.internal \
MODULE_TOKEN=<secret> \
MODULE_ID=gpu-host-2 \
MODULE_PUBLIC_URL=http://10.0.1.23:8001 \
uvicorn module.app:app --host 0.0.0.0 --port 8001
```

Two GPUs in one chassis? Run two instances, each seeing only its own card, and
**split the CPU between them** — otherwise both take the machine-wide defaults and
oversubscribe each other:

```bash
CUDA_VISIBLE_DEVICES=0 uvicorn module.app:app --port 8001 --env-file module/.env
CUDA_VISIBLE_DEVICES=1 uvicorn module.app:app --port 8002 --env-file module/.env.gpu1
```

Give each a distinct `MODULE_ID`, and halve `num_workers`/`infer_threads` in
config for that host.

## Reading the TIMING log

`log_timing = True` prints per frame per feed — useful for tuning, expensive at
scale (turn it off once tuned).

```
TIMING a4e721eb f=960 | 15 pulled | decode=44ms pace=0ms detect=73ms post=1ms emit=0ms | gap=119ms (8.4 proc-fps)
```

| Field | Meaning |
|---|---|
| `pulled` | Frames decoded to produce this one (>1 under a stride) |
| `decode` | Real decode cost. ~1-2ms = NVDEC; ~5-9ms = software |
| `pace` | Sleep to hold real time. **`0ms` means the feed is behind and dropping frames** |
| `detect` | Round-trip to the shared model: IPC + queue wait + preprocess + GPU |
| `post` | This feed's track/rules/annotate/JPEG — only large when someone is watching |
| `gap` | Wall time between processed frames |

`BATCH n=<k> predict=<t>ms` from the inference process is the one that tells you
whether the **GPU** is the limit: `n` at `batch_max_size` with sub-linear `predict`
growth means batching is working; small `n` while feeds fall behind means the
bottleneck is upstream.

## Capacity

Measured on an RTX 4070 Ti SUPER at imgsz 1280, FP16:

| `BATCH n=` | per frame | aggregate |
|---|---|---|
| 1 | 16 ms | ~63 fps |
| 4 | 7 ms | **~143 fps** |

```
demand = cameras × source_fps ÷ frame_stride     must stay under ~150 fps
```

So 30 cameras at 30 fps needs `frame_stride ≈ 6`. At `frame_stride = 1` those same
30 cameras demand 900 fps — 6x the GPU — and every feed sits at `pace=0ms` dropping
frames. Note stride does **not** reduce decode cost: you still decode every frame
at source rate, which is why NVDEC matters independently.

## Tests

```bash
python tests/test_ring.py         # shared-memory frame transport
python tests/test_e2e.py          # coordinator + 2 workers + inference process
python tests/test_wired.py        # this module registering with a live central
python tests/test_batch_wait.py   # batch-window latency
```

They lower `imgsz`/`stride` for speed on a CPU box — raise them on the GPU host.

## Notes

**No authentication on any endpoint.** The module serves its dashboard at `/` and
`/feeds/*` accepts writes from anyone who can reach it. Don't put a module on an
untrusted network as-is.

`module/static/index.html` is the old single-box dashboard, kept for standalone
debugging. The product UI is central's.
