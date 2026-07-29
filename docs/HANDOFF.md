# Handoff — multi-process / single-GPU rework

Written 2026-07-30. Read `docs/ARCHITECTURE.md` §3, §4.4b, §5, phases 6–7 and §8
first; this file is only the live state and the open thread.

## The open problem (unsolved)

On the deployment box (**RTX 4070 Ti SUPER**, 17 cores visible, ~80 feeds of 720p
H.264): **CPU pinned at 99.9% across all cores, GPU nearly idle.** Per-feed logs
looked like:

```
TIMING c7e7b23e f=295 | 1 pulled | decode=9ms pace=0ms infer=205ms emit=0ms | gap=217ms (4.6 proc-fps)
```

`infer=205ms` is **not** GPU time — it is queue wait + CPU preprocessing + GPU.
The diagnostic that separates those is the `BATCH n=<k> predict=<t>ms` line from
the inference process: `n` at `batch_max_size` means the GPU is genuinely the
limit; small `n` means the queue is starved and the bottleneck is upstream.

### Why CPU is saturated — leading hypothesis, only partly addressed

Per-feed CPU cost measured on `input/operator.mp4` (848x478, 964 frames):

| | time | share |
|---|---|---|
| H.264 decode alone | 1442 ms | 67% |
| + YUV→BGR convert every frame | 2153 ms | (old behaviour) |
| + convert only every 10th | 1454 ms | (new behaviour) |

1. **Conversion waste — FIXED (uncommitted).** `frame_stride` only skipped
   *inference*; `sources.py` still ran `to_ndarray('bgr24')` on every decoded
   frame before the stride check, so 90% of that work was computed and discarded.
   Now `StreamURLSource(stride=...)` skips the conversion for frames it will not
   yield. Worth ~32% of per-feed decode CPU at stride 10, more at 720p.
2. **Decode itself is the remaining 67%, and NVDEC is the intended fix — UNVERIFIED
   ON GPU.** PyAV 18.0.0 bundles ffmpeg 8.1.2 and
   `av.codec.hwaccel.hwdevices_available()` returns `['cuda','qsv','drm','amf']`,
   so the build supports it. **Next step: grep the server log for which branch
   ran.** `sources.py::_open` logs exactly one of:
   - `NVDEC hardware decode active.` → hardware decode is working; CPU load must
     be coming from somewhere else (annotate/JPEG encode for viewers, or
     `to_ndarray` still downloading frames from GPU memory — see caveat below).
   - `CUDA decode unavailable (<reason>); using software decode.` → every feed is
     decoding on CPU. This is almost certainly the cause of the 99.9%.
3. **Caveat nobody has checked:** even with NVDEC, `frame.to_ndarray('bgr24')`
   downloads the frame to system RAM and converts there. If that conversion is
   still on the CPU, NVDEC saves the decode but not the convert. Verify with
   `nvidia-smi dmon` (look for non-zero `dec` utilisation) rather than trusting
   the log line alone.

## Known remaining ceiling (documented, not fixed)

The inference process is **one Python thread doing preprocess → GPU → respond in
sequence, with no pipelining**. While the GPU computes batch K, nothing prepares
batch K+1, so GPU utilisation cannot reach 100% no matter how many threads
`infer_threads` grants. Real fixes: double-buffer (prepare the next batch during
the current GPU call), or export to TensorRT/ONNX so preprocessing moves onto the
GPU. See `docs/ARCHITECTURE.md` §8.

## Git state

- Committed by the user as `2a30a18` / `5c5be53` ("added multi processs").
- **Uncommitted** (all verified, tests pass): `src/sources.py` (stride-aware
  conversion skip), `src/workers.py` + `web/server.py` (pass `stride=`),
  `src/config.py` (one stale comment).
- The user set `frame_stride = 10` themselves in `config.py`. Do not revert.

## What changed in the rework (all committed)

- **`src/inference.py` (new)** — one process owns the GPU. `ModelRunner` (the only
  place the model is loaded/called, power-of-two batch warmup), `FrameRing`
  (shared-memory frame slots; consumer releases the slot after reading;
  detections converted to arrays *before* release because Ultralytics holds
  `Results.orig_img` into that memory), `InferenceService`/`InferenceClient`, and
  `LocalInferencer` for `num_workers == 0`.
- **Deleted `src/detector.py` and `src/batch.py`**; removed `batched_inference` and
  `max_concurrent_inferences`. One inference design, not two.
- **Thread caps** — `cv_threads`/`torch_threads` = 1 per worker; `num_workers`
  sized to *physical* cores. **`infer_threads` is deliberately NOT 1** (0 = auto):
  capping the inference process starved it, because Ultralytics preprocesses every
  frame on the CPU inside it, serialized against the GPU. That was a real
  regression introduced mid-rework and then fixed.
- **`drop_when_behind` was dead code** — the pacer reset `next_t = now` every
  frame, capping measured lag at one frame interval, so with a 40ms interval and a
  0.5s tolerance the drop branch was unreachable. A feed that could not keep up
  played in slow motion instead of skipping. Fixed; measured 2.3s→11.4s of video
  per 12s wall-clock with a 200ms/frame consumer.

## Test scripts

Copied into `tests/` (run with `.venv/bin/python tests/<name>.py` from the repo
root): `test_ring.py` (shared-memory round-trip, oversize + pool-exhaustion
fallbacks), `test_e2e.py` (coordinator + 2 workers + inference process, 2 feeds),
`test_detect.py` (parts A/B/C). They lower `inference_imgsz`/`frame_stride` for the
test only — they were written for a 2-core CPU box, so raise those on the GPU host.

All pass except **`test_detect.py` part A, which is a known test artifact, not a
bug**: it samples only the first ~2s of the clip at imgsz 320 and that clip has
sparse detections. At imgsz 1280 the same path returns person (class 0) *and*
phone (class 67). Do not chase it.

## Verified vs not

- **Verified on CPU-only hardware (2 physical cores):** shared-memory ring across
  processes; in-process mode produces boxes + terminal `done`; coordinator mode
  with 1 and 2 workers runs a 964-frame file to `done`; full HTTP path
  (`POST /feeds/stream` → `GET /feeds`); server startup does not block on the 38s
  model warmup.
- **Never verified on GPU:** NVDEC actually engaging, FP16 batch throughput, and
  anything about behaviour at 80 concurrent feeds.

## Other loose ends

- `GET /feeds` reports `frame_index: 0` / `progress: 0.0` for **unwatched** feeds,
  because payloads are only emitted when a viewer is attached or an event fires.
  The dashboard progress looks stuck on idle feeds. Pre-existing; not fixed.
- `supervision.ByteTrack` is deprecated and removed in supervision 0.30, and the
  shared-model design depends on it for per-feed identity. Pinned `<0.30` in
  `requirements.txt`. Needs a plan before that pin has to move.
- Shutdown logs `resource_tracker: There appear to be N leaked semaphore objects`.
  Artifact of daemon processes + `mp.Queue` being terminated; cosmetic, not
  investigated.
- `max_feeds = 80` is an admission limit, not a throughput promise. Aggregate
  demand = feeds × (source_fps / frame_stride); see `docs/ARCHITECTURE.md` §5.
