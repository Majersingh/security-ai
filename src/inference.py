"""Central inference service — exactly ONE process owns the GPU.

Every feed, in every worker process, sends its frames here; a single process
holds the one model and combines whatever frames are waiting into one
``predict()`` call. That gives one CUDA context, one copy of the weights, and
batches that actually fill up. (The previous design gave each worker process its
own batcher, so with N workers each batcher saw 1/N of the frames — N CUDA
contexts, N sets of weights, and batches too small to be worth batching.)

Two deployment shapes, one design:

* ``num_workers == 0`` -> :class:`LocalInferencer`. Same batching, in-process, no
  shared memory. The simple path, and the only one that makes sense CPU-only.
* ``num_workers > 0``  -> :class:`InferenceService` in the coordinator spawns the
  inference process; each worker talks to it through an :class:`InferenceClient`.

Frames move over a pool of fixed-size shared-memory slots
(:class:`FrameRing`), not by pickling: a 1080p BGR frame is ~6 MB, and pickling
that through a queue at hundreds of frames/second would cost more than the
inference it feeds. A frame that doesn't fit a slot (or arrives when the pool is
momentarily empty) falls back to the queue, so correctness never depends on the
fast path.

Detection here is **stateless** — identity tracking is per-feed, downstream, in
``FrameProcessor``'s own ``supervision.ByteTrack``.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import queue as pyqueue
import threading
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import supervision as sv

from utils import (
    limit_process_threads, physical_cores, resolve_device, setup_logging,
)

logger = logging.getLogger("operator_monitor")

# Wire formats (plain tuples — these cross a process boundary per frame).
#   request : (worker_id, req_id, slot, shape, dtype, inline_frame_or_None)
#             slot < 0 means the frame is carried inline in the message.
#   response: (req_id, ok, payload)
#             ok -> payload is (xyxy, confidence, class_id); else an error string.
Request = Tuple[int, int, int, Optional[tuple], Optional[str], Optional[np.ndarray]]
Response = Tuple[int, bool, Any]

_SHUTDOWN = None


# --------------------------------------------------------------- the one model

class ModelRunner:
    """The single YOLO model: load it, warm fixed shapes, predict a list.

    Shared by both deployment shapes so "how we call the model" is defined once.
    """

    def __init__(self, cfg) -> None:
        from ultralytics import YOLO      # heavy; only imported where a model lives

        self._cfg = cfg
        self._device = resolve_device(cfg.device)
        dev = self._device.lower()
        self._half = ("cuda" in dev) or dev.isdigit()      # FP16 on CUDA
        self._keep = sorted({cfg.person_class_id, cfg.phone_class_id})
        self.max_batch = max(1, int(cfg.batch_max_size))

        logger.info("Loading the shared model '%s' ...", cfg.model_path)
        self._model = YOLO(str(cfg.model_path))

        # Only ever run these batch sizes (powers of two up to max_batch) and pad
        # up to them, so cuDNN autotunes a handful of fixed shapes instead of
        # re-tuning on every new size (~1s+ each, stalling every feed).
        self._sizes: List[int] = []
        s = 1
        while s < self.max_batch:
            self._sizes.append(s)
            s <<= 1
        self._sizes.append(self.max_batch)

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for n in self._sizes:
            try:
                self._predict_raw([dummy] * n)
            except Exception as exc:  # noqa: BLE001 - warmup is optional
                logger.warning("Warmup at batch=%d failed: %s", n, exc)
        logger.info(
            "Inference ready (device=%s, half=%s, imgsz=%d, batch sizes=%s).",
            self._device, self._half, cfg.inference_imgsz, self._sizes,
        )

    def pad_size(self, n: int) -> int:
        """Round a frame count up to the nearest pre-warmed batch size."""
        for s in self._sizes:
            if n <= s:
                return s
        return self.max_batch

    def _predict_raw(self, frames: Sequence[np.ndarray]):
        return self._model.predict(
            source=list(frames),
            conf=self._cfg.confidence_threshold,
            iou=self._cfg.iou_threshold,
            imgsz=self._cfg.inference_imgsz,
            device=self._device,
            half=self._half,
            classes=self._keep,
            verbose=False,
        )

    def predict(self, frames: Sequence[np.ndarray]) -> List[sv.Detections]:
        """One padded forward pass -> per-frame detections (padding discarded)."""
        n = len(frames)
        padded = list(frames) + [frames[-1]] * (self.pad_size(n) - n)
        results = self._predict_raw(padded)
        out: List[sv.Detections] = []
        for r in results[:n]:
            det = sv.Detections.from_ultralytics(r)
            if det.class_id is not None and len(det):
                det = det[np.isin(det.class_id, self._keep)]
            out.append(det)
        return out


def detections_to_wire(det: sv.Detections) -> Tuple[Any, Any, Any]:
    """Reduce detections to the three arrays worth sending (a few KB)."""
    if len(det) == 0:
        return None, None, None
    return det.xyxy, det.confidence, det.class_id


def detections_from_wire(payload: Tuple[Any, Any, Any]) -> sv.Detections:
    """Rebuild detections on the worker side."""
    xyxy, conf, cls = payload
    if xyxy is None or len(xyxy) == 0:
        return sv.Detections.empty()
    # ByteTrack needs confidences; synthesize them if the model somehow omitted.
    if conf is None:
        conf = np.ones(len(xyxy), dtype=np.float32)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)


# ------------------------------------------------------- shared-memory transport

@dataclass
class RingSpec:
    """Picklable handle to a :class:`FrameRing` (what a child process needs)."""

    name: str
    n_slots: int
    slot_bytes: int
    free_q: Any          # mp.Queue of free slot indices


class FrameRing:
    """Pool of fixed-size shared-memory slots for frame handoff.

    A slot is claimed by the producer, filled, and handed to the inference
    process by index; the *consumer* returns it to the free pool once it has
    finished reading, which is what keeps a producer from overwriting a frame
    that is still in flight.
    """

    def __init__(self, spec: RingSpec, shm: shared_memory.SharedMemory) -> None:
        self.spec = spec
        self.slot_bytes = spec.slot_bytes
        self.n_slots = spec.n_slots
        self._shm = shm
        self._arr = np.ndarray((shm.size,), dtype=np.uint8, buffer=shm.buf)

    @classmethod
    def create(cls, cfg, ctx) -> "FrameRing":
        slot_bytes = (
            int(cfg.infer_slot_max_height) * int(cfg.infer_slot_max_width) * 3
        )
        n_slots = max(4, int(cfg.infer_slots))
        shm = shared_memory.SharedMemory(create=True, size=slot_bytes * n_slots)
        free_q = ctx.Queue()
        for i in range(n_slots):
            free_q.put(i)
        logger.info(
            "Frame ring: %d slots x %.1f MB = %.0f MB shared.",
            n_slots, slot_bytes / 1e6, slot_bytes * n_slots / 1e6,
        )
        return cls(RingSpec(shm.name, n_slots, slot_bytes, free_q), shm)

    @classmethod
    def attach(cls, spec: RingSpec) -> "FrameRing":
        return cls(spec, shared_memory.SharedMemory(name=spec.name))

    def acquire(self, timeout: float = 2.0) -> int:
        """Claim a free slot index. Raises ``queue.Empty`` if none frees up."""
        return self.spec.free_q.get(timeout=timeout)

    def release(self, slot: int) -> None:
        try:
            self.spec.free_q.put(slot)
        except Exception:  # noqa: BLE001 - shutting down
            pass

    def write(self, slot: int, frame: np.ndarray) -> Tuple[tuple, str]:
        """Copy a frame into a slot. Raises ``ValueError`` if it doesn't fit."""
        if frame.nbytes > self.slot_bytes:
            raise ValueError(
                f"frame {frame.shape} ({frame.nbytes} B) exceeds slot "
                f"({self.slot_bytes} B)"
            )
        flat = np.ascontiguousarray(frame).reshape(-1).view(np.uint8)
        off = slot * self.slot_bytes
        self._arr[off:off + flat.size] = flat
        return tuple(frame.shape), frame.dtype.str

    def read(self, slot: int, shape: tuple, dtype: str) -> np.ndarray:
        """View (no copy) of a slot's frame. Valid until the slot is released."""
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        off = slot * self.slot_bytes
        return self._arr[off:off + nbytes].view(dtype).reshape(shape)

    def close(self) -> None:
        try:
            self._arr = None
            self._shm.close()
        except Exception:  # noqa: BLE001
            pass

    def unlink(self) -> None:
        """Creator only: destroy the shared block."""
        self.close()
        try:
            self._shm.unlink()
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------- the inference process

def _inference_process(cfg, spec: RingSpec, req_q, resp_qs: List[Any]) -> None:
    """Entry point: batch whatever is waiting, run it, answer the requester."""
    # Deliberately NOT capped to 1 like the workers: this process does the CPU-side
    # preprocessing (letterbox/convert/stack) for every frame of every batch, and
    # that work is serialized against the GPU. One thread here idles the GPU.
    n_threads = int(getattr(cfg, "infer_threads", 0) or 0)
    if n_threads <= 0:
        n_threads = max(2, physical_cores() - max(0, int(getattr(cfg, "num_workers", 0))))
    limit_process_threads(n_threads, n_threads)
    setup_logging(getattr(cfg, "log_level", "INFO"))
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    log = logging.getLogger("operator_monitor")

    ring = FrameRing.attach(spec)
    try:
        runner = ModelRunner(cfg)
    except Exception:  # noqa: BLE001 - nothing can work without a model
        log.exception("Inference process failed to load the model")
        ring.close()
        return

    max_wait = max(1, int(cfg.batch_max_wait_ms)) / 1000.0
    log.info("Inference process up (batch<=%d, wait=%.0fms, threads=%d).",
             runner.max_batch, max_wait * 1000, n_threads)
    stopping = False
    while not stopping:
        try:
            first = req_q.get()
        except (EOFError, OSError):
            break
        if first is _SHUTDOWN:
            break

        batch: List[Request] = [first]
        deadline = time.monotonic() + max_wait
        while len(batch) < runner.max_batch:          # brief window to fill up
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                nxt = req_q.get(timeout=remaining)
            except pyqueue.Empty:
                break
            except (EOFError, OSError):
                stopping = True
                break
            if nxt is _SHUTDOWN:
                stopping = True                        # serve this batch, then stop
                break
            batch.append(nxt)

        frames = [
            inline if slot < 0 else ring.read(slot, shape, dtype)
            for (_w, _r, slot, shape, dtype, inline) in batch
        ]
        try:
            t0 = time.monotonic()
            dets = runner.predict(frames)
            if getattr(cfg, "log_timing", False):
                dt = (time.monotonic() - t0) * 1000.0
                log.info("BATCH n=%d predict=%.0fms (%.0fms/frame)",
                         len(frames), dt, dt / max(1, len(frames)))
            # Convert BEFORE releasing slots: ultralytics keeps a reference to the
            # source array (Results.orig_img), which points into shared memory.
            wire = [detections_to_wire(d) for d in dets]
            for (wid, rid, *_), payload in zip(batch, wire):
                _put(resp_qs[wid], (rid, True, payload))
        except Exception as exc:  # noqa: BLE001 - fail this batch, keep serving
            log.exception("Batch predict failed (%d frames)", len(frames))
            for (wid, rid, *_) in batch:
                _put(resp_qs[wid], (rid, False, str(exc)))
        finally:
            frames = None                     # drop views before reusing slots
            for (_w, _r, slot, *_rest) in batch:
                if slot >= 0:
                    ring.release(slot)

    ring.close()
    log.info("Inference process exiting.")


def _put(q, msg) -> None:
    try:
        q.put(msg)
    except Exception:  # noqa: BLE001 - worker gone
        pass


class InferenceService:
    """Coordinator side: owns the inference process and the shared frame ring."""

    def __init__(self, cfg, ctx, n_workers: int) -> None:
        self._cfg = cfg
        self._ctx = ctx
        self._ring = FrameRing.create(cfg, ctx)
        self._req_q = ctx.Queue(maxsize=max(64, 4 * int(cfg.batch_max_size)))
        self._resp_qs = [ctx.Queue() for _ in range(n_workers)]
        self._proc = None

    def start(self) -> "InferenceService":
        self._proc = self._ctx.Process(
            target=_inference_process,
            args=(self._cfg, self._ring.spec, self._req_q, self._resp_qs),
            name="cctv-inference", daemon=True,
        )
        self._proc.start()
        logger.info("Inference process started (pid=%s).", self._proc.pid)
        return self

    def client_args(self, worker_id: int) -> tuple:
        """The (spec, req_q, resp_q) triple a worker process needs."""
        return (self._ring.spec, self._req_q, self._resp_qs[worker_id])

    def shutdown(self) -> None:
        try:
            self._req_q.put(_SHUTDOWN)
        except Exception:  # noqa: BLE001
            pass
        if self._proc is not None:
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.terminate()
        self._ring.unlink()


class InferenceClient:
    """Worker-side handle: ``await client.infer(frame) -> sv.Detections``.

    Same interface as :class:`LocalInferencer`, so ``FeedManager`` neither knows
    nor cares whether the GPU is in this process or another one.
    """

    def __init__(self, worker_id: int, spec: RingSpec, req_q, resp_q) -> None:
        self._wid = worker_id
        self._ring = FrameRing.attach(spec)
        self._req_q = req_q
        self._resp_q = resp_q
        self._ids = itertools.count()
        self._pending: Dict[int, asyncio.Future] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = threading.Event()
        self._reader: Optional[threading.Thread] = None

    def start(self) -> "InferenceClient":
        self._loop = asyncio.get_event_loop()
        self._reader = threading.Thread(
            target=self._read_loop, name="infer-responses", daemon=True
        )
        self._reader.start()
        return self

    async def infer(self, frame: np.ndarray) -> sv.Detections:
        """Submit one frame and await its detections."""
        loop = asyncio.get_event_loop()
        req_id = next(self._ids)
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        try:
            # Claiming a slot and copying the frame both block -> off the loop.
            await loop.run_in_executor(None, self._submit, req_id, frame)
        except BaseException:
            self._pending.pop(req_id, None)
            raise
        return await fut

    def _submit(self, req_id: int, frame: np.ndarray) -> None:
        slot, inline = -1, None
        shape, dtype = tuple(frame.shape), frame.dtype.str
        try:
            slot = self._ring.acquire()
            shape, dtype = self._ring.write(slot, frame)
        except (pyqueue.Empty, ValueError) as exc:
            # Pool momentarily empty, or the frame is bigger than a slot: carry it
            # inline instead. Slower, but never wrong.
            if slot >= 0:
                self._ring.release(slot)
                slot = -1
            inline = frame
            logger.debug("Inline frame fallback: %s", exc)
        self._req_q.put((self._wid, req_id, slot, shape, dtype, inline))

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._resp_q.get(timeout=0.5)
            except pyqueue.Empty:
                continue
            except (EOFError, OSError):
                break
            if msg is _SHUTDOWN:
                break
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._resolve, *msg)

    def _resolve(self, req_id: int, ok: bool, payload) -> None:
        fut = self._pending.pop(req_id, None)
        if fut is None or fut.done():
            return
        if ok:
            fut.set_result(detections_from_wire(payload))
        else:
            fut.set_exception(RuntimeError(f"inference failed: {payload}"))

    def shutdown(self) -> None:
        self._stop.set()
        self._ring.close()


class LocalInferencer:
    """In-process batcher for ``num_workers == 0``: same batching, no IPC."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._runner = ModelRunner(cfg)
        self._max_wait = max(1, int(cfg.batch_max_wait_ms)) / 1000.0
        self._queue: "asyncio.Queue[Tuple[np.ndarray, asyncio.Future]]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    def start(self) -> "LocalInferencer":
        if self._task is None:
            self._task = asyncio.create_task(self._loop())
        return self

    async def infer(self, frame: np.ndarray) -> sv.Detections:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put((frame, fut))
        return await fut

    async def _loop(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            frame, fut = await self._queue.get()
            batch = [(frame, fut)]
            deadline = loop.time() + self._max_wait
            while len(batch) < self._runner.max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(),
                                                        timeout=remaining))
                except asyncio.TimeoutError:
                    break

            frames = [b[0] for b in batch]
            futs = [b[1] for b in batch]
            try:
                t0 = loop.time()
                dets = await loop.run_in_executor(None, self._runner.predict, frames)
                if getattr(self._cfg, "log_timing", False):
                    dt = (loop.time() - t0) * 1000.0
                    logger.info("BATCH n=%d predict=%.0fms (%.0fms/frame)",
                                len(frames), dt, dt / max(1, len(frames)))
                for f, d in zip(futs, dets):
                    if not f.done():
                        f.set_result(d)
            except Exception as exc:  # noqa: BLE001 - fail this batch, keep serving
                logger.exception("Batch predict failed (%d frames)", len(frames))
                for f in futs:
                    if not f.done():
                        f.set_exception(exc)

    def shutdown(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
