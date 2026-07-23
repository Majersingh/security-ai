"""Batched inference: ONE shared YOLO model + micro-batching across all feeds.

Running one GPU call per feed thrashes a single GPU (see the 20-feed wall). This
service lets every feed submit a frame and `await` its detections; frames that
arrive within a short window are combined into a **single** ``model.predict()``
call, which is far more GPU-efficient. Detection is stateless/shared here;
per-feed tracking (identity) is done by each feed's own ByteTrack downstream.

Usage:
    batcher = BatchInferencer(config).start()      # inside a running event loop
    detections = await batcher.infer(frame)        # sv.Detections (no track ids)
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Tuple

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import Config
from utils import resolve_device

logger = logging.getLogger("operator_monitor")


class BatchInferencer:
    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._device = resolve_device(config.device)
        dev = self._device.lower()
        self._half = ("cuda" in dev) or dev.isdigit()   # FP16 on CUDA
        self._keep = sorted({config.person_class_id, config.phone_class_id})
        self._max_batch = max(1, int(config.batch_max_size))
        self._max_wait = max(1, int(config.batch_max_wait_ms)) / 1000.0

        logger.info("BatchInferencer loading shared model '%s' ...", config.model_path)
        self._model = YOLO(str(config.model_path))
        self._queue: "asyncio.Queue[Tuple[np.ndarray, asyncio.Future]]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

        # The set of batch sizes we'll ever run (powers of two up to max_batch).
        # We pad every batch up to one of these so cuDNN only autotunes a few
        # fixed sizes — otherwise a varying size (4,16,7,...) re-tunes each time,
        # costing ~1s+ per batch and stalling every feed.
        self._sizes = []
        s = 1
        while s < self._max_batch:
            self._sizes.append(s)
            s <<= 1
        self._sizes.append(self._max_batch)

        # Warm up each fixed size once now, so the first runtime batch of each
        # size doesn't pay the autotune spike.
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for n in self._sizes:
            try:
                self._model.predict(source=[dummy] * n, imgsz=config.inference_imgsz,
                                     device=self._device, half=self._half, verbose=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Warmup at batch=%d failed: %s", n, exc)
        logger.info(
            "BatchInferencer ready (device=%s, half=%s, imgsz=%d, batch sizes=%s, wait=%.0fms).",
            self._device, self._half, config.inference_imgsz, self._sizes, self._max_wait * 1000,
        )

    def _pad_size(self, n: int) -> int:
        """Round a batch count up to the nearest fixed (pre-warmed) size."""
        for s in self._sizes:
            if n <= s:
                return s
        return self._max_batch

    def start(self) -> "BatchInferencer":
        if self._task is None:
            self._task = asyncio.create_task(self._loop())
        return self

    async def infer(self, frame: np.ndarray) -> sv.Detections:
        """Submit a frame; returns its detections once its batch has run."""
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put((frame, fut))
        return await fut

    async def _loop(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            frame, fut = await self._queue.get()          # block for the first item
            batch = [(frame, fut)]
            deadline = loop.time() + self._max_wait
            while len(batch) < self._max_batch:            # fill the batch briefly
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break

            frames = [b[0] for b in batch]
            futs = [b[1] for b in batch]
            try:
                t0 = loop.time()
                dets = await loop.run_in_executor(None, self._predict, frames)
                if getattr(self._cfg, "log_timing", False):
                    dt = (loop.time() - t0) * 1000.0
                    logger.info(
                        "BATCH n=%d predict=%.0fms (%.0fms/frame)",
                        len(frames), dt, dt / max(1, len(frames)),
                    )
                for f, d in zip(futs, dets):
                    if not f.done():
                        f.set_result(d)
            except Exception as exc:  # noqa: BLE001 - fail this batch, keep serving
                logger.exception("Batch predict failed (%d frames)", len(frames))
                for f in futs:
                    if not f.done():
                        f.set_exception(exc)

    def _predict(self, frames: List[np.ndarray]) -> List[sv.Detections]:
        """Run ONE batched forward pass, padded to a fixed size, return per-frame
        detections (padding results are discarded)."""
        n = len(frames)
        padded = frames + [frames[-1]] * (self._pad_size(n) - n)  # pad to fixed size
        results = self._model.predict(
            source=padded,
            conf=self._cfg.confidence_threshold,
            iou=self._cfg.iou_threshold,
            imgsz=self._cfg.inference_imgsz,
            device=self._device,
            half=self._half,
            classes=self._keep,
            verbose=False,
        )
        out: List[sv.Detections] = []
        for r in results[:n]:                          # ignore padded frames
            det = sv.Detections.from_ultralytics(r)
            if det.class_id is not None and len(det):
                det = det[np.isin(det.class_id, self._keep)]
            out.append(det)
        return out
