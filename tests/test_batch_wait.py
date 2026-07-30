"""Does `batch_max_wait_ms` add latency when batches can't fill?

Runs the real coordinator + worker + inference process with ONE feed (so a batch
can never fill) at wait=12ms vs wait=0ms, and compares end-to-end detect latency.
"""
import asyncio
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "module" / "core"))

from config import Config
from utils import limit_process_threads, setup_logging

setup_logging("ERROR")
limit_process_threads(1, 1)

SCRATCH = Path("/tmp/claude-1000/-home-am-lp-04-security-ai/"
               "7d37a8fe-088c-4790-947f-f743b20c9a62/scratchpad/bw")
VIDEO = str(ROOT / "module" / "input" / "operator.mp4")


def cfg_for(wait_ms: int) -> Config:
    cfg = Config()
    cfg.inference_imgsz = 320          # CPU box; keep predict small so the
    cfg.frame_stride = 10              #   wait window is visible against it
    cfg.batch_max_size = 16
    cfg.batch_max_wait_ms = wait_ms
    cfg.num_workers = 1
    cfg.max_feeds = 2
    cfg.log_timing = False
    cfg.events_csv = SCRATCH / f"w{wait_ms}" / "events.csv"
    cfg.snapshots_dir = SCRATCH / f"w{wait_ms}" / "snapshots"
    cfg.write_output_video = False
    return cfg


async def run(wait_ms: int, n_frames: int = 25) -> list:
    """Measure per-frame detect latency through the real IPC path."""
    from inference import InferenceClient, InferenceService
    import multiprocessing as mp
    import numpy as np

    cfg = cfg_for(wait_ms)
    ctx = mp.get_context("spawn")
    svc = InferenceService(cfg, ctx, 1).start()
    client = InferenceClient(0, *svc.client_args(0)).start()

    frame = np.random.randint(0, 255, (478, 848, 3), dtype=np.uint8)
    await client.infer(frame)                      # absorb model load + warmup
    lat = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        await client.infer(frame)                  # ONE in flight, like one feed
        lat.append((time.perf_counter() - t0) * 1000.0)
    client.shutdown()
    svc.shutdown()
    return lat


async def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    results = {}
    for wait_ms in (12, 0):
        lat = await run(wait_ms)
        results[wait_ms] = lat
        print(f"batch_max_wait_ms={wait_ms:2d}: median detect "
              f"{statistics.median(lat):6.1f} ms  (min {min(lat):5.1f}, "
              f"max {max(lat):6.1f}, n={len(lat)})")
    saved = statistics.median(results[12]) - statistics.median(results[0])
    print(f"\nlatency removed per frame: {saved:.1f} ms")
    # The window can only ever ADD latency when a batch cannot fill.
    ok = saved > 0
    print("RESULT:", "PASS" if ok else "FAIL (expected wait=0 to be faster)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
