"""Measure this GPU's real detection throughput, to set Config.fps_budget.

Run on the GPU host. Times the actual ModelRunner (so preprocessing, the forward
pass and NMS are all included, exactly as the inference process pays them) at each
batch size, back to back with no queue in the way.

    python tests/bench_gpu.py                 # defaults from Config
    python tests/bench_gpu.py 1920 1080 20    # width height iterations

The number to put in `fps_budget` is the aggregate at the batch size you actually
expect. Batch depth grows with feed count, so:
  - a handful of cameras   -> use the low-n row
  - many cameras           -> use the high-n row
Take the sustained (median) column, not the best.
"""
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "module" / "core"))

import numpy as np

from config import Config
from utils import limit_process_threads, physical_cores, setup_logging


def main() -> int:
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 1920
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 1080
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    cfg = Config()
    # Same thread budget the real inference process gets.
    n_threads = int(cfg.infer_threads or 0) or max(
        2, physical_cores() - max(0, int(cfg.num_workers)))
    limit_process_threads(n_threads, n_threads)
    setup_logging("ERROR")

    from inference import ModelRunner

    print(f"model={cfg.model_path.name}  imgsz={cfg.inference_imgsz}  "
          f"frame={width}x{height}  threads={n_threads}  iters={iters}")
    runner = ModelRunner(cfg)          # includes warmup at configured shapes
    print(f"device={runner._device}  precision="
          f"{'fp16' if runner._quantize == 16 else 'fp32'}\n")

    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    print(f"{'batch':>5} {'median':>9} {'best':>9} {'per frame':>10} "
          f"{'aggregate fps':>14}")
    print("-" * 52)
    results = {}
    for n in runner._sizes:
        frames = [frame] * n
        runner.predict(frames)                      # settle this shape
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            runner.predict(frames)
            times.append(time.perf_counter() - t0)
        med, best = statistics.median(times), min(times)
        per_frame = med / n
        agg = n / med
        results[n] = agg
        print(f"{n:>5} {med*1000:>8.1f}ms {best*1000:>8.1f}ms "
              f"{per_frame*1000:>9.1f}ms {agg:>13.0f}")

    print("\nHow to use this:")
    print(f"  Config.fps_budget = <aggregate at your expected batch depth>")
    print(f"  cameras that fit  = fps_budget / (source_fps / frame_stride)")
    best_n = max(results, key=results.get)
    agg = results[best_n]
    for src_fps in (15, 25, 30):
        for stride in (1, 6, 10):
            fits = agg / (src_fps / stride)
            print(f"    at {agg:.0f} fps: {src_fps} fps sources, stride {stride:>2} "
                  f"-> {fits:.0f} cameras")
        break
    print(f"\n  Sustained throughput is BELOW these figures: this loop has no IPC,")
    print(f"  no queue wait and no competing decode. Treat it as the ceiling, and")
    print(f"  set fps_budget somewhat under the row you pick.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
