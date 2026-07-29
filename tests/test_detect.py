"""Does the centralized model actually detect, and does 'done' propagate?

Part A: ModelRunner directly on real frames -> expect person detections.
Part B: in-process FeedManager (num_workers=0) on the full file -> expect boxes
        in payloads and a terminal 'done'.
Part C: coordinator mode, one feed, run past EOF -> expect status 'done'.
"""
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path("/home/am-lp-04/security-ai")
sys.path.insert(0, str(ROOT / "src"))

from config import Config
from utils import limit_process_threads, setup_logging

setup_logging("WARNING")
limit_process_threads(1, 1)

SCRATCH = Path(__file__).parent / "detect-out"
VIDEO = str(ROOT / "input" / "operator.mp4")


def base_cfg() -> Config:
    cfg = Config()
    cfg.inference_imgsz = 320
    cfg.frame_stride = 15
    cfg.batch_max_size = 4
    cfg.log_timing = False
    cfg.events_csv = SCRATCH / "events.csv"
    cfg.snapshots_dir = SCRATCH / "snapshots"
    cfg.write_output_video = False
    return cfg


def part_a() -> bool:
    from inference import ModelRunner
    from sources import StreamURLSource

    cfg = base_cfg()
    runner = ModelRunner(cfg)
    src = StreamURLSource(VIDEO, target_fps=1000).start()   # no pacing
    frames, total = [], 0
    for idx, img in src.frames():
        if idx % 20 == 0:
            frames.append(img)
        total += 1
        if len(frames) >= 4:
            break
    src.close()
    dets = runner.predict(frames)
    counts = [len(d) for d in dets]
    classes = sorted({int(c) for d in dets if d.class_id is not None for c in d.class_id})
    print(f"[A] file has >={total} frames, {src.width}x{src.height}")
    print(f"[A] detections per frame: {counts}, classes seen: {classes}")
    return sum(counts) > 0


async def part_b() -> bool:
    from feeds import FeedManager
    from sources import StreamURLSource

    cfg = base_cfg()
    mgr = FeedManager(cfg)                      # LocalInferencer, no IPC
    src = StreamURLSource(VIDEO, target_fps=1000, hw_decode=cfg.hw_decode).start()
    feed = mgr.create(src, cfg, None, None, None, name="local", kind="stream",
                      emit_image=False)        # boxes-only payloads
    seen = {"frame": 0, "done": 0, "error": 0}
    boxes = 0

    async def on_update(p):
        nonlocal boxes
        t = p.get("type")
        seen[t] = seen.get(t, 0) + 1
        boxes += len(p.get("boxes") or [])

    await asyncio.wait_for(feed.run(on_update), timeout=180)
    print(f"[B] payload types: {seen}, total boxes: {boxes}, status={feed.info.status}")
    mgr.shutdown()
    return seen["frame"] > 0 and boxes > 0 and seen["done"] == 1


async def part_c() -> bool:
    from workers import WorkerPool

    cfg = base_cfg()
    cfg.num_workers = 1
    cfg.max_feeds = 2
    pool = WorkerPool(cfg).start()
    fid = pool.add_stream(VIDEO, "cam", None, None, None)
    q = None
    deadline = time.monotonic() + 90
    got = []
    while time.monotonic() < deadline:
        if q is None:
            q = pool.subscribe(fid)
        else:
            try:
                p = q.get_nowait()
                got.append(p.get("type"))
                if p.get("type") in ("done", "error"):
                    break
            except asyncio.QueueEmpty:
                pass
        await asyncio.sleep(0.05)
    rec = pool.get_info(fid)
    print(f"[C] payload types seen: {sorted(set(got))}")
    print(f"[C] coordinator status={rec['status']} frame_index={rec['frame_index']}")
    pool.shutdown()
    return rec["status"] == "done"


async def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    a = part_a()
    b = await part_b()
    c = await part_c()
    print(f"RESULT A(detects)={'PASS' if a else 'FAIL'} "
          f"B(local+done)={'PASS' if b else 'FAIL'} "
          f"C(coord done)={'PASS' if c else 'FAIL'}")
    return 0 if (a and b and c) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
