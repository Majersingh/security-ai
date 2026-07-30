"""End-to-end: coordinator + 2 worker processes + 1 inference process, 2 feeds.

Detection only — the pipeline serves no video (that is module/rawapp.py), so this
checks events, progress and completion, and asserts no payload carries an image.

Runs the real pipeline on input/operator.mp4. imgsz/stride are lowered FOR THE
TEST ONLY (this box is CPU-only, 2 physical cores) — the shipped defaults are
untouched.
"""
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "module" / "core"))

from config import Config
from utils import limit_process_threads, setup_logging

setup_logging("INFO")
limit_process_threads(1, 1)

SCRATCH = Path(__file__).parent / "e2e-out"
VIDEO = str(ROOT / "module" / "input" / "operator.mp4")


def make_cfg() -> Config:
    cfg = Config()
    cfg.inference_imgsz = 320        # CPU-only box; 1280 would take minutes/frame
    cfg.frame_stride = 15            # ~2 processed fps from a 30 fps file
    cfg.num_workers = 2
    cfg.batch_max_size = 4
    cfg.max_feeds = 4
    cfg.log_timing = False
    cfg.events_csv = SCRATCH / "events.csv"
    cfg.snapshots_dir = SCRATCH / "snapshots"
    cfg.output_video = SCRATCH / "annotated.mp4"
    cfg.write_output_video = False
    return cfg


async def main() -> int:
    from workers import WorkerPool

    cfg = make_cfg()
    pool = WorkerPool(cfg).start()
    print(f"[test] pool up: {cfg.num_workers} workers + inference process")

    events_q = pool.subscribe_events()
    ids = []
    for i in range(2):
        fid = pool.add_stream(VIDEO, f"cam-{i}", None, None, None)
        assert fid, "add_stream returned None"
        ids.append(fid)
    print(f"[test] added feeds: {[f[:8] for f in ids]}")

    # No per-feed video queues any more: progress comes from the coordinator's
    # records, violations from the global events channel.
    n_events = 0
    images = 0
    t_end = time.monotonic() + 45
    while time.monotonic() < t_end:
        try:
            while True:
                ev = events_q.get_nowait()
                n_events += len(ev.get("events") or [])
                if ev.get("image"):
                    images += 1          # must never happen: nothing encodes now
        except asyncio.QueueEmpty:
            pass
        if all(f["status"] in ("done", "error") for f in pool.list()):
            break
        await asyncio.sleep(0.2)

    info = {f["feed_id"][:8]: (f["status"], f["frame_index"], f["worker_id"])
            for f in pool.list()}
    print(f"[test] feed status (status, frame_index, worker): {info}")
    print(f"[test] events seen: {n_events} | payloads carrying an image: {images}")

    workers_used = {f["worker_id"] for f in pool.list()}
    print(f"[test] distinct workers used: {sorted(workers_used)}")

    pool.shutdown()
    print("[test] pool shut down")

    recs = pool.list()
    ok = (
        len(workers_used) == 2                       # both workers were used
        and all(f["status"] == "done" for f in recs)  # both feeds ran to completion
        and all(f["frame_index"] > 0 for f in recs)   # progress reached the coordinator
        and images == 0                              # nothing encodes video any more
    )
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    SCRATCH.mkdir(parents=True, exist_ok=True)
    sys.exit(asyncio.run(main()))
