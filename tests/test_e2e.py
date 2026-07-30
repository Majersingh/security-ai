"""End-to-end: coordinator + 2 worker processes + 1 inference process, 2 feeds.

Runs the real pipeline on input/operator.mp4. imgsz/stride are lowered FOR THE
TEST ONLY (this box is CPU-only, 2 physical cores) — the shipped defaults are
untouched.
"""
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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

    # Subscribe as a viewer so the annotate + JPEG-encode path runs too.
    view_qs = {}
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and len(view_qs) < len(ids):
        for fid in ids:
            if fid not in view_qs:
                q = pool.subscribe(fid)
                if q is not None:
                    view_qs[fid] = q
        await asyncio.sleep(0.2)

    frames = {f: 0 for f in ids}
    images = {f: 0 for f in ids}
    types = {}
    terminal = {}
    n_events = 0
    boxes_seen = 0
    t_end = time.monotonic() + 45
    while time.monotonic() < t_end:
        for fid, q in view_qs.items():
            try:
                payload = q.get_nowait()
            except asyncio.QueueEmpty:
                continue
            t = payload.get("type")
            types[t] = types.get(t, 0) + 1
            if t in ("done", "error"):
                terminal[fid[:8]] = (t, payload.get("message", ""))
            if t == "frame":
                frames[fid] += 1
                if payload.get("image"):
                    images[fid] += 1
                boxes_seen += len(payload.get("boxes") or [])
        try:
            while True:
                ev = events_q.get_nowait()
                n_events += len(ev.get("events") or [])
        except asyncio.QueueEmpty:
            pass
        await asyncio.sleep(0.05)

    info = {f["feed_id"][:8]: (f["status"], f["frame_index"], f["worker_id"])
            for f in pool.list()}
    print(f"[test] feed status (status, frame_index, worker): {info}")
    print(f"[test] viewer frames: { {k[:8]: v for k, v in frames.items()} }")
    print(f"[test] with JPEG image: { {k[:8]: v for k, v in images.items()} }")
    print(f"[test] boxes in payloads: {boxes_seen} | events: {n_events}")
    print(f"[test] payload types: {types}")
    print(f"[test] terminal payloads: {terminal}")

    workers_used = {f["worker_id"] for f in pool.list()}
    print(f"[test] distinct workers used: {sorted(workers_used)}")

    pool.shutdown()
    print("[test] pool shut down")

    ok = (
        all(v > 0 for v in frames.values())
        and all(v > 0 for v in images.values())
        and len(workers_used) == 2
        and not any(t[0] == "error" for t in terminal.values())
    )
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    SCRATCH.mkdir(parents=True, exist_ok=True)
    sys.exit(asyncio.run(main()))
