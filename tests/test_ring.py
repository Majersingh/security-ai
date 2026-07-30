"""Shared-memory frame ring: cross-process round-trip + fallback paths."""
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "module" / "core"))

import numpy as np

from config import Config
from inference import FrameRing


def child(spec, req_q, resp_q):
    ring = FrameRing.attach(spec)
    for _ in range(3):
        slot, shape, dtype, checksum = req_q.get()
        got = ring.read(slot, shape, dtype)
        resp_q.put(("read", int(got.sum()), got.shape, checksum))
        ring.release(slot)
    ring.close()


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    cfg = Config()
    cfg.infer_slots = 8
    cfg.infer_slot_max_height, cfg.infer_slot_max_width = 240, 320
    ring = FrameRing.create(cfg, ctx)
    print(f"ring: {ring.n_slots} slots x {ring.slot_bytes} B")

    req_q, resp_q = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=child, args=(ring.spec, req_q, resp_q))
    p.start()

    ok = True
    for i in range(3):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        slot = ring.acquire()
        shape, dtype = ring.write(slot, frame)
        req_q.put((slot, shape, dtype, int(frame.sum())))
        tag, got_sum, got_shape, expect = resp_q.get(timeout=30)
        same = got_sum == expect and tuple(got_shape) == frame.shape
        ok &= same
        print(f"  frame {i}: shape {got_shape} sum {got_sum} == {expect} -> {same}")
    p.join(timeout=10)

    # A frame larger than a slot must raise (the caller then sends it inline).
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    slot = ring.acquire()
    try:
        ring.write(slot, big)
        print("  oversize: NO raise -> FAIL")
        ok = False
    except ValueError as exc:
        print(f"  oversize raises ValueError -> PASS ({str(exc)[:50]}...)")
    ring.release(slot)

    # Slots must be reusable: drain and refill the whole pool twice.
    for _ in range(2):
        slots = [ring.acquire() for _ in range(ring.n_slots)]
        for s in slots:
            ring.release(s)
    print(f"  pool drain/refill x2 -> PASS")

    ring.unlink()
    print("RESULT:", "PASS" if ok else "FAIL")
