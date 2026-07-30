"""Raw video path: independent of detection, and actually smooth.

  1. a ticket hides the stream URL from the browser
  2. an unknown/expired ticket is refused
  3. raw playback delivers frames at ~SOURCE rate, not detection rate
  4. it works with NO feed running — nothing in the pipeline is involved
  5. the concurrency cap is enforced so a wall's "all" can't swamp the box
"""
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "module"))
sys.path.insert(0, str(ROOT / "module" / "core"))

from config import Config
from utils import limit_process_threads, setup_logging

setup_logging("ERROR")
limit_process_threads(1, 1)

VIDEO = str(ROOT / "module" / "input" / "operator.mp4")


class FakeWS:
    """Collects what stream_raw would send, and can cut the connection."""

    def __init__(self, stop_after=None):
        self.msgs = []
        self.stop_after = stop_after

    async def send_json(self, m):
        self.msgs.append(m)
        if self.stop_after and len([x for x in self.msgs
                                    if x.get("type") == "frame"]) >= self.stop_after:
            raise RuntimeError("client went away")

    def frames(self):
        return [m for m in self.msgs if m.get("type") == "frame"]

    def meta(self):
        return next((m for m in self.msgs if m.get("type") == "meta"), None)

    def error(self):
        return next((m for m in self.msgs if m.get("type") == "error"), None)


async def main() -> int:
    import rawstream

    cfg = Config()
    ok = True

    # 1. ticket hides the URL
    secret = "rtsp://admin:s3cret@10.0.0.5/stream1"
    ticket, ttl = rawstream.issue_ticket(secret)
    hidden = secret not in ticket and rawstream.resolve_ticket(ticket) == secret
    print(f"  1. ticket hides the stream URL ............ {'PASS' if hidden else 'FAIL'}"
          f"  (ticket={ticket[:10]}…, ttl={ttl:.0f}s)")
    ok &= hidden

    # 2. bad ticket refused
    bad = rawstream.resolve_ticket("deadbeef") is None
    print(f"  2. unknown ticket refused ................. {'PASS' if bad else 'FAIL'}")
    ok &= bad

    # 3 + 4. plays at source rate, with NO feed anywhere
    ws = FakeWS(stop_after=40)
    t0 = time.perf_counter()
    await rawstream.stream_raw(ws, VIDEO, cfg, target_fps=0, max_width=640)
    dt = time.perf_counter() - t0
    meta, frames = ws.meta(), ws.frames()
    src_fps = (meta or {}).get("fps") or 0
    got_fps = len(frames) / dt if dt else 0
    # Source-paced: should land near the source rate, well above a detection rate.
    smooth = meta is not None and len(frames) >= 30 and got_fps > src_fps * 0.5
    jpeg_ok = frames and len(base64.b64decode(frames[0]["image"])) > 500
    print(f"  3. plays at ~source rate .................. {'PASS' if smooth else 'FAIL'}"
          f"  (source {src_fps} fps, delivered {got_fps:.1f} fps, "
          f"{len(frames)} frames, decode={meta.get('decode') if meta else '?'})")
    print(f"  4. no feed / no model involved ............ "
          f"{'PASS' if jpeg_ok else 'FAIL'}  (real JPEGs, "
          f"{len(base64.b64decode(frames[0]['image'])) if frames else 0} bytes)")
    ok &= smooth and jpeg_ok

    # 5. concurrency cap
    cfg2 = Config()
    cfg2.raw_max_streams = 1
    held = asyncio.Event()

    class Blocker(FakeWS):
        async def send_json(self, m):
            self.msgs.append(m)
            if m.get("type") == "frame":
                held.set()
                await asyncio.sleep(3)      # hold the slot open

    a = asyncio.create_task(rawstream.stream_raw(Blocker(), VIDEO, cfg2))
    await asyncio.wait_for(held.wait(), timeout=60)
    b = FakeWS()
    await rawstream.stream_raw(b, VIDEO, cfg2)
    err = b.error()
    capped = err is not None and "limit" in (err.get("message") or "")
    print(f"  5. concurrency cap enforced ............... {'PASS' if capped else 'FAIL'}"
          f"  ({(err or {}).get('message','no error returned')[:44]})")
    ok &= capped
    a.cancel()
    try:
        await a
    except (asyncio.CancelledError, Exception):
        pass

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
