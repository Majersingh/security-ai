"""Central + module contract, end to end, with a stub module.

Verifies the things that actually matter for horizontal scale:
  1. a module registers and shows up with capacity
  2. a camera added at central is PLACED on that module (central calls its API)
  3. events reported by the module land in central's store
  4. a second module absorbs cameras the first one has no room for
  5. when a module stops heartbeating, its cameras are released for re-placement
  6. the event spool survives central being unreachable, then delivers
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))          # reporting.py
sys.path.insert(0, str(ROOT / "module" / "core"))  # config, utils, ...

CENTRAL = "http://127.0.0.1:9411"
TOKEN = "test-token"


# ----------------------------------------------------------- stub module API
class StubModule(BaseHTTPRequestHandler):
    """Pretends to be module/app.py: accepts /feeds/stream and /feeds/*/stop.

    Feed state is keyed by listening port so two stubs are genuinely independent
    modules with their own capacity — sharing one pool made an earlier version of
    this test blame central for the stub's own 429.
    """

    _by_port: dict = {}
    capacity = 2

    @property
    def feeds(self) -> dict:
        return StubModule._by_port.setdefault(self.server.server_address[1], {})

    def log_message(self, *a):  # silence
        pass

    def _json(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        _ = self.rfile.read(n)
        if self.path == "/feeds/stream":
            feeds = self.feeds
            if len(feeds) >= StubModule.capacity:
                return self._json(429, {"error": "feed limit reached"})
            fid = f"feed{len(feeds)+1:03d}"
            feeds[fid] = True
            return self._json(200, {"feed_id": fid})
        if self.path.endswith("/stop"):
            return self._json(200, {"stopped": True})
        if self.path.endswith("/geometry"):
            return self._json(200, {"applied": True})
        self._json(404, {})

    def do_GET(self):
        self._json(200, {"feeds": [], "active": len(self.feeds),
                         "max_feeds": StubModule.capacity})


def post(path, body, token=TOKEN, base=CENTRAL):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-Module-Token", token)
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


def get(path, base=CENTRAL):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.loads(r.read())


def main() -> int:
    import os
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    os.environ["CENTRAL_DB"] = str(tmp / "central.db")
    os.environ["CENTRAL_TOKEN"] = TOKEN
    os.environ["CENTRAL_STRIDE"] = "6"
    os.environ["CENTRAL_STALE_AFTER"] = "4"      # so the reaper runs fast

    import uvicorn

    from central.app import app

    # stub module on 9412
    for port in (9412, 9413):          # two independent stub modules
        srv = HTTPServer(("127.0.0.1", port), StubModule)
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    cfg = uvicorn.Config(app, host="127.0.0.1", port=9411, log_level="error")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            get("/api/fleet")
            break
        except Exception:
            time.sleep(0.1)

    ok = True

    # 1. register
    post("/api/modules/register", {
        "id": "mod-a", "url": "http://127.0.0.1:9412",
        "gpu": "RTX 4070 Ti SUPER", "max_feeds": 2, "fps_budget": 150.0})
    fleet = get("/api/fleet")
    step = fleet["modules_online"] == 1
    print(f"  1. module registers + online .............. {'PASS' if step else 'FAIL'}")
    ok &= step

    # 2. camera gets placed
    r = post("/api/cameras", {"url": "rtsp://cam/1", "name": "Cam 1"}, token=None)
    cam1 = r["camera_id"]
    cams = get("/api/cameras")["cameras"]
    placed = cams[0]["module_id"] == "mod-a" and cams[0]["status"] == "running"
    # Stub module advertises no raw_url, so it must be unplayable AND say why in a
    # way a user can act on — this string is shown in the dashboard and the wall.
    has_video = cams[0]["playable"] is False
    reason = cams[0].get("playable_reason") or ""
    has_video = has_video and "raw video service" in reason and "rawapp" in reason
    print(f"  2. camera placed on module ................ {'PASS' if placed else 'FAIL'}"
          f"  (unplayable + actionable reason: {has_video})")
    ok &= placed and has_video

    # 3. events ingested
    post("/api/events", {"camera_id": cam1, "module_id": "mod-a", "events": [
        {"timestamp": "00:00:12.500", "frame_number": 375, "person_id": 3,
         "event": "phone_usage", "confidence": 0.81}]})
    evs = get("/api/events")["events"]
    step = len(evs) == 1 and evs[0]["kind"] == "phone_usage" and evs[0]["person_id"] == 3
    print(f"  3. events ingested + queryable ............ {'PASS' if step else 'FAIL'}")
    ok &= step

    # 4. capacity respected, second module absorbs the overflow
    post("/api/cameras", {"url": "rtsp://cam/2", "name": "Cam 2"}, token=None)
    r3 = post("/api/cameras", {"url": "rtsp://cam/3", "name": "Cam 3"}, token=None)
    overflow_unplaced = r3["placed"] is None
    post("/api/modules/register", {
        "id": "mod-b", "url": "http://127.0.0.1:9413",
        "gpu": "RTX 4070 Ti SUPER", "max_feeds": 2, "fps_budget": 150.0})
    time.sleep(0.5)
    cams = {c["name"]: c for c in get("/api/cameras")["cameras"]}
    absorbed = cams["Cam 3"]["module_id"] == "mod-b"
    print(f"  4. full module refused, mod-b absorbed .... "
          f"{'PASS' if overflow_unplaced and absorbed else 'FAIL'}"
          f"  (unplaced first: {overflow_unplaced}, then on mod-b: {absorbed})")
    ok &= overflow_unplaced and absorbed

    # 5. dead module releases its cameras
    print("     waiting for heartbeat timeout ...")
    time.sleep(7)
    cams = {c["name"]: c for c in get("/api/cameras")["cameras"]}
    released = all(c["module_id"] in (None, "mod-a", "mod-b") for c in cams.values())
    any_pending = any(c["status"] in ("pending", "unplaced", "running")
                      for c in cams.values())
    print(f"  5. stale module reaped, cameras released .. "
          f"{'PASS' if released and any_pending else 'FAIL'}")
    ok &= released and any_pending

    # 6. spool survives central being down, then delivers
    from reporting import CentralReporter

    class Cfg:
        events_csv = tmp / "out" / "events.csv"
        max_feeds = 2

    rep = CentralReporter(Cfg(), central_url="http://127.0.0.1:9999",  # nothing there
                          module_id="mod-spool", public_url="http://x", token=TOKEN)
    rep.report_events(cam1, [{"event": "zone_intrusion", "person_id": 9,
                              "confidence": 0.9, "frame_number": 1, "timestamp": "0"}])
    spooled = rep.spool.pending() == 1
    rep._flush_spool()                     # central unreachable -> must NOT drop
    kept = rep.spool.pending() == 1
    rep.central = CENTRAL                  # central comes back
    rep._flush_spool()
    drained = rep.spool.pending() == 0
    evs = get("/api/events")["events"]
    delivered = any(e["kind"] == "zone_intrusion" for e in evs)
    step = spooled and kept and drained and delivered
    print(f"  6. event spool durable then delivers ..... {'PASS' if step else 'FAIL'}"
          f"  (spooled:{spooled} kept-while-down:{kept} drained:{drained} "
          f"in-db:{delivered})")
    ok &= step

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
