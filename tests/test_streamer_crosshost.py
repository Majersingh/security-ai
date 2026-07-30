"""Any streamer can play any camera — even one detected on a different host.

Streaming shares nothing with detection (own decode, no feed, no model), so the host
serving video need not be the one analysing the camera. That is what allows a
VIDEO-ONLY host: the streamer registers itself, so nothing else need run there.

  1. camera detected on host-a, only host-b runs a streamer -> host-b serves it
  2. a streamer on the SAME machine as the detector is preferred (proven route)
  3. cameras are `playable` if ANY streamer is up, not just one on their own host
  4. with no streamer anywhere, the error names the command to run
  5. a streamer is never a placement target, and cannot be pinned to
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

PORT = 9451
CENTRAL = f"http://127.0.0.1:{PORT}"


class Stub(BaseHTTPRequestHandler):
    """Detection stub on 9452/9453; raw-ticket stub on 9462/9463."""

    def log_message(self, *a):
        pass

    def _j(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw))); self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/feeds/stream":
            return self._j(200, {"feed_id": f"feed-{self.server.server_address[1]}"})
        if self.path == "/stream/ticket":
            return self._j(200, {"ticket": f"tkt{self.server.server_address[1]}",
                                 "expires_in": 120})
        return self._j(200, {"stopped": True, "applied": True})

    def do_GET(self):
        self._j(200, {"feeds": [], "active": 0, "max_feeds": 50})


def call(path, body=None, method=None):
    req = urllib.request.Request(
        CENTRAL + path, data=json.dumps(body).encode() if body is not None else None,
        method=method or ("POST" if body is not None else "GET"))
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, (json.loads(raw) if raw else {})


def register_detector(mid, port, host):
    return call("/api/modules/register",
                {"id": mid, "url": f"http://127.0.0.1:{port}", "role": "detection",
                 "host": host, "max_feeds": 50, "fps_budget": 150.0})


def register_streamer(sid, port, host):
    """What module/streamer.py sends for itself — no detection involved."""
    return call("/api/modules/register",
                {"id": sid, "url": f"http://127.0.0.1:{port}", "role": "streamer",
                 "host": host, "max_feeds": 0})


def cam_named(name):
    return next(c for c in call("/api/cameras")[1]["cameras"] if c["name"] == name)


def main() -> int:
    import os
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    os.environ.update({"CENTRAL_DB": str(tmp / "x.db"), "CENTRAL_TOKEN": "",
                       "CENTRAL_STRIDE": "10", "CENTRAL_STALE_AFTER": "120"})
    import uvicorn

    from central.app import app

    for port in (9452, 9453, 9462, 9463):
        srv = HTTPServer(("127.0.0.1", port), Stub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    cfg = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()
    for _ in range(120):
        try:
            call("/api/fleet"); break
        except Exception:
            time.sleep(0.1)

    ok = True

    # 4 first: a detector, no streamer anywhere yet.
    register_detector("mod-a", 9452, host="host-a")
    _, r = call("/api/cameras", {"url": "rtsp://c/1", "name": "Cam1",
                                "module_id": "mod-a"})
    cam1 = r["camera_id"]
    st, r = call(f"/api/cameras/{cam1}/stream", {})
    msg = r.get("error", "")
    told = st == 503 and "module.streamer" in msg and "STREAMER_PUBLIC_URL" in msg
    print(f"  4. no streamer -> names the command ...... {'PASS' if told else 'FAIL'}"
          f"  (HTTP {st})")
    ok &= told
    unplayable = cam_named("Cam1")["playable"] is False
    print(f"     and the camera reports playable=False ... {'PASS' if unplayable else 'FAIL'}")
    ok &= unplayable

    # 1. A VIDEO-ONLY host: a streamer on host-b, no detection there at all.
    register_streamer("host-b-streamer", 9463, host="host-b")
    st, r = call(f"/api/cameras/{cam1}/stream", {})
    cross = (st == 200 and r.get("served_by") == "host-b-streamer"
             and r.get("detected_by") == "mod-a"
             and "9463" in r.get("ws_url", ""))
    print(f"  1. detected host-a, served by host-b .... {'PASS' if cross else 'FAIL'}"
          f"  (served_by={r.get('served_by')}, detected_by={r.get('detected_by')})")
    ok &= cross

    # 3. playable now, because SOME streamer is up
    playable = cam_named("Cam1")["playable"] is True
    print(f"  3. playable via another host ............. {'PASS' if playable else 'FAIL'}")
    ok &= playable

    # 2. a streamer co-located with the detector wins (proven route to the camera)
    register_streamer("host-a-streamer", 9462, host="host-a")
    st, r = call(f"/api/cameras/{cam1}/stream", {})
    prefers_local = (st == 200 and r.get("served_by") == "host-a-streamer"
                     and "9462" in r["ws_url"])
    print(f"  2. co-located streamer preferred ......... "
          f"{'PASS' if prefers_local else 'FAIL'}  (served_by={r.get('served_by')})")
    ok &= prefers_local

    # 5. a streamer must never receive cameras
    fleet = call("/api/fleet")[1]
    listed = {m["module_id"] for m in fleet["modules"]}
    not_target = "host-b-streamer" not in listed and "host-a-streamer" not in listed
    st, _ = call("/api/cameras", {"url": "rtsp://c/9", "name": "Bad",
                                 "module_id": "host-a-streamer"})
    rejected = st == 400
    print(f"  5. streamer not placeable / not pinnable . "
          f"{'PASS' if not_target and rejected else 'FAIL'}"
          f"  (absent from placement: {not_target}, pin rejected: {rejected})")
    ok &= not_target and rejected

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
