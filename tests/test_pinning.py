"""Pinning a camera to a specific module.

  1. auto placement still works when no module is chosen
  2. an explicit module_id is honoured even when another has more headroom
  3. an unknown module_id is rejected, not silently auto-placed
  4. a pin to an OFFLINE module leaves the camera unplaced and says why —
     it must never be quietly relocated, since a pin usually encodes network
     reachability that central cannot see
  5. the pin survives module death: when it comes back, the camera lands there
  6. an old database without the column is migrated, not broken
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

PORT = 9441
CENTRAL = f"http://127.0.0.1:{PORT}"
TOKEN = "pin-token"


class Stub(BaseHTTPRequestHandler):
    _by_port: dict = {}

    @property
    def feeds(self):
        return Stub._by_port.setdefault(self.server.server_address[1], {})

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
            fid = f"f{self.server.server_address[1]}-{len(self.feeds)+1}"
            self.feeds[fid] = True
            return self._j(200, {"feed_id": fid})
        return self._j(200, {"stopped": True, "applied": True})

    def do_GET(self):
        self._j(200, {"feeds": [], "active": len(self.feeds), "max_feeds": 50})


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


def register(mid, port, slots=50, budget=150.0):
    return call("/api/modules/register",
                {"id": mid, "url": f"http://127.0.0.1:{port}",
                 "max_feeds": slots, "fps_budget": budget})


def beat(mid):
    return call(f"/api/modules/{mid}/heartbeat", {"active_feeds": 0, "feeds": []})


def cams():
    return {c["name"]: c for c in call("/api/cameras")[1]["cameras"]}


def main() -> int:
    import os
    import sqlite3
    import tempfile

    tmp = Path(tempfile.mkdtemp())

    # 6. pre-create a DB with the OLD cameras schema (no pinned_module column)
    old = tmp / "old.db"
    con = sqlite3.connect(old)
    con.executescript("""CREATE TABLE cameras (id TEXT PRIMARY KEY, name TEXT NOT NULL,
        url TEXT NOT NULL, module_id TEXT, feed_id TEXT,
        status TEXT NOT NULL DEFAULT 'pending', geometry TEXT,
        source_fps REAL NOT NULL DEFAULT 30, created_at REAL NOT NULL,
        updated_at REAL NOT NULL);""")
    con.commit(); con.close()

    os.environ.update({"CENTRAL_DB": str(old), "CENTRAL_TOKEN": "",
                       "CENTRAL_STRIDE": "10", "CENTRAL_STALE_AFTER": "4"})
    import uvicorn

    from central.app import app

    for port in (9442, 9443):
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
    migrated = "pinned_module" in {r[1] for r in
                                   sqlite3.connect(old).execute(
                                       "PRAGMA table_info(cameras)")}
    print(f"  6. old DB migrated (column added) ......... {'PASS' if migrated else 'FAIL'}")
    ok &= migrated

    register("mod-a", 9442, slots=50)
    register("mod-b", 9443, slots=50)

    # 1. auto
    call("/api/cameras", {"url": "rtsp://c/1", "name": "Auto"})
    auto_ok = cams()["Auto"]["module_id"] in ("mod-a", "mod-b")
    print(f"  1. auto placement still works ............. {'PASS' if auto_ok else 'FAIL'}"
          f"  (-> {cams()['Auto']['module_id']})")
    ok &= auto_ok

    # 2. explicit pin honoured (mod-b, even though mod-a has equal/more room)
    st, r = call("/api/cameras", {"url": "rtsp://c/2", "name": "Pinned",
                                 "module_id": "mod-b"})
    c = cams()["Pinned"]
    pin_ok = c["module_id"] == "mod-b" and c["pinned_module"] == "mod-b"
    print(f"  2. explicit module honoured ............... {'PASS' if pin_ok else 'FAIL'}"
          f"  (-> {c['module_id']}, pinned_to={r.get('pinned_to')})")
    ok &= pin_ok

    # 3. unknown module rejected
    st, r = call("/api/cameras", {"url": "rtsp://c/3", "name": "Bad",
                                 "module_id": "mod-nope"})
    rejected = st == 400 and "Bad" not in cams()
    print(f"  3. unknown module rejected ................ {'PASS' if rejected else 'FAIL'}"
          f"  (HTTP {st})")
    ok &= rejected

    # 4. pin to a module that goes offline -> stays unplaced, NOT relocated
    print("     letting mod-b go stale ...")
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        beat("mod-a")                      # keep A alive, let B die
        time.sleep(1)
    c = cams()["Pinned"]
    held = c["module_id"] is None and c["pinned_module"] == "mod-b" \
        and "pinned" in (c["status"] or "")
    print(f"  4. pinned camera NOT relocated ........... {'PASS' if held else 'FAIL'}"
          f"  (module={c['module_id']}, status='{c['status']}')")
    ok &= held

    # 5. pinned module returns -> camera lands back on it
    register("mod-b", 9443, slots=50)
    back = False
    for _ in range(30):
        if cams()["Pinned"]["module_id"] == "mod-b":
            back = True; break
        time.sleep(0.5)
    print(f"  5. returns to its module when back ....... {'PASS' if back else 'FAIL'}")
    ok &= back

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
