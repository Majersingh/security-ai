"""Preview-then-configure flow: probe a stream, then add it WITH geometry.

  1. module POST /feeds/probe returns a still + TRUE source dimensions
  2. probing creates NO feed and consumes no max_feeds slot
  3. central POST /api/cameras/probe proxies via an online module
  4. central POST /api/cameras {url,name,geometry} starts a feed already configured
  5. the module received the geometry (engine applies it, not left unconfigured)
  6. a bad URL fails cleanly rather than hanging
"""
import base64
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CENTRAL_PORT, MODULE_PORT = 9431, 9432
CENTRAL = f"http://127.0.0.1:{CENTRAL_PORT}"
MODULE = f"http://127.0.0.1:{MODULE_PORT}"
TOKEN = "prev-token"


def call(url, body=None, method=None, timeout=60):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode() if body is not None else None,
        method=method or ("POST" if body is not None else "GET"))
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, (json.loads(raw) if raw else {})


def wait_for(fn, timeout=240, interval=1.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if fn():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def main() -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    os.environ.update({
        "CENTRAL_DB": str(tmp / "c.db"), "CENTRAL_TOKEN": TOKEN,
        "CENTRAL_STRIDE": "10", "CENTRAL_STALE_AFTER": "120",
    })
    import uvicorn

    from central.app import app as central_app

    cfg = uvicorn.Config(central_app, host="127.0.0.1", port=CENTRAL_PORT,
                         log_level="error")
    threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()
    wait_for(lambda: call(f"{CENTRAL}/api/fleet")[0] == 200, 30)

    env = dict(os.environ)
    env.update({"CENTRAL_URL": CENTRAL, "MODULE_TOKEN": TOKEN,
                "MODULE_ID": "mod-prev", "MODULE_PUBLIC_URL": MODULE,
                "PYTHONPATH": str(ROOT)})
    proc = subprocess.Popen(
        [str(ROOT / ".venv/bin/python"), "-m", "uvicorn", "module.app:app",
         "--host", "127.0.0.1", "--port", str(MODULE_PORT), "--log-level", "error"],
        cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)

    ok = True
    video = str(ROOT / "module" / "input" / "operator.mp4")
    try:
        if not wait_for(lambda: call(f"{MODULE}/feeds")[0] == 200, 240):
            print("module never came up"); return 1

        # 1. probe returns a still + true dimensions
        st, r = call(f"{MODULE}/feeds/probe", {"url": video})
        good = (st == 200 and r.get("width") == 848 and r.get("height") == 478
                and len(base64.b64decode(r.get("image", ""))) > 1000)
        print(f"  1. module probe -> still + true size ...... {'PASS' if good else 'FAIL'}"
              f"  ({r.get('width')}x{r.get('height')} @ {r.get('fps')} fps, "
              f"jpeg {len(r.get('image',''))} b64 chars)")
        ok &= good

        # 2. probing created no feed
        _, feeds = call(f"{MODULE}/feeds")
        clean = feeds.get("active") == 0
        print(f"  2. probe created NO feed .................. {'PASS' if clean else 'FAIL'}"
              f"  (active={feeds.get('active')})")
        ok &= clean

        # 3. central proxies the probe
        st, r = call(f"{CENTRAL}/api/cameras/probe", {"url": video})
        proxied = st == 200 and r.get("width") == 848 and r.get("probed_by") == "mod-prev"
        print(f"  3. central proxies probe ................. {'PASS' if proxied else 'FAIL'}"
              f"  (probed_by={r.get('probed_by')})")
        ok &= proxied

        # 4+5. add WITH geometry -> feed starts already configured
        geom = {"zone_polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                "line_start": [0.2, 0.5], "line_end": [0.8, 0.5]}
        st, r = call(f"{CENTRAL}/api/cameras",
                     {"url": video, "name": "Preview Cam", "geometry": geom})
        cam_id = r.get("camera_id")
        placed = wait_for(lambda: next(
            c for c in call(f"{CENTRAL}/api/cameras")[1]["cameras"]
            if c["id"] == cam_id)["module_id"] == "mod-prev", 90)
        cam = next(c for c in call(f"{CENTRAL}/api/cameras")[1]["cameras"]
                   if c["id"] == cam_id)
        stored = cam.get("geometry") or {}
        kept = stored.get("zone_polygon") == geom["zone_polygon"] and \
            stored.get("line_start") == geom["line_start"]
        print(f"  4. added WITH geometry, placed ........... {'PASS' if placed else 'FAIL'}")
        print(f"  5. geometry persisted on the camera ...... {'PASS' if kept else 'FAIL'}"
              f"  (zone {len(stored.get('zone_polygon') or [])} pts, "
              f"line {'yes' if stored.get('line_start') else 'no'})")
        ok &= placed and kept

        # 6. a bad URL fails fast and clearly
        t0 = time.monotonic()
        st, r = call(f"{MODULE}/feeds/probe", {"url": "rtsp://127.0.0.1:1/nope"},
                     timeout=60)
        dt = time.monotonic() - t0
        clean_fail = st in (400, 502, 504) and "error" in r and dt < 40
        print(f"  6. bad URL fails cleanly ................. {'PASS' if clean_fail else 'FAIL'}"
              f"  (HTTP {st} in {dt:.1f}s: {str(r.get('error'))[:52]})")
        ok &= clean_fail
    finally:
        proc.terminate()
        try:
            out = proc.communicate(timeout=15)[0] or ""
        except subprocess.TimeoutExpired:
            proc.kill(); out = proc.communicate()[0] or ""
        for line in out.splitlines():
            if any(k in line for k in ("Probed", "BehaviorEngine", "Traceback")):
                print(f"     module | {line.strip()[:104]}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
