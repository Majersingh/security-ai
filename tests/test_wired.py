"""The real module wired to central — no stubs on the module side.

Starts central, then starts module/app.py as a subprocess with CENTRAL_URL set,
and checks the module actually reports in and accepts a placed camera:

  1. the real module registers itself with central (module -> central)
  2. heartbeats keep it online
  3. a camera added at CENTRAL starts a real feed on the module (central -> module)
  4. the module learns central's camera_id and reports it back in heartbeats
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CENTRAL_PORT = 9421
MODULE_PORT = 9422
CENTRAL = f"http://127.0.0.1:{CENTRAL_PORT}"
TOKEN = "wire-token"


def get(url):
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())


def post(url, body, token=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-Module-Token", token)
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


def wait_for(fn, timeout=180, interval=1.0):
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
    os.environ["CENTRAL_DB"] = str(tmp / "central.db")
    os.environ["CENTRAL_TOKEN"] = TOKEN
    os.environ["CENTRAL_STRIDE"] = "10"
    os.environ["CENTRAL_STALE_AFTER"] = "60"

    import uvicorn

    from central.app import app as central_app

    cfg = uvicorn.Config(central_app, host="127.0.0.1", port=CENTRAL_PORT,
                         log_level="error")
    threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()
    if not wait_for(lambda: get(f"{CENTRAL}/api/fleet") is not None, 30):
        print("central did not start"); return 1

    # Real module, as a subprocess, pointed at central. Small imgsz so warmup on a
    # CPU-only box finishes in reasonable time.
    env = dict(os.environ)
    env.update({
        "CENTRAL_URL": CENTRAL,
        "MODULE_TOKEN": TOKEN,
        "MODULE_ID": "mod-real",
        "MODULE_PUBLIC_URL": f"http://127.0.0.1:{MODULE_PORT}",
        "PYTHONPATH": str(ROOT),
    })
    proc = subprocess.Popen(
        [str(ROOT / ".venv/bin/python"), "-m", "uvicorn", "module.app:app",
         "--host", "127.0.0.1", "--port", str(MODULE_PORT), "--log-level", "error"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    ok = True
    try:
        # 1 + 2. module registers itself and shows online
        registered = wait_for(
            lambda: any(m["id"] == "mod-real" and m["online"]
                        for m in get(f"{CENTRAL}/api/fleet")["modules"]
                        if "id" in m) or
                    any(m["module_id"] == "mod-real" and m["online"]
                        for m in get(f"{CENTRAL}/api/fleet")["modules"]),
            timeout=240)
        print(f"  1. real module self-registers ............. "
              f"{'PASS' if registered else 'FAIL'}")
        ok &= registered
        if not registered:
            return 1

        fleet = get(f"{CENTRAL}/api/fleet")
        mine = [m for m in fleet["modules"] if m["module_id"] == "mod-real"][0]
        print(f"     reported: slots={mine['slots_total']} "
              f"fps_budget={mine['fps_budget']}")

        # 3. camera added at CENTRAL lands on the real module
        video = str(ROOT / "module" / "input" / "operator.mp4")
        r = post(f"{CENTRAL}/api/cameras", {"url": video, "name": "Wired Cam"})
        cam_id = r["camera_id"]
        placed = wait_for(
            lambda: next(c for c in get(f"{CENTRAL}/api/cameras")["cameras"]
                         if c["id"] == cam_id)["module_id"] == "mod-real", 60)
        cam = next(c for c in get(f"{CENTRAL}/api/cameras")["cameras"]
                   if c["id"] == cam_id)
        print(f"  2. central placed camera on it ............ "
              f"{'PASS' if placed else 'FAIL'}  (feed_id={cam.get('feed_id')}, "
              f"playable={cam.get('playable')})")
        ok &= placed

        # 4. module reports OUR camera_id back in its heartbeat -> status tracks
        tracked = wait_for(
            lambda: next(c for c in get(f"{CENTRAL}/api/cameras")["cameras"]
                         if c["id"] == cam_id)["status"] in ("running", "done"), 90)
        status = next(c for c in get(f"{CENTRAL}/api/cameras")["cameras"]
                      if c["id"] == cam_id)["status"]
        print(f"  3. module maps camera_id, status flows .... "
              f"{'PASS' if tracked else 'FAIL'}  (status={status})")
        ok &= tracked
    finally:
        proc.terminate()
        try:
            out = proc.communicate(timeout=15)[0] or ""
        except subprocess.TimeoutExpired:
            proc.kill()
            out = proc.communicate()[0] or ""
        for line in out.splitlines():
            if any(k in line for k in ("Registered with central", "Reporting to central",
                                       "Traceback", "Error", "error")):
                print(f"     module log | {line.strip()[:110]}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
