#!/usr/bin/env python3
"""Loop a local video as a test stream so the dashboard has something to point at.

Serves HLS (HTTP) by default, or RTSP with ``--rtsp``. Uses the system ``ffmpeg``
if present, otherwise the static binary bundled with the ``imageio-ffmpeg`` pip
package (``pip install imageio-ffmpeg`` — no sudo needed).

Examples
--------
    python scripts/serve_test_stream.py                 # HLS from input/operator.mp4
    python scripts/serve_test_stream.py my_clip.mp4     # HLS from another file
    python scripts/serve_test_stream.py --port 8090
    python scripts/serve_test_stream.py --rtsp          # RTSP (ffmpeg listen mode)

Then paste the printed URL into the dashboard's "Add Stream" box. Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import socketserver
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resolve_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit(
            "ffmpeg not found. Install it with either:\n"
            "  pip install imageio-ffmpeg      (no sudo)\n"
            "  sudo apt install -y ffmpeg"
        )


def serve_rtsp(ffmpeg: str, video: Path, port: int, name: str) -> None:
    url = f"rtsp://localhost:{port}/{name}"
    cmd = [
        ffmpeg, "-re", "-stream_loop", "-1", "-i", str(video),
        "-c", "copy", "-f", "rtsp", "-rtsp_flags", "listen", url,
    ]
    print(f"\n  Stream URL:  {url}\n  Paste it into the dashboard. Ctrl-C to stop.\n")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


def serve_hls(ffmpeg: str, video: Path, port: int, name: str) -> None:
    outdir = Path(tempfile.mkdtemp(prefix="teststream_"))
    m3u8 = outdir / f"{name}.m3u8"
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-re", "-stream_loop", "-1", "-i", str(video),
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-g", "48", "-an",
        "-f", "hls", "-hls_time", "2", "-hls_list_size", "6",
        "-hls_flags", "delete_segments+omit_endlist", str(m3u8),
    ]
    ff = subprocess.Popen(cmd)
    for _ in range(75):                 # wait up to ~15s for the first segment
        if m3u8.exists():
            break
        if ff.poll() is not None:
            sys.exit("ffmpeg exited early — check the codec/input file.")
        time.sleep(0.2)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(outdir))

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    httpd = Server(("0.0.0.0", port), handler)
    url = f"http://localhost:{port}/{name}.m3u8"
    print(f"\n  Stream URL:  {url}\n  Paste it into the dashboard. Ctrl-C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ff.terminate()
        httpd.shutdown()
        shutil.rmtree(outdir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Loop a video as a test HLS/RTSP stream.")
    ap.add_argument("video", nargs="?", default=str(ROOT / "input" / "operator.mp4"),
                    help="video file to loop (default: input/operator.mp4)")
    ap.add_argument("--port", type=int, default=8080, help="HLS HTTP port (default 8080)")
    ap.add_argument("--rtsp", action="store_true", help="serve RTSP instead of HLS")
    ap.add_argument("--rtsp-port", type=int, default=8554, help="RTSP port (default 8554)")
    ap.add_argument("--name", default="operator", help="stream path/name (default 'operator')")
    args = ap.parse_args()

    # Line-buffer stdout so the URL shows immediately even when piped to a file.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"video not found: {video}")
    ffmpeg = resolve_ffmpeg()
    print(f"[serve] ffmpeg: {ffmpeg}")
    print(f"[serve] looping: {video}")

    if args.rtsp:
        serve_rtsp(ffmpeg, video, args.rtsp_port, args.name)
    else:
        serve_hls(ffmpeg, video, args.port, args.name)


if __name__ == "__main__":
    main()
