"""Pull training frames out of the CCTV videos in ``uploads/``.

Your own footage is worth far more than public images here: a helmet detector
trained on web photos sees faces at eye level in good light, and yours will see
the tops of heads from a ceiling corner. That domain gap, not model size, is what
usually makes a PPE model disappoint in production.

CCTV is mostly *identical* frames, so sampling every Nth frame gives you hundreds
of near-duplicates that add labelling work and teach the model nothing. This keeps
a frame only when it differs enough from the last one kept.

Put your clips in ``training/videos/`` first — that folder, not ``uploads/``, is
what this reads, so a run only ever touches footage you deliberately put there.

    python training/extract_frames.py                  # everything in training/videos/
    python training/extract_frames.py --every 60 --max 300
    python training/extract_frames.py --src uploads --max 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".m4v"}


def frame_signature(frame: np.ndarray) -> np.ndarray:
    """Tiny grayscale thumbnail — enough to tell 'new scene' from 'same scene'."""
    small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def extract(video: Path, out_dir: Path, every: int, limit: int, min_diff: float) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  !! cannot open {video.name}")
        return 0

    kept, index, last_sig = 0, 0, None
    while kept < limit:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if index % every:
            continue

        sig = frame_signature(frame)
        if last_sig is not None and float(np.abs(sig - last_sig).mean()) < min_diff:
            continue          # visually the same as the frame we already kept
        last_sig = sig

        out = out_dir / f"{video.stem}_{index:06d}.jpg"
        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept += 1

    cap.release()
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "training" / "videos",
                    help="directory of videos (default: training/videos/). Kept "
                         "separate from uploads/ so a run only ever touches clips "
                         "you deliberately put there")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "training" / "dataset" / "images" / "train",
                    help="where to write frames")
    ap.add_argument("--every", type=int, default=30,
                    help="consider every Nth frame (default 30 = ~1/s at 30fps)")
    ap.add_argument("--max", type=int, default=200, dest="limit",
                    help="max frames to keep PER VIDEO (default 200)")
    ap.add_argument("--min-diff", type=float, default=3.0,
                    help="0-255 mean pixel change needed to count as a new scene; "
                         "raise it if you still get near-duplicates (default 3.0)")
    args = ap.parse_args()

    videos = sorted(p for p in args.src.glob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        print(f"No videos found in {args.src}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    print(f"{len(videos)} video(s) -> {args.out}")
    for video in videos:
        n = extract(video, args.out, args.every, args.limit, args.min_diff)
        total += n
        print(f"  {video.name}: {n} frames")

    print(f"\n{total} frames written.")
    print("Next: python training/prelabel.py   (auto-labels person + phone)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
