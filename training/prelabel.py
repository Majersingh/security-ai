"""Auto-label person + phone so a human only has to draw helmets.

Two of your three classes are already known to any COCO model, so labelling them
by hand is wasted effort. This runs a stock YOLO over the extracted frames and
writes YOLO-format labels for person and phone in the *dataset's* id space,
leaving helmet (class 1) for you.

**These are a head start, not ground truth.** Review every file before training.
The failure mode that matters is a MISSED person: YOLO treats every unlabelled
region as background, so a person the pre-labeller skipped actively teaches the
new model that people are not people. That is why the confidence default is low
(0.25) — deleting a wrong box in the editor takes a second, spotting a missing
one takes real attention.

    python training/prelabel.py                          # label the train split
    python training/prelabel.py --model module/models/yolo26n.pt
    python training/prelabel.py --images training/dataset/images/val

Then load the folder in an editor (CVAT, Label Studio, Roboflow — all read this
format) with the class list from ppe.yaml, fix mistakes, and draw the helmets.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def hms(seconds: float) -> str:
    """Seconds -> "4m12s" / "1h04m". Progress on 30k images needs a readable ETA."""
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"

# COCO class id -> the id this dataset uses (see training/ppe.yaml).
# Helmet has no COCO equivalent, which is the entire reason for this exercise.
COCO_TO_DATASET = {0: 0, 67: 2}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def class_ids_in(label: Path) -> set:
    """The class ids already present in a label file. Missing/empty file -> set()."""
    if not label.exists():
        return set()
    ids = set()
    for line in label.read_text().splitlines():
        first = line.split(maxsplit=1)
        if first:
            try:
                ids.add(int(first[0]))
            except ValueError:          # not a label line; ignore rather than crash
                continue
    return ids


def label_path_for(image: Path) -> Path:
    """``.../images/train/x.jpg`` -> ``.../labels/train/x.txt`` (YOLO convention)."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    else:
        raise ValueError(f"{image} is not under an 'images' directory")
    return Path(*parts).with_suffix(".txt")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path,
                    default=ROOT / "training" / "dataset" / "images" / "train")
    ap.add_argument("--model", type=str, default="yolo26x.pt",
                    help="COCO model used for labelling. Bigger = better labels; "
                         "this runs once offline, so accuracy beats speed "
                         "(default yolo26x.pt, auto-downloaded)")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="match your deployment imgsz so small phones are found")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="deliberately low: a missed object becomes 'background' "
                         "and teaches the model the wrong thing (default 0.25)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-label images that already have a label file. OFF by "
                         "default so this can never destroy your hand-drawn helmets")
    ap.add_argument("--append", action="store_true",
                    help="ADD the detected boxes to existing label files instead of "
                         "skipping them. For a downloaded PPE dataset that labels "
                         "helmets but no people: this fills in the missing person "
                         "boxes and keeps every existing line untouched")
    args = ap.parse_args()

    if args.append and args.overwrite:
        print("--append and --overwrite are opposites: one keeps existing lines, "
              "the other replaces them. Pick one.")
        return 1

    images = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"No images in {args.images} — run extract_frames.py first.")
        return 1

    written = set(COCO_TO_DATASET.values())
    if args.append:
        # Skip a file that ALREADY holds one of the ids we write, so a second run
        # cannot double-label the same person. Without this, re-running (which the
        # default mode makes safe and habitual) silently duplicates every box.
        todo = [p for p in images if not (class_ids_in(label_path_for(p)) & written)]
    else:
        todo = [p for p in images if args.overwrite or not label_path_for(p).exists()]
    skipped = len(images) - len(todo)
    if not todo:
        hint = ("every label file already contains person/phone boxes"
                if args.append else "--overwrite to redo")
        print(f"Nothing to do for {len(images)} images ({hint}).")
        return 0

    from ultralytics import YOLO           # heavy import, only needed here

    print(f"Labelling {len(todo)} image(s) with {args.model} at imgsz={args.imgsz}"
          + (f"  ({skipped} already labelled, skipped)" if skipped else ""))
    model = YOLO(args.model)

    counts = {0: 0, 2: 0}
    done = 0
    interrupted = False
    start = time.time()
    last_tick = 0.0
    total = len(todo)

    try:
        for image in todo:
            result = model.predict(source=str(image), imgsz=args.imgsz, conf=args.conf,
                                   classes=sorted(COCO_TO_DATASET), verbose=False)[0]
            lines = []
            for box in result.boxes:
                dataset_id = COCO_TO_DATASET.get(int(box.cls))
                if dataset_id is None:
                    continue
                cx, cy, w, h = box.xywhn[0].tolist()     # already normalised 0-1
                lines.append(f"{dataset_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                counts[dataset_id] += 1

            out = label_path_for(image)
            out.parent.mkdir(parents=True, exist_ok=True)
            if args.append and out.exists():
                existing = out.read_text()
                # A file whose last line has no newline would otherwise be joined
                # onto our first line, corrupting both boxes.
                if existing and not existing.endswith("\n"):
                    existing += "\n"
                out.write_text(existing + "\n".join(lines) + ("\n" if lines else ""))
            else:
                # An empty file is meaningful: "reviewed, nothing here". Ultralytics
                # reads it as a background image, a useful training signal.
                out.write_text("\n".join(lines) + ("\n" if lines else ""))

            # Each file is written before the next image is read, so the run is
            # resumable: whatever finished stays done and a re-run skips it.
            done += 1
            now = time.time()
            if now - last_tick >= 0.5 or done == total:
                last_tick = now
                rate = done / max(now - start, 1e-6)
                print(f"\r  {done}/{total} ({100 * done / total:.0f}%)  "
                      f"{rate:.1f} img/s  elapsed {hms(now - start)}  "
                      f"eta {hms((total - done) / rate) if rate else '?'}  "
                      f"person={counts[0]} phone={counts[2]}",
                      end="", flush=True)
    except KeyboardInterrupt:
        interrupted = True
    print()

    print(f"Wrote {done} label file(s): {counts[0]} person, {counts[2]} phone.")
    if interrupted:
        remaining = total - done
        print(f"INTERRUPTED with {remaining} image(s) left. Every file already "
              f"written is complete and kept — re-run the SAME command and it "
              f"resumes, skipping the {done} done.")
        return 130
    print("Now open the folder in your labelling tool, CHECK for missed people, "
          "and draw the helmets (class 1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
