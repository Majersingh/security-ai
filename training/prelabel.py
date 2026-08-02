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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# COCO class id -> the id this dataset uses (see training/ppe.yaml).
# Helmet has no COCO equivalent, which is the entire reason for this exercise.
COCO_TO_DATASET = {0: 0, 67: 2}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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
    args = ap.parse_args()

    images = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"No images in {args.images} — run extract_frames.py first.")
        return 1

    todo = [p for p in images if args.overwrite or not label_path_for(p).exists()]
    skipped = len(images) - len(todo)
    if not todo:
        print(f"All {len(images)} images already have labels (--overwrite to redo).")
        return 0

    from ultralytics import YOLO           # heavy import, only needed here

    print(f"Labelling {len(todo)} image(s) with {args.model} at imgsz={args.imgsz}"
          + (f"  ({skipped} already labelled, skipped)" if skipped else ""))
    model = YOLO(args.model)

    counts = {0: 0, 2: 0}
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
        # An empty file is meaningful: "reviewed, nothing here". Ultralytics reads
        # it as a background image, which is a useful training signal.
        out.write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"\nWrote {len(todo)} label files: "
          f"{counts[0]} person, {counts[2]} phone, 0 helmet.")
    print("Now open the folder in your labelling tool, CHECK for missed people, "
          "and draw the helmets (class 1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
