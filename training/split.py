"""Move a slice of the labelled train set into the val set.

Run this **after** labelling, never before: an image and its label file must move
together, and doing it by hand in a file manager is how you end up with images
whose labels stayed behind — which trains the model on "this frame contains
nothing".

Validation frames should come from YOUR cameras. If the val set is public
photos, the score tells you how well you do on the internet, which is not what
you are shipping. Use ``--only`` to restrict the split to your own frames when
the train folder also holds a downloaded dataset.

    python training/split.py                     # move 15% to val
    python training/split.py --frac 0.2
    python training/split.py --only cam3_        # only files starting cam3_
    python training/split.py --undo              # move everything back to train
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "training" / "dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def label_for(image: Path) -> Path:
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def move(image: Path, dest_split: str) -> bool:
    """Move one image and its label into ``dest_split``. False if unlabelled."""
    label = label_for(image)
    if not label.exists():
        return False
    for src in (image, label):
        # .../images/train/x.jpg -> .../images/val/x.jpg  (same for labels)
        dest = src.parent.parent / dest_split / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dest)
    return True


def listing(split: str) -> list[Path]:
    d = DATASET / "images" / split
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frac", type=float, default=0.15,
                    help="fraction of the train set to move (default 0.15)")
    ap.add_argument("--only", type=str, default="",
                    help="only consider files whose name starts with this")
    ap.add_argument("--seed", type=int, default=0, help="for a repeatable split")
    ap.add_argument("--undo", action="store_true",
                    help="move every val image back into train")
    args = ap.parse_args()

    if args.undo:
        moved = sum(move(p, "train") for p in listing("val"))
        print(f"Moved {moved} image(s) back to train.")
        return 0

    train = [p for p in listing("train") if p.name.startswith(args.only)]
    if not train:
        print(f"Nothing to split in {DATASET / 'images' / 'train'}"
              + (f" matching '{args.only}'" if args.only else ""))
        return 1

    unlabelled = [p for p in train if not label_for(p).exists()]
    if unlabelled:
        print(f"WARNING: {len(unlabelled)} image(s) have no label file and will be "
              f"left in train. Label them first, or they teach the model that the "
              f"frame is empty. e.g. {unlabelled[0].name}")

    labelled = [p for p in train if label_for(p).exists()]
    n = max(1, round(len(labelled) * args.frac))
    random.Random(args.seed).shuffle(labelled)
    moved = sum(move(p, "val") for p in labelled[:n])

    print(f"train {len(listing('train'))}   val {len(listing('val'))}   "
          f"(moved {moved})")
    print("Next: python training/train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
