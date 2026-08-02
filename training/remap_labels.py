"""Renumber and filter the classes of a downloaded YOLO dataset, in place.

A public dataset never uses your class ids. The Roboflow construction-safety set,
for example, is 14 classes with ``Hardhat=3`` and ``Person=11``, while
``training/ppe.yaml`` wants ``person=0, helmet=1, phone=2, head=3``. Training on
the raw download teaches the model that a person is a hardhat — and nothing
errors, because ids are just integers.

This rewrites every label line to YOUR ids and DROPS every class you did not map.
Dropping is the right default: an unmapped class becomes background, which is
what you want for gloves, ladders and safety cones.

    # inspect first — changes nothing
    python training/remap_labels.py --labels path/to/labels --map 11:0,3:1 --dry-run

    # then apply
    python training/remap_labels.py --labels path/to/labels --map 11:0,3:1

Edits files IN PLACE. The dataset is re-downloadable; your own labelled frames
are not, so point this at the download, never at training/dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

# For readable output only — the mapping itself is whatever --map says.
# Matches training/ppe7.yaml.
OUR_NAMES = {0: "person", 1: "hardhat", 2: "phone", 3: "fall-detected",
             4: "gloves", 5: "safety-cone", 6: "safety-vest"}


def parse_map(text: str) -> dict:
    """``"11:0,3:1"`` -> ``{11: 0, 3: 1}``."""
    mapping = {}
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        try:
            src, dst = pair.split(":")
            mapping[int(src)] = int(dst)
        except ValueError:
            raise SystemExit(f"Bad --map entry {pair!r}. Use SRC:DST,SRC:DST")
    if not mapping:
        raise SystemExit("--map is empty: every class would be dropped.")
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True,
                    help="labels directory (searched recursively for *.txt)")
    ap.add_argument("--map", type=str, required=True,
                    help="SRC:DST pairs, e.g. 11:0,3:1 — unmapped classes are dropped")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")
    ap.add_argument("--force", action="store_true",
                    help="remap again even though this directory was already remapped")
    args = ap.parse_args()

    mapping = parse_map(args.map)
    files = sorted(args.labels.rglob("*.txt"))
    if not files:
        print(f"No .txt files under {args.labels}")
        return 1

    # This operation is NOT idempotent: source and destination ids overlap, so a
    # second pass silently re-maps already-mapped classes (with 3:1 and 1:4, every
    # hardhat becomes a glove). A marker file is cruder than detecting it, but it
    # cannot be fooled, and the failure it prevents is invisible in the data.
    marker = args.labels / ".remap-applied"
    if marker.exists() and not (args.dry_run or args.force):
        print(f"{args.labels} was already remapped:\n  {marker.read_text().strip()}\n"
              f"Running again would re-map the NEW ids (hardhat -> gloves, etc).\n"
              f"Restore the originals first, or pass --force if you know better.")
        return 1

    before, after, dropped = Counter(), Counter(), Counter()
    emptied = 0
    for path in files:
        out = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                src = int(parts[0])
            except ValueError:
                continue
            before[src] += 1
            if src in mapping:
                parts[0] = str(mapping[src])
                after[mapping[src]] += 1
                out.append(" ".join(parts))
            else:
                dropped[src] += 1
        # An emptied file is a legitimate background frame, but a whole dataset of
        # them means the mapping is wrong, so it is counted and reported.
        if not out:
            emptied += 1
        if not args.dry_run:
            path.write_text("\n".join(out) + ("\n" if out else ""))

    print(f"{len(files)} label file(s) in {args.labels}")
    print(f"\nsource classes seen : {dict(sorted(before.items()))}")
    print("kept + renumbered   :")
    for dst, count in sorted(after.items()):
        name = OUR_NAMES.get(dst, "?")
        print(f"    {dst} {name:<14} {count:>8}")
    print(f"dropped             : {dict(sorted(dropped.items()))}")
    print(f"files now empty     : {emptied} of {len(files)}")
    if args.dry_run:
        print("\n--dry-run: nothing written. Re-run without it to apply.")
    else:
        marker.write_text(f"{datetime.now().isoformat(timespec='seconds')} "
                          f"--map {args.map}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
