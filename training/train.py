"""Fine-tune one YOLO over person + helmet + phone, then report per-class recall.

A thin wrapper over ``yolo detect train`` that pins the defaults this project
actually needs and checks the two things people get wrong:

* **imgsz must match deployment.** Train at 640 and deploy at 1280 and the model
  sees objects at a scale it never learned. `Config.inference_imgsz` is 1280
  because small objects (phones, and helmets on distant workers) vanish at 640.
* **Recall is the number that matters here**, not mAP. `HelmetComplianceRule`
  infers a violation from a MISSING helmet, so every helmet the model fails to
  detect is a false accusation against a compliant worker.

    python training/train.py                      # 100 epochs from yolo11n.pt
    python training/train.py --epochs 200 --model yolo11s.pt
    python training/train.py --batch 4            # if you hit CUDA OOM at 1280
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "module" / "core"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "training" / "ppe.yaml")
    ap.add_argument("--model", type=str, default="yolo11n.pt",
                    help="starting weights. Fine-tuning from the COCO checkpoint "
                         "beats training from scratch on a small dataset")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=0,
                    help="0 = take it from Config.inference_imgsz (recommended)")
    ap.add_argument("--batch", type=int, default=8,
                    help="lower this first if training OOMs at imgsz 1280")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--name", type=str, default="ppe")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"Missing {args.data}")
        return 1

    imgsz = args.imgsz
    if imgsz <= 0:
        from config import Config
        imgsz = int(Config().inference_imgsz)
        print(f"imgsz={imgsz} (from Config.inference_imgsz — keep these equal)")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        # CCTV is fixed-mount: a camera never sees the world upside down or
        # mirrored, so those augmentations spend capacity on poses that will
        # never occur. Scale and translate DO happen (people walk toward and
        # across the camera), so they stay on.
        fliplr=0.0,
        degrees=0.0,
        patience=30,          # stop early rather than overfit a small dataset
    )

    metrics = model.val(data=str(args.data), imgsz=imgsz, device=args.device)
    print("\nPer-class results (recall is the one to watch):")
    print(f"{'class':>10} {'precision':>10} {'recall':>10} {'mAP50':>10}")
    print("-" * 44)
    names = model.names
    for i, cid in enumerate(metrics.box.ap_class_index):
        p, r, ap50 = metrics.box.p[i], metrics.box.r[i], metrics.box.ap50[i]
        print(f"{names[int(cid)]:>10} {p:>10.3f} {r:>10.3f} {ap50:>10.3f}")

    print(f"\nWeights: runs/detect/{args.name}/weights/best.pt")
    print("Next:")
    print("  1. Copy best.pt to module/models/ and point Config.model_path at it.")
    print("  2. Set person_class_id=0, helmet_class_id=1, phone_class_id=2,")
    print("     head_class_id=3  (fine-tuned ids are YOUR data.yaml order, not")
    print("     COCO's 0/67). head_class_id enables helmet/head position")
    print("     matching, which is what tells worn from carried.")
    print("  3. Set ppe_enabled=True.")
    print("  4. Re-measure throughput: python tests/bench_gpu.py -> fps_budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
