"""Phase 3 - train a baseline detector on the NEU-DET closed-vocabulary split.

Three variants, all trained under identical settings so the comparison is fair:

    stock  YOLOv11n            closed-vocabulary upper bound. Defines the
                               "within 3 mAP retention" criterion.
    cbam   YOLOv11n + CBAM     visual-attention control. Isolates "attention as
                               such" from "text conditioning" - without this,
                               any TG-FEM gain could just be the attention.
    tgfem  YOLOv11n + TG-FEM   identity mode in Phase 3; becomes the real model
                               in Phase 5. Run now to confirm the identity
                               module matches stock, validating ablation (a).

Run:
    python scripts/train_baseline.py --variant stock
    python scripts/train_baseline.py --variant cbam
    python scripts/train_baseline.py --variant stock --epochs 50
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tgfem import register  # noqa: E402

DATA = REPO_ROOT / "datasets" / "neu-det-yolo" / "neu_closed.yaml"
RESULTS = REPO_ROOT / "results"

VARIANTS = {
    # name  : (model spec, pretrained weights or None)
    "stock": ("yolo11n.yaml", "yolo11n.pt"),
    "stock_scratch": ("yolo11n.yaml", None),
    "cbam": (str(REPO_ROOT / "cfg" / "yolo11-cbam.yaml"), None),
    "tgfem": (str(REPO_ROOT / "cfg" / "yolo11-tgfem.yaml"), None),
}

# WHY stock_scratch EXISTS
# -----------------------
# Inserting a module at layers 5/8/13 shifts every downstream index, so the
# COCO-pretrained yolo11n.pt state dict no longer maps onto the modified
# architecture. Every custom-module variant (cbam, tgfem) therefore trains from
# scratch, while plain `stock` starts pretrained.
#
# Comparing them directly measures the value of COCO pretraining, not of the
# architecture - and CBAM already fails the report's 3-point retention
# threshold on that basis alone, despite containing no text conditioning.
#
# `stock_scratch` is the like-for-like retention anchor: identical architecture
# to `stock`, identical schedule, but random init like the module variants.
# Retention should be judged against THIS number; `stock` stays in the table as
# the absolute ceiling. See phase-notes/PHASE-3.md section 3.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="stock")
    ap.add_argument("--epochs", type=int, default=150,
                    help="report section 4.4 specifies 150")
    ap.add_argument("--batch", type=int, default=16,
                    help="Phase 2 measured 1.57 GB at batch 8, so 16 fits 6 GB")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    register()
    from ultralytics import YOLO
    import torch

    spec, pretrained = VARIANTS[args.variant]
    run_name = f"phase3_{args.variant}"

    print("=" * 62)
    print(f"Phase 3 baseline - {args.variant}")
    print("=" * 62)
    print(f"config     : {Path(spec).name}")
    print(f"pretrained : {pretrained or 'no (from scratch)'}")
    print(f"epochs     : {args.epochs}   batch: {args.batch}   imgsz: {args.imgsz}")
    print(f"device     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print()

    if not DATA.exists():
        raise SystemExit(f"missing {DATA}\nrun: python scripts/prepare_neu_det.py")

    # Build from YAML (architecture), then load COCO-pretrained weights where a
    # compatible checkpoint exists. With only 1434 training images, training a
    # detector from scratch is not competitive - the report's retention
    # criterion assumes a properly initialised baseline.
    model = YOLO(spec)
    if pretrained:
        model = YOLO(pretrained)

    model.train(
        data=str(DATA),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
        amp=True,
        workers=2,
        project=str(REPO_ROOT / "runs"),
        name=run_name,
        exist_ok=True,
        plots=True,
        val=True,
    )

    # --- evaluate on the held-out TEST split -----------------------------
    # model.train() reports val-split numbers. The report's criteria are defined
    # on test, which has been untouched until now.
    metrics = model.val(
        data=str(DATA),
        split="test",
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(REPO_ROOT / "runs"),
        name=f"{run_name}_test",
        exist_ok=True,
        plots=False,
    )

    summary = {
        "variant": args.variant,
        "config": Path(spec).name,
        "pretrained": pretrained,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "split": "test",
        "mAP50": round(float(metrics.box.map50), 5),
        "mAP50_95": round(float(metrics.box.map), 5),
        "precision": round(float(metrics.box.mp), 5),
        "recall": round(float(metrics.box.mr), 5),
        "per_class_AP50": {
            metrics.names[c]: round(float(metrics.box.ap50[i]), 5)
            for i, c in enumerate(metrics.box.ap_class_index)
        },
        "speed_ms": {k: round(v, 2) for k, v in metrics.speed.items()},
    }

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"phase3_{args.variant}.json"
    out.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 62)
    print(f"TEST split results - {args.variant}")
    print("=" * 62)
    print(f"  mAP@0.5      : {summary['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95 : {summary['mAP50_95']:.4f}")
    print(f"  precision    : {summary['precision']:.4f}")
    print(f"  recall       : {summary['recall']:.4f}")
    print("\n  per-class AP@0.5:")
    for cls, ap in summary["per_class_AP50"].items():
        print(f"    {cls:<18} {ap:.4f}")
    infer = summary["speed_ms"].get("inference", 0)
    if infer:
        print(f"\n  inference    : {infer:.1f} ms  (~{1000 / infer:.0f} FPS)")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
