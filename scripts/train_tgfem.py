"""Phase 6 - full training runs for TG-FEM and its two load-bearing ablations.

Companion to scripts/train_baseline.py (Phase 3), which produced the
reference numbers (stock/CBAM/YOLO-World-S) this script's results are judged
against. Unlike Phase 3's baselines, every variant here needs the language
branch (Phase 4) and the real TG-FEM math + WorldDetect head (Phase 5), so it
goes through TGFEMTrainer rather than the plain `YOLO(...).train()` API.

Variants:
    tgfem            the model - TG-FEM gates, text-conditioned, learnable
                      context tokens (n_ctx > 0) unless --n-ctx 0
    tgfem_identity    ablation (a): same parameter budget, gates forced to
                      identity - isolates "does the mechanism do anything"
                      from "does TG-FEM have more parameters than stock"
    cbam_worlddetect  ablation (g), the load-bearing comparison ("this one is
                      the paper" - README): same WorldDetect head and same
                      text feeding it as tgfem, but P3/P4/P5 gates come from
                      CBAM (image statistics) instead of TG-FEM (text). Only
                      this one structural difference is allowed to vary.

Ablation (d) (context-token count / hand-written vs learned prompts) is
--n-ctx and --tier on the `tgfem` variant, not a separate cfg.

Run (needs a real GPU + internet access to huggingface.co - see language.py):
    python scripts/train_tgfem.py --variant tgfem --dataset neu --protocol closed
    python scripts/train_tgfem.py --variant tgfem_identity --dataset neu --protocol closed
    python scripts/train_tgfem.py --variant cbam_worlddetect --dataset neu --protocol closed
    python scripts/train_tgfem.py --variant tgfem --n-ctx 0 --tier bare   # ablation (d), M=0 arm
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tgfem import register  # noqa: E402
from tgfem.data import class_phrase_pools, class_texts_for  # noqa: E402

RESULTS = REPO_ROOT / "results"

CFG = {
    "tgfem": REPO_ROOT / "cfg" / "yolo11-tgfem.yaml",
    "tgfem_identity": REPO_ROOT / "cfg" / "yolo11-tgfem-ablation-a.yaml",
    "cbam_worlddetect": REPO_ROOT / "cfg" / "yolo11-cbam-worlddetect.yaml",
}

DATA_YAML = {
    ("neu", "closed"): REPO_ROOT / "datasets" / "neu-det-yolo" / "neu_closed.yaml",
    ("neu", "openvocab"): REPO_ROOT / "datasets" / "neu-det-yolo" / "neu_openvocab.yaml",
    ("gc10", "closed"): REPO_ROOT / "datasets" / "gc10-det-yolo" / "gc10_closed.yaml",
    ("gc10", "openvocab"): REPO_ROOT / "datasets" / "gc10-det-yolo" / "gc10_openvocab.yaml",
    # NEU + GC10 merged, 16 classes (scripts/prepare_combined.py). Closed only:
    # a combined open-vocabulary split would need its held-out classes chosen
    # afresh across both taxonomies, which is a separate design decision.
    ("combined", "closed"): REPO_ROOT / "datasets" / "combined-yolo" / "combined_closed.yaml",
    # DeepCrack specialist (concrete/asphalt, single class). Training on this
    # SPENDS the out-of-distribution holdout - deliberate, see PHASE-1b.md.
    ("deepcrack", "closed"): REPO_ROOT / "datasets" / "deepcrack-yolo" / "deepcrack_closed.yaml",
    # The concrete specialist that actually ships: ~9.7k pooled crack images
    # with DeepCrack excluded, so DeepCrack survives as the OOD probe. This is
    # what "deepcrack" above should have been - keep both so the 255-image
    # result stays reproducible as the small-data comparison point.
    ("crack", "closed"): REPO_ROOT / "datasets" / "crack-merged-yolo" / "crack_merged.yaml",
    # NEU + crack, 7 classes. The point is not mAP: a single-class crack model
    # is not text-guided at all (PHASE-9.md section 4), and this is the test of
    # whether giving crack six steel classes to compete against fixes that.
    ("neu_crack", "closed"): REPO_ROOT / "datasets" / "neu-crack-yolo" / "neucrack_closed.yaml",
    # Concrete structural defects, 6 classes (Roboflow export, CC BY 4.0).
    # Kept SEPARATE from neu_crack on purpose: its classes sit on the same
    # material as `crack` and are plausibly confusable with it, which is the
    # NEU+GC10 situation that measurably hurt both taxonomies (PHASE-9.md
    # section 1). Training it alone contains that risk to this model.
    ("concrete", "closed"): REPO_ROOT / "datasets" / "concrete-yolo" / "concrete_closed.yaml",
}

# Which scripts/prepare_*.py builds each dataset, for the error message when one
# has not been built yet.
PREPARE_SCRIPT = {
    "neu": "neu_det", "gc10": "gc10", "combined": "combined",
    "deepcrack": "deepcrack", "crack": "crack_merged", "neu_crack": "neu_crack",
    "concrete": "concrete",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(CFG), default="tgfem")
    ap.add_argument("--dataset",
                    choices=["neu", "gc10", "combined", "deepcrack", "crack", "neu_crack",
                             "concrete"],
                    default="neu")
    ap.add_argument("--protocol", choices=["closed", "openvocab"], default="closed")
    ap.add_argument("--tier", choices=["bare", "natural", "visual", "material", "alias"], default="natural",
                     help="ablation (d): which phrase tier forms the fixed training vocabulary")
    ap.add_argument("--phrase-aug", action="store_true",
                    help="ablation (d), robustness arm: resample a phrasing per class "
                         "every TRAINING step from the corpus (~10 per class) instead of "
                         "fixing one for the whole run. Evaluation stays deterministic "
                         "on --tier. See scripts/probe_wording.py for why.")
    ap.add_argument("--n-ctx", type=int, default=8, help="ablation (d): learnable context tokens (0 = hand-written prompts only)")
    ap.add_argument("--epochs", type=int, default=150, help="report section 4.4 specifies 150")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    register()
    import torch

    from tgfem.trainer import TGFEMTrainer

    data_path = DATA_YAML[(args.dataset, args.protocol)]
    if not data_path.exists():
        raise SystemExit(
            f"missing {data_path}\n"
            f"run: python scripts/prepare_{PREPARE_SCRIPT[args.dataset]}.py"
        )
    names = [v for _, v in sorted(yaml.safe_load(data_path.read_text())["names"].items())]
    class_texts = class_texts_for(names, tier=args.tier)
    phrase_pools = class_phrase_pools(names) if args.phrase_aug else None

    run_name = f"phase6_{args.dataset}_{args.protocol}_{args.variant}"
    if args.n_ctx != 8:
        run_name += f"_nctx{args.n_ctx}"
    if args.phrase_aug:
        run_name += "_phraseaug"
    if args.tier != "natural":
        run_name += f"_{args.tier}"

    print("=" * 62)
    print(f"Phase 6 - {run_name}")
    print("=" * 62)
    print(f"config     : {CFG[args.variant].name}")
    print(f"data       : {data_path.name}")
    print(f"vocabulary : {len(class_texts)} classes, tier={args.tier}, n_ctx={args.n_ctx}, "
          f"phrase_aug={args.phrase_aug}"
          + (f" ({sum(len(p) for p in phrase_pools)} phrases in pool)" if phrase_pools else ""))
    for c, t in zip(names, class_texts):
        print(f"  {c:<20} -> {t!r}")
    print(f"epochs     : {args.epochs}   batch: {args.batch}   imgsz: {args.imgsz}")
    print(f"device     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print()

    trainer = TGFEMTrainer(
        overrides={
            "model": str(CFG[args.variant]),
            "data": str(data_path),
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "seed": args.seed,
            "amp": torch.cuda.is_available(),
            "workers": args.workers,
            "project": str(REPO_ROOT / "runs"),
            "name": run_name,
            "exist_ok": True,
            "plots": True,
            "val": True,
            "class_texts": class_texts,
            "n_ctx": args.n_ctx,
            "phrase_pools": phrase_pools,
        }
    )
    trainer.train()

    if not trainer.text_conditioner.encoder.pretrained_loaded:
        print("\n" + "!" * 70)
        print("WARNING: this run used a RANDOMLY-INITIALISED text encoder (no route to")
        print("huggingface.co). Every number below is structurally meaningless - see")
        print("language.py's module docstring. Re-run with real internet access.")
        print("!" * 70 + "\n")

    # --- evaluate on the held-out TEST split, mirroring train_baseline.py ---
    # trainer.validator (built by _setup_train) validates on the data yaml's
    # "val" split during training; the report's criteria are defined on
    # "test", untouched until now (Phase 3 finding) - build a fresh
    # validator pointed at it explicitly rather than reuse that one.
    from copy import copy

    from ultralytics.models.yolo.detect import DetectionValidator

    test_args = copy(trainer.args)
    test_args.split = "test"
    test_args.plots = False
    validator = DetectionValidator(args=test_args, save_dir=REPO_ROOT / "runs" / f"{run_name}_test")
    validator(model=trainer.best if trainer.best.exists() else trainer.model)
    metrics = validator.metrics

    summary = {
        "variant": args.variant,
        "dataset": args.dataset,
        "protocol": args.protocol,
        "config": CFG[args.variant].name,
        "n_ctx": args.n_ctx,
        "phrase_aug": args.phrase_aug,
        "tier": args.tier,
        "pretrained_clip_loaded": trainer.text_conditioner.encoder.pretrained_loaded,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "mAP50": round(float(metrics.box.map50), 5),
        "mAP50_95": round(float(metrics.box.map), 5),
        "precision": round(float(metrics.box.mp), 5),
        "recall": round(float(metrics.box.mr), 5),
        "per_class_AP50": {
            names[c]: round(float(metrics.box.ap50[i]), 5)
            for i, c in enumerate(metrics.box.ap_class_index)
        },
        "speed_ms": {k: round(v, 2) for k, v in metrics.speed.items()},
    }

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{run_name}.json"
    out.write_text(json.dumps(summary, indent=2))

    infer = summary["speed_ms"].get("inference", 0)
    if infer:
        print(f"\ninference    : {infer:.1f} ms  (~{1000 / infer:.0f} FPS)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
