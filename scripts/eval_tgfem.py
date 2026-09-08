"""Phase 7 - evaluation on top of a Phase 6 checkpoint.

train_tgfem.py already reports per-class test-split AP for whatever data yaml
it was pointed at - for the openvocab protocol that test split contains both
seen AND held-out classes (Phase 1 design: the openvocab YAML declares all
classes in `names`, held-out ones just have zero training images), so the
zero-shot numbers this project's whole premise rests on are already sitting
in results/phase6_*_openvocab_*.json's per_class_AP50. This script covers
what that run does NOT:

    1. negative control - swap the trained checkpoint's text for nonsense
       prompts *without retraining* and confirm mAP collapses. If it does
       not, the classification head is not actually being driven by text
       (the exact bug eval_yoloworld.py caught in Phase 3 - see its
       docstring). Re-checking this per checkpoint matters more here than it
       did for YOLO-World, because TG-FEM's context tokens are learned
       jointly with detection; a broken text path could still fit the seen
       classes and look fine right up until a real test.
    2. a compact tgfem vs tgfem_identity vs cbam_worlddetect comparison
       table (ablations (a) and (g)) pulled from results/phase6_*.json,
       plus the falsifiable per-class prediction from PHASE-3.md section 3:
       any TG-FEM gain over CBAM should concentrate on crazing and
       rolled-in_scale.

Run:
    python scripts/eval_tgfem.py --checkpoint runs/phase6_neu_openvocab_tgfem/weights/best.pt
    python scripts/eval_tgfem.py --compare-only   # just tabulate existing results/phase6_*.json
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS = REPO_ROOT / "results"

ABSURD = ["banana", "elephant", "skateboard", "umbrella", "giraffe", "pizza"]

# The falsifiable prediction from phase-notes/PHASE-3.md section 3: attention's
# gain over the from-scratch anchor concentrated on these two texture-confusable,
# NEU-DET-held-out classes. If text conditioning works, TG-FEM's gain over CBAM
# should concentrate here too.
FALSIFIABLE_CLASSES = ["crazing", "rolled-in_scale"]


def negative_control(checkpoint: Path, data_yaml: Path, device: str, imgsz: int, batch: int):
    import torch

    from tgfem import register
    from tgfem.detection_model import TGFEMModel  # noqa: F401 - needed for unpickling
    from tgfem.language import TextConditioner
    from tgfem.module import TGFEM

    register()

    names = [v for _, v in sorted(yaml.safe_load(data_yaml.read_text())["names"].items())]

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) else ckpt
    model = model.float().to(device)

    # Detach whatever TextConditioner came pickled with the checkpoint and
    # attach a fresh one, n_ctx=0 (no learned component - the absurd prompts
    # were never trained on, so there is nothing to learn a context for
    # anyway), so the swap is a clean drop-in replacement.
    for h in list(model._forward_pre_hooks.values()):
        if isinstance(h, TextConditioner):
            model._forward_pre_hooks.clear()

    tgfem_layers = [m for m in model.model if isinstance(m, TGFEM)]

    from ultralytics.models.yolo.detect import DetectionValidator

    def run(class_texts, tag):
        tc = TextConditioner(class_texts, n_ctx=0, device=device)
        tc.attach(model, tgfem_layers)
        # model.args may not exist on a bare loaded model - build minimal args instead
        from ultralytics.cfg import get_cfg
        val_args = get_cfg(overrides=dict(
            data=str(data_yaml), split="test", imgsz=imgsz, batch=batch, device=device, plots=False,
        ))
        validator = DetectionValidator(args=val_args, save_dir=REPO_ROOT / "runs" / f"eval_control_{tag}")
        validator(model=model)
        return validator.metrics

    real_texts = [next(d["text"] for d in json.loads((REPO_ROOT / "prompts" / "defect_corpus.json").read_text())
                        ["classes"][c]["descriptions"] if d["tier"] == "natural") for c in names]
    real_metrics = run(real_texts, "real")
    absurd_metrics = run((ABSURD * ((len(names) // len(ABSURD)) + 1))[: len(names)], "absurd")

    print(f"real prompts   mAP@0.5 = {real_metrics.box.map50:.4f}")
    print(f"absurd prompts mAP@0.5 = {absurd_metrics.box.map50:.4f}")
    if absurd_metrics.box.map50 >= real_metrics.box.map50 * 0.5 and real_metrics.box.map50 > 0.01:
        print("\nWARNING: nonsense prompts scored close to real ones - the classification")
        print("head may not actually be driven by text. See this script's docstring.")
        return False
    print("negative control OK: nonsense prompts collapse relative to real ones.")
    return True


def compare(dataset: str, protocol: str):
    rows = []
    for path in sorted(RESULTS.glob(f"phase6_{dataset}_{protocol}_*.json")):
        d = json.loads(path.read_text())
        if "variant" not in d:
            continue
        rows.append(d)
    if not rows:
        print(f"no results/phase6_{dataset}_{protocol}_*.json found yet")
        return

    print("=" * 78)
    print(f"Phase 6/7 comparison - {dataset} / {protocol}")
    print("=" * 78)
    print(f"{'variant':<20}{'n_ctx':>6}{'tier':>10}{'mAP@0.5':>10}{'CLIP real?':>12}")
    for r in rows:
        print(f"{r['variant']:<20}{r['n_ctx']:>6}{r['tier']:>10}{r['mAP50']:>10.4f}"
              f"{'yes' if r.get('pretrained_clip_loaded') else 'NO':>12}")

    tgfem = next((r for r in rows if r["variant"] == "tgfem"), None)
    cbam = next((r for r in rows if r["variant"] == "cbam_worlddetect"), None)
    if tgfem and cbam:
        print("\nablation (g) - TG-FEM vs CBAM gates, same WorldDetect head:")
        print(f"  TG-FEM mAP@0.5 = {tgfem['mAP50']:.4f}   CBAM mAP@0.5 = {cbam['mAP50']:.4f}"
              f"   delta = {tgfem['mAP50'] - cbam['mAP50']:+.4f}")
        print("\n  per-class delta (falsifiable prediction: gain concentrates on "
              f"{', '.join(FALSIFIABLE_CLASSES)}):")
        for cls in tgfem["per_class_AP50"]:
            t, c = tgfem["per_class_AP50"][cls], cbam["per_class_AP50"].get(cls)
            if c is None:
                continue
            flag = "  <-- falsifiable prediction target" if cls in FALSIFIABLE_CLASSES else ""
            print(f"    {cls:<18} {t - c:+.4f}{flag}")

    identity = next((r for r in rows if r["variant"] == "tgfem_identity"), None)
    if tgfem and identity:
        print("\nablation (a) - TG-FEM vs identity (same param budget):")
        print(f"  TG-FEM mAP@0.5 = {tgfem['mAP50']:.4f}   identity mAP@0.5 = {identity['mAP50']:.4f}"
              f"   delta = {tgfem['mAP50'] - identity['mAP50']:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--dataset", choices=["neu", "gc10"], default="neu")
    ap.add_argument("--protocol", choices=["closed", "openvocab"], default="openvocab")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--compare-only", action="store_true", help="skip the negative control, just tabulate results/phase6_*.json")
    args = ap.parse_args()

    if not args.compare_only:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint required unless --compare-only")
        data_yaml = REPO_ROOT / "datasets" / f"{'neu-det' if args.dataset == 'neu' else 'gc10-det'}-yolo" / f"{'neu' if args.dataset == 'neu' else 'gc10'}_{args.protocol}.yaml"
        negative_control(args.checkpoint, data_yaml, args.device, args.imgsz, args.batch)
        print()

    compare(args.dataset, args.protocol)


if __name__ == "__main__":
    main()
