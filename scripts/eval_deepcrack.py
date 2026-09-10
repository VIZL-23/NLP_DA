"""Phase 7 - DeepCrack out-of-distribution zero-shot evaluation.

DeepCrack was held out entirely for OOD zero-shot eval since Phase 1b, but
nothing ever actually queried a trained checkpoint against it - this script
was the missing piece.

Two things distinguish this from eval_tgfem.py's within-dataset zero-shot
(the NEU/GC10 openvocab protocol's held-out classes, which come for free
from train_tgfem.py's test-split evaluation - see PHASE-7.md §2):

  1. DeepCrack is a genuinely different domain, not a held-out class inside
     the training dataset - the checkpoint has never seen a DeepCrack image,
     a crack-shaped object, or concrete/asphalt texture at all. It is the
     hardest zero-shot test this project has.
  2. Its boxes are measurably loose (Phase 1b: median fill ratio 16.6%, one
     box per connected-component segment of a diagonal crack). A low mAP
     here does not by itself mean "the model can't find cracks" - dense
     predictions can be pointing at the right pixels and still disagree with
     a loose ground-truth box on IoU. Report this caveat alongside the
     number, not instead of it.

Reuses the checkpoint's LEARNED context tokens throughout, including for the
absurd-prompt control (same fixed ctx prefix, different class text) - CoOp's
whole premise is that a learned context generalises to new class names
(demo.py does the same for arbitrary live queries), so this exercises that
premise under the hardest condition available: a class from a dataset the
model has never touched.

Run:
    python scripts/eval_deepcrack.py --checkpoint runs/phase6_neu_openvocab_tgfem/weights/best.pt
    python scripts/eval_deepcrack.py --checkpoint ... --tiers bare natural material alias
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DATA = REPO_ROOT / "datasets" / "deepcrack-yolo" / "deepcrack_ood.yaml"
CORPUS = REPO_ROOT / "prompts" / "defect_corpus.json"
RESULTS = REPO_ROOT / "results"

ABSURD_TEXT = "banana"


def phrases_for_crack() -> dict[str, list[str]]:
    corpus = json.loads(CORPUS.read_text())["classes"]["crack"]["descriptions"]
    by_tier: dict[str, list[str]] = {}
    for d in corpus:
        by_tier.setdefault(d["tier"], []).append(d["text"])
    return by_tier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tiers", nargs="+", default=["bare", "natural"],
                     choices=["bare", "natural", "visual", "material", "alias"],
                     help="corpus tiers to query with, one AP number per tier")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect import DetectionValidator

    from tgfem import register
    from tgfem.detection_model import TGFEMModel  # noqa: F401 - needed for unpickling
    from tgfem.inference import _resolve_device
    from tgfem.language import TextConditioner

    register()

    # Ultralytics tolerates a bare "0", torch.load and Module.to do not: they
    # raise "don't know how to restore data location" and "Invalid device
    # string" respectively. Normalise once, here. `torch_device` is for torch
    # calls; args.device stays as-is for the Ultralytics validator config.
    torch_device = _resolve_device(args.device)

    print("=" * 62)
    print("Phase 7 - DeepCrack OOD zero-shot evaluation")
    print("=" * 62)

    if not DATA.exists():
        raise SystemExit(
            f"missing {DATA}\n"
            "download DeepCrack per the README, then: python scripts/prepare_deepcrack.py"
        )

    ckpt = torch.load(args.checkpoint, map_location=torch_device, weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) else ckpt
    model = model.float().to(torch_device)

    hooks = [h for h in model._forward_pre_hooks.values() if isinstance(h, TextConditioner)]
    if not hooks:
        raise SystemExit("checkpoint has no TextConditioner attached - was it trained with train_tgfem.py?")
    tc = hooks[0]
    print(f"checkpoint trained on : {tc.class_texts}")
    print(f"pretrained CLIP loaded: {tc.encoder.pretrained_loaded}")
    if not tc.encoder.pretrained_loaded:
        print("\n!!! WARNING: this checkpoint's text encoder is randomly-initialised (no real")
        print("!!! CLIP weights). Every number below is structurally meaningless. See")
        print("!!! phase-notes/PHASE-4.md §5.\n")

    by_tier = phrases_for_crack()
    val_args = get_cfg(overrides={
        "data": str(DATA), "split": "test", "imgsz": args.imgsz, "batch": args.batch,
        "device": args.device, "plots": False,
    })

    def run(class_text: str, tag: str):
        tc.class_texts = [class_text]
        model.model[-1].nc = 1  # WorldDetect.forward derives self.no from this - see demo.py
        # DetectionValidator.init_metrics() sets self.names = model.names for
        # its printed per-class table - left alone, that's still whatever the
        # training dataset's class 0 was (e.g. "crazing"), not the DeepCrack
        # query, since ground truth/prediction matching is by index and never
        # actually needed the string to be right. Override it so the printed
        # label matches what's actually being queried.
        model.names = {0: class_text}
        validator = DetectionValidator(args=val_args, save_dir=REPO_ROOT / "runs" / f"eval_deepcrack_{tag}")
        validator(model=model)
        return validator.metrics

    results = {"checkpoint": str(args.checkpoint), "pretrained_clip_loaded": tc.encoder.pretrained_loaded,
               "trained_vocabulary": list(tc.class_texts), "per_tier": {}}

    print()
    for tier in args.tiers:
        phrase = by_tier[tier][0]
        print(f"--- tier: {tier} ({phrase!r}) ---")
        metrics = run(phrase, tier)
        results["per_tier"][tier] = {
            "phrase": phrase,
            "mAP50": round(float(metrics.box.map50), 5),
            "mAP50_95": round(float(metrics.box.map), 5),
            "precision": round(float(metrics.box.mp), 5),
            "recall": round(float(metrics.box.mr), 5),
        }
        print(f"    mAP@0.5 = {results['per_tier'][tier]['mAP50']:.4f}\n")

    print(f"--- negative control: {ABSURD_TEXT!r} (same learned ctx, nonsense class text) ---")
    absurd_metrics = run(ABSURD_TEXT, "absurd")
    results["absurd_control"] = {
        "phrase": ABSURD_TEXT,
        "mAP50": round(float(absurd_metrics.box.map50), 5),
    }
    print(f"    mAP@0.5 = {results['absurd_control']['mAP50']:.4f}\n")

    best_real = max(v["mAP50"] for v in results["per_tier"].values())
    if results["absurd_control"]["mAP50"] >= best_real * 0.5 and best_real > 0.01:
        print("WARNING: nonsense text scored close to real descriptions - the classification")
        print("head may not be responding to text content. See this script's docstring.\n")
    else:
        print("negative control OK: nonsense text does not match real descriptions.\n")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"phase7_deepcrack_{args.checkpoint.parent.parent.name}.json"
    out.write_text(json.dumps(results, indent=2))

    print("=" * 62)
    print("SUMMARY - DeepCrack OOD zero-shot")
    print("=" * 62)
    for tier, r in results["per_tier"].items():
        print(f"  {tier:<10} mAP@0.5 = {r['mAP50']:.4f}   ({r['phrase']!r})")
    print("\nReminder: DeepCrack boxes are loose by construction (Phase 1b - median")
    print("fill ratio 16.6%). A low number here is not conclusive evidence the model")
    print("can't localise cracks - it may be an IoU/geometry mismatch. Inspect")
    print("predictions visually before treating this as a negative result.")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
