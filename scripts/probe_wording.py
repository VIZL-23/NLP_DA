"""Phase 7 - how robust is the text interface to WORDING?

The project's premise is that an inspector types a plain-English description
and gets boxes. That only holds if the model responds to the MEANING of a
phrase, not to the exact string it was trained on.

This probes exactly that: one image, one true defect class, many ways of
phrasing it, plus controls. It reuses the checkpoint's trained context tokens
and swaps only the query text (same mechanism as scripts/demo.py).

Run:
    python scripts/probe_wording.py --checkpoint runs/phase6_neu_closed_tgfem/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# (label, query) - graded from the exact training string outwards.
QUERIES = [
    ("EXACT (trained)",   "scratches on the steel surface"),
    ("near paraphrase",   "scratches on the metal surface"),
    ("corpus, other tier","long thin scratch marks on the metal"),
    ("bare class name",   "scratches"),
    ("descriptive",       "a long straight bright line running across the surface"),
    ("loose paraphrase",  "long thin gouges scored into metal"),
    ("wrong class",       "a pitted steel surface"),
    ("CONTROL nonsense",  "a wooden door"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "runs/phase6_neu_closed_tgfem/weights/best.pt")
    ap.add_argument("--image", type=Path,
                    default=REPO_ROOT / "datasets/neu-det-yolo/images/scratches_1.jpg")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import cv2
    import torch

    from tgfem import register
    from tgfem.detection_model import TGFEMModel  # noqa: F401 - needed for unpickling
    from tgfem.language import TextConditioner

    register()
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) else ckpt
    model = model.float().to(args.device).eval()

    tc = [h for h in model._forward_pre_hooks.values() if isinstance(h, TextConditioner)][0]

    img = cv2.imread(str(args.image))
    img_r = cv2.resize(img, (args.imgsz, args.imgsz))
    img_t = torch.from_numpy(img_r[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    print(f"image  : {args.image.name}  (ground truth: scratches)")
    print(f"n_ctx  : {tc.learner.n_ctx}   trained phrase: {tc.class_texts[5]!r}")
    print()
    print(f"{'kind':<20}{'query':<56}{'best conf':>10}{'n>0.25':>8}")
    print("-" * 94)

    for kind, q in QUERIES:
        # One query at a time, so each phrase is scored on its own merits
        # rather than competing in a softmax against the others.
        tc.class_texts = [q]
        model.model[-1].nc = 1
        with torch.no_grad():
            y, _ = model(img_t)
        dets = non_max_suppression(y, conf_thres=0.01, iou_thres=0.7, nc=1)[0]
        best = float(dets[:, 4].max()) if len(dets) else 0.0
        n_strong = int((dets[:, 4] > 0.25).sum()) if len(dets) else 0
        print(f"{kind:<20}{q[:54]!r:<56}{best:>10.3f}{n_strong:>8}")

    print()
    print("If only the EXACT trained string scores, the model has keyed on that")
    print("specific string rather than learning a general text interface - which")
    print("is the difference between open-vocabulary detection and a lookup table.")


if __name__ == "__main__":
    main()
