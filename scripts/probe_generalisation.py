"""Does phrase augmentation GENERALISE, or just cover the phrases it was shown?

Phase 7 found that training on 10 phrasings per class (instead of 1) made the
text interface robust. But 4 of the 5 phrasings that started working were IN
the training pool - so that result alone cannot distinguish:

    (a) wider coverage of the listed phrases  -> a bigger lookup table
    (b) genuine generalisation to unlisted phrases -> a real text interface

This probes (b) directly: every query below appears NOWHERE in
prompts/defect_corpus.json. Controls are included, because a model that scores
everything has become indiscriminate rather than robust.

Run:
    python scripts/probe_generalisation.py --checkpoint <ckpt>
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# NONE of these appear in the corpus. Ground truth for the image is `scratches`.
NOVEL = [
    ("novel paraphrase", "long thin gouges scored into metal"),
    ("novel paraphrase", "a linear defect running down the plate"),
    ("novel paraphrase", "scoring marks left by a sharp edge"),
    ("novel paraphrase", "a slender vertical streak on the metal"),
    ("novel paraphrase", "drag marks across the steel"),
    ("novel paraphrase", "a fine incision in the surface"),
    ("novel, wrong class", "a surface covered in tiny holes"),
    ("novel, wrong class", "dark oxide flakes on the steel"),
    ("CONTROL nonsense", "a wooden door"),
    ("CONTROL nonsense", "a bowl of soup"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
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
    model = (ckpt["model"] if isinstance(ckpt, dict) else ckpt).float().to(args.device).eval()
    tc = [h for h in model._forward_pre_hooks.values() if isinstance(h, TextConditioner)][0]

    img = cv2.imread(str(args.image))
    img_r = cv2.resize(img, (args.imgsz, args.imgsz))
    img_t = torch.from_numpy(img_r[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    print(f"checkpoint : {args.checkpoint}")
    print(f"image      : {args.image.name}   ground truth: scratches")
    print("every query below is ABSENT from the corpus\n")
    print(f"{'kind':<20}{'query':<48}{'conf':>8}")
    print("-" * 78)

    hits = 0
    for kind, q in NOVEL:
        tc.class_texts = [q]
        model.model[-1].nc = 1
        with torch.no_grad():
            y, _ = model(img_t)
        dets = non_max_suppression(y, conf_thres=0.01, iou_thres=0.7, nc=1)[0]
        best = float(dets[:, 4].max()) if len(dets) else 0.0
        if kind == "novel paraphrase" and best > 0.25:
            hits += 1
        print(f"{kind:<20}{q[:46]!r:<48}{best:>8.3f}")

    n = sum(1 for k, _ in NOVEL if k == "novel paraphrase")
    print(f"\nnovel paraphrases detected above 0.25: {hits}/{n}")
    print("High -> genuine generalisation, more phrases will help.")
    print("Low  -> coverage only; a larger corpus just adds lookup entries.")


if __name__ == "__main__":
    main()
