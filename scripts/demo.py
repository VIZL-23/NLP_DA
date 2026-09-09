"""Phase 8 - the live demo: type a description, get boxes.

This is the whole point of the project (README, "What this project does") -
an inspector types a plain-English phrase and gets back bounding boxes,
including for defect types never seen in training. Given a Phase 6
checkpoint, this script does exactly that for an arbitrary image + arbitrary
text queries (not limited to prompts/defect_corpus.json - open vocabulary
means literally any phrase).

How re-querying without retraining works: the checkpoint's embedded
TextConditioner (Phase 4) already carries LEARNED context tokens (`ctx`),
tuned during training to work well with CLIP's frozen text tower. Inference
reuses that exact `ctx` (CoOp's whole point: the learned prefix generalises
to new class names, not just the training vocabulary) and only swaps in the
new query strings - found by locating the TextConditioner already sitting in
the model's forward_pre_hooks (Phase 4/5's pickling design lets it travel
with the checkpoint) and mutating `.class_texts` in place.

Run:
    python scripts/demo.py --checkpoint runs/phase6_neu_closed_tgfem/weights/best.pt \\
        --image datasets/neu-det-yolo/images/crazing_1.jpg \\
        --query "crazing on a steel surface" "a pitted corroded surface"
Writes:
    <image_stem>_demo.png  (boxes drawn, labelled by query text + confidence)
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--query", nargs="+", required=True,
                     help="one or more free-text descriptions - any phrase, not limited to the training corpus")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import cv2
    import torch

    from tgfem import register
    from tgfem.detection_model import TGFEMModel  # noqa: F401 - needed for unpickling
    from tgfem.language import TextConditioner

    register()

    print(f"Loading {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) else ckpt
    model = model.float().to(args.device).eval()

    hooks = [h for h in model._forward_pre_hooks.values() if isinstance(h, TextConditioner)]
    if not hooks:
        raise SystemExit("checkpoint has no TextConditioner attached - was it trained with train_tgfem.py?")
    tc = hooks[0]

    print(f"Trained vocabulary ({len(tc.class_texts)}): {tc.class_texts}")
    print(f"Reusing learned context tokens (n_ctx={tc.learner.n_ctx}), swapping in new query text:")
    for q in args.query:
        print(f"  -> {q!r}")
    tc.class_texts = list(args.query)  # ctx stays the trained one - see module docstring
    # WorldDetect.forward derives self.no = nc + reg_max*4 from self.nc, which
    # is only updated when the vocabulary size changes here - not implicitly
    # from len(text) - mirrors WorldModel.set_classes()'s own `self.model[-1].nc = len(text)`.
    model.model[-1].nc = len(args.query)

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    h0, w0 = img.shape[:2]
    img_r = cv2.resize(img, (args.imgsz, args.imgsz))
    img_t = torch.from_numpy(img_r[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_t = img_t.to(args.device)

    with torch.no_grad():
        y, _ = model(img_t)  # forward_pre_hook fires here, encodes args.query fresh

    # y: (1, 4 + n_queries, n_anchors) after _inference() NMS-ready decode
    try:
        from ultralytics.utils.nms import non_max_suppression  # newer layout
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression  # older layout

    dets = non_max_suppression(y, conf_thres=args.conf, iou_thres=0.7, nc=len(args.query))[0]
    print(f"\n{len(dets)} detection(s) above conf={args.conf}:")

    # NEU-DET images are 200x200, so drawing at native size makes a query like
    # "scratches on the steel surface" several times wider than the whole frame.
    # Render onto an upscaled canvas instead, and size the text to the canvas.
    canvas_min = 640
    up = max(1.0, canvas_min / max(h0, w0))
    canvas = cv2.resize(img, (int(w0 * up), int(h0 * up)), interpolation=cv2.INTER_CUBIC)
    ch, cw = canvas.shape[:2]

    sx, sy = (w0 / args.imgsz) * up, (h0 / args.imgsz) * up
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    RED, WHITE = (0, 0, 255), (255, 255, 255)

    for *xyxy, conf, cls in dets.tolist():
        x1, y1, x2, y2 = (int(xyxy[0] * sx), int(xyxy[1] * sy),
                          int(xyxy[2] * sx), int(xyxy[3] * sy))
        label = args.query[int(cls)]
        # Report the box in ORIGINAL image pixels, not canvas pixels.
        print(f"  {label!r:<40} conf={conf:.3f}  "
              f"box=({int(x1 / up)},{int(y1 / up)},{int(x2 / up)},{int(y2 / up)})")

        cv2.rectangle(canvas, (x1, y1), (x2, y2), RED, 2)

        # Shorten the caption until it fits the canvas width.
        text = f"{label} {conf:.2f}"
        while cv2.getTextSize(text, font, fs, th)[0][0] > cw - 8 and len(text) > 12:
            label = label[:-4] + "..." if not label.endswith("...") else label[:-4] + "..."
            text = f"{label} {conf:.2f}"

        (tw, tht), base = cv2.getTextSize(text, font, fs, th)
        # Prefer above the box; drop inside it when there is no room at the top.
        ty = y1 - 4 if y1 - tht - base - 4 >= 0 else min(y1 + tht + 4, ch - 2)
        tx = max(0, min(x1, cw - tw - 4))
        cv2.rectangle(canvas, (tx, ty - tht - base), (tx + tw + 4, ty + 2), RED, -1)
        cv2.putText(canvas, text, (tx + 2, ty - 2), font, fs, WHITE, th, cv2.LINE_AA)

    out = args.out or Path(f"{args.image.stem}_demo.png")
    cv2.imwrite(str(out), canvas)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
