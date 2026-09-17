"""Phase 7 - how robust is the text interface to WORDING?

The project's premise is that an inspector types a plain-English description
and gets boxes. That only holds if the model responds to the MEANING of a
phrase, not to the exact string it was trained on.

This probes exactly that: one image, one true defect class, many ways of
phrasing it, plus controls. It reuses the checkpoint's trained context tokens
and swaps only the query text (same mechanism as scripts/demo.py).

One caveat that decides how to read the output: with --phrase-aug, EVERY
corpus phrasing was seen during training. So the "corpus, other tier" row
measures vocabulary COVERAGE, not generalisation. Only the "descriptive" and
"loose paraphrase" rows are genuinely novel wording, and those are the ones
that answer whether the text interface generalises - see PHASE-7.md section 4
and scripts/probe_generalisation.py.

Run:
    python scripts/probe_wording.py --checkpoint runs/... --set scratches
    python scripts/probe_wording.py --checkpoint runs/... --set spalling
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# (label, query) - graded from the exact training string outwards.
# `truth` is the ground-truth class of the default image; `wrong class` must be
# a class the model WAS trained on, so a null there is discrimination rather
# than unfamiliarity.
QUERY_SETS = {
    "scratches": {
        "truth": "scratches",
        "image": "app/samples/neu_crack",
        "queries": [
            ("EXACT (trained)",    "scratches on the steel surface"),
            ("near paraphrase",    "scratches on the metal surface"),
            ("corpus, other tier", "long thin scratch marks on the metal"),
            ("bare class name",    "scratches"),
            ("descriptive",        "a long straight bright line running across the surface"),
            ("loose paraphrase",   "long thin gouges scored into metal"),
            ("wrong class",        "a pitted steel surface"),
            ("CONTROL nonsense",   "a wooden door"),
        ],
    },
    "spalling": {
        "truth": "spalling",
        "image": "app/samples/concrete",
        "queries": [
            ("EXACT (trained)",    "spalling on the concrete surface"),
            ("near paraphrase",    "spalling on the concrete wall"),
            ("corpus, other tier", "concrete broken away from the surface"),
            ("bare class name",    "spalling"),
            ("descriptive",        "a broken-out hollow exposing the stony interior"),
            ("loose paraphrase",   "chunks missing from the face of the wall"),
            ("wrong class",        "efflorescence on the concrete surface"),
            ("CONTROL nonsense",   "a wooden door"),
        ],
    },
    "crack": {
        "truth": "crack",
        "image": "app/samples/concrete",
        "queries": [
            ("EXACT (trained)",    "a crack in the concrete surface"),
            ("near paraphrase",    "a crack in the concrete wall"),
            ("corpus, other tier", "a thin fracture line running across the pavement"),
            ("bare class name",    "crack"),
            ("descriptive",        "a dark split running through the stone"),
            ("loose paraphrase",   "a thin break travelling across the slab"),
            ("wrong class",        "spalling on the concrete surface"),
            ("CONTROL nonsense",   "a wooden door"),
        ],
    },
}


def corpus_phrases() -> set[str]:
    """Every phrase the corpus contains, for the `seen?` column.

    The hand-written labels below say what a phrase is MEANT to be; this says
    whether the model was actually trained on it. They disagree more often than
    you would expect - "a long straight bright line running across the surface"
    reads like free description but is a visual-tier corpus entry, so scoring on
    it proves coverage and nothing about generalisation. Compute it, never
    assume it.
    """
    import json
    corpus = json.loads((REPO_ROOT / "prompts" / "defect_corpus.json").read_text())
    return {d["text"] for c in corpus["classes"].values() for d in c["descriptions"]}


def default_image(spec) -> Path:
    """First bundled sample whose filename names the ground-truth class."""
    d = REPO_ROOT / spec["image"]
    hits = sorted(d.glob(f"{spec['truth']}__*")) if d.is_dir() else []
    if not hits:
        raise SystemExit(
            f"no '{spec['truth']}' sample in {d} - run scripts/make_samples.py, "
            "or pass --image explicitly"
        )
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "runs/phase6_neu_crack_closed_tgfem_phraseaug/weights/best.pt")
    ap.add_argument("--set", dest="qset", choices=sorted(QUERY_SETS), default="scratches",
                    help="which graded query set to probe with")
    ap.add_argument("--image", type=Path, default=None,
                    help="default: a bundled sample of the set's ground-truth class")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    spec = QUERY_SETS[args.qset]
    queries = spec["queries"]
    if args.image is None:
        args.image = default_image(spec)

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

    trained = next((t for t in tc.class_texts if spec["truth"] in t), tc.class_texts[0])
    print(f"model  : {args.checkpoint.parent.parent.name}")
    print(f"image  : {args.image.name}  (ground truth: {spec['truth']})")
    print(f"n_ctx  : {tc.learner.n_ctx}   trained phrase: {trained!r}")
    print()
    seen = corpus_phrases()
    print(f"{'kind':<20}{'query':<56}{'seen?':>7}{'best conf':>10}{'n>0.25':>8}")
    print("-" * 101)
    novel_hits = novel_total = 0

    for kind, q in queries:
        # One query at a time, so each phrase is scored on its own merits
        # rather than competing in a softmax against the others.
        tc.class_texts = [q]
        model.model[-1].nc = 1
        with torch.no_grad():
            y, _ = model(img_t)
        dets = non_max_suppression(y, conf_thres=0.01, iou_thres=0.7, nc=1)[0]
        best = float(dets[:, 4].max()) if len(dets) else 0.0
        n_strong = int((dets[:, 4] > 0.25).sum()) if len(dets) else 0
        is_seen = q in seen
        if not is_seen and not kind.startswith(("wrong", "CONTROL")):
            novel_total += 1
            novel_hits += n_strong > 0
        mark = "seen" if is_seen else "NOVEL"
        print(f"{kind:<20}{q[:54]!r:<56}{mark:>7}{best:>10.3f}{n_strong:>8}")

    print()
    print(f"GENERALISATION: {novel_hits}/{novel_total} genuinely novel phrasings "
          f"(seen=NOVEL, excluding controls) returned a box above 0.25.")
    print()
    print("If only the EXACT trained string scores, the model has keyed on that")
    print("specific string rather than learning a general text interface - which")
    print("is the difference between open-vocabulary detection and a lookup table.")
    print()
    print("Rows marked `seen` were in the training corpus and, under --phrase-aug,")
    print("were trained on directly - they measure COVERAGE. Only `NOVEL` rows say")
    print("anything about whether the text interface generalises.")


if __name__ == "__main__":
    main()
