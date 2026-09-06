"""Visual sanity check of a YOLO conversion.

Draws the converted (normalised) boxes back onto the images. If the conversion
maths is wrong, the boxes visibly miss the defects.

Run:
    python scripts/verify_labels.py --dataset neu
    python scripts/verify_labels.py --dataset gc10
    python scripts/verify_labels.py --dataset deepcrack
Writes:
    phase-notes/assets/label_check_<dataset>.png
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DATASETS = {
    "neu": (REPO_ROOT / "datasets" / "neu-det-yolo", "neu_closed.yaml", 2),
    "gc10": (REPO_ROOT / "datasets" / "gc10-det-yolo", "gc10_closed.yaml", 2),
    "deepcrack": (REPO_ROOT / "datasets" / "deepcrack-yolo", "deepcrack_ood.yaml", 3),
}

PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#008080",
]
SEED = 7


def load_boxes(data_dir, stem):
    boxes = []
    for line in (data_dir / "labels" / f"{stem}.txt").read_text().strip().splitlines():
        parts = line.split()
        boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="neu")
    ap.add_argument("--samples", type=int, default=0,
                    help="override number of sample images")
    args = ap.parse_args()

    data_dir, yaml_name, per_group = DATASETS[args.dataset]
    if not data_dir.exists():
        raise SystemExit(f"missing {data_dir} - run the matching prepare_*.py first")

    names = yaml.safe_load((data_dir / yaml_name).read_text())["names"]
    classes = [names[i] for i in sorted(names)]

    rng = random.Random(SEED)
    stems = sorted(p.stem for p in (data_dir / "labels").glob("*.txt"))

    # Pick samples that between them exercise as many classes as possible.
    chosen, seen = [], set()
    pool = stems[:]
    rng.shuffle(pool)
    for stem in pool:
        present = {b[0] for b in load_boxes(data_dir, stem)}
        if present - seen or len(chosen) < per_group:
            chosen.append(stem)
            seen |= present
        if len(chosen) >= (args.samples or max(len(classes), 6)):
            break

    cols = 3
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.2))
    axes = axes.reshape(rows, cols)

    total = 0
    for i, stem in enumerate(chosen):
        ax = axes[i // cols][i % cols]
        img_path = next((data_dir / "images").glob(f"{stem}.*"))
        img = Image.open(img_path)
        W, H = img.size
        ax.imshow(img, cmap="gray")

        for cls_idx, cx, cy, w, h in load_boxes(data_dir, stem):
            px, py = (cx - w / 2) * W, (cy - h / 2) * H
            color = PALETTE[cls_idx % len(PALETTE)]
            ax.add_patch(patches.Rectangle(
                (px, py), w * W, h * H,
                linewidth=1.4, edgecolor=color, facecolor="none"))
            ax.text(px, max(py - 3, 8), classes[cls_idx],
                    color="white", fontsize=6,
                    bbox=dict(facecolor=color, edgecolor="none", pad=0.8))
            total += 1

        ax.set_title(f"{stem}  ({W}x{H})", fontsize=7)
        ax.axis("off")

    for j in range(len(chosen), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(f"Label verification - {args.dataset}: converted YOLO boxes "
                 f"drawn on source images", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = REPO_ROOT / "phase-notes" / "assets" / f"label_check_{args.dataset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"checked {len(chosen)} images, {total} boxes")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
