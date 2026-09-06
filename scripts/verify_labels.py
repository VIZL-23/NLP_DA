"""Phase 1 - Visual sanity check of the YOLO conversion.

Draws the converted (normalised) boxes back onto the images. If the conversion
maths is wrong, the boxes will visibly miss the defects.

Run:
    python scripts/verify_labels.py
Writes:
    phase-notes/assets/label_check.png
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "datasets" / "neu-det-yolo"
OUT_PNG = REPO_ROOT / "phase-notes" / "assets" / "label_check.png"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4"]

SEED = 7
PER_CLASS = 2  # sample images per class


def load_boxes(stem):
    """Read a YOLO label file -> [(cls_idx, cx, cy, w, h)]."""
    path = DATA / "labels" / f"{stem}.txt"
    boxes = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def main():
    rng = random.Random(SEED)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    # Pick sample images whose filename starts with each class name.
    all_stems = sorted(p.stem for p in (DATA / "labels").glob("*.txt"))
    chosen = []
    for cls in CLASSES:
        pool = [s for s in all_stems if s.startswith(cls)]
        rng.shuffle(pool)
        chosen.extend(pool[:PER_CLASS])

    cols = PER_CLASS
    rows = len(CLASSES)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = axes.reshape(rows, cols)

    total_boxes = 0
    for i, stem in enumerate(chosen):
        ax = axes[i // cols][i % cols]
        img = Image.open(DATA / "images" / f"{stem}.jpg")
        W, H = img.size
        ax.imshow(img, cmap="gray")

        for cls_idx, cx, cy, w, h in load_boxes(stem):
            # Denormalise back to pixels; if the maths is right these land on
            # the defect.
            px = (cx - w / 2) * W
            py = (cy - h / 2) * H
            pw, ph = w * W, h * H
            ax.add_patch(patches.Rectangle(
                (px, py), pw, ph,
                linewidth=1.8, edgecolor=COLORS[cls_idx], facecolor="none",
            ))
            ax.text(
                px, max(py - 3, 8), CLASSES[cls_idx],
                color="white", fontsize=7,
                bbox=dict(facecolor=COLORS[cls_idx], edgecolor="none", pad=1),
            )
            total_boxes += 1

        ax.set_title(f"{stem}  ({W}x{H})", fontsize=8)
        ax.axis("off")

    fig.suptitle(
        "Phase 1 label verification - converted YOLO boxes drawn on source images",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_PNG, dpi=110)
    print(f"checked {len(chosen)} images, {total_boxes} boxes")
    print(f"written: {OUT_PNG}")


if __name__ == "__main__":
    main()
