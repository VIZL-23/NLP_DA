"""Phase 1b - Convert DeepCrack segmentation masks to YOLO boxes.

DeepCrack ships binary pixel masks, not boxes. Report section 4.4 holds it out
entirely for out-of-distribution zero-shot evaluation, so this dataset is
EVAL-ONLY - it is never trained on.

The mask->box problem
---------------------
A crack is thin and usually diagonal, so its bounding box is far larger than
the crack itself. Taking ONE box per image would produce a near-full-image
rectangle that carries no information.

We therefore run connected-component analysis and emit one box per crack
SEGMENT. That is much better, but boxes are still loose by construction - a
diagonal segment fills only a fraction of its own box. The script reports this
"fill ratio" explicitly so the limitation is measured rather than hidden; it
belongs in the report rather than being quietly ignored.

Run:
    python scripts/prepare_deepcrack.py
    python scripts/prepare_deepcrack.py --min-area 40
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tgfem.data import DEEPCRACK_CLASSES  # noqa: E402

SRC = REPO_ROOT / "datasets" / "DeepCrack"
OUT = REPO_ROOT / "datasets" / "deepcrack-yolo"


def find_split_dir(name):
    """Locate a split folder, preferring the most populated match.

    Extracting the release can leave empty duplicates of these folders.
    """
    candidates = [d for d in SRC.rglob(name) if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: len(list(d.glob("*"))))


def mask_to_boxes(mask, min_area):
    """Connected components -> [(x1, y1, x2, y2, pixel_area)]."""
    binary = (mask > 127).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    for i in range(1, n):  # 0 is background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append((x, y, x + w, y + h, int(area)))
    return boxes


def convert_split(split, img_dir, lab_dir, min_area, stats):
    images_out = OUT / "images"
    labels_out = OUT / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    written = []
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".png", ".bmp"}:
            continue
        mask_path = next((p for p in lab_dir.glob(f"{img_path.stem}.*")), None)
        if mask_path is None:
            stats["no_mask"] += 1
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            stats["unreadable"] += 1
            continue

        h_img, w_img = mask.shape
        boxes = mask_to_boxes(mask, min_area)
        if not boxes:
            stats["no_boxes"] += 1
            continue

        lines = []
        for x1, y1, x2, y2, pix in boxes:
            cx = ((x1 + x2) / 2) / w_img
            cy = ((y1 + y2) / 2) / h_img
            bw = (x2 - x1) / w_img
            bh = (y2 - y1) / h_img
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            box_px = max((x2 - x1) * (y2 - y1), 1)
            stats["fill_ratios"].append(pix / box_px)
            stats["box_areas"].append(bw * bh)
            stats["boxes"] += 1

        # Prefix the stem so train/test names can never collide in the flat pool.
        stem = f"{split}_{img_path.stem}"
        shutil.copy2(img_path, images_out / f"{stem}{img_path.suffix}")
        (labels_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        written.append(f"./images/{stem}{img_path.suffix}")
        stats["images"] += 1
        stats["crack_pixel_frac"].append(float((mask > 127).mean()))

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-area", type=int, default=20,
                    help="drop connected components smaller than this many pixels")
    args = ap.parse_args()

    print("=" * 62)
    print("Phase 1b - DeepCrack mask -> box conversion")
    print("=" * 62)
    print(f"min component area: {args.min_area} px\n")

    if OUT.exists():
        shutil.rmtree(OUT)

    stats = {
        "images": 0, "boxes": 0, "no_mask": 0, "unreadable": 0, "no_boxes": 0,
        "fill_ratios": [], "box_areas": [], "crack_pixel_frac": [],
    }

    manifests = {}
    for split, img_name, lab_name in (
        ("train", "train_img", "train_lab"),
        ("test", "test_img", "test_lab"),
    ):
        img_dir, lab_dir = find_split_dir(img_name), find_split_dir(lab_name)
        if img_dir is None or lab_dir is None:
            print(f"{split:<6}: MISSING ({img_name}/{lab_name} not found)")
            continue
        files = convert_split(split, img_dir, lab_dir, args.min_area, stats)
        manifests[split] = files
        print(f"{split:<6}: {len(files)} images")

    every = [f for files in manifests.values() for f in files]
    for name, files in [*manifests.items(), ("all", every)]:
        (OUT / f"{name}.txt").write_text("\n".join(sorted(files)) + "\n")

    print(f"\nimages converted : {stats['images']}")
    print(f"boxes written    : {stats['boxes']}")
    print(f"  avg boxes/image: {stats['boxes'] / max(stats['images'], 1):.1f}")
    for k, label in (("no_mask", "image without mask"),
                     ("unreadable", "unreadable mask"),
                     ("no_boxes", "mask with no component above min-area")):
        if stats[k]:
            print(f"  {label}: {stats[k]}")

    fills = np.array(stats["fill_ratios"])
    areas = np.array(stats["box_areas"])
    crackpx = np.array(stats["crack_pixel_frac"])

    print("\nTHE MASK->BOX LIMITATION, MEASURED:")
    print(f"  crack pixels per image : {crackpx.mean()*100:.2f}% (mean)")
    print(f"  median box area        : {np.median(areas)*100:.2f}% of image")
    print(f"  median box fill ratio  : {np.median(fills)*100:.1f}% "
          f"(crack pixels / box pixels)")
    print(f"  boxes filled < 25%     : {(fills < 0.25).mean()*100:.1f}%")
    print("  ^ a box is mostly background by construction: a diagonal crack")
    print("    segment cannot fill an axis-aligned rectangle. State this in the")
    print("    report rather than letting a reader assume tight boxes.")

    (OUT / "deepcrack_ood.yaml").write_text(
        "# Generated by scripts/prepare_deepcrack.py - do not edit by hand.\n"
        "# EVAL ONLY. Report section 4.4 holds DeepCrack out entirely for\n"
        "# out-of-distribution zero-shot evaluation - never train on it.\n"
        "# Boxes are derived from segmentation masks and are loose by\n"
        "# construction; see the fill-ratio statistics in the Phase 1b notes.\n\n"
        f"path: {OUT.as_posix()}\n"
        "train: all.txt   # present only because Ultralytics requires the key\n"
        "val: all.txt\n"
        "test: all.txt\n\n"
        "names:\n" + "\n".join(f"  {i}: {c}" for i, c in enumerate(DEEPCRACK_CLASSES)) + "\n"
    )

    print(f"\nOutput written to: {OUT}")
    print("Done.\n")


if __name__ == "__main__":
    main()
