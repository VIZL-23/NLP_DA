"""Phase 0 environment check.

Verifies that the toolchain and the datasets are in a usable state before
any training work begins. Run this first on any new machine:

    python scripts/check_env.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = REPO_ROOT / "datasets"

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def line(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_python():
    line("Python")
    print(f"version    : {sys.version.split()[0]}")
    print(f"executable : {sys.executable}")
    in_venv = sys.prefix != sys.base_prefix
    print(f"in venv    : {in_venv}")
    if not in_venv:
        print("WARNING    : not running inside .venv - activate it first")


def check_torch():
    line("PyTorch / GPU")
    try:
        import torch
    except ImportError:
        print("torch      : NOT INSTALLED - install from https://pytorch.org")
        return
    print(f"torch      : {torch.__version__}")
    print(f"cuda avail : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        print(f"gpu        : {props.name}")
        print(f"vram       : {vram_gb:.1f} GB")
        if vram_gb < 8:
            print("NOTE       : <8 GB VRAM - use small batch + gradient accumulation")
    else:
        print("WARNING    : CUDA unavailable, training will fall back to CPU")


def check_ultralytics():
    line("Ultralytics")
    try:
        import ultralytics
        print(f"version    : {ultralytics.__version__}")
    except ImportError:
        print("ultralytics: NOT INSTALLED - pip install -r requirements.txt")


def check_neu_det():
    line("Dataset: NEU-DET")
    root = DATASETS / "NEU-DET"
    if not root.exists():
        print(f"MISSING    : {root}")
        return

    total_images, total_labels = 0, 0
    for split in ("train", "validation"):
        images_dir = root / split / "images"
        ann_dir = root / split / "annotations"
        if not images_dir.exists():
            print(f"{split:<11}: images/ missing")
            continue

        n_images = 0
        per_class = []
        for cls in NEU_CLASSES:
            cls_dir = images_dir / cls
            count = len(list(cls_dir.glob("*.jpg"))) if cls_dir.exists() else 0
            per_class.append(f"{cls}={count}")
            n_images += count

        n_labels = len(list(ann_dir.glob("*.xml"))) if ann_dir.exists() else 0
        total_images += n_images
        total_labels += n_labels

        status = "OK" if n_images == n_labels else "MISMATCH"
        print(f"{split:<11}: {n_images} images / {n_labels} annotations  [{status}]")
        print(f"             {', '.join(per_class)}")

    print(f"{'TOTAL':<11}: {total_images} images / {total_labels} annotations")
    if total_images == total_labels:
        print("             per-split counts may differ (a stray XML sits in the "
              "wrong split) - resolved during Phase 1 conversion")


def check_sdnet():
    line("Dataset: SDNET2018")
    root = DATASETS / "SDNET2018"
    if not root.exists():
        print(f"MISSING    : {root}")
        return

    grand_total = 0
    for surface in ("Decks", "Pavements", "Walls"):
        surface_dir = root / surface
        if not surface_dir.exists():
            print(f"{surface:<11}: missing")
            continue
        counts = {}
        for label in ("Cracked", "Non-cracked"):
            label_dir = surface_dir / label
            counts[label] = len(list(label_dir.glob("*.jpg"))) if label_dir.exists() else 0
        subtotal = sum(counts.values())
        grand_total += subtotal
        print(f"{surface:<11}: {subtotal} images "
              f"(cracked={counts['Cracked']}, non-cracked={counts['Non-cracked']})")

    print(f"{'TOTAL':<11}: {grand_total} images")
    print("NOTE       : classification labels only - no bounding boxes. "
          "Usable for backbone warm-up / hard negatives, NOT detection training.")


def main():
    print("TG-FEM project - Phase 0 environment check")
    check_python()
    check_torch()
    check_ultralytics()
    check_neu_det()
    check_sdnet()
    print("\nDone.\n")


if __name__ == "__main__":
    main()
