"""Phase 0 environment check.

Verifies that the toolchain and the datasets are in a usable state before
any training work begins. Run this first on any new machine:

    python scripts/check_env.py
"""

import sys
from collections import Counter
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

# GC10-DET labels its objects with Chinese pinyin codes, not the English folder
# names. This mapping was derived empirically by cross-referencing every XML
# against the image's class folder.
#
# Two data-quality fixes are baked in:
#   * "10_yaozhe" and "10_yaozhed" are the same class (a typo variant) - merged.
#   * "d" is a corrupt label with a single object - deliberately absent, so the
#     converter drops it.
GC10_NAME_MAP = {
    "1_chongkong": "punching_hole",
    "2_hanfeng": "welding_line",
    "3_yueyawan": "crescent_gap",
    "4_shuiban": "water_spot",
    "5_youban": "oil_spot",
    "6_siban": "silk_spot",
    "7_yiwu": "inclusion",
    "8_yahen": "rolled_pit",
    "9_zhehen": "crease",
    "10_yaozhe": "waist_folding",
    "10_yaozhed": "waist_folding",
}


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


def check_gc10():
    line("Dataset: GC10-DET")
    root = DATASETS / "GC10-DET"
    if not root.exists():
        print(f"MISSING    : {root}")
        print("             download: kaggle datasets download -d "
              "zhangyunsheng/defects-class-and-location")
        return

    images_dir = root / "images" / "images"
    label_dir = root / "label" / "label"
    if not images_dir.exists() or not label_dir.exists():
        print("STRUCTURE  : unexpected - expected images/images/<class>/ and label/label/*.xml")
        return

    stems = {}
    for cls_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        for img in cls_dir.glob("*.jpg"):
            stems[img.stem] = cls_dir.name

    xmls = list(label_dir.glob("*.xml"))
    print(f"images     : {len(stems)}")
    print(f"annotations: {len(xmls)}")
    print(f"unlabelled : {len(stems) - len(xmls)} images have no XML")

    import xml.etree.ElementTree as ET
    raw_names = Counter()
    sizes = Counter()
    objects = 0
    for x in xmls:
        root_el = ET.parse(x).getroot()
        size = root_el.find("size")
        sizes[(size.find("width").text, size.find("height").text)] += 1
        for obj in root_el.findall("object"):
            raw_names[obj.find("name").text] += 1
            objects += 1

    print(f"objects    : {objects}")
    print(f"image size : {sizes.most_common(1)[0][0]} for "
          f"{sizes.most_common(1)[0][1]} images")
    print(f"raw classes: {len(raw_names)} distinct names in the XML")

    known = set(GC10_NAME_MAP)
    unknown = {k: v for k, v in raw_names.items() if k not in known}
    if unknown:
        print(f"UNMAPPED   : {unknown}  <- dropped during conversion")

    print("\n  XML name -> canonical class (objects):")
    merged = Counter()
    for raw, count in raw_names.items():
        if raw in GC10_NAME_MAP:
            merged[GC10_NAME_MAP[raw]] += count
    for cls in sorted(merged, key=lambda c: -merged[c]):
        print(f"    {cls:<16} {merged[cls]:>5}")
    print("  NOTE: folders show the DOMINANT defect only - images are multi-label,")
    print("        so the XML is the ground truth, not the folder name.")


def check_deepcrack():
    line("Dataset: DeepCrack")
    root = DATASETS / "DeepCrack"
    if not root.exists():
        print(f"MISSING    : {root}")
        return

    # The correct release (Liu et al., 537 images) ships train_img/train_lab/
    # test_img/test_lab. The Zou et al. code repo has none of these.
    candidates = list(root.rglob("train_img")) + list(root.rglob("test_img"))
    n_images = len([p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".bmp"}])

    if not candidates:
        print("WRONG CONTENT: this looks like the DeepCrack *code* repository,")
        print("               not the dataset. Expected train_img/ train_lab/")
        print(f"               test_img/ test_lab/ - found {n_images} images total")
        print("               (those are paper figures).")
        print("  correct source: https://github.com/yhlleo/DeepCrack  ->  ./dataset")
        return

    for sub in ("train_img", "train_lab", "test_img", "test_lab"):
        # Extracting the zip can leave empty duplicates of these folders, so
        # take the most populated match rather than the first one found.
        found = list(root.rglob(sub))
        count = max((len(list(d.glob("*"))) for d in found), default=0)
        status = "OK" if count else "MISSING"
        extra = f"  ({len(found)} dirs match, using the populated one)" if len(found) > 1 else ""
        print(f"{sub:<11}: {count} files  [{status}]{extra}")
    print("NOTE       : segmentation masks, not boxes - needs mask->box conversion.")


def main():
    print("TG-FEM project - environment & dataset check")
    check_python()
    check_torch()
    check_ultralytics()
    check_neu_det()
    check_sdnet()
    check_gc10()
    check_deepcrack()
    print("\nDone.\n")


if __name__ == "__main__":
    main()
