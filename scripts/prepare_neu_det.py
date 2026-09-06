"""Phase 1 - Convert NEU-DET (Pascal VOC) to YOLO format and build split protocols.

Source layout (as shipped):
    datasets/NEU-DET/{train,validation}/images/<class>/*.jpg   # per-class subfolders
    datasets/NEU-DET/{train,validation}/annotations/*.xml      # flat

Output layout (Ultralytics-compatible):
    datasets/neu-det-yolo/images/*.jpg     # flat pool, all 1800
    datasets/neu-det-yolo/labels/*.txt     # flat pool, global 6-class indices
    datasets/neu-det-yolo/<protocol>_<subset>.txt   # image-path manifests

Two protocols are produced:

  closed/    - all 6 classes, stratified 80/10/10. The closed-vocabulary
               baseline that defines the "within 3 mAP" retention criterion.

  openvocab/ - crazing and rolled-in_scale are HELD OUT. Any image containing
               even one held-out object is excluded from train/val entirely,
               so the model never sees those classes during training.

Run:
    python scripts/prepare_neu_det.py
"""

import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "datasets" / "NEU-DET"
OUT = REPO_ROOT / "datasets" / "neu-det-yolo"

# Global class indices - stable across BOTH protocols so label files are shared.
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Held out from training in the open-vocabulary protocol (report section 4.4).
HELD_OUT = {"crazing", "rolled-in_scale"}

SEED = 42
INCLUDE_DIFFICULT = True  # NEU-DET convention keeps difficult boxes


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def index_images():
    """Map image stem -> path, searching per-class subfolders in both splits.

    Pairing is done by filename stem rather than by folder, which transparently
    fixes the off-by-one annotation/image split mismatch found in Phase 0.
    """
    index = {}
    duplicates = []
    for split in ("train", "validation"):
        for img in (SRC / split / "images").rglob("*.jpg"):
            if img.stem in index:
                duplicates.append(img.stem)
            index[img.stem] = img
    return index, duplicates


def parse_annotation(xml_path):
    """Return (width, height, [(class_name, xmin, ymin, xmax, ymax, difficult)])."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        diff_node = obj.find("difficult")
        difficult = diff_node is not None and diff_node.text.strip() == "1"
        box = obj.find("bndbox")
        objects.append((
            name,
            float(box.find("xmin").text),
            float(box.find("ymin").text),
            float(box.find("xmax").text),
            float(box.find("ymax").text),
            difficult,
        ))
    return width, height, objects


def voc_to_yolo(xmin, ymin, xmax, ymax, width, height):
    """Pascal VOC corners -> YOLO normalised (cx, cy, w, h), clamped to [0, 1]."""
    cx = ((xmin + xmax) / 2.0) / width
    cy = ((ymin + ymax) / 2.0) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    clamp = lambda v: max(0.0, min(1.0, v))
    return clamp(cx), clamp(cy), clamp(w), clamp(h)


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

def convert():
    """Convert every annotation to YOLO format. Returns per-image metadata."""
    img_index, duplicates = index_images()
    if duplicates:
        print(f"WARNING: {len(duplicates)} duplicate image stems across splits")

    images_out = OUT / "images"
    labels_out = OUT / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(SRC.rglob("*.xml"))
    records = {}
    stats = {
        "boxes": 0,
        "skipped_difficult": 0,
        "degenerate": 0,
        "missing_image": [],
        "per_class": Counter(),
        "areas": [],
        "aspects": [],
    }

    for xml_path in xml_files:
        stem = xml_path.stem
        img_path = img_index.get(stem)
        if img_path is None:
            stats["missing_image"].append(stem)
            continue

        width, height, objects = parse_annotation(xml_path)

        lines = []
        classes_present = set()
        for name, xmin, ymin, xmax, ymax, difficult in objects:
            if difficult and not INCLUDE_DIFFICULT:
                stats["skipped_difficult"] += 1
                continue
            if name not in CLASS_TO_IDX:
                continue
            if xmax <= xmin or ymax <= ymin:
                stats["degenerate"] += 1
                continue

            cx, cy, w, h = voc_to_yolo(xmin, ymin, xmax, ymax, width, height)
            lines.append(f"{CLASS_TO_IDX[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            classes_present.add(name)
            stats["boxes"] += 1
            stats["per_class"][name] += 1
            stats["areas"].append(w * h)
            stats["aspects"].append(max(w, h) / min(w, h) if min(w, h) > 0 else 0)

        if not lines:
            continue

        shutil.copy2(img_path, images_out / img_path.name)
        (labels_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")

        records[stem] = {
            "image": f"./images/{img_path.name}",
            "classes": classes_present,
        }

    return records, stats


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def stratify_key(rec):
    """Group images by their sorted class signature so splits stay balanced."""
    return ",".join(sorted(rec["classes"]))


def split_closed(records, rng):
    """All 6 classes, stratified 80/10/10."""
    buckets = defaultdict(list)
    for stem, rec in records.items():
        buckets[stratify_key(rec)].append(stem)

    train, val, test = [], [], []
    for _, stems in sorted(buckets.items()):
        stems = sorted(stems)
        rng.shuffle(stems)
        n = len(stems)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train += stems[:n_train]
        val += stems[n_train:n_train + n_val]
        test += stems[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def split_openvocab(records, rng):
    """Hold out crazing + rolled-in_scale, with strict leakage exclusion."""
    seen_pool, heldout_pool = [], []
    for stem, rec in records.items():
        if rec["classes"] & HELD_OUT:
            # Contains at least one held-out object -> never used for training.
            heldout_pool.append(stem)
        else:
            seen_pool.append(stem)

    buckets = defaultdict(list)
    for stem in seen_pool:
        buckets[stratify_key(records[stem])].append(stem)

    train, val = [], []
    for _, stems in sorted(buckets.items()):
        stems = sorted(stems)
        rng.shuffle(stems)
        n_train = int(len(stems) * 0.9)
        train += stems[:n_train]
        val += stems[n_train:]

    heldout_pool = sorted(heldout_pool)
    rng.shuffle(heldout_pool)
    return {"train": train, "val": val, "test": heldout_pool}


def write_splits(name, splits, records):
    # Manifests live at the dataset ROOT, not in a subfolder: Ultralytics
    # expands a leading "./" against the manifest file's own parent directory,
    # so "./images/x.jpg" only resolves correctly from here.
    split_dir = OUT
    split_dir.mkdir(parents=True, exist_ok=True)
    for subset, stems in splits.items():
        path = split_dir / f"{name}_{subset}.txt"
        path.write_text("\n".join(records[s]["image"] for s in sorted(stems)) + "\n")
    return {k: len(v) for k, v in splits.items()}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def write_dataset_yaml(protocol, note):
    """Emit an Ultralytics data YAML.

    The `path` is written absolute on purpose: Ultralytics resolves a relative
    `path` against its global DATASETS_DIR setting, not against the YAML's own
    location, which would silently point somewhere else on another machine.
    Regenerate by re-running this script.
    """
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    content = f"""# Generated by scripts/prepare_neu_det.py - do not edit by hand.
# {note}

path: {OUT.as_posix()}
train: {protocol}_train.txt
val: {protocol}_val.txt
test: {protocol}_test.txt

# Global 6-class indices, identical across both protocols so the label files
# can be shared. In the openvocab protocol the held-out classes simply have
# no training examples.
names:
{names}
"""
    path = OUT / f"neu_{protocol}.yaml"
    path.write_text(content)
    return path


def verify_no_leakage(splits, records):
    """Assert that no training or validation image contains a held-out class."""
    offenders = []
    for subset in ("train", "val"):
        for stem in splits[subset]:
            if records[stem]["classes"] & HELD_OUT:
                offenders.append(stem)
    return offenders


def class_distribution(stems, records):
    counts = Counter()
    for stem in stems:
        for cls in records[stem]["classes"]:
            counts[cls] += 1
    return counts


def main():
    rng = random.Random(SEED)

    print("=" * 62)
    print("Phase 1 - NEU-DET conversion")
    print("=" * 62)

    if OUT.exists():
        shutil.rmtree(OUT)

    records, stats = convert()

    print(f"\nimages converted : {len(records)}")
    print(f"boxes written    : {stats['boxes']}")
    print(f"degenerate boxes : {stats['degenerate']}")
    if stats["missing_image"]:
        print(f"xml w/o image    : {len(stats['missing_image'])} {stats['missing_image'][:5]}")
    else:
        print("xml w/o image    : 0  (all annotations paired successfully)")

    print("\nboxes per class:")
    for cls in CLASSES:
        print(f"  {cls:<18} {stats['per_class'][cls]}")

    # Validate the constraints asserted in the problem statement.
    areas = sorted(stats["areas"])
    aspects = sorted(stats["aspects"])
    median_area = areas[len(areas) // 2] * 100
    mean_area = sum(areas) / len(areas) * 100
    under_2pct = sum(1 for a in areas if a < 0.02) / len(areas) * 100
    over_10to1 = sum(1 for a in aspects if a > 10) / len(aspects) * 100

    print("\nbox geometry (validates the problem-statement constraints):")
    print(f"  median box area   : {median_area:.2f}% of image")
    print(f"  mean box area     : {mean_area:.2f}% of image")
    print(f"  boxes < 2% area   : {under_2pct:.1f}%")
    print(f"  boxes > 10:1 ratio: {over_10to1:.1f}%")

    # -- Protocol 1: closed vocabulary --------------------------------------
    closed = split_closed(records, rng)
    closed_counts = write_splits("closed", closed, records)
    print("\n" + "-" * 62)
    print("Protocol: CLOSED vocabulary (all 6 classes, 80/10/10)")
    print("-" * 62)
    for subset in ("train", "val", "test"):
        print(f"  {subset:<6} {closed_counts[subset]:>5} images")

    # -- Protocol 2: open vocabulary ----------------------------------------
    openv = split_openvocab(records, rng)
    open_counts = write_splits("openvocab", openv, records)
    print("\n" + "-" * 62)
    print("Protocol: OPEN vocabulary (held out: " + ", ".join(sorted(HELD_OUT)) + ")")
    print("-" * 62)
    for subset in ("train", "val", "test"):
        print(f"  {subset:<6} {open_counts[subset]:>5} images")

    offenders = verify_no_leakage(openv, records)
    if offenders:
        print(f"  LEAKAGE: {len(offenders)} train/val images contain a held-out class!")
    else:
        print("  leakage check : PASS (no held-out class in train/val)")

    print("\n  seen-class distribution in openvocab train:")
    for cls, n in sorted(class_distribution(openv["train"], records).items()):
        print(f"    {cls:<18} {n}")

    # -- Dataset YAMLs ------------------------------------------------------
    y1 = write_dataset_yaml("closed", "All 6 classes. Closed-vocabulary baseline.")
    y2 = write_dataset_yaml(
        "openvocab",
        "Held out from training: " + ", ".join(sorted(HELD_OUT)) + ".",
    )
    print("\ndataset yamls:")
    print(f"  {y1.name}")
    print(f"  {y2.name}")

    print(f"\nOutput written to: {OUT}")
    print("Done.\n")


if __name__ == "__main__":
    main()
