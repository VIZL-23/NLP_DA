"""Phase 1b - Convert GC10-DET (Pascal VOC) to YOLO format and build splits.

Source layout (as shipped):
    datasets/GC10-DET/images/images/<class>/*.jpg   # folder = DOMINANT defect only
    datasets/GC10-DET/label/label/*.xml             # flat, pinyin class names

Three source quirks are handled here; getting any of them wrong produces
silently corrupt labels:

  1. Class names in the XML are Chinese pinyin codes ("3_yueyawan"), not the
     English folder names. Mapped via GC10_NAME_MAP.
  2. "10_yaozhe" and "10_yaozhed" are the same class (typo) -> merged.
     A corrupt class named "d" (1 object) is dropped.
  3. THE FOLDER IS NOT THE LABEL. Images are multi-label; the folder records
     only the dominant defect. The XML is ground truth.

Run:
    python scripts/prepare_gc10.py
"""

import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tgfem.data import GC10_CLASSES, GC10_HELD_OUT, GC10_NAME_MAP  # noqa: E402

SRC = REPO_ROOT / "datasets" / "GC10-DET"
OUT = REPO_ROOT / "datasets" / "gc10-det-yolo"

CLASS_TO_IDX = {c: i for i, c in enumerate(GC10_CLASSES)}
SEED = 42


def index_images():
    """Map image stem -> path across every class folder."""
    index = {}
    for cls_dir in (SRC / "images" / "images").iterdir():
        if cls_dir.is_dir():
            for img in cls_dir.glob("*.jpg"):
                index[img.stem] = img
    return index


def voc_to_yolo(xmin, ymin, xmax, ymax, width, height):
    cx = ((xmin + xmax) / 2.0) / width
    cy = ((ymin + ymax) / 2.0) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    clamp = lambda v: max(0.0, min(1.0, v))
    return clamp(cx), clamp(cy), clamp(w), clamp(h)


def convert():
    img_index = index_images()

    images_out = OUT / "images"
    labels_out = OUT / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    records = {}
    stats = {
        "boxes": 0,
        "dropped_unmapped": Counter(),
        "degenerate": 0,
        "no_image": 0,
        "per_class": Counter(),
        "areas": [],
        "aspects": [],
        "folder_mismatch": 0,
    }

    for xml_path in sorted((SRC / "label" / "label").glob("*.xml")):
        img_path = img_index.get(xml_path.stem)
        if img_path is None:
            stats["no_image"] += 1
            continue

        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        width, height = int(size.find("width").text), int(size.find("height").text)

        lines, present = [], set()
        for obj in root.findall("object"):
            raw = obj.find("name").text.strip()
            if raw not in GC10_NAME_MAP:
                stats["dropped_unmapped"][raw] += 1
                continue
            cls = GC10_NAME_MAP[raw]

            box = obj.find("bndbox")
            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)
            if xmax <= xmin or ymax <= ymin:
                stats["degenerate"] += 1
                continue

            cx, cy, w, h = voc_to_yolo(xmin, ymin, xmax, ymax, width, height)
            lines.append(f"{CLASS_TO_IDX[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            present.add(cls)
            stats["boxes"] += 1
            stats["per_class"][cls] += 1
            stats["areas"].append(w * h)
            stats["aspects"].append(max(w, h) / min(w, h) if min(w, h) > 0 else 0)

        if not lines:
            continue

        # Evidence for "the folder is not the label": count images whose folder
        # name is not even among the classes the XML actually annotates.
        folder = img_path.parent.name.replace(" ", "_")
        folder = "gc10_inclusion" if folder == "inclusion" else folder
        if folder not in present:
            stats["folder_mismatch"] += 1

        shutil.copy2(img_path, images_out / img_path.name)
        (labels_out / f"{xml_path.stem}.txt").write_text("\n".join(lines) + "\n")
        records[xml_path.stem] = {"image": f"./images/{img_path.name}", "classes": present}

    return records, stats


def split_protocol(records, rng, held_out):
    """Split into train/val/test.

    With held_out empty this is the closed protocol (stratified 80/10/10).
    Otherwise any image containing a held-out object goes to test and is never
    seen during training - the leakage guard.
    """
    if not held_out:
        buckets = defaultdict(list)
        for stem, rec in records.items():
            buckets[",".join(sorted(rec["classes"]))].append(stem)
        train, val, test = [], [], []
        for _, stems in sorted(buckets.items()):
            stems = sorted(stems)
            rng.shuffle(stems)
            n = len(stems)
            n_tr, n_va = int(n * 0.8), int(n * 0.1)
            train += stems[:n_tr]
            val += stems[n_tr:n_tr + n_va]
            test += stems[n_tr + n_va:]
        return {"train": train, "val": val, "test": test}

    seen, heldout = [], []
    for stem, rec in records.items():
        (heldout if rec["classes"] & held_out else seen).append(stem)

    buckets = defaultdict(list)
    for stem in seen:
        buckets[",".join(sorted(records[stem]["classes"]))].append(stem)

    train, val = [], []
    for _, stems in sorted(buckets.items()):
        stems = sorted(stems)
        rng.shuffle(stems)
        n_tr = int(len(stems) * 0.9)
        train += stems[:n_tr]
        val += stems[n_tr:]

    return {"train": train, "val": val, "test": sorted(heldout)}


def write_splits(name, splits, records):
    for subset, stems in splits.items():
        path = OUT / f"{name}_{subset}.txt"
        path.write_text("\n".join(records[s]["image"] for s in sorted(stems)) + "\n")
    return {k: len(v) for k, v in splits.items()}


def write_yaml(protocol, note):
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(GC10_CLASSES))
    (OUT / f"gc10_{protocol}.yaml").write_text(
        f"# Generated by scripts/prepare_gc10.py - do not edit by hand.\n"
        f"# {note}\n\n"
        f"path: {OUT.as_posix()}\n"
        f"train: {protocol}_train.txt\n"
        f"val: {protocol}_val.txt\n"
        f"test: {protocol}_test.txt\n\n"
        f"names:\n{names}\n"
    )


def main():
    rng = random.Random(SEED)
    print("=" * 62)
    print("Phase 1b - GC10-DET conversion")
    print("=" * 62)

    if OUT.exists():
        shutil.rmtree(OUT)

    records, stats = convert()

    print(f"\nimages converted : {len(records)}")
    print(f"boxes written    : {stats['boxes']}")
    print(f"xml w/o image    : {stats['no_image']}")
    print(f"degenerate boxes : {stats['degenerate']}")
    if stats["dropped_unmapped"]:
        print(f"dropped (unmapped classes): {dict(stats['dropped_unmapped'])}")
    print(f"images whose FOLDER name is not among their XML classes: "
          f"{stats['folder_mismatch']}")
    print("  ^ proof the folder is not the label; the XML is ground truth")

    print("\nboxes per class:")
    for cls in GC10_CLASSES:
        flag = "  [HELD OUT]" if cls in GC10_HELD_OUT else ""
        print(f"  {cls:<18} {stats['per_class'][cls]:>5}{flag}")

    areas = sorted(stats["areas"])
    aspects = sorted(stats["aspects"])
    print("\nbox geometry:")
    print(f"  median area    : {areas[len(areas)//2]*100:.2f}% of image")
    print(f"  boxes < 2% area: {sum(1 for a in areas if a < 0.02)/len(areas)*100:.1f}%")
    print(f"  boxes > 10:1   : {sum(1 for a in aspects if a > 10)/len(aspects)*100:.1f}%")

    closed = split_protocol(records, rng, set())
    c = write_splits("closed", closed, records)
    write_yaml("closed", "All 10 classes. Closed-vocabulary baseline.")
    print("\n" + "-" * 62)
    print("Protocol: CLOSED (all 10 classes, 80/10/10)")
    print("-" * 62)
    for k in ("train", "val", "test"):
        print(f"  {k:<6} {c[k]:>5} images")

    openv = split_protocol(records, rng, GC10_HELD_OUT)
    o = write_splits("openvocab", openv, records)
    write_yaml("openvocab", "Held out: " + ", ".join(sorted(GC10_HELD_OUT)) + ".")
    print("\n" + "-" * 62)
    print("Protocol: OPEN VOCAB (held out: " + ", ".join(sorted(GC10_HELD_OUT)) + ")")
    print("-" * 62)
    for k in ("train", "val", "test"):
        print(f"  {k:<6} {o[k]:>5} images")

    leaks = [s for k in ("train", "val") for s in openv[k]
             if records[s]["classes"] & GC10_HELD_OUT]
    print(f"  leakage check : {'PASS (none)' if not leaks else f'FAIL {len(leaks)}'}")

    kept = Counter()
    for s in openv["train"]:
        for cls in records[s]["classes"]:
            kept[cls] += 1
    total = Counter()
    for rec in records.values():
        for cls in rec["classes"]:
            total[cls] += 1
    print("\n  seen-class retention in openvocab train:")
    for cls in sorted(kept, key=lambda c: -kept[c]):
        print(f"    {cls:<18} {kept[cls]:>4} / {total[cls]:<4} ({kept[cls]/total[cls]*100:>3.0f}%)")

    print(f"\nOutput written to: {OUT}")
    print("Done.\n")


if __name__ == "__main__":
    main()
