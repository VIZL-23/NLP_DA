"""Convert the merged crack-segmentation parquet set to YOLO boxes.

Why this exists
---------------
The concrete/pavement specialist was trained on DeepCrack's 255 images, which
is far too small to hold up on anything but DeepCrack itself. This dataset is
the same task at ~44x the scale (11,298 image/mask pairs pooled from several
public crack sets - CFD, DeepCrack, GAPs, Rissbilder, Volker, cracktree and
others; the composition is printed at the end of a run).

It ships as HuggingFace parquet with two Image columns (`pixel_values`,
`label`), so unlike prepare_deepcrack.py there are no folders to locate - the
images are decoded straight out of the parquet row groups.

Two things the source composition forces
----------------------------------------
* ~1,400 rows are `noncrack_*`: crack-free concrete with an all-zero mask. They
  are NOT junk - they are exactly the hard negatives that stop the model firing
  on a blank wall. They are kept with an empty label file, which is how
  Ultralytics represents a background image.
* ~520 rows are `DeepCrack_*`. DeepCrack is the held-out out-of-distribution
  probe (see prepare_deepcrack.py), so training on those rows would quietly
  invalidate every OOD number we report. They are excluded by default;
  --keep-source DeepCrack overrides that if you know what you are giving up.

Mask threshold
--------------
The masks are binary with antialiased (JPEG-softened) edges: 15-17 unique grey
levels per mask, overwhelmingly 0 and 255, with a thin ramp between. `>127`
takes the crack core; `>0` would add the one-pixel halo. Verified on samples
before writing this - both produce boxes, `>127` gives tighter ones, so the
threshold matches prepare_deepcrack.py and the two datasets stay comparable.

The mask->box looseness documented in prepare_deepcrack.py applies here
identically and is re-measured below rather than assumed.

Run:
    python scripts/prepare_crack_merged.py
    python scripts/prepare_crack_merged.py --min-area 40 --limit 200
"""

import argparse
import io
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tgfem.data import DEEPCRACK_CLASSES  # noqa: E402

SRC = REPO_ROOT / "datasets" / "CRACK500" / "parquet"
OUT = REPO_ROOT / "datasets" / "crack-merged-yolo"
SEED = 42
# Held out as the OOD probe - see the module docstring.
EXCLUDE_SOURCES = {"DeepCrack"}
VAL_FRACTION = 0.1
BATCH = 64


def mask_to_boxes(mask, min_area):
    """Connected components -> [(x1, y1, x2, y2, pixel_area)]. Same rule as
    prepare_deepcrack.py so the two datasets stay directly comparable."""
    binary = (mask > 127).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return [(x, y, x + w, y + h, int(a))
            for x, y, w, h, a in (stats[i] for i in range(1, n)) if a >= min_area]


def source_of(path: str) -> str:
    """Sub-dataset name from the original filename, e.g. CFD_001.jpg -> CFD.
    Used for the composition report."""
    stem = Path(path).stem
    return stem.split("_")[0] if "_" in stem else "misc"


def convert(files, split_of, min_area, limit, exclude, stats):
    """Decode every row of every parquet file and write image + label pairs."""
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[str]] = {}
    seen: set[str] = set()
    done = 0

    for path, split_name in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=BATCH):
            rows = batch.to_pydict()
            for img_rec, mask_rec in zip(rows["pixel_values"], rows["label"]):
                if limit and done >= limit:
                    return manifests
                orig = img_rec["path"] or f"row{done}"
                src = source_of(orig)
                stats["sources"][src] += 1
                if src in exclude:
                    stats["excluded"] += 1
                    continue

                try:
                    img = np.array(Image.open(io.BytesIO(img_rec["bytes"])).convert("RGB"))
                    mask = np.array(Image.open(io.BytesIO(mask_rec["bytes"])).convert("L"))
                except Exception:
                    stats["unreadable"] += 1
                    continue
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

                h_img, w_img = mask.shape
                boxes = mask_to_boxes(mask, min_area)
                if not boxes:
                    # An empty mask on a `noncrack_*` row is a labelled negative,
                    # so keep it as a background image (empty label file). An
                    # empty mask on a row that claims to show a crack is a broken
                    # annotation, so drop it.
                    if not src.startswith("noncrack"):
                        stats["no_boxes"] += 1
                        continue
                    stats["backgrounds"] += 1

                lines = []
                for x1, y1, x2, y2, pix in boxes:
                    lines.append(
                        f"0 {((x1 + x2) / 2) / w_img:.6f} {((y1 + y2) / 2) / h_img:.6f} "
                        f"{(x2 - x1) / w_img:.6f} {(y2 - y1) / h_img:.6f}")
                    stats["fill_ratios"].append(pix / max((x2 - x1) * (y2 - y1), 1))
                    stats["box_areas"].append(((x2 - x1) / w_img) * ((y2 - y1) / h_img))
                    stats["boxes"] += 1

                # The pooled sources reuse filenames, and a collision here would
                # silently overwrite an image and orphan its label. Disambiguate.
                stem = Path(orig).stem
                if stem in seen:
                    stem = f"{stem}_{done}"
                seen.add(stem)

                cv2.imwrite(str(OUT / "images" / f"{stem}.jpg"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                (OUT / "labels" / f"{stem}.txt").write_text(
                    "\n".join(lines) + "\n" if lines else "")

                manifests.setdefault(split_of(split_name), []).append(f"./images/{stem}.jpg")
                stats["images"] += 1
                stats["crack_pixel_frac"].append(float((mask > 127).mean()))
                done += 1
                if done % 500 == 0:
                    print(f"  {done} images ...", flush=True)
    return manifests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--keep-source", nargs="*", default=[],
                    help=f"train on a normally-excluded source {sorted(EXCLUDE_SOURCES)}")
    args = ap.parse_args()

    parquets = sorted(SRC.glob("*.parquet"))
    if not parquets:
        raise SystemExit(f"no parquet files in {SRC} - run scripts/download_cracks.py")

    print("=" * 62)
    print("Merged crack dataset - parquet -> YOLO boxes")
    print("=" * 62)
    for p in parquets:
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.0f} MB)")
    exclude = EXCLUDE_SOURCES - set(args.keep_source)
    print(f"\nmin component area: {args.min_area} px   mask threshold: >127")
    print(f"excluded sources  : {sorted(exclude) or 'none'}\n")

    if OUT.exists():
        shutil.rmtree(OUT)

    # The upstream test parquet is the held-out split; carve val out of train
    # deterministically so re-running gives the same partition.
    rng = random.Random(SEED)

    def split_of(parquet_split):
        if parquet_split == "test":
            return "test"
        return "val" if rng.random() < VAL_FRACTION else "train"

    files = [(p, "test" if p.name.startswith("test") else "train") for p in parquets]
    stats = {
        "images": 0, "boxes": 0, "unreadable": 0, "no_boxes": 0,
        "fill_ratios": [], "box_areas": [], "crack_pixel_frac": [],
        "backgrounds": 0, "excluded": 0, "sources": Counter(),
    }
    manifests = convert(files, split_of, args.min_area, args.limit, exclude, stats)

    every = [f for v in manifests.values() for f in v]
    for name, v in [*manifests.items(), ("all", every)]:
        (OUT / f"{name}.txt").write_text("\n".join(sorted(v)) + "\n")

    print(f"\nimages converted : {stats['images']}")
    print(f"boxes written    : {stats['boxes']}"
          f"  (avg {stats['boxes'] / max(stats['images'], 1):.1f}/image)")
    print(f"  crack-free backgrounds (empty label): {stats['backgrounds']}")
    for k, label in (("unreadable", "undecodable row"),
                     ("excluded", "row from a held-out source"),
                     ("no_boxes", "mask with no component above min-area")):
        if stats[k]:
            print(f"  {label}: {stats[k]}")
    print("\nsplits:")
    for name in ("train", "val", "test"):
        print(f"  {name:<6} {len(manifests.get(name, [])):>6}")

    print("\nsource composition (top 12):")
    for src, n in stats["sources"].most_common(12):
        print(f"  {src:<16} {n:>6}")

    fills, areas = np.array(stats["fill_ratios"]), np.array(stats["box_areas"])
    crackpx = np.array(stats["crack_pixel_frac"])
    print("\nTHE MASK->BOX LIMITATION, MEASURED:")
    print(f"  crack pixels per image : {crackpx.mean() * 100:.2f}% (mean)")
    print(f"  median box area        : {np.median(areas) * 100:.2f}% of image")
    print(f"  median box fill ratio  : {np.median(fills) * 100:.1f}%")
    print(f"  boxes filled < 25%     : {(fills < 0.25).mean() * 100:.1f}%")

    (OUT / "crack_merged.yaml").write_text(
        "# Generated by scripts/prepare_crack_merged.py - do not edit by hand.\n"
        "# Pooled public crack-segmentation sets, boxes derived from masks and\n"
        "# loose by construction (see fill-ratio stats in the Phase 9 notes).\n"
        "# Unlike deepcrack_ood.yaml this one IS trained on - it is the concrete\n"
        "# specialist training set; DeepCrack stays the held-out OOD probe.\n\n"
        f"path: {OUT.as_posix()}\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "test: test.txt\n\n"
        "names:\n" + "\n".join(f"  {i}: {c}" for i, c in enumerate(DEEPCRACK_CLASSES)) + "\n"
    )
    print(f"\nOutput written to: {OUT}\nDone.\n")


if __name__ == "__main__":
    main()
