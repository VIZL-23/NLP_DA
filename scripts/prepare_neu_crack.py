"""Merge NEU-DET and the merged crack set into one 7-class dataset.

Why
---
The concrete specialist is not text-guided. With `nc=1` its contrastive head was
never asked to separate one text embedding from another - "crack vs background"
is the entire discrimination the loss demanded - so it returns the same boxes
for `banana` as for `a crack in the concrete surface` (PHASE-9.md section 4).
Giving crack six steel classes to compete against is the only route to a
text-guided concrete model that does not require inventing a taxonomy.

Two ideas were rejected before this one, and the reasons are worth keeping:

  * **Severity sub-classes (hairline / medium / wide) derived from the masks.**
    Stroke width does span 3-20 px, but 56% of that variance is explained by
    which source dataset the image came from - `cracktree200` is *exactly*
    2.00 px on every mask, `Rissbilder` is 3.00 +/- 0.35. Those are annotation
    conventions (centreline vs full extent), not crack severity. Sub-classing on
    it would teach the model to identify the dataset.
  * **Provenance sub-classes.** Honest, but "which dataset is this" is not a
    query an inspector would ever type.

Expectations, stated up front so the result can be judged honestly
-----------------------------------------------------------------
  * Steel mAP will probably drop somewhat. Merging NEU+GC10 cost 1.6-4.0 points
    (PHASE-9.md section 1). Steel and concrete are visually unmistakable, unlike
    two steel datasets with near-synonymous classes, so the interference should
    be smaller - but that is the hypothesis under test, not a prediction.
  * Discrimination will be partly IMAGE-driven ("this is concrete, so crack")
    rather than genuinely semantic. That is consistent with everything Phase 7
    found. The claim being tested is narrow: that a nonsense query stops
    returning boxes.

Balance
-------
NEU has 1,434 training images against the crack set's 8,126. Used whole, 85% of
every batch would be concrete and the steel classes would be starved. The crack
train/val splits are therefore subsampled by fixed seed, **stratified by source
dataset** so the subsample keeps the same mix of annotation conventions and
surface types as the full set.

The TEST splits of both datasets are kept **whole**. Only training data changes,
so crack mAP here is directly comparable with the 0.4647 specialist number, and
NEU mAP with the 0.7205 one.

Run:
    python scripts/prepare_neu_crack.py
    python scripts/prepare_neu_crack.py --crack-train 3000
"""

import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tgfem.data import DEEPCRACK_CLASSES, NEU_CLASSES  # noqa: E402

NEU = REPO_ROOT / "datasets" / "neu-det-yolo"
CRACK = REPO_ROOT / "datasets" / "crack-merged-yolo"
OUT = REPO_ROOT / "datasets" / "neu-crack-yolo"
SEED = 42

# 7 classes: NEU's 6 keep indices 0-5 (unchanged from the NEU-only runs, so
# per-class numbers stay directly comparable), crack is 6.
COMBINED = NEU_CLASSES + DEEPCRACK_CLASSES


def source_of(stem: str) -> str:
    return stem.split("_")[0] if "_" in stem else "misc"


def stratified_subsample(rels: list[str], target: int, rng: random.Random) -> list[str]:
    """Take `target` items, keeping each source's share of the whole.

    Sampling uniformly would let the large sources swamp the small ones and
    could drop an annotation convention entirely; the point of the merged crack
    set is its diversity, so the subsample has to preserve the mix.
    """
    if target >= len(rels):
        return rels
    by_src: dict[str, list[str]] = defaultdict(list)
    for r in rels:
        by_src[source_of(Path(r).stem)].append(r)

    out: list[str] = []
    for src, items in sorted(by_src.items()):
        items = sorted(items)
        rng.shuffle(items)
        # At least one from every source, even a tiny one.
        n = max(1, round(target * len(items) / len(rels)))
        out.extend(items[:n])
    rng.shuffle(out)
    return sorted(out[:target])


def copy_split(src_root: Path, src_classes: list[str], prefix: str,
               rels: list[str], stats: dict) -> list[str]:
    """Copy images and rewrite label indices into the combined class space."""
    offset = {c: COMBINED.index(c) for c in src_classes}
    written = []
    for rel in rels:
        src_img = src_root / rel
        if not src_img.exists():
            stats["missing_image"] += 1
            continue
        stem = f"{prefix}_{src_img.stem}"
        shutil.copy2(src_img, OUT / "images" / f"{stem}{src_img.suffix}")

        src_lbl = src_root / "labels" / f"{src_img.stem}.txt"
        out_lines = []
        for row in src_lbl.read_text().strip().splitlines():
            parts = row.split()
            cls = src_classes[int(parts[0])]
            out_lines.append(" ".join([str(offset[cls])] + parts[1:]))
            stats["per_class"][cls] += 1
        if not out_lines:
            # A crack-free background image. Ultralytics represents it as an
            # empty label file - it must survive the merge, it is a hard
            # negative, not a broken row.
            stats["backgrounds"] += 1
        (OUT / "labels" / f"{stem}.txt").write_text(
            "\n".join(out_lines) + "\n" if out_lines else "")
        written.append(f"./images/{stem}{src_img.suffix}")
        stats["images"] += 1
    return written


def read_manifest(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"missing {path} - run the matching prepare_*.py first")
    return [ln.strip().lstrip("./") for ln in path.read_text().split() if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crack-train", type=int, default=2150,
                    help="crack training images to keep (default ~1.5x NEU's 1434)")
    args = ap.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)

    print("=" * 62)
    print("Merging NEU-DET + merged crack set -> 7 classes")
    print("=" * 62)

    rng = random.Random(SEED)
    stats = {"images": 0, "missing_image": 0, "backgrounds": 0, "per_class": Counter()}

    neu_splits = {s: read_manifest(NEU / f"closed_{s}.txt") for s in ("train", "val", "test")}
    crack_splits = {s: read_manifest(CRACK / f"{s}.txt") for s in ("train", "val", "test")}

    # Val is subsampled by the same ratio as train so the validation mix matches
    # what the model is being trained on; test is untouched.
    ratio = min(1.0, args.crack_train / len(crack_splits["train"]))
    crack_splits["train"] = stratified_subsample(crack_splits["train"], args.crack_train, rng)
    crack_splits["val"] = stratified_subsample(
        crack_splits["val"], round(len(crack_splits["val"]) * ratio), rng)

    counts = {}
    for subset in ("train", "val", "test"):
        neu_rel = copy_split(NEU, NEU_CLASSES, "neu", neu_splits[subset], stats)
        crk_rel = copy_split(CRACK, DEEPCRACK_CLASSES, "crk", crack_splits[subset], stats)
        merged = sorted(neu_rel + crk_rel)
        (OUT / f"neucrack_{subset}.txt").write_text("\n".join(merged) + "\n")
        counts[subset] = len(merged)
        print(f"  {subset:<6} {len(merged):>5} images  (neu {len(neu_rel)}, crack {len(crk_rel)})")

    print(f"\ncrack train subsample: {len(crack_splits['train'])} of "
          f"{len(read_manifest(CRACK / 'train.txt'))}  (seed {SEED}, stratified by source)")
    print(f"crack-free backgrounds kept: {stats['backgrounds']}")
    print(f"\ntotal images : {stats['images']}")
    if stats["missing_image"]:
        print(f"missing      : {stats['missing_image']}")
    print(f"total boxes  : {sum(stats['per_class'].values())}")
    print("\nboxes per class:")
    for c in COMBINED:
        print(f"  {c:<18} {stats['per_class'][c]:>6}")

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(COMBINED))
    (OUT / "neucrack_closed.yaml").write_text(
        "# Generated by scripts/prepare_neu_crack.py - do not edit by hand.\n"
        "# NEU-DET (indices 0-5) + crack (6). NEU keeps its original indices so\n"
        "# per-class results stay directly comparable with the NEU-only runs.\n"
        "# The crack training images are a seeded, source-stratified subsample;\n"
        "# both TEST splits are whole, so mAP here is comparable with the\n"
        "# specialists' numbers.\n\n"
        f"path: {OUT.as_posix()}\n"
        "train: neucrack_train.txt\n"
        "val: neucrack_val.txt\n"
        "test: neucrack_test.txt\n\n"
        f"names:\n{names}\n"
    )
    print(f"\nOutput: {OUT}\nDone.\n")


if __name__ == "__main__":
    main()
