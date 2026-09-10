"""Bundle demo sample images for the local app.

Selection rules, which matter for how defensible the demo is:

  * Images come from the **held-out TEST split** - never trained on. Using
    training images would show memorised performance.
  * Chosen by **fixed random seed**, NOT by how well the model scores on them.
    Cherry-picking the best cases is the obvious failure mode here; "random
    sample, fixed seed, from data the model never saw" is a much better answer
    when someone asks how these were chosen.

Run:
    python scripts/make_samples.py                 # NEU-DET (default checkpoint)
    python scripts/make_samples.py --dataset gc10
    python scripts/make_samples.py --dataset crack
"""

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

DATASETS = {
    "neu": (REPO_ROOT / "datasets" / "neu-det-yolo", "neu_closed.yaml", "closed_test.txt"),
    "gc10": (REPO_ROOT / "datasets" / "gc10-det-yolo", "gc10_closed.yaml", "closed_test.txt"),
    "combined": (REPO_ROOT / "datasets" / "combined-yolo", "combined_closed.yaml", "combined_test.txt"),
    "crack": (REPO_ROOT / "datasets" / "crack-merged-yolo", "crack_merged.yaml", "test.txt"),
    "neu_crack": (REPO_ROOT / "datasets" / "neu-crack-yolo", "neucrack_closed.yaml", "neucrack_test.txt"),
}
# One folder per model. The app serves each model its own samples, so a steel
# checkpoint is never demoed on a concrete image (and vice versa) by accident.
SAMPLES_ROOT = REPO_ROOT / "app" / "samples"
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="neu")
    ap.add_argument("--per-class", type=int, default=2)
    args = ap.parse_args()

    root, yaml_name, test_manifest = DATASETS[args.dataset]
    out = SAMPLES_ROOT / args.dataset
    if not (root / test_manifest).exists():
        raise SystemExit(f"missing {root / test_manifest} - run the matching prepare_*.py first")

    names = yaml.safe_load((root / yaml_name).read_text())["names"]
    classes = [names[i] for i in sorted(names)]

    stems = [ln.strip().lstrip("./") for ln in (root / test_manifest).read_text().split() if ln.strip()]

    # Group test images by the classes their labels actually contain, so the
    # bundle spans the taxonomy rather than over-representing common classes.
    by_class = defaultdict(list)
    for rel in stems:
        stem = Path(rel).stem
        lbl = root / "labels" / f"{stem}.txt"
        if not lbl.exists():
            continue
        present = {int(r.split()[0]) for r in lbl.read_text().strip().splitlines() if r.strip()}
        for c in present:
            by_class[c].append(rel)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rng = random.Random(SEED)
    picked: set[str] = set()
    print(f"dataset: {args.dataset}   test pool: {len(stems)} images\n")
    for idx, cls in enumerate(classes):
        pool = sorted(by_class.get(idx, []))
        if not pool:
            print(f"  {cls:<18} no test images")
            continue
        rng.shuffle(pool)
        chosen = [p for p in pool if p not in picked][: args.per_class]
        for rel in chosen:
            picked.add(rel)
            src = root / rel
            shutil.copy2(src, out / f"{cls}__{src.name}")
        print(f"  {cls:<18} {len(chosen)} sample(s)")

    print(f"\n{len(picked)} images -> {out}")


if __name__ == "__main__":
    main()
