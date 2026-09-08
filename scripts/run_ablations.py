"""Phase 7 - orchestrates the seven planned ablations by shelling out to
scripts/train_tgfem.py once per variant.

Of the seven, three are load-bearing (README, "Priority ablations") - run
these first, and treat the rest as time permits:

    (a) TG-FEM removed (identity check)         variant=tgfem_identity
    (d) context-token count M                   --n-ctx 0/4/8/16, --tier
    (g) gates driven by image vs by text         variant=cbam_worlddetect
        ("this one is the paper")

This script does not itself invoke any accuracy-affecting logic - it is a
thin, resumable dispatcher so a real run can be stopped and restarted per
variant without re-running everything. Each call is a separate `python`
subprocess (not an in-process import) so one variant's failure can't corrupt
another's CUDA/optimizer state.

Run (needs GPU + real CLIP weights - see language.py's module docstring for
why this needs a GPU machine to actually run):
    python scripts/run_ablations.py
    python scripts/run_ablations.py --dataset neu --protocol openvocab
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

# (variant, n_ctx, tier, label) - label is just for the printed plan.
PRIORITY = [
    ("tgfem", 8, "natural", "(g) baseline - the active model"),
    ("tgfem_identity", 8, "natural", "(a) identity check, same param budget"),
    ("cbam_worlddetect", 8, "natural", "(g) image-gated control - THE load-bearing comparison"),
    ("tgfem", 0, "bare", "(d) M=0, hand-written prompts"),
    ("tgfem", 4, "natural", "(d) M=4"),
    ("tgfem", 16, "natural", "(d) M=16"),
]

# The remaining four (b, c, e, f from the report's seven) are not wired up as
# separate cfgs yet - they are protocol/schedule variations on the existing
# cfg files (different datasets, different tiers, different seeds for
# variance) rather than new architectures, so they are just more
# --dataset/--protocol/--tier/--seed calls to this same script, not listed
# individually here.


def run_one(variant, n_ctx, tier, dataset, protocol, epochs, batch, imgsz, device, seed):
    out = RESULTS / f"phase6_{dataset}_{protocol}_{variant}" \
        f"{'' if n_ctx == 8 else f'_nctx{n_ctx}'}{'' if tier == 'natural' else f'_{tier}'}.json"
    if out.exists():
        print(f"SKIP (already have) {out.name}")
        return True

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "train_tgfem.py"),
        "--variant", variant, "--dataset", dataset, "--protocol", protocol,
        "--n-ctx", str(n_ctx), "--tier", tier,
        "--epochs", str(epochs), "--batch", str(batch), "--imgsz", str(imgsz),
        "--device", device, "--seed", str(seed),
    ]
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)  # return code checked explicitly below, not raised
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["neu", "gc10"], default="neu")
    ap.add_argument("--protocol", choices=["closed", "openvocab"], default="closed",
                     help="closed for the ablations as run in Phase 3-style tables; "
                          "openvocab to also get the zero-shot numbers")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # PRIORITY is the only plan that exists (see the module docstring: the
    # other four ablations are flag variations on the same script, not
    # separate configs) - there used to be a --priority-only flag implying
    # an opt-out into some larger plan, but no such plan was ever wired up,
    # so it was a no-op that misrepresented what this script does. Removed.
    plan = PRIORITY
    print("=" * 62)
    print(f"Phase 7 ablation plan - {args.dataset}/{args.protocol}")
    print("=" * 62)
    for variant, n_ctx, tier, label in plan:
        print(f"  {variant:<18} n_ctx={n_ctx:<3} tier={tier:<10} {label}")
    print()

    failures = []
    for variant, n_ctx, tier, label in plan:
        ok = run_one(variant, n_ctx, tier, args.dataset, args.protocol,
                     args.epochs, args.batch, args.imgsz, args.device, args.seed)
        if not ok:
            failures.append((variant, n_ctx, tier))

    print("\n" + "=" * 62)
    if failures:
        print(f"{len(failures)} run(s) FAILED:")
        for f in failures:
            print(f"  {f}")
        raise SystemExit(1)
    print("All planned runs completed (or were already present).")
    print("Next: python scripts/eval_tgfem.py --compare-only "
          f"--dataset {args.dataset} --protocol {args.protocol}")


if __name__ == "__main__":
    main()
