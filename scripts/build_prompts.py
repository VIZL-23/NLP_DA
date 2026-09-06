"""Phase 1 - Validate and inspect the defect description corpus.

Phase 4 will extend this to encode the corpus with the frozen CLIP text encoder
and cache the resulting embedding matrix T. For now it validates structure and
reports what the corpus contains.

Run:
    python scripts/build_prompts.py
    python scripts/build_prompts.py --protocol openvocab
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "prompts" / "defect_corpus.json"

VALID_TIERS = {"bare", "natural", "visual", "material", "alias"}


def load_corpus():
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def validate(corpus):
    """Return a list of problems found. Empty list means the corpus is clean."""
    problems = []
    seen_text = {}

    for cls, entry in corpus["classes"].items():
        if "canonical" not in entry:
            problems.append(f"{cls}: missing 'canonical'")
        if "held_out" not in entry:
            problems.append(f"{cls}: missing 'held_out'")

        descriptions = entry.get("descriptions", [])
        if not descriptions:
            problems.append(f"{cls}: no descriptions")

        tiers = Counter()
        for d in descriptions:
            text = d.get("text", "").strip()
            tier = d.get("tier", "")

            if not text:
                problems.append(f"{cls}: empty description text")
            if tier not in VALID_TIERS:
                problems.append(f"{cls}: unknown tier '{tier}' for '{text}'")
            tiers[tier] += 1

            # A phrase appearing under two classes would make the contrastive
            # target ambiguous.
            key = text.lower()
            if key in seen_text and seen_text[key] != cls:
                problems.append(
                    f"duplicate phrase '{text}' in both {seen_text[key]} and {cls}"
                )
            seen_text[key] = cls

        if tiers["bare"] != 1:
            problems.append(f"{cls}: expected exactly 1 'bare' tier, got {tiers['bare']}")

    return problems


def select(corpus, protocol):
    """Return [(class, text, tier)] visible to the given protocol.

    In the openvocab protocol the held-out classes contribute no descriptions to
    the training vocabulary - they may only be introduced at test time.
    """
    rows = []
    for cls, entry in corpus["classes"].items():
        if protocol == "openvocab" and entry.get("held_out"):
            continue
        for d in entry["descriptions"]:
            rows.append((cls, d["text"], d["tier"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--protocol", choices=["closed", "openvocab"], default="closed",
        help="closed = all classes; openvocab = held-out classes excluded",
    )
    ap.add_argument("--list", action="store_true", help="print every phrase")
    args = ap.parse_args()

    corpus = load_corpus()

    print("=" * 62)
    print(f"Defect corpus - {corpus['meta']['name']}")
    print(f"version: {corpus['meta']['version']}  status: {corpus['meta']['review_status']}")
    print("=" * 62)

    problems = validate(corpus)
    if problems:
        print(f"\nVALIDATION FAILED ({len(problems)} problems):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nvalidation: PASS")

    total = 0
    by_dataset = defaultdict(list)
    for cls, entry in corpus["classes"].items():
        by_dataset[entry.get("dataset", "unknown")].append((cls, entry))

    print("\nper class:")
    for dataset in sorted(by_dataset):
        print(f"  [{dataset}]")
        for cls, entry in by_dataset[dataset]:
            n = len(entry["descriptions"])
            total += n
            flag = "HELD OUT" if entry.get("held_out") else "seen"
            print(f"    {cls:<18} {n:>3} phrases   [{flag}]")
    print(f"  {'TOTAL':<20} {total:>3} phrases")

    tiers = Counter(d["tier"] for e in corpus["classes"].values() for d in e["descriptions"])
    print("\nper tier:")
    for tier in sorted(tiers):
        print(f"  {tier:<18} {tiers[tier]:>3}")

    rows = select(corpus, args.protocol)
    print(f"\nprotocol '{args.protocol}': {len(rows)} phrases form the training vocabulary")
    if args.protocol == "openvocab":
        held = [c for c, e in corpus["classes"].items() if e.get("held_out")]
        print(f"  excluded at training time: {', '.join(held)}")

    if args.list:
        print("\nphrases:")
        for cls, text, tier in rows:
            print(f"  [{cls:<16}] ({tier:<8}) {text}")

    print()


if __name__ == "__main__":
    main()
