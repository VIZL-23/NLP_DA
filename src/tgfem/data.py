"""Canonical dataset constants.

Single source of truth for class taxonomies, imported by both the conversion
scripts and the environment audit so the two can never drift apart.
"""

# --------------------------------------------------------------------------
# NEU-DET
# --------------------------------------------------------------------------

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Held out from training in the NEU-DET open-vocabulary protocol.
NEU_HELD_OUT = {"crazing", "rolled-in_scale"}


# --------------------------------------------------------------------------
# GC10-DET
# --------------------------------------------------------------------------

# GC10-DET labels its objects with Chinese pinyin codes, NOT the English folder
# names. This mapping was derived empirically by cross-referencing every XML
# against the class folder its image lives in.
#
# Two data-quality fixes are baked in:
#   * "10_yaozhe" and "10_yaozhed" are the same class (typo variant) -> merged.
#   * a corrupt class literally named "d" (1 object) is absent, so it is
#     dropped during conversion.
#
# CRITICAL: the image's folder is NOT its label. GC10-DET images are
# multi-label and the folder records only the dominant defect. The XML is
# ground truth.
GC10_NAME_MAP = {
    "1_chongkong": "punching_hole",
    "2_hanfeng": "welding_line",
    "3_yueyawan": "crescent_gap",
    "4_shuiban": "water_spot",
    "5_youban": "oil_spot",
    "6_siban": "silk_spot",
    "7_yiwu": "gc10_inclusion",  # namespaced - see note below
    "8_yahen": "rolled_pit",
    "9_zhehen": "crease",
    "10_yaozhe": "waist_folding",
    "10_yaozhed": "waist_folding",
}

# NAMESPACING NOTE
# ----------------
# "inclusion" exists in BOTH NEU-DET and GC10-DET. They are the same concept
# but different imaging regimes (200x200 grayscale close-up vs 2048x1000 steel
# sheet), so they are kept as distinct label identities rather than merged.
#
# GC10's is namespaced here as `gc10_inclusion`. NEU-DET's stays plain
# `inclusion` inside its own label space, which is unambiguous because each
# dataset has its own YAML and its own class indices. It must be renamed to
# `neu_inclusion` if the two are ever merged into one training run.

GC10_CLASSES = sorted(set(GC10_NAME_MAP.values()))

# Held out from training in the GC10-DET open-vocabulary protocol.
#
# Chosen by exhaustively evaluating all 120 possible triples on worst-case
# seen-class retention (GC10 images are multi-label, so holding out a class
# also removes every image it appears in).
#
#   * These three co-occur heavily with EACH OTHER (punching_hole+welding_line
#     in 222 images, crescent_gap+welding_line in 150), so holding them out as
#     a block lets that cluster leave together. Splitting it strands
#     welding_line, which then loses 70% of its training images.
#   * `gc10_inclusion` is deliberately NOT held out despite scoring well:
#     its twin is in NEU-DET's training set, so querying it with near-identical
#     wording would not be a genuine zero-shot test.
#
# Result: 1572 train images (69%), every seen class retaining 92-100%
# except rolled_pit at 73% (inherently rare, 44 images total).
GC10_HELD_OUT = {"crescent_gap", "punching_hole", "welding_line"}


# --------------------------------------------------------------------------
# DeepCrack
# --------------------------------------------------------------------------

# Single-class, and namespaced for the same reason as above: "crack" is a
# generic term that would collide with any concrete-crack vocabulary added
# later.
DEEPCRACK_CLASSES = ["crack"]


# Concrete structural defects (Roboflow "Concrete defect detection", CC BY 4.0).
#
# ORDER IS LOAD-BEARING: it is the class order in that dataset's own data.yaml
# (['Exposed_reinforcement', 'Ruststrain', 'Scaling', 'Spalling', 'crack',
# 'efflorescence']), so the label files can be used verbatim with no index
# rewriting. Only the NAMES change, into the snake_case English the corpus
# uses. Reordering this list silently mislabels every box in the dataset.
#
# `crack` is reused rather than namespaced: this dataset's cracks ARE concrete
# cracks, so the existing corpus wording applies unchanged.
CONCRETE_CLASSES = [
    "exposed_reinforcement",
    "rust_stain",
    "scaling",
    "spalling",
    "crack",
    "efflorescence",
]


# --------------------------------------------------------------------------
# Phase 4/6 - training-vocabulary helper
# --------------------------------------------------------------------------
# Shared by scripts/train_tgfem.py and scripts/eval_tgfem.py so the two never
# pick different phrases for the same class - see build_prompts.py for corpus
# *validation* (this only does selection, deliberately not re-implementing
# that logic).

import json
from pathlib import Path

CORPUS_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "defect_corpus.json"


def load_corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def class_texts_for(classes: list[str], tier: str = "natural", corpus: dict | None = None) -> list[str]:
    """One phrase per class, in the given (class-index) order - order MUST
    match the dataset YAML's `names`, since WorldDetect/TG-FEM match text to
    ground truth by index, never by string (Phase 3 finding, eval_yoloworld.py).
    Falls back to the class's `bare` phrase if the requested tier is absent."""
    corpus = corpus or load_corpus()
    texts = []
    for c in classes:
        descs = corpus["classes"][c]["descriptions"]
        text = next((d["text"] for d in descs if d["tier"] == tier), None)
        if text is None:
            text = next(d["text"] for d in descs if d["tier"] == "bare")
        texts.append(text)
    return texts


def class_phrase_pools(
    classes: list[str],
    corpus: dict | None = None,
    tiers: list[str] | None = None,
) -> list[list[str]]:
    """ALL phrases per class, in class-index order - the pool for phrase augmentation.

    `class_texts_for` fixes ONE phrase per class for a whole run, which is what
    every run up to Phase 6 used. Probing that model (scripts/probe_wording.py)
    showed the text interface had keyed on those exact strings: the trained
    phrase scored 0.639, a one-word change ("steel" -> "metal") 0.561, and any
    real rephrasing 0.000. A model shown one phrasing per class for 150 epochs
    is never pressured to generalise across wording, so it doesn't - which also
    explains why it cannot handle an unseen class's phrasing.

    Returning the pool lets the training loop resample per batch instead, so
    each class is seen through all ~10 of its corpus phrasings.

    Args:
        classes: class names, in the dataset YAML's index order.
        tiers: restrict to these tiers (default: all).
    """
    corpus = corpus or load_corpus()
    pools = []
    for c in classes:
        descs = corpus["classes"][c]["descriptions"]
        texts = [d["text"] for d in descs if tiers is None or d["tier"] in tiers]
        if not texts:
            raise ValueError(f"class {c!r} has no phrases for tiers={tiers}")
        pools.append(texts)
    return pools
