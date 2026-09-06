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
