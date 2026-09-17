"""Generate docs/query-phrases.md from the corpus and the shipped models.

Every phrase listed is one the named model was trained on. Re-run this after
changing prompts/defect_corpus.json or the models in app/server.py:

    python scripts/make_phrase_list.py
"""
import sys, json
sys.path.insert(0, 'src')
from tgfem.data import NEU_CLASSES, GC10_CLASSES, CONCRETE_CLASSES

C = json.load(open('prompts/defect_corpus.json', encoding='utf-8'))['classes']
R = lambda f: json.load(open(f'results/{f}.json'))['per_class_AP50']

# Added by the synonym run (PHASE-9 s8) - only neu_crack was retrained with them.
SYN_CRACK = {"a split in the concrete surface", "a fissure running through the concrete",
             "a fracture line in the concrete"}

MODELS = [
    ("Steel + concrete", "neu_crack", NEU_CLASSES + ["crack"],
     R('phase6_neu_crack_closed_tgfem_phraseaug_syn'), set(), "0.25"),
    ("Steel · GC10-DET", "gc10", GC10_CLASSES,
     R('phase6_gc10_closed_tgfem_phraseaug'), set(), "0.25"),
    ("Concrete · structural", "concrete", CONCRETE_CLASSES,
     R('phase6_concrete_closed_tgfem_phraseaug'), SYN_CRACK, "0.10"),
]
TIER_ORDER = {"natural": 0, "bare": 1, "visual": 2, "material": 3, "alias": 4}

def title(c):
    return {"gc10_inclusion": "Inclusion (GC10)", "rolled-in_scale": "Rolled-in scale",
            "pitted_surface": "Pitted surface"}.get(c, c.replace("_", " ").capitalize())

out, total = [], 0
for label, key, classes, ap, skip, conf in MODELS:
    out.append(f"## Model: {label}\n")
    out.append(f"Select **{label}** in the app. Default confidence **{conf}**.\n")
    if key == "concrete":
        out.append("AP is low across this model because its boxes are loose field "
                   "photography, not because it cannot find defects: at the 0.10 "
                   "default it hit 80% of test images, including efflorescence 6/6 "
                   "and spalling 6/6. Lead with **spalling**.\n")
    for c in classes:
        phrases = [d for d in C[c]['descriptions'] if d['text'] not in skip]
        phrases.sort(key=lambda d: TIER_ORDER[d['tier']])
        a = ap[c]
        note = ""
        if a < 0.10:
            note = " — **does not work, avoid in a demo**"
        elif a < 0.40:
            note = " — weak class, expect misses"
        out.append(f"### {title(c)}\n")
        out.append(f"`{key}` · class `{c}` · AP@0.5 {a:.2f}{note}\n")
        for i, d in enumerate(phrases, 1):
            out.append(f"{i}. {d['text']}")
            total += 1
        out.append("")
    out.append("---\n")

head = f"""# Query phrases

Every phrase below is one the named model was **trained on**, so each should
return a box on an image that actually shows that defect. {total} phrases across
{sum(len(m[2]) for m in MODELS)} class entries in the three shipped models.

## How to use this

- **Pick the model first.** A phrase only works with the model listed above it.
  Steel phrases do nothing on the concrete model, and the other way round.
- **Paste one phrase per line** into the description box. Several lines at once
  is fine — each gets its own colour.
- **Phrase 1 in each list is the one the model is scored on.** Start with it.
- **Keep the defect noun.** The model is keyed on the noun ("crack",
  "scratches", "spalling"). You can reword everything around it —
  `scratches on the metal surface` still works — but swap the noun for a
  synonym that is not in this list (`split`, `rupture`, `cleft`) and it usually
  returns nothing. See `phase-notes/PHASE-9.md` sections 7–8.
- **Nothing returned is not always a failure.** If the image does not show that
  defect, an empty result is the correct answer. Try the confidence slider
  before assuming the model is broken.

`crack` appears twice because it belongs to two models. The concrete-model list
omits three synonyms that were added after that model was trained.

Generated from `prompts/defect_corpus.json` by `python scripts/make_phrase_list.py`. If the corpus changes, regenerate
this file rather than editing it by hand.

---
"""
open('docs/query-phrases.md', 'w', encoding='utf-8').write(head + "\n".join(out) + "\n")
print("phrases:", total)
