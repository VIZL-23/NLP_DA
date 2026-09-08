# Phase 7 — Evaluation & Ablations

**Status:** Scripts complete and smoke-tested. Numbers blocked on Phase 6 (no GPU here).
**Goal:** mAP, zero-shot, FPS + the 7 ablations.

---

## 1. What we set out to do

| Task | Result |
|---|---|
| A way to run the seven planned ablations, three of them flagged priority | Done — `scripts/run_ablations.py` |
| Negative control re-check (Phase 3's `eval_yoloworld.py` bug, re-verified against a trained TG-FEM checkpoint rather than assumed fixed) | Done — `scripts/eval_tgfem.py` |
| A compact tgfem vs tgfem_identity vs cbam_worlddetect comparison table, including the falsifiable per-class prediction from `PHASE-3.md` §3 | Done |
| Zero-shot per-class AP on held-out classes | Comes for free from Phase 6's openvocab-protocol runs — no separate script needed, see §2 |
| Actual ablation results | **Blocked — depends on Phase 6 runs, which need a GPU** |

---

## 2. Zero-shot evaluation doesn't need a separate script

The openvocab data YAMLs (`neu_openvocab.yaml`, etc.) declare **all**
classes in `names` — held-out ones simply have zero training images (Phase
1 design, and the README's "Locked decisions" section). So a checkpoint
trained with `--protocol openvocab` already has WorldDetect comparing
against the full class vocabulary, including the held-out ones, at
inference time — the model was just never shown a training image containing
one. Running `scripts/train_tgfem.py --protocol openvocab`'s test-split
evaluation (test pool includes every held-out-class instance, by
construction — Phase 1 leakage guard) *is* the zero-shot evaluation. Its
`per_class_AP50` output already separates seen from held-out classes by
name. A dedicated "zero-shot eval script" would just be re-deriving numbers
Phase 6 already produces; `eval_tgfem.py` instead covers what that run does
**not**: a negative control and a cross-variant comparison (§3, §4).

---

## 3. Negative control, re-verified against a real checkpoint

Phase 3's `eval_yoloworld.py` caught a real bug: `WorldValidator.__call__`
silently overwrites whatever prompts `set_classes()` was given with the
dataset YAML's own class names, so every prompt style was secretly
evaluating the same vocabulary. That bug was specific to `WorldValidator`;
this project's Phase 6 pipeline uses a plain `DetectionValidator` (Phase 5,
`train_tgfem.py`), which has no `set_classes` override at all — but "the
class most likely to reintroduce this exact failure mode" is different
here: TG-FEM's context tokens are **learned jointly with detection**, so a
broken or short-circuited text path could still fit the seen classes during
training and only reveal itself at held-out-class query time.

`scripts/eval_tgfem.py`'s `negative_control()` loads a **trained** checkpoint,
strips its embedded `TextConditioner` hook, and re-attaches a fresh one with
absurd prompts (`"banana"`, `"elephant"`, ...) — no retraining. If real and
nonsense prompts score similarly, the classification head isn't actually
being driven by text. Smoke-tested against a 1-epoch toy checkpoint (both
scored 0.0, as expected at that scale — the check is written to require
`real mAP@0.5 > 0.01` before it treats a close absurd/real gap as a failure,
so it doesn't false-positive on an undertrained model with genuinely near-
zero accuracy).

---

## 4. Ablation plan (`scripts/run_ablations.py`)

Of the seven planned, three are load-bearing (README) — the script runs
these first:

| # | What | How |
|---|---|---|
| **(a)** | TG-FEM removed (identity check) | `variant=tgfem_identity`, same param budget as active |
| **(d)** | context-token count M | `--n-ctx 0/4/8/16`, `--tier` for the M=0 hand-written-prompt arm |
| **(g)** | gates driven by image vs text — **"this one is the paper"** | `variant=cbam_worlddetect` vs `variant=tgfem`, same WorldDetect head both times |

The other four (dataset/protocol/seed variations rather than new
architectures) are `--dataset`/`--protocol`/`--seed` flags on the same
`train_tgfem.py`, not separate configs — not enumerated individually since
they're combinations of flags already exposed, not new code.

Each variant is a separate `python` subprocess (not an in-process loop), so
one failing/OOMing run can't corrupt another's state, and already-completed
runs (`results/phase6_*.json` present) are skipped — the script is
resumable.

---

## 5. Comparison table (`eval_tgfem.py --compare-only`)

Reads every `results/phase6_{dataset}_{protocol}_*.json`, tabulates mAP@0.5
per variant, and computes:

- **ablation (g):** `tgfem.mAP50 - cbam_worlddetect.mAP50`, plus **per-class**
  delta with the two texture-confusable, NEU-DET-held-out classes
  (`crazing`, `rolled-in_scale`) flagged explicitly — this is Phase 3 §3's
  falsifiable prediction: if text conditioning works, TG-FEM's gain over
  CBAM should concentrate on exactly these two classes, the same ones
  attention already helped most versus the from-scratch anchor.
- **ablation (a):** `tgfem.mAP50 - tgfem_identity.mAP50`.

No results exist yet (Phase 6 blocked), so this has only been smoke-tested
on a single toy variant's output, not a real multi-variant table — the
table-building logic itself is straightforward (read JSON, subtract), the
risk was in the upstream scripts producing consistent, correctly-keyed
JSON, which is what the smoke tests actually exercised.

---

## 6. Files created

```
scripts/eval_tgfem.py       negative control + comparison table
scripts/run_ablations.py    resumable dispatcher over train_tgfem.py
```

---

## 7. Gate check

| Criterion | Status |
|---|---|
| Ablation plan covers all 3 priority ablations (a, d, g) | Pass |
| Zero-shot evaluation path identified (no redundant script needed) | Pass |
| Negative control re-implemented against a real checkpoint, not assumed | Pass, smoke-tested |
| Comparison table includes the falsifiable per-class prediction | Pass |
| Dispatcher is resumable (skips completed variants) | Pass |
| Real ablation numbers produced | **Blocked — needs Phase 6's runs, which need a GPU** |

**Phase 7 gate: PASSED structurally, blocked on Phase 6's compute.**

---

## 8. Carried forward

1. Once Phase 6 has run the three priority variants:
   `python scripts/eval_tgfem.py --checkpoint runs/phase6_neu_closed_tgfem/weights/best.pt --dataset neu --protocol closed --device 0`
   then `python scripts/eval_tgfem.py --compare-only --dataset neu --protocol closed`.
2. FPS/speed: `train_tgfem.py`'s validator already reports `metrics.speed`
   the same way `train_baseline.py` does (Phase 3 §5) — not separately
   surfaced in `eval_tgfem.py`'s comparison table yet; add a column if the
   report needs it tabulated alongside mAP rather than read from each raw
   JSON.
3. Report per-class zero-shot AP for held-out classes specifically (Phase 3
   §2's finding: they're the hardest classes for a fully-supervised model
   too, so a modest zero-shot number isn't automatically a method failure) —
   already in `per_class_AP50`, just needs pulling out into the write-up
   once real numbers exist.
