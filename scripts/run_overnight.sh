#!/usr/bin/env bash
# Phase 6 overnight queue - runs sequentially, in PRIORITY order.
#
# Sequential on purpose: each run needs ~4.5 GB of a 6 GB card, so two at once
# would OOM. A failing run does NOT abort the queue - it is logged and the next
# one starts, so one bad config cannot waste the whole night.
#
# Needs no network: CLIP now loads from the local weights/clip/ViT-B-32.pt
# (see src/tgfem/language.py::_find_local_openai_clip), and every dataset and
# checkpoint is already cached.

cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
mkdir -p runs results

run () {                      # run <name> <args...>
  local name="$1"; shift
  echo ""
  echo "=================================================================="
  echo ">>> START $name   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=================================================================="
  if $PY scripts/train_tgfem.py "$@" --device 0 > "runs/phase6_${name}.log" 2>&1; then
    echo ">>> OK    $name   $(date '+%H:%M:%S')"
  else
    echo ">>> FAILED $name (exit $?) - see runs/phase6_${name}.log"
    tail -n 15 "runs/phase6_${name}.log" | tr '\r' '\n' | tail -8
  fi
}

echo "Phase 6 overnight queue started $(date)"

# 1. Ablation (g) - the load-bearing comparison. CBAM gates (image-derived)
#    vs TG-FEM gates (text-derived) with the SAME WorldDetect head, so the
#    head no longer confounds the comparison (PHASE-5.md section 5).
run cbam_worlddetect_closed --variant cbam_worlddetect --dataset neu --protocol closed

# 2. The zero-shot numbers - the project's actual premise, still unmeasured.
run tgfem_openvocab        --variant tgfem            --dataset neu --protocol openvocab

# 3. Ablation (a) - identity at the same parameter budget.
run tgfem_identity_closed  --variant tgfem_identity   --dataset neu --protocol closed

# 4. Ablation (g) on the zero-shot side. Lowest priority: if the night runs
#    out here, nothing above it is lost.
run cbam_worlddetect_openvocab --variant cbam_worlddetect --dataset neu --protocol openvocab

echo ""
echo "=================================================================="
echo "QUEUE FINISHED $(date)"
echo "results/ now contains:"
ls -1 results/ | sed 's/^/  /'
