#!/usr/bin/env bash
# Train the concrete/pavement specialist on the ~9.7k merged crack set.
#
# Waits for the 255-image DeepCrack run to release the GPU first - the two must
# not overlap on a 6 GB card. That small run is kept because it is the honest
# small-data comparison point for this one.
#
# 60 epochs, not the 150 used elsewhere: this dataset is ~38x larger than
# DeepCrack, so the same number of gradient steps arrives far sooner, and 150
# passes over 8.6k images does not fit in a night on a 4050.
#
# Run:
#     bash scripts/run_crack_specialist.sh
set -u

cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=runs/phase7_crack_merged.log

echo "waiting for the DeepCrack run to finish ..."
while [ ! -f results/phase6_deepcrack_closed_tgfem_phraseaug.json ]; do
  # A crashed run would leave us waiting forever, so notice when no python is
  # training any more and stop pretending to queue behind it.
  if ! tasklist 2>/dev/null | grep -qi python; then
    echo "no python process left and no DeepCrack result - it failed; check runs/phase7_deepcrack_phraseaug.log"
    exit 1
  fi
  sleep 60
done
echo "DeepCrack done $(date)"

echo "=== crack specialist START $(date) ===" | tee "$LOG"
"$PY" scripts/train_tgfem.py \
  --variant tgfem --dataset crack --protocol closed \
  --phrase-aug --epochs 60 --batch 16 >>"$LOG" 2>&1
status=$?
echo "=== crack specialist EXIT $status $(date) ===" | tee -a "$LOG"
exit $status
