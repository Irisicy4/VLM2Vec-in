#!/bin/bash
# Q4 no-context utility filter on the PL path (the winning recipe).
#   bash grid_q4gate.sh <NAME> <GPU> "<extra args>"
# Matched control is mix2m-v3pure / mix2m-v3pure-s1 at ck750: identical COMMON, ONE flag apart.
# The flag was inert on plgrpo until 2026-09-12 — the gate lived only in the non-plgrpo reward
# branch — so this is the first test of the Q4 lever on the recipe that actually wins.
set -uo pipefail
D=/raid/icy/mmrag_data
NAME=${1:?name}; GPU=${2:?gpu}; EXTRA=${3:-}
COMMON="--pool built/pool_train_1M.jsonl --corpus built/corpus_mix2M.jsonl \
 --n_distractor_articles 42484 --image_dataset infoseek --max_train_rows 0 \
 --max_steps 1500 --lr_schedule constant --no_force_gold --reward judge --algo plgrpo \
 --pl_group 4 --pl_k 4 --pl_support 24 --learning_rate 2e-5 --mem_frac 0.25"
for try in 1 2 3; do
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME complete"; exit 0; }
  echo "$(date +%T) $NAME attempt $try on GPU $GPU : $EXTRA"
  bash "$D/run_cell.sh" "$GPU" "$NAME" "$COMMON $EXTRA" >> "$D/local_$NAME.log" 2>&1
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME done"; exit 0; }
  echo "$(date +%T) $NAME died at step $(wc -l < "$D/runs/$NAME/metrics.jsonl" 2>/dev/null || echo 0); retry"
  sleep 120
done
