#!/bin/bash
# FULL-CORPUS RL index + random partial re-embed.
# --n_distractor_articles 21143 takes every non-gold article in corpus_small, so the RL index IS the
# full 422,378-passage corpus — identical to the EVAL index, which removes the train/eval index gap
# that every previous cell here carried (41% or 9.2% overlap).
# --refresh_frac re-encodes a fresh UNIFORM RANDOM sample each period. Note the property that makes
# this a real test rather than a re-run: random selection bounds MEAN age but not MAX age, so at 10%
# every 100 steps ~20.6% of rows are never refreshed at all; the stalest-first sweep in --index_ema
# bounds max age by construction. If the mediator is max age, random should underperform stalest at
# matched coverage.
set -uo pipefail
D=/raid/icy/mmrag_data
NAME=$1; GPU=$2; EXTRA=$3
COMMON="--pool built/pool_train_1M.jsonl --corpus built/corpus_small.jsonl \
 --n_distractor_articles 21143 --image_dataset infoseek --max_train_rows 0 \
 --max_steps 1500 --lr_schedule constant --no_force_gold --reward judge --algo plgrpo \
 --pl_group 4 --pl_k 4 --pl_support 24 --contrastive_coef 0 --seed 0 --learning_rate 2e-5 \
 --mem_frac 0.25"
for try in 1 2 3; do
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME complete"; exit 0; }
  echo "$(date +%T) $NAME attempt $try gpu $GPU : $EXTRA"
  bash "$D/run_cell.sh" "$GPU" "$NAME" "$COMMON $EXTRA" >> "$D/local_$NAME.log" 2>&1
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME done"; exit 0; }
  echo "$(date +%T) $NAME stopped at $(wc -l < "$D/runs/$NAME/metrics.jsonl" 2>/dev/null||echo 0)"; sleep 120
done
