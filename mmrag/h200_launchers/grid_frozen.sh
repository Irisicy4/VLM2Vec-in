#!/bin/bash
# FROZEN-INDEX CONTROL of the winning recipe: --refresh_steps 0, everything else identical to
# mix2m-v3pure. paper.md line 71 records this as never measured and as "the control that would price
# the dynamism itself". The text side's 2x2 now makes it urgent: on their carve-out a trained query
# tower reading a FROZEN base-encoded index (0.766) beats both the untrained base (0.660) and a
# self-consistent LIVE index (0.626), so refreshing with the trained encoder may be harmful.
# Also the cheapest cell we can run: skips 15 full re-encodes, ~3.4 h of the wall clock.
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
