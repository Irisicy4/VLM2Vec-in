#!/bin/bash
# Q4 OFFLINE POOL FILTER vs SIZE-MATCHED RANDOM CONTROL. One flag apart: --pool.
# This is the experiment Q4 asked for. The online mask (q4gate-*) is a different intervention.
# Both pools are 6,789 rows drawn from the same 20,000 scored candidates, so composition is the
# only variable -- pool SIZE is held fixed, which a comparison against the 810k pool would not do.
set -uo pipefail
D=/raid/icy/mmrag_data
NAME=${1:?name}; GPU=${2:?gpu}; EXTRA=${3:-}
COMMON="--corpus built/corpus_mix2M.jsonl \
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
