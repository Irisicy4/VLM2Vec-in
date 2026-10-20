#!/bin/bash
# LOCAL CONTROL ARM for the erosion finding: identical recipe MINUS the two EMA flags.
#
# WHY: at step ~460 emaidx-1M-s3000 shows group_std falling and degenerate-group fraction rising,
# both CIs excluding zero. That CANNOT be attributed to index-EMA without a matched arm — it may
# simply be what v3-pure does on the 810k pool. The origin cluster's runs/*/metrics.jsonl is not
# in the published bundle, so the comparison is impossible here; this reproduces the comparator
# locally, on the SAME hardware, which also removes the cross-site fp/hardware caveat.
# Args are emaidx's minus --index_ema / --index_refresh_extra. Distinct name: never overwrites
# the published v3-rows1M-s3000 rows.
set -uo pipefail
D=/raid/icy/mmrag_data
NAME=v3pure-1M-s3000-loc
GPU=${GPU:-3}
ARGS="--pool built/pool_train_1M.jsonl --image_dataset infoseek --max_train_rows 0 \
 --max_steps 3000 --lr_schedule constant --no_force_gold --reward judge --algo plgrpo \
 --pl_group 4 --pl_k 4 --pl_support 24 --contrastive_coef 0 --seed 0 --learning_rate 2e-5 \
 --mem_frac 0.25"
for try in $(seq 1 20); do
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) CTRL complete"; exit 0; }
  echo "$(date +%T) CTRL attempt $try on GPU $GPU"
  bash "$D/run_cell.sh" "$GPU" "$NAME" "$ARGS" >> "$D/local_$NAME.log" 2>&1
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) CTRL finished"; exit 0; }
  echo "$(date +%T) CTRL died at ~step $(wc -l < "$D/runs/$NAME/metrics.jsonl" 2>/dev/null || echo 0) — retry in 120s"
  sleep 120
done
