#!/bin/bash
# Larger initial index + partial write-back.
# "the small pool selected initially" = build_rl_corpus's distractor-article sample. Raising it
# 3,000 -> 10,000 on corpus_small takes the RL index 173,755 -> 268,718 passages, i.e. 63.6% of the
# 422,378-passage EVAL corpus instead of 41.1% — so it also shrinks the train/eval index gap.
# Partial write-back is fixed at 112 rows/step (pool + boundary), so coverage falls 0.97x -> 0.63x:
# a bigger index at the SAME per-step budget is a strictly staler index, which is the text side's
# failing direction. Control: identical index size with the FULL periodic refresh, so index size and
# refresh scheme are not confounded.
set -uo pipefail
D=/raid/icy/mmrag_data
NAME=$1; GPU=$2; EXTRA=$3
COMMON="--pool built/pool_train_1M.jsonl --corpus built/corpus_small.jsonl \
 --n_distractor_articles 10000 --image_dataset infoseek --max_train_rows 0 \
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
