#!/bin/bash
# Diverse-QUERY pool x diverse-DOCUMENT index, +/- index-EMA.
# pool_div45k_v3 spans InfoSeek/EVQA/OVEN/OKVQA image sources; with the Infoseek + EVQA archives
# staged, 42,194 of 45,000 rows are servable (93.8%). train_rl.py silently drops unservable rows,
# so that 6.2% gap (the OVEN/OKVQA slice) is stated here rather than hidden.
# Index: corpus_mix2M, --n_distractor_articles 45848 -> ~184,660 passages, matched to the other
# four grid cells so index SIZE is held constant while its composition varies.
# Eval stays InfoSeek/corpus_small, which is what div45kv3-v3pure (76.83/64.57/33.00) used, so this
# is directly comparable to that row.
set -uo pipefail
D=/raid/icy/mmrag_data
NAME=$1; GPU=$2; EXTRA=$3
COMMON="--pool built/pool_div45k_v3.jsonl --corpus built/corpus_mix2M.jsonl \
 --n_distractor_articles 45848 --image_dataset div --max_train_rows 0 \
 --max_steps 1500 --lr_schedule constant --no_force_gold --reward judge --algo plgrpo \
 --pl_group 4 --pl_k 4 --pl_support 24 --contrastive_coef 0 --seed 0 --learning_rate 2e-5 \
 --mem_frac 0.25"
for try in 1 2 3; do
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME complete"; exit 0; }
  echo "$(date +%T) $NAME attempt $try gpu $GPU : $EXTRA"
  bash "$D/run_cell.sh" "$GPU" "$NAME" "$COMMON $EXTRA" >> "$D/local_$NAME.log" 2>&1
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME done"; exit 0; }
  echo "$(date +%T) $NAME died at $(wc -l < "$D/runs/$NAME/metrics.jsonl" 2>/dev/null||echo 0); retry"
  sleep 120
done
