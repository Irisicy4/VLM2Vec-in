#!/bin/bash
# EMA x recipe grid on the 2M mixed-domain corpus.
#
# WHY THESE FOUR: the record has index-EMA tested on v3-pure only — at 45k it was parity-to-
# negative (div45kv3-emaidx 75.77/63.14/32.93 vs v3pure 76.83/64.57/33.00), at 810k/3000 steps
# it is ansR@5 +2.27 callable. It has NEVER been tested on the ANCHORED arm (v2), which is the
# recipe with the best retrieval on record (80.13 entR@5). v1 (GRPO-8 + anchor) is skipped: the
# ledger records v1->v2 as acc-unchanged and retrieval-up, so v2 dominates it.
#
# DATA CHANGE: --corpus built/corpus_mix2M.jsonl. n_distractor_articles is NOT 3000 —
# mix2M averages 1.278 passages/url (98.1% single-passage BEIR/OVEN/OKVQA docs) against ~16 for
# corpus_small, so 3000 would give ~4.2k distractor passages instead of ~48k. 42,484 urls gives a
# 184,660-passage index (10.9% cross-domain), matched to the 173,755 our corpus_small runs used.
# EVAL is deliberately left on corpus_small so every number stays comparable to the whole campaign;
# the diversity change is training-index-only.
set -uo pipefail
D=/raid/icy/mmrag_data
NAME=$1; GPU=$2; EXTRA=$3
COMMON="--pool built/pool_train_1M.jsonl --corpus built/corpus_mix2M.jsonl \
 --n_distractor_articles 42484 --image_dataset infoseek --max_train_rows 0 \
 --max_steps 1500 --lr_schedule constant --no_force_gold --reward judge --algo plgrpo \
 --pl_group 4 --pl_k 4 --pl_support 24 --seed 0 --learning_rate 2e-5 --mem_frac 0.25"
for try in 1 2 3; do
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME complete"; exit 0; }
  echo "$(date +%T) $NAME attempt $try on GPU $GPU : $EXTRA"
  bash "$D/run_cell.sh" "$GPU" "$NAME" "$COMMON $EXTRA" >> "$D/local_$NAME.log" 2>&1
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME done"; exit 0; }
  echo "$(date +%T) $NAME died at step $(wc -l < "$D/runs/$NAME/metrics.jsonl" 2>/dev/null||echo 0); retry"
  sleep 120
done
