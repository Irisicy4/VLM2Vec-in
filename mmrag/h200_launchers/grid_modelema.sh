#!/bin/bash
# SCHEME 3 — model-EMA. The doc tower is encoded with EMA'd weights while queries stay live
# (train_rl.py:291-310 doc_tower()). Never cleanly run: the only prior attempt (div45kv3-emaenc,
# 72.47/60.49/32.20) trained q(live)x d(EMA) but EVALUATED both towers live — a geometry mismatch,
# not a result. The fix (saving ema_tower/) landed at train_rl.py:668 and was never exercised.
# Matched no-EMA twin CO-LOCATED on the same card so contention is identical for both arms.
set -uo pipefail
D=/raid/icy/mmrag_data
NAME=$1; GPU=$2; EXTRA=$3
COMMON="--pool built/pool_train_1M.jsonl --corpus built/corpus_mix2M.jsonl \
 --n_distractor_articles 42484 --image_dataset infoseek --max_train_rows 0 \
 --max_steps 1500 --lr_schedule constant --no_force_gold --reward judge --algo plgrpo \
 --pl_group 4 --pl_k 4 --pl_support 24 --contrastive_coef 0 --seed 0 --learning_rate 2e-5 \
 --mem_frac 0.25"
for try in 1 2 3; do
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME complete"; exit 0; }
  echo "$(date +%T) $NAME attempt $try gpu $GPU : $EXTRA"
  bash "$D/run_cell.sh" "$GPU" "$NAME" "$COMMON $EXTRA" >> "$D/local_$NAME.log" 2>&1
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME done"; exit 0; }
  echo "$(date +%T) $NAME died at $(wc -l < "$D/runs/$NAME/metrics.jsonl" 2>/dev/null||echo 0)"; sleep 120
done
