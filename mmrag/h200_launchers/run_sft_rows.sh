#!/bin/bash
# MATCHED SFT DATA-SCALING ARM, the comparator the v3-pure rows sweep never had.
#
# The RL side already has a training-query-volume curve in the latest recipe (v3-rows25k / 200k / 1M,
# 1/3/2 seeds) and it is FLAT: acc 42.80 -> 43.07 +- 0.41 -> 43.30 +- 0.05, every step inside the 1.83
# floor. That result is only interesting next to a supervised curve on the same pool sizes, and the SFT
# points on disk (sft-200k-short, sft-810k-short, sft-big-1M, sft-div45k) use different step budgets and
# were never matched to it.
#
# WHAT "MATCHED" MEANS HERE, stated because it is not step-matching. Both train_rl.py and train_sft.py
# take --max_train_rows with identical semantics (random.Random(seed).sample of the pool), so the two
# curves see the SAME subset of the SAME pool at each x. Each objective then keeps its own tuned
# hyperparameters -- SFT bs 32 / lr 1e-4 / 1200 steps, RL bs 4 / lr 2e-5 / 1500 steps -- which is the
# convention tab:mm_main already uses. Queries CONSUMED therefore differ (SFT 1200x32 = 38.4k, RL
# 1500x4 = 6k); the x axis is pool size available, not tokens seen. Do not relabel it as compute.
#
# sft-local-s0 is already the full-pool point (max_train_rows 0 -> all 810,276 rows), same
# hyperparameters, so only 25k and 200k are missing.
set -uo pipefail
D=/raid/icy/mmrag_data
NAME=${1:?name}; GPU=${2:?gpu}; ROWS=${3:?rows}
if [ ! -f "$D/runs/$NAME/adapter_model.safetensors" ]; then
  echo "$(date +%T) training $NAME (rows=$ROWS) on gpu $GPU"
  bash "$D/grid_sft.sh" "$NAME" "$GPU" "--max_train_rows $ROWS --seed 0" >> "$D/local_$NAME.log" 2>&1 \
    || { echo "$(date +%T) $NAME TRAIN FAILED"; exit 1; }
fi
echo "$(date +%T) evaluating $NAME on gpu $GPU"
MMRAG_MEM_FRAC=0.40 bash "$D/run_eval.sh" "$GPU" "$NAME" "runs/$NAME" >> "$D/local_eval_$NAME.log" 2>&1 \
  || { echo "$(date +%T) $NAME EVAL FAILED"; exit 1; }
# the wrapper prints EVAL_DONE unconditionally; the real check is the vqa file
[ -f "$D/results/$NAME.vqa_top5.json" ] || { echo "$(date +%T) $NAME vqa MISSING"; exit 1; }
MMRAG_DATA=$D $D/venv/bin/python3 /raid/icy/VLM2Vec-in/mmrag/rescore.py --arms "$NAME" --write >/dev/null 2>&1
echo "$(date +%T) $NAME complete"
