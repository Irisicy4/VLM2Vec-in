#!/bin/bash
# The 3K-query row of tab:ab_refresh is the frozen arm at step 750 (750 steps x batch 4 = 3K queries);
# the 6K row is the same run at step 1500. frozen-idx-s1 has now given the 6K row a second seed, so this
# evaluates its step-750 checkpoint to give the 3K row one too. Without it the two rows of the same table
# would rest on different seed counts.
#
# MEM_FRAC lowered to 0.35: a vLLM engine took ~44 GiB of card 4 after the earlier launch, and 0.55 would
# let this claim more than the card has left.
set -uo pipefail
D=/raid/icy/mmrag_data
GPU=${1:-4}
export MMRAG_MEM_FRAC=0.35
NAME=frozen-idx-s1-ck750
[ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME already done"; exit 0; }
rm -f "$D/cache/corpus_$NAME.pt"
echo "$(date +%T) evaluating $NAME on gpu $GPU"
bash "$D/run_eval.sh" "$GPU" "$NAME" "runs/frozen-idx-s1/checkpoint-750" >> "$D/local_eval_$NAME.log" 2>&1 \
  || echo "$(date +%T) $NAME FAILED"
echo "$(date +%T) done"
