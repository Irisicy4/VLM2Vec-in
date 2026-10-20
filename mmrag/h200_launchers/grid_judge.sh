#!/bin/bash
# INTERMITTENT EXTERNAL JUDGE as the reward grader. Exactly mix2m-v3pure plus --judge_every,
# so the ONLY difference from the 2-seed control is which grader scores the reader's answers.
#
# The reader is unchanged (3B, 4 rollouts at T=0.7). Only the GRADER alternates: on judge steps the
# reader's generated answers go to Qwen3-30B instead of cover-EM. Switching is PER STEP, so every
# GRPO group is graded by one grader and its within-group z-scored advantage stays commensurable --
# that was the user's constraint and it is the one that matters.
#
# Motivation, measured on 1,500 held-out queries: re-grading with the 30B moves the live-gradient
# population 32.3% -> 39.6% (net +7.7pp). Of the 25.4pp of "hopeless" it recovers, only ~3/5 becomes
# usable gradient; the rest saturates, and it destroys 8.1pp of existing gradient by crediting
# no-context answers. So the expected effect is real but modest.
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
