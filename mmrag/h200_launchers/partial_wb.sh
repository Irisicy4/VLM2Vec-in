#!/bin/bash
# Reproduce the TEXT-SIDE FAILING REGIME on multimodal data, to test whether their turnover is
# about the refresh OPERATION (partial-and-accumulating vs full-and-periodic) rather than the model
# or modality. Their dmixDynCov: 1.06x coverage, blend 0.5, doc tower training -> -10.4/-11.2 R@5
# at 4k/5k queries, both seeds.
# Here: --index_ema 0.5 (retention 0.5 == their blend 0.5) and --index_refresh_extra 0, giving
# 112 rows/step = 0.91x coverage. Everything else identical to mix2m-v3pure, whose FULL periodic
# refresh gives 76.23 entR@5 at 6k queries and is the control.
# Prediction registered before running: if inconsistency is the mediator, THIS arm turns over
# (>2.40 entR@5 below its own ck750) while mix2m-v3pure did not (-2.20, inside the floor).
set -uo pipefail
D=/raid/icy/mmrag_data
NAME=mix2m-partialwb; GPU=${GPU:-3}
ARGS="--pool built/pool_train_1M.jsonl --corpus built/corpus_mix2M.jsonl \
 --n_distractor_articles 42484 --image_dataset infoseek --max_train_rows 0 \
 --max_steps 1500 --lr_schedule constant --no_force_gold --reward judge --algo plgrpo \
 --pl_group 4 --pl_k 4 --pl_support 24 --contrastive_coef 0 --seed 0 --learning_rate 2e-5 \
 --mem_frac 0.25 --index_ema 0.5 --index_refresh_extra 0"
for try in 1 2 3; do
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) done"; exit 0; }
  echo "$(date +%T) $NAME attempt $try gpu $GPU"
  bash "$D/run_cell.sh" "$GPU" "$NAME" "$ARGS" >> "$D/local_$NAME.log" 2>&1
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) done"; exit 0; }
  echo "$(date +%T) died at $(wc -l < "$D/runs/$NAME/metrics.jsonl" 2>/dev/null||echo 0)"; sleep 120
done
