#!/bin/bash
# Keep emaidx-1M-s3000 alive across external kills.
#
# WHY: train_rl.py has NO resume path -- a SIGKILL loses everything back to step 0, including the
# ~36 min init corpus encode. On 2026-09-07 the trainer was SIGKILLed at step ~113, 18 s after
# another tenant launched vLLM servers on this host. Restarting is scientifically clean: the run
# is seeded, so a fresh start reproduces the identical row sequence. It is only expensive in time,
# and only if nobody notices for hours -- which is what this fixes.
#
# The entry script is itself idempotent: it skips training once $RUN/adapter_model.safetensors
# exists and skips the whole cell once results/<name>.vqa_top5.json exists, so re-invoking it
# after a completed run is a no-op rather than a redo.
set -uo pipefail
D=/raid/icy/mmrag_data
NAME=emaidx-1M-s3000
GPU=${GPU:-1}
MAXTRIES=${MAXTRIES:-20}
ARGS="--pool built/pool_train_1M.jsonl --image_dataset infoseek --max_train_rows 0 \
 --max_steps 3000 --lr_schedule constant --no_force_gold --reward judge --algo plgrpo \
 --pl_group 4 --pl_k 4 --pl_support 24 --contrastive_coef 0 --seed 0 --learning_rate 2e-5 \
 --index_ema 0.9 --index_refresh_extra 256 --mem_frac 0.25"

for try in $(seq 1 "$MAXTRIES"); do
  if [ -f "$D/results/$NAME.vqa_top5.json" ]; then
    echo "$(date +%T) SUPERVISOR: cell complete, nothing to do"; exit 0
  fi
  echo "$(date +%T) SUPERVISOR: attempt $try/$MAXTRIES launching $NAME on GPU $GPU"
  bash "$D/run_cell.sh" "$GPU" "$NAME" "$ARGS" >> "$D/local_$NAME.log" 2>&1
  rc=$?
  if [ -f "$D/results/$NAME.vqa_top5.json" ]; then
    echo "$(date +%T) SUPERVISOR: cell finished cleanly (rc=$rc)"; exit 0
  fi
  STEPS=$(wc -l < "$D/runs/$NAME/metrics.jsonl" 2>/dev/null || echo 0)
  echo "$(date +%T) SUPERVISOR: attempt $try DIED rc=$rc at ~step $STEPS — restarting in 120s"
  sleep 120
done
echo "$(date +%T) SUPERVISOR: gave up after $MAXTRIES attempts"; exit 1
