#!/bin/bash
# Evaluate the CONTROL arm at the registered checkpoints, so the EMA-vs-no-EMA comparison is
# same-hardware, same-seed, same-args-except-the-two-EMA-flags. Without this the only available
# comparison is cross-site against the origin cluster's published baseline.
D=/raid/icy/mmrag_data
RUN=$D/runs/v3pure-1M-s3000-loc
GPU=${GPU:-6}
for STEP in ${STEPS:-750 1500}; do
  CK=$RUN/checkpoint-$STEP; NAME=v3pure-1M-s3000-loc-ck$STEP
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME done"; continue; }
  echo "$(date +%T) waiting for $CK ..."
  while [ ! -f "$CK/adapter_model.safetensors" ]; do
    pgrep -f "train_rl.py --profile gme2b" >/dev/null || { echo "$(date +%T) no trainer, stop"; exit 3; }
    sleep 120
  done
  sleep 30
  echo "$(date +%T) $CK -> eval on GPU $GPU"
  bash $D/run_eval.sh "$GPU" "$NAME" "runs/v3pure-1M-s3000-loc/checkpoint-$STEP" > "$D/local_$NAME.eval.log" 2>&1
  echo "$(date +%T) exit=$? $NAME"
  RM=$D/results/$NAME.retrieval.metrics.json
  [ -f "$RM" ] && python3 -c "
import json;d=json.load(open('$RM'));e,a=d['entity_recall'],d['answer_recall']
print('CONTROL %s  entR@5=%.2f ansR@5=%.2f  n=%s'%(d['name'],e['5']*100,a['5']*100,d['n']))"
done
echo "$(date +%T) control eval watcher finished"
