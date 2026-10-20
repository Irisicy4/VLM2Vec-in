#!/bin/bash
# Evaluate a grid cell at the registered checkpoints. Eval index stays corpus_small for
# comparability with the whole campaign; only the TRAINING index differed.
D=/raid/icy/mmrag_data
NAME=$1; GPU=$2
for STEP in ${STEPS:-750 1500}; do
  CK=$D/runs/$NAME/checkpoint-$STEP; RN=$NAME-ck$STEP
  [ -f "$D/results/$RN.vqa_top5.json" ] && { echo "$(date +%T) $RN done"; continue; }
  while [ ! -f "$CK/adapter_model.safetensors" ]; do
    pgrep -f "output_dir /raid/icy/mmrag_data/runs/$NAME" >/dev/null || { echo "$(date +%T) $NAME trainer gone"; exit 3; }
    sleep 180
  done
  sleep 30
  echo "$(date +%T) $RN -> eval gpu $GPU"
  bash $D/run_eval.sh "$GPU" "$RN" "runs/$NAME/checkpoint-$STEP" > "$D/local_$RN.eval.log" 2>&1
  RM=$D/results/$RN.retrieval.metrics.json
  [ -f "$RM" ] && python3 -c "
import json;d=json.load(open('$RM'));e,a=d['entity_recall'],d['answer_recall']
print('GRID %s  entR@5=%.2f ansR@5=%.2f'%(d['name'],e['5']*100,a['5']*100))"
done
