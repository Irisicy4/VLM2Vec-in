#!/bin/bash
# Fire the estimator-matched eval as soon as a checkpoint appears, on a GPU in OUR lane
# (1/3/5/7; 0/2/4/6 belong to the h200-rag session). Training holds GPU 1, so evals go to 3.
# Result names mirror the baseline's so compare_ema.py pairs them automatically.
D=/raid/icy/mmrag_data
RUN=$D/runs/emaidx-1M-s3000
GPU=${GPU:-5}   # 5/7 idle; GPU 3 hosts the control arm v3pure-1M-s3000-loc
for STEP in ${STEPS:-750 1500}; do
  CK=$RUN/checkpoint-$STEP
  NAME=emaidx-1M-s3000-ck$STEP
  # already evaluated? skip (run_eval.sh is idempotent too, but avoid the model load)
  if [ -f "$D/results/$NAME.vqa_top5.json" ]; then echo "$(date +%T) $NAME already done"; continue; fi
  echo "$(date +%T) waiting for $CK ..."
  while [ ! -f "$CK/adapter_model.safetensors" ]; do
    # bail out if training died — no point waiting forever
    pgrep -f "train_rl.py --profile gme2b" >/dev/null || { echo "$(date +%T) TRAINER GONE, stopping watcher"; exit 3; }
    sleep 120
  done
  sleep 30   # let the checkpoint finish writing
  echo "$(date +%T) $CK present -> eval on GPU $GPU"
  bash $D/run_eval.sh "$GPU" "$NAME" "runs/emaidx-1M-s3000/checkpoint-$STEP" \
      > "$D/local_$NAME.eval.log" 2>&1
  echo "$(date +%T) eval exit=$? for $NAME"
  # RETRIEVAL is the primary product and is written before the VQA stage — report it on its own,
  # so a slow or failed VQA stage never hides entR@5/ansR@5 (ansR@5 is the registered primary).
  RM=$D/results/$NAME.retrieval.metrics.json
  if [ -f "$RM" ]; then
    python3 -c "
import json;d=json.load(open('$RM'))
e,a=d['entity_recall'],d['answer_recall']
print('RETRIEVAL %s  entR@1=%.2f entR@5=%.2f entR@10=%.2f  ansR@5=%.2f  n=%s n_ans=%s'%(
      d['name'],e['1']*100,e['5']*100,e['10']*100,a['5']*100,d['n'],d.get('n_ans')))"
  else
    echo "RETRIEVAL MISSING for $NAME — primary metric absent, investigate before reading acc"
  fi
  grep -aE "encode=|gen=|L2 VQA" "$D/local_$NAME.eval.log" | tail -4
done
echo "$(date +%T) auto_eval watcher finished"
