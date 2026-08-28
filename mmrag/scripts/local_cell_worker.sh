#!/bin/bash
# Local fallback for MLX cell jobs: pops "name|profile|args" lines from cell_queue.txt and runs
# the same idempotent pipeline as mlx_entry_cell.sh, one worker per GPU.
#   usage: local_cell_worker.sh <gpu>
GPU=${1:?gpu}
D=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data
Q=$D/cell_queue.txt
LOCK=$D/.cellq_lock
ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-rl

pop() {
  while ! mkdir "$LOCK" 2>/dev/null; do sleep 2; done
  local line=""
  if [ -s "$Q" ]; then line=$(head -1 "$Q"); tail -n +2 "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"; fi
  rmdir "$LOCK"
  echo "$line"
}

while true; do
  LINE=$(pop); [ -z "$LINE" ] && { echo "[w$GPU] queue empty"; break; }
  IFS='|' read -r NAME PROFILE ARGS <<< "$LINE"
  echo "[w$GPU] START $NAME $(date +%m-%d_%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU CELL_NAME=$NAME CELL_PROFILE=$PROFILE CELL_TRAIN_ARGS=$ARGS \
      bash "$ROOT/mmrag/mlx_entry_cell.sh" > "$D/local_$NAME.log" 2>&1
  if grep -q "CELL_DONE\|already complete" "$D/local_$NAME.log"; then
    echo "[w$GPU] DONE $NAME $(date +%m-%d_%H:%M) $(grep -hE 'acc=' $D/results/$NAME.vqa_top5.json 2>/dev/null | head -c 0; python3 -c "import json;print('acc=%.4f'%json.load(open('$D/results/$NAME.vqa_top5.json'))['acc'])" 2>/dev/null)"
  else
    echo "[w$GPU] FAILED $NAME — requeueing once"
    if ! grep -q "RETRY:$NAME" "$D/cellq_retries.txt" 2>/dev/null; then
      echo "RETRY:$NAME" >> "$D/cellq_retries.txt"
      while ! mkdir "$LOCK" 2>/dev/null; do sleep 2; done
      printf '%s\n' "$LINE" >> "$Q"; rmdir "$LOCK"
    fi
  fi
done
