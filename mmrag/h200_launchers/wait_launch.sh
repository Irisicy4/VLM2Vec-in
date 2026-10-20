#!/bin/bash
# Launch a cell only once its card actually has headroom, and keep waiting rather than failing.
# Three OOM-at-startup losses on the same cell came from launching blind onto a momentarily full card.
#   bash wait_launch.sh <script> <name> <gpu> "<extra args>" <need_GiB>
set -uo pipefail
SCRIPT=$1; NAME=$2; GPU=$3; EXTRA=$4; NEED=${5:-34}
D=/raid/icy/mmrag_data
for i in $(seq 1 240); do            # up to 4 h of waiting, checked every minute
  free=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
         | tr -d ',' | awk -v g="$GPU" '$1==g{printf "%d",($3-$2)/1024}')
  if [ "${free:-0}" -ge "$NEED" ]; then
    echo "$(date +%H:%M) $NAME: gpu $GPU has ${free} GiB free (need $NEED) — launching"
    exec bash "$D/$SCRIPT" "$NAME" "$GPU" "$EXTRA"
  fi
  [ $((i % 10)) -eq 1 ] && echo "$(date +%H:%M) $NAME: gpu $GPU only ${free:-?} GiB free, waiting"
  sleep 60
done
echo "$(date +%H:%M) $NAME: gave up waiting for gpu $GPU"
