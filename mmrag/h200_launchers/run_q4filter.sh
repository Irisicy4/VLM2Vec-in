#!/bin/bash
# Offline Q4 pool filter, gated on real headroom on MY cards only (5 and 7).
# Sized at 20k candidates: at the eval-set RAG-BENEFITS rate (~32%) that yields ~6.5k kept rows,
# and the winning recipe draws 1500 steps x batch 4 = 6,000 queries, so ~1 epoch of filtered pool.
# Resumable: re-running continues from <out>.decisions.jsonl, so a kill costs only the current chunk.
set -uo pipefail
D=/raid/icy/mmrag_data
NEED=${NEED:-20}; GPUS="${GPUS:-5 7}"; N=${N:-20000}
export MMRAG_DATA=$D HF_HOME=/raid/icy/iris/.cache/huggingface
export PATH=$D/venv/bin:$PATH
export MMRAG_MEM_FRAC=${MMRAG_MEM_FRAC:-0.25}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /raid/icy/VLM2Vec-in
for i in $(seq 1 480); do        # up to 8 h of waiting, checked every minute
  g=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
      | tr -d ',' | awk -v ok=" $GPUS " 'index(ok," "$1" ")==0{next}{print}' \
      | awk -v m="$NEED" '{f=($3-$2)/1024; if(f>m) print f, $1}' | sort -rn | head -1 | awk '{print $2}')
  if [ -n "${g:-}" ]; then
    echo "$(date +%H:%M) q4filter: gpu $g has headroom (need ${NEED} GiB) — starting, N=$N"
    export CUDA_VISIBLE_DEVICES=$g
    exec python3 -u mmrag/filter_ragbenefit.py --n "$N" --device cuda:0 \
         --out built/pool_train_q4filt.jsonl
  fi
  [ $((i % 15)) -eq 1 ] && echo "$(date +%H:%M) q4filter: waiting for ${NEED} GiB on [$GPUS]"
  sleep 60
done
echo "$(date +%H:%M) q4filter: gave up waiting"
