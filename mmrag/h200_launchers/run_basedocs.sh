#!/bin/bash
# SECOND EVAL PROTOCOL: trained query tower against a BASE-encoded index.
#
# Every number in this project so far is self-consistent (the eval corpus is re-encoded by the
# trained checkpoint). h200-rag's 2x2 found that on their setup the trained query tower does BETTER
# against base-encoded documents than against its own (0.766 vs 0.626), and that protocol is also the
# cheap deployment: encode once, never re-encode. We have never measured it.
#
# The base encode is done ONCE and shared by every arm, so the marginal cost per arm is a query pass
# plus a reader pass. Gated on real headroom; the cross-eval grid has priority.
set -uo pipefail
D=/raid/icy/mmrag_data
GPUS="${GPUS:-5 7}"; NEED=${NEED:-32}
export MMRAG_DATA=$D HF_HOME=/raid/icy/iris/.cache/huggingface
export PATH=$D/venv/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MMRAG_MEM_FRAC=${MMRAG_MEM_FRAC:-0.25}
CACHE=$D/cache/corpus_BASEENC_small.pt
pick_gpu(){ nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  | tr -d ',' | awk -v ok=" $GPUS " 'index(ok," "$1" ")==0{next}{print}' \
  | awk -v m="$NEED" '{f=($3-$2)/1024; if(f>m) print f,$1}' | sort -rn | head -1 | awk '{print $2}'; }
wait_gpu(){ for i in $(seq 1 720); do g=$(pick_gpu); [ -n "${g:-}" ] && { echo "$g"; return; }; sleep 60; done; echo ""; }
cd /raid/icy/VLM2Vec-in
if [ ! -f "$CACHE" ]; then
  G=$(wait_gpu); [ -z "$G" ] && { echo "gave up waiting for a card"; exit 1; }
  echo "$(date +%T) base-encoding corpus_small on gpu $G (shared by every arm)"
  CUDA_VISIBLE_DEVICES=$G python3 -u mmrag/encode_corpus.py --profile gme2b \
      --checkpoint __profile__ --cache "$CACHE" --device cuda:0 --bs 128 || exit 7
fi
for ARM in mix2m-v3pure mix2m-v3pure-s1 frozen-idx q4gate-s1; do
  NAME="$ARM-basedocs"
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME done"; continue; }
  [ -f "$D/runs/$ARM/adapter_model.safetensors" ] || { echo "$(date +%T) $ARM has no adapter yet, skipping"; continue; }
  G=$(wait_gpu); [ -z "$G" ] && { echo "gave up waiting"; exit 1; }
  echo "$(date +%T) $NAME on gpu $G"
  ( export CUDA_VISIBLE_DEVICES=$G
    python3 -u mmrag/eval_retrieval.py --profile gme2b --checkpoint "$D/runs/$ARM" --name "$NAME" \
        --max-q 3000 --device cuda:0 --corpus_bs 128 --cache_corpus "$CACHE" &&
    python3 -u mmrag/eval_vqa.py --retrieval "$D/results/$NAME.retrieval.json" \
        --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500 --device cuda:0 --batch_size 8
  ) >> "$D/local_$NAME.log" 2>&1 || echo "$(date +%T) $NAME FAILED"
done
echo "$(date +%T) basedocs protocol done"
