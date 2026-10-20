#!/bin/bash
# Self-driving eval orchestrator. Every cycle: find checkpoints that exist but have no result,
# pick a GPU with real headroom, launch an eval there. Caps concurrency so evals do not starve
# the trainers. Runs until the deadline.
#   bash orchestrate.sh <hours>
D=/raid/icy/mmrag_data
DEADLINE=$(( $(date +%s) + ${1:-24}*3600 ))
MAXEVAL=${MAXEVAL:-3}          # concurrent evals
MINFREE=${MINFREE:-32}         # GiB a card must have spare to take an eval
GPUS="${GPUS:-5 7}"        # ALLOWLIST — 5 and 7 are mine; 4 and 6 belong to h200-rag

CELLS="fullcorp-rand05 fullcorp-rand10 fullcorp-rand25 fullcorp-rand50 fullcorp-stale \
fullcorp-rand10p50 idx10k-partial idx10k-full mix2m-v3pure-s1 mix2m-v3pure-ema-s1 \
mix2m-partialwb mix2m-modelema-ctrl mix2m-v2anchor mix2m-modelema divmix-v3pure"

free_gpu() {   # echo the index of the emptiest card if it clears MINFREE, else nothing
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
   | tr -d ',' | awk -v ok=" $GPUS " -v m="$MINFREE" 'index(ok," "$1" ")==0{next}{print}' \
   | awk -v m="$MINFREE" '{f=($3-$2)/1024; if(f>m) print f, $1}' | sort -rn | head -1 | awk '{print $2}'
}
n_evals() {   # single integer, always
  local n
  n=$(pgrep -fc "mmrag/(encode_corpus|eval_retrieval|eval_vqa)" 2>/dev/null)
  [ -z "$n" ] && n=0
  printf '%d' "$n"
}

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  for c in $CELLS; do
    [ -d "$D/runs/$c" ] || continue
    # endpoint: final adapter present, no result yet
    if [ -f "$D/runs/$c/adapter_model.safetensors" ] && [ ! -f "$D/results/$c.vqa_top5.json" ]; then
      if [ "$(n_evals)" -lt "$MAXEVAL" ]; then
        g=$(free_gpu); [ -z "$g" ] && continue
        echo "$(date +%H:%M) EVAL endpoint $c -> gpu $g"
        rm -f "$D/cache/corpus_$c.pt"
        nohup bash "$D/run_eval.sh" "$g" "$c" "runs/$c" > "$D/local_$c.endeval.log" 2>&1 &
        sleep 90
      fi
    fi
    # ck750: checkpoint present, no result yet
    if [ -f "$D/runs/$c/checkpoint-750/adapter_model.safetensors" ] && [ ! -f "$D/results/$c-ck750.vqa_top5.json" ]; then
      if [ "$(n_evals)" -lt "$MAXEVAL" ]; then
        g=$(free_gpu); [ -z "$g" ] && continue
        echo "$(date +%H:%M) EVAL ck750 $c -> gpu $g"
        rm -f "$D/cache/corpus_$c-ck750.pt"
        nohup bash "$D/run_eval.sh" "$g" "$c-ck750" "runs/$c/checkpoint-750" > "$D/local_$c-ck750.endeval.log" 2>&1 &
        sleep 90
      fi
    fi
  done
  sleep 300
done
echo "$(date +%H:%M) ORCHESTRATOR DEADLINE REACHED"
