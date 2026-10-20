#!/bin/bash
# Dedicated eval watcher for the q4gate cells.
#
# Why a separate script rather than adding them to orchestrate.sh's CELLS: that orchestrator is
# ALREADY RUNNING (its CELLS was assigned once, before the while loop, so an edit would not be
# picked up anyway) and editing a live bash script IN PLACE can corrupt it, because bash re-reads
# from a byte offset in the same inode. Replacing the inode is safe; sed -i is not. So: new file.
#
# Evaluates ck750 (the matched-control depth: mix2m-v3pure / -s1 have ck750 results) and the
# endpoint. Same GPU allowlist and concurrency guard as orchestrate.sh.
set -uo pipefail
D=/raid/icy/mmrag_data
GPUS="${GPUS:-5 7}"          # 5 and 7 are mine; 4 and 6 belong to h200-rag
MINFREE=${MINFREE:-34}
MAXEVAL=${MAXEVAL:-2}
HOURS=${1:-24}
DEADLINE=$(( $(date +%s) + HOURS*3600 ))
CELLS="fullcorp-rand10p50 frozen-idx q4pool-filt q4pool-rand q4gate-s1 mix2m-modelema divmix-v3pure"

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
   | tr -d ',' | awk -v ok=" $GPUS " 'index(ok," "$1" ")==0{next}{print}' \
   | awk -v m="$MINFREE" '{f=($3-$2)/1024; if(f>m) print f, $1}' | sort -rn | head -1 | awk '{print $2}'
}
n_evals() {   # ALWAYS one integer. pgrep -fc prints 0 AND returns 1 when nothing matches, so a
  local n    # bare `|| echo 0` emits TWO values and every downstream [ ] test dies. Assign, then test.
  n=$(pgrep -fc "mmrag/(encode_corpus|eval_retrieval|eval_vqa)" 2>/dev/null)
  [ -z "$n" ] && n=0
  printf '%d' "$n"
}
launch() {  # $1=result name  $2=ckpt path rel to D
  [ -f "$D/results/$1.vqa_top5.json" ] && return
  # PER-CELL running check. Without this the loop relaunches the SAME eval every cycle until it
  # hits MAXEVAL, because the result file does not appear until the eval finishes (~15+ min) while
  # the cycle is far shorter. orchestrate.sh has exactly this defect and duplicated two endpoint
  # evals today, each burning ~30 GiB of a card on work already in flight. The total-concurrency
  # guard below does NOT catch it: two duplicates of one cell satisfy it perfectly.
  pgrep -f "run_eval.sh [0-9]* $1 " >/dev/null 2>&1 && return
  pgrep -f "corpus_$1\.pt" >/dev/null 2>&1 && return
  [ "$(n_evals)" -ge "$MAXEVAL" ] && return
  local g; g=$(free_gpu); [ -z "$g" ] && return
  echo "$(date +%H:%M) EVAL $1 -> gpu $g"
  rm -f "$D/cache/corpus_$1.pt"
  nohup bash "$D/run_eval.sh" "$g" "$1" "$2" > "$D/local_$1.endeval.log" 2>&1 &
  sleep 90
}
echo "$(date +%H:%M) watch_q4gate: cards[$GPUS] minfree=$MINFREE for ${HOURS}h"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  for c in $CELLS; do
    [ -d "$D/runs/$c" ] || continue
    [ -f "$D/runs/$c/checkpoint-750/adapter_model.safetensors" ] && launch "$c-ck750" "runs/$c/checkpoint-750"
    [ -f "$D/runs/$c/adapter_model.safetensors" ] && launch "$c" "runs/$c"
  done
  sleep 300
done
echo "$(date +%H:%M) watch_q4gate: deadline reached"
