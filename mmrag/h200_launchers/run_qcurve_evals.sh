#!/bin/bash
# Score the matched SFT query-scaling curve. runs/sft-qcurve finished all 6250 steps at bs 32 with a
# checkpoint every 625 steps, i.e. every 20k queries consumed: 20k, 40k ... 200k, ten points on exactly
# the x grid the RL bs-16 arms (v3-b16-local 10k-60k, v3-b16-1M 50k-200k) already occupy.
#
# IN-DOMAIN eval (422K index, ~40 min per point) on purpose: the RL curve it is being matched against is
# scored the same way, so the comparison holds everything but the objective fixed. Scoring these on the
# 2M index instead would cost 7-13 h EACH and answer a different question.
#
# run_eval.sh is the wrapper that produced every other in-domain number: encode_corpus.py re-encodes the
# 422K corpus with the checkpoint, then retrieval, then the reader pass. The wrapper prints EVAL_DONE
# unconditionally after its && chain, so success is tested by the vqa file existing, never by that string.
set -uo pipefail
D=/raid/icy/mmrag_data
GPU=${1:-7}
export MMRAG_MEM_FRAC=${MMRAG_MEM_FRAC:-0.40}
for CK in 625 1250 1875 2500 3125 3750 4375 5000 5625; do
  NAME="sft-qcurve-ck${CK}"
  [ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$(date +%T) $NAME done"; continue; }
  echo "$(date +%T) $NAME  ($((CK*32)) queries consumed)"
  bash "$D/run_eval.sh" "$GPU" "$NAME" "runs/sft-qcurve/checkpoint-$CK" >> "$D/local_eval_$NAME.log" 2>&1 || true
  [ -f "$D/results/$NAME.vqa_top5.json" ] \
    && MMRAG_DATA=$D $D/venv/bin/python3 /raid/icy/VLM2Vec-in/mmrag/rescore.py --arms "$NAME" --write >/dev/null 2>&1 \
    || echo "$(date +%T) $NAME INCOMPLETE"
done
# the endpoint: the final adapter = 6250 steps = 200k queries
NAME=sft-qcurve
if [ ! -f "$D/results/$NAME.vqa_top5.json" ]; then
  echo "$(date +%T) $NAME (200k queries, endpoint)"
  bash "$D/run_eval.sh" "$GPU" "$NAME" "runs/sft-qcurve" >> "$D/local_eval_$NAME.log" 2>&1 || true
  [ -f "$D/results/$NAME.vqa_top5.json" ] \
    && MMRAG_DATA=$D $D/venv/bin/python3 /raid/icy/VLM2Vec-in/mmrag/rescore.py --arms "$NAME" --write >/dev/null 2>&1 \
    || echo "$(date +%T) $NAME INCOMPLETE"
fi
echo "$(date +%T) qcurve evals done"
