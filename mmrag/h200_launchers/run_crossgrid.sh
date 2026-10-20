#!/bin/bash
# Complete the 2x2: TRAINING corpus x EVAL corpus, to separate in-domain gain from
# robustness to a larger, partly off-domain candidate pool.
#
#                          eval on 422k (corpus_small)      eval on 2M (corpus_mix2M)
#   base (untrained)       68.53  HAVE                      14.27  HAVE   <- collapses
#   trained on 422k        76.83  HAVE (emaidx ck1500)      69.83  HAVE   <- holds
#   trained on 2M          76.23  HAVE (mix2m-v3pure)       ????   THIS SCRIPT
#   trained on 422k        74.97  HAVE (idx10k-full ck750)  ????   THIS SCRIPT
#
# The base loses 54.3 points of entity recall on the bigger corpus while a trained arm loses 7.0.
# That 7.8x robustness gap is far larger than the in-domain accuracy margin and is the axis the
# eval table was missing. Runs SEQUENTIALLY: each does an inline 2M encode (~2.5 h at the measured
# 225 passages/s) and two of them on one card would just contend.
set -uo pipefail
D=/raid/icy/mmrag_data
GPU=${1:-7}
for spec in "mix2m-v3pure-on-mix2M:runs/mix2m-v3pure" "idx10k-full-on-mix2M:runs/idx10k-full"; do
  NAME=${spec%%:*}; CK=${spec##*:}
  if [ -f "$D/results/$NAME.vqa_top5.json" ]; then echo "$(date +%T) $NAME already done"; continue; fi
  echo "$(date +%T) starting $NAME"
  bash "$D/cross_eval.sh" "$GPU" "$NAME" "$D/$CK" built/corpus_mix2M.jsonl \
       >> "$D/local_$NAME.log" 2>&1 || echo "$(date +%T) $NAME FAILED (see local_$NAME.log)"
done
echo "$(date +%T) crossgrid done"
