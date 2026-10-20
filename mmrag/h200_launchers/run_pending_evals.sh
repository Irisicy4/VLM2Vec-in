#!/bin/bash
# The two runs that finished TRAINING on card 5 (step 1499/1500, adapters saved) but whose post-training
# eval died with torch.OutOfMemoryError: Tried to allocate 25.43 GiB -- they were sharing that card with
# each other at MMRAG_MEM_FRAC=0.25 (35 GiB). Card 4 is now empty, so they get real headroom.
#
#   frozen-idx-s1  the second seed tab:mm_refresh needs (matched to frozen-idx s0: cc 0.3, mix2M,
#                  refresh_steps 0, seed 1)
#   judge-alt2     the intermittent 3B/30B grader arm; its dead-gradient result is already settled from
#                  the training log (-0.1145, CI [-0.155,-0.074]), this is its RETRIEVAL read
#
# run_eval.sh is the same wrapper that produced the s0 numbers: encode_corpus.py re-encodes the corpus
# with the trained checkpoint into cache/corpus_<NAME>.pt, then retrieval, then vqa. Self-consistent, and
# protocol-identical to s0 -- do not substitute cross_eval.sh, which takes a different corpus.
#
# SEQUENTIAL on one card: each encode is GPU-bound, so running both at once buys nothing and reintroduces
# exactly the contention that OOM'd them.
set -uo pipefail
D=/raid/icy/mmrag_data
GPU=${1:-4}
export MMRAG_MEM_FRAC=0.55          # card is free; the failed allocation alone was 25.43 GiB
for NAME in frozen-idx-s1 judge-alt2; do
  if [ -f "$D/results/$NAME.vqa_top5.json" ]; then echo "$(date +%T) $NAME already evaluated"; continue; fi
  [ -f "$D/runs/$NAME/adapter_model.safetensors" ] || { echo "$(date +%T) $NAME has no adapter, skip"; continue; }
  rm -f "$D/cache/corpus_$NAME.pt"   # a partial cache from the OOM'd attempt would be silently reused
  echo "$(date +%T) evaluating $NAME on gpu $GPU"
  bash "$D/run_eval.sh" "$GPU" "$NAME" "runs/$NAME" >> "$D/local_eval_$NAME.log" 2>&1 \
    || echo "$(date +%T) $NAME EVAL FAILED"
done
echo "$(date +%T) pending evals done"
