#!/bin/bash
# LOCAL RELEVANCE-SFT ARM, built to answer one question: is the "robustness to a bigger corpus"
# effect actually about the indirect reward, or is it just what ANY fine-tuning on this task format
# buys? The untrained backbone loses 54 points of entity recall going from a 422k to a 2M corpus and
# every RL arm loses ~7. If a SUPERVISED arm also loses ~7 (or less), the effect is task adaptation,
# not something the indirect reward earns, and the claim must be withdrawn.
#
# No SFT adapter was ever staged on this box -- every supervised result we quote came from the
# original cluster as metrics only -- so the cross-corpus comparison could not be made until now.
# SFT needs no --corpus: positives are the gold passages carried in the pool, so it never sees an
# index during training, which is itself part of why this comparison is interesting.
set -uo pipefail
D=/raid/icy/mmrag_data
NAME=${1:?name}; GPU=${2:?gpu}; EXTRA=${3:-}
export MMRAG_DATA=$D HF_HOME=/raid/icy/iris/.cache/huggingface
export PATH=$D/venv/bin:$PATH CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MMRAG_MEM_FRAC=${MMRAG_MEM_FRAC:-0.25}
cd /raid/icy/VLM2Vec-in
echo "$(date +%T) SFT $NAME on gpu $GPU : $EXTRA"
exec python3 -u mmrag/train_sft.py --profile gme2b --pool built/pool_train_1M.jsonl \
    --image_dataset infoseek --output_dir "$D/runs/$NAME" \
    --batch_size 32 --learning_rate 1e-4 --max_steps 1200 --save_steps 150 \
    --device cuda:0 $EXTRA
