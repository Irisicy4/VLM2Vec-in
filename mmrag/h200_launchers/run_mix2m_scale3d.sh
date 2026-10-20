#!/bin/bash
# MIXED-DOMAIN QUERY-SCALING CURVE, sized to finish inside a 3-day budget.
#
# THE BINDING CONSTRAINT IS EVALUATION, NOT TRAINING. Measured from the bench2m chain that just
# finished, one checkpoint evaluated on the 2M index costs 7.2-13.5 h (it re-encodes 2M passages). A
# four-point curve scored on the haystack is 30-54 h of card time before a single training step. So:
#
#   curve points   evaluated IN-DOMAIN on the 422K index, ~40 min each -> 6-8 points affordable
#   haystack       2 points only, at the endpoints, on the 2M index
#
# The in-domain eval is a deliberate choice, not a concession. v3-b16-1M -- the only large-query curve
# the project has -- is trained AND evaluated in-domain. Scoring this curve the same way means the only
# thing that differs is the TRAINING index, so the contrast isolates the training-corpus effect, which
# is the actual question. The two haystack endpoints then test whether the conclusion transfers.
#
# CHECKPOINTS ARE DENSE AND UNIFORM (every 500 steps = 2k queries) and max_steps is set far beyond what
# 3 days can reach. The run is meant to be STOPPED at the deadline and a geometric subset of the landed
# checkpoints evaluated. That removes the need to predict throughput in advance, which matters because
# the reward backend changes it several-fold.
#
# REWARD BACKEND: --reward_server_url points at mmrag/reward_server_vllm.py. The reader is the
# bottleneck (up to 384 generations of 24 tokens per step under HF's padded generate), and vLLM's
# continuous batching attacks exactly that. The trainer also stops holding a reader locally, which frees
# card memory. Only switch this on after mmrag/verify_reward_parity.py shows HF-vs-vLLM disagreement no
# worse than HF-vs-HF -- every arm in the paper was trained against the HF reader.
set -uo pipefail
D=/raid/icy/mmrag_data
GPU=${1:?gpu}; NAME=${2:-mix2m-scale3d}; URL=${3:-}
export MMRAG_DATA=$D HF_HOME=/raid/icy/iris/.cache/huggingface
export PATH=$D/venv/bin:$PATH CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MMRAG_MEM_FRAC=${MMRAG_MEM_FRAC:-0.25}
cd /raid/icy/VLM2Vec-in
EXTRA=""
[ -n "$URL" ] && EXTRA="--reward_server_url $URL"
echo "$(date +%T) $NAME on gpu $GPU  reward=${URL:-in-process HF}  ckpt every 2k queries"
exec python3 -u mmrag/train_rl.py --profile gme2b \
    --output_dir "$D/runs/$NAME" --pool built/pool_train_1M.jsonl \
    --corpus built/corpus_mix2M.jsonl --image_dataset infoseek \
    --algo plgrpo --pl_group 4 --pl_k 4 --pl_support 24 --temperature 0.02 \
    --contrastive_coef 0 --no_force_gold --reward judge \
    --batch_size 4 --learning_rate 2e-5 --max_steps 50000 --save_steps 500 \
    --refresh_steps 500 --seed 0 --device cuda:0 $EXTRA
