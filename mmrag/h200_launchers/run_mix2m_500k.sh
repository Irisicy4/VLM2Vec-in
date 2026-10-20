#!/bin/bash
# MIXED-DOMAIN RL AT LARGE QUERY VOLUME: 500k queries consumed against the 2M index.
#
# WHY IT DOES NOT EXIST YET. Every mixed-domain arm so far is 6k queries (1500 steps x bs 4). The only
# large-query curve in the project (v3-b16-1M, 50k-200k queries) is IN-DOMAIN -- mlx_entry_cell.sh pins
# --corpus built/corpus_small.jsonl and its CELL_TRAIN_ARGS carry no override. So the in-domain finding
# that RL's entity R@5 decays past ~60k queries while SFT's does not has never been tested on a haystack.
#
# COST, measured on this box rather than assumed (steps/min from checkpoint mtimes):
#     refresh_steps 100 (the mix2m-v3pure default)   2.4-3.5 queries/min  ->  99-145 days for 500k
#     refresh_steps 0   (frozen index)               6.0-10.2             ->  34-58 days
#   Batch size barely moves this: the cost is the frozen reader scoring pl_group x pl_k candidates per
#   QUERY, so it scales with queries, not optimizer steps. bs 16 buys utilisation, not a 4x.
#
# refresh_steps 500 is the compromise actually launched: ~50 days for the full 500k. It is a deliberate
# departure from mix2m-v3pure's 100, and any table using this arm must say so. It is defensible because
# refresh was measured to matter early and to be inside the noise floor by 6k queries (tab:ab_refresh),
# and at this horizon a 100-step refresh spends four fifths of the budget re-encoding.
#
# CHECKPOINTS EVERY 20k QUERIES (5000 steps). The run is designed to be cut short: at 50k queries it has
# already passed every existing mixed-domain arm, and at 200k it reaches the in-domain crossover point.
# Killing it early costs nothing already earned.
set -uo pipefail
D=/raid/icy/mmrag_data
GPU=${1:?gpu}; NAME=${2:-mix2m-q500k}
export MMRAG_DATA=$D HF_HOME=/raid/icy/iris/.cache/huggingface
export PATH=$D/venv/bin:$PATH CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MMRAG_MEM_FRAC=${MMRAG_MEM_FRAC:-0.35}
cd /raid/icy/VLM2Vec-in
echo "$(date +%T) $NAME on gpu $GPU: 125000 steps x bs4 = 500k queries, 2M index, ckpt every 20k queries"
exec python3 -u mmrag/train_rl.py --profile gme2b \
    --output_dir "$D/runs/$NAME" --pool built/pool_train_1M.jsonl \
    --corpus built/corpus_mix2M.jsonl --image_dataset infoseek \
    --algo plgrpo --pl_group 4 --pl_k 4 --pl_support 24 --temperature 0.02 \
    --contrastive_coef 0 --no_force_gold --reward judge \
    --batch_size 4 --learning_rate 2e-5 --max_steps 125000 --save_steps 5000 \
    --refresh_steps 500 --seed 0 --device cuda:0
