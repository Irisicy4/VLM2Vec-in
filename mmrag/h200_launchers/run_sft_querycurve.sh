#!/bin/bash
# MATCHED SFT CURVE ON THE QUERIES-CONSUMED AXIS (x = steps x batch, NOT pool size).
#
# The supervised side of that plot currently splices two families -- sft-curve (4.8k-38.4k queries) and
# sft-big-1M (67.2k-873.6k) -- with a gap across 38.4k-67.2k and no guarantee the two share a pool or an
# lr. One run with uniform checkpoints removes the splice and lands on the same x grid as the RL bs-16
# arms (v3-b16-local 10k-60k, v3-b16-1M 50k-200k).
#
# bs 32, save_steps 625 -> checkpoints at 20k, 40k, 60k, ... 200k queries consumed (10 points).
# Full pool (max_train_rows 0 = all 810,276) so pool size is NOT the varying quantity here; only the
# number of queries the optimiser actually sees varies. seed 0.
#
# Measured: the sft-rows pair did 1200 steps in 3h18m on one card = 6.1 steps/min, so 6250 steps is
# roughly 17 h. Checkpoints are usable as they land, so partial results are still a partial curve.
set -uo pipefail
D=/raid/icy/mmrag_data
GPU=${1:?gpu}; NAME=sft-qcurve
export MMRAG_DATA=$D HF_HOME=/raid/icy/iris/.cache/huggingface
export PATH=$D/venv/bin:$PATH CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MMRAG_MEM_FRAC=0.40
cd /raid/icy/VLM2Vec-in
echo "$(date +%T) SFT query-curve on gpu $GPU: 6250 steps x bs32 = 200k queries, ckpt every 20k"
exec python3 -u mmrag/train_sft.py --profile gme2b --pool built/pool_train_1M.jsonl \
    --image_dataset infoseek --output_dir "$D/runs/$NAME" \
    --batch_size 32 --learning_rate 1e-4 --max_steps 6250 --save_steps 625 \
    --max_train_rows 0 --seed 0 --device cuda:0
