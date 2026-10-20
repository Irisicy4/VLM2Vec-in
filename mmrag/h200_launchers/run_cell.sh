#!/bin/bash
# Local (non-MLX) cell launcher for the shared H200 box.
#   bash run_cell.sh <GPU> <CELL_NAME> "<train args>"
# Pins the isolated venv (never the shared system python), maps cuda:0 onto the chosen
# physical GPU, and caps VRAM so the co-resident job on this GPU is not starved.
set -uo pipefail
GPU=${1:?gpu index}; NAME=${2:?cell name}; ARGS=${3:-}
export MMRAG_DATA=/raid/icy/mmrag_data
export HF_HOME=/raid/icy/iris/.cache/huggingface
export PATH=$MMRAG_DATA/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Caps EVERY stage, incl. the three eval stages that --mem_frac cannot reach (CELL_TRAIN_ARGS
# goes only to train_rl.py). Sized against the neighbour verl job's ~80.6 GiB vLLM-rollout
# peak, not its ~56.4 GiB training phase: its sleep-mode vLLM RECLAIMS ~64 GiB on wake and
# dies if we are holding it. Applied by venv/lib/python3.10/site-packages/mmrag_memcap.py.
export MMRAG_MEM_FRAC=${MMRAG_MEM_FRAC:-0.25}
export TMPDIR=$MMRAG_DATA/tmp
export CELL_NAME=$NAME CELL_PROFILE=gme2b CELL_TRAIN_ARGS="$ARGS"
cd /raid/icy/VLM2Vec-in
echo "launcher: gpu=$GPU cell=$NAME python=$(which python3)"
exec bash mmrag/mlx_entry_cell.sh
