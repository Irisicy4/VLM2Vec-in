#!/bin/bash
# Interim-checkpoint eval on the shared box.
#   bash run_eval.sh <GPU> <RESULT_NAME> <CKPT_DIR_RELATIVE_TO_MMRAG_DATA>
set -uo pipefail
GPU=${1:?gpu}; NAME=${2:?result name}; CK=${3:?ckpt dir rel to MMRAG_DATA}
export MMRAG_DATA=/raid/icy/mmrag_data
export HF_HOME=/raid/icy/iris/.cache/huggingface
export PATH=$MMRAG_DATA/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPU
# Same shared-box cap as run_cell.sh; see mmrag_memcap.py.
export MMRAG_MEM_FRAC=${MMRAG_MEM_FRAC:-0.25}
export CELL_NAME=$NAME CELL_CKPT=$CK CELL_PROFILE=gme2b
cd /raid/icy/VLM2Vec-in
echo "eval launcher: gpu=$GPU name=$NAME ckpt=$CK python=$(which python3)"
exec bash mmrag/mlx_entry_eval.sh
