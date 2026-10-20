#!/bin/bash
# Evaluate one adapter against a DIFFERENT eval index than the campaign has ever used.
#   bash cross_eval.sh <GPU> <RESULT_NAME> <CKPT|__profile__> <CORPUS_REL>
# encode_corpus.py hardcodes built/corpus_small.jsonl, so it cannot pre-encode another corpus —
# eval_retrieval.py encodes inline instead (no --cache_corpus). eval_vqa MUST get the same
# --corpus or it will look passage pids up in the wrong file.
set -uo pipefail
D=/raid/icy/mmrag_data
GPU=$1; NAME=$2; CK=$3; CORPUS=$4
export MMRAG_DATA=$D HF_HOME=/raid/icy/iris/.cache/huggingface
export PATH=$D/venv/bin:$PATH CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MMRAG_MEM_FRAC=${MMRAG_MEM_FRAC:-0.25}
cd /raid/icy/VLM2Vec-in
echo "$(date +%T) cross-eval $NAME  ckpt=$CK  corpus=$CORPUS  gpu=$GPU"
python3 -u mmrag/eval_retrieval.py --profile gme2b --checkpoint "$CK" --name "$NAME" \
    --corpus "$CORPUS" --max-q 3000 --device cuda:0 --corpus_bs 128 || exit 7
python3 -u mmrag/eval_vqa.py --retrieval "$D/results/$NAME.retrieval.json" \
    --corpus "$CORPUS" --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500 \
    --device cuda:0 --batch_size 8 || exit 8
echo "$(date +%T) CROSS_EVAL_DONE $NAME"
