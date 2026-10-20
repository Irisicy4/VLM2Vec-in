#!/bin/bash
# Reader-only rerun for a cell whose retrieval SUCCEEDED but whose VQA pass OOM'd.
#
# frozen-idx-s1-ck750 is the case: encode and retrieval completed on card 4, then eval_vqa died needing
# a further 78 MiB because two vLLM engines had filled the card underneath it. mlx_entry_eval.sh prints
# "EVAL_DONE" from an unconditional line after the && chain, so a half-finished cell looks finished in
# the log -- check for results/<NAME>.vqa_top5.json, not for that string.
#
# Only the reader pass is redone. The retrieval.json is already on disk and is what defines the slate,
# so this changes nothing about the retrieval protocol or its self-consistency.
set -uo pipefail
D=/raid/icy/mmrag_data
GPU=${1:?gpu}; NAME=${2:?result name}
export MMRAG_DATA=$D HF_HOME=/raid/icy/iris/.cache/huggingface
export PATH=$D/venv/bin:$PATH PYTHONPATH=/raid/icy/VLM2Vec-in
export CUDA_VISIBLE_DEVICES=$GPU MMRAG_MEM_FRAC=${MMRAG_MEM_FRAC:-0.40}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "$D/results/$NAME.retrieval.json" ] || { echo "no retrieval.json for $NAME"; exit 1; }
[ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "$NAME already has vqa"; exit 0; }
cd /raid/icy/VLM2Vec-in
echo "$(date +%T) reader pass for $NAME on gpu $GPU"
python3 -u mmrag/eval_vqa.py --retrieval "$D/results/$NAME.retrieval.json" \
    --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500 --device cuda:0 --batch_size 8 \
    && python3 mmrag/rescore.py --arms "$NAME" --write >/dev/null \
    && echo "$(date +%T) $NAME VQA OK" || echo "$(date +%T) $NAME VQA FAILED"
