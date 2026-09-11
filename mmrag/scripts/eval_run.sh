#!/bin/bash
# eval_run.sh <run_name> <profile> <gpu>
export HF_HOME=/raid/icy/iris/.cache/huggingface
export MMRAG_DATA=/raid/icy/mmrag_data
export PYTHONPATH=/raid/icy/VLM2Vec-in
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$MMRAG_DATA; NAME=$1; PROF=$2; GPU=$3; CKPT=$D/runs/$1
cd "$PYTHONPATH"
CACHE=$D/cache/corpus_$NAME.pt
python3 -u mmrag/encode_corpus.py --profile $PROF --checkpoint "$CKPT" --cache "$CACHE" --device cuda:$GPU --bs 128 &&
python3 -u mmrag/eval_retrieval.py --profile $PROF --checkpoint "$CKPT" --name "$NAME" --max-q 3000 --device cuda:$GPU --corpus_bs 128 --cache_corpus "$CACHE" &&
python3 -u mmrag/eval_vqa.py --retrieval "$D/results/$NAME.retrieval.json" --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500 --device cuda:$GPU --batch_size 8 &&
rm -f "$CACHE" && echo "EVAL_DONE $NAME"
