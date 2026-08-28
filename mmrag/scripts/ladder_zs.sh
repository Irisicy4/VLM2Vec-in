#!/bin/bash
# Zero-shot evals for the base-model ladder (local, eval-only). usage: ladder_zs.sh <profile> <name> <gpu>
export HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/hf_home
export MMRAG_DATA=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data
export PYTHONPATH=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-rl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$MMRAG_DATA; PROFILE=$1; NAME=$2; GPU=$3
cd "$PYTHONPATH"
until grep -q LADDER_DL_DONE $D/dl_ladder.log 2>/dev/null; do sleep 60; done
CACHE=$D/cache/corpus_$NAME.pt
python3 -u "$PYTHONPATH/mmrag/encode_corpus.py" --profile "$PROFILE" --checkpoint __profile__ \
    --cache "$CACHE" --device cuda:$GPU --bs 128 &&
python3 -u mmrag/eval_retrieval.py --profile "$PROFILE" --checkpoint __profile__ --name "$NAME" \
    --max-q 3000 --device cuda:$GPU --corpus_bs 128 --cache_corpus "$CACHE" &&
python3 -u mmrag/eval_vqa.py --retrieval "$D/results/$NAME.retrieval.json" \
    --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500 --device cuda:$GPU --batch_size 8 &&
rm -f "$CACHE" && echo "ZS_DONE $NAME"
