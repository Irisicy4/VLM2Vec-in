#!/bin/bash
# Eval-only MLX worker: encode -> retrieval -> vqa for one checkpoint.
# Env: CELL_NAME (result name), CELL_CKPT (checkpoint dir under $MMRAG_DATA), CELL_PROFILE.
set -uo pipefail
[ -n "${ARNOLD_ID:-}${ARNOLD_TRIAL_ID:-}" ] && pkill -f keep_gpu 2>/dev/null || true  # MLX workers only
ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-rl
export MMRAG_DATA=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data
export HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/hf_home
export PYTHONPATH=$ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$MMRAG_DATA; NAME=$CELL_NAME; CK=$D/$CELL_CKPT; PROF=${CELL_PROFILE:-gme2b}
[ -f $D/results/$NAME.vqa_top5.json ] && exit 0
cd $ROOT
CACHE=$D/cache/corpus_$NAME.pt
python3 -u mmrag/encode_corpus.py --profile $PROF --checkpoint "$CK" --cache "$CACHE" --device cuda:0 --bs 128 &&
python3 -u mmrag/eval_retrieval.py --profile $PROF --checkpoint "$CK" --name "$NAME" --max-q 3000 --device cuda:0 --corpus_bs 128 --cache_corpus "$CACHE" &&
python3 -u mmrag/eval_vqa.py --retrieval "$D/results/$NAME.retrieval.json" --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500 --device cuda:0 --batch_size 8 &&
python3 mmrag/adapter_stats.py "$CK" > $D/results/$NAME.lorab.txt
rm -f "$CACHE"; echo "EVAL_DONE $NAME"
