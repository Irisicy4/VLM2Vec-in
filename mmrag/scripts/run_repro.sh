#!/bin/bash
# mmrag headline repro: gme2b zero-shot vs no-gold RL checkpoint, aligned protocol
# (retrieval n=3000 seed-0 subsample of queries_test, VQA n=1500 seed-0 subsample of that, k=5,
#  reader Qwen2.5-VL-7B, v2 token-boundary cover-EM).
#   usage: bash run_repro.sh <arm>   with arm in {zs, nogold}
set -euo pipefail
ARM=${1:?arm: zs|nogold}
export HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/hf_home
export MMRAG_DATA=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data
export PYTHONPATH=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-in
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PYTHONPATH"

case "$ARM" in
  zs)
    NAME=gme2b-zeroshot-3k-repro
    CKPT_ARGS=(--checkpoint "__profile__")   # gme2b profile checkpoint is None -> zero-shot
    CACHE=$MMRAG_DATA/cache/corpus_zs.pt
    DEV=cuda:0
    ;;
  nogold)
    NAME=rl-j2e5-nogold-det-repro
    CK=$(ls -d "$HF_HOME"/hub/models--Icey444--mmrag-rl-nogold-judge-2b/snapshots/*/ | head -1)
    CKPT_ARGS=(--checkpoint "${CK%/}")
    CACHE=$MMRAG_DATA/cache/corpus_nogold.pt
    DEV=cuda:1
    ;;
  *) echo "bad arm"; exit 1;;
esac

echo "=== [$NAME] L1 retrieval on $DEV ==="
python3 mmrag/eval_retrieval.py --profile gme2b "${CKPT_ARGS[@]}" --name "$NAME" \
    --max-q 3000 --device "$DEV" --corpus_bs 256 --cache_corpus "$CACHE"

echo "=== [$NAME] L2 VQA (top-5, 7B reader) on $DEV ==="
python3 mmrag/eval_vqa.py --retrieval "$MMRAG_DATA/results/$NAME.retrieval.json" \
    --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500 --device "$DEV" --batch_size 8

echo "REPRO_DONE $NAME"
