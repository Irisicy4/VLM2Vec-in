#!/bin/bash
# Matched SFT checkpoint sweep on an MLX/Arnold worker (1 GPU): train -> eval every checkpoint.
# Exists because the 810k RL line has a full curve (ck750/1500/2250/3000) but its SFT
# comparator has a SINGLE point (sft-810k-short-ck600). Comparing an RL curve against one
# SFT checkpoint understates SFT the same way a per-seed-best comparison overstates RL.
# Env: SWEEP_NAME, SWEEP_POOL, SWEEP_IMAGE_DATASET, SWEEP_STEPS, SWEEP_SAVE, SWEEP_CKPTS.
set -uo pipefail
LH=$(cat /mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data/local_hostname.txt 2>/dev/null)
[ "$(hostname)" != "$LH" ] && pkill -x keep_gpu 2>/dev/null; true  # worker images only, never the dev box
ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-rl
export MMRAG_DATA=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data
export HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/hf_home
export PYTHONPATH=$ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=${TMPDIR:-/mnt/bn/tns-algo-video-public-my2/yijiangli/data/tmp}
mkdir -p "$TMPDIR"; cd "$ROOT"
D=$MMRAG_DATA

NAME=${SWEEP_NAME:?SWEEP_NAME not set}
POOL=${SWEEP_POOL:?SWEEP_POOL not set}
IMGDS=${SWEEP_IMAGE_DATASET:-infoseek}
STEPS=${SWEEP_STEPS:-1500}
SAVE=${SWEEP_SAVE:-250}
CKPTS=${SWEEP_CKPTS:-"250 500 750 1000 1250 1500"}
RUN=$D/runs/$NAME
echo "=== sft sweep $NAME (pool=$POOL steps=$STEPS save=$SAVE) start $(date) host=$(hostname) ==="
nvidia-smi -L

python3 - <<'PY' || pip install -q torch==2.8.0 transformers==4.57.0 peft==0.17.1 \
    huggingface_hub==0.36.2 accelerate==1.13.0 qwen-vl-utils torchvision pillow
import sys, torch, transformers, peft, qwen_vl_utils  # noqa
mj, mn = (int(x) for x in transformers.__version__.split('.')[:2])
sys.exit(0 if (mj, mn) >= (4, 51) else 1)
PY
python3 -c "import flash_attn" 2>/dev/null || pip install -q flash-attn==2.8.3 --no-build-isolation
python3 -c "import flash_attn" || { echo "FATAL: flash_attn unavailable"; exit 3; }

# --- train (skipped if the sweep's last checkpoint already exists) ---
if [ ! -d "$RUN/checkpoint-$STEPS" ]; then
  python3 -u mmrag/train_sft.py --profile gme2b --pool "$POOL" --image_dataset "$IMGDS" \
      --batch_size 24 --learning_rate 1e-4 --max_steps "$STEPS" --save_steps "$SAVE" \
      --seed 0 --output_dir "$RUN" --device cuda:0 || exit 5
fi

# --- eval every checkpoint with the SAME estimator the RL curve used ---
for CK in $CKPTS; do
  CKD=$RUN/checkpoint-$CK; RN=$NAME-ck$CK
  [ -d "$CKD" ] || { echo "skip $RN (no checkpoint)"; continue; }
  [ -f "$D/results/$RN.vqa_top5.json" ] && { echo "skip $RN (already evaluated)"; continue; }
  CACHE=$D/cache/corpus_$RN.pt
  python3 -u mmrag/encode_corpus.py --profile gme2b --checkpoint "$CKD" --cache "$CACHE" \
      --device cuda:0 --bs 128 &&
  python3 -u mmrag/eval_retrieval.py --profile gme2b --checkpoint "$CKD" --name "$RN" \
      --max-q 3000 --device cuda:0 --corpus_bs 128 --cache_corpus "$CACHE" &&
  python3 -u mmrag/eval_vqa.py --retrieval "$D/results/$RN.retrieval.json" \
      --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500 --device cuda:0 --batch_size 8
  rm -f "$CACHE"
done
echo "SFT_SWEEP_DONE $NAME $(date)"
