#!/bin/bash
# Generic mm-RAG cell on an MLX/Arnold worker (1 GPU): train -> corpus encode -> L1 -> L2.
# Env: CELL_NAME, CELL_PROFILE (default gme2b), CELL_TRAIN_ARGS (extra train_rl.py flags).
# Idempotent: exits immediately if results/<name>.vqa_top5.json already exists.
set -uo pipefail
ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-in
export MMRAG_DATA=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data
export HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/hf_home
export PYTHONPATH=$ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=${TMPDIR:-/mnt/bn/tns-algo-video-public-my2/yijiangli/data/tmp}
mkdir -p "$TMPDIR"; cd "$ROOT"

NAME=${CELL_NAME:?CELL_NAME not set}
PROFILE=${CELL_PROFILE:-gme2b}
D=$MMRAG_DATA
echo "=== cell $NAME (profile=$PROFILE) start $(date) host=$(hostname) ==="
echo "    train args: ${CELL_TRAIN_ARGS:-}"
nvidia-smi -L

python3 - <<'PY' || pip install -q torch==2.8.0 transformers==4.57.0 peft==0.17.1 \
    huggingface_hub==0.36.2 accelerate==1.13.0 qwen-vl-utils pillow
import sys, torch, transformers, peft, qwen_vl_utils  # noqa
mj, mn = (int(x) for x in transformers.__version__.split('.')[:2])
sys.exit(0 if (mj, mn) >= (4, 51) else 1)
PY
python3 -c "import flash_attn" 2>/dev/null || pip install -q flash-attn==2.8.1 --no-build-isolation
python3 -c "import flash_attn" || { echo "FATAL: flash_attn unavailable"; exit 3; }

[ -f "$D/results/$NAME.vqa_top5.json" ] && { echo "already complete"; exit 0; }
RUN=$D/runs/$NAME

if [ ! -f "$RUN/adapter_model.safetensors" ]; then
  # shellcheck disable=SC2086
  python3 -u mmrag/train_rl.py --profile "$PROFILE" --output_dir "$RUN" \
      --corpus built/corpus_small.jsonl --algo grpo --temperature 0.02 \
      --batch_size 4 --num_candidates 8 --lora_r 32 --lora_alpha 64 --epsilon 0.2 \
      --vf_coef 0.0 --beta 0.0 --contrastive_coef 0.3 --contrastive_temperature 0.03 \
      --reward judge --reader_model Qwen/Qwen2.5-VL-3B-Instruct \
      --reader_rollouts 4 --reader_temperature 0.7 --reader_batch_size 16 \
      --n_distractor_articles 3000 --refresh_steps 100 \
      --encode_bs 64 --max_len_doc 512 --save_steps 250 --log_every 25 \
      --device cuda:0 ${CELL_TRAIN_ARGS:-} || exit 5
fi

CACHE=$D/cache/corpus_$NAME.pt
python3 -u $D/encode_corpus.py --profile "$PROFILE" --checkpoint "$RUN" --cache "$CACHE" --device cuda:0 --bs 192 || exit 6
python3 -u mmrag/eval_retrieval.py --profile "$PROFILE" --checkpoint "$RUN" --name "$NAME" \
    --max-q 3000 --device cuda:0 --corpus_bs 192 --cache_corpus "$CACHE" || exit 7
python3 -u mmrag/eval_vqa.py --retrieval "$D/results/$NAME.retrieval.json" \
    --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500 --device cuda:0 --batch_size 8 || exit 8
rm -f "$CACHE"
echo "CELL_DONE $NAME $(date)"
