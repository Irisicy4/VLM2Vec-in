#!/bin/bash
# One mm-RAG data-scaling cell on an MLX/Arnold worker (1 GPU per job).
#
# Extends the recorded data-scaling ladder (2k/8k/20k/41k at fixed 500 steps) up to 200k distinct
# training queries, using pool_train_big.jsonl (250k rows, built with --train_per_entity 200; its
# corpus_small was verified byte-identical to the shipped one, so pids are compatible).
#
# Everything reads/writes $MMRAG_DATA on the shared bytenas volume, so results land next to the
# local runs and collect_results.py picks them up unchanged.
#
# Cell spec comes from the SCALE_CELL env var: "<name>:<max_train_rows>:<max_steps>".
set -uo pipefail

ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-in
export MMRAG_DATA=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data
export HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/hf_home
export PYTHONPATH=$ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=${TMPDIR:-/mnt/bn/tns-algo-video-public-my2/yijiangli/data/tmp}
mkdir -p "$TMPDIR"
cd "$ROOT"

# spec: <name>:<max_train_rows>:<max_steps>[:<seed>[:<pool_file>]]
IFS=: read -r NAME ROWS STEPS SEED POOL <<< "${SCALE_CELL:?SCALE_CELL not set}"
SEED=${SEED:-0}
POOL=${POOL:-pool_train_big.jsonl}
D=$MMRAG_DATA
echo "=== mm-RAG scale cell $NAME (rows=$ROWS steps=$STEPS seed=$SEED pool=$POOL) start $(date) host=$(hostname) ==="
nvidia-smi -L

# ---- deps. flash-attn is REQUIRED: both reader_vlm.py and src/model/model.py hard-select
# flash_attention_2, and silently falling back to sdpa would make these numbers non-comparable
# to the locally-run points on the same curve. Fail loudly instead.
python3 - <<'PY' || pip install -q torch==2.8.0 transformers==4.57.0 peft==0.17.1 \
    huggingface_hub==0.36.2 accelerate==1.13.0 qwen-vl-utils torchvision pillow
import sys, torch, transformers, peft, qwen_vl_utils  # noqa
mj, mn = (int(x) for x in transformers.__version__.split('.')[:2])
sys.exit(0 if (mj, mn) >= (4, 51) else 1)
PY
python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)" || {
  echo "flash_attn missing — installing"; pip install -q flash-attn==2.8.1 --no-build-isolation; }
python3 -c "import flash_attn" || { echo "FATAL: flash_attn unavailable; aborting rather than \
silently switching to sdpa (would break comparability with the recorded ladder)"; exit 3; }
python3 -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__,
'cuda',torch.cuda.is_available(),torch.cuda.device_count())"

# ---- inputs must exist on the shared volume
for f in "built/$POOL" built/corpus_small.jsonl built/queries_test.jsonl \
         images/Infoseek/infoseek_train_images.tar images/Infoseek/infoseek_val_images.tar; do
  [ -e "$D/$f" ] || { echo "FATAL: missing $D/$f"; exit 4; }
done

RUN=$D/runs/$NAME
if [ -f "$D/results/$NAME.vqa_top5.json" ]; then
  echo "already complete — nothing to do"; exit 0
fi

# ---- train
if [ ! -f "$RUN/adapter_model.safetensors" ]; then
  python3 -u mmrag/train_rl.py \
      --profile gme2b --output_dir "$RUN" \
      --pool "built/$POOL" --corpus built/corpus_small.jsonl \
      --algo grpo --temperature 0.02 --learning_rate 2e-5 \
      --batch_size 4 --num_candidates 8 --max_steps "$STEPS" \
      --lora_r 32 --lora_alpha 64 --epsilon 0.2 \
      --vf_coef 0.0 --beta 0.0 --contrastive_coef 0.3 --contrastive_temperature 0.03 \
      --no_force_gold \
      --reward judge --reader_model Qwen/Qwen2.5-VL-3B-Instruct \
      --reader_rollouts 4 --reader_temperature 0.7 --reader_batch_size 16 \
      --n_distractor_articles 3000 --refresh_steps 100 \
      --encode_bs 64 --max_len_doc 512 --max_train_rows "$ROWS" \
      --save_steps 250 --log_every 25 --seed "$SEED" --device cuda:0 || exit 5
fi

# ---- eval: corpus encode -> L1 retrieval -> L2 VQA (same protocol as every other point)
CACHE=$D/cache/corpus_$NAME.pt
python3 -u $D/encode_corpus.py --checkpoint "$RUN" --cache "$CACHE" --device cuda:0 --bs 192 || exit 6
python3 -u mmrag/eval_retrieval.py --profile gme2b --checkpoint "$RUN" --name "$NAME" \
    --max-q 3000 --device cuda:0 --corpus_bs 192 --cache_corpus "$CACHE" || exit 7
python3 -u mmrag/eval_vqa.py --retrieval "$D/results/$NAME.retrieval.json" \
    --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500 --device cuda:0 --batch_size 8 || exit 8
rm -f "$CACHE"

echo "SCALE_CELL_DONE $NAME $(date)"
grep -hE "acc=" "$D/results/$NAME.vqa_top5.json" 2>/dev/null || true
