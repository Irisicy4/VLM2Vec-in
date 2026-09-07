#!/bin/bash
set -uo pipefail
ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-rl
export MMRAG_DATA=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data
export HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/hf_home
export PYTHONPATH=$ROOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=${TMPDIR:-/mnt/bn/tns-algo-video-public-my2/yijiangli/data/tmp}; mkdir -p "$TMPDIR"; cd "$ROOT"
python3 - <<'PY' || pip install -q torch==2.8.0 transformers==4.57.0 peft==0.17.1 huggingface_hub==0.36.2 accelerate==1.13.0 qwen-vl-utils torchvision pillow
import sys, torch, transformers, peft, qwen_vl_utils
sys.exit(0)
PY
python3 -c "import flash_attn" 2>/dev/null || pip install -q flash-attn==2.8.1 --no-build-isolation
D=$MMRAG_DATA; NAME=gme7b-sft; RUN=$D/runs/$NAME
[ -f "$D/results/$NAME.vqa_top5.json" ] && { echo done; exit 0; }
if [ ! -f "$RUN/adapter_model.safetensors" ]; then
  python3 -u mmrag/train_sft.py --profile gme7b --pool built/pool_train.jsonl \
    --output_dir "$RUN" --learning_rate 1e-4 --num_hard_negs 0 --batch_size 16 \
    --max_steps 600 --save_steps 150 --lora_r 32 --lora_alpha 64 \
    --temperature 0.03 --max_len_doc 512 --seed 0 --device cuda:0 || exit 5
fi
python3 -u mmrag/encode_corpus.py --profile gme7b --checkpoint "$RUN" --cache "$D/cache/$NAME.pt" --device cuda:0 --bs 96 || exit 6
python3 -u mmrag/eval_retrieval.py --profile gme7b --checkpoint "$RUN" --name "$NAME" \
  --max-q 3000 --device cuda:0 --corpus_bs 96 --cache_corpus "$D/cache/$NAME.pt" || exit 7
python3 -u mmrag/eval_vqa.py --retrieval "$D/results/$NAME.retrieval.json" \
  --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500 --device cuda:0 --batch_size 8 || exit 8
rm -f "$D/cache/$NAME.pt"; echo "CELL_DONE gme7b-sft"
