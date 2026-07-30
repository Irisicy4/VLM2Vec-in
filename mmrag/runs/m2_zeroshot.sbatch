#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH -J mmrag_m2
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/m2.%j.out
set -o pipefail
SCRATCH=/lus/lfs1aip2/scratch/u6ko/icywang.u6ko
source $SCRATCH/tis_env.sh
cd $SCRATCH/VLM2Vec-in
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MMRAG_DATA=$SCRATCH/mmrag_data
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
L=$MMRAG_DATA/logs
echo "=== node $(hostname) | $(date) ==="

# GPU0: zero-shot retrieval (VLM2Vec-2B) on test queries over corpus_small; cache corpus embs
CUDA_VISIBLE_DEVICES=0 python3 -u mmrag/eval_retrieval.py \
  --name vlm2vec2b-zeroshot --max-q 5000 \
  --cache_corpus $MMRAG_DATA/results/corpus_small_vlm2vec2b.pt \
  > $L/m2_retrieval_g0.log 2>&1 &
# GPU1: hard-negative mining for the SFT baseline (train pool; needs train image tar)
CUDA_VISIBLE_DEVICES=1 python3 -u mmrag/mine_hard_negs.py \
  --out built/pool_train_negs.jsonl --num_negs 8 \
  --cache_corpus results/corpus_small_vlm2vec2b_negs.pt \
  > $L/m2_mine_g1.log 2>&1 &
wait
echo "=== retrieval + mining done $(date) ==="

# L2 VQA evals with the 7B reader (needs the retrieval.json from GPU0)
RET=$MMRAG_DATA/results/vlm2vec2b-zeroshot.retrieval.json
CUDA_VISIBLE_DEVICES=0 python3 -u mmrag/eval_vqa.py --retrieval $RET --k 5 --max-q 1500 > $L/m2_vqa_top5_g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 -u mmrag/eval_vqa.py --retrieval $RET --no-context --max-q 1500 > $L/m2_vqa_noctx_g1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 -u mmrag/eval_vqa.py --retrieval $RET --gold-context --k 5 --max-q 1500 > $L/m2_vqa_gold_g2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 -u mmrag/eval_vqa.py --retrieval $RET --k 1 --max-q 1500 > $L/m2_vqa_top1_g3.log 2>&1 &
wait
echo "=== done $(date) ==="
grep -h "L2 VQA\|entity-level\|answer-level" $L/m2_*.log
