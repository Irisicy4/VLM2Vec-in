#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=02:30:00
#SBATCH -J mmrag_m2b
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/m2b.%j.out
set -o pipefail
SCRATCH=/lus/lfs1aip2/scratch/u6ko/icywang.u6ko
source $SCRATCH/tis_env.sh
cd $SCRATCH/VLM2Vec-in
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MMRAG_DATA=$SCRATCH/mmrag_data
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
L=$MMRAG_DATA/logs
echo "=== node $(hostname) | $(date) ==="

# E-VQA (OOD) zero-shot: gme2b retrieval then VQA over top-5 / no-context
CUDA_VISIBLE_DEVICES=0 python3 -u mmrag/eval_retrieval.py \
  --profile gme2b --dataset evqa --name gme2b-evqa-zeroshot --max-q 3000 \
  --cache_corpus $MMRAG_DATA/results/corpus_evqa_gme2b.pt \
  > $L/m2b_ret_g0.log 2>&1
RET=$MMRAG_DATA/results/gme2b-evqa-zeroshot.retrieval.json
CUDA_VISIBLE_DEVICES=0 python3 -u mmrag/eval_vqa.py --retrieval $RET --dataset evqa --k 5 --max-q 1500 > $L/m2b_vqa_top5_g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 -u mmrag/eval_vqa.py --retrieval $RET --dataset evqa --no-context --max-q 1500 > $L/m2b_vqa_noctx_g1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 -u mmrag/eval_vqa.py --retrieval $RET --dataset evqa --gold-context --k 5 --max-q 1500 > $L/m2b_vqa_gold_g2.log 2>&1 &
wait
echo "=== done $(date) ==="
grep -h "L2 VQA\|entity-level\|answer-level" $L/m2b_*.log
