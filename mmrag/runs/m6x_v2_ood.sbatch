#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH -J m6x
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/m6x.%j.out
set -o pipefail
SCRATCH=/lus/lfs1aip2/scratch/u6ko/icywang.u6ko
source $SCRATCH/tis_env.sh
cd $SCRATCH/VLM2Vec-in
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MMRAG_DATA=$SCRATCH/mmrag_data
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export L=$MMRAG_DATA/logs
echo "=== node $(hostname) | $(date) ==="
export L=$MMRAG_DATA/logs
CUDA_VISIBLE_DEVICES=0 python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/rl-j2e5-evqa.retrieval.json --dataset evqa --k 5 --max-q 1500 > $L/m6x_vqa_goldrl_evqa_g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/sft-lr1e4-evqa.retrieval.json --dataset evqa --k 5 --max-q 1500 > $L/m6x_vqa_sft_evqa_g1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/gme2b-evqa-zeroshot.retrieval.json --dataset evqa --gold-context --k 5 --max-q 1500 > $L/m6x_vqa_zs_gold_evqa_g2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/rl-ragacc-nogold-lr1e5.retrieval.json --k 5 --max-q 1500 > $L/m6x_vqa_ragacc1e5_g3.log 2>&1 &
wait
echo "=== done $(date) ==="
grep -h "L2 VQA" $L/m6x_*.log
