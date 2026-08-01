#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH -J m6h
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/m6h.%j.out
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
full_eval() { # gpu name checkpoint
  local g=$1 name=$2 ckpt=$3
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py --profile gme2b --checkpoint "$ckpt" \
    --name $name-full --corpus built/corpus_full.jsonl --max-q 3000 \
    > $L/m6h_ret_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/$name-full.retrieval.json \
    --corpus built/corpus_full.jsonl --k 5 --max-q 1500 > $L/m6h_vqa_${name}_g${g}.log 2>&1
}
full_eval 0 gme2b-zs "__profile__" &
full_eval 1 rl-j2e5 $MMRAG_DATA/runs/rl-grpo-judge-lr2e5 &
full_eval 2 sft-lr1e4-h0 $MMRAG_DATA/runs/sft-lr1e4-h0 &
full_eval 3 rl-logit2e5 $MMRAG_DATA/runs/rl-grpo-logit-lr2e5 &
wait
echo "=== done $(date) ==="
grep -h "entity-level\|answer-level\|L2 VQA" $L/m6h_*.log
