#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH -J mmrag_m5
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/m5.%j.out
set -o pipefail
SCRATCH=/lus/lfs1aip2/scratch/u6ko/icywang.u6ko
source $SCRATCH/tis_env.sh
cd $SCRATCH/VLM2Vec-in
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MMRAG_DATA=$SCRATCH/mmrag_data
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
L=$MMRAG_DATA/logs
echo "=== node $(hostname) | $(date) ==="

# Capacity-ladder weak rungs: CLIP-L-336 and SigLIP2-so400m dual-tower encoders.
# Per GPU: zero-shot eval -> train (RL or SFT) -> eval. Weak encoders want HIGH LR (text finding).
run_rl() { # gpu profile lr
  local g=$1 p=$2 lr=$3 name=$2-rl-grpo-judge-lr$3
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py --profile $p --name $p-zeroshot \
    --max-q 3000 --cache_corpus $MMRAG_DATA/results/corpus_small_$p.pt > $L/m5_zs_${p}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/train_rl.py --profile $p \
    --output_dir $MMRAG_DATA/runs/$name --algo grpo --reward judge --reader_rollouts 4 \
    --learning_rate $lr --batch_size 4 --num_candidates 8 --max_steps 500 --save_steps 100 \
    --refresh_steps 100 --n_distractor_articles 3000 --encode_bs 256 > $L/m5_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py --profile $p \
    --checkpoint $MMRAG_DATA/runs/$name --name $name --max-q 3000 > $L/m5_eval_${name}_g${g}.log 2>&1
}
run_sft() { # gpu profile lr
  local g=$1 p=$2 lr=$3 name=$2-sft-lr$3
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/train_sft.py --profile $p \
    --output_dir $MMRAG_DATA/runs/$name --learning_rate $lr --batch_size 64 --max_steps 600 \
    > $L/m5_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py --profile $p \
    --checkpoint $MMRAG_DATA/runs/$name --name $name --max-q 3000 > $L/m5_eval_${name}_g${g}.log 2>&1
}
run_rl 0 clip 1e-4 &
run_sft 1 clip 2e-4 &
run_rl 2 siglip2 1e-4 &
run_sft 3 siglip2 2e-4 &
wait
echo "=== done $(date) ==="
grep -h "entity-level\|answer-level" $L/m5_*_g*.log
