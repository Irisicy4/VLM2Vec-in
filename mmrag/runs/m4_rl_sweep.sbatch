#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH -J mmrag_m4
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/m4.%j.out
set -o pipefail
SCRATCH=/lus/lfs1aip2/scratch/u6ko/icywang.u6ko
source $SCRATCH/tis_env.sh
cd $SCRATCH/VLM2Vec-in
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MMRAG_DATA=$SCRATCH/mmrag_data
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
L=$MMRAG_DATA/logs
echo "=== node $(hostname) | $(date) ==="

# 4-way sweep, one config per GPU (encoder + reward reader share the GPU).
# LR is the key knob (text finding). 500 steps, checkpoints every 100 (walltime lesson).
run() { # gpu name extra-args...
  local g=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/train_rl.py \
    --output_dir $MMRAG_DATA/runs/$name \
    --batch_size 4 --num_candidates 8 --max_steps 500 --save_steps 100 \
    --refresh_steps 100 --n_distractor_articles 3000 "$@" \
    > $L/m4_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py \
    --checkpoint $MMRAG_DATA/runs/$name --name $name --max-q 3000 \
    > $L/m4_eval_${name}_g${g}.log 2>&1
}
run 0 rl-grpo-logit-lr2e5 --algo grpo --reward logit --learning_rate 2e-5 &
run 1 rl-grpo-logit-lr5e5 --algo grpo --reward logit --learning_rate 5e-5 &
run 2 rl-grpo-judge-lr2e5 --algo grpo --reward judge --reader_rollouts 4 --learning_rate 2e-5 &
run 3 rl-ppo-logit-lr2e5  --algo ppo  --reward logit --learning_rate 2e-5 &
wait
echo "=== done $(date) ==="
grep -h "entity-level\|answer-level" $L/m4_eval_*_g*.log
