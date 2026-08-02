#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH -J m6z
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/m6z.%j.out
set -o pipefail
SCRATCH=/lus/lfs1aip2/scratch/u6ko/icywang.u6ko
source $SCRATCH/tis_env.sh
cd $SCRATCH/VLM2Vec-in
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MMRAG_DATA=$SCRATCH/mmrag_data
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export L=$MMRAG_DATA/logs
echo "=== node $(hostname) | $(date) ==="

# RL cell: train -> retrieval eval -> VQA eval, all on one GPU
rl_cell() { # gpu name extra-args...
  local g=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/train_rl.py --output_dir $MMRAG_DATA/runs/$name \
    --batch_size 4 --num_candidates 8 --max_steps 500 --save_steps 100 \
    --refresh_steps 100 --n_distractor_articles 3000 "$@" > $L/m6z_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py --profile ${PROF:-gme2b} --checkpoint $MMRAG_DATA/runs/$name \
    --name $name --max-q 3000 > $L/m6z_eval_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/$name.retrieval.json \
    --k 5 --max-q 1500 > $L/m6z_vqa_${name}_g${g}.log 2>&1
}
sft_cell() { # gpu name extra-args...
  local g=$1 name=$2; shift 2
  POOL=built/pool_train_negs.jsonl; [ -s $MMRAG_DATA/$POOL ] || POOL=built/pool_train.jsonl
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/train_sft.py --pool $POOL --output_dir $MMRAG_DATA/runs/$name \
    --batch_size 24 --max_steps 600 "$@" > $L/m6z_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py --profile ${PROF:-gme2b} --checkpoint $MMRAG_DATA/runs/$name \
    --name $name --max-q 3000 > $L/m6z_eval_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/$name.retrieval.json \
    --k 5 --max-q 1500 > $L/m6z_vqa_${name}_g${g}.log 2>&1
}
rl_cell 0 rl-grpo-judge-lr1e5 --algo grpo --reward judge --reader_rollouts 4 --learning_rate 1e-5 &
rl_cell 1 rl-grpo-judge-lr3e5 --algo grpo --reward judge --reader_rollouts 4 --learning_rate 3e-5 &
rl_cell 2 rl-ppo-judge-lr2e5  --algo ppo  --reward judge --reader_rollouts 4 --learning_rate 2e-5 &
# seeds for the BEST SFT config (lr2e-4 h0) + fill the s1 no-gold eval if its VQA is missing
sft_cell 0 sft-lr2e4-h0-s1 --learning_rate 2e-4 --num_hard_negs 0 --seed 1 &
sft_cell 1 sft-lr2e4-h0-s2 --learning_rate 2e-4 --num_hard_negs 0 --seed 2 &
sft_cell 2 sft-lr2e4-h0-s3 --learning_rate 2e-4 --num_hard_negs 0 --seed 3 &
CUDA_VISIBLE_DEVICES=3 bash -c '
python3 -u mmrag/eval_retrieval.py --profile gme2b --checkpoint $MMRAG_DATA/runs/rl-j2e5-nogold-det-s1 \
  --name rl-j2e5-nogold-det-s1 --max-q 3000 > $L/m6z_ret_nogold-s1_g3.log 2>&1
python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/rl-j2e5-nogold-det-s1.retrieval.json \
  --k 5 --max-q 1500 > $L/m6z_vqa_nogold-s1_g3.log 2>&1' &
wait
echo "=== done $(date) ==="
grep -h "entity-level\|answer-level\|L2 VQA" $L/m6z_*.log
