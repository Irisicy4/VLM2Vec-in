#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH -J m6s
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/m6s.%j.out
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
    --refresh_steps 100 --n_distractor_articles 3000 "$@" > $L/m6s_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py --profile ${PROF:-gme2b} --checkpoint $MMRAG_DATA/runs/$name \
    --name $name --max-q 3000 > $L/m6s_eval_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/$name.retrieval.json \
    --k 5 --max-q 1500 > $L/m6s_vqa_${name}_g${g}.log 2>&1
}
sft_cell() { # gpu name extra-args...
  local g=$1 name=$2; shift 2
  POOL=built/pool_train_negs.jsonl; [ -s $MMRAG_DATA/$POOL ] || POOL=built/pool_train.jsonl
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/train_sft.py --pool $POOL --output_dir $MMRAG_DATA/runs/$name \
    --batch_size 24 --max_steps 600 "$@" > $L/m6s_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py --profile ${PROF:-gme2b} --checkpoint $MMRAG_DATA/runs/$name \
    --name $name --max-q 3000 > $L/m6s_eval_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/$name.retrieval.json \
    --k 5 --max-q 1500 > $L/m6s_vqa_${name}_g${g}.log 2>&1
}
rl_cell 0 rl-grpo-judge-lr1e5 --algo grpo --reward judge --reader_rollouts 4 --learning_rate 1e-5 &
rl_cell 1 rl-grpo-judge-lr3e5 --algo grpo --reward judge --reader_rollouts 4 --learning_rate 3e-5 &
rl_cell 2 rl-ppo-judge-lr2e5  --algo ppo  --reward judge --reader_rollouts 4 --learning_rate 2e-5 &
# data-mix experiment (paper finding to test: diversity pays within task structure):
# 50/50 InfoSeek+EVQA-train mix vs EVQA-only control, RL and SFT arms; eval BOTH test sets
for i in $(seq 1 40); do [ -s $MMRAG_DATA/built/pool_mix.jsonl ] && break; echo "waiting for pool_mix..."; sleep 60; done
[ -s $MMRAG_DATA/built/pool_mix.jsonl ] || { echo "SKIP: pool_mix missing"; exit 0; }

rl_cell 0 rl-j2e5-mix  --algo grpo --reward judge --reader_rollouts 4 --learning_rate 2e-5 --pool built/pool_mix.jsonl --corpus built/corpus_mix.jsonl --image_dataset mix &
rl_cell 1 rl-j2e5-evqaonly --algo grpo --reward judge --reader_rollouts 4 --learning_rate 2e-5 --pool built/pool_evqa_train.jsonl --corpus built/corpus_evqa_train.jsonl --image_dataset evqa --n_distractor_articles 0 &
sft_cell 2 sft-lr1e4-mix --learning_rate 1e-4 --num_hard_negs 0 --pool built/pool_mix.jsonl --image_dataset mix &
sft_cell 3 sft-lr1e4-evqaonly --learning_rate 1e-4 --num_hard_negs 0 --pool built/pool_evqa_train.jsonl --image_dataset evqa &
wait
# cross-dataset eval: every trained model also on E-VQA test
g=0
for name in rl-j2e5-mix rl-j2e5-evqaonly sft-lr1e4-mix sft-lr1e4-evqaonly; do
  [ -d $MMRAG_DATA/runs/$name ] && CUDA_VISIBLE_DEVICES=$g bash -c "
    python3 -u mmrag/eval_retrieval.py --profile gme2b --checkpoint $MMRAG_DATA/runs/$name \
      --dataset evqa --name $name-evqa --max-q 3000 > \$L/m6s_ret_${name}_evqa_g$g.log 2>&1
    python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/$name-evqa.retrieval.json \
      --dataset evqa --k 5 --max-q 1500 > \$L/m6s_vqa_${name}_evqa_g$g.log 2>&1" &
  g=$((g+1))
done
wait
echo "=== done $(date) ==="
grep -h "entity-level\|answer-level\|L2 VQA" $L/m6s_*.log
