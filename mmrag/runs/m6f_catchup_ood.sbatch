#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH -J m6f
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/m6f.%j.out
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
    --refresh_steps 100 --n_distractor_articles 3000 "$@" > $L/m6f_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py --profile ${PROF:-gme2b} --checkpoint $MMRAG_DATA/runs/$name \
    --name $name --max-q 3000 > $L/m6f_eval_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/$name.retrieval.json \
    --k 5 --max-q 1500 > $L/m6f_vqa_${name}_g${g}.log 2>&1
}
sft_cell() { # gpu name extra-args...
  local g=$1 name=$2; shift 2
  POOL=built/pool_train_negs.jsonl; [ -s $MMRAG_DATA/$POOL ] || POOL=built/pool_train.jsonl
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/train_sft.py --pool $POOL --output_dir $MMRAG_DATA/runs/$name \
    --batch_size 24 --max_steps 600 "$@" > $L/m6f_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py --profile ${PROF:-gme2b} --checkpoint $MMRAG_DATA/runs/$name \
    --name $name --max-q 3000 > $L/m6f_eval_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/$name.retrieval.json \
    --k 5 --max-q 1500 > $L/m6f_vqa_${name}_g${g}.log 2>&1
}
# catch-up VQA for the SFT arm + OOD transfer of the best RL checkpoint to E-VQA
CUDA_VISIBLE_DEVICES=0 python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/sft-lr1e4-h0.retrieval.json --k 5 --max-q 1500 > $L/m6f_vqa_sft-lr1e4-h0_g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/clip-zeroshot.retrieval.json --k 5 --max-q 1500 > $L/m6f_vqa_clip-zs_g1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash -c '
python3 -u mmrag/eval_retrieval.py --profile gme2b --checkpoint $MMRAG_DATA/runs/rl-grpo-judge-lr2e5 \
  --dataset evqa --name rl-j2e5-evqa --max-q 3000 > $L/m6f_ret_rl-evqa_g2.log 2>&1
python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/rl-j2e5-evqa.retrieval.json \
  --dataset evqa --k 5 --max-q 1500 > $L/m6f_vqa_rl-evqa_g2.log 2>&1' &
CUDA_VISIBLE_DEVICES=3 bash -c '
python3 -u mmrag/eval_retrieval.py --profile gme2b --checkpoint $MMRAG_DATA/runs/sft-lr1e4-h0 \
  --dataset evqa --name sft-lr1e4-evqa --max-q 3000 > $L/m6f_ret_sft-evqa_g3.log 2>&1
python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/sft-lr1e4-evqa.retrieval.json \
  --dataset evqa --k 5 --max-q 1500 > $L/m6f_vqa_sft-evqa_g3.log 2>&1' &
wait
echo "=== done $(date) ==="
grep -h "entity-level\|answer-level\|L2 VQA" $L/m6f_*.log
