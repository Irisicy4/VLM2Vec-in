#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH -J m6e
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/m6e.%j.out
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
    --refresh_steps 100 --n_distractor_articles 3000 "$@" > $L/m6e_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py --profile ${PROF:-gme2b} --checkpoint $MMRAG_DATA/runs/$name \
    --name $name --max-q 3000 > $L/m6e_eval_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/$name.retrieval.json \
    --k 5 --max-q 1500 > $L/m6e_vqa_${name}_g${g}.log 2>&1
}
sft_cell() { # gpu name extra-args...
  local g=$1 name=$2; shift 2
  POOL=built/pool_train_negs.jsonl; [ -s $MMRAG_DATA/$POOL ] || POOL=built/pool_train.jsonl
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/train_sft.py --pool $POOL --output_dir $MMRAG_DATA/runs/$name \
    --batch_size 24 --max_steps 600 "$@" > $L/m6e_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py --profile ${PROF:-gme2b} --checkpoint $MMRAG_DATA/runs/$name \
    --name $name --max-q 3000 > $L/m6e_eval_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/$name.retrieval.json \
    --k 5 --max-q 1500 > $L/m6e_vqa_${name}_g${g}.log 2>&1
}
# capacity ladder top rung: GME-7B (strong encoder -> even lower LRs; slower corpus refresh)
CUDA_VISIBLE_DEVICES=0 bash -c '
python3 -u mmrag/eval_retrieval.py --profile gme7b --name gme7b-zeroshot --max-q 3000 \
  --corpus_bs 32 --cache_corpus $MMRAG_DATA/results/corpus_small_gme7b.pt > $L/m6e_zs_gme7b_g0.log 2>&1
python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/gme7b-zeroshot.retrieval.json \
  --k 5 --max-q 1500 > $L/m6e_vqa_gme7b-zeroshot_g0.log 2>&1' &
PROF=gme7b rl_cell 1 rl7b-j-lr1e5 --profile gme7b --algo grpo --reward judge --reader_rollouts 4 --learning_rate 1e-5 --refresh_steps 250 --encode_bs 32 --reader_batch_size 8 &
PROF=gme7b rl_cell 2 rl7b-j-lr5e6 --profile gme7b --algo grpo --reward judge --reader_rollouts 4 --learning_rate 5e-6 --refresh_steps 250 --encode_bs 32 --reader_batch_size 8 &
PROF=gme7b sft_cell 3 sft7b-lr5e5-h0 --profile gme7b --learning_rate 5e-5 --num_hard_negs 0 --batch_size 12 &
wait
echo "=== done $(date) ==="
grep -h "entity-level\|answer-level\|L2 VQA" $L/m6e_*.log
