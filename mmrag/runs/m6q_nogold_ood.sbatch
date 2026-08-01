#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH -J m6q
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/m6q.%j.out
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
ood() { # gpu name ckptdir
  local g=$1 name=$2 ckpt=$3
  [ -d "$ckpt" ] || { echo "SKIP $name (no ckpt)"; return; }
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py --profile gme2b --checkpoint $ckpt \
    --dataset evqa --name $name-evqa --max-q 3000 --cache_corpus $MMRAG_DATA/results/corpus_evqa_gme2b_$name.pt \
    > $L/m6q_ret_${name}_g${g}.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/$name-evqa.retrieval.json \
    --dataset evqa --k 5 --max-q 1500 > $L/m6q_vqa_${name}_g${g}.log 2>&1
}
ood 0 rl-j2e5-nogold-det $MMRAG_DATA/runs/rl-j2e5-nogold-det &
ood 1 rl-j2e5-nogold-det-N16 $MMRAG_DATA/runs/rl-j2e5-nogold-det-N16 &
ood 2 rl-ragacc-nogold-lr2e5 $MMRAG_DATA/runs/rl-ragacc-nogold-lr2e5 &
ood 3 rl-judge-nogold-frozen $MMRAG_DATA/runs/rl-judge-nogold-frozen &
wait
echo "=== done $(date) ==="
grep -h "entity-level\|answer-level\|L2 VQA\|SKIP" $L/m6q_*.log
