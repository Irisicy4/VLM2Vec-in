#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH -J mmrag_m3
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/m3.%j.out
set -o pipefail
SCRATCH=/lus/lfs1aip2/scratch/u6ko/icywang.u6ko
source $SCRATCH/tis_env.sh
cd $SCRATCH/VLM2Vec-in
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MMRAG_DATA=$SCRATCH/mmrag_data
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
L=$MMRAG_DATA/logs
echo "=== node $(hostname) | $(date) ==="

# wait (max 90 min) for the train image tar (login-node download may still be finishing)
TT=$MMRAG_DATA/images/Infoseek/infoseek_train_images.tar
for i in $(seq 1 90); do [ -f $TT ] && break; echo "waiting for train tar..."; sleep 60; done

POOL=built/pool_train_negs.jsonl
[ -s $MMRAG_DATA/$POOL ] || POOL=built/pool_train.jsonl   # -s: a crashed miner must not leave an empty decoy
echo "pool: $POOL"

# 4-way LR x hard-neg sweep, one per GPU (own log each — no shared names)
run() { # gpu name extra-args...
  local g=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/train_sft.py --pool $POOL \
    --output_dir $MMRAG_DATA/runs/$name --max_steps 600 --batch_size 24 "$@" \
    > $L/m3_${name}_g${g}.log 2>&1
  # eval retrieval right after training on the same GPU
  CUDA_VISIBLE_DEVICES=$g python3 -u mmrag/eval_retrieval.py \
    --checkpoint $MMRAG_DATA/runs/$name --name $name --max-q 3000 \
    > $L/m3_eval_${name}_g${g}.log 2>&1
}
run 0 sft-lr1e4-h2 --learning_rate 1e-4 --num_hard_negs 2 &
run 1 sft-lr5e5-h2 --learning_rate 5e-5 --num_hard_negs 2 &
run 2 sft-lr2e4-h2 --learning_rate 2e-4 --num_hard_negs 2 &
run 3 sft-lr1e4-h0 --learning_rate 1e-4 --num_hard_negs 0 &
wait
echo "=== done $(date) ==="
grep -h "entity-level\|answer-level" $L/m3_eval_*_g*.log
