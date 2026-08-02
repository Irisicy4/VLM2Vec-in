#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH -J m6ab
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/m6ab.%j.out
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
# zero-shot eval on the SAME 3000-q sample as trained runs -> full-n paired tests vs zero-shot
CUDA_VISIBLE_DEVICES=0 bash -c '
python3 -u mmrag/eval_retrieval.py --profile gme2b --name gme2b-zeroshot-3k --max-q 3000 \
  --cache_corpus $MMRAG_DATA/results/corpus_small_gme2b.pt > $L/m6ab_ret_zs3k_g0.log 2>&1
python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/gme2b-zeroshot-3k.retrieval.json \
  --k 5 --max-q 1500 > $L/m6ab_vqa_zs3k_g0.log 2>&1' &
wait
echo "=== done $(date) ==="
grep -h "L2 VQA\|entity-level" $L/m6ab_*.log
