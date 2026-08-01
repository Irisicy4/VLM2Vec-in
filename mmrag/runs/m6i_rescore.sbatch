#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH -J m6i
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/m6i.%j.out
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
# re-score headline VQA comparisons with the v2 token-boundary metric (+ per-question preds)
CUDA_VISIBLE_DEVICES=0 bash -c '
R=$MMRAG_DATA/results/gme2b-zeroshot.retrieval.json
python3 -u mmrag/eval_vqa.py --retrieval $R --k 5 --max-q 1500 > $L/m6i_vqa_zs_top5_g0.log 2>&1
python3 -u mmrag/eval_vqa.py --retrieval $R --no-context --max-q 1500 > $L/m6i_vqa_zs_noctx_g0.log 2>&1
python3 -u mmrag/eval_vqa.py --retrieval $R --gold-context --k 5 --max-q 1500 > $L/m6i_vqa_zs_gold_g0.log 2>&1' &
CUDA_VISIBLE_DEVICES=1 bash -c '
python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/rl-grpo-judge-lr2e5.retrieval.json --k 5 --max-q 1500 > $L/m6i_vqa_rljudge_g1.log 2>&1
python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/rl-grpo-logit-lr2e5.retrieval.json --k 5 --max-q 1500 > $L/m6i_vqa_rllogit_g1.log 2>&1' &
CUDA_VISIBLE_DEVICES=2 bash -c '
python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/rl-ppo-logit-lr2e5.retrieval.json --k 5 --max-q 1500 > $L/m6i_vqa_rlppo_g2.log 2>&1
python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/rl-grpo-logit-lr5e5.retrieval.json --k 5 --max-q 1500 > $L/m6i_vqa_rl5e5_g2.log 2>&1
python3 -u mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/vlm2vec2b-zeroshot.retrieval.json --k 5 --max-q 1500 > $L/m6i_vqa_vlm2vec_g2.log 2>&1' &
# E-VQA: re-run retrieval (queries now carry section-based ans_pids) then corrected gold/top5/noctx
CUDA_VISIBLE_DEVICES=3 bash -c '
python3 -u mmrag/eval_retrieval.py --profile gme2b --dataset evqa --name gme2b-evqa-zeroshot --max-q 3000 \
  --cache_corpus $MMRAG_DATA/results/corpus_evqa_gme2b.pt > $L/m6i_ret_evqa_g3.log 2>&1
R=$MMRAG_DATA/results/gme2b-evqa-zeroshot.retrieval.json
python3 -u mmrag/eval_vqa.py --retrieval $R --dataset evqa --k 5 --max-q 1500 > $L/m6i_vqa_evqa_top5_g3.log 2>&1
python3 -u mmrag/eval_vqa.py --retrieval $R --dataset evqa --gold-context --k 5 --max-q 1500 > $L/m6i_vqa_evqa_gold_g3.log 2>&1
python3 -u mmrag/eval_vqa.py --retrieval $R --dataset evqa --no-context --max-q 1500 > $L/m6i_vqa_evqa_noctx_g3.log 2>&1' &
wait
echo "=== done $(date) ==="
grep -h "L2 VQA\|entity-level\|answer-level" $L/m6i_*.log
