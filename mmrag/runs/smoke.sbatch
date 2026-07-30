#!/bin/bash
#SBATCH -p workq
#SBATCH -A brics.u6ko
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:50:00
#SBATCH -J mmrag_smoke
#SBATCH -o /lus/lfs1aip2/scratch/u6ko/icywang.u6ko/mmrag_data/logs/smoke.%j.out
set -o pipefail                       # NEVER set -u (breaks conda init in tis_env.sh)
SCRATCH=/lus/lfs1aip2/scratch/u6ko/icywang.u6ko
source $SCRATCH/tis_env.sh
cd $SCRATCH/VLM2Vec-in
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MMRAG_DATA=$SCRATCH/mmrag_data
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader

# wait (max 20 min) for the val image tar to be fully downloaded (login-node curl may still run)
VAL_TAR=$MMRAG_DATA/images/Infoseek/infoseek_val_images.tar
for i in $(seq 1 40); do
  S1=$(stat -c %s $VAL_TAR 2>/dev/null || echo 0)
  sleep 20
  S2=$(stat -c %s $VAL_TAR 2>/dev/null || echo 0)
  if [ "$S1" = "$S2" ] && [ "$S2" -gt 9000000000 ]; then break; fi
  echo "waiting for val tar ($S2 bytes)..."
done

python3 -u mmrag/smoke_test.py 2>&1
echo "=== smoke_test exit: $? ==="

# build a small val-image-based pool for the RL smoke (train image tar may still be downloading)
python3 - <<'EOF'
import json, os
D = os.environ["MMRAG_DATA"]
qs = [json.loads(l) for l in open(f"{D}/built/queries_test.jsonl")][:800]
need = {(q["ans_pids"] or q["gold_pids"])[0] for q in qs}
pid2text = {}
for line in open(f"{D}/built/corpus_small.jsonl"):
    p = json.loads(line)
    if p["pid"] in need:
        pid2text[p["pid"]] = p["text"]
with open(f"{D}/built/pool_smoke.jsonl", "w") as f:
    for q in qs:
        pid = (q["ans_pids"] or q["gold_pids"])[0]
        if pid not in pid2text: continue
        f.write(json.dumps({"qid": q["qid"], "image_id": q["image_id"], "question": q["question"],
                            "answer": q["answer"], "entity_url": q["entity_url"],
                            "pos": [pid2text[pid]]}) + "\n")
print("pool_smoke written")
EOF

# tiny end-to-end RL loop: 5 steps, small corpus — exercises rollout+reward+backward+save
python3 -u mmrag/train_rl.py \
  --output_dir $MMRAG_DATA/runs/rl-smoke \
  --pool built/pool_smoke.jsonl \
  --image_tars $MMRAG_DATA/images/Infoseek/infoseek_val_images.tar \
  --algo grpo --reward logit \
  --batch_size 2 --num_candidates 6 --max_steps 5 \
  --max_train_rows 300 --n_distractor_articles 100 \
  --refresh_steps 100 --encode_bs 32 --save_steps 0 --log_every 1 2>&1
echo "=== train_rl smoke exit: $? ==="
echo "=== done $(date) ==="
