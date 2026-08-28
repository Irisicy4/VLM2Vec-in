#!/bin/bash
D=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data
P=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-rl
# wait for the RL arms to land (GPU 0 frees when div45kv3-v3pure's eval completes)
until [ -f $D/results/div45kv3-v3pure.vqa_top5.json ]; do sleep 600; done
cd $P
MMRAG_DATA=$D PYTHONPATH=$P HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u mmrag/train_sft.py --profile gme2b --pool built/pool_div45k_v3.jsonl --image_dataset div \
  --batch_size 24 --learning_rate 1e-4 --max_steps 1200 --save_steps 150 --seed 0 \
  --output_dir $D/runs/sft-div45k --device cuda:0 || exit 1
for CK in 450 600 750 900 1050 1200; do
  CKD=$D/runs/sft-div45k/checkpoint-$CK; RN=sft-div45k-ck$CK
  [ -d "$CKD" ] || continue
  CACHE=$D/cache/corpus_$RN.pt
  MMRAG_DATA=$D PYTHONPATH=$P HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/hf_home python3 -u mmrag/encode_corpus.py --profile gme2b --checkpoint "$CKD" --cache "$CACHE" --device cuda:0 --bs 128 &&
  MMRAG_DATA=$D PYTHONPATH=$P HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/hf_home python3 -u mmrag/eval_retrieval.py --profile gme2b --checkpoint "$CKD" --name "$RN" --max-q 3000 --device cuda:0 --corpus_bs 128 --cache_corpus "$CACHE" &&
  MMRAG_DATA=$D PYTHONPATH=$P HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/hf_home python3 -u mmrag/eval_vqa.py --retrieval "$D/results/$RN.retrieval.json" --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500 --device cuda:0 --batch_size 8
  rm -f "$CACHE"
done
echo SFT_DIV_DONE
