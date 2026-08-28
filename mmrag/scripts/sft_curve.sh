#!/bin/bash
# SFT max-over-checkpoints: retrain the exact tuned-baseline config with checkpoints every 150
# steps AND run 1200 total (the recorded s1200 cell suggests SFT was still rising at 600).
# Then eval every checkpoint. SFT training is cheap (no reader) — the whole curve in one night.
export HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/hf_home
export MMRAG_DATA=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data
export PYTHONPATH=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-rl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$MMRAG_DATA; cd "$PYTHONPATH"
RUN=$D/runs/sft-curve-lr1e4
if [ ! -f "$RUN/adapter_model.safetensors" ]; then
  python3 -u mmrag/train_sft.py --profile gme2b --pool built/pool_train.jsonl \
      --output_dir "$RUN" --learning_rate 1e-4 --num_hard_negs 0 --batch_size 24 \
      --max_steps 1200 --save_steps 150 --lora_r 32 --lora_alpha 64 \
      --temperature 0.03 --max_len_doc 512 --seed 0 --device cuda:0 || exit 1
fi
for CK in 150 300 450 600 750 900 1050 1200; do
  SRC=$RUN/checkpoint-$CK; [ $CK -eq 1200 ] && SRC=$RUN
  NAME=sft-curve-ck$CK
  [ -f "$D/results/$NAME.vqa_top5.json" ] && continue
  python3 -u mmrag/encode_corpus.py --profile gme2b --checkpoint "$SRC" \
      --cache "$D/cache/$NAME.pt" --device cuda:0 --bs 192 &&
  python3 -u mmrag/eval_retrieval.py --profile gme2b --checkpoint "$SRC" --name "$NAME" \
      --max-q 3000 --device cuda:0 --corpus_bs 192 --cache_corpus "$D/cache/$NAME.pt" &&
  python3 -u mmrag/eval_vqa.py --retrieval "$D/results/$NAME.retrieval.json" \
      --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500 --device cuda:0 --batch_size 8
  rm -f "$D/cache/$NAME.pt"
  echo "SFT_CK_DONE $CK"
done
echo "SFT_CURVE_DONE"
