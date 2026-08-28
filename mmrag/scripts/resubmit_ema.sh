#!/bin/bash
D=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data
R=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/VLM2Vec-rl
until grep -q OKVQA_REDL_DONE $D/logs/okvqa_redl.log 2>/dev/null; do sleep 120; done
python3 -c "
import zipfile
z=zipfile.ZipFile('$D/images/OKVQA/OKVQA/train2014.zip' if __import__('os').path.exists('$D/images/OKVQA/OKVQA/train2014.zip') else '$D/images/OKVQA/train2014.zip')
print('zip ok,', len(z.namelist()), 'members')" || { echo ZIP_STILL_BAD; exit 1; }
[ -f $D/images/OKVQA/OKVQA/train2014.zip ] && mv $D/images/OKVQA/OKVQA/train2014.zip $D/images/OKVQA/train2014.zip && rmdir $D/images/OKVQA/OKVQA
# v3-core slice replaces the older subsamples (diversity-first per user directive)
head -n 45000 $D/built/pool_div_1M_v3.jsonl > $D/built/pool_div45k_v3.jsonl
BASE="--pool built/pool_div45k_v3.jsonl --corpus built/corpus_div_v2.jsonl --image_dataset div --max_train_rows 0 --max_steps 500 --no_force_gold --reward judge --algo plgrpo --pl_group 4 --pl_k 4 --pl_support 24 --contrastive_coef 0 --seed 0 --learning_rate 2e-5"
bash $R/mmrag/mlx_submit_cell.sh div45kv3-v3pure gme2b "$BASE"
bash $R/mmrag/mlx_submit_cell.sh div45kv3-emaidx gme2b "$BASE --index_ema 0.9 --index_refresh_extra 256"
bash $R/mmrag/mlx_submit_cell.sh div45kv3-emaenc gme2b "$BASE --index_ema 0.9 --index_refresh_extra 256 --model_ema 0.99"
echo EMA_RESUBMITTED
