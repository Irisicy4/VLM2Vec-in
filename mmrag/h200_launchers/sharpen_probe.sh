#!/bin/bash
# Why does reward stop discriminating as training proceeds?
# HYPOTHESIS: it is the POLICY sharpening, not the reward changing.
# pl_sample_lists is Gumbel-top-k over sims/temperature at tau=0.02. Gumbel noise has scale ~1.28.
# Early, the top-24 cosines are close together, so scaled gaps are comparable to the noise and the
# G=4 sampled lists differ -> different docs -> different rewards -> usable advantage.
# As training sharpens the similarity profile, the scaled gaps grow past the noise, sampling becomes
# near-deterministic, the 4 lists collapse onto the SAME docs -> identical reward -> zero advantage.
# PREDICTION: unique-docs-per-query FALLS and top-24 cosine spread RISES monotonically with steps.
D=/raid/icy/mmrag_data
GPU=${GPU:-6}
for CK in checkpoint-250 checkpoint-750 checkpoint-1500 checkpoint-2250 ""; do
  P="$D/runs/emaidx-1M-s3000/$CK"; TAG=${CK:-final}
  [ -z "$CK" ] && P="$D/runs/emaidx-1M-s3000"
  [ -f "$P/adapter_model.safetensors" ] || { echo "SKIP $TAG"; continue; }
  echo "=== $TAG ==="
  CKPT="$P" NBIG=40000 NQ=16 MMRAG_MEM_FRAC=0.25 CUDA_VISIBLE_DEVICES=$GPU \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $D/venv/bin/python3 $D/bench_uniq.py 2>/dev/null \
    | grep -E "^  index|encoder checkpoint|MEASURED" | sed "s/^/  [$TAG] /"
done
echo "SHARPEN_PROBE_DONE"
