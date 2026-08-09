# mm-RAG reproduction on the new cluster (2026-08-07)

Everything below was rebuilt from public sources on `n116-042-202` (8×H100 80GB); the old
Isambard-AI scratch (`$SCRATCH/mmrag_data`) was lost on transfer and was **not** needed.

## 0. What was rebuilt

- `MMRAG_DATA=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data`
- Raw: EchoSight `infoseek_{train,test}_filtered.csv` + `wiki_100_dict_v4.json` (Dropbox),
  official `infoseek_val.jsonl` (GCS), M2KR `infoseek_{val,train}_images.tar` (HF, 9+48GB,
  read in place via `image_store.py`; train tar indexed = 770,602 members).
- `build_infoseek_data.py --train_per_entity 10 --max_train 60000 --distractor_articles 20000`

Build output matches the recorded protocol **exactly**, which is what makes the numbers below
comparable:

| quantity | RESULTS.md / README | rebuild |
|---|---|---|
| corpus_full passages | 1,421,665 | 1,421,665 |
| corpus_small passages / articles | 422,378 / 26,454 | 422,378 / 26,454 |
| queries_train | 45,248 | 45,248 |
| queries_test | 71,335 | 71,335 |
| answer-level eval subset | 2,450 | 2,450 |

## 1. Released-checkpoint reproduction (InfoSeek, aligned protocol)

Protocol: L1 = 3,000 seed-0 test queries over corpus_small; L2 = 1,500 seed-0 subsample, top-5,
reader Qwen2.5-VL-7B, v2 token-boundary cover-EM.

| run | metric | recorded | repro | Δ |
|---|---|---|---|---|
| gme2b-zeroshot-3k | VQA acc | 0.3100 | 0.3080 | −0.0020 |
| | entity R@5 | 0.6843 | 0.6853 | +0.0010 |
| sft-lr1e4-h0 | VQA acc | 0.3100 | **0.3100** | 0.0000 |
| | strictEM / F1 | 0.2387 / 0.3213 | 0.2387 / 0.3213 | 0.0000 |
| | entity R@1/5/10 | 0.5890 / 0.7373 / 0.8003 | 0.5913 / 0.7363 / 0.7983 | ≤0.002 |
| **rl-j2e5-nogold-det (s0)** | **VQA acc** | **0.3413** | **0.3393** | −0.0020 |
| | entity R@1/5/10 | 0.5907 / 0.7960 / 0.8607 | 0.5920 / 0.7977 / 0.8593 | ≤0.002 |
| | answer R@5 | 0.6351 | 0.6371 | +0.0020 |
| **rl-j2e5-nogold-det-s3** | **VQA acc** | **0.3427** | **0.3433** | +0.0006 |
| | strictEM / F1 | 0.2567 / 0.3415 | 0.2580 / 0.3426 | ≤0.002 |
| | entity R@1/5 | 0.5737 / 0.7747 | 0.5737 / 0.7763 | ≤0.002 |

**Headline effect reproduces exactly**: RL − zero-shot = +0.0313 recorded, +0.0313 repro.
The qualitative claim holds too — no-gold RL (0.339/0.343) ≫ tuned SFT (0.310) ≈ zero-shot
(0.308), i.e. SFT-vs-zero-shot remains the non-significant comparison.

Residual −0.002 on two arms is fp16 matmul / top-k tie nondeterminism on a freshly built index,
not a protocol difference (the SFT arm came out bit-exact on all three L2 metrics).

## 2. MMEB(-V2) image-task subset — new control

Question: does annotation-free RL on one knowledge-VQA corpus damage general multimodal
embedding quality? Subset: 3,700 queries (100/task × 37 tasks, seed 0), **full per-task
candidate pools retained** so difficulty is unchanged; MMEB Precision@1; identical pools for
both arms. Saved at `/mnt/.../data/mmeb_eval/subset_seed0/`.

| | macro P@1 | micro P@1 (n=3700) |
|---|---|---|
| gme2b zero-shot | 0.4381 | 0.4381 |
| + no-gold RL adapter | **0.4468** | **0.4468** |
| Δ | **+0.0087** | +0.0086 |

Better on 25 tasks / worse on 9 / tied 3 — sign test **p = 0.009**.

- Gains: TextVQA +0.06, Visual7W-Pointing +0.05, SUN397 +0.05, ScienceQA +0.04, Place365 +0.04,
  ImageNet-1K +0.04, A-OKVQA +0.03, ImageNet-R +0.03, OVEN/EDIS/VisDial/ChartQA +0.02.
- Losses: ImageNet-A −0.07, MSCOCO_t2i −0.04, NIGHTS −0.04, RefCOCO-Matching −0.03.

Reading: the adapter tilts the encoder toward image→text knowledge retrieval (helping
classification/knowledge tasks) at a small cost on pure visual-similarity / composed image
retrieval. No catastrophic forgetting. Caveat: at 100 queries/task a ±0.01 per-task delta is one
query — trust the aggregate and the sign test, not individual rows.

Re-score any other checkpoint on the identical subset:

    python3 $MMRAG_DATA/mmeb_subset.py eval --subset .../subset_seed0 \
        --checkpoint <adapter_dir|__profile__> --name <label> --device cuda:N

## 3. From-scratch retrain (running)

4 seeds of the headline cell, config copied verbatim from the released checkpoint's `args.json`
(GRPO, judge reward, lr 2e-5, batch 4, N=8, 500 steps, `--no_force_gold`, cc 0.3, 4 rollouts of
Qwen2.5-VL-3B). Step 0 matches the original `metrics.jsonl` on the retrieval-side quantities
(reward/gold_mean 0.750, gold_in_topN 0.75); reward/InfoNCE differ slightly because the
temperature-0.7 reader rollouts are not bit-reproducible across hardware.
Results land in `runs/rl-j2e5-nogold-det-repro-s{0..3}` and `eval_retrain_s*.log`.

## 4. Gotchas hit during the rebuild (for the next person)

- `hf_hub_download` has no `max_workers` kwarg (only `snapshot_download` does).
- Corpus encoding at `--corpus_bs 64` runs ~50 passages/s; at 192–256 it is ~300/s. The default
  in the eval scripts is 64 — raise it.
- MMEB `images.zip` is 285K small files; extracting onto this NFS runs at ~1GB/10min. Read the
  images in place from the zip's central directory instead (as `image_store.ZipImageStore` does).
- MMEB queries must be encoded with `max_length ≥ 2048`. At 1024 the image pads get truncated and
  the vendored `get_rope_index` throws a shape mismatch — the same trap `encoder.py`'s
  `query_max_len=2048` default already documents.
- Pre-build the 48GB train-tar index once (`image_store.build_tar_index`, ~7.5 min) or every
  training process rescans it.
