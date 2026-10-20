# H200 harvest index (branch h200-raid, 2026-10-20)

Results: private HF dataset https://huggingface.co/datasets/Icey444/h200-results-VLM2Vec-in (`mmrag_data/results`, `mmrag_data/logs`, `mmrag_data/runs`).
Launchers copied from /raid/icy/mmrag_data -> `mmrag/h200_launchers/`. Paper tables -> `mmrag/docs/tab_*.tex`. Memory/skills -> `.claude/`.

**INCOMPLETE:** harvest was cut off by a usage limit. The version index (tagv) and the number maps for "The Indirect Reward" and Sept artifacts are not yet included; the fragments, if finished, are in /raid/icy/tmp/claude-1013/-raid-icy-VLM2Vec-in/e138b2ce-b664-42d9-b18c-e2a9f2094798/scratchpad/frag_*.md. Headline seeded runs live on the MLX cluster (/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data) — not on this box.

## Artifact harvest: frag_aug (5 artifacts, read 2026-10-20)

Out of scope (other project, skipped):
- **The No-Gold Retriever Loop** (https://claude.ai/artifact/2TsqHCsgn2jSgMZ2G33p2E): other project. It is the *text twin* (Qwen3-Embedding-0.6B, Qwen2.5-3B text reader, NQ/TriviaQA, BIGRET 2M). That is the text-retrieval collaborator's work, not mmrag.
- **Annotation-Free Retrieval on DMIX** (https://claude.ai/artifact/57riG7FC5bwRnBtnW51mYT): other project. Text DMIX corpus, `trl-projects` repo, branch `cluster-move-ema`, `projects/indirect/rag/*`.
- **Retrieval Scaling Axes** (https://claude.ai/artifact/KdSgcNd5Vy5DP9KSjbtojX): other project. Text BIGRET 2M-passage corpus (Trivia/NQ/SQuAD/…), the same text-twin lane.
- **Does Agreement Predict Correctness** (https://claude.ai/artifact/FzFSDyvBSHBVyPsg4AAkdN): other project. Qwen3-8B pseudo-label calibration on SciKnowEval chemistry (the judge/pseudo-label line).

### Artifact: The VQA Retriever Loop (https://claude.ai/artifact/UigS2G1erhmbGSVckbPJqZ)

Conventions:
- `$D`: the artifact's commands use `$D=/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mmrag_data` (the MLX cluster). The local mirror is /raid/icy/mmrag_data.
- HF prefix `HF/` = https://huggingface.co/datasets/Icey444/h200-results-VLM2Vec-in/blob/main/mmrag_data/results/
- Output file path prefix `R/` = /raid/icy/mmrag_data/results/
- Metrics files:
  - R@5 values come from `R/<name>.retrieval.metrics.json` (`entity_recall.5`, `answer_recall.5`).
  - Every InfoSeek "VQA acc" in this artifact equals `acc` in `*.vqa_top5.json`, which is the **broken metric**. It also equals `acc_orig` in `*.vqa_rescored.json`. The corrected value (`acc_rescored_any`) is given in brackets as **resc=**. Orderings are unchanged; levels shift by about +8.5.
  - E-VQA is unaffected: rescored = top5.
- Run directories: none of these runs has a run dir on this box. /raid/icy/mmrag_data/runs holds only 30 other runs (mix2m-*, divmix-*, sft-rows*, …). The args.json and adapter for each run are on the remote MLX cluster at `$D/runs/<name>`.
- Code base:
  - Training: /raid/icy/VLM2Vec-in/mmrag/train_rl.py, launched through /raid/icy/VLM2Vec-in/mmrag/mlx_entry_cell.sh.
  - Eval chain: /raid/icy/VLM2Vec-in/mmrag/encode_corpus.py → eval_retrieval.py → eval_vqa.py.
  - Rescoring: /raid/icy/VLM2Vec-in/mmrag/rescore.py.
  - The artifact's own source copy is /raid/icy/VLM2Vec-in/mmrag/docs/vqa_retriever_loop.html.

| # | claim / table cell | value | script (full path) | config / args | output file (full path) | HF link | status |
|---|---|---|---|---|---|---|---|
| 1 | T1 v3-pure GME-2B entR@5 / ansR@5 / acc (3 seeds) | 76.09 / 63.46 / 34.71±0.19 (seeds 75.27/63.59/34.47 · 75.77/62.45/34.73 · 77.23/64.33/34.93) **resc=43.44±0.34** (43.07/43.73/43.53) | /raid/icy/VLM2Vec-in/mmrag/train_rl.py via mlx_entry_cell.sh | /raid/icy/VLM2Vec-in/mmrag/mlx_configs/v3-pure{,-s1,-s2}.yaml; plgrpo G4 k4 sup24, cc0, lr2e-5, 500 steps; run dir on MLX only | R/v3-pure{,-s1,-s2}.{retrieval.metrics,vqa_top5,vqa_rescored}.json | HF/v3-pure.vqa_rescored.json (and -s1, -s2) | verified (acc on broken metric) |
| 2 | T1 v2 = v3+anchor (seeds 0/2/3) | 80.13 / 65.46 / 33.82±0.60 **resc=42.00±0.87** | same train_rl.py | `--contrastive_coef 0.3`; runs plgrpo-gme2b-nogold, plgrpo-s2, plgrpo-s3 (no yaml; local_cell_worker.sh) | R/plgrpo-gme2b-nogold.*, R/plgrpo-s2.*, R/plgrpo-s3.* | HF/plgrpo-gme2b-nogold.vqa_rescored.json etc. | verified |
| 3 | v2 seed 1 not in mean; 4-seed acc 33.60±0.70 | 78.73/61.55/32.93 (resc 39.67) | same | plgrpo-gme2b-nogold-s1 | R/plgrpo-gme2b-nogold-s1.* | HF/plgrpo-gme2b-nogold-s1.vqa_rescored.json | verified |
| 4 | T1 Rel-SFT acc, per-seed best ckpt | 32.85±0.90 (s0 ck750 33.00 · s1 ck1050 33.67 · s2 ck1050 31.87). **resc** per-seed max: 41.13 (ck750) / 42.73 (ck1050) / 40.67 (ck600) → 41.51 | /raid/icy/VLM2Vec-in/mmrag/train_sft.py; /raid/icy/VLM2Vec-in/mmrag/scripts/sft_curve.sh | lr1e-4, bs24, 1200 steps, save 150, no hard negs; runs sft-curve-lr1e4{,-s1,-s2} (MLX) | R/sft-curve-ck750.*, R/sft-curve-lr1e4-s1-ck1050.*, R/sft-curve-lr1e4-s2-ck1050.* | HF/sft-curve-ck750.vqa_rescored.json etc. | verified (on rescored, the s2 best ckpt shifts to ck600) |
| 5 | T1 Rel-SFT entR@5 / ansR@5 | 73.63 / 59.67 (acc 31.00, resc 39.20) | train_sft.py | 600-step released config, sft-lr1e4-h0-repro; the run dir was not retained | R/sft-lr1e4-h0-repro.* | HF/sft-lr1e4-h0-repro.retrieval.metrics.json | verified (mixed sources: R@5 from a different run than acc) |
| 6 | T1 frozen base GME-2B | 68.53 / 54.94 / 30.80 **resc=38.40** | /raid/icy/VLM2Vec-in/mmrag/scripts/run_repro.sh zs | `--checkpoint __profile__` | R/gme2b-zeroshot-3k-repro.* | HF/gme2b-zeroshot-3k-repro.vqa_rescored.json | verified |
| 7 | Deltas vs base | +7.56 entR@5 / +8.52 ansR@5 / +3.91 acc (**resc +5.04**) | arithmetic on rows 1 and 6 | — | — | — | verified |
| 8 | McNemar vs base, every seed | p ≤ 4.2e-6 | /raid/icy/VLM2Vec-in/mmrag/significance.py | `--pairs v3-pure{,-s1,-s2}:gme2b-zeroshot-3k-repro` | stdout only, re-run 2026-10-20: p=4.2e-6 / 3.2e-7 / 5.3e-7 | — | verified on the old metric; paper.md:253 quotes p ≤ 4.8e-05 (likely rescored) |
| 9 | RL − SFT margin | +1.86 ≈ 3.5 SE (**resc +1.93**: 43.44 vs 41.51) | arithmetic on rows 1 and 4 | — | — | — | verified |
| 10 | T2 GME-7B v3-pure (2 seeds) | 80.97/69.02/34.60 · 80.30/68.41/35.73 (**resc 45.00 / 45.47**) | train_rl.py | /raid/icy/VLM2Vec-in/mmrag/mlx_configs/gme7b-v3pure{,-s1}.yaml; lr1e-5, encode_bs32, reader_bs8 | R/gme7b-v3pure{,-s1}.* | HF/gme7b-v3pure.vqa_rescored.json | verified |
| 11 | T2 GME-7B Rel-SFT | 77.67 / 63.02 / 33.20 (**resc 41.60**) | /raid/icy/VLM2Vec-in/mmrag/mlx_entry_sft7b.sh → train_sft.py | mlx_configs/gme7b-sft.yaml; bs16, 600 steps | R/gme7b-sft.* | HF/gme7b-sft.vqa_rescored.json | verified |
| 12 | T2 GME-7B base | 69.37 / 53.59 / 30.67 (**resc 39.67**) | /raid/icy/VLM2Vec-in/mmrag/scripts/ladder_zs.sh gme7b | __profile__ | R/gme7b-zeroshot.* | HF/gme7b-zeroshot.vqa_rescored.json | verified |
| 13 | T3 BGE-VL-large lr1e-5 vs zs | 21.23 (zs 15.73) / 10.24 (7.10) / 16.93 (15.87) [**resc 20.00 vs 17.73**] | train_rl.py; ladder_zs.sh | mlx_configs/bgevl_l-v3pure-lr1e5.yaml | R/bgevl_l-v3pure-lr1e5.*, R/bgevl_l-zeroshot.* | HF/bgevl_l-v3pure-lr1e5.retrieval.metrics.json | verified |
| 14 | BGE-VL-L LR window: 3e-6 / 3e-5 / 1e-4 | 15.23 / 14.77 / 0.70 entR@5 | train_rl.py | bgevl_l-v3pure-lr3e6, -lr3e5, bgevl_l-v3pure | R/bgevl_l-v3pure{-lr3e6,-lr3e5,}.retrieval.metrics.json | HF/… | verified |
| 15 | SigLIP2+anchor retracted: s0 17.47 (+9.2), s1/s2 1.10/0.13, zs 8.23 | as stated | train_rl.py | mlx_configs/siglip2-nogold.yaml (GRPO-8 + anchor, lr1e-4); siglip2-anchor-s1/-s2 (plgrpo + cc0.3, no yaml) | R/siglip2-nogold.*, R/siglip2-anchor-s{1,2}.*, R/siglip2-zs-fresh.* | HF/siglip2-nogold.retrieval.metrics.json | verified |
| 16 | E-VQA OOD: v3-pure +2.6±1.3 acc / +3.9±1.2 entR@5 (3 seeds) | acc 45.87/43.33/44.80, entR@5 81.34/78.93/80.41; zs 42.07 / 76.31 | eval_retrieval.py / eval_vqa.py `--dataset evqa` | ckpts v3-pure{,-s1,-s2}; artifact appendix name `v3-pure-evqa` does not exist, real name is evqa-v3pure | R/evqa-v3pure{,-s1,-s2}.*, R/evqa-zs.* | HF/evqa-v3pure.vqa_top5.json | verified (E-VQA metric not affected) |
| 17 | E-VQA SFT best ckpt +1.4 acc / +2.1 R@5; older released SFT | 43.47 / 78.38; evqa-sft 39.33 / 71.05 | same | `--checkpoint $D/runs/sft-curve-lr1e4/checkpoint-750` | R/evqa-sft-ck750.*, R/evqa-sft.* | HF/evqa-sft-ck750.vqa_top5.json | verified |
| 18 | Text-twin KL cells: β=0.01 −0.7, β=0.1 −1.7, β=0.05 | v3kl001 76.73/63.39/33.80 · v3kl01 74.50/61.35/32.73 · v3-kl 74.73/61.22/32.93 (**resc 42.93 / 41.20 / 41.60** vs 43.07: −0.13 / −1.87) | train_rl.py `--kl_beta` | v3kl001, v3kl01 (no yaml); mlx_configs/v3-kl.yaml | R/v3kl001.*, R/v3kl01.*, R/v3-kl.* | HF/v3kl001.vqa_rescored.json | verified (on rescored, β=0.01 shrinks to −0.13, inside noise) |
| 19 | Consumption: 2k draws 34.00±0.47 → 6k 34.80 → 12k 34.07 | 33.47/34.33/34.20; s1500 34.80; s3000 34.07 (**resc 43.07±0.41 → 43.80 → 43.60**) | train_rl.py | mlx_configs/v3-rows200k.yaml (pool_train_big, max_train_rows 200000); -s1500, -s3000 | R/v3-rows200k{,-s1,-s2,-s1500,-s3000}.* | HF/v3-rows200k.vqa_rescored.json | verified |
| 20 | Batch-16: v3-b16-s750 34.87; 1M draws 34.40 → 33.60 | 34.87 (resc 44.33); ck3125 34.40 → ck12500 33.60 (resc 43.53 → 41.80) | train_rl.py | mlx_configs/v3-b16-s750.yaml, v3-b16-1M.yaml (+ ck yamls) | R/v3-b16-s750.*, R/v3-b16-1M-ck{3125,6250,9375,12500}.* | HF/v3-b16-1M-ck12500.vqa_rescored.json | verified |
| 21 | Pool size at matched 2k draws | v3-pure 34.47 · rows25k 34.00 · rows200k 34.00 · rows1M 34.73 / s1 34.40; 810k-pool entR@5 78.70 vs 74.37 | train_rl.py | mlx_configs/v3-rows25k.yaml, v3-rows1M-s1.yaml; **no v3-rows1M.yaml on disk** (only -s1 / ck variants) | R/v3-rows25k.*, R/v3-rows1M{,-s1}.* | HF/v3-rows1M.retrieval.metrics.json | verified (values); yaml for seed 0 not found |
| 22 | Movement law (displacement ~linear in LR, 3× steps → ~1.7×) | qualitative | /raid/icy/VLM2Vec-in/mmrag/adapter_stats.py | needs $D/runs adapters (MLX) | not found (no saved output) | — | not found |
| 23 | Text-twin column ("anchor decays 67.4→63.8", 4/4 seeds, 0.6B vs 4B) | — | — | — | — | — | other project (text twin) |
| 24 | Ladder: GME-2B and GME-7B rows | = rows 1, 6, 10, 12 | | | | | verified |
| 25 | BGE-VL-base: zs 14.73; 8.90 @1e-5 · 0.07 @1e-4 · 3.77 anchored | as stated | train_rl.py / ladder_zs.sh | bgevl_b-v3pure-lr1e5 (no yaml); mlx_configs/bgevl_b-v3pure.yaml; bgevl_b-anchor | R/bgevl_b-{v3pure-lr1e5,v3pure,anchor,zeroshot}.* | HF/bgevl_b-zeroshot.retrieval.metrics.json | verified |
| 26 | VLM2Vec-2B: zs 15.67; 0.70 @2e-5 · 0.17 @5e-6 · 1.30 anchored | as stated | same | mlx_configs/vlm2vec2b-v3pure.yaml, vlm2vec2b-nogold.yaml; -lr5e6 | R/vlm2vec2b-{v3pure,v3pure-lr5e6,nogold,zs-fresh}.* | HF/vlm2vec2b-zs-fresh.retrieval.metrics.json | verified |
| 27 | VLM2Vec-7B: zs 14.37; 2.90 @2e-5; 1.90 anchored | as stated | same | mlx_configs/vlm2vec7b-v3pure.yaml; vlm2vec7b-nogold | R/vlm2vec7b-{v3pure,nogold,zeroshot}.* | HF/vlm2vec7b-zeroshot.retrieval.metrics.json | verified |
| 28 | VLM2Vec-4B: zs 12.77; RL unfinished (stopped at step 250, MLX jobs f1a4695950630a51 / 885a9aadfb7b8fe6) | zs 12.77; no RL result | same | mlx_configs/vlm2vec4b-v3pure.yaml | R/vlm2vec4b-zeroshot.*; there is no R/vlm2vec4b-v3pure.* (confirmed absent); its run dir is MLX-only | HF/vlm2vec4b-zeroshot.retrieval.metrics.json | verified (absence confirmed) |
| 29 | Qwen2.5-VL-3B raw: zs 3.73; 0.00 @1e-4 · 0.03 @2e-5 · 0.60 @3e-5 | as stated | same | mlx_configs/qwen25_3b-v3pure.yaml; -lr2e5, -lr3e5 | R/qwen25_3b-{v3pure,v3pure-lr2e5,v3pure-lr3e5,zeroshot}.* | HF/qwen25_3b-zeroshot.retrieval.metrics.json | verified |
| 30 | Qwen2-VL-2B raw: zs 0.03; 0.00 everywhere | as stated | same | mlx_configs/qwen2b-v3pure.yaml, qwen2b-nogold.yaml; -lr5e6 | R/qwen2b-{v3pure,v3pure-lr5e6,nogold,zeroshot}.* | HF/qwen2b-zeroshot.retrieval.metrics.json | verified |
| 31 | SigLIP2: zs 8.23; 6.60 @1e-5 · 0.40 @1e-4 | as stated | same | siglip2-v3pure-lr1e5 (no yaml); mlx_configs/siglip2-v3pure.yaml | R/siglip2-v3pure{-lr1e5,}.* | HF/siglip2-v3pure-lr1e5.retrieval.metrics.json | verified |
| 32 | CLIP-L/14: zs 5.37; 4.10 / 1.70 / 0.43; anchored 29.13; gate 3.80 | as stated | same | mlx_configs/clip-v3pure-lr1e5.yaml, clip-v3pure.yaml, clip-nogold-base.yaml; -lr3e5; clip-nogold-gate | R/clip-{v3pure-lr1e5,v3pure-lr3e5,v3pure,nogold-base,nogold-gate,zeroshot}.* | HF/clip-nogold-base.retrieval.metrics.json | verified |
| 33 | CLIP-B/16: zs 3.10; 0.03 @1e-4; anchored 1.13; "96 distinct top-5 lists" | 3.10 / 0.03 / 1.13 verified; 96 not checked | same; /raid/icy/VLM2Vec-in/mmrag/collect_results.py (distinct-top-5 count) | clipb16-v3pure, clipb16-anchor (no yaml) | R/clipb16-{v3pure,anchor,zeroshot}.* | HF/clipb16-v3pure.retrieval.metrics.json | verified (R@5); plausible (96) |
| 34 | CLIP-B/32: zs 2.60; 2.57 @1e-4 (inert, large adapter movement) | as stated | same + adapter_stats.py | clipb32-v3pure (no yaml) | R/clipb32-{v3pure,zeroshot}.* | HF/clipb32-v3pure.retrieval.metrics.json | verified (R@5); movement not found |
| 35 | Diverse 45k pool: v3-pure | 76.83 / 64.57 / 33.00 (**resc 42.20**) | train_rl.py; /raid/icy/VLM2Vec-in/mmrag/build_div_mix_v2.py; /raid/icy/VLM2Vec-in/mmrag/scripts/resubmit_ema.sh | mlx_configs/div45kv3-v3pure.yaml (pool_div45k_v3, corpus_div_v2) | R/div45kv3-v3pure.* | HF/div45kv3-v3pure.vqa_rescored.json | verified |
| 36 | + index EMA 0.9 / sweep 256 | 75.77 / 63.14 / 32.93 (**resc 41.80**) | train_rl.py (ema_write) | mlx_configs/div45kv3-emaidx.yaml | R/div45kv3-emaidx.* | HF/div45kv3-emaidx.vqa_rescored.json | verified |
| 37 | + EMA doc tower 0.99 (eval confounded) | 72.47 / 60.49 / 32.20 (**resc 40.67**) | train_rl.py | mlx_configs/div45kv3-emaenc.yaml | R/div45kv3-emaenc.* | HF/div45kv3-emaenc.vqa_rescored.json | verified (caveat: the eval is confounded) |
| 38 | Rel-SFT div45k ck450…1050; best ck750 | 73.67/59.02/31.67 · 74.43/58.00/30.87 · **75.87/60.41/32.07** · 72.77/59.80/31.93 · 71.57/54.53/30.73 (resc 39.40 / 37.80 / 39.33 / 39.40 / 35.87; on rescored the best ckpt is ck450 or ck900 at 39.40) | /raid/icy/VLM2Vec-in/mmrag/scripts/launch_sft_div.sh → train_sft.py | sft-div45k, bs24, lr1e-4, 1200 steps | R/sft-div45k-ck{450,600,750,900,1050}.* | HF/sft-div45k-ck750.vqa_rescored.json | verified |
| 39 | RL − SFT-best on div pool | +0.96 entR@5 / +4.16 ansR@5 / +0.93 acc (**resc +2.80** vs 39.40) | arithmetic | — | — | — | verified |
| 40 | Geometry, docs (2,000): base / v3-pure / +idxEMA / +EMA tower | 0.0726 / 0.0486 / 0.0567 / 0.0895 | /raid/icy/VLM2Vec-in/mmrag/geom_probe.py | runs div45kv3-* | R/geom_probe_div45kv3.json | HF/geom_probe_div45kv3.json | verified |
| 41 | Geometry, queries (1,000) | 0.2212 / 0.0594 / 0.0612 / 0.0976 | /raid/icy/VLM2Vec-in/mmrag/geom_probe_q.py | same | R/geom_probe_q_div45kv3.json | HF/geom_probe_q_div45kv3.json | verified |
| 42 | Text twin sees −0.17 query-cos | — | — | — | — | — | other project |
| 43 | Spec values: GME-2B LoRA r32 α64, τ0.02, reader Qwen2.5-VL-3B ×4 @T0.7, 3,000 distractors, refresh 100, eval reader Qwen2.5-VL-7B k=5, 1,500 VQA / 3,000 retrieval queries | as stated | train_rl.py, mlx_entry_cell.sh, eval_vqa.py | headline command in the artifact appendix; args.json on MLX only | — | — | plausible (flags match code paths; no local args.json) |

