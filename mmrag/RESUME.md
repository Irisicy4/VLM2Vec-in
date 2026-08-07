# RESUME.md — continuing mm-RAG on a new cluster

Cluster access to Isambard-AI ended 2026-08-03 (AssocGrpCPUMinutesLimit). Every submitted job
completed; everything below is reproducible from this repo + public sources. Session state
lived in the shared CLAUDE_CONFIG_DIR (`.claude/` on the old scratch, pushed separately by the
main session) — all project knowledge a fresh agent needs is in THIS directory instead:
`README.md` (setup/zoo), `RESULTS.md` + `results_summary.json` (all numbers), `LEDGER.md`
(tested-vs-works), this file (state + next steps).

## 1. Environment

aarch64 or x86 both fine. Known-good stack: torch 2.9.0+cu126, transformers 4.57.6,
peft 0.19.1, flash-attn 2.8.3, qwen-vl-utils, pillow, ijson, matplotlib. Compute nodes offline →
pre-download everything, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` inside jobs.
`export PYTHONPATH=<repo> MMRAG_DATA=<data_root>`.

## 2. Models (all public HF)

Retrievers: `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` (headline), `-7B-Instruct`,
`TIGER-Lab/VLM2Vec-Qwen2VL-2B`/`7B` (+ base `Qwen/Qwen2-VL-2B/7B-Instruct`),
`openai/clip-vit-large-patch14-336`, `google/siglip2-so400m-patch16-384`.
Readers: `Qwen/Qwen2.5-VL-3B-Instruct` (reward), `Qwen/Qwen2.5-VL-7B-Instruct` (eval).
Trained adapters (LoRA, load via `--checkpoint`): `Icey444/mmrag-*` — see README zoo table.

## 3. Data (rebuildable from public sources; old scratch copy at $SCRATCH/mmrag_data is LOST on transfer)

Raw downloads (login node):
- EchoSight InfoSeek CSVs + 100K wiki KB: Dropbox links in github.com/Go2Heart/EchoSight README
  (`infoseek_{train,test}_filtered.csv`, `infoseek_100k_wiki.zip`)
- Official aliases: `storage.googleapis.com/gresearch/open-vision-language/infoseek/infoseek_val.jsonl`
- InfoSeek images: HF `BByrneLab/M2KR_Images` → `Infoseek/infoseek_{val,train}_images.tar` (9+48GB;
  read IN PLACE via mmrag/image_store.py — never extract on inode-limited filesystems)
- E-VQA: `storage.googleapis.com/encyclopedic-vqa/{encyclopedic_kb_wiki.zip,val.csv,test.csv}` +
  EchoSight's `train_full_image_cleaned.csv` (Dropbox); images: M2KR_Images `EVQA/inat.zip`,
  `EVQA/google-landmark.tar` + iNat-2021 val: `ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz`
  (gunzip to .tar); id maps `{train,val}_id2name.json` (EchoSight Dropbox)
Then: `build_infoseek_data.py`, `build_evqa_data.py`, `build_evqa_train_pool.py` (see README).
Gotchas already fixed in code: InfoSeek `answer` is '|'-joined aliases (split!), E-VQA gold =
evidence_section_title match, images per-sample LISTS for the vendored Qwen2VL forward,
query-side max_len 2048, explicit mrope position_ids (encoder.py).

## 4. What is DONE (v2 metric, top-5, n=1500; full tables in RESULTS.md/LEDGER.md)

- **Headline (5 seeds, all SIG)**: annotation-free [no-gold] RL (judge reward, det top-8 pools,
  lr 2e-5) = 0.342 ± 0.002 vs tuned SFT 0.313 ± 0.005 (per-seed McNemar p≤0.027) vs zero-shot
  0.310 (p≤5e-4). SFT vs zero-shot: n.s. (p=0.25). Text side could not achieve this.
- 7B: zs 0.306 → SFT 0.326 → RL[gold] 0.359 / [no-gold] 0.352±0.001 (p=1.6e-07) — RL−SFT gap
  grows with capacity; inverts at CLIP/SigLIP rungs.
- Reward-source bench (same loop): no-gold judge 0.341 > ansmatch 0.334 > goldrel-oracle 0.317;
  dense logit FAILS no-gold (0.315); slate ragacc collapses; 3B (even 2B) reward reader suffices.
- **Transfer**: single-set no-gold under-transfers (E-VQA 0.397 < zs 0.426), but the 50/50
  InfoSeek+EVQA no-gold mix fixes it: 0.463–0.473 E-VQA + 0.331–0.341 InfoSeek; 7B no-gold-mix
  0.495 + 0.344 (best transfer). Gold-RL transfers at 0.462.
- Scaling (fig_mm_scaling.pdf/tex): RL monotone in feedback data at fixed compute, SFT flat/noisy;
  longer training doesn't help; 2k+1500-steps overfits.
- Paper materials: `sec_mm_rag.tex` + figure pushed to Irisicy4/Overleaf_Indirect (NOT \input yet).

## 5. What is UNFINISHED / next steps (priority order)

1. **Review sign-off** (main session): metric v2 + labels done; check LEDGER caveats. Then
   un-provisionalize RESULTS.md/README and \input sec_mm_rag.tex into the paper.
2. **Update fig_mm_scaling + sec_mm_rag numbers** with the final 5-seed means (script:
   plot_scaling.py — seed lists inside are one wave behind; refresh from results_summary.json).
3. Not yet run (nice-to-have): no-gold OOD seeds (mix cells have 3 seeds in-domain, 3 evqa evals);
   E-VQA-primary training with InfoSeek as OOD (symmetric transfer); 7B goldrel/logit controls;
   iterative/multi-hop retrieval (future work per brief); ColPali/late-interaction (future work).
4. Data-mix ratios other than 50/50 untested; reward-reader < 2B untested.
5. `value_head.pt` files for PPO runs live only on the lost scratch (GRPO headline unaffected).

## 6. Session-state note

Recurring 3h experiment loop + monitors existed in the old session (cron ad0d1505) — do NOT
recreate until compute exists. HF token comes from the env (tis_env.sh equivalent); git identity
Irisicy4 / wangby.icy@gmail.com.
