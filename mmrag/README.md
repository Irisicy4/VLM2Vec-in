# mmrag — Multimodal Indirect-Feedback RAG
### Training a multimodal retriever by RL from a frozen VLM reader's answer success

Multimodal counterpart of the text indirect-RL experiment (`trl-projects/projects/indirect/rag`):
cast retrieval as a one-step MDP with policy `π(p|q) = softmax(⟨e(q),e(p)⟩/τ)` over live top-N
pools, reward the retriever's LoRA adapters with a frozen **Qwen2.5-VL-3B** reader's answer
success on knowledge-VQA (**InfoSeek**; OOD **Encyclopedic-VQA**), and compare against
(a) the zero-shot encoder and (b) a relevance-SFT contrastive baseline at equal trainable budget.

> **STATUS: ALL NUMBERS PROVISIONAL** — under main-session review (see `RESULTS.md` header).
> Every RL run is labeled **[gold]** (gold evidence passage force-inserted into pools + used as
> the InfoNCE positive → same supervision level as SFT) or **[no-gold]** (annotation-free:
> pure top-N pools, self-labeled InfoNCE anchor). The reward is an **in-process** reader — no
> server: `logit` = teacher-forced mean logP(answer | image, question, passage); `judge` =
> sampled-answer accuracy (token-boundary cover-EM vs official aliases), both z-scored per pool.

## Setup (Isambard-AI, aarch64 GH200)

```bash
source $SCRATCH/tis_env.sh          # torch 2.9 cu126, transformers 4.57.6, peft, flash-attn 2.8.3
cd $SCRATCH/VLM2Vec-in && export PYTHONPATH=$PWD MMRAG_DATA=$SCRATCH/mmrag_data
# compute nodes are offline: export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 inside jobs
```

Models (pre-downloaded; profiles in `encoder.py:ENCODER_PROFILES`): retrievers
`Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` (headline), `gme-7B`, `TIGER-Lab/VLM2Vec-Qwen2VL-2B/7B`,
`openai/clip-vit-large-patch14-336`, `google/siglip2-so400m-patch16-384` (weak rungs);
readers `Qwen/Qwen2.5-VL-3B-Instruct` (reward) / `7B` (eval).

## Data build (login node, ~10 min + downloads)

```bash
# raw: EchoSight filtered InfoSeek CSVs + 100K wiki KB (Dropbox), official infoseek_val.jsonl
#      (GCS, answer aliases), M2KR_Images Infoseek val/train tars (HF), E-VQA KB+CSVs (GCS),
#      M2KR EVQA inat.zip / google-landmark.tar + iNat-2021 val tar. See git log for exact URLs.
python3 mmrag/build_infoseek_data.py --data_dir $MMRAG_DATA \
    --train_per_entity 10 --max_train 60000 --distractor_articles 20000
python3 mmrag/build_evqa_data.py --data_dir $MMRAG_DATA --distractor_articles 25000
```

Outputs (all packed — 1M-inode quota): `corpus_full` 1.42M passages / `corpus_small` 422K
(gold entities ∪ 20K distractor articles), 45K train / 71K test queries (with official answer
aliases + unseen-entity/-question split labels), SFT pool, E-VQA test 4.7K + 382K-passage corpus.
Images are read **inside** the tar/zip archives via `image_store.py` offset indexes.

## Running experiments

Every sbatch in `runs/` is a self-contained 4-GPU cell grid (`rl_cell`/`sft_cell` =
train → retrieval eval → VQA eval on one GPU). Core commands the cells wrap:

```bash
# Ours: online RL (GRPO default; PPO with --algo ppo). [gold] unless --no_force_gold.
python3 mmrag/train_rl.py --profile gme2b --output_dir $MMRAG_DATA/runs/NAME \
    --algo grpo --reward judge --reader_rollouts 4 --learning_rate 2e-5 \
    --batch_size 4 --num_candidates 8 --max_steps 500 [--no_force_gold] [--pool_sampling]

# Baseline: relevance-SFT (InfoNCE, in-batch negs; --num_hard_negs N for mined negs)
python3 mmrag/train_sft.py --profile gme2b --pool built/pool_train.jsonl \
    --output_dir $MMRAG_DATA/runs/NAME --learning_rate 1e-4 --num_hard_negs 0 --max_steps 600

# Eval: L1 retrieval (entity + answer-level R@k) then L2 downstream VQA (7B reader)
python3 mmrag/eval_retrieval.py --profile gme2b --checkpoint $MMRAG_DATA/runs/NAME --name NAME --max-q 3000
python3 mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/NAME.retrieval.json --k 5 --max-q 1500

# Harvest everything into results_summary.json + RESULTS.md (idempotent)
python3 mmrag/collect_results.py --data_dir $MMRAG_DATA --repo_out mmrag
```

## Experiment zoo (row → driver → config → checkpoint)

Numbers live in [`RESULTS.md`](RESULTS.md) / [`results_summary.json`](results_summary.json)
(regenerate with `collect_results.py`; per-run config in `$MMRAG_DATA/runs/<name>/args.json`).
VQA-metric note: runs evaluated before 2026-08-01 used the v1 substring cover-EM (over-credits);
job m6i re-scores headline rows with the v2 token-boundary metric + strict-EM.

| run | label | driver (job) | checkpoint |
|---|---|---|---|
| gme2b-zeroshot | — | `runs/m2_zeroshot.sbatch` (5853831) | [Alibaba-NLP/gme-Qwen2-VL-2B-Instruct](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct) |
| vlm2vec2b-zeroshot | — | same job | TIGER-Lab/VLM2Vec-Qwen2VL-2B |
| clip / siglip2 zero-shot | — | `runs/m5_clip_ladder.sbatch` (5853835, retry 5856656) | openai / google originals |
| sft-lr{3e5,5e5,1e4,2e4}-h{0,2}[-s1200] | [gold] | `runs/m3_sft_sweep.sbatch` (5853833), `m6c_sft_fair.sbatch` (5856652) | **best: [Icey444/mmrag-sft-lr1e4-h0](https://huggingface.co/Icey444/mmrag-sft-lr1e4-h0)** |
| rl-grpo-judge-lr2e5 | [gold] | `runs/m4_rl_sweep.sbatch` (5853834) | **[Icey444/mmrag-rl-grpo-judge-lr2e5](https://huggingface.co/Icey444/mmrag-rl-grpo-judge-lr2e5)** |
| rl-{grpo,ppo}-logit-lr{2e5,5e5} | [gold] | same job | [Icey444/mmrag-rl-grpo-logit-lr2e5](https://huggingface.co/Icey444/mmrag-rl-grpo-logit-lr2e5) |
| rl-grpo-judge-lr{1e5,3e5}, rl-ppo-judge, N=16 | [gold] | `runs/m6a_rl_refine.sbatch` (5856650) | scratch runs/ |
| grounding×pool 2×2 (gold/nogold × det/sampled) + seed1 | mixed | `runs/m6b_grounding2x2.sbatch` (5856651) | scratch runs/ (competitive no-gold cell → HF post-review) |
| data scaling rows{2k,8k} RL vs SFT | [gold] | `runs/m6d_datascale.sbatch` (5856653) | scratch runs/ |
| GME-7B rung (zs, RL lr{1e-5,5e-6}, SFT) | [gold] | `runs/m6e_7b_ladder.sbatch` (5856654) | scratch runs/ |
| E-VQA OOD transfer (best RL/SFT) | [gold] | `runs/m6f_catchup_ood.sbatch` (5856655) | — |
| winner ablations (1500 steps, 8 rollouts, cc∈{0,.1}) | [gold] | `runs/m6g_winner_ablate.sbatch` (5856743) | scratch runs/ |
| full-corpus (1.42M) triple | — | `runs/m6h_fullcorpus.sbatch` (5856744) | — |
| metric-v2 rescore + fixed E-VQA gold | — | `runs/m6i_rescore.sbatch` (5856745) | — |
| seed replicates (RL s2/s3, SFT s1/s2) | [gold] | `runs/m6j_seeds.sbatch` (5856751) | — |

Key facts for interpretation (documented per review):
- **Exposure asymmetry**: RL visits ~2,000 queries (500 steps × batch 4, ~16K reward calls);
  SFT visits ~19,200 (600 × 32). m6d controls data; m6j adds seed error bars.
- Retrieval gold is **entity-level**; answer-level R@k on the ~83% of rows with an
  alias-bearing passage. E-VQA gold is evidence-section-based (4,744/4,750 rows).
- CLIP/SigLIP2 rungs fuse image+text by normalized mean of the two towers (`clip_encoder.py`);
  their 77/64-token doc cap is intrinsic to the rung.

## Files

See module docstrings; entry points: `train_rl.py` (ours), `train_sft.py` (baseline),
`eval_retrieval.py` / `eval_vqa.py` (L1/L2), `build_*_data.py` (data),
`mine_hard_negs.py`, `collect_results.py`, `rl_core.py` (pure-torch PPO/GRPO math),
`encoder.py` + `clip_encoder.py` (profile-dispatched encoders), `reader_vlm.py` (reward/eval reader),
`image_store.py` (in-archive image access), `smoke_test.py`, `runs/*.sbatch` (drivers).
