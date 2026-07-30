# mmrag — Multimodal Indirect-Feedback RAG (retriever RL from VLM answer success)

Multimodal counterpart of the text experiment (`trl-projects/projects/indirect/rag`): train a
**multimodal dense retriever** (VLM2Vec, Qwen2-VL backbone) with **RL where the reward is a frozen
VLM reader's answer success** on knowledge-VQA — no relevance labels — and compare against
(a) the zero-shot encoder and (b) a relevance-SFT contrastive baseline trained on gold-evidence pairs.

Task: **InfoSeek** (EchoSight-filtered splits), corpus = EchoSight's 100K-article Wikipedia KB
chunked into ~100-word passages. Query = image + question; reader = Qwen2.5-VL (3B reward / 7B eval).

## Design (mirrors the text experiment)

- Policy: `π(p|q) = softmax(⟨e(q), e(p)⟩ / τ)` over an N-candidate pool retrieved live from the KB
  by the current encoder (τ = 0.02). Deterministic embeddings; LoRA adapters are the only trainable
  weights (fresh adapters on top of the merged VLM2Vec checkpoint; same budget for SFT arm).
- Reward: `log p_VLM(answer | image, question, passage)` (dense, `--reward logit`) or sampled
  answer accuracy (`--reward judge`), z-scored per pool.
- Algorithms: GRPO (default, group-relative advantage, no critic) or PPO (value head V(q) on the
  detached query embedding, separate high LR, critic warmup).
- Anchors against collapse: InfoNCE (in-batch, gold at slot 0) + optional KL-to-init (`--beta`).
- Gold passage is force-included at slot 0 (InfoNCE positive) unless `--no_force_gold`
  (annotation-free variant).

## Files

| file | role |
|---|---|
| `build_infoseek_data.py` | M1: CSVs + KB → `corpus_{full,small}.jsonl`, `queries_{train,test}.jsonl`, `pool_train.jsonl` |
| `image_store.py` | read images straight from the M2KR tars (inode quota: never extract) |
| `encoder.py` | encode API over VLM2Vec `MMEBModel` (+fresh LoRA for training arms) |
| `reader_vlm.py` | in-process Qwen2.5-VL reader: `score_logit` / `score_judge` (cover-EM) |
| `rl_core.py` | pure-torch PPO/GRPO math (ported from trl embedding_ppo) |
| `train_rl.py` | **Ours**: online RL loop (rollout → VLM reward → PPO/GRPO + InfoNCE) |
| `train_sft.py` | relevance-SFT baseline (InfoNCE, in-batch + mined hard negs) |
| `mine_hard_negs.py` | top-k non-gold-entity passages as hard negatives for SFT |
| `eval_retrieval.py` | L1: entity-level + answer-level R@k, writes `<name>.retrieval.json` |
| `eval_vqa.py` | L2: reader answer accuracy over top-k (modes: topk / no-context / gold oracle) |
| `smoke_test.py` | M0 gate: store + encoder top-k sanity + reader sanity |
| `runs/*.sbatch` | SLURM drivers (Isambard-AI; 4×GH200; compute nodes offline) |

## Data layout (`$MMRAG_DATA = $SCRATCH/mmrag_data`)

```
raw/        infoseek_{train,test}_filtered.csv (EchoSight), wiki_100_dict_v4.json (100K KB)
built/      corpus_full.jsonl (1.42M passages) corpus_small.jsonl (422K; gold∪20K distractor articles)
            queries_train.jsonl (45K, ≤10/entity) queries_test.jsonl (71K = InfoSeek val)
            pool_train.jsonl (+_negs after mining)
images/     Infoseek/infoseek_{val,train}_images.tar  (M2KR_Images; read in place via index)
results/    *.retrieval.json, *.vqa_*.json, cached corpus embeddings (*.pt)
runs/       training outputs (adapters, metrics.jsonl)
```

Provenance: EchoSight (github.com/Go2Heart/EchoSight) filtered CSVs + 100K KB;
images from HF `BByrneLab/M2KR_Images` (`Infoseek/infoseek_{val,train}_images.tar`);
retriever `TIGER-Lab/VLM2Vec-Qwen2VL-2B` (LoRA on `Qwen/Qwen2-VL-2B-Instruct`).
Gold relevance is **entity-level** (any passage of the question's Wikipedia article); the answer
string appears verbatim in the article for only ~28%/44% of train/test rows (`ans_pids`), so
answer-level recall is reported on that subset only.

## Typical commands

```bash
# login node (internet): data build
python3 mmrag/build_infoseek_data.py --data_dir $MMRAG_DATA

# compute node (offline): M2 zero-shot triple + mining
sbatch mmrag/runs/m2_zeroshot.sbatch

# M3 relevance-SFT baseline
python3 mmrag/train_sft.py --pool built/pool_train_negs.jsonl --num_hard_negs 2 \
    --output_dir $MMRAG_DATA/runs/sft-2b --max_steps 600

# M4 indirect RL (Ours)
python3 mmrag/train_rl.py --output_dir $MMRAG_DATA/runs/rl-2b --algo grpo --reward logit \
    --batch_size 4 --num_candidates 8 --max_steps 500 --reader_device cuda:1

# eval any checkpoint
python3 mmrag/eval_retrieval.py --checkpoint $MMRAG_DATA/runs/rl-2b --name rl-2b --max-q 5000
python3 mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/rl-2b.retrieval.json --k 5
```
