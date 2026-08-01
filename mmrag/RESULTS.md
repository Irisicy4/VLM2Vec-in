# mm-RAG results (PROVISIONAL — under review, do not cite yet)

PROVISIONAL — pending main-session review (SFT tuning fairness, gold-seeded pools vs annotation-free claims, PPO/GRPO advantage shapes, cover-EM crediting, split/seed noise)

Reward reader is IN-PROCESS (no server): Qwen2.5-VL-3B-Instruct on the trainer GPU.

## 1. PRIMARY: annotation-free RL ([no-gold])

Pure top-N pools under the live policy; the ONLY supervision is the gold ANSWER inside the reward (no passage-level labels). Key question: does this beat zero-shot and the relevance-supervised arms? (The text-side experiment could NOT achieve this.)

| run | arm | label | axes | entity R@1/R@5 | answer R@1/R@5 | VQA top5 acc / strictEM (n) |
|---|---|---|---|---|---|---|
| *(m6b no-gold cells still training/queued)* | | | | | | |

## 2. Baselines (no training)

| run | arm | label | axes | entity R@1/R@5 | answer R@1/R@5 | VQA top5 acc / strictEM (n) |
|---|---|---|---|---|---|---|
| gme2b-evqa-zeroshot | zero-shot |  | profile=gme2b | 0.4994/0.7645 | 0.362/0.6499 | 0.426 / 0.2407 (1500) |
| clip-zeroshot | zero-shot |  | profile=clip | 0.0203/0.054 | 0.0053/0.0176 | 0.16 / 0.1173 (1500) |
| gme2b-zeroshot | zero-shot |  | profile=gme2b | 0.4494/0.6938 | 0.3211/0.5471 | 0.3033 / 0.2333 (1500) |
| siglip2-zeroshot | zero-shot |  | profile=siglip2 | 0.0273/0.0827 | 0.0065/0.0294 | — |
| vlm2vec2b-zeroshot | zero-shot |  | profile=vlm2vec2b | 0.0672/0.1534 | 0.0451/0.1097 | 0.1693 / 0.1353 (1500) |

## 3. Gold-seeded reference (relevance-supervised) — SECONDARY

These RL runs force-insert the gold evidence passage into every pool AND use it as the InfoNCE positive — the same passage-level supervision SFT consumes. They compare supervision *form* (RL vs contrastive) at equal labels, NOT label-free learning.

| run | arm | label | axes | entity R@1/R@5 | answer R@1/R@5 | VQA top5 acc / strictEM (n) |
|---|---|---|---|---|---|---|
| rl-j2e5-evqa | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.615/0.8293 | 0.4128/0.6981 | 0.462 / 0.256 (1500) |
| sft-lr1e4-evqa | sft |  | lr=1e4, num_hard_negs=0, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.5309/0.709 | 0.3145/0.5538 | 0.3953 / 0.2153 (1500) |
| rl-grpo-judge-lr2e5 | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6093/0.8043 | 0.4298/0.6796 | 0.3433 / 0.2607 (1500) |
| rl-grpo-logit-lr2e5 | rl | [gold] | algo=grpo, reward=logit, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6027/0.8213 | 0.3927/0.6339 | 0.3293 / 0.2453 (1500) |
| rl-grpo-logit-lr5e5 | rl | [gold] | algo=grpo, reward=logit, lr=5e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5643/0.6823 | 0.2755/0.4788 | 0.298 / 0.2213 (1500) |
| rl-ppo-logit-lr2e5 | rl | [gold] | algo=ppo, reward=logit, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6087/0.778 | 0.3682/0.6045 | 0.3287 / 0.2407 (1500) |
| clip-sft-lr2e-4 | sft |  | lr=0.0002, num_hard_negs=0, max_steps=600, batch_size=64, max_train_rows=all(~41k), profile=clip, seed=0 | 0.4373/0.613 | 0.1976/0.4135 | — |
| sft-lr1e4-h0 | sft |  | lr=1e4, num_hard_negs=0, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.589/0.7373 | 0.3531/0.5947 | 0.31 / 0.2387 (1500) |
| sft-lr1e4-h2 | sft |  | lr=1e4, num_hard_negs=2, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.5673/0.6877 | 0.2506/0.4678 | — |
| sft-lr2e4-h2 | sft |  | lr=2e4, num_hard_negs=2, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.566/0.6923 | 0.2392/0.4669 | — |
| sft-lr5e5-h2 | sft |  | lr=5e5, num_hard_negs=2, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.5723/0.6983 | 0.2527/0.469 | — |

## 4. Feedback-data scaling (arms labeled; no-gold arms lead when available)

| feedback size | arm | label | entity R@5 | answer R@5 | VQA top5 acc | n_retr / n_vqa |
|---|---|---|---|---|---|---|
| 0 (zero-shot) | — | — | 0.6938 | 0.5471 | 0.3033 | 5000 / 1500 |
| 41k (full) | rl | [gold] | 0.8043 | 0.6796 | 0.3433 | 3000 / 1500 |
| 41k (full) | sft | — | 0.7373 | 0.5947 | 0.31 | 3000 / 1500 |

All scaling rows are currently [gold] (gold-seeded pools) unless labeled [no-gold]; the m6b no-gold cells provide the annotation-free scaling anchor when they land.
