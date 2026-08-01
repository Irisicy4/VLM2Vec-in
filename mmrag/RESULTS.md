# mm-RAG results (PROVISIONAL — under review, do not cite yet)

PROVISIONAL — pending main-session review (SFT tuning fairness, gold-seeded pools vs annotation-free claims, PPO/GRPO advantage shapes, cover-EM crediting, split/seed noise)

Reward reader is IN-PROCESS (no server): Qwen2.5-VL-3B-Instruct on the trainer GPU.
Each RL row lists `gold_in_pool` (was the gold passage force-inserted at slot 0 of every candidate pool — i.e. the run is NOT annotation-free) and the exact reward.

| run | arm | label | axes | entity R@1/R@5 | answer R@1/R@5 | VQA top5 acc / strictEM (n) |
|---|---|---|---|---|---|---|
| gme2b-evqa-zeroshot | zero-shot |  | profile=gme2b | 0.4994/0.7645 | 0.2859/0.5621 | 0.436 / None (1500) ⚠v1-metric |
| rl-grpo-judge-lr2e5 | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6093/0.8043 | 0.4298/0.6796 | 0.352 / None (1500) ⚠v1-metric |
| rl-grpo-logit-lr2e5 | rl | [gold] | algo=grpo, reward=logit, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6027/0.8213 | 0.3927/0.6339 | 0.3407 / None (1500) ⚠v1-metric |
| rl-grpo-logit-lr5e5 | rl | [gold] | algo=grpo, reward=logit, lr=5e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5643/0.6823 | 0.2755/0.4788 | 0.3113 / None (1500) ⚠v1-metric |
| rl-ppo-logit-lr2e5 | rl | [gold] | algo=ppo, reward=logit, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6087/0.778 | 0.3682/0.6045 | 0.3373 / None (1500) ⚠v1-metric |
| sft-lr1e4-h0 | sft |  | lr=1e4, num_hard_negs=0, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.589/0.7373 | 0.3531/0.5947 | — |
| sft-lr1e4-h2 | sft |  | lr=1e4, num_hard_negs=2, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.5673/0.6877 | 0.2506/0.4678 | — |
| sft-lr2e4-h2 | sft |  | lr=2e4, num_hard_negs=2, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.566/0.6923 | 0.2392/0.4669 | — |
| sft-lr5e5-h2 | sft |  | lr=5e5, num_hard_negs=2, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.5723/0.6983 | 0.2527/0.469 | — |
| clip-zeroshot | zero-shot |  | profile=clip | 0.0197/0.0543 | 0.0053/0.018 | — |
| gme2b-zeroshot | zero-shot |  | profile=gme2b | 0.4494/0.6938 | 0.3211/0.5471 | 0.3133 / None (1500) ⚠v1-metric |
| vlm2vec2b-zeroshot | zero-shot |  | profile=vlm2vec2b | 0.0672/0.1534 | 0.0451/0.1097 | 0.182 / None (1500) ⚠v1-metric |

## Feedback-data scaling (required deliverable; auto-fills as m6d lands)

| feedback size | arm | entity R@5 | answer R@5 | VQA top5 acc | n_retr / n_vqa |
|---|---|---|---|---|---|
| 0 (zero-shot) | — | 0.6938 | 0.5471 | 0.3133 | 5000 / 1500 |
| 41k (full) | rl | 0.8043 | 0.6796 | 0.352 | 3000 / 1500 |
| 41k (full) | sft | 0.7373 | 0.5947 | — | 3000 / — |

RL runs above with `gold_in_pool=True` use the gold-evidence passage as a forced pool member (grounding); the annotation-free variants are the `nogold` runs of m6b.
