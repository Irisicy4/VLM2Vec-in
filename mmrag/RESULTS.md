# mm-RAG results (PROVISIONAL — under review, do not cite yet)

PROVISIONAL — pending main-session review (SFT tuning fairness, gold-seeded pools vs annotation-free claims, PPO/GRPO advantage shapes, cover-EM crediting, split/seed noise)

Reward reader is IN-PROCESS (no server): Qwen2.5-VL-3B-Instruct on the trainer GPU.

## 1. PRIMARY: annotation-free RL ([no-gold])

Pure top-N pools under the live policy; the ONLY supervision is the gold ANSWER inside the reward (no passage-level labels). Key question: does this beat zero-shot and the relevance-supervised arms? (The text-side experiment could NOT achieve this.)

| run | arm | label | axes | entity R@1/R@5 | answer R@1/R@5 | VQA top5 acc / strictEM (n) |
|---|---|---|---|---|---|---|
| rl-j2e5-nogold-det-N16-evqa | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=16, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5202/0.7742 | 0.1517/0.3531 | 0.272 / 0.1567 (1500) |
| rl-j2e5-nogold-det-evqa | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5739/0.8008 | 0.3034/0.5831 | 0.3967 / 0.2173 (1500) |
| rl-j2e5-nogold-evqaonly | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5233/0.7547 | 0.3829/0.6155 | 0.3127 / 0.2387 (1500) |
| rl-j2e5-nogold-mix-evqa | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6164/0.8338 | 0.4477/0.7333 | 0.4633 / 0.2547 (1500) |
| rl-j2e5-nogold-mix-s1-evqa | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=1 | 0.6309/0.8404 | 0.4926/0.7485 | 0.4727 / 0.254 (1500) |
| rl-judge-nogold-frozen-evqa | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5546/0.7908 | 0.3283/0.595 | 0.398 / 0.2247 (1500) |
| rl-ragacc-nogold-lr2e5-evqa | rl | [no-gold] | algo=grpo, reward=ragacc, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.0004/0.0007 | 0.0/0.0 | 0.0967 / 0.0433 (1500) |
| rl7b-j-nogold-mix-evqa | rl | [no-gold] | algo=grpo, reward=judge, lr=1e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme7b, seed=0 | 0.6849/0.8819 | 0.5241/0.8056 | 0.4953 / 0.2827 (1500) |
| rl-ansmatch-nogold | rl | [no-gold] | algo=grpo, reward=ansmatch, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5263/0.7443 | 0.378/0.6073 | 0.334 / 0.2533 (1500) |
| rl-gain2e5-nogold | rl | [no-gold] | algo=grpo, reward=gain, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.59/0.7957 | 0.4049/0.649 | 0.3373 / 0.2513 (1500) |
| rl-goldrel-nogold | rl | [no-gold] | algo=grpo, reward=goldrel, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6057/0.7833 | 0.3796/0.6057 | 0.3173 / 0.2393 (1500) |
| rl-j-nogold-det-lr1e5 | rl | [no-gold] | algo=grpo, reward=judge, lr=1e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.598/0.797 | 0.3971/0.6518 | 0.332 / 0.25 (1500) |
| rl-j-nogold-frozen-lr1e5 | rl | [no-gold] | algo=grpo, reward=judge, lr=1e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.579/0.7973 | 0.3967/0.6518 | 0.3367 / 0.2527 (1500) |
| rl-j2e5-nogold-2breward | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5873/0.7817 | 0.3963/0.6212 | 0.3393 / 0.25 (1500) |
| rl-j2e5-nogold-N16-1500 | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=16, max_steps=1500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5947/0.7873 | 0.3931/0.6237 | 0.3327 / 0.2507 (1500) |
| rl-j2e5-nogold-N32 | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=32, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5927/0.784 | 0.3776/0.6041 | 0.328 / 0.246 (1500) |
| rl-j2e5-nogold-cc0 | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.0, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5907/0.8067 | 0.4/0.6588 | 0.3373 / 0.2527 (1500) |
| rl-j2e5-nogold-cc01 | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.1, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5817/0.7913 | 0.4016/0.651 | 0.3347 / 0.2527 (1500) |
| rl-j2e5-nogold-det | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5907/0.796 | 0.4008/0.6351 | 0.3413 / 0.2513 (1500) |
| rl-j2e5-nogold-det-1500 | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=1500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5753/0.7907 | 0.382/0.6094 | 0.3193 / 0.2427 (1500) |
| rl-j2e5-nogold-det-N16 | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=16, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5683/0.7917 | 0.3657/0.5984 | 0.3193 / 0.242 (1500) |
| rl-j2e5-nogold-det-s2 | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=2 | 0.6023/0.805 | 0.3971/0.6388 | 0.3393 / 0.2533 (1500) |
| rl-j2e5-nogold-det-s3 | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=3 | 0.5737/0.7747 | 0.4012/0.6408 | 0.3427 / 0.2567 (1500) |
| rl-j2e5-nogold-det-s4 | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=4 | 0.5947/0.7937 | 0.4073/0.6608 | 0.3427 / 0.2587 (1500) |
| rl-j2e5-nogold-mix | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.582/0.7923 | 0.3947/0.6363 | 0.3313 / 0.25 (1500) |
| rl-j2e5-nogold-mix-s1 | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=1 | 0.5813/0.792 | 0.4135/0.6629 | 0.342 / 0.2607 (1500) |
| rl-j2e5-nogold-mix-s2 | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=2 | 0.59/0.8007 | 0.4127/0.6698 | 0.3407 / 0.2487 (1500) |
| rl-j2e5-nogold-rows2k | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=2000, profile=gme2b, seed=0 | 0.589/0.7827 | 0.3837/0.6102 | 0.3333 / 0.2493 (1500) |
| rl-j2e5-nogold-rows8k | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=8000, profile=gme2b, seed=0 | 0.5897/0.7957 | 0.3927/0.642 | 0.3407 / 0.2533 (1500) |
| rl-j2e5-nogold-rows8k-1500 | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=1500, contrastive_coef=0.3, max_train_rows=8000, profile=gme2b, seed=0 | 0.5543/0.788 | 0.3714/0.5914 | 0.322 / 0.244 (1500) |
| rl-j2e5-nogold-samp | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=True, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.58/0.7867 | 0.3898/0.6265 | 0.3327 / 0.2473 (1500) |
| rl-j2e5-nogold-samp-t01 | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=True, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.592/0.729 | 0.3567/0.5776 | 0.326 / 0.2467 (1500) |
| rl-judge-nogold-frozen | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.569/0.7843 | 0.3927/0.6302 | 0.336 / 0.26 (1500) |
| rl-judge7b-nogold-lr2e5 | rl | [no-gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5827/0.785 | 0.3878/0.6163 | 0.3373 / 0.2513 (1500) |
| rl-logit2e5-nogold-det | rl | [no-gold] | algo=grpo, reward=logit, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.564/0.7493 | 0.3441/0.5404 | 0.3153 / 0.232 (1500) |
| rl-ragacc-nogold-N16 | rl | [no-gold] | algo=grpo, reward=ragacc, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=16, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.427/0.668 | 0.2555/0.482 | 0.2927 / 0.2313 (1500) |
| rl-ragacc-nogold-frozen | rl | [no-gold] | algo=grpo, reward=ragacc, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.2473/0.4693 | 0.1641/0.3506 | 0.256 / 0.1953 (1500) |
| rl-ragacc-nogold-lr1e5 | rl | [no-gold] | algo=grpo, reward=ragacc, lr=1e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.1083/0.3853 | 0.0649/0.2433 | 0.21 / 0.1733 (1500) |
| rl-ragacc-nogold-lr1e5-1500 | rl | [no-gold] | algo=grpo, reward=ragacc, lr=1e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=1500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.0013/0.003 | 0.0016/0.0037 | 0.144 / 0.1087 (1500) |
| rl-ragacc-nogold-lr2e5 | rl | [no-gold] | algo=grpo, reward=ragacc, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.0007/0.0017 | 0.0008/0.002 | 0.148 / 0.1113 (1500) |
| rl-ragacc-nogold-lr2e5-s1 | rl | [no-gold] | algo=grpo, reward=ragacc, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=1 | 0.2827/0.5633 | 0.1792/0.4029 | 0.2653 / 0.208 (1500) |
| rl-ragacc7b-nogold-lr2e5 | rl | [no-gold] | algo=grpo, reward=ragacc, lr=2e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.159/0.403 | 0.0914/0.2808 | 0.2273 / 0.1773 (1500) |
| rl7b-ansmatch-nogold | rl | [no-gold] | algo=grpo, reward=ansmatch, lr=1e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme7b, seed=0 | 0.6343/0.817 | 0.4531/0.6796 | 0.352 / 0.272 (1500) |
| rl7b-j-nogold-lr1e5 | rl | [no-gold] | algo=grpo, reward=judge, lr=1e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme7b, seed=0 | 0.651/0.829 | 0.451/0.6894 | 0.3513 / 0.268 (1500) |
| rl7b-j-nogold-lr1e5-s1 | rl | [no-gold] | algo=grpo, reward=judge, lr=1e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme7b, seed=1 | 0.6467/0.8257 | 0.4571/0.6959 | 0.3527 / 0.2647 (1500) |
| rl7b-j-nogold-lr1e5-s2 | rl | [no-gold] | algo=grpo, reward=judge, lr=1e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme7b, seed=2 | 0.6397/0.83 | 0.4465/0.6939 | 0.3533 / 0.2627 (1500) |
| rl7b-j-nogold-lr5e6 | rl | [no-gold] | algo=grpo, reward=judge, lr=5e-06, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme7b, seed=0 | 0.6373/0.833 | 0.4522/0.698 | 0.3447 / 0.2593 (1500) |
| rl7b-j-nogold-mix | rl | [no-gold] | algo=grpo, reward=judge, lr=1e-05, gold_in_pool=False, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme7b, seed=0 | 0.6343/0.832 | 0.4645/0.709 | 0.344 / 0.258 (1500) |

## 2. Baselines (no training)

| run | arm | label | axes | entity R@1/R@5 | answer R@1/R@5 | VQA top5 acc / strictEM (n) |
|---|---|---|---|---|---|---|
| rl-j2e5-evqaonly-evqa | ? |  |  | 0.6438/0.8382 | 0.5438/0.7678 | 0.4787 / 0.2633 (1500) |
| rl-j2e5-nogold-evqaonly-evqa | ? |  |  | 0.6279/0.8327 | 0.5104/0.7474 | 0.4687 / 0.258 (1500) |
| gme2b-evqa-zeroshot | zero-shot |  | profile=gme2b | 0.4994/0.7645 | 0.362/0.6499 | 0.426 / 0.2407 (1500) |
| clip-zeroshot | zero-shot |  | profile=clip | 0.0203/0.054 | 0.0053/0.0176 | 0.16 / 0.1173 (1500) |
| gme2b-zeroshot | zero-shot |  | profile=gme2b | 0.4494/0.6938 | 0.3211/0.5471 | 0.3033 / 0.2333 (1500) |
| gme2b-zeroshot-3k | zero-shot |  | profile=gme2b | 0.4443/0.6843 | 0.3151/0.5482 | 0.31 / 0.2387 (1500) |
| gme2b-zs-full | zero-shot |  | profile=gme2b | 0.3987/0.6303 | 0.282/0.5004 | 0.2887 / 0.2273 (1500) |
| gme7b-zeroshot | zero-shot |  | profile=gme7b | 0.447/0.6953 | 0.3118/0.5376 | 0.306 / 0.236 (1500) |
| siglip2-zeroshot | zero-shot |  | profile=siglip2 | 0.0273/0.0827 | 0.0065/0.0294 | — |
| vlm2vec2b-zeroshot | zero-shot |  | profile=vlm2vec2b | 0.0672/0.1534 | 0.0451/0.1097 | 0.1693 / 0.1353 (1500) |

## 3. Gold-seeded reference (relevance-supervised) — SECONDARY

These RL runs force-insert the gold evidence passage into every pool AND use it as the InfoNCE positive — the same passage-level supervision SFT consumes. They compare supervision *form* (RL vs contrastive) at equal labels, NOT label-free learning.

| run | arm | label | axes | entity R@1/R@5 | answer R@1/R@5 | VQA top5 acc / strictEM (n) |
|---|---|---|---|---|---|---|
| rl-j2e5-evqa | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.615/0.8293 | 0.4128/0.6981 | 0.462 / 0.256 (1500) |
| rl-j2e5-evqaonly | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.505/0.743 | 0.3624/0.609 | 0.3133 / 0.236 (1500) |
| rl-j2e5-mix-evqa | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6346/0.8301 | 0.5026/0.747 | 0.4773 / 0.2667 (1500) |
| sft-lr1e4-evqa | sft |  | lr=1e4, num_hard_negs=0, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.5309/0.709 | 0.3145/0.5538 | 0.3953 / 0.2153 (1500) |
| sft-lr1e4-evqaonly | sft |  | lr=0.0001, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.561/0.7447 | 0.3824/0.6086 | 0.3167 / 0.2433 (1500) |
| sft-lr1e4-evqaonly-evqa | sft |  | lr=?, num_hard_negs=?, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.6098/0.8093 | 0.4811/0.7255 | 0.45 / 0.2593 (1500) |
| sft-lr1e4-mix-evqa | sft |  | lr=0.0001, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.585/0.786 | 0.4058/0.6721 | 0.4467 / 0.2513 (1500) |
| clip-rl-grpo-judge-lr1e-4 | rl | [gold] | algo=grpo, reward=judge, lr=0.0001, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=clip, seed=0 | 0.3377/0.575 | 0.1922/0.36 | — |
| rl-ansmatch-gold | rl | [gold] | algo=grpo, reward=ansmatch, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6/0.7783 | 0.4318/0.6718 | 0.346 / 0.2547 (1500) |
| rl-gain2e5-gold | rl | [gold] | algo=grpo, reward=gain, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5993/0.8153 | 0.4135/0.6792 | 0.3367 / 0.2513 (1500) |
| rl-goldrel-gold | rl | [gold] | algo=grpo, reward=goldrel, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.612/0.8037 | 0.402/0.662 | 0.3413 / 0.2553 (1500) |
| rl-grpo-judge-lr1e5 | rl | [gold] | algo=grpo, reward=judge, lr=1e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.596/0.8083 | 0.4098/0.6698 | 0.3353 / 0.2513 (1500) |
| rl-grpo-judge-lr2e5 | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6093/0.8043 | 0.4298/0.6796 | 0.3433 / 0.2607 (1500) |
| rl-grpo-judge-lr2e5-N16 | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=16, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5977/0.8147 | 0.4106/0.6722 | 0.3313 / 0.2547 (1500) |
| rl-grpo-judge-lr3e5 | rl | [gold] | algo=grpo, reward=judge, lr=3e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.614/0.7893 | 0.4253/0.6755 | 0.334 / 0.25 (1500) |
| rl-grpo-logit-lr2e5 | rl | [gold] | algo=grpo, reward=logit, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6027/0.8213 | 0.3927/0.6339 | 0.3293 / 0.2453 (1500) |
| rl-grpo-logit-lr5e5 | rl | [gold] | algo=grpo, reward=logit, lr=5e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5643/0.6823 | 0.2755/0.4788 | 0.298 / 0.2213 (1500) |
| rl-j2e5-1500steps | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=1500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.598/0.8043 | 0.4184/0.6714 | 0.344 / 0.2533 (1500) |
| rl-j2e5-cc0 | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.0, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.594/0.8083 | 0.4139/0.6673 | 0.3327 / 0.2527 (1500) |
| rl-j2e5-cc01 | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.1, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6003/0.8027 | 0.4245/0.6751 | 0.3413 / 0.25 (1500) |
| rl-j2e5-full | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5653/0.764 | 0.3992/0.638 | 0.328 / 0.2553 (1500) |
| rl-j2e5-gold-det-s1 | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=1 | 0.6017/0.8107 | 0.431/0.6878 | 0.3427 / 0.2553 (1500) |
| rl-j2e5-gold-samp | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=True, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6027/0.811 | 0.4163/0.6759 | 0.3427 / 0.2587 (1500) |
| rl-j2e5-mix | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.607/0.778 | 0.4163/0.6669 | 0.344 / 0.252 (1500) |
| rl-j2e5-roll8 | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6067/0.797 | 0.4249/0.6727 | 0.3393 / 0.254 (1500) |
| rl-j2e5-rows20k | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=20000, profile=gme2b, seed=0 | 0.6133/0.7953 | 0.431/0.6833 | 0.346 / 0.2587 (1500) |
| rl-j2e5-rows2k | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=2000, profile=gme2b, seed=0 | 0.5827/0.804 | 0.3918/0.6441 | 0.3287 / 0.2573 (1500) |
| rl-j2e5-rows2k-1500 | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=1500, contrastive_coef=0.3, max_train_rows=2000, profile=gme2b, seed=0 | 0.5273/0.772 | 0.351/0.5551 | 0.3 / 0.2273 (1500) |
| rl-j2e5-rows8k | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=8000, profile=gme2b, seed=0 | 0.606/0.7893 | 0.4257/0.669 | 0.3367 / 0.2513 (1500) |
| rl-j2e5-rows8k-1500 | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=1500, contrastive_coef=0.3, max_train_rows=8000, profile=gme2b, seed=0 | 0.5823/0.813 | 0.3959/0.6555 | 0.3313 / 0.2547 (1500) |
| rl-j2e5-s2 | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=2 | 0.62/0.8077 | 0.4245/0.6747 | 0.35 / 0.2573 (1500) |
| rl-j2e5-s3 | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=3 | 0.6033/0.8107 | 0.4208/0.6812 | 0.3393 / 0.2527 (1500) |
| rl-judge-roll1 | rl | [gold] | algo=grpo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6103/0.8087 | 0.4233/0.6755 | 0.344 / 0.2607 (1500) |
| rl-logit2e5-full | rl | [gold] | algo=grpo, reward=logit, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.566/0.7857 | 0.3673/0.5947 | 0.3167 / 0.238 (1500) |
| rl-mix2e5-gold | rl | [gold] | algo=grpo, reward=mix, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.624/0.749 | 0.3682/0.602 | 0.3327 / 0.2427 (1500) |
| rl-ppo-judge-lr2e5 | rl | [gold] | algo=ppo, reward=judge, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6033/0.7903 | 0.4122/0.6657 | 0.3427 / 0.2513 (1500) |
| rl-ppo-logit-lr2e5 | rl | [gold] | algo=ppo, reward=logit, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6087/0.778 | 0.3682/0.6045 | 0.3287 / 0.2407 (1500) |
| rl-ragacc-gold-lr2e5 | rl | [gold] | algo=grpo, reward=ragacc, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.3713/0.6157 | 0.2229/0.44 | 0.2787 / 0.2193 (1500) |
| rl-ragacc7b-gold-lr2e5 | rl | [gold] | algo=grpo, reward=ragacc, lr=2e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.548/0.787 | 0.3645/0.6086 | 0.3247 / 0.2487 (1500) |
| rl7b-j-lr1e5 | rl | [gold] | algo=grpo, reward=judge, lr=1e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme7b, seed=0 | 0.6537/0.83 | 0.4653/0.7073 | 0.3587 / 0.2653 (1500) |
| rl7b-j-lr1e5-s1 | rl | [gold] | algo=grpo, reward=judge, lr=1e-05, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme7b, seed=1 | 0.6573/0.8323 | 0.4739/0.7176 | 0.3573 / 0.268 (1500) |
| rl7b-j-lr5e6 | rl | [gold] | algo=grpo, reward=judge, lr=5e-06, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=gme7b, seed=0 | 0.6417/0.8337 | 0.4624/0.7082 | 0.3473 / 0.26 (1500) |
| siglip2-rl-grpo-judge-lr1e-4 | rl | [gold] | algo=grpo, reward=judge, lr=0.0001, gold_in_pool=True, pool_sampling=False, num_candidates=8, max_steps=500, contrastive_coef=0.3, max_train_rows=all(~41k), profile=siglip2, seed=0 | 0.145/0.2907 | 0.0837/0.1698 | — |
| clip-sft-lr2e-4 | sft |  | lr=0.0002, num_hard_negs=0, max_steps=600, batch_size=64, max_train_rows=all(~41k), profile=clip, seed=0 | 0.4373/0.613 | 0.1976/0.4135 | — |
| sft-lr1e4-h0 | sft |  | lr=1e4, num_hard_negs=0, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.589/0.7373 | 0.3531/0.5947 | 0.31 / 0.2387 (1500) |
| sft-lr1e4-h0-full | sft |  | lr=1e4, num_hard_negs=0, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.537/0.6903 | 0.3208/0.5563 | 0.3053 / 0.2353 (1500) |
| sft-lr1e4-h0-rows20k | sft |  | lr=0.0001, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=20000, profile=gme2b, seed=0 | 0.5987/0.745 | 0.3163/0.5718 | 0.3107 / 0.2327 (1500) |
| sft-lr1e4-h0-rows2k | sft |  | lr=0.0001, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=2000, profile=gme2b, seed=0 | 0.5933/0.7547 | 0.3461/0.6102 | 0.334 / 0.2473 (1500) |
| sft-lr1e4-h0-rows2k-s1 | sft |  | lr=0.0001, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=2000, profile=gme2b, seed=1 | 0.6127/0.7457 | 0.342/0.5763 | 0.32 / 0.2407 (1500) |
| sft-lr1e4-h0-rows8k | sft |  | lr=0.0001, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=8000, profile=gme2b, seed=0 | 0.5903/0.7177 | 0.3082/0.5416 | 0.3007 / 0.23 (1500) |
| sft-lr1e4-h0-s1 | sft |  | lr=0.0001, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=all(~41k), profile=gme2b, seed=1 | 0.6063/0.747 | 0.3257/0.5747 | 0.308 / 0.2327 (1500) |
| sft-lr1e4-h0-s1200 | sft |  | lr=0.0001, num_hard_negs=0, max_steps=1200, batch_size=24, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5883/0.7317 | 0.3298/0.5751 | 0.3173 / 0.246 (1500) |
| sft-lr1e4-h0-s2 | sft |  | lr=0.0001, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=all(~41k), profile=gme2b, seed=2 | 0.572/0.722 | 0.3359/0.5845 | 0.3033 / 0.2287 (1500) |
| sft-lr1e4-h2 | sft |  | lr=1e4, num_hard_negs=2, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.5673/0.6877 | 0.2506/0.4678 | — |
| sft-lr1e4-mix | sft |  | lr=0.0001, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.6/0.7573 | 0.3849/0.6355 | 0.328 / 0.2467 (1500) |
| sft-lr2e4-h0 | sft |  | lr=0.0002, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.614/0.7533 | 0.3673/0.6012 | 0.3207 / 0.2387 (1500) |
| sft-lr2e4-h0-s1 | sft |  | lr=0.0002, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=all(~41k), profile=gme2b, seed=1 | 0.5797/0.715 | 0.3286/0.5555 | 0.3087 / 0.2387 (1500) |
| sft-lr2e4-h0-s2 | sft |  | lr=0.0002, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=all(~41k), profile=gme2b, seed=2 | 0.571/0.723 | 0.3237/0.5694 | 0.3127 / 0.2327 (1500) |
| sft-lr2e4-h0-s3 | sft |  | lr=0.0002, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=all(~41k), profile=gme2b, seed=3 | 0.6047/0.7267 | 0.3473/0.5714 | 0.3093 / 0.232 (1500) |
| sft-lr2e4-h2 | sft |  | lr=2e4, num_hard_negs=2, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.566/0.6923 | 0.2392/0.4669 | — |
| sft-lr3e5-h0 | sft |  | lr=3e-05, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.5957/0.7453 | 0.3576/0.6131 | 0.3127 / 0.236 (1500) |
| sft-lr5e5-h0 | sft |  | lr=5e-05, num_hard_negs=0, max_steps=600, batch_size=24, max_train_rows=all(~41k), profile=gme2b, seed=0 | 0.585/0.7417 | 0.36/0.6024 | 0.3147 / 0.238 (1500) |
| sft-lr5e5-h2 | sft |  | lr=5e5, num_hard_negs=2, profile=gme2b, note=axes parsed from name (pre-args.json run) | 0.5723/0.6983 | 0.2527/0.469 | — |
| sft7b-lr1e4-h0 | sft |  | lr=0.0001, num_hard_negs=0, max_steps=600, batch_size=12, max_train_rows=all(~41k), profile=gme7b, seed=0 | 0.626/0.7567 | 0.3706/0.5984 | 0.3307 / 0.2487 (1500) |
| sft7b-lr2e4-h0 | sft |  | lr=0.0002, num_hard_negs=0, max_steps=600, batch_size=12, max_train_rows=all(~41k), profile=gme7b, seed=0 | 0.6077/0.7287 | 0.2996/0.5188 | 0.3087 / 0.2293 (1500) |
| sft7b-lr5e5-h0 | sft |  | lr=5e-05, num_hard_negs=0, max_steps=600, batch_size=12, max_train_rows=all(~41k), profile=gme7b, seed=0 | 0.645/0.7523 | 0.3367/0.5743 | 0.326 / 0.242 (1500) |
| siglip2-sft-lr2e-4 | sft |  | lr=0.0002, num_hard_negs=0, max_steps=600, batch_size=64, max_train_rows=all(~41k), profile=siglip2, seed=0 | 0.275/0.4827 | 0.1371/0.3155 | — |

## 4. Feedback-data scaling (arms labeled; no-gold arms lead when available)

| feedback size | arm | label | entity R@5 | answer R@5 | VQA top5 acc | n_retr / n_vqa |
|---|---|---|---|---|---|---|
| 0 (zero-shot) | — | — | 0.6938 | 0.5471 | 0.3033 | 5000 / 1500 |
| 2k | rl | [gold] | 0.804 | 0.6441 | 0.3287 | 3000 / 1500 |
| 8k | rl | [gold] | 0.7893 | 0.669 | 0.3367 | 3000 / 1500 |
| 2k | sft | — | 0.7547 | 0.6102 | 0.334 | 3000 / 1500 |
| 8k | sft | — | 0.7177 | 0.5416 | 0.3007 | 3000 / 1500 |
| 41k (full) | rl | [gold] | 0.8043 | 0.6796 | 0.3433 | 3000 / 1500 |
| 41k (full) | sft | — | 0.7373 | 0.5947 | 0.31 | 3000 / 1500 |

All scaling rows are currently [gold] (gold-seeded pools) unless labeled [no-gold]; the m6b no-gold cells provide the annotation-free scaling anchor when they land.
