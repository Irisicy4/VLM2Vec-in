# v23-base — the working set (branch pinned 2026-08-12)

This branch is the clean base for everything going forward: the two RL recipes that work
(v2 anchored, v3 pure) plus the SFT baseline, with their exact commands, verified numbers,
and provenance. Prior history: v1 = tag `v1-listwise-top8` on `mmrag` (the published listwise
recipe); design rationale and the anchor-mechanism story = `RL_REDESIGN.md`.

All numbers: InfoSeek, corpus_small (422,378 passages), L1 n=3000 / L2 top-5 n=1500,
reader Qwen2.5-VL-7B, v2 token-boundary cover-EM vs official aliases. Fresh same-cluster
retrains throughout (fresh-retrain seed sd ≈ 0.002–0.006; cross-hardware ±0.005).

## The three arms

| arm | acc (seeds) | entity R@5 | what it is |
|---|---|---|---|
| **v3 pure** | **0.3471 ± 0.0019** (3) | 0.7609 | PL policy gradient, sampled ordered lists, exact conditional log-probs, **no anchor, no gold passages** — supervision is only the answer string inside the reward. **Best accuracy. SIG vs SFT and base per seed (McNemar p ≤ 4.8e-05).** |
| **v2 anchored** | 0.3382 ± 0.006 (3) | **0.8013** | same policy gradient + InfoNCE anchor (cc 0.3, self-argmax positive). **Best retrieval recall.** The anchor buys ~+4pts tail recall for ~−1pt accuracy (mechanism: measured margin pressure, see RL_REDESIGN). |
| **SFT baseline** | 0.3100 (bit-exact repro) | 0.7363 | InfoNCE vs the answer-bearing gold passage. Equal-supervision baseline (same answers, different form). Fairest tuned variant: lr1e-4 × 2k rows = 0.327. |

Reference: base zero-shot 0.3080 / R@5 0.6853. Ordering v3 > v2 > SFT > base holds on acc;
v2 > v1 > v3 > SFT > base on R@5 — both RL arms beat SFT on BOTH metrics.

## Commands

```bash
export PYTHONPATH=<this repo> MMRAG_DATA=<data> HF_HOME=<cache>
B="--profile gme2b --pool built/pool_train.jsonl --corpus built/corpus_small.jsonl \
   --no_force_gold --reward judge --reader_model Qwen/Qwen2.5-VL-3B-Instruct \
   --reader_rollouts 4 --batch_size 4 --max_steps 500 --learning_rate 2e-5 \
   --n_distractor_articles 3000 --refresh_steps 100 --seed 0"

# v3 pure (the headline arm)
python3 mmrag/train_rl.py $B --algo plgrpo --pl_group 4 --pl_k 4 --pl_support 24 \
    --contrastive_coef 0 --output_dir $MMRAG_DATA/runs/NAME

# v2 anchored (the recall arm) — identical + anchor
python3 mmrag/train_rl.py $B --algo plgrpo --pl_group 4 --pl_k 4 --pl_support 24 \
    --contrastive_coef 0.3 --output_dir $MMRAG_DATA/runs/NAME

# SFT baseline
python3 mmrag/train_sft.py --profile gme2b --pool built/pool_train.jsonl \
    --learning_rate 1e-4 --num_hard_negs 0 --batch_size 24 --max_steps 600 \
    --output_dir $MMRAG_DATA/runs/NAME

# eval (any arm)
python3 mmrag/encode_corpus.py --profile gme2b --checkpoint $MMRAG_DATA/runs/NAME \
    --cache $MMRAG_DATA/cache/NAME.pt --bs 192
python3 mmrag/eval_retrieval.py --profile gme2b --checkpoint $MMRAG_DATA/runs/NAME \
    --name NAME --max-q 3000 --cache_corpus $MMRAG_DATA/cache/NAME.pt
python3 mmrag/eval_vqa.py --retrieval $MMRAG_DATA/results/NAME.retrieval.json \
    --reader Qwen/Qwen2.5-VL-7B-Instruct --k 5 --max-q 1500
# significance vs a baseline
python3 mmrag/significance.py --results_dir $MMRAG_DATA/results --pairs NAME:sft-lr1e4-h0-repro
```

## Extra knobs on the v3 path (all measured, all default-off)

`--kl_beta` (hurts: 0.3293) · `--entropy_coef` (neutral: 0.3440) · `--baseline rloo` ·
`--pl_behavior_temperature` (off-policy exploration; hurt in combo) · `--inner_epochs`
(real PPO clip; hurt: 0.3213 with sampling, 0.3353 on v1). Simplest is best — additions to
v3-pure have never helped.

## Known limits (don't rediscover)

- Ceiling ≈0.345–0.350 at 2B regardless of data (×16 flat), compute (×3 flat/negative, ×6
  degrades), estimator, or anchor. Capacity moves it (recorded 7B: 0.351–0.359).
- Recipes are a few-thousand-visited-query correction; peak early, then drift.
- Reward metric must be alias-aware and audited against the eval metric AS A PAIR
  (the text side's historical no-gold failure was 57% false-zero rewards from dropped aliases).
- MLX quota kills silent at >~6 concurrent 1-GPU jobs; cells are idempotent — resubmit.
