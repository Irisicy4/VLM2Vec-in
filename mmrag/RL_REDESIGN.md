# RL redesign (branch `rl-redesign`, sibling repo VLM2Vec-rl)

## Thesis

The v1 recipe is RL-shaped supervision with supervised-learning scaffolding. Strip the
scaffolding and replace it with RL-native machinery:

| v1 scaffold | why it's a crutch | RL-native replacement |
|---|---|---|
| InfoNCE anchor (cc=0.3) | supervised contrastive loss inside a "policy gradient"; its positive is a pseudo-label (top-1 / gold) | KL-to-init trust region + entropy bonus |
| deterministic top-8 pools | no exploration; policy only re-ranks its own argmax set | Plackett–Luce sampled lists (in since a3b734e) |
| uniform all-actions weighting | not a policy-gradient estimator | exact PL conditional log-probs (a3b734e) |
| per-pool z-score baseline | std-division couples advantage scale to pool degeneracy | RLOO leave-one-out baseline |
| sampling temp == scoring temp (0.02) | near-argmax sampling → no exploration even when "sampling" | decoupled behavior temperature + clipped IS correction (off-policy PPO) |

Evidence this is plausible: recorded cc=0 no-gold = 0.337 (vs 0.341 anchored) — RL already
works without the anchor, losing only ~0.4pt that principled regularization may recover.
Discovered while building this: `--beta` (KL-to-init) was DEAD CODE in v1 — `ref_logps` was
never computed — so the anchor was the only anti-collapse device ever actually active.

## New knobs (all on the plgrpo path)

- `--kl_beta B` — k3 KL-to-init on sampled lists. Reference policy = the base encoder
  (LoRA-B init is zero, so the initial policy IS the base; ref log-probs are computed with
  adapters disabled — no second model in memory).
- `--entropy_coef C` — first-position PL entropy bonus over the support (direct anti-collapse:
  keeps the sampling distribution from spiking, which is what the anchor was papering over).
- `--baseline {group_z,rloo}` — RLOO: A_g = R_g − mean of the OTHER G−1 list rewards.
  Unbiased, no std division (z-score's std blows up advantages precisely when the pool is
  degenerate, e.g. all-zero-reward pools).
- `--pl_behavior_temperature T_b` — sample lists at T_b (explore), score at `--temperature`
  (target). old_logps become the BEHAVIOR log-probs, so the PPO ratio is the true importance
  weight and the clip bounds the off-policyness. This is off-policy clipped PG, the standard
  fix for "the deployed policy is near-argmax but training needs exploration".

## First wave (all cc=0 — the point is no InfoNCE anywhere)

| cell | config | isolates |
|---|---|---|
| v3-pure   | plgrpo, cc0, nothing else | does pure RL collapse without the anchor? |
| v3-kl     | + kl_beta 0.05 | KL trust region as the anchor replacement |
| v3-ent    | + entropy_coef 0.01 | entropy as the anchor replacement |
| v3-rloo   | + baseline rloo | baseline quality |
| v3-full   | kl 0.05 + ent 0.01 + rloo + T_b 0.05 | the whole redesign |

References on identical protocol: v1 anchored 0.3413 (recorded) / 0.3333 (fresh retrain);
v1 cc0 0.3373 (recorded); plgrpo anchored 0.3427/0.3293 (s0/s1). Fresh-retrain noise is
~±0.005 — judge against fresh-retrain controls, not recorded numbers.

Success criterion: any cc0 cell ≥ anchored v1 fresh-retrain (0.333) means the anchor is
replaceable; matching the anchored plgrpo mean (~0.336) means the redesign is strictly
cleaner at no cost; beating 0.342 means it's better.
