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

## Cross-side notes (from the text session, 2026-08-12)

- **Binding vs inert anchor.** Our anchored runs' InfoNCE loss stays at 0.48–0.63 through the
  final quarter of training (nogate 0.80→0.48, plgrpo 1.02→0.61, plgrpo-s3 1.04→0.63) — the
  anchor gradient never dies. The text side's equivalent decayed to 0.0000 by step 240 (its
  positive = own top-1 was trivially satisfied). Registered joint prediction: anchor removal
  helps where the anchor is binding (us: +0.009–0.011, observed) and does nothing where it is
  inert (them: cells rag-plgcc0 in flight). If both hold, the rule is "check your anchor's
  loss curve before concluding anything about anchors."
- **Scope warning (theirs, adopted):** dropping the anchor is safe for POOL-RESTRICTED PL
  (top-M support, our v3) but collapsed outright on text when the partition function was
  normalized over the full corpus (R@5 1–11 vs base 55) until a frozen-base trust region was
  added. If v4 ever moves to corpus-wide normalization, expect to need KL-to-frozen-base —
  and note their failure was a drift-blind KL reference, not KL itself.
- **Ceiling story confirmed on both sides.** Their step sweep (SQuAD 10k pool, 0.6B):
  47.2 → 49.0 (peak at ~6.4k visited) → 47.8 → 48.6 with recall monotonically falling; our 2B:
  plateau ~0.345 at 2–6k visited, degradation at 12k. Joint conclusion: these recipes are a
  few-thousand-example correction, not a scalable training paradigm; capacity is the only
  lever that moved either ceiling.

## Anchor taxonomy, amended (2026-08-12, after cross-side falsification)

The text side proposed: external positive -> binding; self-argmax positive -> inert BY
CONSTRUCTION; stochastic positive -> destructive. Our data falsifies the middle clause as
stated: every binding anchor quoted above (nce 0.48-0.63 at end of training) is a SELF-ARGMAX
positive (no_force_gold: candidate 0 = own top-1), not an external one.

The repair is the negative SCOPE, not the positive type alone. Our infonce_loss ranks the
positive against ALL B x P docs in the batch — cross-query negatives from other entities'
pools — while the argmax was taken only over the query's own pool. The CE task therefore
contains comparisons the argmax does not automatically win, and at contrastive temperature
0.03 those cross-entity margins keep the loss alive.

  Amended rule: an anchor is inert iff its negative set is contained in the comparison set
  over which its positive was selected. Self-argmax + own-pool negatives = born satisfied.
  Self-argmax + cross-query negatives = binding (ours). External positive = binding
  regardless. Stochastic positive = harmful regardless.

Consequence for the joint writeup: our +0.009-0.011 from anchor removal is the removal of a
binding SELF-ARGMAX anchor — i.e. the proxy being traded away was "top-1 stability under
cross-entity contrast," not gold relevance. That makes the objective/proxy trade purer: no
label enters even through the anchor, and pure RL still wins.

## Anchor taxonomy, final (margin-at-temperature) — 2026-08-12

Both single-factor rules died on the other side's data: "self-argmax -> inert by construction"
(theirs) is falsified by our binding self-argmax runs; "inert iff negatives within the
positive's comparison set" (ours) is falsified by their inert self-argmax + cross-query cell.

Surviving rule (theirs, adopted): **an anchor is inert iff no in-scope negative sits within
~O(few*tau) of the positive.** Positive type and negative scope matter only through the
margins they induce.

Direct measurement on our side (base policy, B=4 batches, P=24 pools, 48 queries, ctemp=0.03):
gap between own-top-1 and the best cross-query doc, in units of tau: min 0.00, p10 2.21,
median 6.38, p90 9.07; 17% of queries competitive (gap < 3*tau); 0% inverted (gap < 0).
Exactly the predicted signature of our runs: a minority of live margins keeps InfoNCE at
~0.5 (binding) while the argmax positive always wins its own row (never destructive).
Their inert cell: same construction, but B=4 Trivia batches put cross-query docs topically
far away — gaps >> tau, loss born at 0.44 and gone by step 240.

Reading of the v3 result under the final rule: what anchor removal deleted was hard-margin
uniformity pressure on the ~17% of queries with near-entity competitors — which is
recall-shaped by definition. Pure RL reallocates those queries' gradient from "keep the top-1
separated" to "rank what the reader can answer from," hence R@5 down, accuracy up.

Discriminating cell (pre-registered, in flight on the text side): plgcc0-ng removes their
inert anchor -> margin rule predicts no-op; our withdrawn scope rule predicted small positive.
plgcc0-gold removes a binding external anchor -> tests proxy-for-objective transfer.
