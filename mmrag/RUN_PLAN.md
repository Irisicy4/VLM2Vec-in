# Run plan — "dynamic" schemes, updated against current results

Scope note: **"dynamic" here means four specific EMA schemes** (user's definition, 2026-09-08):

1. **index-EMA** — re-encode a subset of candidates each step, write back a weighted average of
   the new and old embedding.
2. **selected-candidate + pool-EMA + rarity + hard negatives** — use the selected top candidates to
   drive the weight update, EMA the pool, deliberately include less-frequently-sampled docs, and
   add hard negatives.
3. **model-EMA** — compute with EMA'd model weights.
4. **EMA-as-regularizer** for full fine-tuning.

---

## Status at a glance

| scheme | implemented? | tested? | verdict so far |
|---|---|---|---|
| 1 index-EMA | yes, `--index_ema` / `--index_refresh_extra` | 45k, 810k, mix2M running | **null so far** |
| 2 candidates+rarity+hard-negs | **mostly — it IS scheme 1's implementation** | same runs as scheme 1 | folded into 1 |
| 3 model-EMA | yes, `--model_ema` | once, **confounded** | **UNANSWERED — priority 1** |
| 4 EMA-as-regularizer | **no — needs code** | never | unstarted |

---

## Scheme 1 — index-EMA. Near-answered, and the answer is null.

`--index_ema m` replaces the periodic full index refresh entirely: one full encode at init, then
per step it writes `index[p] <- l2norm(m*old + (1-m)*new)` for a chosen subset. **It is a compound
treatment** — blending coefficient *and* refresh schedule change together — so no result can be
attributed to `m` alone.

Evidence:

| comparison | entR@5 | ansR@5 | acc |
|---|---|---|---|
| 45k pool, 500 steps: emaidx vs v3-pure | −1.06 | −1.43 | −0.07 |
| 810k, ck750, vs **same-hardware control** | +1.60 | +1.06 | +0.60 |
| floors | 2.40 | 1.91 | 0.61 |

**Nothing is callable against the matched control.** Against the cross-cluster published baseline
the same arm read +2.29 ansR@5 and looked callable — roughly half of that was site artifact, which
is why the control existed. Across ck750/1500/2250 the arm's slope matches the baseline's
(−0.60 vs −0.58 per 1k steps): EMA shifts the level, not the trajectory.

**Remaining:** the mix2M grid (running) tests it on a diverse index and, for the first time, on the
anchored recipe. Cells `mix2m-v3pure{,-ema}` and `mix2m-v2anchor{,-ema}`, 1500 steps, matched pairs
on the same hardware. This closes scheme 1.

## Scheme 2 — already inside scheme 1. Only one component is genuinely new.

Reading `train_rl.py:401-422`, the `--index_ema` branch already does all three things the scheme
asks for, every step:
- **selected top candidates** — `pool_ids = topk[b][:N]` are EMA-written from the fresh encodings
  the policy just computed (line 401, 417);
- **hard negatives** — `boundary_ids = topk[b][N:]`, commented in-source as *"near-boundary = free
  hard negatives"*, re-encoded and written back (402, 419-422);
- **less-frequently-sampled docs** — `ema_refresh_stale()` re-encodes the *stalest*
  `--index_refresh_extra` docs by `doc_last_upd` (322-330).

**So scheme 2 is not a separate arm; it is what scheme 1 has been running all along**, and it
inherits scheme 1's null. Do not schedule it as a new cell — that would re-run the same treatment
under a new name.

**The one genuinely new part**: hard negatives entering the **loss**, not just the index.
`--num_hard_negs` exists in `train_sft.py` only; `mine_hard_negs.py` exists. Adding mined negatives
to the RL objective is a code change, and note the nearest existing evidence is discouraging — the
contrastive anchor (whose negatives are cross-query docs) costs ~0.9 acc for ~4 recall.

## Scheme 3 — model-EMA. The real gap. **Priority 1.**

Implemented: `--model_ema m` keeps an EMA copy of the encoder and a `doc_tower()` context manager
that swaps EMA weights in for **document** encoding while queries stay live.

**It has been run exactly once and that run is unusable.** `div45kv3-emaenc` (72.47/60.49/32.20)
trained with q(live)·d(EMA) but *evaluated* with live weights for both towers — a train/eval
geometry mismatch, not a result. The fix landed (the EMA tower is now saved to `ema_tower/`,
line 668) but the clean rerun was never done. It is also the cell the EMA handoff registered as
"cell 2" and it has been outstanding since.

**Prior that makes it worth running rather than assuming:** the geometry probes show RL moves the
**query** tower a great deal (mean pairwise cosine 0.2212 → 0.0594) while the doc side barely
moves. Model-EMA damps the doc tower — a priori aimed at the tower that is not moving. That is an
argument for a *null*, and it is cheap to settle.

**Cell:** `emaenc2-clean` — v3-pure args + `--model_ema 0.99`, plus a matched no-EMA twin on the
same hardware. Verify before trusting: `ema_tower/` exists in the run dir and the eval loads it.

## Scheme 4 — EMA as a regularizer. Needs code; weakest prior.

Not implemented. Two readings, both requiring changes:
- **KL or L2 to the EMA weights** as a trust region. Cheap to add. But KL-to-init is already
  measured and it *hurts*: β=0.01 → −0.7 acc, β=0.1 → −1.7 acc, and the ledger closed that axis
  with *"bare > KL on MM; the regularizer must match the failure mode the base actually has, and
  gme2b has none."* An EMA reference is a moving target rather than a fixed one, so it is not
  strictly the same experiment — but the prior is poor.
- **Full fine-tuning with EMA weight averaging.** Everything on record is LoRA r32. Full FT changes
  the optimisation regime, the memory profile and the collapse risk simultaneously, so a null or a
  win would be hard to attribute.

**Recommendation: last.** Only after scheme 3 resolves, and only in the KL-to-EMA form first.

---

## The cross-cutting risk, stated plainly

**All four schemes act on the index or the encoder weights. The measured bottleneck is neither.**
Even when the gold passage is retrieved for all four queries in a step, 53% of reward groups carry
zero advantage, because the 3B reward reader converts a good retrieval into a correct answer only
20-25% of the time. A matched control reproduces that within 1-2%, so it is a property of the
recipe under this reward. **If the gradient is absent for ~60% of steps, no change to what the
retriever searches or how its weights are averaged can restore it** — which is a concrete reason
to expect schemes 1-4 to come out null, and scheme 1 already has.

That does not make them not worth finishing; scheme 3 is registered and cheap. But the ordering
should reflect it.

## Ordering

1. **Finish the mix2M grid** (running) — closes scheme 1 on a diverse index and on the anchored
   recipe. No new compute needed.
2. **Scheme 3, clean model-EMA + matched twin.** The only registered cell never cleanly run; two
   GPUs, ~1500 steps.
3. **Reward-side diagnostic, one cell.** Not one of the four schemes, but it tests the mechanism
   that predicts all four are null: run v3-pure with `--reward_gate_noctx` (drops queries retrieval
   cannot help; tried once, +0.94 ansR@5, 1 seed) or `--reward logit` (continuous, always defined —
   degeneracy should collapse toward zero; caveat: it lost badly on the *older* GRPO-8 estimator,
   ansR@5 0.5404, and has never been tried on PL-GRPO). Instrument `reward/group_degenerate_frac`:
   if it does not fall well below 0.5, the mechanism claim is wrong and schemes 1-4 deserve more
   weight, not less.
4. **Scheme 2's new component** — mined hard negatives in the RL loss. Code change.
5. **Scheme 4** — KL-to-EMA first, full FT only if that shows anything.

## Standing protocol for every cell above

Matched no-EMA twin on the **same hardware in the same week** — the cross-site artifact was ~half
the apparent scheme-1 effect. Floors entR@5 2.40 / ansR@5 1.91 / acc 0.61; below them, not callable
on one seed. Name the metric in every claim. Register the reading before the numbers exist.
