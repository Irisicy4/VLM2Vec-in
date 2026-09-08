# Annotation-free retrieval RL for multimodal RAG — running results

Living document. Numbers are copied from `$MMRAG_DATA/results/*.json`; every row names its seed
count. House rules apply throughout: **1 seed = recorded, not claimed**; no margin without matched
seeds and a matched estimator; **name the metric in every claim** (accuracy and recall routinely
move in opposite directions here).

**Setup.** Policy = a multimodal encoder (GME-Qwen2-VL-2B, LoRA r32) that embeds an
(image, question) pair. Environment = a frozen VLM reader that answers the question from the
retrieved passages *while seeing the same image*. Reward = whether it answers correctly. No
relevance label is used anywhere (`--no_force_gold`); the image is never annotated.
Estimator = PL-GRPO: G=4 Plackett-Luce sampled ordered lists of k=4 from the policy's own top-24,
group-relative advantage, no critic, no KL, no contrastive anchor ("v3-pure").
Eval = held-out InfoSeek, retrieval over the full corpus (n=3000), top-5 passages to a frozen
Qwen2.5-VL-**7B** reader (n=1500). Train and eval readers deliberately differ.

**Metrics.** entR@5 = entity recall@5 · ansR@5 = answer recall@5 · acc = VQA cover-EM accuracy.

**Seed noise floors** (2x pooled between-seed SD over every multi-seed family on disk;
n=4/3/2/2 families): **entR@5 2.40 · ansR@5 1.91 · acc 0.61**. A single-seed gap below these is
not callable. These supersede an earlier bound taken from one two-seed pair, which overstated the
entR@5 floor 1.8x and understated acc 2x.

---

## Main result — best DYNAMIC no-gold RL vs relevance-SFT vs frozen base

**What makes the winning recipe dynamic.** The retrieval index is not precomputed and fixed: the
RL corpus is **re-encoded with the current policy weights every 100 steps** (`--refresh_steps 100`),
so the action space the policy samples from moves with the policy.

> **Definition — "periodic full refresh"** (`train_rl.py:262-271, 340-356`)
>
> *Index membership is fixed for the whole run.* At startup `build_rl_corpus` keeps every passage
> of every gold-entity URL in the training rows, plus `--n_distractor_articles` randomly chosen
> (seeded) non-gold article URLs, again keeping *all* passages of each kept URL. For
> `corpus_small` + 3000 that is 173,755 passages. Refresh never adds, removes or reweights a
> passage — **only its embedding changes.**
>
> *"Full"* = `refresh_corpus()` re-encodes **all** of `corpus_texts` in chunks of `--encode_bs` and
> replaces the whole `corpus_emb` matrix (fp32, on device). Not incremental, not a subset.
>
> *"Periodic"* = once at init (step 0), then at every optimizer step where
> `step % refresh_steps == 0`. For a 3000-step run at `refresh_steps 100` that is **30 full
> re-encodes** (init + steps 100, 200, … 2900), measured at 749 s each on a 173,755-passage index,
> i.e. **6.24 h = 53% of that run's 11.77 h wall clock.**
>
> *Consequence — bounded staleness.* Between refreshes the index is stale: retrieval at step *k*
> scores a query embedded with step-*k* weights against document embeddings from step
> 100·⌊*k*/100⌋. The train/index skew is therefore **0 to 99 steps, sawtooth**, reset to zero at
> each refresh.
>
> *Interactions.* The branch is gated `elif args.index_ema == 0`, so **index-EMA replaces periodic
> refresh entirely** rather than adding to it — that is why index-EMA is a compound treatment
> (blend coefficient *and* schedule) and why its staleness profile is different: monotonically
> rising toward a plateau of |index| / docs-per-step ≈ 472 instead of a 0-99 sawtooth. The encode
> also runs inside `doc_tower()`, a no-op unless model-EMA is on, in which case documents are
> encoded with the EMA weights while queries stay live.
>
> *What has never been measured:* `--refresh_steps 0`, i.e. the same recipe with the index frozen
> at its init embeddings. That is the control that would price the dynamism itself. That is the dynamism carried by
every RL row below, and it is part of the recipe that beats SFT. The EMA variants (index-EMA,
model-EMA) are *different* dynamic schedules layered on top; none of them has yet added anything
measurable — see "dynamic-schedule variants" below.

GME-Qwen2-VL-2B, InfoSeek, 500 steps (~2k query draws).

| arm | entR@5 | ansR@5 | acc | seeds |
|---|---|---|---|---|
| frozen base (zero-shot) | 68.53 | 54.94 | 30.80 | 1 |
| relevance-SFT (annotated) | 73.63 | 59.67 | 32.85 ± 0.90 | 3, per-seed best ckpt |
| **v3-pure — no-gold RL, no anchor** | **76.09** | **63.46** | **34.71 ± 0.19** | 3 |
| v2 — same + contrastive anchor | 80.13 | 65.46 | 33.82 ± 0.60 | 3 |

**RL beats SFT on every column.** acc margin **+1.86 = 3.5 SE** with matched seed counts and the
same estimator on both legs (SFT scored at its *best* checkpoint — the estimator that favours SFT).
Per-seed McNemar vs SFT-best: significant in 2 of 3 seeds. Against base, every seed is significant
(p ≤ 4.8e-05). RL − base = **+7.56 entR@5 / +8.52 ansR@5 / +3.91 acc**.

**The anchor is a trade, not an improvement.** Adding the contrastive term buys ~+4 entR@5 and
+2 ansR@5 and costs ~0.9 acc. Ordering is metric-dependent: v3 > v2 > SFT > base on accuracy;
v2 > v3 > SFT > base on recall. Neither RL arm loses to SFT on any column.

**Out of domain (E-VQA, never trained on).** base 76.31/64.84/42.07 · SFT-best 78.38/66.73/43.47 ·
v3-pure 81.34/72.18/45.87. Over 3 seeds RL gains +2.6 ± 1.3 acc / +3.9 ± 1.2 entR@5 vs base,
about **2x the SFT transfer** (+1.4/+2.1, 1 seed).

**Standing caveat — exposure asymmetry.** RL visits ~2,000 queries (500 steps x batch 4); SFT
visits ~19,200 (600 x 32). RL wins on ~10x fewer query visits, and with no relevance labels.

### Dynamic-schedule variants — which flavour of "dynamic" actually pays

| schedule | what changes | best result | vs its matched control |
|---|---|---|---|
| **periodic full refresh (`--refresh_steps 100`)** | full index re-encode every 100 steps | **76.09 / 63.46 / 34.71** (3 seeds) | this IS the winning recipe; beats SFT +1.86 acc |
| index-EMA (`--index_ema 0.9`) | no periodic refresh; per-step blended write-back of the top candidates, near-boundary hard negatives and the stalest docs | 77.23 / 65.84 / 34.47 @ck750 | **+1.60 / +1.06 / +0.60 — all inside floors, not callable** |
| model-EMA (`--model_ema 0.99`) | doc tower encoded with EMA'd weights | 72.47 / 60.49 / 32.20 | **UNUSABLE** — trained q(live)·d(EMA), evaluated both live |
| frozen-index variant (`*-frozen`) | undocumented | 78.43 / 63.02 / 33.60 (v1 family, 1 seed) | **cannot be used** — the axis it varies is not recorded anywhere |

**Reading.** The dynamism that is established as load-bearing is the plain periodic re-encode; it
is inside the recipe that beats SFT. Every *refinement* of that schedule is so far null or
unusable. Index-EMA's apparent +2.29 ansR@5 came from a cross-cluster baseline; against a
same-hardware control it halves to +1.06 and falls under the floor, and its slope matches the
control's, so it moves the level rather than the trajectory. Model-EMA has never been cleanly run.
A `*-frozen` v1 arm exists whose numbers sit ~1 point below its `-det` twin, but neither
`RESULTS.md` nor any doc records what "frozen" varies, so it cannot serve as the static-index
control the comparison needs. **Building that control — the same recipe with the refresh disabled —
is the missing measurement that would price the dynamism itself.**

---

## Q1 — Does RL work better on multi-source training data?

**Answer so far: no measurable gain, and a small cost on the in-domain benchmark. Unresolved on a
diverse INDEX, which is the version now running.**

Two different things are being varied and they must not be confused: the *training pool* (which
queries) and the *retrieval index* (what the policy searches).

**(a) Diverse training pool, same InfoSeek eval** — 1 seed per leg:

| arm (pool_div45k_v3) | entR@5 | ansR@5 | acc |
|---|---|---|---|
| relevance-SFT, best ckpt | 75.87 | 60.41 | 32.07 |
| v3-pure | 76.83 | 64.57 | 33.00 |
| RL − SFT | +0.96 | **+4.16** | +0.93 |

RL still beats SFT on the same pool, and the answer-recall gap (+4.16) is the largest of the three.
But v3-pure on the *diverse* pool scores acc 33.00 against 34.71 ± 0.19 on the InfoSeek-only pool —
i.e. **diversity costs ~1.7 acc on an InfoSeek benchmark**, plausibly a train/test distribution
mismatch rather than a failure of diversity. 1 seed; not claimed.

**(b) Mixed-source pools at matched draws** — flat:

| pool | rows | entR@5 | ansR@5 | acc |
|---|---|---|---|---|
| InfoSeek 25k | 25k | 77.57 | 66.45 | 34.00 |
| InfoSeek 200k | 200k | 77.00 | 66.37 | 33.47 |
| mixed 275k | 275k | 76.53 | 66.49 | 34.67 |
| mixed 1.06M | 1.06M | 76.77 | 66.16 | 34.33 |

Source diversity in the pool neither helps nor hurts at matched draws.

**(c) Diverse retrieval INDEX — IN PROGRESS.** Every index used above is homogeneous encyclopedic
Wikipedia, so nothing has ever tested whether the retriever can reject an *off-domain* passage.
`corpus_mix2M` (2,000,000 passages: 63% the existing multimodal corpus, 37% BEIR across msmarco /
hotpotqa / quora / trec-covid / touche / fiqa / scidocs / arguana / scifact / nfcorpus) is now the
training index for a 4-cell grid. Eval stays on the original corpus so numbers remain comparable.
Trap recorded: this corpus averages **1.278 passages per document** (98% single-passage) against
~16 for Wikipedia, so the standard distractor setting of 3000 yields ~4.2k index passages instead
of ~48k. It must be scaled ~12x (42,484 used here) or the index is silently 12x too easy.

---

## Q2 — Does RL keep learning from more data, where SFT does not?

**Answer: no. Both plateau, and the plateau is caused by the reward, not by the data.**

**Relevance-SFT scaling** — 11 checkpoints, steps 2100 to 27300, least squares per 10k steps:

| metric | slope | 95% CI | verdict |
|---|---|---|---|
| entR@5 | +0.131 | [−0.565, +0.827] | flat |
| ansR@5 | +0.399 | [−0.517, +1.315] | flat |
| acc | +0.427 | [+0.032, +0.822] | rising, but only ~1.2 acc across the whole line |

**RL consumption** — 34.00 ± 0.47 at 2k draws, 34.80 at 6k, 34.07 at 12k. A plateau in the 34-35
band; 34.80 is not quotable as a maximum because the span is smaller than the 3-seed spread.
**Pool size at matched draws is fully null on both metrics** — 25k, 200k and 810k rows all land in
the same place, and an apparent retrieval gain at 810k failed seed validation (78.70 vs 74.37).

**Scale context that reframes the question.** 810k rows at batch 4 is ~202,500 steps = one epoch.
Every run on record is ≤3000 steps = 12k draws = **1.5% of an epoch**, and turnover appears at
~2-3k draws, i.e. ~1% of an epoch. "Keeps improving through one epoch" is not merely untested; the
recipe stops improving two orders of magnitude before that point.

**Mechanism (measured, and the reason more data cannot fix it).** The reward is *starved*, not
saturated. Over 924 steps:

| gold in pool for | steps | degenerate groups | mean reward |
|---|---|---|---|
| all 4 queries | 294 | **0.532** | 0.255 |
| 3 of 4 | 379 | 0.613 | 0.200 |
| 2 of 4 | 215 | 0.666 | 0.155 |

Degeneracy-heavy steps have *low* reward (median 0.125; none above 0.9), so degenerate groups are
all-wrong, not all-right. **Even when the gold passage is retrieved for every query, 53% of groups
carry zero advantage**: the 3B reward reader converts a correct retrieval into a correct answer
only 20-25% of the time, and when it fails on all sampled lists the rewards tie, the advantage is
exactly zero, and the step teaches nothing. The signal also thins over training (degenerate
0.586 → 0.621, group spread 0.0403 → 0.0376, both CIs excluding zero) while retrieval and reward
stay flat. A matched control with the EMA flags removed reproduces this within 1-2%, so it is a
property of the recipe under this reward, **not** of any index treatment.

**Consequence.** The bottleneck sits downstream of retrieval. That predicts what the campaign
observed and could not explain: every pool size plateaus in the same place, accuracy and recall
peak at different draws, and the SFT line is flat too. A recipe that keeps improving must first
stop wasting ~60% of its steps on zero-advantage groups.

**Index-EMA, tested as one candidate fix — null.** Against a same-hardware, same-seed control with
only the two EMA flags removed, at ck750: entR@5 +1.60, ansR@5 +1.06, acc +0.60 — **all inside the
noise floors**. Against the published cross-cluster baseline the same arm read +2.29 ansR@5 and
looked callable, so roughly half the apparent effect was cross-site artifact. Over 3 checkpoints
the arm's slope matches the baseline's (−0.60 vs −0.58 per 1k steps): EMA moves the level, not the
trajectory — the wrong shape for "keeps improving", even where the level shift is real.

---

## Q3 — How big, and what pretraining, must the backbone be for RL to help?

**Answer: size is nearly irrelevant; retrieval pretraining is decisive.** A 2B retrieval-finetuned
encoder learns; a 7B embedding-trained one destroys its own retrieval.

Entity R@5, zero-shot → after RL at the best tested learning rate:

| base | retrieval-finetuned? | zs entR@5 | after RL | outcome |
|---|---|---|---|---|
| GME-Qwen2-VL-7B | yes | 69.37 | **80.97 / 80.30** | learns (2 seeds) |
| GME-Qwen2-VL-2B | yes | 68.53 | **76.09** | learns (3 seeds) — the headline |
| BGE-VL-large | yes | 15.73 | **21.23** @1e-5 | learns, but only in a one-decade LR window |
| BGE-VL-base | yes | 14.73 | 8.90 @1e-5 | **fails** at its larger twin's rate |
| VLM2Vec-2B | no (embedding-trained) | 15.67 | 0.17–0.70 | drift — retrieval destroyed |
| VLM2Vec-7B | no | 14.37 | 2.90 | drift — capacity does not rescue |
| VLM2Vec-4B | no | 12.77 | no result | reward degenerate (group spread 0) at step 249 |
| SigLIP2-so400m | no (contrastive only) | 8.23 | 6.60 @1e-5 · 0.40 @1e-4 | no learning |
| Qwen2.5-VL-3B (raw) | no | 3.73 | 0.00–0.60 | dead at every rate |
| CLIP-L | no | ~5.4 | 0.0043 | contraction; SFT on the same base reaches 61.30 |
| Qwen2-VL-2B (raw) | no | 0.03 | 0.00 | no zero-shot signal to train on |

**Three failure phenotypes, distinguishable by their signature.** *Contraction* — the index
collapses onto a handful of documents (CLIP-B16 retrieves 96 distinct docs out of 3000). *Drift* —
embeddings move without collapsing and retrieval dies anyway; robust to a 4x learning-rate change
and to capacity. *Inert* — measurable parameter movement buying no measurable retrieval change.

**Working range: zero-shot entR@5 in (0.157, 0.685].** RL fails at 0.157 and works at 0.685; the
boundary has not been narrowed further. **Learning-rate windows are per-model, not per-family** —
BGE-VL-large learns at 1e-5 where BGE-VL-base fails at the same rate, and 1e-5 is an interior
maximum for the large model, not a boundary.

**Retracted along the way:** "the contrastive anchor rescues contraction-prone bases" is dead. It
rested on one seed reaching 17.47 on SigLIP2; seeds 1 and 2 gave 1.10 and 0.13, a 3-seed mean below
zero-shot. Anchored runs on CLIP-B16 (1.13) and BGE-VL-base (3.77) still contract.

**Caveat on the weak rungs.** "Learns" means retrieval improves. For BGE-VL-large it does *not* yet
mean useful RAG: its accuracy (16.93) remains below the reader's no-context floor (19.27), so its
retrieval is still net-harmful to the reader.

---

## Open / in flight

- **mix2M grid (4 cells, running).** {v3-pure, v2-anchored} x {EMA, no EMA} on the 2M mixed-domain
  index, 1500 steps, matched pairs on the same hardware. First test of index-EMA on the anchored
  recipe, and first test of any recipe against off-domain distractors.
- **Reward-side levers, none tested at scale on the current estimator**: `--reward logit`
  (continuous, always defined — should collapse degeneracy toward zero; note it lost badly on the
  *older* GRPO-8 estimator, ansR@5 0.5404), `--reward gain`, `--reward mix`, `--reward_gate_std`,
  and `--reward_gate_noctx` (tried once: +0.94 ansR@5, 1 seed, right direction). These attack the
  Q2 mechanism directly and are the highest-leverage untried axis.
- **Ceiling.** ~0.345-0.350 acc at 2B regardless of data, compute, estimator or anchor; the 7B rung
  reaches 0.351-0.357. Capacity is the only lever that has ever moved it.
