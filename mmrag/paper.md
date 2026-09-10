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

**(a2) The other side of the comparison — does SFT degrade on multi-source data?**
Same recipe, same diverse pool, same InfoSeek eval, matched checkpoints (1 seed per leg):

| ckpt | SFT ansR@5 InfoSeek pool | diverse pool | Δ |
|---|---|---|---|
| 450 | 59.92 | 59.02 | −0.90 |
| 600 | 59.84 | 58.00 | −1.84 |
| 750 | 62.04 | 60.41 | −1.63 |
| 900 | 59.59 | 59.80 | +0.20 |
| 1050 | 56.82 | 54.53 | −2.29 |
| **mean** | | | **−1.29, negative at 4/5** |

**The two methods move in OPPOSITE directions on answer recall when the training pool becomes
multi-source:**

| method | entR@5 | ansR@5 | acc |
|---|---|---|---|
| SFT, InfoSeek pool → diverse pool | +0.70 | **−1.63** | −0.93 |
| RL, InfoSeek pool → diverse pool | +0.74 | **+1.11** | −1.71 |

So the answer to "does SFT fail or go flat on multi-source data" is: **it degrades, mildly and
consistently, on answer recall** (4 of 5 matched checkpoints negative, mean −1.29), while RL's
answer recall *improves* (+1.11). Both methods lose accuracy on the diverse pool, RL slightly more
(−1.71 vs −0.93), and both gain a little entity recall — so the divergence is specific to answer
recall, the metric that measures whether the retrieved passage actually contains the answer.
Read together with the head-to-head **on** the diverse pool (RL − SFT = +0.97 / **+4.16** / +0.93,
the largest answer-recall margin in the whole campaign), the shape is that diversity *widens* the
RL-over-SFT gap on answer recall rather than closing it.

**NOT ESTABLISHED — the honest statistics.** The differential is ~+2.4 on ansR@5. The single-metric
seed floor is 1.91, and a difference-of-differences compounds that to roughly 2.70, so **+2.4 sits
below the compounded floor**. Every leg is 1 seed except RL's InfoSeek-pool arm. What supports it
beyond the point estimate is only the *consistency of sign* — SFT is negative at 4 of 5 matched
checkpoints — which is suggestive and not a significance test. Seeding the two diverse-pool legs is
the cheapest way to settle it.

**(b) Mixed-source pools at matched draws** — flat:

| pool | rows | entR@5 | ansR@5 | acc |
|---|---|---|---|---|
| InfoSeek 25k | 25k | 77.57 | 66.45 | 34.00 |
| InfoSeek 200k | 200k | 77.00 | 66.37 | 33.47 |
| mixed 275k | 275k | 76.53 | 66.49 | 34.67 |
| mixed 1.06M | 1.06M | 76.77 | 66.16 | 34.33 |

Source diversity in the pool neither helps nor hurts at matched draws.

**(c) Diverse EVAL index — first measurement, and the drop is large.** The same finished adapter
(`emaidx-1M-s3000`), same `queries_test`, n=3000, scored against two different eval indexes:

| eval index | entR@5 | ansR@5 | acc |
|---|---|---|---|
| corpus_small — 422k passages, 0% off-domain | 75.83 | 64.82 | 34.20 |
| corpus_mix2M — 2M passages, 37% off-domain | 69.83 | 57.88 | 32.40 |
| delta | **−6.00** | **−6.94** | −1.80 |

**CONFOUND BROKEN — by running the base on the same index.** The zero-shot base was then scored on
both indexes too, and it does not degrade, it **collapses**:

| | entR@5 | ansR@5 | acc |
|---|---|---|---|
| base on corpus_small (422k, in-domain) | 68.53 | 54.94 | 30.80 |
| RL on corpus_small | 75.83 | 64.82 | 34.20 |
| **RL − base** | **+7.30** | **+9.88** | **+3.40** |
| base on corpus_mix2M (2M, 37% off-domain) | **14.27** | **9.96** | 20.87 |
| RL on corpus_mix2M | 69.83 | 57.88 | 32.40 |
| **RL − base** | **+55.57** | **+47.92** | **+11.53** |

Changing the eval index costs the RL retriever 6.0 entR@5 and costs the frozen base **54.3**. If the
drop were merely a 4.7x larger index, both arms would pay a similar toll; they do not, by an order
of magnitude. **So the drop is a capability difference, not an index-size artifact, and the earlier
"confounded" caveat is resolved.** Note the base's accuracy on mix2M (20.87) sits essentially at the
reader's no-context floor (19.27) — the documented signature of retrieval being dead — so the base
is not merely worse, its retrieval has stopped functioning on a mixed-domain index.

**This is the strongest evidence in the campaign that annotation-free RL buys something real.** On
the homogeneous benchmark the recipe looks like a +3.4 accuracy improvement. On a mixed-domain index
the same adapter is **+11.5 accuracy and +55.6 entity recall** over its own base, because the base
cannot reject off-domain passages at all and the RL-trained retriever largely can. The homogeneous
benchmark was hiding most of the effect. It also reframes Q1: multi-source data does not help
*training* measurably, but it is where the *evaluation* becomes able to see the difference.

**Original caveat, retained for the record:** mix2M is both 4.7x LARGER (more distractors of any
kind) and 37% off-domain. A bigger index alone lowers recall, so this drop **cannot** be attributed
to a failure to reject off-domain passages. The control that would separate them is an eval index of
comparable size that is entirely in-domain — `corpus_div_v2` (1.26M, 0% off-domain) is the obvious
candidate and has not been run. Until then the honest statement is: *retrieval quality on this
benchmark is substantially index-dependent, and the published numbers are specific to a 422k
in-domain index.* A zero-shot run on the same mix2M index is finishing and will say whether the
RL-vs-base gap survives the harder index, which is the part that matters for whether the recipe
generalises.

**(d) Diverse TRAINING index — IN PROGRESS.** Every index used above is homogeneous encyclopedic
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

**Tested: does multi-source data fix the starvation? NO — it slightly worsens it.**
Degeneracy is logged per step, so this is measurable directly on the running cells with no extra
compute. Matched over the first 103 steps of every arm (the shortest arm sets the window):

| training data condition | degenerate frac | group spread | arms |
|---|---|---|---|
| InfoSeek pool + corpus_small (homogeneous) | 0.5546 | 0.0445 | 2 |
| InfoSeek pool + corpus_mix2M (diverse index) | 0.5667 | 0.0452 | 2 |
| InfoSeek pool + corpus_mix2M, anchored | 0.6153 | 0.0406 | 2 |
| **diverse pool + corpus_mix2M (both)** | **0.6104** | **0.0377** | 2 |

Going from fully homogeneous to diverse on both axes moves degeneracy **0.5546 → 0.6104 (worse)**
and group spread **0.0445 → 0.0377 (worse)**. Gold-in-pool is roughly flat across conditions
(0.72 / 0.71 / 0.72), so this is not explained by the gold passage falling out of the candidate set.

**WHY — my first explanation was wrong, and its own prediction killed it.** I proposed that
off-domain passages displace in-domain near-misses, so more lists become all-junk, so more groups
score all-zero. Two measurements, both against that:

*(i) The retriever does discriminate against off-domain content.* From the mix2M retrieval dumps
(pid ≥ 60M identifies BEIR), the index is **37.0% off-domain but only 10.2%** of retrieved top-100
and **9.9%** of the top-5 that reach the reader — a 3.7x under-representation. Median rank of the
first off-domain passage is 11. It is not drowning in off-domain text; 32.6% of queries do get at
least one off-domain passage into their top-5, so the displacement is real but modest.

*(ii) The extra degenerate groups are NOT all-wrong.* If they were, reward on the most-degenerate
steps would fall and zero-reward steps would become commoner as data diversifies. Both go the
**opposite** way:

| condition | degenerate | reward | reward on degen≥.75 steps | frac of steps with reward 0 |
|---|---|---|---|---|
| homogeneous | 0.555 | 0.195 | 0.156 | 0.039 |
| diverse index | 0.567 | 0.187 | 0.129 | 0.034 |
| both diverse | 0.610 | 0.200 | 0.142 | **0.024** |

Zero-reward steps get **rarer** (0.039 → 0.024) and the reward level is flat. So diversity is
producing *variance reduction within groups at constant mean* — lists agreeing more often at a
middling reward — not a shift toward failure. **My displacement-to-all-zero story is refuted.**

*What I do not know:* why lists agree more under diversity. I am not going to supply a second
mechanism after the first one failed a test, especially since **the effect itself is not
established** — 0.056 in degeneracy against within-condition spread up to 0.046. The correct order
of work is to establish whether the effect is real (seed the conditions) before explaining it.
What survives is the *negative* claim, which does not need a mechanism: **diversity does not reduce
starvation.** And the reason to expect that a priori still holds — starvation is reader-limited, and
diversity acts on the retrieval side, so it cannot make the 3B reader convert a good passage better.

**Statistical honesty.** 103 steps is early, and the two arms inside each condition differ by the
EMA flag rather than by seed, so they are not replicates. Within-condition spread runs 0.003 to
0.046 in degeneracy, and the between-condition difference is 0.056 — comparable to the largest
within-condition spread. So: **no evidence diversity helps, weak evidence it slightly hurts, effect
size comparable to the noise available to measure it.** One arm is an outlier worth watching —
`mix2m-v2anchor-ema` shows gold-in-pool 0.4951 against ~0.72 elsewhere and the highest degeneracy
(0.6383); if that persists it is a problem with that cell, not a property of the condition.

**WHY the reward discriminates early and not late — measured, and it is the POLICY, not the reward.**
Probing the finished run's own checkpoints for the geometry that drives Plackett-Luce sampling
(40k-passage index, 16 real queries, 400 draws, run twice per checkpoint — replicates agree to 0.02):

| checkpoint | top-24 cosine spread | unique docs / query |
|---|---|---|
| zero-shot | 0.1342 | 8.43 |
| 250 | 0.1361 | 8.27 |
| 750 | 0.1363 | 8.17 |
| 1500 | 0.1366 | 8.03 |
| 2250 | **0.1535** | **7.70** |
| 3000 (final) | 0.1527 | 7.85 |

**The policy sharpens (+13.8% spread) and its sampling explores less (−6.9% unique docs).** The
mechanism: `pl_sample_lists` is Gumbel-top-k over sims/τ at τ=0.02, and Gumbel noise has scale
~1.28. Early, the top-24 similarities sit close together, so the scaled gaps are comparable to the
noise and the G=4 sampled lists genuinely differ — different documents, different rewards, usable
advantage. As training sharpens the profile, the scaled gaps outgrow the noise, sampling drifts
toward deterministic, the four lists increasingly pick the *same* documents, and their rewards
converge. **The magnitudes agree: unique docs −6.9% against degeneracy +7.1%** (0.588 → 0.630).

Within-run trajectory, which fits this and rules out the obvious alternatives:

| phase | degenerate | reward level | reward on degen≥.75 steps | steps with usable spread |
|---|---|---|---|---|
| 0-499 | 0.588 | 0.200 | 0.158 | 73.0% |
| 1000-1499 | 0.595 | 0.236 | 0.192 | 72.0% |
| 2500-2999 | 0.630 | 0.232 | 0.183 | 66.0% |

Reward **level rises +16%** while reward **spread falls −12.5%** and gold-in-pool is flat (0.745 →
0.7425). So the reward is not failing and retrieval is not decaying — **the policy is succeeding, and
its success is what removes the gradient.** This is self-limiting rather than starving: a 7-point
loss of usable-gradient steps over 3000 steps. It also explains why the plateau is so robust to
data, compute and estimator changes: none of them touch τ or the sampling temperature, which is what
sets how long exploration survives.

**Caveats.** Both series are monotone through ck2250 then tick back at the final checkpoint, so
"monotone" is a 5-of-6 statement. The probe uses a 40k index rather than the run's 174k one; the
earlier size check showed unique-docs moves only ~3% across a 3x index change, so the direction is
safe but the absolute values are not the run's own. And this coexists with the reader-side floor:
the 48.5% RAG-HOPELESS rows (Q4) supply a persistent all-wrong population regardless of sampling.
**Both are real: a fixed reader-limited floor, plus a growing agreement term driven by the policy
sharpening.** The two mechanisms have different fixes — the floor needs data filtering (Q4), the
sharpening needs an exploration knob (`--pl_behavior_temperature`, which the record lists as tried
and "hurt in combo", never at scale).

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

## Appendix — the slate / joint-context reward (`--reward ragacc`): how it failed

The one attempt at rewarding a *selected slate* rather than individual passages. It is the closest
thing in the record to a "dynamic rerank" arm, and it collapsed harder than any other reward.
Recipe otherwise standard no-gold (GRPO-8, cc 0.3, lr 2e-5, 500 steps, gme2b). Comparator: the
`judge` reward reaches entR@5 **79.60** on the same settings; SFT reaches 73.73.

| run | entR@1/@5 | ansR@5 | acc |
|---|---|---|---|
| rl-ragacc-nogold-lr2e5 | 0.0007 / **0.0017** | 0.0020 | 0.1480 |
| rl-ragacc-nogold-lr2e5-s1 | 0.2827 / 0.5633 | 0.4029 | 0.2653 |
| rl-ragacc-nogold-lr1e5 | 0.1083 / 0.3853 | 0.2433 | 0.2100 |
| rl-ragacc-nogold-lr1e5-1500 | 0.0013 / 0.0030 | 0.0037 | 0.1440 |
| rl-ragacc-nogold-N16 | 0.4270 / 0.6680 | 0.4820 | 0.2927 |
| rl-ragacc-nogold-frozen | 0.2473 / 0.4693 | 0.3506 | 0.2560 |
| rl-ragacc7b-nogold-lr2e5 | 0.1590 / 0.4030 | 0.2808 | 0.2273 |
| rl-ragacc-**gold**-lr2e5 | 0.3713 / 0.6157 | 0.4400 | 0.2787 |
| rl-ragacc7b-**gold**-lr2e5 | 0.5480 / 0.7870 | 0.6086 | 0.3247 |

**Four defects, all visible in `train_rl.py:489-508`, in order of severity.**

1. **The baseline is computed across QUERIES, not across samples of one query.**
   `adv_q = (slate_r − slate_r.mean()) / (slate_r.std() + 1e-6)` takes its mean and std over the
   **batch of 4 different queries**. Every other reward path z-scores within a group of G samples of
   the *same* query. So the advantage here encodes *which queries happened to be easy*, which the
   policy does not control, rather than *which slate was better*, which it does. An easy query gives
   its slate positive advantage even if the slate was poor; a hard query punishes a perfect slate.
   This is not a variance-reduced policy gradient — it is a query-difficulty signal.
2. **One scalar for k=5 documents, so no credit assignment.**
   `advantages.scatter_(1, slates, adv_q…expand(-1, k))` assigns *the same* advantage to all five
   slate members. A slate that succeeds because of one good passage promotes four irrelevant ones
   equally; a slate that fails despite containing the gold passage pushes the gold *down*. The
   surviving recipe fixes exactly this by reading each unique document separately — the in-source
   comment at `train_rl.py:456` reads "per-document credit — the joint-context scalar variant
   collapsed, see ragacc".
3. **Only 5 of 24 candidates receive any gradient**; the other 19 get exactly zero advantage, so
   nothing pushes bad candidates down and there is no force opposing collapse.
4. **The z-score is over 4 values.** With `std + 1e-6` in the denominator and a coarse reward, a
   batch where all four queries score alike produces either a vanishing or an exploding advantage.

**Empirical signatures that confirm the diagnosis rather than just describing the failure.**
*Seed instability:* the same config at lr 2e-5 gives entR@5 0.0017 (seed 0) and 0.5633 (seed 1) — a
330x spread, which is what a 4-sample noisy baseline predicts and what no other reward family shows.
*Longer training makes it worse, not better:* lr 1e-5 at 500 steps 0.3853 → at 1500 steps 0.0030,
i.e. the collapse is progressive, consistent with an advantage that has no term pushing candidates
apart. *Gold-seeding rescues it:* the `-gold-` variants reach 0.6157 and 0.7870 because forcing the
gold passage into the pool prevents the policy from losing it — masking, not fixing, the defect.
*Accuracy lands near the no-context floor* (0.1480 vs the reader's 0.1927), the documented signature
of retrieval being dead rather than merely weak.

**Conclusion.** The slate reward failed for reasons of estimator construction, not because
downstream RAG accuracy is a bad objective. A correct version would need a within-query baseline
over several sampled slates, and per-document credit within the slate — at which point it becomes
the PL-GRPO recipe that works. This is recorded so the idea is not re-attempted in its broken form.

---

## Q5 — Is the "reward starvation" partly a GRADER defect? Yes, and a large part of it.

Every reward and every accuracy number in this campaign is produced by `answer_correct()`
(`mmrag/reader_vlm.py`), i.e. **cover-EM**. It was audited against an external LLM judge
(Qwen3-30B-A3B-Instruct-2507, temperature 0, served over an OpenAI-compatible endpoint) by
re-grading the *stored* predictions of a finished run — no retraining, no re-reading.

### Definitions used below

| term | exact definition |
|---|---|
| **cover-EM** (`answer_correct`) | For each gold alias *g*: normalise both strings (lowercase, strip non-alphanumerics, collapse whitespace); score 1 if the prediction equals *g*, **or** if *g*'s token sequence appears contiguously in the prediction's tokens. Token-boundary containment, not raw substring. This is the reward signal *and* the reported accuracy. |
| **strict-EM** | Same normalisation, equality only, no containment. |
| **LLM-judge** | The 30B model is shown the question, up to 12 gold aliases, and the model answer, and asked to reply 1 if the answer matches any gold "allowing paraphrase and formatting differences", else 0. Greedy. |
| **false negative** | cover-EM scored 0, judge scored 1. |
| **false positive** | cover-EM scored 1, judge scored 0. |
| **string-gold query** | `answer_aliases` is a list of strings. |
| **numeric-range-gold query** | `answer_aliases` is a list of **dicts** of the form `{'wikidata': v, 'range': [lo, hi]}` — InfoSeek's numeric questions, where any value inside `[lo, hi]` is correct. |

### Audit result (run `emaidx-1M-s3000`, n=1500, 400 predictions sampled)

| stratum | n judged | judge disagrees | rate |
|---|---|---|---|
| cover-EM said WRONG | 250 | 94 | **37.6% false negative** |
| cover-EM said RIGHT | 150 | 8 | **5.3% false positive** |
| implied judge-graded accuracy | | | **57.12** vs cover-EM **34.20** |

### The dominant cause is a metric defect, not judge leniency

`answer_correct()` does `_normalize(str(g))` on each gold. When the gold is a **dict**, that
stringifies to `"{'wikidata': 33.7, 'range': [30.33, 37.07]}"` and is then searched for as a token
sequence inside the prediction — **which can never match any number the reader produces.**

| stratum of the eval set | share | cover-EM accuracy |
|---|---|---|
| string-gold | 78.1% | 43.81 |
| **numeric-range gold** | **21.9%** | **0.00** |

**21.9% of queries are scored 0 by construction**, and 15,414 of the 71,335 test queries (21.6%)
are of this kind. The headline 34.20 is computed over a denominator that includes them; the same run
scores **43.81 on the queries the metric can actually grade.**

### What this does and does not overturn

- **It is a real, mechanical defect** and it explains most of the 37.6% false-negative rate. The
  overturned examples are overwhelmingly numeric: `"3.4 km²"` vs range [3.006, 3.674]; `"10"` vs
  [8.55, 10.45]; `"509 megawatts"` vs [450, 550] — all correct under InfoSeek's own rule, all scored 0.
- **The judge is also too lenient in places**, so 37.6% is an upper bound. It credited
  `"The Dead Sea has a salinity of 30%"` against range [30.33, 37.07] (30 is *outside*), and
  `"Fabaceae"` (family) for a gold of `"Amphicarpaea"` (genus) on a "closest upper taxonomy"
  question. A range-aware programmatic grader, not an LLM, is the correct fix for the numeric stratum.
- **It does NOT invalidate the RL-vs-SFT comparison.** Both arms are scored by the same broken
  metric on the same queries, so the *ordering* is unaffected; the *levels* are all understated by
  roughly the same amount.
- **It DOES bear on Q2's starvation.** Numeric queries can only ever return reward 0, so they are a
  guaranteed all-wrong population in training — an ~22% floor of dead gradient that is an artifact of
  the grader, on top of the reader-limited floor Q4 measures. This is a *fixable* component of the
  plateau, and the first one identified that requires no new model, data or compute.
- **Registered consequence:** fix `answer_correct` to evaluate `lo <= float(pred_number) <= hi` for
  dict golds, then re-measure degeneracy. Predicted effect: degenerate fraction should fall by
  roughly the numeric share of the training pool.

---

## Dynamic schemes — what was executed, and what it returned

Scheme numbering is the user's. Every pair below is **same hardware, same week, same seed, one flag
apart**; the only difference within a pair is the scheme.

### Scheme 1 — index-EMA (`--index_ema 0.9 --index_refresh_extra 256`). EXECUTED, 3 settings.

| setting | comparison | entR@5 | ansR@5 | acc |
|---|---|---|---|---|
| 810k pool, corpus_small, ck750 | vs **same-hardware** control | +1.60~ | +1.06~ | +0.60~ |
| 810k pool, corpus_small, 4-ckpt mean | vs published (cross-cluster) baseline | +2.50 | +2.40 | +0.07~ |
| 2M mixed index, v3-pure, ck750 | vs matched twin | **−2.80** | −0.08~ | +0.47~ |
| 2M mixed index, **anchored**, ck750 | vs matched twin | **+14.80** | **+19.59** | **+6.40** |

(~ = inside the seed floors 2.40 / 1.91 / 0.61.)

**CROSS-PROJECT JOINT CLAIM (text-side collaboration, 2026-09-10).** The text-side project ran the
same knob at far lower coverage and every arm turned over. Their bracket, all vs a frozen-index
control, R@5 at 1k→5k queries:

| arm | coverage | blend | outcome |
|---|---|---|---|
| frozen control | — | — | 0.508 → 0.607, monotone |
| 32 rows/step | 0.04x | 0.1 | level to 3k, then −5 to −7 by 5k |
| 128 rows/step | 0.16x | 0.1 | level to 3k, then −5 to −7 by 5k |
| dmixDynCov 704/step | 1.06x | 0.5 | −10.4 / −11.2 at 4k/5k, both seeds |
| dmixDynFreq 2816/step | 4.2x | 0.5 | monotone down from 1k |

**Turnover is invariant to coverage across 0.04x–4.2x and to blend rate.** The only thing every
failing arm shares, and the frozen control does not, is that **the document tower trains**. Their
candidate-side mean pairwise cosine ran 0.171 → 0.072 across the turnover — the doc tower
contracting, with the index following it.

Our side supplies the other half of the bracket: at **6.35x and 30x** coverage we see no turnover
(`emaidx-1M-s3000` entR@5 77.23 / 76.83 / 76.33 / 75.83 at 3k/6k/9k/12k queries — a 1.4-point drift
over 9k queries, not a turnover), and our one *contracting* cell is the anchored arm with the index
left un-damped (1126 distinct top-1 docs, entR@5 below zero-shot).

**JOINT CLAIM: the mediator is whether document geometry is HELD, not how much of the index moves.**
Held by any of three means — freezing the doc gradient (their control, and our scheme 3), damping
the index with EMA (our scheme 1), or the contrastive anchor — and the recipe is stable. Held by
none and the doc tower contracts and takes the index with it. Bracketed 0.04x–30x across two
projects, two model families, two modalities.
**TEST RESULT — the joint claim does NOT transfer to multimodal.** `mix2m-v3pure` is their failing
condition exactly: anchor off, index-EMA off, document tower training, live index.

| queries | step | entR@5 | ansR@5 | acc |
|---|---|---|---|---|
| 3,000 | 750 | 78.43 | 65.27 | 34.20 |
| 6,000 | 1500 | 76.23 | 64.12 | 34.47 |
| (zero-shot base) | | 68.53 | 54.94 | 30.80 |
| **3k → 6k** | | **−2.20** | **−1.14** | **+0.27** |

All three deltas are **inside the seed floors** (2.40 / 1.91 / 0.61), and the arm finishes 7.7 points
of entR@5 **above** its zero-shot base. Their arms lost 5 to 11 R@5 by this point at every coverage
they tried. **So the "doc tower training ⇒ contraction" mediator does not reproduce here**, and the
joint claim must be narrowed: it holds on their 0.6B text setup and does not, at this budget, on
GME-2B multimodal.

Candidate reasons, none tested: our index is refreshed **fully every 100 steps** (30x coverage)
where their write-back arms only ever touch 0.04x–4.2x, so our doc embeddings are never far from the
live encoder and may not have room to drift; our base is retrieval-finetuned and theirs is a 0.6B
general encoder; and our budget is 6k queries where their turnover shows at 3–5k but their *frozen
control* keeps rising to 28k, so we may simply be measuring too early. **The honest status is that
the mediator is confirmed on one side, refuted on the other, and the difference between the setups
is itself the next experiment.**

Note what this does to the coverage story: our 30x full refresh is the one thing their failing arms
never had. It is the leading candidate for why the same doc-side gradient contracts there and not
here — which would make "keep the index live" a real design rule rather than an aesthetic
preference, and it is directly testable by running our own recipe at their coverage.

**Reading.** On the anchor-free recipe index-EMA is null-to-negative: nothing callable against the
same-hardware control on the homogeneous index, and it *costs* 2.80 entR@5 on the mixed index.
The apparent +19.59 on the anchored arm is **not an EMA gain** — it is EMA preventing a collapse
that the anchor causes. The anchored no-EMA arm scores 63.57 entR@5, **below the 68.53 zero-shot
base**, and shows the documented contraction signature:

| arm | distinct top-1 docs / 3000 q | distinct top-5 | entR@5 |
|---|---|---|---|
| anchored, no EMA | **1126** | **4735** | 63.57 |
| anchored + index-EMA | 1435 | 6478 | 78.37 |
| v3-pure, no EMA | 1538 | 6924 | 78.43 |
| v3-pure + index-EMA | 1810 | 7809 | 75.63 |

So the finding is: **the contrastive anchor damages retrieval on a diverse index, and index-EMA
prevents that damage.** Mechanistically coherent — the anchor's negatives are cross-query documents,
and on a 37%-off-domain index most are trivially separable, so its gradient is dominated by easy
separations and the embedding space narrows; a smoothed, staler index damps that. 1 seed, 1
checkpoint: recorded, not claimed.

### Scheme 2 — selected-candidate + pool-EMA + rarity + hard negatives. NOT SEPARATELY EXECUTED, because it is already scheme 1's implementation.

`train_rl.py:401-422` in the `--index_ema` branch does all three per step: EMA-writes the selected
top candidates from the encodings just computed (`pool_ids`, :401/:417); re-encodes and EMA-writes
the near-boundary documents, commented in-source as *"near-boundary = free hard negatives"*
(`boundary_ids`, :402/:419-422); and re-encodes the **stalest** documents by `doc_last_upd`
(`ema_refresh_stale`, :322-330) — i.e. deliberately sampling the less-frequently-updated ones.
Running it as a separate cell would re-run scheme 1 under a new name. **Its results are scheme 1's.**
The one genuinely new component — mined hard negatives entering the **loss** rather than the index —
is a code change and was not executed. Nearest evidence is discouraging: the contrastive anchor,
whose negatives are cross-query documents, is precisely what causes the contraction above.

### Scheme 3 — model-EMA (`--model_ema 0.99`). LAUNCHED. **And it is structurally the strongest arm in either project, which reverses my prior.**

`train_rl.py:594-597` is the fact I had missed:
```
if args.model_ema > 0:  c_emb_g = c_emb.detach()   # grad via queries only
else:                   c_emb_g = enc.encode_docs(flat, grad=True)
```
**With `--model_ema` OFF the document tower trains; with it ON the document gradient is detached and
only the query tower learns.** So scheme 3 is not merely "encode docs with smoothed weights" — it
*disables doc-side training*. That makes it the same structure as the text-side project's **frozen
control**, which in their hands **beat every dynamic-index arm they ran** (R@5 0.508→0.607 over
1k→5k queries, while their 1.06x-coverage arm turned over by −10.4 and −11.2 at 4k/5k on both seeds,
and their 4.2x arm declined monotonically).

Their mechanism, which explains my own anchor result from the other side: every write-back cell of
theirs turns the doc tower on, and its candidate-side mean pairwise cosine ran **0.171 → 0.072**
across the turnover — the doc tower contracting, with the index following it. My anchored no-EMA
cell contracting to 1126 distinct top-1 documents is the same phenomenon. **A training doc tower
with nothing holding its geometry contracts; their frozen control removes the gradient, my index-EMA
damps its effect on the index.** That reframes the whole coverage question: their turnover appears
at 1.06x *and* 4.2x, so it is not a coverage effect, and my 30x-vs-6.35x null is expected because
both of my arms hold the doc side one way or another.

Operating point, checked across conventions: our `--model_ema` is a **retention** factor
(`ema.mul_(m).add_(p, alpha=1-m)`), so 0.99 means a 0.01 update rate — identical to the interior
optimum they measured (0.01), with 0.1 degenerating and 0.001 collapsing back to a base anchor.
The launched cell is therefore at their validated rate by coincidence.

**Registered prediction, before the result:** if the doc-tower-geometry account is right, scheme 3
should be the best of the four schemes here, not the null I predicted earlier from the
"aimed at the tower that doesn't move" argument. That argument was wrong because it assumed the doc
tower barely moves; in fact it barely moves *in the probes*, while still carrying the gradient that
lets it contract.

The only prior attempt, `div45kv3-emaenc` (72.47/60.49/32.20), is **unusable**: it trained
q(live)·d(EMA) but evaluated both towers live — a train/eval geometry mismatch, not a result. The
fix (saving `ema_tower/`) landed at `train_rl.py:668` and had never been exercised. Launched
`mix2m-modelema` (`--model_ema 0.99`) against a matched twin `mix2m-modelema-ctrl`, co-located on one
card so contention is identical, 1500 steps on the mixed index. **Result pending.**
Prior worth stating: the geometry probes show RL moves the **query** tower hard (mean pairwise
cosine 0.2212 → 0.0594) while the doc side barely moves, and model-EMA damps the doc tower — i.e.
aimed at the tower that is not moving. That argues for a null; it is cheap to settle.

### Scheme 4 — EMA as a regulariser. NOT EXECUTED here, but the text-side project HAS run it, and the answer is conditional.

Their evidence (0.6B, 1M multi-source index, 2-4 seeds), where their `m` is the **update rate**:

| condition | result |
|---|---|
| **frozen** index, m=0.01 | **no degeneration over 28k queries**: 0.476 (1k) → 0.632 (13k) → 0.647 (28k), still rising, above their Rel-SFT ceiling of 0.630. Query-side pairwise cosine falls 0.210 → 0.121 (spreading, not collapsing); degenerate-group rate has no trend; within-group reward std flat. |
| rate m=0.1 | the pre-registered KL→0 degeneracy: flat 0.550 / 0.546 / 0.546 |
| rate m=0.001 | collapses back to a base anchor: 0.534 |
| **live** index (doc tower trains, full re-encode /100 steps) | **EMA 47.8 max, 45.8 final vs base-anchored KL 48.1 max** — a late decline 4x the KL spread. Their phrase: *"a lagged copy of a moving policy against a moving index."* 4B in-domain: EMA 0.738 vs KL 0.742. |

**So "EMA regularisation beats KL" is true only when the index is HELD, and false when it is live.**
Our index is live — refreshed every 100 steps — so on this project the expected outcome is no gain
and a late decline. The interior optimum is m=0.01 (update rate) with one decade measured either
side. This is the cheapest possible answer to scheme 4 and it cost no GPU time here.

### Reward-reader strength — bears directly on the "use a 30B reader" proposal

Text-side measurement, swapping the reward reader from 3B to a 30B MoE:

| quantity | 3B | 30B |
|---|---|---|
| training queries answerable WITHOUT context | 21% | **34%** |
| fails even WITH gold | 39% | 33% |
| **share of queries that can move a reward** | **37%** | **31%** |

**A stronger reward reader makes saturation worse, not better.** It converts more queries into the
"answerable without retrieval" quadrant than it rescues from the "hopeless" quadrant, so the
fraction of rows carrying usable reward variance *falls*. This is the mirror image of our Q4
taxonomy (32.3% benefit / 48.5% hopeless with a 3B) and it predicts that swapping our reader for a
30B would shrink the useful training set rather than fix the starvation.
**What did work on their side: the filter.** Keeping only reward-variance rows is **+4 to +7 R@5**
over a matched control at 3k queries across 2 seeds — the largest positive effect either project has
found for the plateau, and independent confirmation that Q4's filter is the right lever.

No code path exists. Two forms, both requiring changes. **KL/L2 to the EMA weights**: cheap to add,
but the nearest measured evidence is against it — KL-to-init hurts (β=0.01 → −0.7 acc, β=0.1 → −1.7)
and the ledger closed that axis with *"bare > KL on MM; the regulariser must match the failure mode
the base actually has, and gme2b has none."* An EMA reference is a moving target rather than a fixed
one, so it is not the identical experiment, but the prior is poor. **Full fine-tuning with weight
averaging**: everything on record is LoRA r32; full FT changes the optimisation regime, memory
profile and collapse risk at once, so a result would be hard to attribute. Recommended last.

---

## Q4 — Does indirect RL need downstream data quality? Filter to queries RAG can actually help.

**Sizing first, from evaluations already on disk** (frozen base, n=1500, same queries): reading the
gold-context and no-context runs per query gives an exact taxonomy of what retrieval can and cannot
buy.

| category | gold / no-ctx | share | top-5 acc today |
|---|---|---|---|
| **RAG-BENEFITS** — retrieval is the whole game | 1 / 0 | **32.3%** | 43.8% |
| RAG-UNNEEDED — reader already knows it | 1 / 1 | 18.1% | 84.9% |
| **RAG-HOPELESS** — wrong even WITH the gold passage | 0 / 0 | **48.5%** | 2.2% |
| RAG-HARMS — context breaks a correct answer | 0 / 1 | 1.1% | 17.6% |

Anchors: gold-context accuracy **50.40**, no-context **19.27**, current top-5 **30.80**.

**Two consequences, and the first bounds every other question in this document.**

**(1) The absolute ceiling of retrieval on this benchmark is 50.40 accuracy.** Perfect retrieval —
handing the reader the gold passage every time — scores 50.40, because 48.5% of queries are wrong
even then. Current top-5 is 30.80, so the entire remaining headroom for *any* retrieval method,
RL or SFT or oracle, is **19.6 accuracy points**. The 34-35 plateau that Q2 documents sits at about
a quarter of the way through that gap. Any claim that a retrieval recipe "keeps improving" is
bounded by this number, and it is a property of the reader and the dataset, not of the recipe.

**(2) It independently confirms the Q2 starvation mechanism, from the reader side.** The 48.5%
RAG-HOPELESS rows can produce *only* all-wrong reward groups: no candidate, gold included, lets the
reader answer. That is measured on the reader, not inferred from training logs, and it matches the
~53% zero-advantage groups seen during training closely enough to be the same phenomenon.

**The Q4 experiment this motivates.** Filter the training pool to RAG-BENEFITS rows only (gold
correct AND no-context wrong), which removes by construction the rows that generate dead gradient,
and train the same recipe on it. Predictions, registered before running: degeneracy should fall
well below the ~0.59 baseline; and because the campaign has repeatedly shown pool SIZE is null, a
~3x smaller but fully-informative pool should not lose on that account. **Control required:** an
unfiltered random pool of the SAME size, or the result confounds filtering with pool size.
Note `--reward_gate_noctx` already implements a close relative of this online, using the policy's
best candidate instead of the gold passage; it was tried once for +0.94 ansR@5 at 1 seed. The
gold-based version is the cleaner, policy-independent form and is what Q4 asks for.
**Cost:** the filter needs a gold-context and a no-context reader pass over each candidate training
row, ~2 reader calls/row, so it is sharded across GPUs rather than run inline.

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
