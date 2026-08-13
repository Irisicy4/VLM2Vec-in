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

## Cross-modal split on anchor removal (2026-08-13) — scope condition for v3

Text-side plgcc0 cells landed: their no-anchor no-gold arms COLLAPSE (Trivia R@5 43.1 vs base
58.1; WQ R@5 6.3 vs base 55.5) while our v3-pure improved both in-domain (+0.9 over anchored)
and OOD (+3.3). So "drop the anchor" is NOT modality-portable as stated. The margin rule
correctly predicted the anchor's RANKING pressure was inert on text — what it missed is the
anchor's second job: absolute cross-query collapse prevention, which becomes load-bearing when
the reward is sparse (their live-pool rates 21-75% pre-fix; ours healthy at mean ~0.26).

MM datapoints against a pure-density story: v1-cc0 (all-actions, no anchor) never collapsed
either (recorded 0.3373), so on MM BOTH estimators survive anchor removal.

PRE-REGISTERED DISCRIMINATOR (in flight, job a1eb5830): clip-v3pure = no anchor + weak
retriever (CLIP-L, zs R@5 0.054) + mostly-junk pools = the text collapse regime reproduced on
MM. If reward density/liveness is the deciding variable, clip-v3pure should collapse like
their WQ cell; if the split is modality- or action-structure-driven (their pool construction
vs our sampled lists), it should merely underperform its anchored twin (clip-nogold-base,
R@5 0.291). Logged here before the cell reports.

Consequence for the paper: the v3 no-anchor claim carries a scope sentence — demonstrated in
a reward-dense regime; the text twin collapses in reward-sparse ones; the anchor's
collapse-prevention role is real and regime-dependent.

## Discriminator verdict (2026-08-12, hours after pre-registration): DENSITY BRANCH CONFIRMED

clip-v3pure landed: entity R@5 = 0.0043 — total retrieval collapse, 12x BELOW its own
zero-shot base (0.054), let alone its anchored twin (clip-nogold-base, 0.291). Accuracy 0.190
= the reader answering from the image alone. This is the text side's WQ collapse (6.3 vs base
55.5) reproduced within our modality, under the identical sampled-list PL estimator (their
plgcc0 confirmed G=8 k=5 Gumbel-top-k exact-factorized — the estimator x anchor confound is
closed on both sides).

Both pre-registrations fired the same way (ours: density branch; theirs: collapse expected).
The joint rule, now measured on a 2x2 across modalities:

                      reward-dense pools        reward-sparse pools
  anchor removed      HELPS (+0.9 ID, +3.3 OOD  COLLAPSES (MM clip: R@5 0.004;
                      — mm v3-pure; both MM      text WQ: 6.3; text Trivia: partial)
                      estimators survive)
  anchor kept         costs accuracy for recall  prevents collapse (its load-bearing job)

The InfoNCE anchor's two jobs are now separable and regime-tagged: ranking pressure (inert or
harmful depending on margins — the margin-at-temperature rule) and collapse prevention
(irrelevant when reward is dense, essential when sparse). "Drop the anchor" is a
reward-density-conditional recommendation, on both modalities, full stop.

Corollary for the ladder: qwen2b-zeroshot R@5 = 0.0003 (raw VLM embeddings = random retrieval).
By the density rule, qwen2b-v3pure should also collapse and qwen2b-nogold (anchored) is the
interesting cell — can the anchor keep a from-scratch policy alive long enough for the reward
to shape it? Registered expectation: anchored survives, unanchored collapses, locating the
boot-strap floor between clipb32 and gme2b.

## The anchor as a restoring force (text-side insight, 2026-08-12, PAPER_TABLES 8f @ 0cdc69c2)

Resolution of why the margin-rule pre-registration missed on collapse: 8d margins are measured
at the BASE model — a POINT statement. Their anchored-ng twins are anchor(self), inert at init
by that measurement, yet removal still collapsed training. The anchor is a RESTORING FORCE:
zero at equilibrium, re-engaging exactly as reward-driven drift shrinks the margins that made
it inert. Static margins predict ranking pressure; they cannot price collapse prevention,
because inertness is about a point and collapse is about a trajectory.

Cross-project vocabulary (both papers): anchor(self) = positive is the policy's own top-1
(all 39 of our anchored runs, audited); anchor(ext) = external/gold positive (their gold arms);
anchor(stoch) = sampled positive (destructive, both sides' history). "Anchored" is NOT the
same column across projects without these tags.

CORRECTION (writer-caught, 8f cross-checked): my earlier claim that collapse prevention is
"demonstrated by both anchor types" was WRONG. Both text-side collapse cells are anchor(self)
— the same type as all 39 of ours — so the cross-modal collapse result is a SAME-TYPE
replication. Their one anchor(ext) removal (mus_gold cc0: Hot cEM 39.6->30.8) lost the gold
advantage WITHOUT retrieval collapse — a different failure mode.

Refinement on the refinement: "collapse prevention belongs to the self type" also overreaches,
because type and regime are CONFOUNDED in the ext cell — external positives exist only where
gold labels exist, and gold seeding keeps pools reward-dense, which is precisely the regime
where NO anchor is needed for stability. The ext cell therefore never faced collapse risk and
cannot test whether ext anchors prevent collapse. Accurate statement: collapse prevention is
demonstrated ONLY for anchor(self), on two modalities; anchor(ext)'s collapse-prevention
capacity is untested and structurally hard to test.

PRE-REGISTERED READINGS for the ladder cells in flight, per the restoring-force account:
- siglip2-nogold (anchor(self), zs R@5 0.082): rescue should repeat (CLIP-L pattern).
- qwen2b-nogold (anchor(self), zs R@5 0.0003 — the top-1 positive is a RANDOM doc):
  * if it still bootstraps -> the anchor needs only a FIXED POINT, not a meaningful one;
  * if it fails where CLIP-L (0.054) succeeded -> there is a liveness floor below which
    anchor(self) has nothing to restore to.
  Either outcome is one clean sentence; both are written here before the cells report.

## Two collapse phenotypes (2026-08-12, joint w/ text side, their 8f @ 1bb562ef)

The distinct-top-k metric run on text-side collapse cells fires the second live outcome:
their TOTAL collapse (wq_ng cc0, R@5 6.3 vs base 55.5) is 242/276 distinct sets (top set
2.2%) — varied-but-useless. Ours is 67-338/3000 with top sets up to 45% — contracted.

  MM collapse   = DEGENERACY: the embedding contracts; policy maps most queries to a few
                  fixed sets. Detectable statically (distinct-count), no labels needed.
  text collapse = MISALIGNMENT DRIFT: query/doc geometry decouples while staying spread out.
                  Invisible to the distinct-count (their anchored-vs-pure pairing: 276 vs 242,
                  1.14x — nothing); visible only against ground truth (recall).

Same anchor term, two restoring jobs by regime AND modality: against contraction here,
against drift there. Joint-table rule: a signature column, never the bare word "collapse".
Symmetric limitation now explicit: our static count would have missed their failure exactly
as their static margin measurement missed ours.

## Boundary probe verdict + phenotype revision (2026-08-12 evening)

vlm2vec2b-v3pure (zs R@5 0.157, the registered boundary probe): **FAILED — acc 0.1720,
R@5 0.007 (22x below its zero-shot). So v3-pure's working range is bracketed: fails at 0.157,
works at 0.685. Pure RL requires a GME-class base.**

**And it failed by the TEXT phenotype: distinct-top5 = 3000/3000 (max 1) — maximal diversity,
zero utility. Misalignment DRIFT, not contraction.** The "degeneracy=MM, drift=text" split
recorded earlier today is falsified as a modality rule: the MM ladder now contains BOTH
phenotypes (contraction at clip/siglip2/raw-qwen2b; drift at vlm2vec2b). Phenotype correlates
with something else — candidate: contrastive-pretraining structure of the base (VLM2Vec is
MMEB-contrastive-trained with strong uniformity, resisting contraction but free to drift;
CLIP/SigLIP dual towers and raw VLMs contract). One-seed evidence, hypothesis only.
Consequence: the diagnostic-PAIR recommendation (distinct-count + recall-vs-base) upgrades
from "covers both modalities" to "required within a single modality."

## Checklist-caught ledger gaps (2026-08-12 late — items sent in messages but never written here)

- HEADLINE, final: v3-pure [InfoSeek acc, top-5, n=1500] = 0.3471 ± 0.0019 (3 seeds:
  0.3447/0.3473/0.3493), SIG vs SFT 0.3100 and base 0.3080 per seed (McNemar p <= 4.8e-05).
  ck250 lower on all seeds (0.3373 mean) -> endpoint is the checkpoint max; headline unrevised.
- PHENOTYPE HYPOTHESIS, corrected name: **domain-uniformity** (not "contrastive-pretrained" —
  CLIP/SigLIP are contrastive yet contract). Bases whose pretraining spread the retrieval-domain
  (passage-side) embedding space fail by drift (VLM2Vec, Qwen3-Emb); bases without passage-side
  uniformity contract (CLIP/SigLIP cross-modal alignment, raw VLMs). 1 seed per cell.
- CAPACITY, corrected: the recorded "gap grows with capacity" was a baseline-pairing artifact
  (gold-RL vs mid-rung SFT). Like-for-like: 2B +2.08 / 7B +2.17 — FLAT. See main-repo LEDGER
  erratum. Fresh v3-7B: 0.3460 (1 seed) — no accuracy gain over 2B, +4.9 entR@5 only.
- v3 boundary bracket: pure RL fails at zs entR@5 0.157 (vlm2vec2b, by DRIFT) and works at
  0.685 (gme2b); working range (0.157, 0.685] pending finer rungs.

## Base-model wave 2 (registered 2026-08-13, pre-launch)
New cells: bgevl_b-v3pure (5310cbb94793e62c), bgevl_l-v3pure (b8803548161beb61), qwen25_3b-v3pure
(7ece3f2262c030cf); zs chains bgevl_b/bgevl_l/qwen25_3b local. BGE-VL = MegaPairs
retrieval-FINETUNED CLIP-arch — same architecture as clipb16/clip, different pretraining.
PREDICTION (registered before any number lands): if the working/failing boundary is about
retrieval pretraining (domain-uniformity account), bgevl_l-v3pure avoids the contraction
phenotype (distinct-top5 >> 338, R@5 within a few points of its own zs) and shows positive
delta-acc; if the boundary is about VLM-class architecture/capacity, bgevl_l collapses like
clip (R@5 ~0). Secondary: bgevl_b (0.15B) tests whether size alone blocks training at
matched pretraining. qwen25_3b slots the raw-VLM size ladder between qwen2b and qwen7b;
expectation from domain-uniformity: drift phenotype like qwen2b, not contraction.

### Wave-2 amendment (2026-08-13, zs landed, TRAINING CELLS STILL BLIND)
zs placements: bgevl_b entR@5 0.1473 / acc 0.1593; bgevl_l entR@5 0.1573 / acc 0.1587 —
both AT the known failure boundary (<=0.157), and both BELOW the no-context floor (0.1927):
their retrieved context actively hurts the reader. MegaPairs CIR finetuning does not
transfer to InfoSeek wiki-KB retrieval; BGE-VL sits in the raw-CLIP zs band here.
AMENDED READING (registered before any training number lands): the original clean
discrimination (same arch, retrieval pretraining, higher start) is unavailable — BGE-VL is
NOT higher-start. The cell now tests: retrieval-style objective at matched-bad zs level.
If bgevl_l-v3pure still trains where clip collapsed -> objective/geometry mechanism result
(stronger than the original claim). If it collapses -> consistent with a zs-competence
floor; pretraining-vs-architecture question stays OPEN (not refuted).

### LR-sensitivity control wave (2026-08-13, user-directed, registered pre-landing)
CORRECTION (writer, from args.json): rates were NOT uniform and 1e-4 is NOT the GME rate.
Actual rates: gme2b/qwen2b/vlm2vec2b 2e-5; clip/siglip2 (and wave-2 bgevl_*/qwen25_3b,
which copied the clip config) 1e-4 (best-for-CLIP in the recorded weak-rung sweep);
qwen7b/gme7b 1e-5. Each failure is a failure at that rung's own inherited rate, chosen on
a different base. Control cells submitted:
clip-v3pure-lr{1e5:MLX b530a228, 3e5:local}, bgevl_l-v3pure-lr{1e5:MLX b319c018, 3e5:local},
siglip2-v3pure-lr1e5 (local), qwen25_3b-v3pure-lr3e5 (local); coverage-gap cells added
after writer's design note: qwen25_3b-v3pure-lr2e5 (its family's rate), qwen2b-v3pure-lr5e6
and vlm2vec2b-v3pure-lr5e6 (low-end controls for the 2e-5 collapses, previously uncovered).
PREDICTION: if the CLIP-class contraction phenotype is an LR overshoot artifact, lower LR
rescues R@5 toward zs level and the boundary claim must be rewritten as "fails at
GME-tuned lr" (per-family tuning required, boundary table gains an lr column). If
contraction persists across the 10x ladder, the boundary claim survives LR as a confound.
Registered expectation from the margin account: contraction persists (the failure is
signal-starvation in the top-M support, not step size) — but genuinely uncertain.

### Wave-2 blind cells landed (2026-08-13 03:13) — SCORED
bgevl_b-v3pure acc 0.1780, entR@5 0.1473->0.0007, distinct-top5 133/3000.
bgevl_l-v3pure acc 0.1827, entR@5 0.1573->0.0070, distinct-top5 197/3000.
Both CONTRACT at lr 1e-4 — same phenotype as clip/siglip2. Amended registered reading
scores as: retrieval finetuning does NOT rescue CLIP-arch at matched-bad zs;
pretraining-vs-architecture stays OPEN (zs-competence-floor account consistent).
Weight shifts to the LR controls (bgevl_l 1e-5/3e-5, running).
ANTI-DIAGNOSTIC INSTANCE #2 (cleanest yet): acc ROSE +2 to +2.4 pts while retrieval
died 20-200x — with R@5 ~0 the context stream approaches uniform junk and the reader
drifts toward its no-ctx floor (0.1927) from below. Accuracy alone would have called
these cells "improved." The diagnostic PAIR is mandatory; log this instance in the paper's
diagnostics paragraph.
qwen25_3b-zeroshot: acc 0.1607, entR@5 0.0373 (distinct 2992/3000) — raw Qwen2.5-VL-3B
zs retrieval is near-random, far below the boundary; its v3 cells now test training from
a near-zero start.
sft-curve ck750 = 0.3300 — NEW SFT MAX, curve non-monotone (.2953/.3207/.3227/.3107/.3300);
provisional fair 2B gap 0.3471-0.3300 = +0.0171; McNemar at fair pair when curve completes.

### Text-side composed cell scored (2026-08-13, their wave) — my registration: HIT
plgTa_triv (PL + anchor(self) cc=1.0 + 0.727 reader, 0.6B) ck175: Trivia R@5 55.6 vs base
58.1 (within 2.5; registered ~2), cEM 68.0 vs 69.8 ~parity. 30%-clears-base branch did NOT
fire (score that miss-of-branch). Decomposition as registered: anchor removed the early
DROP (+5.5-7.4 R@5 vs both cc0 arms), PL removed the decay. Headline: NQ-transfer cEM 47.6
= +3.0 over base = matches gold-seeded RL exactly; swept SFT 49.2 still +1.6 ahead (their
SFT verdict survives). Estimator x capacity, text column: PL > enumeration at 0.6B AND 4B
(plgT4b_tj ck175 61.4/72.0 vs enum 57.5/70.8). MM column: our gme 2B/7B pair — with the
caveat that MM 7B RL-SFT gap compresses to n.s.; text 4B still shows PL>enum clearly
(different comparison: estimator-vs-estimator, not RL-vs-SFT).

### LR-confound propagation from text side (2026-08-13, their config audit)
Their anchored plg twins/klfix/ppofix ran 1e-5; T-era family (plgT*, plgTa, cc0 arms) ran
2e-5. CONSEQUENCE FOR THIS LEDGER: the "text no-anchor no-gold arms COLLAPSE" attribution
(entry above citing Trivia R@5 43.1) compared unanchored-2e-5 vs anchored-1e-5 — the
anchor-prevents-collapse reading carries an LR confound there. PARTIAL RESCUE already in
hand: the composed-cell pair plgTa vs plgT_cc0 is same-rate (both 2e-5) and shows the
anchor removing the early drop (+5.5-7.4 R@5) at matched LR — anchor attribution survives
at matched rate, magnitude of the old "collapse" rows awaits their wave-3 plgTlr1e5 cell.
Cross-modality claims should cite the T-era same-rate pair, not the old twins.

### clipb32-v3pure landed (2026-08-13 03:55): THIRD PHENOTYPE — INERT
acc 0.1513->0.1473, entR@5 0.0260->0.0257, distinct 2825/3000 (healthy). Training was a
no-op. New failure vocabulary: INERT (nothing moves) vs CONTRACTION vs DRIFT.
HYPOTHESIS (registered now, qwen25_3b-v3pure still blind): phenotype tracks starting
signal level — zs entR@5 ~0.03 -> inert (reward ~uniform, no gradient direction);
~0.15 -> contraction (bgevl_b 0.147, bgevl_l 0.157, clip, siglip2 — enough signal to
chase, not enough to rank within); ~0.16 raw-VLM -> drift (vlm2vec2b, qwen2b);
0.69 -> healthy (gme). AMENDED PREDICTION for qwen25_3b-v3pure (zs entR@5 0.0373,
supersedes my earlier drift registration, recorded pre-landing): INERT, not drift.
If it drifts instead, the ladder hypothesis loses the level-not-family part (raw-VLMs
may drift at any level); if inert, phenotype-by-signal-level gains its second point.

### Taxonomy correction (writer catch #8, accepted) + boundary bracket
My "inert is where distinct-count IS informative alone" INVERTED — distinct for inert
(2825) is indistinguishable from healthy (2899). Corrected taxonomy (theirs, adopted):
recall-vs-base separates healthy/inert/damaged but cannot split contraction from drift;
distinct-count detects ONLY contraction (healthy/drift/inert all 2800-3000). The pair is
minimal and sufficient — neither dominates, neither redundant.
BRACKET (sharper than my three-band sketch): clip-L already CONTRACTS at zs entR@5 0.054,
clipb32 INERT at 0.026 -> the inert/contraction boundary lies in (0.026, 0.054). My
ladder bands ("~0.03 inert, ~0.15 contraction") were sloppy — the contraction band
extends down to at least 0.054. qwen25_3b (zs 0.0373) sits MID-BRACKET: whichever
phenotype it lands, the bracket halves. Queued clipb16-zeroshot to place a further
intermediate point (expected zs between clipb32 and clip-L); its v3 cell follows if its
zs lands inside the surviving half-bracket.

### Bracket wave-audit (writer) + fresh clip-zeroshot launched
Writer caught the (0.026, 0.054) bracket mixing waves: clip-L's 0.054 upper bound is
RECORDED (no fresh clip-zeroshot existed on this cluster). Fully-fresh primary bracket:
(0.026, 0.082) — clipb32 inert / siglip2 contracting, both endpoints fresh-zs + fresh-v3.
Fresh clip-zeroshot chain launched (GPU 5) — makes the tight bracket within-wave and is
PREREQUISITE for interpreting clipb16-v3pure as a tight-bracket bisection. qwen25_3b
(0.0373) bisects either bracket. Print-once discipline: nothing writable until qwen25_3b
lands.

### LR ladder first landings + gme7b-s1 (2026-08-13 04:21) — MY LR PREDICTION: MISS
**bgevl_l-v3pure-lr1e5: TRAINS.** entR@5 0.1573 -> 0.2123 (+5.5, 1.35x), distinct
2966/3000, acc 0.1587 -> 0.1693. Contraction at 1e-4 was an LR ARTIFACT for BGE-VL-L.
My registered prediction ("contraction persists; failure is signal-starvation not step
size") SCORES AS MISS on the phenotype claim. User's tuning instinct was right.
**clip-v3pure-lr1e5: rescued phenotype, NO learning.** entR@5 0.0410 (vs recorded zs
0.054), distinct 2882 — no contraction, but no improvement either (slight sag).
=> THE WAVE-2 DISCRIMINATION RETURNS WITH AN ANSWER: at matched tuned LR (1e-5), the
retrieval-pretrained CLIP-arch (BGE-VL-L) LEARNS (+5.5 R@5) while raw CLIP-L does not
(-1.3). Retrieval pretraining is an enabling ingredient; step size was a masking artifact
on top. Margin account survives only its signal-starvation half (clip has nothing to
learn from); its phenotype half was wrong.
HONESTY CLAUSE: bgevl_l-lr1e5 acc 0.1693 remains BELOW the no-ctx floor (0.1927) — its
retrieval improved but is still net-harmful to the reader. "Trains" means retrieval
learns; it does NOT yet mean useful RAG. 500 steps; longer runs unregistered.
**gme7b-v3pure-s1 = 0.3573** (entR@5 0.8030, distinct 2895): vs gme7b-sft +0.0253
McNemar p=0.0014 SIG (87/49); vs 2B base +0.0493 p=5.9e-08. 7B RL 2-seed: 0.3460/0.3573
(mean 0.3517, spread 0.0113). The s0-only "n.s. convergence at 7B" verdict does NOT
survive s1: RL>SFT at 7B is SIG in the stronger seed, mean gap +2.0 vs 2B's +3.7.
Capacity story v4: margin persists at 7B, somewhat smaller than 2B; "capacity helps SFT
not RL" weakens (RL-7B mean 0.3517 vs RL-2B 0.3471, +0.5 within seed noise).
sft-curve ck900 = 0.3133 (max still ck750 0.3300; ck1050/1200 pending).

### 05:18 landings: bracket promotes; LR dose-response; b16 consumption point
clip-zeroshot FRESH entR@5 0.0537 (recorded 0.054, delta 0.0003 — cluster repro fidelity
holds). Tight bracket (0.026, 0.054) PROMOTES to fully-fresh. clipb16-zeroshot 0.0310 —
inside the bracket, below qwen25_3b 0.0373; bracket coverage now .026/.031/.0373/.054.
clipb16-v3pure stays queued (in-bracket condition met).
LR DOSE-RESPONSE (phenotype=f(zs,lr) now has gradients, not just endpoints):
clip-L: 1e-5 no contraction/no learning (2882, R@5 .041) | 3e-5 PARTIAL contraction
(1795, .0170) | 1e-4 full (214, .0043).
bgevl_l: 1e-5 LEARNS (+5.5, 2966) | 3e-5 mild damage (2364, R@5 .1477 < zs) | 1e-4 full
contraction (197). Learning window closes between 1e-5 and 3e-5. QUEUED bgevl_l-v3pure-lr3e6
to probe below 1e-5 — 1e-5 is a boundary cell, not established as the peak.
sft-curve ck1050 = 0.3133 (plateau; max still ck750 0.3300; ck1200 last).
v3-b16-s750 = 0.3487 acc, entR@5 0.7590, distinct 2891 — batch-16, 12k draws: consumption
curve gains a >=headline point (0.3487 vs 0.3471), consistent with consumption-rising.

### 05:59 landings — fairness audit COMPLETE; qwen25_3b prediction HIT; bracket halves
**2B fairness verdict (final): SFT curve complete** (.2953/.3207/.3227/.3107/.3300/.3133/
.3133/.3153, ck150-1200; max ck750 = 0.3300; released 600-step ckpt confirmed post-peak).
v3-pure fresh seeds vs SFT-max McNemar: s0 +0.0147 p=0.084 n.s.; s1 +0.0173 p=0.042 SIG;
s2 +0.0193 p=0.02 SIG. AUDITED HEADLINE: RL 0.3471 (3-seed mean) vs SFT-best-checkpoint
0.3300 = +1.7, SIG in 2/3 seeds. (Old +3.7-vs-endpoint number retired from headline use.)
Residual asymmetry, disclosed: SFT max is from ONE retrain (8 ckpts); RL endpoints
verified >= their ck250 on s0 only.
**qwen25_3b-v3pure: INERT, exactly** — entR@5 0.0373 -> 0.0373, distinct 2992, acc
identical. Amended prediction (drift->inert, registered pre-landing) SCORES HIT. Bracket
halves: at lr 1e-4, inert/contraction boundary now (0.0373, 0.054), fully fresh. Note the
bracket is LR-CONDITIONED (at 1e-5 nothing contracts anywhere). clipb16-v3pure (zs .031,
queued) now tests inert-region consistency rather than bisecting.
**siglip2-v3pure-lr1e5: no contraction (2917), no learning (R@5 .0823 -> .0660)** — same
class as clip-L at 1e-5. At 1e-5 the only base that LEARNS remains bgevl_l. Wording
nuance for the discrimination claim: BGE-VL is retrieval-FINETUNED (MegaPairs CIR);
clip/siglip are image-text contrastive — say "retrieval-finetuned", not "-pretrained".
**v3-rows200k-s3000 = 0.3407** (R@5 .764, distinct 2885): big-pool consumption curve now
2k:0.3400(3s) / 6k:0.3480(1s) / 12k:0.3407(1s) — plateau ~0.34-0.35, no clear rise
beyond 6k; 1-seed points, do not claim a sag.
ERRATUM (writer catch): I doubled these draw labels in the 05:59 entry and in messages
(4k/12k/24k); batch is 4, so draws = 4 x steps = 2k/6k/12k. Accuracies were mapped
correctly. The b16-s750 line above (16 x 750 = 12k) was already correct — note it now
COINCIDES in draws with rows200k-s3000 (12k): 0.3487 (b16, 3k pool) vs 0.3407 (b4,
200k pool) — a matched-consumption batch/pool comparison, confounded pairwise, listed
as observation only.

### Zero-gradient note (writer suggestion, adopted)
qwen25_3b-v3pure's retrieval metrics unchanged to FOUR DECIMALS is evidence the policy
gradient was exactly zero (uniform reward -> zero advantage -> no update), not merely
small. Verifiable from train logs (loss/adv should be ~0 throughout); stronger inertness
evidence than "flat metrics". Same check applies to clipb32 (near-flat, small nonzero).

### RETRACTED (writer control): the entry below claimed verification from an identity
loss/policy = -mean(A) with A group-z-scored is ZERO BY CONSTRUCTION for every plgrpo
run — the healthy v3-pure logs |loss/policy|max 1.49e-08 too. A zero scalar loss also
does not imply zero gradient (grad = -mean(A*gradlogpi) at rho=1, generally nonzero).
The within-group-degeneracy account is DEMOTED to hypothesis (still the best one; the
across-query reward variation stands). Verification requires per-group reward std,
which was not logged. FIX: train_rl.py now logs reward/group_std_mean and
reward/group_degenerate_frac; 50-step probe cells queued (qwen25_3b + gme2b control).
Prediction, registered: qwen25_3b degenerate_frac ~1.0, gme2b well below 1.

### Zero-gradient claim (RETRACTED ABOVE — kept for the record) — mechanism sharpened
loss/policy and ppo/advantage_mean = 0.000000 exactly, all 500 steps. BUT reward/raw_mean
= 0.135 (max 0.75) — rewards are NOT zero/uniform across queries. Mechanism: inertness =
WITHIN-GROUP reward degeneracy. With near-random retrieval, the G=4 sampled lists from a
junk top-24 are interchangeable to the reader — identical reward within every group —
so the group-z advantage is identically zero even though the reader sometimes answers
right (parametric knowledge). Inert cells are where the reward cannot DISCRIMINATE
between lists, not where it is absent. (This also predicts: any base whose top-24
contains at least occasional signal escapes inertness — consistent with contraction
starting by 0.054.)

### Conventions (adopted from writer exchange, 2026-08-13 morning)
Diagnostic runs use name suffixes -stdprobe/-probe/-smoke/-debug; the writer's collectors
exclude these BY NAME (EXCLUDE_RUNS), because config rules deliberately free max_steps on
the consumption axis and a 50-step diagnostic would otherwise be admitted as a legitimate
200-draw curve point. General shape, theirs, worth keeping: "the more permissive an axis
is by design, the more it admits things that are not results" — any future axis that
frees a config key inherits the same exposure. New diagnostic suffixes must be announced
(protocol runs both directions). Methods-narrative decision: process near-misses stay
OUT of the paper; rigour is shown by scored misses, withdrawn registrations, and the
audited headline, not by narrating the catch.

### INVALIDATION (2026-08-13 08:47): qwen25_3b-v3pure "inert" was an EVAL ARTIFACT
Chain that exposed it: stdprobe prediction MISSED (qwen25_3b degenerate_frac 0.71, NOT
~1.0; healthy gme2b control itself 0.55) -> partial degeneracy cannot give four-decimal
flatness -> checked adapter: lora_B abs-sum 44k (HUGE — bgevl_l's +5.5 R@5 came from 812)
-> loaded ckpt vs base embeddings: diff 0.0 EXACTLY -> src/model/model.py reload path
attached qwen2_5_vl adapters to base_model.model, keys matched NOTHING, merge_and_unload
was identity. FIXED (QWEN2_5_VL added to full-model-attach set; verified diff 0.328) and
pushed. SCOPE: only qwen2_5_vl CHECKPOINT RELOADS — training was real (fresh-LoRA attach
is a different path); qwen25_3b-zeroshot valid (no ckpt); qwen2_vl/gme/clip families
unaffected (their landed metrics visibly moved); reader unaffected.
CONSEQUENCES: (1) qwen25_3b-v3pure verdict UNKNOWN pending re-eval (running, GPU 0) —
adapter moved a lot, could be anything; (2) "amended prediction HIT" un-scored — pending;
(3) bracket REVERTS to (0.026, 0.054) with interior point unresolved; phenotype-ladder
%-comment must drop the qwen25_3b point; (4) lr2e5/lr3e5 cells' upcoming evals are
CORRECT (fix landed before their eval step); (5) stdprobe data reinterpreted: 29% live
groups is CONSISTENT with the large adapter movement — there never was a reward-side
inertness at this rung.

### Zero-gradient account DEAD on both rungs (writer refutation, accepted)
Writer's checks, both sharper than mine: (1) artifact signature is BIT-IDENTICAL outputs
— qwen25_3b identical on every metric; clipb32 moved (.0260->.0257, 2971->2825, .1513->
.1473) => different phenomenon, clipb32 verdict stands (test works without reading code).
(2) lora_B abs-sums: clipb32 3048 — FOUR TIMES bgevl_l-lr1e5's 812 (which gained +5.5
R@5). clipb32's gradient was emphatically nonzero; retrieval didn't follow. With gme2b
control 0.55 degenerate, uniform-reward -> zero-advantage -> no-update is dead on BOTH
rungs it was proposed for. INERTNESS IS NOW AN OBSERVATION, NOT A MECHANISM: substantial
parameter movement buying no measurable retrieval change, cause unresolved. Do not
restore the mechanism without new evidence.
CONVENTIONS: lora_B abs-sum added to the standard harvest (mmrag/adapter_stats.py, no
GPU) — separates "didn't train" from "trained and it didn't help"; would have flagged
44041 on landing. Re-eval of qwen25_3b treated as a FRESH CELL: first check outputs
differ from base, then interpret.

### clipb16-v3pure (09:14): CONTRACTS — bracket tightens to (0.026, 0.031) CLIP-family
zs entR@5 0.0310 -> 0.0003, distinct 96/3000 (deepest contraction recorded), acc ROSE
0.1680 -> 0.1947 = the no-ctx floor (0.1927) within noise. ANTI-DIAGNOSTIC INSTANCE #3,
the cleanest: retrieval fully dead, reader reverts exactly to context-free behavior —
acc-at-floor is now a predicted signature of full contraction (bgevl cells were en route:
0.178/0.183 rising toward floor from 0.159).
BRACKET (lr 1e-4, CLIP family, both endpoints fresh): inert/contraction boundary in
(0.0260, 0.0310) — clipb32 inert, clipb16 contracts. Print-once: this version can print;
the qwen25_3b point (zs 0.0373) now sits ABOVE the bracket and its re-eval tests whether
the boundary TRANSFERS ACROSS FAMILIES (raw-VLM), not bisection. FAMILY-SCOPE the claim.
lora_B: clipb16 3126 ~= clipb32 3048 — same optimiser movement, opposite phenotype;
movement magnitude does not determine outcome (second confirmation).

### qwen25_3b corrected verdicts (10:41, fresh-cell rule applied: outputs differ from base)
qwen25_3b-v3pure (lr 1e-4, RE-EVAL): **CONTRACTS, totally** — entR@5 0.0373 -> 0.0000,
distinct 32/3000 (deepest recorded), acc 0.1607 -> 0.1707 (climbing toward floor 0.1927;
acc-at-floor pathway consistent). My original amended prediction (INERT) now definitively
scores MISS — the artifact had hidden a contraction.
qwen25_3b-v3pure-lr2e5 (family rate): retrieval also dies — entR@5 0.0003, distinct
2033/3000 (partial diversity loss), acc 0.1853 (near floor). Raw Qwen2.5-VL-3B fails at
BOTH rates by retrieval-death; dose-response in diversity (32 @1e-4 vs 2033 @2e-5).
CONSEQUENCES: (1) BOUNDARY TRANSFERS ACROSS FAMILIES at the one interior point — zs
0.0373 > bracket upper 0.031 predicts contraction; observed. (2) The INERT phenotype has
exactly ONE member (clipb32, zs 0.026) — it is an edge case at near-zero signal, not a
band; the taxonomy's load-bearing split is contract-vs-drift-vs-learn, with inert as the
degenerate corner. (3) stdprobe data now coherent: 29% live groups = gradient existed =
contraction had fuel.

### bgevl_l-v3pure-lr3e6 (11:52): flat — 1e-5 is an INTERIOR max of the tested window
entR@5 0.1573 -> 0.1523 (~flat), distinct 2952 (healthy), acc 0.1640, lora_B 260 (1/3 of
the 1e-5 run's 812) — undertrained at 500 steps. BGE-VL-L window over the full ladder:
3e-6 flat | 1e-5 LEARNS +5.5 | 3e-5 damaged -1.0 | 1e-4 contracts. 1e-5 upgrades from
boundary cell to interior maximum; the +5.5 rescue number is no longer an underestimate
by construction (longer-run/finer-grid gains remain possible but unregistered).
